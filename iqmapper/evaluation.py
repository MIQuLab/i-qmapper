"""
evaluation.py — offline comparison of the layout-search engines.

The guided beam search (`guided_search.guided_sweep`) is a heuristic; the VF2
sweep (`ancilla_sweep.sweep_ancilla_layouts`) is exhaustive up to a 2000-mapping
cap that it routinely hits on large devices. Which one is actually better is an
empirical question, and this module is how it gets answered before anything is
made the default.

It replays both engines over the calibration snapshots already sitting in the
local cache — no network, no QPU time, no credentials. For each (date, norb, k)
it records the best circuit-level LQS each engine reached, the wall time, how
many distinct chain scaffolds each produced, and whether VF2 truncated.

Reading the table
-----------------
`delta_lqs` is `guided - vf2` in LQS points. Positive means guided found a
better layout on that snapshot. `vf2_capped` matters: where VF2 truncated, it
was ranking an arbitrary 2% sample, so a guided win there says the guided
sample is better-chosen, not that beam search beats exhaustive search. Where
VF2 did NOT truncate it enumerated everything, so a guided win is a genuine
win and a guided loss is the heuristic's real cost.

Both engines get the same graph, the same properties and the same
`circuit_score_fn`, so the comparison is like-for-like.

Where the topology comes from
-----------------------------
The calibration cache stores ONLY the property arrays: `data._save_calibration`
writes `qubit_props` and `gate_props`, and `_load_calibration` returns exactly
that pair. The coupling map is not cached — the gate arrays are positional
against `state.edges_list`, but that list lives on the session, not in the
cache. `data.py` is a frozen engine, so this module cannot start caching it.

So the coupling has to come from somewhere else, in this order:

  1. an explicit `coupling=[(u, v), ...]` argument;
  2. a saved session for that backend under `.iqmapper_sessions/`, which does
     persist `edges_list` (see `state.to_dict`);
  3. otherwise the run reports a clear error rather than guessing — an
     invented topology would produce numbers that look real and mean nothing.

Save a session once for the backend you want to evaluate and every cached
calibration for it becomes replayable.

CLI
---
    python -m iqmapper.evaluation ibm_boston --norb 8 12 16

prints the table and writes `.scratch/eval_<backend>_<timestamp>.md`.
"""
import argparse
import json as _json
import logging
import os
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np
import rustworkx

from . import data as _data
from .ancilla_sweep import sweep_ancilla_layouts
from .guided_search import guided_sweep

_log = logging.getLogger(__name__)

DEFAULT_NORBS = (8, 12, 16)


# ── replaying a cached calibration as a backend/props pair ──────────

class _CachedProps:
    """BackendProperties stand-in reading a cached calibration snapshot.

    The engines only ever call `readout_error(q)` and `gate_error(name, edge)`,
    so that is the whole surface. Missing entries return NaN, which both
    engines already map to their uncalibrated penalty.
    """

    def __init__(self, qubit_props, gate_props, edges):
        self._readout = np.asarray(
            qubit_props.get('readout_error', []), dtype=float)
        gate = None
        for key in ('cz_error', 'rzz_error'):
            if key in gate_props:
                gate = np.asarray(gate_props[key], dtype=float)
                break
        self._gate = gate if gate is not None else np.array([], dtype=float)
        self._edge_index = {}
        for i, (u, v) in enumerate(edges):
            self._edge_index[frozenset((int(u), int(v)))] = i

    def readout_error(self, q):
        q = int(q)
        if 0 <= q < len(self._readout):
            return float(self._readout[q])
        return float('nan')

    def gate_error(self, _name, edge):
        u, v = edge
        idx = self._edge_index.get(frozenset((int(u), int(v))))
        if idx is None or idx >= len(self._gate):
            return float('nan')
        return float(self._gate[idx])


def _backend_from_edges(edges, n_qubits, name='cached'):
    """A BackendV2-shaped stand-in carrying just the coupling graph."""
    graph = rustworkx.PyDiGraph()
    for _ in range(n_qubits):
        graph.add_node(None)
    for u, v in edges:
        graph.add_edge(int(u), int(v), None)
        graph.add_edge(int(v), int(u), None)
    return SimpleNamespace(name=name,
                           coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


def _product_scorer():
    """A stand-in `circuit_score_fn` usable without the app's state.

    The real UI scorer is state-bound (`export.scoring.estimate_circuit_fidelity`
    needs dict-shaped props on a SessionState). For an engine COMPARISON what
    matters is that both engines are ranked by the same monotone function of
    layout quality, so the product score serves: it is the same quantity the
    UI's LQS is a refinement of, and using it keeps the harness runnable
    against a bare cache. Reported as `lqs_proxy` rather than `lqs` for that
    reason.
    """
    def _score(entry):
        return float(entry.get('product_score') or 0.0)
    return _score


# ── report ──────────────────────────────────────────────────────────

class Row:
    """One (date, norb, k) comparison."""

    __slots__ = ('date', 'norb', 'k', 'vf2_lqs', 'guided_lqs', 'vf2_time',
                 'guided_time', 'vf2_scaffolds', 'guided_scaffolds',
                 'vf2_capped', 'refined')

    def __init__(self, date, norb, k, vf2_lqs, guided_lqs, vf2_time,
                 guided_time, vf2_scaffolds, guided_scaffolds, vf2_capped,
                 refined):
        self.date = date
        self.norb = norb
        self.k = k
        self.vf2_lqs = vf2_lqs
        self.guided_lqs = guided_lqs
        self.vf2_time = vf2_time
        self.guided_time = guided_time
        self.vf2_scaffolds = vf2_scaffolds
        self.guided_scaffolds = guided_scaffolds
        self.vf2_capped = vf2_capped
        self.refined = refined

    @property
    def delta(self):
        if self.vf2_lqs is None or self.guided_lqs is None:
            return None
        return self.guided_lqs - self.vf2_lqs

    @property
    def guided_at_least(self):
        d = self.delta
        return None if d is None else d >= -1e-12


class Report:
    """The rows plus the summary a human actually reads."""

    def __init__(self, backend_name, rows, errors=None):
        self.backend_name = backend_name
        self.rows = list(rows)
        self.errors = list(errors or [])

    # ── aggregates ──────────────────────────────────────────────
    def _comparable(self):
        return [r for r in self.rows if r.delta is not None]

    def summary(self):
        comp = self._comparable()
        if not comp:
            return 'No comparable cases.'
        wins = sum(1 for r in comp if r.guided_at_least)
        deltas = [r.delta * 100.0 for r in comp]      # LQS points
        vf2_t = sum(r.vf2_time for r in comp)
        guided_t = sum(r.guided_time for r in comp)
        speed = (vf2_t / guided_t) if guided_t > 0 else float('inf')
        faster = (f'{speed:.1f}x faster' if speed >= 1
                  else f'{1 / speed:.1f}x slower')
        return (f'guided >= VF2 in {wins}/{len(comp)} cases; '
                f'mean dLQS {statistics.mean(deltas):+.1f} pt; '
                f'max {max(deltas):+.1f} pt; min {min(deltas):+.1f} pt; '
                f'{faster}')

    def to_markdown(self):
        lines = [f'# Engine comparison - {self.backend_name}', '',
                 self.summary(), '']
        if self.errors:
            lines += ['## Errors', '']
            lines += [f'- {e}' for e in self.errors]
            lines += ['']
        lines += [
            '| date | norb | k | VF2 | guided | dLQS (pt) | VF2 s | guided s |'
            ' VF2 scaffolds | guided scaffolds | VF2 capped | refined |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|']
        for r in self.rows:
            def _f(value, digits=4):
                return '-' if value is None else f'{value:.{digits}f}'
            delta = ('-' if r.delta is None else f'{r.delta * 100:+.2f}')
            lines.append(
                f'| {r.date} | {r.norb} | {r.k} | {_f(r.vf2_lqs)} | '
                f'{_f(r.guided_lqs)} | {delta} | {r.vf2_time:.2f} | '
                f'{r.guided_time:.2f} | {r.vf2_scaffolds} | '
                f'{r.guided_scaffolds} | {"yes" if r.vf2_capped else "no"} | '
                f'{r.refined} |')
        return '\n'.join(lines) + '\n'

    def __repr__(self):                       # pragma: no cover - debug aid
        return f'<Report {self.backend_name} rows={len(self.rows)}>'


# ── the comparison ──────────────────────────────────────────────────

def _scaffolds(result):
    """Distinct chain scaffolds across all strata."""
    seen = set()
    for entries in (result.get('strata') or {}).values():
        for entry in entries:
            seen.add(frozenset(entry['layout']))
    return len(seen)


def _best_lqs(result, k):
    entries = (result.get('strata') or {}).get(k) or []
    if not entries:
        return None
    values = [e.get('lqs_circuit') for e in entries
              if e.get('lqs_circuit') is not None]
    if values:
        return max(values)
    return max(e['product_score'] for e in entries)


def compare_engines(backend_name, dates=None, norbs=DEFAULT_NORBS, top_n=3,
                    min_bridges=0, cache_dir=None, beam_width=200,
                    circuit_score_fn=None, limit=None, coupling=None,
                    session_dir=None):
    """Run both engines over the cached calibrations and report.

    Read-only and offline: reads the local calibration cache, never the
    network. `min_bridges=0` is clamped to 1, the engines' floor.

    Args:
        backend_name: the cached backend to replay.
        dates: explicit `last_update` strings, or None for everything cached.
        norbs: orbital counts to test.
        cache_dir: override `data._CACHE_DIR` for the duration (tests).
        limit: cap the number of calibrations replayed.
    """
    scorer = circuit_score_fn or _product_scorer()
    min_bridges = max(1, int(min_bridges))

    old_cache = None
    if cache_dir is not None:
        old_cache = _data._CACHE_DIR
        _data._CACHE_DIR = str(cache_dir)
    try:
        entries = _data.list_known_calibrations(backend_name)
        stamps = ([e['last_update'] for e in entries] if dates is None
                  else list(dates))
        if limit is not None:
            stamps = stamps[:int(limit)]

        rows, errors = [], []
        # The cache has no topology; resolve it once for the whole run.
        if coupling is None:
            coupling = coupling_from_sessions(backend_name, session_dir)
        if coupling is None:
            errors.append(
                f'No coupling map for {backend_name}: the calibration cache '
                f'stores only property arrays. Pass coupling=[(u, v), ...] or '
                f'save a session for this backend so .iqmapper_sessions/ '
                f'carries its edges_list.')
            return Report(backend_name, [], errors)

        for stamp in stamps:
            snapshot = _data._load_calibration(backend_name, stamp)
            if not snapshot:
                errors.append(f'{stamp}: no cached calibration')
                continue
            qubit_props, gate_props, edges = _unpack(snapshot)
            if edges is None:
                edges = coupling
            n_qubits = len(qubit_props.get('readout_error', []))
            if n_qubits == 0:
                errors.append(f'{stamp}: no readout_error array; skipped')
                continue
            backend = _backend_from_edges(edges, n_qubits, backend_name)
            props = _CachedProps(qubit_props, gate_props, edges)

            for norb in norbs:
                rows.extend(_compare_one(stamp, norb, backend, props,
                                         top_n, min_bridges, beam_width,
                                         scorer, errors))
        return Report(backend_name, rows, errors)
    finally:
        if old_cache is not None:
            _data._CACHE_DIR = old_cache


def _unpack(snapshot):
    """Cached snapshots have varied in shape; accept what we find.

    `data._load_calibration` returns a 2-tuple `(qubit_props, gate_props)` and
    carries no topology, so `edges` is normally None and the caller supplies
    the coupling — see the module docstring.
    """
    if isinstance(snapshot, tuple):
        qubit_props, gate_props = snapshot[0], snapshot[1]
        edges = snapshot[2] if len(snapshot) > 2 else None
    else:
        qubit_props = snapshot.get('qubit_props', {})
        gate_props = snapshot.get('gate_props', {})
        edges = snapshot.get('edges')
    if edges is not None:
        edges = [tuple(int(x) for x in e) for e in edges]
    return dict(qubit_props or {}), dict(gate_props or {}), edges


def coupling_from_sessions(backend_name, session_dir=None):
    """Recover a coupling map for `backend_name` from a saved session.

    Sessions persist `edges_list` (and `qubit_ids`) via `state.to_dict`, so one
    saved session makes every cached calibration for that backend replayable.
    Returns a list of (u, v), or None when nothing matches.
    """
    directory = session_dir or os.path.join(os.getcwd(), '.iqmapper_sessions')
    try:
        names = sorted(os.listdir(directory))
    except OSError:
        return None
    for name in names:
        if not name.endswith('.json'):
            continue
        try:
            with open(os.path.join(directory, name), 'r',
                      encoding='utf-8') as fh:
                payload = _json.load(fh)
        except Exception:
            continue
        if str(payload.get('active_backend_name') or '') != str(backend_name):
            continue
        edges = payload.get('edges_list') or []
        if edges:
            return [tuple(int(x) for x in e) for e in edges]
    return None


def _compare_one(stamp, norb, backend, props, top_n, min_bridges, beam_width,
                 scorer, errors):
    """One (date, norb): run both engines, emit a row per shared stratum."""
    try:
        t0 = time.time()
        vf2 = sweep_ancilla_layouts(norb, backend, props,
                                    min_bridges=min_bridges, top_n=top_n,
                                    circuit_score_fn=scorer)
        vf2_time = time.time() - t0
    except Exception as exc:                  # noqa: BLE001 - recorded
        errors.append(f'{stamp} norb={norb}: VF2 failed ({exc})')
        return []
    try:
        t0 = time.time()
        guided = guided_sweep(norb, backend, props, min_bridges=min_bridges,
                              top_n=top_n, circuit_score_fn=scorer,
                              beam_width=beam_width)
        guided_time = time.time() - t0
    except Exception as exc:                  # noqa: BLE001 - recorded
        errors.append(f'{stamp} norb={norb}: guided failed ({exc})')
        return []

    vf2_scaffolds = _scaffolds(vf2)
    guided_scaffolds = _scaffolds(guided)
    capped = bool(vf2.get('vf2_truncated'))
    refined = int(guided.get('refined_count') or 0)

    rows = []
    for k in sorted(set(vf2.get('strata', {})) | set(guided.get('strata', {})),
                    reverse=True):
        rows.append(Row(stamp, norb, k, _best_lqs(vf2, k),
                        _best_lqs(guided, k), vf2_time, guided_time,
                        vf2_scaffolds, guided_scaffolds, capped, refined))
    return rows


# ── CLI ─────────────────────────────────────────────────────────────

def main(argv=None):
    parser = argparse.ArgumentParser(
        prog='python -m iqmapper.evaluation',
        description='Compare the VF2 and guided layout-search engines over '
                    'the locally cached calibrations. Offline and read-only.')
    parser.add_argument('backend', help='cached backend name, e.g. ibm_boston')
    parser.add_argument('--norb', type=int, nargs='+', default=list(DEFAULT_NORBS))
    parser.add_argument('--top-n', type=int, default=3)
    parser.add_argument('--min-bridges', type=int, default=1)
    parser.add_argument('--beam-width', type=int, default=200)
    parser.add_argument('--limit', type=int, default=None,
                        help='replay at most N cached calibrations')
    parser.add_argument('--out-dir', default='.scratch')
    parser.add_argument('--session-dir', default=None,
                        help='where to look for a saved session carrying the '
                             'coupling map (default .iqmapper_sessions)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    report = compare_engines(args.backend, norbs=tuple(args.norb),
                             top_n=args.top_n, min_bridges=args.min_bridges,
                             beam_width=args.beam_width, limit=args.limit,
                             session_dir=args.session_dir)
    markdown = report.to_markdown()
    print(markdown)

    try:
        os.makedirs(args.out_dir, exist_ok=True)
        stamp = time.strftime('%Y%m%dT%H%M%S')
        path = os.path.join(args.out_dir,
                            f'eval_{args.backend}_{stamp}.md')
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(markdown)
        print(f'written: {path}')
    except Exception as exc:                  # noqa: BLE001 - informational
        print(f'could not write report: {exc}', file=sys.stderr)
    return 0


if __name__ == '__main__':                    # pragma: no cover - CLI
    raise SystemExit(main())
