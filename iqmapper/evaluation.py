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
Optionally it also runs the hybrid engine (`engine_select.hybrid_sweep`) at one
or more thresholds, and `suggest_guided_max_k` turns the table into the value
to enter as "Guided up to k".

The metric
----------
`scorer='lqs'` ranks with the real circuit-level LQS — the same
`export.scoring.estimate_circuit_fidelity` call the app makes — rebuilt for
every snapshot, because the properties change from date to date.
`scorer='proxy'` ranks with the product score instead (reported as
`lqs_proxy`); it needs nothing but the property arrays. An explicit
`circuit_score_fn` overrides both (reported as `custom`).

Reading the table
-----------------
`delta_lqs` is `guided - vf2` in LQS points. Positive means guided found a
better layout on that snapshot. `vf2_capped` matters: where VF2 truncated, it
was ranking an arbitrary sample, so a guided win there says the guided
sample is better-chosen, not that beam search beats exhaustive search. Where
VF2 did NOT truncate it enumerated everything, so a guided win is a genuine
win and a guided loss is the heuristic's real cost.

Both engines get the same graph, the same properties and the same scorer,
so the comparison is like-for-like.

Where the topology comes from
-----------------------------
The calibration cache stores ONLY the property arrays: `data._save_calibration`
writes `qubit_props` and `gate_props`, and `_load_calibration` returns exactly
that pair. The coupling map is not cached — the gate arrays are positional
against `state.edges_list`, but that list lives on the session, not in the
cache. `data.py` is a frozen engine, so this module cannot start caching it.

So the coupling has to come from somewhere else, in this order:

  1. an explicit `coupling=[(u, v), ...]` argument;
  2. an explicit saved session file (`session_path=`, CLI `--session PATH`),
     which must be a session for the requested backend and carry an
     `edges_list`, or the run reports why it is not usable;
  3. a saved session for that backend found by scanning
     `<cwd>/.iqmapper_sessions/` (any `.json`) and `<cwd>/`
     (`*.qpusession.json`, where "Export Session" saves by default);
  4. otherwise the run reports a clear error rather than guessing — an
     invented topology would produce numbers that look real and mean nothing.

Sessions persist `backend_name`, `edges_list` and `qubit_ids` (see
`state.to_session_dict`). Save a session once for the backend you want to
evaluate and every cached calibration for it becomes replayable; if it was
saved elsewhere, pass it with `--session`.

The gate arrays are positional, so the edges must be the backend's own edge
list in the app's order (`data._generate_ibm`: `coupling_map.get_edges()`,
both directions). A saved session's `edges_list` is exactly that; an explicit
`coupling=` must be too. Qubit ids are `str(i)` for i in 0..n-1 in the app,
which is also the fallback when no session supplies them.

CLI
---
    python -m iqmapper.evaluation ibm_boston --norb 8 12 16 --hybrid-k 1 2

prints the table and the suggested "Guided up to k" per norb, and writes both
to `.scratch/eval_<backend>_<timestamp>.md`. `--scorer` defaults to `lqs`;
`--session PATH` names the saved session to take the coupling map from.
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
from .engine_select import hybrid_sweep
from .guided_search import guided_sweep

_log = logging.getLogger(__name__)

DEFAULT_NORBS = (8, 12, 16)

#: Scorer names accepted by `compare_engines`, and the metric each reports.
SCORERS = {'proxy': 'lqs_proxy', 'lqs': 'lqs'}

#: Extension "Export Session" gives a saved session.
SESSION_SUFFIX = '.qpusession.json'


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


def _lqs_scorer(qubit_props, gate_props, edges, qubit_ids):
    """The app's circuit-level LQS for one calibration snapshot.

    Mirrors `ui.lucj_panel._circuit_score`: a scratch one-layout state with the
    candidate's qubits as string ids, scored by
    `export.scoring.estimate_circuit_fidelity(fake, 0, 'lucj')`. The arrays
    are bound once, so build a new scorer for every snapshot.
    """
    from .export.scoring import estimate_circuit_fidelity
    qubit_props_data = {k: np.asarray(v) for k, v in qubit_props.items()}
    gate_props_data = {k: np.asarray(v) for k, v in gate_props.items()}
    edges_list = [tuple(int(x) for x in e) for e in edges]
    ids = [str(q) for q in qubit_ids]

    def _score(entry):
        layout = {
            'alpha': [str(q) for q in entry['alpha']],
            'beta': [str(q) for q in entry['beta']],
            'other': [],
            'ancilla': [str(q) for q in entry['ancillas']],
        }
        fake = SimpleNamespace(
            layouts_data={0: layout},
            qubit_props_data=qubit_props_data,
            gate_props_data=gate_props_data,
            qubit_ids=ids,
            edges_list=edges_list,
        )
        res = estimate_circuit_fidelity(fake, 0, 'lucj')
        return res['fidelity'] if res else None
    return _score


# ── report ──────────────────────────────────────────────────────────

class Row:
    """One (date, norb, k) comparison."""

    __slots__ = ('date', 'norb', 'k', 'vf2_lqs', 'guided_lqs', 'vf2_time',
                 'guided_time', 'vf2_scaffolds', 'guided_scaffolds',
                 'vf2_capped', 'refined', 'hybrid_lqs')

    def __init__(self, date, norb, k, vf2_lqs, guided_lqs, vf2_time,
                 guided_time, vf2_scaffolds, guided_scaffolds, vf2_capped,
                 refined, hybrid_lqs=None):
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
        # {guided_max_k: best LQS at this k}, one entry per hybrid threshold.
        self.hybrid_lqs = dict(hybrid_lqs or {})

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

    def __init__(self, backend_name, rows, errors=None, metric='lqs_proxy',
                 hybrid_ks=()):
        self.backend_name = backend_name
        self.rows = list(rows)
        self.errors = list(errors or [])
        self.metric = metric
        self.hybrid_ks = tuple(hybrid_ks)

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
                 f'Metric: {self.metric}', '', self.summary(), '']
        if self.errors:
            lines += ['## Errors', '']
            lines += [f'- {e}' for e in self.errors]
            lines += ['']
        hybrid_head = ''.join(f' hybrid k<={t} |' for t in self.hybrid_ks)
        hybrid_rule = '---:|' * len(self.hybrid_ks)
        lines += [
            '| date | norb | k | VF2 | guided | dLQS (pt) | VF2 s | guided s |'
            ' VF2 scaffolds | guided scaffolds | VF2 capped | refined |'
            + hybrid_head,
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|'
            + hybrid_rule]
        for r in self.rows:
            def _f(value, digits=4):
                return '-' if value is None else f'{value:.{digits}f}'
            delta = ('-' if r.delta is None else f'{r.delta * 100:+.2f}')
            hybrid_cells = ''.join(f' {_f(r.hybrid_lqs.get(t))} |'
                                   for t in self.hybrid_ks)
            lines.append(
                f'| {r.date} | {r.norb} | {r.k} | {_f(r.vf2_lqs)} | '
                f'{_f(r.guided_lqs)} | {delta} | {r.vf2_time:.2f} | '
                f'{r.guided_time:.2f} | {r.vf2_scaffolds} | '
                f'{r.guided_scaffolds} | {"yes" if r.vf2_capped else "no"} | '
                f'{r.refined} |' + hybrid_cells)
        return '\n'.join(lines) + '\n'

    def __repr__(self):                       # pragma: no cover - debug aid
        return f'<Report {self.backend_name} rows={len(self.rows)}>'


def suggest_guided_max_k(report, tol=0.0):
    """The highest k up to which guided holds its own, per norb.

    For each norb: the largest `t` such that, for every k from the floor up
    to `t`, the median `delta` (guided - VF2) over the snapshots is
    `>= -tol`. `tol` is in LQS points, the unit of the dLQS column. `None`
    when guided already loses at the floor (or the floor has no comparable
    rows). This is the value for "Guided up to k".
    """
    by_norb = {}
    for r in report.rows:
        if r.delta is not None:
            by_norb.setdefault(r.norb, {}).setdefault(r.k, []).append(r.delta)
    suggestion = {}
    for norb in sorted({r.norb for r in report.rows}):
        per_k = by_norb.get(norb, {})
        ks = sorted({r.k for r in report.rows if r.norb == norb})
        best = None
        for k in ks:
            deltas = per_k.get(k)
            if not deltas or statistics.median(deltas) * 100.0 < -tol:
                break
            best = k
        suggestion[norb] = best
    return suggestion


def format_suggestion(suggestion):
    """One line per norb, for the CLI."""
    lines = ['Suggested "Guided up to k":']
    for norb, t in sorted(suggestion.items()):
        lines.append(f'- norb {norb}: '
                     + ('none (guided loses at the floor)' if t is None
                        else f'{t}'))
    return '\n'.join(lines) + '\n'


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
                    session_dir=None, scorer='proxy', hybrid_ks=(),
                    session_path=None):
    """Run both engines over the cached calibrations and report.

    Read-only and offline: reads the local calibration cache, never the
    network. `min_bridges=0` is clamped to 1, the engines' floor.

    Args:
        backend_name: the cached backend to replay.
        dates: explicit `last_update` strings, or None for everything cached.
        norbs: orbital counts to test.
        cache_dir: override `data._CACHE_DIR` for the duration (tests).
        limit: cap the number of calibrations replayed.
        scorer: 'proxy' (product score, `lqs_proxy`) or 'lqs' (the app's
            circuit-level LQS, rebuilt per snapshot). An explicit
            `circuit_score_fn` wins over both.
        hybrid_ks: thresholds at which to also run the hybrid engine; each
            adds a `hybrid k<=t` column. Empty = the two-engine table.
        session_path: a saved session file to take the coupling map from,
            instead of scanning for one. Ignored when `coupling` is given.
    """
    if scorer not in SCORERS:
        raise ValueError(f"scorer={scorer!r}; expected one of "
                         f"{sorted(SCORERS)}.")
    metric = 'custom' if circuit_score_fn is not None else SCORERS[scorer]
    fixed_scorer = circuit_score_fn or (
        _product_scorer() if scorer == 'proxy' else None)
    hybrid_ks = tuple(int(t) for t in hybrid_ks)
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
        session_ids = None
        if coupling is None and session_path is not None:
            try:
                coupling, session_ids = _session_from_path(backend_name,
                                                           session_path)
            except ValueError as exc:
                errors.append(f'No coupling map for {backend_name}: {exc}')
                return Report(backend_name, [], errors, metric, hybrid_ks)
        if coupling is None:
            topology = _session_topology(backend_name, session_dir)
            if topology is not None:
                coupling, session_ids = topology
        if coupling is None:
            errors.append(_no_session_message(backend_name, session_dir))
            return Report(backend_name, [], errors, metric, hybrid_ks)

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
            if fixed_scorer is not None:
                snapshot_scorer = fixed_scorer
            else:
                ids = (session_ids if session_ids is not None
                       and len(session_ids) == n_qubits
                       else [str(i) for i in range(n_qubits)])
                snapshot_scorer = _lqs_scorer(qubit_props, gate_props, edges,
                                              ids)

            for norb in norbs:
                rows.extend(_compare_one(stamp, norb, backend, props,
                                         top_n, min_bridges, beam_width,
                                         snapshot_scorer, errors, hybrid_ks))
        return Report(backend_name, rows, errors, metric, hybrid_ks)
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

    Sessions persist `edges_list` (and `qubit_ids`) via
    `state.to_session_dict`, so one saved session makes every cached
    calibration for that backend replayable. Returns a list of (u, v), or None
    when nothing matches.
    """
    topology = _session_topology(backend_name, session_dir)
    return None if topology is None else topology[0]


def _session_backend(payload):
    """The backend a session belongs to. `to_session_dict` writes
    `backend_name`; `active_backend_name` is accepted as well."""
    return str(payload.get('backend_name')
               or payload.get('active_backend_name') or '')


def _read_session(path):
    """A saved session as a dict. Raises ValueError saying what is wrong."""
    from .export import MAX_IMPORT_BYTES
    try:
        size = os.path.getsize(path)
    except OSError as exc:
        raise ValueError(f'cannot read session file {path} ({exc}).') from exc
    if size > MAX_IMPORT_BYTES:
        raise ValueError(f'session file {path} is larger than '
                         f'{MAX_IMPORT_BYTES // 1_048_576} MB.')
    try:
        with open(path, 'r', encoding='utf-8') as fh:
            payload = _json.load(fh)
    except (OSError, ValueError) as exc:
        raise ValueError(f'{path} is not a readable session file '
                         f'({exc}).') from exc
    if not isinstance(payload, dict):
        raise ValueError(f'{path} is not a session file.')
    return payload


def _topology_of(payload):
    """`(edges, qubit_ids)` from a session, or None without an edges_list."""
    edges = payload.get('edges_list') or []
    if not edges:
        return None
    ids = payload.get('qubit_ids')
    return ([tuple(int(x) for x in e) for e in edges],
            [str(q) for q in ids] if ids else None)


def _session_from_path(backend_name, path):
    """`(edges, qubit_ids)` from one named session file.

    Raises ValueError when the file is unreadable, belongs to another backend,
    or carries no edges_list.
    """
    payload = _read_session(path)
    found = _session_backend(payload)
    if found != str(backend_name):
        raise ValueError(f'{path} is a session for '
                         f'{found or "no named backend"}, not {backend_name}.')
    topology = _topology_of(payload)
    if topology is None:
        raise ValueError(f'{path} is a session for {backend_name} but has no '
                         f'edges_list; save it again with the backend '
                         f'connected.')
    return topology


def _session_dirs(session_dir=None):
    """(managed session directory, working directory) — where to scan."""
    return (session_dir or os.path.join(os.getcwd(), '.iqmapper_sessions'),
            os.getcwd())


def _session_files(session_dir=None):
    """Candidate session files: any `.json` in the managed directory, then
    `*.qpusession.json` in the working directory. Sorted, deduplicated."""
    managed, cwd = _session_dirs(session_dir)
    out, seen = [], set()
    for directory, any_json in ((managed, True), (cwd, False)):
        try:
            names = sorted(os.listdir(directory))
        except OSError:
            continue
        for name in names:
            if not (name.endswith(SESSION_SUFFIX)
                    or (any_json and name.endswith('.json'))):
                continue
            path = os.path.realpath(os.path.join(directory, name))
            if path not in seen and os.path.isfile(path):
                seen.add(path)
                out.append(path)
    return out


def _session_topology(backend_name, session_dir=None):
    """`(edges, qubit_ids)` from the first matching saved session, or None.

    `qubit_ids` is None when the session does not carry it. Unreadable files,
    other backends' sessions and sessions without an edges_list are skipped.
    """
    for path in _session_files(session_dir):
        try:
            payload = _read_session(path)
        except ValueError:
            continue
        if _session_backend(payload) != str(backend_name):
            continue
        topology = _topology_of(payload)
        if topology is not None:
            return topology
    return None


def _no_session_message(backend_name, session_dir=None):
    """The error for a run with no coupling map: where it looked, and how
    to point it at a session saved somewhere else."""
    managed, cwd = _session_dirs(session_dir)
    return (f'No coupling map for {backend_name}: the calibration cache '
            f'stores only property arrays, and no saved session for this '
            f'backend with an edges_list was found in {managed} or in {cwd} '
            f'(*{SESSION_SUFFIX}). Save a session for {backend_name} in the '
            f'app and pass it with --session PATH (session_path= in '
            f'compare_engines), or pass coupling=[(u, v), ...].')


def _compare_one(stamp, norb, backend, props, top_n, min_bridges, beam_width,
                 scorer, errors, hybrid_ks=()):
    """One (date, norb): run the engines, emit a row per stratum."""
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

    hybrid = {}
    for t in hybrid_ks:
        try:
            hybrid[t] = hybrid_sweep(norb, backend, props, guided_max_k=t,
                                     min_bridges=min_bridges, top_n=top_n,
                                     circuit_score_fn=scorer,
                                     guided_kwargs={'beam_width': beam_width})
        except Exception as exc:              # noqa: BLE001 - recorded
            errors.append(f'{stamp} norb={norb}: hybrid k<={t} failed '
                          f'({exc})')

    vf2_scaffolds = _scaffolds(vf2)
    guided_scaffolds = _scaffolds(guided)
    capped = bool(vf2.get('vf2_truncated'))
    refined = int(guided.get('refined_count') or 0)

    rows = []
    for k in sorted(set(vf2.get('strata', {})) | set(guided.get('strata', {})),
                    reverse=True):
        rows.append(Row(stamp, norb, k, _best_lqs(vf2, k),
                        _best_lqs(guided, k), vf2_time, guided_time,
                        vf2_scaffolds, guided_scaffolds, capped, refined,
                        {t: _best_lqs(res, k) for t, res in hybrid.items()}))
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
    parser.add_argument('--session', default=None, metavar='PATH',
                        help='saved session file (.qpusession.json) for this '
                             'backend to take the coupling map from')
    parser.add_argument('--scorer', choices=sorted(SCORERS), default='lqs',
                        help='lqs = the app\'s circuit-level LQS (default); '
                             'proxy = product score')
    parser.add_argument('--hybrid-k', type=int, nargs='+', default=[],
                        help='also run the hybrid engine at these thresholds')
    parser.add_argument('--tol', type=float, default=0.0,
                        help='LQS points guided may trail VF2 by and still '
                             'count as holding its own (default 0)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.WARNING)
    report = compare_engines(args.backend, norbs=tuple(args.norb),
                             top_n=args.top_n, min_bridges=args.min_bridges,
                             beam_width=args.beam_width, limit=args.limit,
                             session_dir=args.session_dir,
                             scorer=args.scorer,
                             hybrid_ks=tuple(args.hybrid_k),
                             session_path=args.session)
    markdown = (report.to_markdown() + '\n'
                + format_suggestion(suggest_guided_max_k(report, args.tol)))
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
