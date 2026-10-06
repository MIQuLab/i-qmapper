"""
joint_plan.py — joint multi-circuit planner: a beam over placement plans.

`multi_sweep` places circuits greedily: circuit 0 takes the best region on the
chip, circuit 1 takes the best of what is left, and so on. That is myopic. The
best region for circuit 0 can sit where its footprint plus buffer cuts the chip
into pieces too small for circuit 1, which is then pushed onto a poor region
while two good regions existed side by side.

This module keeps several partial plans alive instead of one. Circuits are
still placed in the given order. A plan is the entries chosen so far plus the
exclusion set they imply; at each circuit every surviving plan is searched on
the chip minus its own exclusions, extended with its top `branch` candidates,
and the best `beam_width` plans survive. `beam_width = branch = 1` is exactly
the sequential greedy.

The single-circuit search is injected (`search_fn`), so the planner runs on
whichever engine the caller selected — VF2, guided or hybrid. Exclusion and
buffer semantics are `auto_lucj._exclude_with_buffer`, applied as
`multi_sweep._orchestrate` applies them, and every circuit record is built by
`multi_sweep._circuit_record`, so the result has the `sweep_multi_circuits`
shape and every multi-circuit display path works unchanged.


PLAN SCORE
----------
    prod(metric_i) * (1 - boundary_penalty) ** n_boundary

`metric_i` is the circuit's displayed metric, chosen as `multi_sweep._best_entry`
chooses it: `lqs_circuit` when the circuit's search returned any, otherwise
`product_score`. `n_boundary` counts physical couplers whose two endpoints
belong to two different circuits. With `buffer_qubits >= 1` that count is zero
by construction (every neighbour of a placed qubit is excluded). At buffer 0 a
gate can sit on the boundary between two active circuits — spectator crosstalk
the LQS does not model — and the penalty is the knob that prices it.


GUARANTEES
----------
* Never worse than sequential: the greedy plan is computed through the same
  code path and the better of the two is returned (`plan_source` says which).
* Fit failures never raise: a circuit that fits in no surviving plan stops the
  run, and the best partial plan is returned with `failed_at` set.
* Deterministic: candidate order follows the search result, pruning ties break
  on candidate rank, and final ties on the sorted qubit tuples.
"""
import logging

from .ancilla_sweep import sweep_ancilla_layouts
from .auto_lucj import _exclude_with_buffer, _make_backend_cmap_pygraph
from .multi_sweep import _circuit_record, _entry_qubits, _per_circuit

_log = logging.getLogger(__name__)

DEFAULT_BEAM_WIDTH = 4
DEFAULT_BRANCH = 3


def worst_case_searches(n_circuits, beam_width):
    """Upper bound on single-circuit searches for one joint run.

    The beam searches once per surviving plan per circuit: 1 for circuit 0,
    at most `beam_width` for each later one. The greedy comparison plan shares
    those searches except when its prefix has dropped out of the beam, which
    can cost one extra search for each circuit from index 2 on.
    """
    if n_circuits <= 0:
        return 0
    return 1 + beam_width * (n_circuits - 1) + max(0, n_circuits - 2)


def ranked_candidates(res):
    """All entries of a sweep result, best first, by the displayed metric.

    Same rule as `multi_sweep._best_entry(res, True)`: `lqs_circuit` when any
    entry carries one (entries without it drop out), else `product_score`.
    The sort is stable over strata order, so element 0 is `_best_entry`'s pick
    even on ties. Returns `(entries, key)`.
    """
    entries = [e for group in res.get('strata', {}).values() for e in group]
    scored = [e for e in entries if e.get('lqs_circuit') is not None]
    if scored:
        return sorted(scored, key=lambda e: -e['lqs_circuit']), 'lqs_circuit'
    return sorted(entries, key=lambda e: -e['product_score']), 'product_score'


def _metric(entry, key):
    value = entry.get(key)
    return entry['product_score'] if value is None else value


class _Plan:
    """A partial placement: one record per placed circuit, plus its footprint."""

    __slots__ = ('records', 'metrics', 'qsets', 'remaining', 'excluded',
                 'placed', 'buffers', 'boundary', 'path')

    def __init__(self, remaining, excluded):
        self.records = ()
        self.metrics = ()
        self.qsets = ()
        self.remaining = remaining
        self.excluded = frozenset(excluded)
        self.placed = ()
        self.buffers = ()
        self.boundary = ()
        self.path = ()

    def extend(self, record, metric, buffer_qubits, adjacency, rank):
        chosen_full = _entry_qubits(record['chosen'])
        qset = frozenset(chosen_full)
        new_boundary = []
        for q in sorted(qset):
            for other in self.qsets:
                for n in adjacency.get(q, ()):
                    if n in other:
                        new_boundary.append((min(q, n), max(q, n)))
        remaining, new_excl = _exclude_with_buffer(
            self.remaining, set(chosen_full), buffer_qubits)
        child = _Plan(remaining, self.excluded | new_excl)
        child.records = self.records + (record,)
        child.metrics = self.metrics + (metric,)
        child.qsets = self.qsets + (qset,)
        child.placed = self.placed + tuple(chosen_full)
        child.buffers = self.buffers + tuple(sorted(new_excl - qset))
        child.boundary = tuple(sorted(set(self.boundary) | set(new_boundary)))
        child.path = self.path + (rank,)
        return child

    def score(self, penalty):
        if not self.records:
            return None
        total = 1.0
        for m in self.metrics:
            total *= m
        return total * (1.0 - penalty) ** len(self.boundary)

    def qkey(self):
        return tuple(tuple(sorted(s)) for s in self.qsets)


def _final_key(plan, penalty):
    """Sort key, best first: more circuits, higher score, then qubit tuples."""
    score = plan.score(penalty)
    return (-len(plan.records), -(score if score is not None else 0.0),
            plan.qkey())


def _beam(ctx, prefix, beam_width, branch):
    """Run the beam from a fixed prefix. Returns (best_plan, failed_at)."""
    norbs = ctx['norbs']
    penalty = ctx['boundary_penalty']
    buffer_qubits = ctx['buffer_qubits']
    adjacency = ctx['_adjacency']

    root = _Plan(ctx['_root_remaining'], ctx['_stability'])
    for i, (rec, chosen) in enumerate(prefix):
        rec = dict(rec)
        rec['chosen'] = chosen
        _entries, key = ranked_candidates(rec)
        root = root.extend(rec, _metric(chosen, key), buffer_qubits,
                           adjacency, 0)

    beam = [root]
    failed_at = None
    for i in range(len(prefix), len(norbs)):
        children = {}
        for plan in beam:
            res = _search(ctx, i, plan.excluded)
            if res is None:
                continue
            entries, key = ranked_candidates(res)
            for rank, entry in enumerate(entries[:branch]):
                rec = _circuit_record(i, norbs[i], res, entry)
                child = plan.extend(rec, _metric(entry, key), buffer_qubits,
                                    adjacency, rank)
                dedupe = child.qsets
                held = children.get(dedupe)
                if held is None or (-child.score(penalty), child.path) < (
                        -held.score(penalty), held.path):
                    children[dedupe] = child
        if not children:
            failed_at = i
            break
        beam = sorted(children.values(),
                      key=lambda p: (-p.score(penalty), p.path))[:beam_width]
    best = min(beam, key=lambda p: _final_key(p, penalty))
    return best, failed_at


def _search(ctx, i, excluded):
    """Memoised single-circuit search. None when the circuit does not fit."""
    norb_i = ctx['norbs'][i]
    mb = _per_circuit(ctx['min_bridges'], i)
    key = (norb_i, mb, frozenset(excluded))
    cache = ctx['_cache']
    if key in cache:
        return cache[key]
    ctx['_calls'][0] += 1
    try:
        res = ctx['search_fn'](norb_i, exclude_qubits=(set(excluded) or None),
                               min_bridges=mb)
    except Exception as exc:
        _log.info("Joint plan: circuit %d (norb=%d) did not fit: %s",
                  i, norb_i, exc)
        res = None
    if res is not None and not ranked_candidates(res)[0]:
        res = None
    cache[key] = res
    return res


def _result(ctx, plan, failed_at, greedy_plan, plan_source, n_searches):
    penalty = ctx['boundary_penalty']
    chosen_list = [r['chosen'] for r in plan.records]
    total_product = None
    if chosen_list:
        total_product = 1.0
        for c in chosen_list:
            total_product *= c['product_score']
    total_lqs = None
    if chosen_list and all(c.get('lqs_circuit') is not None
                           for c in chosen_list):
        total_lqs = 1.0
        for c in chosen_list:
            total_lqs *= c['lqs_circuit']
    return {
        'circuits': list(plan.records),
        'total_product': total_product,
        'total_lqs_circuit': total_lqs,
        'placed_qubits': list(plan.placed),
        'buffer_qubits_used': list(plan.buffers),
        'failed_at': failed_at,
        'planner': 'joint',
        'beam_width': ctx['beam_width'],
        'branch': ctx['branch'],
        'boundary_penalty': penalty,
        'boundary_couplers': [tuple(e) for e in plan.boundary],
        'greedy_total': greedy_plan.score(penalty),
        'greedy_placed': len(greedy_plan.records),
        'joint_total': plan.score(penalty),
        'plan_source': plan_source,
        'n_searches': n_searches,
        '_context': ctx,
    }


def _run(ctx, prefix):
    calls_before = ctx['_calls'][0]
    greedy, greedy_failed = _beam(ctx, prefix, 1, 1)
    joint, joint_failed = _beam(ctx, prefix, ctx['beam_width'], ctx['branch'])
    penalty = ctx['boundary_penalty']
    if _final_key(greedy, penalty) < _final_key(joint, penalty):
        plan, failed_at, source = greedy, greedy_failed, 'greedy'
    else:
        plan, failed_at, source = joint, joint_failed, 'beam'
    return _result(ctx, plan, failed_at, greedy, source,
                   ctx['_calls'][0] - calls_before)


def plan_multi_circuits(norbs, backend, props, *, search_fn=None,
                        beam_width=DEFAULT_BEAM_WIDTH, branch=DEFAULT_BRANCH,
                        buffer_qubits=1, exclude_qubits=None,
                        min_bridges=None, top_n=3, circuit_score_fn=None,
                        boundary_penalty=0.0) -> dict:
    """Place `len(norbs)` LUCJ circuits on one backend, planning jointly.

    Args:
        norbs: spatial-orbital counts, one per circuit, in placement order.
        backend, props: as for `ancilla_sweep.sweep_ancilla_layouts`.
        search_fn: ``callable(norb, *, exclude_qubits, min_bridges) -> sweep
            result dict``. Default: VF2 (`sweep_ancilla_layouts`) with
            `top_n` and `circuit_score_fn`, the same call `multi_sweep`
            makes. Pass a closure to plan on another engine.
        beam_width: plans kept after each circuit (B).
        branch: candidates per circuit each plan is extended with (M).
        buffer_qubits: idle coupling-graph hops kept clear between circuits.
        exclude_qubits: physical ids barred from every circuit.
        min_bridges: global floor or a per-circuit list.
        top_n, circuit_score_fn: used only by the default `search_fn`.
        boundary_penalty: in [0, 1); each coupler shared by two circuits
            multiplies the plan score by `1 - boundary_penalty`. 0 = off.

    Returns the `multi_sweep.sweep_multi_circuits` keys, plus `planner`,
    `beam_width`, `branch`, `boundary_penalty`, `boundary_couplers`,
    `greedy_total` and `joint_total` (plan scores, penalty included;
    `joint_total` is the returned plan's), `greedy_placed` (circuits the
    greedy plan placed — when the joint plan places more, its score is a
    product of more factors and the two totals are not comparable),
    `plan_source` ('beam' | 'greedy'), and `n_searches` (actual `search_fn`
    calls after memoisation).
    """
    if not (0.0 <= boundary_penalty < 1.0):
        raise ValueError(f"boundary_penalty={boundary_penalty} must be in "
                         f"[0, 1).")
    if beam_width < 1 or branch < 1:
        raise ValueError("beam_width and branch must both be >= 1.")
    if search_fn is None:
        def search_fn(norb, *, exclude_qubits, min_bridges):
            return sweep_ancilla_layouts(
                norb, backend, props, exclude_qubits=exclude_qubits,
                min_bridges=min_bridges, top_n=top_n,
                circuit_score_fn=circuit_score_fn)

    # Physical-id-tagged coupling graph for buffer expansion, pruned of the
    # stability exclusions first (buffer 0), exactly as _orchestrate does.
    remaining = _make_backend_cmap_pygraph(backend=backend)
    for idx in remaining.node_indices():
        remaining[idx] = idx
    adjacency = {}
    for u, v in remaining.edge_list():
        adjacency.setdefault(u, set()).add(v)
        adjacency.setdefault(v, set()).add(u)
    stability = set(int(q) for q in (exclude_qubits or []))
    if stability:
        remaining, _ = _exclude_with_buffer(remaining, stability, 0)

    ctx = {
        'norbs': list(norbs), 'backend': backend, 'props': props,
        'search_fn': search_fn, 'beam_width': beam_width, 'branch': branch,
        'buffer_qubits': buffer_qubits, 'exclude_qubits': exclude_qubits,
        'min_bridges': min_bridges, 'top_n': top_n,
        'circuit_score_fn': circuit_score_fn,
        'boundary_penalty': boundary_penalty,
        '_root_remaining': remaining, '_stability': frozenset(stability),
        '_adjacency': adjacency, '_cache': {}, '_calls': [0],
    }
    return _run(ctx, [])


def replan_from(result, index, new_choice):
    """Fix circuits `< index` and `new_choice` at `index`; plan the rest.

    Same contract as `multi_sweep.replan_from`: pure, reads the `_context` the
    result carries (searches already made are reused from its memo), returns a
    fresh result of the same shape.
    """
    ctx = result['_context']
    circuits = result['circuits']
    if not (0 <= index < len(circuits)):
        raise IndexError(f"replan index {index} out of range "
                         f"(have {len(circuits)} placed circuits)")
    prefix = [(circuits[i], circuits[i]['chosen']) for i in range(index)]
    prefix.append((circuits[index], new_choice))
    return _run(ctx, prefix)
