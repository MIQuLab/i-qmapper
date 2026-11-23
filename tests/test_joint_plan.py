"""Tests for the joint multi-circuit planner (beam over placement plans).

Offline, deterministic. Hosts:
  * the two-block ladder of test_multi_sweep — the greedy-equivalence check;
  * hand-built line graphs, where nine consecutive qubits hold exactly one
    norb=4 layout (a3-a2-a1-a0-r-b0-b1-b2-b3), so regions are easy to reason
    about;
  * heavy_hex_graph(5) with seeded calibrations.
"""
from types import SimpleNamespace

import pytest
import rustworkx

from iqmapper import multi_sweep
from iqmapper.engine_select import hybrid_sweep
from iqmapper.joint_plan import (plan_multi_circuits, ranked_candidates,
                                 replan_from, worst_case_searches)
from iqmapper.multi_sweep import _best_entry, sweep_multi_circuits

from test_engine_select import heavy_hex_backend
from test_multi_sweep import _Props, _make_backend


# ── hosts ───────────────────────────────────────────────────────────

def _lines(lengths):
    """Disjoint line graphs, ids allocated in order. Returns (backend, spans)."""
    edges, spans, base = [], [], 0
    for length in lengths:
        edges += [(base + i, base + i + 1) for i in range(length - 1)]
        spans.append(range(base, base + length))
        base += length
    graph = rustworkx.PyDiGraph()
    for i in range(base):
        graph.add_node(i)
    for u, v in edges:
        graph.add_edge(u, v, None)
    backend = SimpleNamespace(coupling_map=SimpleNamespace(graph=graph),
                              operation_names=['cz', 'sx', 'rz', 'x'],
                              name='lines')
    return backend, spans


class _ErrProps:
    """Per-qubit error; a gate takes the worse of its two qubits."""

    def __init__(self, err):
        self.err = err

    def readout_error(self, q):
        return self.err[q]

    def gate_error(self, name, edge):
        return max(self.err[edge[0]], self.err[edge[1]])


def _myopia_host():
    """X: 17-qubit line, excellent in its middle 9 — the only place norb=8
    fits well. W: a separate, very good 9-qubit line. Y: a bad 17-qubit line.

    Greedy puts the norb=4 circuit on X's middle (its single best layout),
    which leaves norb=8 only Y. The joint plan puts norb=4 on W and norb=8 on X.
    """
    backend, (x, w, y) = _lines([17, 9, 17])
    err = {q: (0.001 if 4 <= q - x[0] <= 12 else 0.02) for q in x}
    err.update({q: 0.0015 for q in w})
    err.update({q: 0.1 for q in y})
    return backend, _ErrProps(err), (set(x), set(w), set(y))


def _adjacent_host():
    """A: an 18-qubit line that holds two norb=4 circuits only back to back
    (sharing coupler 8-9). Z: a separate, slightly worse 9-qubit line."""
    backend, (a, z) = _lines([18, 9])
    err = {q: 0.005 + 0.0001 * q for q in a}
    err.update({q: 0.02 for q in z})
    return backend, _ErrProps(err), (set(a), set(z))


def _qubits(entry):
    return set(entry['alpha']) | set(entry['beta']) | set(entry['ancillas'])


def _adjacency(backend):
    adj = {}
    for u, v in backend.coupling_map.graph.edge_list():
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def _strip(res):
    return {k: v for k, v in res.items() if k != '_context'}


# ── 1. greedy equivalence ───────────────────────────────────────────

@pytest.mark.parametrize('norbs,kw', [
    ([8, 8], dict(buffer_qubits=1)),
    ([8, 8], dict(buffer_qubits=0)),
    ([8, 6], dict(buffer_qubits=1, min_bridges=[2, 1])),
    ([8, 8], dict(buffer_qubits=1,
                  circuit_score_fn=lambda e: e['product_score'] ** 2)),
    ([8, 8, 8], dict(buffer_qubits=1)),                   # third does not fit
    ([8, 8], dict(buffer_qubits=1, exclude_qubits={0, 30})),
])
def test_beam_one_branch_one_is_the_sequential_greedy(norbs, kw):
    backend, props = _make_backend(), _Props()
    seq = sweep_multi_circuits(norbs, backend, props, **kw)
    joint = plan_multi_circuits(norbs, backend, props, beam_width=1,
                                branch=1, **kw)
    assert ([c['chosen'] for c in joint['circuits']]
            == [c['chosen'] for c in seq['circuits']])
    for key in ('placed_qubits', 'buffer_qubits_used', 'total_product',
                'total_lqs_circuit', 'failed_at'):
        assert joint[key] == seq[key], key
    assert joint['circuits'] == seq['circuits']
    assert joint['greedy_total'] == joint['joint_total']


def test_candidate_order_matches_best_entry():
    backend, props = _make_backend(), _Props()
    from iqmapper.ancilla_sweep import sweep_ancilla_layouts
    plain = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=5)
    scored = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=5,
                                   circuit_score_fn=lambda e: 0.5)  # all tie
    for res, has_scorer in ((plain, False), (plain, True), (scored, True)):
        ranked, _key = ranked_candidates(res)
        assert ranked[0] is _best_entry(res, has_scorer)


# ── 2. the myopia case ──────────────────────────────────────────────

def test_joint_beats_greedy_when_greedy_is_myopic():
    backend, props, (x, w, y) = _myopia_host()
    seq = sweep_multi_circuits([4, 8], backend, props)
    res = plan_multi_circuits([4, 8], backend, props)
    assert _qubits(seq['circuits'][1]['chosen']) <= y         # greedy's trap
    assert res['greedy_total'] == pytest.approx(seq['total_product'])
    assert res['joint_total'] > res['greedy_total']
    assert res['plan_source'] == 'beam'
    assert _qubits(res['circuits'][0]['chosen']) <= w
    assert _qubits(res['circuits'][1]['chosen']) <= x
    assert res['total_product'] == pytest.approx(res['joint_total'])


# ── 3. never worse than greedy ──────────────────────────────────────

@pytest.mark.parametrize('seed', range(10))
def test_never_worse_than_greedy_on_random_calibrations(seed):
    # Three circuits on 57 qubits with floor 1: tight enough that the plans
    # differ (on several seeds joint scores higher, on one it fits a circuit
    # greedy cannot), so this is not a tie-only check.
    backend, props = heavy_hex_backend(5, seed=seed)
    norbs = [5, 5, 5]
    seq = sweep_multi_circuits(norbs, backend, props, min_bridges=1)
    res = plan_multi_circuits(norbs, backend, props, min_bridges=1)
    assert res['greedy_total'] == pytest.approx(seq['total_product'])
    assert res['greedy_placed'] == len(seq['circuits'])
    assert len(res['circuits']) >= res['greedy_placed']
    if len(res['circuits']) == res['greedy_placed']:
        assert res['joint_total'] >= res['greedy_total']


def test_joint_can_fit_a_circuit_greedy_cannot():
    backend, props = heavy_hex_backend(5, seed=6)
    norbs = [5, 5, 5]
    seq = sweep_multi_circuits(norbs, backend, props, min_bridges=1)
    res = plan_multi_circuits(norbs, backend, props, min_bridges=1)
    assert seq['failed_at'] == 2 and res['greedy_placed'] == 2
    assert res['failed_at'] is None and len(res['circuits']) == 3
    assert res['plan_source'] == 'beam'


# ── 4. boundary couplers ────────────────────────────────────────────

def test_no_boundary_couplers_with_a_buffer():
    backend, props = heavy_hex_backend(5, seed=1)
    res = plan_multi_circuits([5, 5, 5], backend, props, buffer_qubits=1)
    assert res['boundary_couplers'] == []


def test_boundary_couplers_reported_and_priced_at_buffer_zero():
    backend, props, (a, z) = _adjacent_host()
    off = plan_multi_circuits([4, 4], backend, props, buffer_qubits=0)
    assert off['boundary_couplers'] == [(8, 9)]
    assert all(_qubits(c['chosen']) <= a for c in off['circuits'])

    on = plan_multi_circuits([4, 4], backend, props, buffer_qubits=0,
                             boundary_penalty=0.5)
    assert on['boundary_couplers'] == []
    used = [_qubits(c['chosen']) for c in on['circuits']]
    assert any(u <= z for u in used) and any(u <= a for u in used)
    # The greedy pair is the adjacent one, priced by the penalty.
    assert on['greedy_total'] == pytest.approx(off['joint_total'] * 0.5)
    assert on['joint_total'] > on['greedy_total']


@pytest.mark.parametrize('bad', [-0.1, 1.0])
def test_penalty_outside_its_range_is_refused(bad):
    backend, props = _make_backend(), _Props()
    with pytest.raises(ValueError, match='boundary_penalty'):
        plan_multi_circuits([8], backend, props, boundary_penalty=bad)


# ── 5. no overlap, exclusions respected ─────────────────────────────

@pytest.mark.parametrize('buffer_qubits', [0, 1, 2])
def test_no_overlap_and_every_exclusion_holds(buffer_qubits):
    backend, props = heavy_hex_backend(5, seed=4)
    stability = {7, 8, 9, 30, 31, 50}
    res = plan_multi_circuits([5, 5, 5], backend, props,
                              buffer_qubits=buffer_qubits,
                              exclude_qubits=stability)
    adj = _adjacency(backend)
    claimed = set()
    for c in res['circuits']:
        used = _qubits(c['chosen'])
        assert used.isdisjoint(stability)
        assert used.isdisjoint(claimed)          # no overlap, no buffer reuse
        ring = set(used)
        for _hop in range(buffer_qubits):
            ring |= {n for q in ring for n in adj.get(q, ())}
        claimed |= ring
    assert set(res['buffer_qubits_used']).isdisjoint(res['placed_qubits'])


# ── 6. fit failure ──────────────────────────────────────────────────

def test_more_circuits_than_the_chip_holds_stops_cleanly():
    backend, props = _make_backend(), _Props()
    res = plan_multi_circuits([8, 8, 8, 8], backend, props)
    assert res['failed_at'] == 2
    assert len(res['circuits']) == 2


def test_nothing_fits_returns_an_empty_plan():
    backend, props = _lines([5])[0], _ErrProps({q: 0.01 for q in range(5)})
    res = plan_multi_circuits([4], backend, props)
    assert res['failed_at'] == 0 and res['circuits'] == []
    assert res['greedy_total'] is None and res['joint_total'] is None


# ── 7. memoisation ──────────────────────────────────────────────────

def test_searches_are_memoised():
    backend, props = heavy_hex_backend(5, seed=2)
    calls = []

    def _counting(norb, *, exclude_qubits, min_bridges):
        calls.append((norb, frozenset(exclude_qubits or ()), min_bridges))
        from iqmapper.ancilla_sweep import sweep_ancilla_layouts
        return sweep_ancilla_layouts(norb, backend, props, top_n=3,
                                     exclude_qubits=exclude_qubits,
                                     min_bridges=min_bridges)

    norbs, beam, branch = [5, 5, 5], 4, 3
    res = plan_multi_circuits(norbs, backend, props, search_fn=_counting,
                              beam_width=beam, branch=branch)
    assert res['n_searches'] == len(calls)
    assert len(set(calls)) == len(calls)                 # never searched twice
    assert res['n_searches'] <= worst_case_searches(len(norbs), beam)
    assert res['n_searches'] < beam * branch * len(norbs)


def test_worst_case_search_count():
    assert worst_case_searches(0, 4) == 0
    assert worst_case_searches(1, 4) == 1
    assert worst_case_searches(2, 4) == 5
    assert worst_case_searches(3, 4) == 10


# ── 8. replan ───────────────────────────────────────────────────────

def test_replan_keeps_the_new_choice_and_replans_the_rest():
    backend, props, (x, w, y) = _myopia_host()
    res = plan_multi_circuits([4, 8], backend, props)
    ranked, _key = ranked_candidates(res['circuits'][0])
    alt = next(e for e in ranked if e is not res['circuits'][0]['chosen'])
    new = replan_from(res, 0, alt)
    assert new['circuits'][0]['chosen'] is alt
    first = _qubits(alt)
    ring = first | {n for q in first for n in _adjacency(backend).get(q, ())}
    assert _qubits(new['circuits'][1]['chosen']).isdisjoint(ring)
    assert set(new) == set(res)


def test_replan_index_out_of_range():
    backend, props = _make_backend(), _Props()
    res = plan_multi_circuits([8, 8], backend, props)
    with pytest.raises(IndexError, match=r'replan index 5 out of range '
                                         r'\(have 2 placed circuits\)'):
        replan_from(res, 5, res['circuits'][0]['chosen'])
    seq = sweep_multi_circuits([8, 8], backend, props)
    with pytest.raises(IndexError) as seq_err:
        multi_sweep.replan_from(seq, 5, seq['circuits'][0]['chosen'])
    with pytest.raises(IndexError) as joint_err:
        replan_from(res, 5, res['circuits'][0]['chosen'])
    assert str(seq_err.value) == str(joint_err.value)


# ── 9. engine pass-through ──────────────────────────────────────────

def test_a_hybrid_search_fn_runs_end_to_end():
    backend, props = heavy_hex_backend(5, seed=6)

    def _hybrid(norb, *, exclude_qubits, min_bridges):
        return hybrid_sweep(norb, backend, props, guided_max_k=1,
                            min_bridges=min_bridges,
                            exclude_qubits=exclude_qubits,
                            circuit_score_fn=lambda e: e['product_score'])

    res = plan_multi_circuits([5, 5], backend, props, search_fn=_hybrid,
                              min_bridges=1)
    assert res['failed_at'] is None and len(res['circuits']) == 2
    engines = {m['engine'] for c in res['circuits']
               for m in c['per_stratum'].values()}
    assert engines == {'guided', 'vf2'}
    assert res['total_lqs_circuit'] is not None


# ── 10. determinism ─────────────────────────────────────────────────

def test_identical_calls_give_identical_results():
    backend, props = heavy_hex_backend(5, seed=8)
    a = plan_multi_circuits([5, 5], backend, props, buffer_qubits=0,
                            boundary_penalty=0.2)
    b = plan_multi_circuits([5, 5], backend, props, buffer_qubits=0,
                            boundary_penalty=0.2)
    assert _strip(a) == _strip(b)


# ── result shape ────────────────────────────────────────────────────

def test_result_carries_every_sequential_key():
    backend, props = _make_backend(), _Props()
    seq = sweep_multi_circuits([8, 8], backend, props)
    res = plan_multi_circuits([8, 8], backend, props)
    assert set(seq) <= set(res)
    assert res['planner'] == 'joint'
    for key in ('beam_width', 'branch', 'boundary_penalty',
                'boundary_couplers', 'greedy_total', 'joint_total',
                'plan_source', 'n_searches'):
        assert key in res
    assert set(res['circuits'][0]) == set(seq['circuits'][0])

