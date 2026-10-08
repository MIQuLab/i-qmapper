"""Guided lists each layout once; the joint planner hands back its complete
plans.

B1: the 1-opt polish can turn a finalist into a layout another candidate
already is (same qubits and ancillas, chains walked differently). The guided
sweep keeps the better-ranked copy, so the slot goes to a different layout.

B2: `plan_multi_circuits` returns `plans`, the distinct complete plans of its
final beams, best first, each carrying what a caller needs to swap it in.
"""
from collections import Counter

import pytest
import rustworkx

from iqmapper import guided_search as gs
from iqmapper.guided_search import guided_sweep
from iqmapper.joint_plan import plan_multi_circuits, replan_from

from test_combined_sweep import _lqs_scorer
from test_engine_select import heavy_hex_backend
from test_guided_induced import _SeededProps, _backend_from
from test_joint_plan import _myopia_host, _qubits
from test_multi_sweep import _Props, _make_backend

fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')


def _identity(entry):
    return (frozenset(entry['alpha']) | frozenset(entry['beta'])
            | frozenset(entry['ancillas']), frozenset(entry['ancillas']))


def _duplicates(result):
    return {k: sum(n - 1 for n in Counter(map(_identity, es)).values())
            for k, es in result['strata'].items()}


# ── B1: no repeated layout in a stratum ─────────────────────────────

def _fez():
    backend = fake_provider.FakeFez()
    return backend, backend.properties()


def _heavy_hex(d, seed):
    undirected = [tuple(e) for e in
                  rustworkx.generators.heavy_hex_graph(d).edge_list()]
    backend = _backend_from(undirected)
    return backend, _SeededProps(1 + max(max(e) for e in undirected), seed)


@pytest.mark.parametrize('host, norbs', [
    (lambda: _heavy_hex(5, 3), (6, 8)),
    (lambda: _heavy_hex(7, 11), (8, 12)),
    (_fez, (8, 12)),
])
def test_no_stratum_lists_a_layout_twice(host, norbs):
    backend, props = host()
    scorer = _lqs_scorer(backend, props)
    for norb in norbs:
        res = guided_sweep(norb, backend, props, min_bridges=1, top_n=10,
                           circuit_score_fn=scorer)
        assert set(_duplicates(res).values()) == {0}, (norb, _duplicates(res))


def test_a_repeated_layout_frees_its_slot_for_another(monkeypatch):
    """FakeFez norb=8: without the dedupe, k=2 repeats layouts. With it, the
    list keeps every layout it had and fills the freed slots with others."""
    backend, props = _fez()
    scorer = _lqs_scorer(backend, props)
    args = dict(min_bridges=1, top_n=10, circuit_score_fn=scorer)

    with monkeypatch.context() as m:
        m.setattr(gs, '_distinct_layouts', lambda ranked: list(ranked))
        before = guided_sweep(8, backend, props, **args)
    after = guided_sweep(8, backend, props, **args)

    assert _duplicates(before)[2] > 0, 'fixture no longer repeats a layout'
    old = set(map(_identity, before['strata'][2]))
    new = [_identity(e) for e in after['strata'][2]]
    assert len(new) == len(set(new)) == 10
    assert old <= set(new) and set(new) - old
    assert [e['rank_in_stratum'] for e in after['strata'][2]] == list(
        range(1, 11))


def test_distinct_layouts_keeps_the_first_copy():
    a = {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': [4]}
    a_again = {'alpha': [1, 0], 'beta': [3, 2], 'ancillas': [4]}
    b = {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': [5]}
    assert gs._distinct_layouts([a, a_again, b]) == [a, b]


# ── B2: the joint planner's complete plans ──────────────────────────

PLAN_KEYS = {'chosen', 'total', 'circuits', 'total_product',
             'total_lqs_circuit', 'placed_qubits', 'buffer_qubits_used',
             'boundary_couplers', 'joint_total', 'failed_at'}


def _check_plans(res, n_circuits, beam_width):
    plans = res['plans']
    assert 1 <= len(plans) <= beam_width
    for plan in plans:
        assert set(plan) == PLAN_KEYS
        assert plan['failed_at'] is None and len(plan['chosen']) == n_circuits
        assert [c['chosen'] for c in plan['circuits']] == plan['chosen']
        assert plan['total'] == pytest.approx(plan['joint_total'])
    footprints = [tuple(frozenset(_qubits(e)) for e in p['chosen'])
                  for p in plans]
    assert len(footprints) == len(set(footprints)), 'a plan is listed twice'
    totals = [p['total'] for p in plans]
    assert totals == sorted(totals, reverse=True)
    return plans


def test_plans_are_complete_distinct_and_best_first():
    backend, props = heavy_hex_backend(5, seed=2)
    res = plan_multi_circuits([5, 5], backend, props, min_bridges=1,
                              beam_width=4)
    plans = _check_plans(res, 2, 4)
    assert len(plans) > 1, 'fixture gives only one complete plan'


def test_the_returned_plan_is_plans_first():
    backend, props, _parts = _myopia_host()
    res = plan_multi_circuits([4, 8], backend, props)
    plans = _check_plans(res, 2, 4)
    assert plans[0]['chosen'] == [c['chosen'] for c in res['circuits']]
    assert plans[0]['total'] == pytest.approx(res['joint_total'])


def test_an_incomplete_run_has_no_complete_plans():
    backend, props = _make_backend(), _Props()
    res = plan_multi_circuits([8, 8, 8, 8], backend, props)
    assert res['failed_at'] == 2 and res['plans'] == []


def test_a_replan_hands_back_plans_for_its_own_prefix():
    backend, props = heavy_hex_backend(5, seed=2)
    res = plan_multi_circuits([5, 5], backend, props, min_bridges=1)
    alt = res['plans'][-1]['chosen'][0]
    new = replan_from(res, 0, alt)
    plans = _check_plans(new, 2, res['beam_width'])
    assert all(_qubits(p['chosen'][0]) == _qubits(alt) for p in plans)
