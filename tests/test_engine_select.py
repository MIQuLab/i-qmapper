"""Tests for the hybrid engine (guided below a k threshold, VF2 above).

Offline and deterministic. The edge cases must reduce to the frozen engines
exactly; the split case must be the two engines' own strata, side by side.
"""
import random
from types import SimpleNamespace

import pytest
import rustworkx

from iqmapper.ancilla_sweep import sweep_ancilla_layouts
from iqmapper.engine_select import hybrid_sweep
from iqmapper.guided_search import guided_sweep

from test_ancilla_sweep import FakeProps, make_backend


def heavy_hex_backend(d=5, seed=3):
    """heavy_hex_graph(d) as a backend, with seeded random calibrations."""
    graph = rustworkx.generators.heavy_hex_graph(d)
    directed = rustworkx.PyDiGraph()
    for _ in range(graph.num_nodes()):
        directed.add_node(None)
    for u, v in graph.edge_list():
        directed.add_edge(u, v, None)
        directed.add_edge(v, u, None)
    rng = random.Random(seed)
    readout = {q: rng.uniform(0.002, 0.04) for q in range(graph.num_nodes())}
    gate = {frozenset(e): rng.uniform(0.001, 0.02) for e in graph.edge_list()}

    class _Props:
        def readout_error(self, q):
            return readout[q]

        def gate_error(self, name, edge):
            return gate[frozenset(edge)]

    backend = SimpleNamespace(coupling_map=SimpleNamespace(graph=directed),
                              operation_names=['cz', 'sx', 'rz', 'x'],
                              name=f'heavy_hex_{d}')
    return backend, _Props()


def _score(entry):
    return entry['product_score']


def _untagged(per_stratum):
    return {k: {f: v for f, v in m.items() if f != 'engine'}
            for k, m in per_stratum.items()}


@pytest.fixture(scope='module')
def split_case():
    """norb=9 on heavy_hex(5): k_max=3, VF2 capped at k=1."""
    backend, props = heavy_hex_backend()
    kw = dict(min_bridges=1, top_n=3, circuit_score_fn=_score)
    hyb = hybrid_sweep(9, backend, props, guided_max_k=1, **kw)
    guided = guided_sweep(9, backend, props, **kw)
    vf2 = sweep_ancilla_layouts(9, backend, props, **dict(kw, min_bridges=2))
    return hyb, guided, vf2


def test_pure_vf2_when_threshold_is_below_the_floor():
    backend, props = make_backend(), FakeProps()
    kw = dict(min_bridges=2, top_n=3, circuit_score_fn=_score)
    hyb = hybrid_sweep(8, backend, props, guided_max_k=1, **kw)
    ref = sweep_ancilla_layouts(8, backend, props, **kw)
    assert hyb['strata'] == ref['strata']
    assert _untagged(hyb['per_stratum']) == ref['per_stratum']
    assert {m['engine'] for m in hyb['per_stratum'].values()} == {'vf2'}
    assert hyb['refined_count'] == 0 and hyb['refine_attempted'] == 0


def test_pure_guided_when_threshold_reaches_k_max():
    backend, props = make_backend(), FakeProps()
    kw = dict(min_bridges=1, top_n=3, circuit_score_fn=_score)
    ref = guided_sweep(8, backend, props, **kw)
    hyb = hybrid_sweep(8, backend, props, guided_max_k=ref['k_max_effective'],
                       **kw)
    assert hyb['strata'] == ref['strata']
    assert _untagged(hyb['per_stratum']) == ref['per_stratum']
    assert {m['engine'] for m in hyb['per_stratum'].values()} == {'guided'}
    assert hyb['vf2_truncated'] is False and hyb['reservoir_notices'] == []
    assert hyb['mapping_cap'] is None


def test_split_takes_each_stratum_from_its_engine(split_case):
    hyb, guided, vf2 = split_case
    assert hyb['k_max_effective'] >= 3
    assert list(hyb['strata']) == sorted(hyb['strata'], reverse=True)
    for k, entries in hyb['strata'].items():
        engine = hyb['per_stratum'][k]['engine']
        if k <= 1:
            assert engine == 'guided'
            assert entries == guided['strata'][k]
        else:
            assert engine == 'vf2'
            assert entries == vf2['strata'][k]
    assert set(hyb['strata']) == set(range(hyb['k_max_effective'], 0, -1))
    assert hyb['vf2_truncated'] == vf2['vf2_truncated']
    assert hyb['reservoir_notices'] == vf2['reservoir_notices']
    assert hyb['refined_count'] == guided['refined_count']
    assert hyb['n_raw_embeddings'] == (vf2['n_raw_embeddings']
                                       + guided['n_raw_embeddings'])
    assert hyb['min_bridges'] == 1 and hyb['guided_max_k'] == 1
    assert hyb['engine'] == 'hybrid'


def test_result_carries_every_vf2_key(split_case):
    hyb, _guided, vf2 = split_case
    assert set(vf2) <= set(hyb)


def test_exclusions_reach_both_halves():
    backend, props = heavy_hex_backend()
    excluded = {0, 1, 2, 3, 20, 21, 40, 41, 60, 61}
    res = hybrid_sweep(9, backend, props, guided_max_k=1, min_bridges=1,
                       exclude_qubits=excluded)
    engines = {m['engine'] for m in res['per_stratum'].values()}
    assert engines == {'guided', 'vf2'}
    assert res['excluded_qubits'] == sorted(excluded)
    for entries in res['strata'].values():
        for e in entries:
            used = set(e['alpha']) | set(e['beta']) | set(e['ancillas'])
            assert used.isdisjoint(excluded)


def test_threshold_is_required():
    backend, props = make_backend(), FakeProps()
    with pytest.raises(ValueError, match='guided_max_k'):
        hybrid_sweep(8, backend, props, guided_max_k=None)


def test_out_of_range_floor_raises_the_engine_text():
    backend, props = make_backend(), FakeProps()
    with pytest.raises(ValueError, match='min_bridges=9 out of range'):
        hybrid_sweep(8, backend, props, guided_max_k=1, min_bridges=9)
