"""The combined search: exhaustive and guided on every stratum, best of both.

The guarantee that matters is per stratum: the combined best is never worse
than either engine's best. Around it: a layout found by both is listed once,
every entry says where it came from, and with the guided half off the result
is the exhaustive sweep's own.
"""
import numpy as np
import pytest
import rustworkx

from iqmapper import ancilla_sweep, engine_select, guided_search
from iqmapper.ancilla_sweep import sweep_ancilla_layouts
from iqmapper.auto_lucj import _make_backend_cmap_pygraph
from iqmapper.engine_select import combined_sweep
from iqmapper.guided_search import guided_sweep

from test_ancilla_sweep import FakeProps, make_backend
from test_guided_induced import _SeededProps, _backend_from

SOURCES = {'exhaustive', 'guided', 'both'}

#: Keeps the exhaustive half quick on the larger lattices. A capped VF2 is
#: also where the guided half has the most to add.
CAP = 300


def _identity(entry):
    return (frozenset(entry['alpha']) | frozenset(entry['beta'])
            | frozenset(entry['ancillas']), frozenset(entry['ancillas']))


def _lqs_scorer(backend, props, gate='cz'):
    """Circuit-level LQS shape: (1 - readout) over the layout's qubits times
    (1 - gate error) over every coupler between two of them, once each."""
    graph = _make_backend_cmap_pygraph(backend=backend)
    neighbours = {q: set(graph.neighbors(q)) for q in graph.node_indices()}
    readout, gate_err = {}, {}

    def _r(q):
        if q not in readout:
            readout[q] = float(props.readout_error(q))
        return readout[q]

    def _e(u, v):
        key = (u, v) if u < v else (v, u)
        if key not in gate_err:
            gate_err[key] = float(props.gate_error(gate, key))
        return gate_err[key]

    def _score(entry):
        qubits = set(entry['alpha']) | set(entry['beta']) | set(entry['ancillas'])
        fid = 1.0
        for q in qubits:
            fid *= 1.0 - _r(q)
            for n in neighbours.get(q, ()):
                if n in qubits and q < n:
                    fid *= 1.0 - _e(q, n)
        return fid
    return _score


def _best(result, k):
    entries = result['strata'].get(k) or []
    return max((e['lqs_circuit'] for e in entries
                if e.get('lqs_circuit') is not None), default=None)


def _assert_best_of_both(norb, backend, props, **kw):
    scorer = _lqs_scorer(backend, props)
    args = dict(min_bridges=1, top_n=3, circuit_score_fn=scorer, **kw)
    vf2 = sweep_ancilla_layouts(norb, backend, props, mapping_cap=CAP, **args)
    guided = guided_sweep(norb, backend, props, **args)
    combined = combined_sweep(norb, backend, props, mapping_cap=CAP, **args)
    assert set(combined['strata']) == set(vf2['strata'])
    for k in combined['strata']:
        engines = [b for b in (_best(vf2, k), _best(guided, k)) if b is not None]
        best = _best(combined, k)
        assert best is not None and best >= max(engines), (
            f'norb={norb} k={k}: combined {best} < engines {engines}')
    return vf2, guided, combined


# ── best of both, per stratum ───────────────────────────────────────

@pytest.mark.parametrize('d, norb', [(5, 6), (5, 8), (7, 12)])
def test_combined_best_is_at_least_either_engine_on_heavy_hex(d, norb):
    undirected = [tuple(e) for e in
                  rustworkx.generators.heavy_hex_graph(d).edge_list()]
    backend = _backend_from(undirected)
    props = _SeededProps(1 + max(max(e) for e in undirected), seed=d + norb)
    _assert_best_of_both(norb, backend, props)


@pytest.mark.parametrize('norb', [8, 12])
def test_combined_best_is_at_least_either_engine_on_fake_fez(norb):
    fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
    backend = fake_provider.FakeFez()
    _assert_best_of_both(norb, backend, backend.properties())


def test_combined_best_holds_with_the_diversity_quota():
    backend, props = make_backend(), FakeProps()
    _assert_best_of_both(6, backend, props, diversity_threshold=0.8)


# ── dedupe and source tags ──────────────────────────────────────────

def test_a_layout_found_by_both_engines_is_listed_once_as_both():
    backend, props = make_backend(), FakeProps()
    scorer = _lqs_scorer(backend, props)
    args = dict(min_bridges=1, top_n=5, circuit_score_fn=scorer)
    vf2 = sweep_ancilla_layouts(4, backend, props, **args)
    guided = guided_sweep(4, backend, props, **args)
    combined = combined_sweep(4, backend, props, **args)

    saw_both = False
    for k, entries in combined['strata'].items():
        keys = [_identity(e) for e in entries]
        assert len(keys) == len(set(keys)), f'duplicate layout at k={k}'
        in_vf2 = {_identity(e) for e in vf2['strata'][k]}
        in_guided = {_identity(e) for e in guided['strata'][k]}
        for entry, key in zip(entries, keys):
            assert entry['source'] in SOURCES
            expected = ('both' if key in in_vf2 and key in in_guided
                        else 'guided' if key in in_guided else 'exhaustive')
            assert entry['source'] == expected, (k, key)
            saw_both = saw_both or expected == 'both'
        assert [e['rank_in_stratum'] for e in entries] == list(
            range(1, len(entries) + 1))
        assert len(entries) <= 5
    assert saw_both, 'fixture no longer has a layout both engines find'


def _entry(alpha, beta, ancillas, lqs):
    return {'layout': alpha + beta, 'alpha': alpha, 'beta': beta,
            'ancillas': ancillas, 'product_score': lqs, 'lqs_circuit': lqs,
            'rank_in_stratum': 1, 'source': 'stub'}


def _sweep(strata):
    return {'strata': strata, 'k_max_effective': 2, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 1,
            'n_candidates_scored': 1, 'excluded_qubits': [],
            'per_stratum': {}, 'mapping_cap': 2000}


def test_guided_winners_lists_strata_whose_best_only_guided_found(
        monkeypatch):
    shared = ([0, 1], [2, 3], [4, 5])
    monkeypatch.setattr(ancilla_sweep, 'sweep_ancilla_layouts',
                        lambda *a, **k: _sweep({
                            2: [_entry(*shared, 0.5)],
                            1: [_entry([0, 1], [2, 3], [4], 0.6)]}))
    monkeypatch.setattr(guided_search, 'guided_sweep',
                        lambda *a, **k: _sweep({
                            2: [_entry(*shared, 0.5)],
                            1: [_entry([6, 7], [8, 9], [10], 0.7)]}))
    res = combined_sweep(2, None, None, min_bridges=1, top_n=3,
                         circuit_score_fn=lambda e: e['lqs_circuit'])
    assert res['guided_winners'] == [1]
    assert [e['source'] for e in res['strata'][2]] == ['both']
    assert [e['source'] for e in res['strata'][1]] == ['guided', 'exhaustive']
    assert [e['lqs_circuit'] for e in res['strata'][1]] == [0.7, 0.6]


def test_a_layout_guided_lists_twice_stays_guided(monkeypatch):
    """Guided can return one layout twice (same qubits, another chain
    order). The second copy must not turn the first into 'both'."""
    twin_a = _entry([6, 7], [8, 9], [10], 0.7)
    twin_b = _entry([7, 6], [9, 8], [10], 0.7)
    monkeypatch.setattr(ancilla_sweep, 'sweep_ancilla_layouts',
                        lambda *a, **k: _sweep({
                            1: [_entry([0, 1], [2, 3], [4], 0.6)]}))
    monkeypatch.setattr(guided_search, 'guided_sweep',
                        lambda *a, **k: _sweep({1: [twin_a, twin_b]}))
    res = combined_sweep(2, None, None, min_bridges=1, top_n=3,
                         circuit_score_fn=lambda e: e['lqs_circuit'])
    assert [(e['lqs_circuit'], e['source']) for e in res['strata'][1]] == [
        (0.7, 'guided'), (0.6, 'exhaustive')]
    assert res['guided_winners'] == [1]


def test_the_diversity_quota_is_the_exhaustive_sweeps_own():
    assert (engine_select._select_with_diversity
            is ancilla_sweep._select_with_diversity)


# ── guided off reproduces the exhaustive sweep ──────────────────────

def _strip(result, extra_keys):
    out = {k: v for k, v in result.items() if k not in extra_keys}
    out['strata'] = {k: [{f: v for f, v in e.items() if f != 'source'}
                         for e in entries]
                     for k, entries in result['strata'].items()}
    return out


@pytest.mark.parametrize('kw', [
    {},
    {'circuit_score_fn': 'lqs'},
    {'circuit_score_fn': 'lqs', 'diversity_threshold': 0.8},
])
def test_without_guided_the_result_is_the_exhaustive_sweep(monkeypatch, kw):
    backend, props = make_backend(), FakeProps()
    if kw.get('circuit_score_fn') == 'lqs':
        kw = dict(kw, circuit_score_fn=_lqs_scorer(backend, props))
    vf2 = sweep_ancilla_layouts(6, backend, props, min_bridges=1, top_n=4,
                                **kw)
    monkeypatch.setattr(guided_search, 'guided_sweep',
                        lambda *a, **k: pytest.fail('guided ran'))
    res = combined_sweep(6, backend, props, use_guided=False, min_bridges=1,
                         top_n=4, beam_width=50, **kw)

    added = {'engine', 'use_guided', 'guided_winners', 'guided_error'}
    assert added <= set(res)
    assert res['engine'] == 'combined' and res['use_guided'] is False
    assert res['guided_winners'] == [] and res['guided_error'] is None
    assert _strip(res, added) == _strip(vf2, set())
    assert all(e['source'] == 'exhaustive'
               for entries in res['strata'].values() for e in entries)


# ── routing and failure ─────────────────────────────────────────────

def test_engine_specific_arguments_reach_their_own_engine(monkeypatch):
    seen = {}

    def _spy(name):
        def _run(norb, backend, props, **k):
            seen[name] = k
            return {'strata': {}, 'k_max_effective': 1, 'min_bridges': 1,
                    'vf2_truncated': False, 'n_raw_embeddings': 0,
                    'n_candidates_scored': 0, 'excluded_qubits': [],
                    'per_stratum': {}, 'mapping_cap': 2000}
        return _run

    monkeypatch.setattr(ancilla_sweep, 'sweep_ancilla_layouts', _spy('vf2'))
    monkeypatch.setattr(guided_search, 'guided_sweep', _spy('guided'))
    combined_sweep(8, None, None, min_bridges=1, top_n=3, exclude_qubits={2},
                   circuit_score_fn=None, diversity_threshold=0.8,
                   hybrid=False, mapping_cap=500, diverse_cap=True,
                   beam_width=64, refine_top=2)
    shared = {'min_bridges', 'top_n', 'exclude_qubits', 'circuit_score_fn',
              'diversity_threshold'}
    assert set(seen['vf2']) == shared | {'hybrid', 'mapping_cap',
                                         'diverse_cap'}
    assert set(seen['guided']) == shared | {'beam_width', 'refine_top'}
    assert seen['vf2']['exclude_qubits'] == seen['guided']['exclude_qubits']


def test_an_unknown_argument_is_refused():
    with pytest.raises(TypeError, match='beam_wdth'):
        combined_sweep(4, make_backend(), FakeProps(), beam_wdth=3)


def test_a_guided_dead_end_leaves_the_exhaustive_result():
    """norb=6, k=2 on the ladder: a 5-wide beam completes nothing."""
    backend, props = make_backend(), FakeProps()
    vf2 = sweep_ancilla_layouts(6, backend, props, min_bridges=2, top_n=3)
    res = combined_sweep(6, backend, props, min_bridges=2, top_n=3,
                         beam_width=5)
    assert 'no complete zigzag embedding' in res['guided_error']
    assert res['guided_winners'] == []
    assert ([[_identity(e) for e in v] for v in res['strata'].values()]
            == [[_identity(e) for e in v] for v in vf2['strata'].values()])


def test_run_level_fields_come_from_the_exhaustive_half():
    backend, props = make_backend(), FakeProps()
    vf2 = sweep_ancilla_layouts(6, backend, props, min_bridges=1, top_n=3)
    guided = guided_sweep(6, backend, props, min_bridges=1, top_n=3)
    res = combined_sweep(6, backend, props, min_bridges=1, top_n=3)
    for key in ('vf2_truncated', 'reservoir', 'reservoir_notices',
                'mapping_cap', 'k_max_effective', 'min_bridges',
                'excluded_qubits', 'per_stratum'):
        assert res[key] == vf2[key], key
    assert res['n_candidates_scored'] == (vf2['n_candidates_scored']
                                          + guided['n_candidates_scored'])
    assert res['n_raw_embeddings'] == (vf2['n_raw_embeddings']
                                       + guided['n_raw_embeddings'])
    assert res['beam_width'] == guided['beam_width']
