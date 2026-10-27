"""Tests for the diverse VF2 reservoir (engine #17).

The reservoir exists because a blind cap keeps a spatially clustered sample:
VF2 yields mappings in traversal order, so the first ones are near-copies of
one region and much of a large lattice is never seen. These tests pin the
sieve's semantics, its agreement with the sweep's own score, and that it
actually widens the sample.
"""
import random

import pytest
import rustworkx

from iqmapper import reservoir as rsv
from iqmapper.ancilla_sweep import sweep_ancilla_layouts
from iqmapper.auto_lucj import (_collect_vf2_mappings,
                                create_lucj_zigzag_layout)
from iqmapper.reservoir import DiverseReservoir

from test_ancilla_sweep import FakeProps, make_backend


# ── helpers ─────────────────────────────────────────────────────────

def _ident(layout):
    """A trivial to_layout: the 'mapping' IS the layout."""
    return list(layout)


def _res(cap, score_fn=None, **kw):
    return DiverseReservoir(cap, score_fn or (lambda lay: 1.0), _ident, **kw)


# ── 1. offer semantics ──────────────────────────────────────────────

def test_fills_before_it_sieves():
    """Below the cap nothing is filtered — the sweep must still see every
    raw embedding, including both halves of each alpha/beta mirror."""
    r = _res(3)
    assert r.offer([0, 1, 2]) == 'kept'
    assert r.offer([0, 1, 3]) == 'kept'        # Jaccard 0.5, would pass anyway
    assert r.offer([2, 1, 0]) == 'kept'        # SAME qubit set, mirror-like
    assert len(r) == 3
    assert r.stats['skipped_similar'] == 0


def test_sieves_once_full():
    scores = {(0, 1, 2): 0.5, (0, 1, 3): 0.9, (7, 8, 9): 0.4}
    r = _res(2, lambda lay: scores[tuple(lay)])
    assert r.offer([0, 1, 2]) == 'kept'
    assert r.offer([7, 8, 9]) == 'kept'        # unrelated region
    # Full. (0,1,3) overlaps (0,1,2) by 2/4 = 0.5 < 0.8, so it is a NEW
    # region and must beat the weakest held candidate (0.4) — it does.
    assert r.offer([0, 1, 3]) == 'replaced'
    assert len(r) == 2


def test_a_near_duplicate_that_does_not_beat_its_rival_is_skipped():
    scores = {(0, 1, 2, 3): 0.9, (0, 1, 2, 4): 0.5, (5, 6, 7, 8): 0.8}
    r = _res(2, lambda lay: scores[tuple(lay)])
    r.offer([0, 1, 2, 3])
    r.offer([5, 6, 7, 8])
    # Jaccard with (0,1,2,3) = 3/5 = 0.6 -> new region, and 0.5 < 0.8 weakest
    assert r.offer([0, 1, 2, 4]) == 'skipped_similar'
    assert len(r) == 2
    assert r.stats['skipped_similar'] == 1


def test_a_better_candidate_displaces_the_weakest_when_full():
    """Full reservoir, new region, better than the worst held -> it gets in.

    This is what lets coverage keep growing after the cap is reached rather
    than freezing on whichever corner VF2 traversed first.
    """
    scores = {(0, 1, 2, 3): 0.5, (0, 1, 2, 3, 5): 0.9}
    r = _res(1, lambda lay: scores[tuple(lay)])
    assert r.offer([0, 1, 2, 3]) == 'kept'
    assert r.offer([0, 1, 2, 3, 5]) == 'replaced'
    assert r.layouts == [[0, 1, 2, 3, 5]]
    assert r.stats['replaced'] == 1


def test_identical_qubit_sets_are_the_same_region():
    scores = {(0, 1, 2, 3): 0.5, (3, 2, 1, 0): 0.9}
    r = _res(1, lambda lay: scores[tuple(lay)])
    r.offer([0, 1, 2, 3])
    # Same set, different order: Jaccard 1.0 -> the better one wins.
    assert r.offer([3, 2, 1, 0]) == 'replaced'
    assert r.layouts == [[3, 2, 1, 0]]


def test_exact_duplicates_are_skipped_via_key_fn():
    r = _res(5)
    assert r.offer([0, 1, 2]) == 'kept'
    assert r.offer([0, 1, 2]) == 'skipped_duplicate'
    assert len(r) == 1
    assert r.stats['skipped_duplicate'] == 1


def test_incomplete_mappings_are_counted_and_ignored():
    r = DiverseReservoir(4, lambda lay: 1.0, lambda m: [0, None, 2])
    assert r.offer({'x': 1}) == 'incomplete'
    assert len(r) == 0 and r.stats['incomplete'] == 1


def test_jaccard_none_disables_similarity():
    r = _res(3, jaccard=None)
    for lay in ([0, 1, 2], [0, 1, 3], [0, 1, 4]):
        assert r.offer(lay) == 'kept'
    assert len(r) == 3
    # Full: with no similarity notion nothing can displace anything.
    assert r.offer([0, 1, 5]) == 'skipped_similar'
    assert len(r) == 3


def test_the_index_is_updated_on_replace():
    """A stale index would let an evicted layout keep matching."""
    scores = {(0, 1, 2, 3): 0.1, (0, 1, 2, 4): 0.9, (50, 51, 52, 53): 0.5}
    r = _res(1, lambda lay: scores[tuple(lay)])
    r.offer([0, 1, 2, 3])
    r.offer([0, 1, 2, 4])                       # replaces it
    assert r.layouts == [[0, 1, 2, 4]]
    # qubit 3 belonged only to the evicted entry and must be gone.
    assert 3 not in r._by_qubit
    assert set(r._by_qubit) == {0, 1, 2, 4}
    assert list(r._keys) == [(0, 1, 2, 4)]


def test_offers_are_deterministic():
    def _run():
        r = _res(3, lambda lay: sum(lay) / 100.0)
        out = [r.offer(list(lay)) for lay in
               ([0, 1, 2], [0, 1, 3], [9, 8, 7], [0, 2, 3], [5, 5, 5])]
        return out, r.layouts
    assert _run() == _run()


def test_the_offer_budget_bounds_the_work():
    r = _res(2, offer_budget=5)
    for i in range(5):
        r.offer([i, i + 100, i + 200])
    assert r.exhausted is True
    assert r.stats['offered'] == 5


def test_summarise_names_the_hybrid_pass():
    text = rsv.summarise({'kept': 2000, 'skipped_similar': 34812}, k=1)
    assert text == ('hybrid k=1: cap reached — kept 2,000 diverse '
                    'candidates, skipped 34,812 near-duplicates')
    assert rsv.summarise({'kept': 0, 'skipped_similar': 0}, k=1) is None
    assert 'hybrid' not in rsv.summarise(
        {'kept': 5, 'skipped_similar': 1}, k=2, hybrid=False)


# ── 2. score parity with the sweep ──────────────────────────────────

def test_reservoir_score_equals_the_sweep_product_score():
    """One formula: what the reservoir sieves on is what the sweep ranks on."""
    backend, props = make_backend(), FakeProps()
    captured = {}
    import iqmapper.ancilla_sweep as sweep_mod
    original = sweep_mod._collect_vf2_mappings

    def _spy(graph, subgraph, cap=None, *, reservoir=None):
        if reservoir is not None:
            captured['reservoir'] = reservoir
        return original(graph, subgraph, cap=cap, reservoir=reservoir)

    sweep_mod._collect_vf2_mappings = _spy
    try:
        res = sweep_ancilla_layouts(4, backend, props, min_bridges=1,
                                    top_n=50, hybrid=False)
    finally:
        sweep_mod._collect_vf2_mappings = original

    reservoir = captured.get('reservoir')
    assert reservoir is not None, 'the sweep did not build a reservoir'
    by_set = {frozenset(lay): reservoir._score_fn(lay)
              for lay in reservoir.layouts}

    compared = 0
    k_max = res['k_max_effective']
    for entry in res['strata'][k_max]:
        used = frozenset(list(entry['layout']) + list(entry['ancillas']))
        if used in by_set:
            compared += 1
            assert by_set[used] == pytest.approx(entry['product_score'],
                                                 abs=1e-12)
    assert compared > 0, 'no shared layouts to compare'


# ── 3. parity paths ─────────────────────────────────────────────────

def test_diverse_cap_off_matches_diverse_cap_on_when_the_cap_is_not_reached():
    """On the fixture VF2 finishes well inside the cap, so the sieve never
    engages and the two results must be identical."""
    backend, props = make_backend(), FakeProps()
    off = sweep_ancilla_layouts(4, backend, props, min_bridges=1, top_n=25,
                                diverse_cap=False)
    on = sweep_ancilla_layouts(4, backend, props, min_bridges=1, top_n=25,
                               diverse_cap=True)

    assert off['n_raw_embeddings'] == on['n_raw_embeddings']
    assert set(off['strata']) == set(on['strata'])
    for k in off['strata']:
        for a, b in zip(off['strata'][k], on['strata'][k]):
            assert a['layout'] == b['layout']
            assert a['ancillas'] == b['ancillas']
            assert a['product_score'] == pytest.approx(b['product_score'],
                                                       abs=1e-12)
            assert a['source'] == b['source']


def test_diverse_cap_off_leaves_the_result_shape_untouched():
    backend, props = make_backend(), FakeProps()
    off = sweep_ancilla_layouts(4, backend, props, min_bridges=1, top_n=3,
                                diverse_cap=False)
    assert off['reservoir'] == {}
    assert off['reservoir_notices'] == []


# ── 4. the measurement that justifies the engine ────────────────────

def _synthetic_calibration(graph, seed=7):
    rng = random.Random(seed)
    readout = {q: (2e-4 if (q // 12) % 3 == 0 else 2e-3) * (1 + rng.random())
               for q in graph.node_indices()}
    edge_err = {frozenset(e): (5e-4 if (e[0] // 12) % 3 == 0 else 5e-3)
                * (1 + rng.random()) for e in graph.edge_list()}
    return readout, edge_err


def _mean_jaccard(sets, trials=5000, seed=1):
    rng = random.Random(seed)
    total = 0.0
    for _ in range(trials):
        a, b = rng.sample(sets, 2)
        total += len(a & b) / len(a | b)
    return total / trials


def test_the_reservoir_widens_the_sample_on_heavy_hex():
    """Deterministic, and the numbers are the point of the engine."""
    graph = rustworkx.generators.heavy_hex_graph(7)
    norb, cap = 8, 300
    template, n_bridges = create_lucj_zigzag_layout(
        num_orbitals=norb, backend_coupling_graph=graph, requested_bridges=1)
    size = 2 * norb + n_bridges
    readout, edge_err = _synthetic_calibration(graph)

    def _to_layout(mapping):
        layout = [None] * size
        for key, value in mapping.items():
            if value < size:
                layout[value] = key
        return layout

    def _score(layout):
        alpha, beta = layout[:norb], layout[norb:2 * norb]
        product = 1.0
        for q in layout[:2 * norb]:
            product *= 1 - readout[q]
        for j in range(norb - 1):
            product *= 1 - edge_err.get(frozenset((alpha[j], alpha[j + 1])),
                                        5e-3)
            product *= 1 - edge_err.get(frozenset((beta[j], beta[j + 1])),
                                        5e-3)
        for i in range(n_bridges):
            bphys = layout[2 * norb + i]
            product *= ((1 - readout[bphys])
                        * (1 - edge_err.get(frozenset((alpha[4 * i], bphys)),
                                            5e-3))
                        * (1 - edge_err.get(frozenset((bphys, beta[4 * i])),
                                            5e-3)))
        return product

    plain = [_to_layout(m) for m in
             _collect_vf2_mappings(graph, template, cap=cap)]
    reservoir = DiverseReservoir(cap, _score, _to_layout)
    diverse = [_to_layout(m) for m in
               _collect_vf2_mappings(graph, template, cap=cap,
                                     reservoir=reservoir)]

    assert len(plain) == len(diverse) == cap, 'both must fill the cap'

    plain_sets = [frozenset(l) for l in plain]
    diverse_sets = [frozenset(l) for l in diverse]

    # (a) less self-similar
    j_plain = _mean_jaccard(plain_sets)
    j_diverse = _mean_jaccard(diverse_sets)
    assert j_diverse < j_plain, (j_plain, j_diverse)

    # (b) the best candidate is not worse
    assert max(_score(l) for l in diverse) >= max(_score(l) for l in plain)

    # (c) more of the chip is seen
    covered_plain = len(set().union(*plain_sets))
    covered_diverse = len(set().union(*diverse_sets))
    assert covered_diverse > covered_plain, (covered_plain, covered_diverse)

    # (d) the work is bounded by the offer budget
    assert reservoir.stats['offered'] <= reservoir.offer_budget
    assert reservoir.stats['skipped_similar'] > 0


# ── 5. validity ─────────────────────────────────────────────────────

def test_every_kept_layout_is_complete():
    graph = rustworkx.generators.heavy_hex_graph(5)
    template, n_bridges = create_lucj_zigzag_layout(
        num_orbitals=4, backend_coupling_graph=graph, requested_bridges=1)
    size = 2 * 4 + n_bridges

    def _to_layout(mapping):
        layout = [None] * size
        for key, value in mapping.items():
            if value < size:
                layout[value] = key
        return layout

    reservoir = DiverseReservoir(50, lambda lay: 1.0, _to_layout)
    _collect_vf2_mappings(graph, template, cap=50, reservoir=reservoir)
    for layout in reservoir.layouts:
        assert len(layout) == size
        assert all(q is not None for q in layout)
        assert len(set(layout)) == size, 'a layout reused a qubit'


def test_summarise_reports_how_many_were_seen():
    """The '(of N seen)' clause carries what the old truncation warning
    said — that the ranking comes from a sample — so the two notices can be
    merged into one."""
    text = rsv.summarise({'kept': 2000, 'skipped_similar': 2498,
                          'offered': 4498}, k=3)
    assert text == ('hybrid k=3: cap reached — kept 2,000 diverse '
                    'candidates, skipped 2,498 near-duplicates '
                    '(of 4,498 seen)')
    # Absent 'offered' (older stats dicts) simply omits the clause.
    assert rsv.summarise({'kept': 5, 'skipped_similar': 1}, k=1).endswith(
        'near-duplicates')
