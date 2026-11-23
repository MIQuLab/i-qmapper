"""Tests for the calibration-guided beam search.

The engine is a heuristic, so the tests are about the things that must hold
regardless: every layout is a valid embedding of the LUCJ template, the scores
agree with the frozen engine to the last bit, the search is deterministic, and
the output dict is indistinguishable from `sweep_ancilla_layouts`'s apart from
the additive keys.
"""
import math
from types import SimpleNamespace

import pytest
import rustworkx

from iqmapper import guided_search as gs
from iqmapper.ancilla_sweep import sweep_ancilla_layouts
from iqmapper.guided_search import guided_sweep

from test_ancilla_sweep import FakeProps, make_backend


# ── helpers ─────────────────────────────────────────────────────────

def _adjacency(backend):
    from iqmapper.auto_lucj import _make_backend_cmap_pygraph
    graph = _make_backend_cmap_pygraph(backend=backend)
    return {q: frozenset(graph.neighbors(q)) for q in graph.node_indices()}


def _engine_for(backend, props, norb):
    """A `_GuidedEngine` wired to a backend the way `guided_sweep` wires it."""
    import numpy as np
    from iqmapper.ancilla_sweep import (_UNCALIBRATED_PENALTY,
                                        _safe_gate_error, _safe_readout)
    from iqmapper.auto_lucj import (_get_two_q_gate_name,
                                    _make_backend_cmap_pygraph)
    graph = _make_backend_cmap_pygraph(backend=backend)
    gate_name = _get_two_q_gate_name(backend)
    nodes = list(graph.node_indices())
    readout = np.full(max(nodes) + 1, _UNCALIBRATED_PENALTY, dtype=float)
    for q in nodes:
        readout[q] = _safe_readout(props, q)
    edge_err = {frozenset((u, v)): _safe_gate_error(props, gate_name, u, v)
                for u, v in graph.edge_list()}
    engine = gs._GuidedEngine(norb, graph, readout, edge_err)
    return engine, norb


def _all_entries(result):
    for k in sorted(result['strata'], reverse=True):
        for entry in result['strata'][k]:
            yield k, entry


def _entry_layout(entry):
    return list(entry['layout']) + list(entry['ancillas'])


# ── 1. validity ─────────────────────────────────────────────────────

def test_every_produced_layout_is_a_valid_template_embedding():
    """Injective, and every template edge lands on a coupling edge."""
    backend, props = make_backend(), FakeProps()
    adjacency = _adjacency(backend)
    for norb in (4, 6, 8):
        res = guided_sweep(norb, backend, props, min_bridges=1, top_n=3,
                           circuit_score_fn=lambda e: e['product_score'])
        assert res['strata'], f'no strata for norb={norb}'
        for k, entry in _all_entries(res):
            layout = _entry_layout(entry)
            assert gs.validate_layout(layout, norb, k, adjacency), (
                f'invalid embedding at norb={norb} k={k}: {layout}')
            assert entry['bridge_slots'] == gs.bridge_slots_for(k), (
                'prefix contract violated: bridge_slots must be '
                '(0, 4, ..., 4*(k-1))')


def test_guided_layouts_are_contained_in_the_vf2_enumeration():
    """On a fixture VF2 can fully enumerate, guided finds nothing exotic.

    VF2 must be given a top_n large enough that its slice is not binding:
    at norb=4 its k=1 pool holds ~564 layouts, so a top_n of 50 would compare
    guided's picks against VF2's shortlist rather than its enumeration.
    """
    backend, props = make_backend(), FakeProps()
    norb = 4
    vf2 = sweep_ancilla_layouts(norb, backend, props, min_bridges=1,
                                top_n=5000)
    guided = guided_sweep(norb, backend, props, min_bridges=1, top_n=50)

    for k in guided['strata']:
        vf2_keys = {(frozenset(e['layout']), frozenset(e['ancillas']))
                    for e in vf2['strata'].get(k, [])}
        for entry in guided['strata'][k]:
            key = (frozenset(entry['layout']), frozenset(entry['ancillas']))
            assert key in vf2_keys, (
                f'guided produced a k={k} layout VF2 never enumerated: {key}')


def test_validate_layout_rejects_the_obvious_breakages():
    adjacency = {0: frozenset({1}), 1: frozenset({0, 2}), 2: frozenset({1})}
    assert gs.validate_layout([0, 1, 2], 1, 0, adjacency) is False  # wrong len
    # duplicate qubit -> not injective
    assert gs.validate_layout([0, 1, 0, 1], 2, 0, adjacency) is False
    # None slot
    assert gs.validate_layout([0, 1, None, 2], 2, 0, adjacency) is False


def test_template_edges_match_the_frozen_construction():
    """Chain paths plus two edges per bridge, bridges at slots 0, 4, 8..."""
    edges = gs.template_edges(norb=8, k=2)
    for j in range(7):
        assert (j, j + 1) in edges
        assert (8 + j, 8 + j + 1) in edges
    assert (0, 16) in edges and (16, 8) in edges       # bridge 0 at slot 0
    assert (4, 17) in edges and (17, 12) in edges      # bridge 1 at slot 4
    assert gs.max_bridges_for(8) == 2
    assert gs.max_bridges_for(9) == 3
    assert gs.bridge_slots_for(3) == (0, 4, 8)


# ── 2. score parity with the frozen engine ──────────────────────────

def test_score_parity_with_the_frozen_sweep():
    """A layout both engines found must carry identical scores."""
    backend, props = make_backend(), FakeProps()
    norb = 6
    vf2 = sweep_ancilla_layouts(norb, backend, props, min_bridges=1, top_n=50)
    guided = guided_sweep(norb, backend, props, min_bridges=1, top_n=50)

    compared = 0
    for k in guided['strata']:
        vf2_by_key = {(frozenset(e['layout']), frozenset(e['ancillas'])): e
                      for e in vf2['strata'].get(k, [])}
        for entry in guided['strata'][k]:
            key = (frozenset(entry['layout']), frozenset(entry['ancillas']))
            twin = vf2_by_key.get(key)
            if twin is None:
                continue
            compared += 1
            assert entry['product_score'] == pytest.approx(
                twin['product_score'], abs=1e-12), (k, key)
            assert entry['additive_score'] == pytest.approx(
                twin['additive_score'], abs=1e-12), (k, key)
    assert compared > 0, 'no shared layouts to compare'


# ── 3. determinism and beam monotonicity ────────────────────────────

def test_same_inputs_give_an_identical_result():
    backend, props = make_backend(), FakeProps()
    a = guided_sweep(6, backend, props, min_bridges=1, top_n=3, seed=0)
    b = guided_sweep(6, backend, props, min_bridges=1, top_n=3, seed=0)
    for k in a['strata']:
        assert [e['layout'] for e in a['strata'][k]] == \
               [e['layout'] for e in b['strata'][k]]
        assert [e['product_score'] for e in a['strata'][k]] == \
               [e['product_score'] for e in b['strata'][k]]


def test_a_wider_beam_finds_a_superset_of_completed_layouts():
    """Widening the beam must not lose a completed layout.

    Asserted on the beam itself rather than on `strata`: the strata are a
    top_n slice, so a wider beam can legitimately push a narrow run's pick off
    the shortlist by finding better ones. The monotonicity that matters is
    that the wider search still REACHED it.
    """
    backend, props = make_backend(), FakeProps()
    engine, norb = _engine_for(backend, props, norb=6)
    k = 2

    narrow = {s.key() for s in engine.run_beam(k, 50)}
    engine.expansions = 0
    wide = {s.key() for s in engine.run_beam(k, 200)}

    assert narrow, 'the narrow beam completed nothing'
    assert narrow <= wide, (
        f'widening the beam dropped {len(narrow - wide)} completed layout(s)')
    assert len(wide) > len(narrow), 'a wider beam should reach more layouts'


def test_a_too_narrow_beam_dead_ends_loudly():
    """Beam search is incomplete: a narrow beam can finish with nothing.

    On the fixture at norb=6, k=2 the beam completes nothing below width ~40 —
    every state it kept ran out of legal extensions. `guided_sweep` must raise
    rather than return an empty result that looks like 'no layout exists on
    this device', because the two mean very different things to a user.
    """
    backend, props = make_backend(), FakeProps()
    engine, _norb = _engine_for(backend, props, norb=6)
    assert engine.run_beam(2, 5) == [], 'fixture assumption changed'

    with pytest.raises(RuntimeError, match='no complete zigzag embedding'):
        guided_sweep(6, backend, props, min_bridges=2, top_n=3, beam_width=5)


# ── 4. the fixture optimum ──────────────────────────────────────────

def test_guided_finds_the_same_optimum_as_vf2_at_every_k():
    """The fixture has a GOOD and a BAD block; guided must pick GOOD.

    On this fixture VF2 enumerates exhaustively, so its best IS the optimum —
    which makes this a real check on the heuristic, not a tautology.
    """
    backend, props = make_backend(), FakeProps()
    for norb in (4, 6, 8):
        vf2 = sweep_ancilla_layouts(norb, backend, props, min_bridges=1,
                                    top_n=3)
        guided = guided_sweep(norb, backend, props, min_bridges=1, top_n=3)
        assert set(guided['strata']) == set(vf2['strata'])
        for k in vf2['strata']:
            best_vf2 = vf2['strata'][k][0]['product_score']
            best_guided = guided['strata'][k][0]['product_score']
            assert best_guided == pytest.approx(best_vf2, abs=1e-12), (
                f'norb={norb} k={k}: guided {best_guided} vs VF2 {best_vf2}')


def test_guided_picks_the_good_block():
    from test_ancilla_sweep import GOOD_QUBITS
    backend, props = make_backend(), FakeProps()
    guided = guided_sweep(6, backend, props, min_bridges=1, top_n=1)
    for k, entry in _all_entries(guided):
        used = set(_entry_layout(entry))
        assert used <= set(GOOD_QUBITS), (
            f'k={k} winner strayed into the BAD block: {sorted(used)}')


# ── 5. refinement ───────────────────────────────────────────────────

def _twin_bridge_backend():
    """A minimal graph with TWO candidate bridges for the same chain pair.

    alpha = 0-1, beta = 2-3, and both 4 and 5 are adjacent to 0 and 2. That
    gives the bridge-replacement move a real alternative to find, which the
    ladder fixture does not (there the bridge is forced).
    """
    graph = rustworkx.PyDiGraph()
    for _ in range(6):
        graph.add_node(None)

    def _add(u, v):
        graph.add_edge(u, v, None)
        graph.add_edge(v, u, None)

    _add(0, 1)                       # alpha chain
    _add(2, 3)                       # beta chain
    for bridge in (4, 5):            # two interchangeable bridges
        _add(0, bridge)
        _add(bridge, 2)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


class _TunableProps:
    def __init__(self, readout=None, gate=None, default_r=0.01,
                 default_e=0.01):
        self._r = dict(readout or {})
        self._g = {frozenset(k): v for k, v in (gate or {}).items()}
        self._dr, self._dg = default_r, default_e

    def readout_error(self, q):
        return self._r.get(int(q), self._dr)

    def gate_error(self, _name, edge):
        return self._g.get(frozenset((int(edge[0]), int(edge[1]))), self._dg)


def _refine_engine(backend, props, norb):
    engine, _ = _engine_for(backend, props, norb)
    return engine


def test_refinement_takes_a_strictly_better_bridge():
    """Bridge 5 is better than bridge 4; refinement must swap and flag it."""
    backend = _twin_bridge_backend()
    props = _TunableProps(readout={4: 0.20, 5: 0.001})
    adjacency = _adjacency(backend)
    engine = _refine_engine(backend, props, norb=2)

    entry = gs._entry_from_state(engine, [0, 1], [2, 3], [4], 1)
    assert gs.validate_layout(_entry_layout(entry), 2, 1, adjacency)
    assert entry['ancillas'] == [4]

    def _score(e):
        return float(e['product_score'])          # bridge 5 scores higher

    refined, changed = gs.refine_entry(engine, entry, 1, _score)

    assert changed is True, 'refinement ignored a strictly better bridge'
    assert refined['ancillas'] == [5], (
        f"expected the good bridge, got {refined['ancillas']}")
    assert refined['lqs_circuit'] > entry['product_score']
    assert gs.validate_layout(_entry_layout(refined), 2, 1, adjacency), (
        'refinement produced an invalid embedding')


def test_refinement_leaves_an_already_optimal_entry_alone():
    backend = _twin_bridge_backend()
    props = _TunableProps(readout={4: 0.001, 5: 0.20})
    engine = _refine_engine(backend, props, norb=2)
    entry = gs._entry_from_state(engine, [0, 1], [2, 3], [4], 1)

    refined, changed = gs.refine_entry(engine, entry, 1,
                                       lambda e: float(e['product_score']))
    assert changed is False, 'refinement moved off the optimum'
    assert refined is entry


def test_refinement_respects_max_iter():
    """A scorer that always improves must still terminate."""
    backend = _twin_bridge_backend()
    props = _TunableProps()
    engine = _refine_engine(backend, props, norb=2)
    entry = gs._entry_from_state(engine, [0, 1], [2, 3], [4], 1)

    calls = []

    def _always_better(_e):
        calls.append(1)
        return 0.1 * len(calls)                  # strictly increasing forever

    refined, changed = gs.refine_entry(engine, entry, 1, _always_better,
                                       max_iter=3)
    assert changed is True
    # Bounded: 3 iterations, each scanning a neighbourhood of a handful of
    # moves. Without the ceiling this scorer would loop forever.
    assert len(calls) <= 3 * 20 + 1, (
        f'max_iter did not bound the search ({len(calls)} scorer calls)')


def test_a_raising_scorer_never_breaks_refinement():
    backend = _twin_bridge_backend()
    engine = _refine_engine(backend, _TunableProps(), norb=2)
    entry = gs._entry_from_state(engine, [0, 1], [2, 3], [4], 1)

    def _boom(_e):
        raise RuntimeError('scorer exploded')

    refined, changed = gs.refine_entry(engine, entry, 1, _boom)
    assert changed is False and refined is entry


# ── 6. no explosion at low k ────────────────────────────────────────

def test_low_k_does_not_explode_on_heavy_hex():
    """The failure mode this engine exists to fix, asserted structurally.

    At k<=1 on a large lattice VF2 finds far more mappings than its cap and
    gets cut off. The beam is bounded by construction: expansions can never exceed
    beam_width x norb x max_degree^2, so the assertion is on a counter rather
    than on wall time (which would be flaky on CI).
    """
    graph_rx = rustworkx.generators.heavy_hex_graph(7)
    directed = rustworkx.PyDiGraph()
    for _ in range(graph_rx.num_nodes()):
        directed.add_node(None)
    for u, v in graph_rx.edge_list():
        directed.add_edge(u, v, None)
        directed.add_edge(v, u, None)
    backend = SimpleNamespace(coupling_map=SimpleNamespace(graph=directed),
                              operation_names=['cz', 'sx', 'rz', 'x'])
    props = _TunableProps()

    norb, beam_width = 6, 50
    res = guided_sweep(norb, backend, props, min_bridges=1, top_n=3,
                       beam_width=beam_width)

    max_degree = max(len(graph_rx.neighbors(n))
                     for n in graph_rx.node_indices())
    # Per growth step: beam_width states x (deg choices for alpha) x (deg for
    # beta). Summed over the norb-1 steps, and over the strata we run.
    ceiling = beam_width * norb * (max_degree ** 2) * (res['k_max_effective'] + 1)
    assert res['expansions'] <= ceiling, (
        f"expansions {res['expansions']} exceeded the structural bound "
        f"{ceiling}")
    assert res['vf2_truncated'] is False, 'the beam must never report a cap'
    assert res['mapping_cap'] is None


# ── 7. output contract ──────────────────────────────────────────────

def test_result_dict_is_a_superset_of_the_sweep_contract():
    backend, props = make_backend(), FakeProps()
    scorer = (lambda e: e['product_score'])
    vf2 = sweep_ancilla_layouts(6, backend, props, min_bridges=1, top_n=3,
                                circuit_score_fn=scorer)
    guided = guided_sweep(6, backend, props, min_bridges=1, top_n=3,
                          circuit_score_fn=scorer)

    # Keys that describe the VF2 ENUMERATION have no guided counterpart:
    # the beam is bounded by construction and has no mapping cap to make
    # diverse. Everything a downstream consumer reads is still shared.
    vf2_only = {'reservoir', 'reservoir_notices'}
    assert (set(vf2) - vf2_only) <= set(guided), (
        f'guided is missing sweep keys: {set(vf2) - vf2_only - set(guided)}')
    assert guided['engine'] == 'guided'
    assert set(guided['strata']) == set(vf2['strata'])
    assert set(guided['per_stratum']) == set(vf2['per_stratum'])
    for k in vf2['per_stratum']:
        assert set(guided['per_stratum'][k]) == set(vf2['per_stratum'][k])

    for k in vf2['strata']:
        if not vf2['strata'][k] or not guided['strata'][k]:
            continue
        vf2_fields = set(vf2['strata'][k][0])
        for entry in guided['strata'][k]:
            missing = vf2_fields - set(entry)
            assert not missing, f'k={k}: entry missing {missing}'


def test_min_bridges_is_validated_like_the_frozen_engine():
    backend, props = make_backend(), FakeProps()
    with pytest.raises(ValueError, match='min_bridges'):
        guided_sweep(4, backend, props, min_bridges=99)


def test_qubit_weight_hook_shifts_the_ranking():
    """The stability/T1 hook must actually influence what the beam keeps."""
    backend, props = make_backend(), FakeProps()
    from test_ancilla_sweep import GOOD_QUBITS
    good = set(GOOD_QUBITS)

    plain = guided_sweep(4, backend, props, min_bridges=1, top_n=1)
    # Punish the GOOD block hard; the winner should leave it.
    penalised = guided_sweep(
        4, backend, props, min_bridges=1, top_n=1,
        qubit_weight=lambda q: 0.01 if q in good else 1.0)

    plain_used = set(_entry_layout(plain['strata'][1][0]))
    pen_used = set(_entry_layout(penalised['strata'][1][0]))
    assert plain_used <= good
    assert not pen_used <= good, 'qubit_weight did not affect the search'


def test_excluded_qubits_never_appear():
    backend, props = make_backend(), FakeProps()
    excluded = {0, 1, 2, 3}
    res = guided_sweep(6, backend, props, min_bridges=1, top_n=3,
                       exclude_qubits=excluded)
    assert res['excluded_qubits'] == sorted(excluded)
    for _k, entry in _all_entries(res):
        assert not (set(_entry_layout(entry)) & excluded)


def test_log_floor_keeps_a_dead_element_comparable():
    """log(0) must not poison a partial state into -inf."""
    assert gs._safe_log(0.0) == gs._LOG_FLOOR
    assert gs._safe_log(-1.0) == gs._LOG_FLOOR
    assert gs._safe_log(1.0) == pytest.approx(0.0)
    assert gs._safe_log(0.5) == pytest.approx(math.log(0.5))
