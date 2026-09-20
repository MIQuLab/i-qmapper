"""Engine tests for iqmapper.benchmark — target selection, planning, costing,
result folding, persistence and plan gating.

Entirely offline: the runner is injected, so nothing imports qiskit_experiments
or touches a network. That is the same discipline qpu_scan.py's tests use, and
it is what makes a feature that spends QPU time safe to develop.
"""
import json
import os
from types import SimpleNamespace

import numpy as np
import pytest
import rustworkx

from iqmapper import benchmark as bench


# ── fixtures / helpers ──────────────────────────────────────────────

def _line_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def _state(n_qubits=8, edges=None, layouts=None):
    edges = _line_edges(n_qubits) if edges is None else edges
    return SimpleNamespace(
        qubit_ids=[str(i) for i in range(n_qubits)],
        edges_list=list(edges),
        layouts_data=layouts or {},
        active_backend_name='ibm_test',
        qubit_props_data={}, gate_props_data={},
        calibration_data_label='', delta_mode=True)


def _entry(alpha, beta, ancillas):
    return {'alpha': list(alpha), 'beta': list(beta),
            'ancillas': list(ancillas)}


def _reported(n_q=8, n_e=7):
    return ({'readout_error': np.full(n_q, 0.05),
             't1': np.full(n_q, 100.0),          # microseconds, as data.py stores
             't2': np.full(n_q, 80.0)},
            {'cz_error': np.full(n_e, 0.01)})


class _Res:
    """Stand-in for a qiskit-experiments AnalysisResult."""

    def __init__(self, name=None, value=None, qubits=None,
                 assignment_matrix=None):
        self.name = name
        self.qubits = qubits
        self.device_components = list(qubits or [])
        if assignment_matrix is not None:
            self.value = SimpleNamespace(assignment_matrix=assignment_matrix)
        else:
            self.value = value


# ── 1. select_targets ───────────────────────────────────────────────

def test_select_targets_unions_top_k_entries():
    state = _state(8)
    entries = [_entry([0, 1], [2], [3]),
               _entry([4, 5], [6], []),
               _entry([7], [], [])]          # outside top_k=2
    targets = bench.select_targets(state, entries, top_k=2)

    assert targets.qubits == [0, 1, 2, 3, 4, 5, 6]
    assert 7 not in targets.qubits, "top_k must bound the union"
    # Every coupling edge whose BOTH endpoints are targets, including (3, 4),
    # which bridges the two candidates: the union is what gets measured, so an
    # edge inside it counts even though no single candidate contains it.
    assert targets.edges == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    # ...and nothing touching the excluded qubit 7.
    assert all(7 not in e for e in targets.edges)


def test_select_targets_accepts_raw_sweep_dict():
    state = _state(8)
    raw = {'strata': {1: [_entry([0, 1], [2], [3])],
                      2: [_entry([4], [5], [])]}}
    targets = bench.select_targets(state, raw, top_k=5)
    assert set(targets.qubits) == {0, 1, 2, 3, 4, 5}


def test_select_targets_falls_back_to_current_layout():
    state = _state(8, layouts={0: {'alpha': ['1', '2'], 'beta': ['3'],
                                   'ancilla': [], 'other': []}})
    targets = bench.select_targets(state, None)
    assert targets.qubits == [1, 2, 3]
    assert targets.edges == [(1, 2), (2, 3)]


def test_select_targets_empty_when_nothing_to_measure():
    targets = bench.select_targets(_state(8), None)
    assert targets == bench.Targets([], [])


def test_select_targets_normalizes_edge_orientation():
    """An edge the device lists as (5, 4) must not be emitted as (4, 5)."""
    state = _state(6, edges=[(1, 0), (2, 1), (3, 2), (5, 4), (4, 3)])
    targets = bench.select_targets(state, [_entry([0, 1, 2], [3, 4], [5])])
    for edge in targets.edges:
        assert edge in state.edges_list, f'{edge} not in device orientation'


# ── 2. build_plan ───────────────────────────────────────────────────

def test_level1_is_readout_on_target_qubits_only():
    targets = bench.Targets([0, 1, 2], [(0, 1), (1, 2)])
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT, targets, 8,
                            _line_edges(8))
    assert [s.kind for s in plan.specs] == ['readout']
    assert plan.specs[0].qubits == (0, 1, 2)


def test_level2_adds_layered_2q_rb_over_target_edges_only():
    targets = bench.Targets([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets, 8, _line_edges(8))

    kinds = [s.kind for s in plan.specs]
    assert kinds.count('readout') == 1
    rb = [s for s in plan.specs if s.kind == 'rb2q_layer']
    assert rb, 'no layered 2Q RB specs'
    covered = {tuple(p) for spec in rb for p in spec.params['pairs']}
    assert covered == {(0, 1), (1, 2), (2, 3)}, \
        'only TARGET edges may be benchmarked'
    assert all(s.params['lengths'] == bench.RB_2Q_LENGTHS for s in rb)
    assert all(s.params['num_samples'] == bench.RB_2Q_SAMPLES for s in rb)

    # Each layer is ONE parallel experiment, so its pairs must be disjoint.
    for spec in rb:
        seen = set()
        for pair in spec.params['pairs']:
            assert not (seen & set(pair)), f'layer {spec.layer} reuses a qubit'
            seen |= set(pair)
        # 30 circuits per layer, whatever the number of pairs in it.
        assert bench._spec_circuits(spec) == 30


def test_level3_covers_the_whole_chip_with_tutorial_parameters():
    plan = bench.build_plan(bench.LEVEL_FULL, bench.Targets([], []), 8,
                            _line_edges(8))
    kinds = {s.kind for s in plan.specs}
    assert kinds == {'t1', 't2hahn', 'readout', 'rb1q', 'rb2q_layer'}

    t1 = next(s for s in plan.specs if s.kind == 't1')
    assert t1.params['delays'] == bench.T1T2_DELAYS
    assert t1.qubits == tuple(range(8)), 'L3 ignores targets, covers the chip'

    rb1 = next(s for s in plan.specs if s.kind == 'rb1q')
    assert rb1.params['lengths'] == bench.RB_1Q_LENGTHS
    assert rb1.params['num_samples'] == bench.RB_1Q_SAMPLES

    rb2 = {tuple(p) for s in plan.specs if s.kind == 'rb2q_layer'
           for p in s.params['pairs']}
    assert rb2 == set(_line_edges(8)), 'L3 benchmarks every coupling edge'
    assert plan.shots == bench.DEFAULT_SHOTS == 1000


def test_empty_targets_produce_an_empty_plan():
    plan = bench.build_plan(bench.LEVEL_TARGETED, bench.Targets([], []), 8,
                            _line_edges(8))
    assert plan.specs == []


# ── 3. estimate_cost ────────────────────────────────────────────────

def test_cost_is_monotonic_in_level_and_in_target_count():
    targets = bench.Targets([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    coupling = _line_edges(20)
    costs = [bench.estimate_cost(
        bench.build_plan(lvl, targets, 20, coupling)).qpu_seconds
        for lvl in (1, 2, 3)]
    assert costs[0] < costs[1] < costs[2], costs

    small = bench.Targets([0, 1], [(0, 1)])
    big = bench.Targets(list(range(10)), _line_edges(10))
    c_small = bench.estimate_cost(bench.build_plan(2, small, 20, coupling))
    c_big = bench.estimate_cost(bench.build_plan(2, big, 20, coupling))
    assert c_big.qpu_seconds > c_small.qpu_seconds
    assert c_big.circuits > c_small.circuits


def test_full_chip_127q_matches_the_tutorial_figure():
    """The cost constants are calibrated against the tutorial's ~240 s."""
    coupling = _line_edges(127)
    plan = bench.build_plan(bench.LEVEL_FULL, bench.Targets([], []), 127,
                            coupling)
    seconds = bench.estimate_cost(plan).qpu_seconds
    assert 240 * 0.7 <= seconds <= 240 * 1.3, (
        f'full-chip estimate {seconds:.0f}s is outside 240s +/-30%')


def test_empty_plan_costs_nothing():
    plan = bench.build_plan(1, bench.Targets([], []), 8, _line_edges(8))
    cost = bench.estimate_cost(plan)
    assert cost.circuits == 0 and cost.qpu_seconds == 0.0


# ── 4. results_to_epoch ─────────────────────────────────────────────

def test_measured_values_override_only_measured_elements():
    qp, gp = _reported()
    targets = bench.Targets([1, 2], [(1, 2)])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets, 8, _line_edges(8))

    raw = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 'readout':
            raw[idx] = [_Res(qubits=[1],
                             assignment_matrix=[[0.9, 0.1], [0.2, 0.8]]),
                        _Res(qubits=[2],
                             assignment_matrix=[[0.97, 0.03], [0.01, 0.99]])]
        elif spec.kind == 'rb2q_layer':
            raw[idx] = [_Res(name='EPG', value=0.004, qubits=[1, 2])]

    epoch = bench.results_to_epoch(raw, plan, qp, gp, _line_edges(8))

    # measured
    assert epoch.qubit_props['readout_error'][1] == pytest.approx(0.15)
    assert epoch.qubit_props['readout_error'][2] == pytest.approx(0.02)
    assert epoch.gate_props['cz_error'][bench_edge_index(1, 2)] == \
        pytest.approx(0.004)
    # untouched
    for q in (0, 3, 4, 5, 6, 7):
        assert epoch.qubit_props['readout_error'][q] == pytest.approx(0.05)
    assert epoch.gate_props['cz_error'][0] == pytest.approx(0.01)
    # masks
    assert list(np.where(epoch.measured_qubit_mask)[0]) == [1, 2]
    assert list(np.where(epoch.measured_edge_mask)[0]) == [bench_edge_index(1, 2)]
    # the reported arrays passed in must not be mutated in place
    assert qp['readout_error'][1] == pytest.approx(0.05)


def bench_edge_index(a, b):
    return _line_edges(8).index((a, b))


def test_fit_failure_keeps_reported_value_and_emits_a_notice():
    qp, gp = _reported()
    targets = bench.Targets([1, 2], [(1, 2)])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets, 8, _line_edges(8))

    raw = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 'readout':
            raw[idx] = [_Res(qubits=[1],
                             assignment_matrix=[[0.9, 0.1], [0.2, 0.8]])]
        elif spec.kind == 'rb2q_layer':
            raw[idx] = [_Res(name='EPG', value=float('nan'),
                             qubits=[1, 2])]                    # bad fit

    epoch = bench.results_to_epoch(raw, plan, qp, gp, _line_edges(8))

    assert epoch.gate_props['cz_error'][bench_edge_index(1, 2)] == \
        pytest.approx(0.01), 'a failed fit must keep the reported value'
    assert not epoch.measured_edge_mask.any()
    assert any('RB fit failed' in n for n in epoch.notices), epoch.notices


def test_missing_results_are_a_notice_not_a_crash():
    qp, gp = _reported()
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT,
                            bench.Targets([1], []), 8, _line_edges(8))
    epoch = bench.results_to_epoch({0: None}, plan, qp, gp, _line_edges(8))
    assert not epoch.measured_qubit_mask.any()
    assert epoch.notices


def test_empty_result_set_is_never_a_silent_no_op():
    """A runner that returns no analysis results must say so.

    The dangerous failure mode is not a crash, it is a benchmark that appears
    to succeed, changes nothing, and lets the user believe the numbers are
    fresh. Empty results therefore produce a notice and leave the mask clear.
    """
    qp, gp = _reported()
    targets = bench.Targets([1, 2], [(1, 2)])
    plan = bench.build_plan(bench.LEVEL_TARGETED, targets, 8, _line_edges(8))

    epoch = bench.results_to_epoch({i: [] for i in range(len(plan.specs))},
                                   plan, qp, gp, _line_edges(8))

    assert not epoch.measured_qubit_mask.any()
    assert not epoch.measured_edge_mask.any()
    assert len(epoch.notices) == len(plan.specs), epoch.notices
    assert all('no results' in n for n in epoch.notices), epoch.notices


def test_t1_t2_are_converted_from_seconds_to_microseconds():
    qp, gp = _reported()
    plan = bench.build_plan(bench.LEVEL_FULL, bench.Targets([], []), 8,
                            _line_edges(8))
    raw = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 't1':
            raw[idx] = [_Res(qubits=[0], value=150e-6)]     # 150 us in seconds
        elif spec.kind == 't2hahn':
            raw[idx] = [_Res(qubits=[0], value=90e-6)]

    epoch = bench.results_to_epoch(raw, plan, qp, gp, _line_edges(8))
    assert epoch.qubit_props['t1'][0] == pytest.approx(150.0)
    assert epoch.qubit_props['t2'][0] == pytest.approx(90.0)
    assert epoch.qubit_props['t1'][1] == pytest.approx(100.0)   # untouched


def test_readout_error_is_the_symmetrised_assignment_error():
    assert bench._readout_error_from_matrix([[0.9, 0.1], [0.2, 0.8]]) == \
        pytest.approx(0.15)


# ── 5. persistence + TTL ────────────────────────────────────────────

def _epoch(when, level=2):
    qp, gp = _reported()
    mask_q = np.zeros(8, dtype=bool); mask_q[1] = True
    mask_e = np.zeros(7, dtype=bool); mask_e[1] = True
    return bench.MeasuredEpoch(qp, gp, mask_q, mask_e, level, when,
                               ['q1: example notice'])


def test_save_list_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    import time as _time
    when = _time.time()
    epoch = _epoch(when)

    assert bench.save_measured_epoch('ibm_test', epoch) is not None
    assert bench.list_measured_epochs('ibm_test')

    back = bench.load_measured_epoch('ibm_test', when)
    assert back is not None
    assert back.level == epoch.level
    assert back.notices == epoch.notices
    np.testing.assert_allclose(back.qubit_props['readout_error'],
                               epoch.qubit_props['readout_error'])
    np.testing.assert_array_equal(back.measured_qubit_mask,
                                  epoch.measured_qubit_mask)


def test_load_missing_epoch_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    assert bench.load_measured_epoch('ibm_test', 1.0) is None


def test_ttl_reuse_window(tmp_path, monkeypatch):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    import time as _time
    now = _time.time()

    fresh = _epoch(now - 10 * 60)                      # 10 minutes old
    assert fresh.younger_than(60, now) is True
    assert fresh.age_minutes(now) == pytest.approx(10, abs=0.1)

    stale = _epoch(now - 120 * 60)                     # 2 hours old
    assert stale.younger_than(60, now) is False

    bench.save_measured_epoch('ibm_test', stale)
    assert bench.latest_reusable_epoch('ibm_test', 60, now) is None
    bench.save_measured_epoch('ibm_test', fresh)
    got = bench.latest_reusable_epoch('ibm_test', 60, now)
    assert got is not None and got.level == fresh.level


# ── 6. apply_measured_epoch ─────────────────────────────────────────

def test_apply_sets_arrays_label_and_resets_delta():
    state = _state(8)
    epoch = _epoch(1_700_000_000.0, level=2)
    assert bench.apply_measured_epoch(state, epoch) is True

    assert state.qubit_props_data is epoch.qubit_props
    assert state.gate_props_data is epoch.gate_props
    assert state.delta_mode is False
    assert 'measured' in state.calibration_data_label
    assert 'L2' in state.calibration_data_label
    assert getattr(state, '_measured_epoch') is epoch


def test_apply_rejects_a_shape_mismatch():
    state = _state(4)                       # 4 qubits, epoch has 8
    assert bench.apply_measured_epoch(state, _epoch(1.0)) is False
    assert state.delta_mode is True, 'a rejected epoch must change nothing'


def test_measured_badge_only_marks_measured_elements():
    state = _state(8)
    assert bench.measured_badge(state, qubit=1) == ''    # nothing applied yet
    bench.apply_measured_epoch(state, _epoch(1.0))
    assert bench.measured_badge(state, qubit=1) == ' (measured)'
    assert bench.measured_badge(state, qubit=0) == ''
    assert bench.measured_badge(state, edge=1) == ' (measured)'
    assert bench.measured_badge(state, edge=0) == ''


# ── 7. plan gating ──────────────────────────────────────────────────

@pytest.mark.parametrize('plan_name,level,allowed,warns', [
    ('premium', 1, True, False),
    ('premium', 2, True, False),
    ('premium', 3, True, False),
    ('dedicated', 3, True, False),
    ('open', 1, True, False),
    ('open', 2, True, True),
    ('open', 3, False, True),
    (None, 1, True, False),
    (None, 2, True, True),
    (None, 3, False, True),
    ('', 3, False, True),
])
def test_plan_gate_matrix(plan_name, level, allowed, warns):
    gate = bench.plan_gate(plan_name, level)
    assert gate.allowed is allowed
    assert (gate.warning is not None) is warns


def test_level3_always_needs_acknowledgement_even_on_premium():
    assert bench.requires_acknowledgement('premium', 3) is True
    assert bench.requires_acknowledgement('premium', 2) is False
    assert bench.requires_acknowledgement('open', 2) is True
    assert bench.requires_acknowledgement(None, 1) is False


# ── run_plan / install hint ─────────────────────────────────────────

def test_run_plan_delegates_to_the_injected_runner():
    plan = bench.build_plan(1, bench.Targets([0], []), 8, _line_edges(8))
    seen = {}
    bench.run_plan(plan, lambda p: seen.setdefault('plan', p) or {'ok': 1})
    assert seen['plan'] is plan


def test_run_plan_skips_the_runner_for_an_empty_plan():
    plan = bench.build_plan(1, bench.Targets([], []), 8, _line_edges(8))

    def _boom(_p):                                  # pragma: no cover
        raise AssertionError('runner must not be called for an empty plan')

    assert bench.run_plan(plan, _boom) == {}


def test_make_runtime_runner_raises_the_documented_install_hint():
    """With no qiskit-experiments installed the hint names the extra.

    Call-shape contracts live in tests/test_benchmark_runtime_contract.py.
    """
    with pytest.raises(ImportError, match=r'i-qmapper\[benchmark\]'):
        bench.make_runtime_runner(object())


# ── v2: target-selection modes ──────────────────────────────────────

def test_current_layouts_mode_unions_every_circuit_slot():
    """Hand-placed or auto-placed, C1..Cn all count — and no duplicate edges."""
    state = _state(8, layouts={
        0: {'alpha': ['0', '1'], 'beta': ['2'], 'ancilla': [], 'other': []},
        1: {'alpha': ['4', '5'], 'beta': ['6'], 'ancilla': ['3'], 'other': []},
    })
    targets = bench.select_targets(state, None, mode=bench.TARGET_CURRENT)

    assert targets.qubits == [0, 1, 2, 3, 4, 5, 6]
    assert len(targets.edges) == len(set(targets.edges)), 'duplicate edges'
    assert targets.edges == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    assert all(7 not in e for e in targets.edges)


def test_whole_qpu_mode_takes_every_qubit_and_edge():
    state = _state(8)
    targets = bench.select_targets(state, None, mode=bench.TARGET_WHOLE)
    assert targets.qubits == list(range(8))
    assert targets.edges == _line_edges(8)


def test_custom_mode_filters_edges_to_both_endpoints_in_the_set():
    state = _state(8)
    targets = bench.select_targets(state, None, mode=bench.TARGET_CUSTOM,
                                   custom='0, 1, 2, 5')
    assert targets.qubits == [0, 1, 2, 5]
    # (2,3) and (4,5) each have one endpoint outside the set.
    assert targets.edges == [(0, 1), (1, 2)]


def test_custom_mode_ignores_junk_and_out_of_range_qubits():
    state = _state(8)
    targets = bench.select_targets(state, None, mode=bench.TARGET_CUSTOM,
                                   custom='[0, 1] q3, 99, -4, , 2')
    assert targets.qubits == [0, 1, 2], 'junk and out-of-range must drop'


def test_sweep_mode_is_unchanged():
    state = _state(8)
    entries = [_entry([0, 1], [2], [3]), _entry([4, 5], [6], [])]
    by_mode = bench.select_targets(state, entries, top_k=2,
                                   mode=bench.TARGET_SWEEP)
    legacy = bench.select_targets(state, entries, top_k=2)
    assert by_mode == legacy


def test_sweep_mode_is_offered_only_once_a_sweep_has_run():
    state = _state(8, layouts={0: {'alpha': ['1'], 'beta': [], 'ancilla': [],
                                   'other': []}})
    assert bench.TARGET_SWEEP not in bench.available_target_modes(state, [])
    assert bench.TARGET_SWEEP in bench.available_target_modes(
        state, [_entry([0, 1], [2], [3])])
    # Order is the one the spec lists.
    assert bench.available_target_modes(state, [_entry([0], [1], [])]) == [
        bench.TARGET_CURRENT, bench.TARGET_SWEEP, bench.TARGET_WHOLE,
        bench.TARGET_CUSTOM]


def test_default_mode_prefers_a_placed_layout():
    placed = _state(8, layouts={0: {'alpha': ['1'], 'beta': [], 'ancilla': [],
                                    'other': []}})
    assert bench.default_target_mode(placed, None) == bench.TARGET_CURRENT
    empty = _state(8)
    assert bench.default_target_mode(empty, [_entry([0], [1], [])]) == \
        bench.TARGET_SWEEP


# ── v2: Level 1 does not scale with qubit count ─────────────────────

def test_level1_cost_is_flat_in_qubit_count():
    """LocalReadoutError is two circuits whatever the width.

    This is what makes `Whole QPU x L1` the cheap "refresh the readout
    heatmap" action rather than a full-chip expense.
    """
    coupling = _line_edges(127)
    ten = bench.estimate_cost(bench.build_plan(
        1, bench.Targets(list(range(10)), []), 127, coupling))
    whole = bench.estimate_cost(bench.build_plan(
        1, bench.Targets(list(range(127)), []), 127, coupling))

    assert whole.qpu_seconds == pytest.approx(ten.qpu_seconds)
    assert whole.circuits == ten.circuits == 2


def test_level2_cost_scales_with_layer_count_not_edge_count():
    """Layering is the whole saving: 44 disjointly-colourable edges cost the
    same as 4 if they fit the same number of layers."""
    coupling = _line_edges(127)
    # A path graph 2-colours, so any number of path edges needs 2 layers.
    few = bench.estimate_cost(bench.build_plan(
        2, bench.Targets(list(range(5)), coupling[:4]), 127, coupling))
    many = bench.estimate_cost(bench.build_plan(
        2, bench.Targets(list(range(40)), coupling[:39]), 127, coupling))
    assert many.circuits == few.circuits, (
        'more edges in the same layers must not add circuits')
    # ...and a plan with MORE layers does cost more.
    one_edge = bench.estimate_cost(bench.build_plan(
        2, bench.Targets([0, 1], coupling[:1]), 127, coupling))
    assert few.circuits >= one_edge.circuits


# ── v3 Part A: the plan is composite, and it fits in a job ──────────

def test_level1_is_two_circuits_at_any_width():
    """LocalReadoutError prepares all targets at once: 2 circuits, always."""
    coupling = _line_edges(156)
    for n in (3, 10, 156):
        plan = bench.build_plan(1, bench.Targets(list(range(n)), []), 156,
                                coupling)
        assert bench.plan_circuit_count(plan) == 2, f'{n} qubits'
        assert len([s for s in plan.specs if s.kind == 'readout']) == 1


def test_level2_is_bounded_by_layers_not_edges():
    """44 edges must cost at most 2 + n_layers x 30 circuits."""
    coupling = _line_edges(64)
    targets = bench.Targets(list(range(45)), coupling[:44])
    plan = bench.build_plan(2, targets, 64, coupling)

    layers = [s for s in plan.specs if s.kind == 'rb2q_layer']
    assert 0 < len(layers) <= 3, f'{len(layers)} layers for a path graph'
    assert bench.plan_circuit_count(plan) <= 2 + 3 * 30
    # Every target edge is covered exactly once, and layers stay disjoint.
    covered = [tuple(p) for s in layers for p in s.params['pairs']]
    assert sorted(covered) == sorted(coupling[:44])
    assert len(covered) == len(set(covered))


def test_full_chip_level3_is_about_134_circuits():
    """The number that decides whether the job is accepted at all.

    The pre-v3 plan emitted one spec per qubit and per edge: ~43,400 circuits
    on this device, which IBM rejected outright (error 1520).
    """
    graph = rustworkx.generators.heavy_hex_graph(9)
    coupling = [tuple(e) for e in graph.edge_list()]
    plan = bench.build_plan(3, bench.Targets([], []), 156, coupling)

    circuits = bench.plan_circuit_count(plan)
    assert abs(circuits - 134) <= 10, f'{circuits} circuits, expected ~134'
    assert bench.plan_shots(plan) == circuits * 1000
    assert len(bench.split_for_submission(plan)) == 1, 'L3 must fit one job'
    ok, reason = bench.plan_is_submittable(plan)
    assert ok, reason


def test_split_never_drops_an_experiment():
    coupling = _line_edges(64)
    plan = bench.build_plan(2, bench.Targets(list(range(45)), coupling[:44]),
                            64, coupling)
    chunks = bench.split_for_submission(plan)
    assert sum(len(c.specs) for c in chunks) >= len(plan.specs)
    kinds_in = sorted(s.kind for s in plan.specs)
    kinds_out = sorted(s.kind for c in chunks for s in c.specs)
    assert kinds_out[:len(kinds_in)] == kinds_in or set(kinds_out) == set(kinds_in)
    for chunk in chunks:
        assert chunk.shots <= bench._JOB_SHOT_BUDGET


def test_an_over_budget_plan_is_split_into_several_jobs():
    """A plan bigger than one job's shot budget must chunk, not fail."""
    coupling = _line_edges(64)
    plan = bench.build_plan(2, bench.Targets(list(range(45)), coupling[:44]),
                            64, coupling)
    huge = 200_000                              # shots per circuit
    chunks = bench.split_for_submission(plan, huge)
    assert len(chunks) > 1, 'an over-budget plan must split'
    for chunk in chunks:
        assert chunk.shots <= bench._JOB_SHOT_BUDGET


def test_planned_count_reflects_the_plan_not_the_targets():
    """L1 targets edges but measures none, so it must still read complete."""
    coupling = _line_edges(8)
    targets = bench.Targets([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    l1 = bench.build_plan(1, targets, 8, coupling)
    assert bench.plan_planned_count(l1) == (4, 0)
    l2 = bench.build_plan(2, targets, 8, coupling)
    assert bench.plan_planned_count(l2) == (4, 3)


def test_summary_line_states_circuits_shots_and_jobs():
    graph = rustworkx.generators.heavy_hex_graph(9)
    coupling = [tuple(e) for e in graph.edge_list()]
    cost = bench.estimate_cost(bench.build_plan(3, bench.Targets([], []), 156,
                                                coupling))
    assert 'circuits' in cost.text and 'shots' in cost.text
    assert '1 job' in cost.text
    assert 'estimate' in cost.text


# ── v3 Part C: an empty epoch is a failure ─────────────────────────

def test_epoch_reports_emptiness_partiality_and_completeness():
    qp, gp = _reported()
    coupling = _line_edges(8)
    plan = bench.build_plan(1, bench.Targets([1, 2], []), 8, coupling)

    empty = bench.results_to_epoch({0: []}, plan, qp, gp, coupling)
    assert empty.is_empty is True
    assert empty.is_complete is False
    assert empty.measured_count == (0, 0)

    full = bench.results_to_epoch(
        {0: [_Res(qubits=[1], assignment_matrix=[[0.99, 0.01], [0.01, 0.99]]),
             _Res(qubits=[2], assignment_matrix=[[0.98, 0.02], [0.02, 0.98]])]},
        plan, qp, gp, coupling)
    assert full.is_empty is False
    assert full.measured_count == (2, 0) == full.planned_count
    assert full.is_complete is True
    assert '2/2 qubits' in full.coverage_text()

    partial = bench.results_to_epoch(
        {0: [_Res(qubits=[1], assignment_matrix=[[0.99, 0.01], [0.01, 0.99]])]},
        plan, qp, gp, coupling)
    assert partial.is_empty is False and partial.is_complete is False
    assert partial.measured_count == (1, 0)


# ── v3 Part B: a failed job is an error, not an empty parse ────────

def test_check_chunk_status_raises_on_an_errored_job():
    class _Errored:
        job_ids = ['abc123']

        def status(self):
            return SimpleNamespace(name='ERROR')

        def errors(self):
            return ('The number of circuit shots 43433984 exceeds the '
                    'system limit 10000000')

    with pytest.raises(bench.BenchmarkRunError) as excinfo:
        bench._check_chunk_status(_Errored(), 0)
    assert 'exceeds the system limit' in str(excinfo.value)
    assert excinfo.value.job_ids == ['abc123']


def test_check_chunk_status_passes_a_healthy_job():
    class _Done:
        job_ids = ['ok1']

        def status(self):
            return SimpleNamespace(name='DONE')

        def errors(self):
            return None

    assert bench._check_chunk_status(_Done(), 0) == ['ok1']


# ── v3 Part E: the analysis-result shape adapter ───────────────────

def test_result_adapter_handles_the_list_shape():
    items = [_Res(name='T1', value=1.0, qubits=[3])]

    class _Data:
        def analysis_results(self, *a, **k):
            return items

        def child_data(self):
            return []

    out = list(bench._iter_analysis_results(_Data()))
    assert len(out) == 1 and out[0].name == 'T1'


def test_result_adapter_handles_the_dataframe_shape():
    """Newer qiskit-experiments returns a DataFrame from analysis_results()."""
    pd = pytest.importorskip('pandas')
    frame = pd.DataFrame([{'name': 'T1', 'value': 1.5,
                           'device_components': ['Q3'], 'quality': 'good'}])

    class _Data:
        def analysis_results(self, *a, **k):
            return frame

        def child_data(self):
            return []

    out = list(bench._iter_analysis_results(_Data()))
    assert len(out) == 1
    assert out[0].name == 'T1'
    assert bench._components_to_qubits(out[0]) == [3]


def test_result_adapter_recurses_into_children():
    child_items = [_Res(name='EPG', value=0.01, qubits=[0, 1])]

    class _Child:
        def analysis_results(self, *a, **k):
            return child_items

        def child_data(self):
            return []

    class _Parent:
        def analysis_results(self, *a, **k):
            return []

        def child_data(self):
            return [_Child()]

    out = list(bench._iter_analysis_results(_Parent()))
    assert len(out) == 1 and out[0].name == 'EPG'


def test_components_to_qubits_parses_both_forms():
    assert bench._components_to_qubits(
        SimpleNamespace(device_components=['Q7'])) == [7]
    assert bench._components_to_qubits(
        SimpleNamespace(device_components=[SimpleNamespace(index=4)])) == [4]
    assert bench._components_to_qubits(
        SimpleNamespace(device_components=[]), fallback=(9,)) == [9]


# ── v3 Part D: persistence and diagnosis ───────────────────────────

def test_job_record_is_written_at_submission(tmp_path, monkeypatch):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    coupling = _line_edges(8)
    plan = bench.build_plan(2, bench.Targets([0, 1, 2, 3], coupling[:3]), 8,
                            coupling)
    when = 1_700_000_000.0

    path = bench.save_job_record('ibm_test', when, plan,
                                 jobs={'0': ['job-1']},
                                 statuses={'0': 'submitted'})
    assert path and os.path.exists(path)
    with open(path, encoding='utf-8') as fh:
        record = json.load(fh)
    for key in ('level', 'shots', 'circuits', 'total_shots', 'targets',
                'specs', 'chunks', 'jobs', 'statuses'):
        assert key in record, key
    assert record['jobs'] == {'0': ['job-1']}
    assert record['circuits'] == bench.plan_circuit_count(plan)

    back = bench.latest_job_record('ibm_test')
    assert back and back['jobs'] == {'0': ['job-1']}


def test_raw_dump_captures_nesting_status_and_errors():
    class _Node:
        def __init__(self, name, results, children=(), status='DONE',
                     errors=None, jobs=()):
            self.experiment_type = name
            self._results = results
            self._children = list(children)
            self._status = status
            self._errors = errors
            self.job_ids = list(jobs)

        def status(self):
            return SimpleNamespace(name=self._status)

        def errors(self):
            return self._errors

        def analysis_results(self, *a, **k):
            return self._results

        def child_data(self):
            return self._children

    leaf = _Node('StandardRB', [_Res(name='EPG_cz', value=0.004,
                                     qubits=[0, 1])])
    root = _Node('BatchExperiment', [], [leaf], status='ERROR',
                 errors='shot limit exceeded', jobs=['j1'])

    dump = bench.dump_raw_results([root])
    assert dump['experiments'][0]['status'] == 'ERROR'
    assert dump['experiments'][0]['job_ids'] == ['j1']
    assert dump['experiments'][0]['children'][0]['results'][0]['name'] == \
        'EPG_cz'

    text = bench.diagnose_raw(dump)
    assert 'ERROR' in text
    assert 'EPG_cz' in text
    assert 'shot limit exceeded' in text
    assert 'depth 1' in text, 'the summary must say WHERE results were found'


def test_diagnose_raw_says_so_when_nothing_was_produced():
    class _Empty:
        job_ids = []
        experiment_type = 'BatchExperiment'

        def status(self):
            return SimpleNamespace(name='ERROR')

        def errors(self):
            return 'error 1520'

        def analysis_results(self, *a, **k):
            return []

        def child_data(self):
            return []

    text = bench.diagnose_raw(bench.dump_raw_results([_Empty()]))
    assert 'NO analysis results at any depth' in text
    assert 'error 1520' in text


def test_diagnose_raw_on_a_missing_file():
    assert 'Could not read' in bench.diagnose_raw('/nonexistent/path.json')


# ── v3.1: the 20 Sept live-run symptoms, reproduced offline ────────
#
# Each of these fails against the pre-v3.1 parser with the exact error the
# live run produced. The shapes are taken from the real raw dump
# (measured_20260920T123808_raw.json), not from what we assumed.

class _Mitigator:
    """LocalReadoutMitigator: assignment_matrix is a METHOD, not an attribute.

    Reading it as an attribute is what produced
    "float() argument must be a string or a real number, not 'method'"
    for all 156 qubits.
    """

    def __init__(self, matrices):
        self._m = dict(matrices)

    def assignment_matrix(self, qubits=None):
        if qubits is None:
            raise TypeError('qubits required')
        q = list(qubits)[0]
        return np.asarray(self._m[int(q)], dtype=float)


def _readout_result(qubits, matrices):
    """ONE result covering every qubit, as the live dump shows."""
    return SimpleNamespace(
        name=bench.RESULT_READOUT_MITIGATOR,
        device_components=[f'Q{q}' for q in qubits],
        value=_Mitigator(matrices))


def test_readout_assignment_matrix_is_called_not_read():
    """Live symptom 2: readout failed for all 156 qubits."""
    qp, gp = _reported()
    coupling = _line_edges(8)
    plan = bench.build_plan(1, bench.Targets([0, 1, 2], []), 8, coupling)

    good = [[0.99, 0.01], [0.01, 0.99]]
    skewed = [[0.9, 0.1], [0.2, 0.8]]
    raw = {0: [_readout_result([0, 1, 2],
                               {0: good, 1: skewed, 2: good})]}

    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    assert epoch.n_measured_qubits == 3, epoch.notices
    assert epoch.qubit_props['readout_error'][1] == pytest.approx(0.15)
    assert epoch.qubit_props['readout_error'][0] == pytest.approx(0.01)
    values = epoch.qubit_props['readout_error'][[0, 1, 2]]
    assert np.all((values >= 0.0) & (values <= 0.5))


def test_t1_t2_are_attributed_by_device_components():
    """Live symptom 3: 156 T1 results existed, but only 1 qubit was measured.

    The parser read `.qubits`, which these results do not carry, and fell back
    to `spec.qubits[0]` — so every one of the 156 results was written onto
    qubit 0.
    """
    qp, gp = _reported()
    coupling = _line_edges(8)
    plan = bench.build_plan(3, bench.Targets([], []), 8, coupling)

    raw = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 't1':
            # Shape from the live dump: device_components=['Q0'], seconds.
            raw[idx] = [SimpleNamespace(name='T1', device_components=[f'Q{q}'],
                                        value=1.0e-4 + q * 1e-5)
                        for q in range(8)]
        elif spec.kind == 't2hahn':
            raw[idx] = [SimpleNamespace(name='T2', device_components=[f'Q{q}'],
                                        value=8.0e-5)
                        for q in range(8)]
        else:
            raw[idx] = []

    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    assert epoch.n_measured_qubits == 8, (
        f'expected every qubit, got {epoch.n_measured_qubits}: {epoch.notices}')
    # Seconds -> microseconds, and each result on its OWN qubit.
    assert epoch.qubit_props['t1'][0] == pytest.approx(100.0)
    assert epoch.qubit_props['t1'][3] == pytest.approx(130.0)
    assert epoch.qubit_props['t2'][5] == pytest.approx(80.0)


def test_one_qubit_rb_counts_qubits_not_gate_results():
    """Live symptom 4: '1Q RB: median EPG over 624 qubit(s)' — 156 x 4 gates."""
    qp, gp = _reported()
    coupling = _line_edges(8)
    plan = bench.build_plan(3, bench.Targets([], []), 8, coupling)

    raw = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 'rb1q':
            raw[idx] = [
                SimpleNamespace(name=f'EPG_{g}', device_components=[f'Q{q}'],
                                value=7e-4)
                for q in range(8) for g in ('rz', 'sx', 'x')]
        else:
            raw[idx] = []

    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    note = next((n for n in epoch.notices if '1Q RB' in n), '')
    assert note, epoch.notices
    assert '8 qubit' in note, f'counted gate results, not qubits: {note!r}'
    assert '24' not in note
    # The per-qubit EPGs are kept for the 2Q analysis (Part B).
    assert epoch.extra.get('epg_1q'), 'per-qubit 1Q EPGs must be retained'
    assert set(epoch.extra['epg_1q']) == set(range(8))


def test_notices_are_summarised_not_a_list_of_indices():
    """Live symptom 5: a notice listing all 156 qubit indices verbatim."""
    qp, gp = _reported(n_q=40, n_e=39)
    coupling = _line_edges(40)
    plan = bench.build_plan(1, bench.Targets(list(range(40)), []), 40,
                            coupling)

    epoch = bench.results_to_epoch({0: []}, plan, qp, gp, coupling)

    joined = ' '.join(epoch.notices)
    assert '[0, 1, 2, 3, 4, 5' not in joined, (
        f'notice still lists indices: {joined[:120]}')
    assert any('40' in n for n in epoch.notices), epoch.notices
    assert len(joined) < 400, f'notice too verbose ({len(joined)} chars)'


# ── v3.1 Part B: 1Q EPG subtraction wiring ─────────────────────────

def test_level2_now_includes_one_qubit_rb():
    """The 2Q analysis cannot subtract what was never measured."""
    coupling = _line_edges(8)
    plan = bench.build_plan(2, bench.Targets([0, 1, 2, 3], coupling[:3]), 8,
                            coupling)
    kinds = [s.kind for s in plan.specs]
    assert 'rb1q' in kinds, kinds
    # ...and it comes before the two-qubit layers.
    assert kinds.index('rb1q') < kinds.index('rb2q_layer')
    assert bench.plan_circuit_count(plan) == 2 + 30 + 2 * 30


def test_epg_options_are_set_from_the_one_qubit_results():
    captured = {}

    class _Analysis:
        def set_options(self, **kw):
            captured.update(kw)

    experiment = SimpleNamespace(analysis=_Analysis())
    notices = []
    ok = bench.apply_epg_1q_options(experiment, [(0, 1)],
                                    {0: 7e-4, 1: 8e-4}, {}, notices)
    assert ok is True
    assert 'epg_1_qubit' in captured
    assert not notices


def test_epg_options_fall_back_to_the_reported_error():
    captured = {}

    class _Analysis:
        def set_options(self, **kw):
            captured.update(kw)

    notices = []
    ok = bench.apply_epg_1q_options(SimpleNamespace(analysis=_Analysis()),
                                    [(0, 1)], {0: 7e-4}, {1: 9e-4}, notices)
    assert ok is True
    assert any('1Q EPG unavailable' in n for n in notices), notices


def test_an_unsupported_epg_option_is_a_notice_not_a_crash():
    class _Analysis:
        def set_options(self, **kw):
            raise TypeError('unexpected keyword epg_1_qubit')

    notices = []
    ok = bench.apply_epg_1q_options(SimpleNamespace(analysis=_Analysis()),
                                    [(0, 1)], {0: 7e-4, 1: 8e-4}, {}, notices)
    assert ok is False
    assert any('did not accept epg_1_qubit' in n for n in notices), notices


# ── v3.1 Part C: consistency and relative mode ─────────────────────

def _epoch_with(measured_gate, reported_gate, mask):
    qp = {'readout_error': np.full(4, 0.01)}
    gp = {'cz_error': np.asarray(measured_gate, dtype=float)}
    return bench.MeasuredEpoch(qp, gp, np.zeros(4, dtype=bool), mask, 2,
                               1_700_000_000.0)


def test_consistency_stats_reports_both_ratio_statistics():
    """They differ on skewed data — quoting one as 'the' ratio misleads."""
    reported = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    measured = np.array([2e-3, 2e-3, 2e-3, 20e-3])
    mask = np.ones(4, dtype=bool)
    epoch = _epoch_with(measured, reported, mask)

    stats = bench.consistency_stats(epoch, {}, {'cz_error': reported})
    cz = stats['cz_error']
    assert cz['n'] == 4
    assert cz['median_ratio'] == pytest.approx(2.0)
    assert cz['ratio_of_medians'] == pytest.approx(2.0)
    assert cz['n_worse_2x'] == 1          # only the 20x outlier is >2x
    assert cz['top_degradations'][0]['index'] == 3


def test_an_untouched_property_is_not_reported():
    reported = np.array([1e-3, 2e-3, 3e-3, 4e-3])
    epoch = _epoch_with(reported.copy(), reported, np.ones(4, dtype=bool))
    assert 'cz_error' not in bench.consistency_stats(
        epoch, {}, {'cz_error': reported})


def test_relative_mode_matches_medians_and_preserves_ratios():
    """The uniform offset goes; the outlier stays an outlier."""
    reported = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    measured = np.array([3e-3, 3e-3, 3e-3, 30e-3])
    mask = np.ones(4, dtype=bool)
    epoch = _epoch_with(measured.copy(), reported, mask)

    stats = bench.consistency_stats(epoch, {}, {'cz_error': reported})
    assert list(bench.consistency_offenders(stats)) == ['cz_error']

    before = epoch.gate_props['cz_error'].copy()
    n = bench.apply_relative_mode(epoch, {}, {'cz_error': reported}, stats)
    after = epoch.gate_props['cz_error']

    assert n == 1 and epoch.relative is True
    assert np.median(after[mask]) == pytest.approx(np.median(reported[mask]))
    # Every pairwise ratio is untouched — that is what makes outliers survive.
    np.testing.assert_allclose(after / after[0], before / before[0])
    assert after[3] / after[0] == pytest.approx(10.0)


def test_relative_mode_is_a_no_op_inside_the_band():
    reported = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    measured = np.array([1.2e-3, 1.1e-3, 1.3e-3, 1.2e-3])
    epoch = _epoch_with(measured.copy(), reported, np.ones(4, dtype=bool))
    stats = bench.consistency_stats(epoch, {}, {'cz_error': reported})
    assert bench.consistency_offenders(stats) == {}
    assert bench.apply_relative_mode(epoch, {}, {'cz_error': reported},
                                     stats) == 0
    assert epoch.relative is False


def test_relative_label_says_so():
    state = _state(4)
    epoch = _epoch_with(np.full(3, 3e-3), np.full(3, 1e-3),
                        np.ones(3, dtype=bool))
    epoch.qubit_props = {'readout_error': np.full(4, 0.01)}
    epoch.relative = True
    assert bench.apply_measured_epoch(state, epoch) is True
    assert 'relative' in state.calibration_data_label
