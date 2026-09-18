"""Engine tests for iqmapper.benchmark — target selection, planning, costing,
result folding, persistence and plan gating.

Entirely offline: the runner is injected, so nothing imports qiskit_experiments
or touches a network. That is the same discipline qpu_scan.py's tests use, and
it is what makes a feature that spends QPU time safe to develop.
"""
from types import SimpleNamespace

import numpy as np
import pytest

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
    rb = [s for s in plan.specs if s.kind == 'rb2q']
    assert {s.qubits for s in rb} == {(0, 1), (1, 2), (2, 3)}, \
        'only TARGET edges may be benchmarked'
    assert all(s.params['lengths'] == bench.RB_2Q_LENGTHS for s in rb)
    assert all(s.params['num_samples'] == bench.RB_2Q_SAMPLES for s in rb)

    # No two experiments in one layer may share a qubit.
    by_layer = {}
    for spec in rb:
        by_layer.setdefault(spec.layer, []).append(set(spec.qubits))
    for layer, pairs in by_layer.items():
        seen = set()
        for pair in pairs:
            assert not (seen & pair), f'layer {layer} reuses a qubit'
            seen |= pair


def test_level3_covers_the_whole_chip_with_tutorial_parameters():
    plan = bench.build_plan(bench.LEVEL_FULL, bench.Targets([], []), 8,
                            _line_edges(8))
    kinds = {s.kind for s in plan.specs}
    assert kinds == {'t1', 't2hahn', 'readout', 'rb1q', 'rb2q'}

    t1 = next(s for s in plan.specs if s.kind == 't1')
    assert t1.params['delays'] == bench.T1T2_DELAYS
    assert t1.qubits == tuple(range(8)), 'L3 ignores targets, covers the chip'

    rb1 = next(s for s in plan.specs if s.kind == 'rb1q')
    assert rb1.params['lengths'] == bench.RB_1Q_LENGTHS
    assert rb1.params['num_samples'] == bench.RB_1Q_SAMPLES

    rb2 = {s.qubits for s in plan.specs if s.kind == 'rb2q'}
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
        elif spec.kind == 'rb2q':
            raw[idx] = [_Res(name='EPG', value=0.004)]

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
        elif spec.kind == 'rb2q':
            raw[idx] = [_Res(name='EPG', value=float('nan'))]   # bad fit

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


def test_level2_cost_scales_with_edge_count():
    coupling = _line_edges(127)
    few = bench.estimate_cost(bench.build_plan(
        2, bench.Targets(list(range(5)), coupling[:4]), 127, coupling))
    many = bench.estimate_cost(bench.build_plan(
        2, bench.Targets(list(range(40)), coupling[:39]), 127, coupling))
    assert many.qpu_seconds > few.qpu_seconds
    assert many.circuits > few.circuits
