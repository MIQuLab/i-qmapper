"""Calibration-epoch audit (main): the auto-LUCJ search must score on the
LOADED snapshot's calibration, not live backend properties.

Headless-stub style: real MapperUIBuilder over a deterministic 5-qubit line
device; auto_lucj.get_all_zigzag_layouts is monkeypatched to CAPTURE the props
object the handler passes (and return a canned layout). state.initial_props is
a deliberately-wrong live sentinel (readout_error == 0.99) so a mismatch is
unmistakable.
"""
import numpy as np
import pytest


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 5
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


class _LiveSentinel:
    """Live BackendProperties stand-in — every value deliberately wrong."""
    def readout_error(self, q): return 0.99
    def gate_error(self, name, edge): return 0.99
    def t1(self, q): return 1e-9
    def t2(self, q): return 1e-9


# Snapshot arrays (what C1 / the error map read): distinct from the sentinel.
_SNAP_RO = np.array([0.001 * (i + 1) for i in range(5)])
_SNAP_CZ = np.array([0.002 * (i + 1) for i in range(4)])
_SNAP_T1 = np.array([100.0, 110.0, 120.0, 130.0, 140.0])   # µs
_SNAP_T2 = np.array([80.0, 82.0, 84.0, 86.0, 88.0])        # µs


def _build(monkeypatch, *, snapshot, ancilla=0):
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    captured = {}
    monkeypatch.setattr(
        "iqmapper.auto_lucj.get_all_zigzag_layouts",
        lambda **kw: (captured.update(props=kw['props']) or
                      ([{'rank': 1, 'score': 0.1, 'alpha': [0, 1],
                         'beta': [2, 3]}], 1)))
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()                 # no .coords → Auto allowed
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = _LiveSentinel()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]
    b = MapperUIBuilder(state)
    b.build()
    # Load a snapshot AFTER build() (build resets time_mode→'live').
    state.qubit_ids = [str(i) for i in range(5)]
    state.edges_list = [(i, i + 1) for i in range(4)]
    state.qubit_props_data = {'readout_error': _SNAP_RO,
                              't1': _SNAP_T1, 't2': _SNAP_T2}
    state.gate_props_data = {'cz_error': _SNAP_CZ}
    if snapshot:
        state.time_mode = 'point'
        state.point_timestamp = '2026-07-01 08:00:00'
    # Run now routes by ancilla count: a fixed count (>= 0) → the legacy
    # get_all_zigzag_layouts search these tests mock; Auto (-1) → the sweep.
    b.lucj_ancilla_dd.value = ancilla
    return b, state, captured


# ── the search receives the snapshot arrays, not the live sentinel ──────────

def test_search_uses_snapshot_not_live(monkeypatch):
    b, state, captured = _build(monkeypatch, snapshot=True)
    b.btn_lucj_run.click()
    pr = captured['props']
    assert pr is not state.initial_props
    # Live sentinel would be 0.99; the snapshot value is 0.004.
    assert abs(pr.readout_error(3) - 0.99) > 1e-6
    assert pr.readout_error(3) == _SNAP_RO[3]
    # gate error comes from the snapshot's cz_error (both edge orders resolve).
    assert pr.gate_error('cz', (2, 3)) == _SNAP_CZ[2]
    assert pr.gate_error('cz', (3, 2)) == _SNAP_CZ[2]


# ── search and C1 read the SAME values (epoch consistency) ──────────────────

def test_search_and_c1_share_epoch(monkeypatch):
    b, state, captured = _build(monkeypatch, snapshot=True)
    b.btn_lucj_run.click()
    pr = captured['props']
    # C1 (export.scoring) reads state.qubit_props_data / gate_props_data. The
    # search props must return those exact arrays' values.
    for q in range(5):
        assert pr.readout_error(q) == state.qubit_props_data['readout_error'][q]
    for i, (u, v) in enumerate(state.edges_list):
        assert pr.gate_error('cz', (u, v)) == state.gate_props_data['cz_error'][i]
    # t1/t2 unit round-trip: search gets seconds, ×1e6 recovers the stored µs.
    assert pr.t1(2) * 1e6 == pytest.approx(_SNAP_T1[2])
    assert pr.t2(2) * 1e6 == pytest.approx(_SNAP_T2[2])


# ── live mode unchanged — the sentinel flows ────────────────────────────────

def test_live_mode_passes_initial_props(monkeypatch):
    b, state, captured = _build(monkeypatch, snapshot=False)   # time_mode='live'
    b.btn_lucj_run.click()
    pr = captured['props']
    assert pr is state.initial_props
    assert pr.readout_error(3) == 0.99                          # the sentinel


# ── the "scored on" note renders only for a non-live snapshot ───────────────

def test_status_note_only_for_snapshot(monkeypatch):
    b, state, captured = _build(monkeypatch, snapshot=True)
    b.btn_lucj_run.click()
    assert 'scored on 2026-07-01' in b.lbl_lucj_status.value

    b2, state2, _ = _build(monkeypatch, snapshot=False)
    b2.btn_lucj_run.click()
    assert 'scored on' not in b2.lbl_lucj_status.value


# ── the Auto sweep path also pre-ranks on the snapshot, not live ─────────────

def test_sweep_uses_snapshot_not_live(monkeypatch):
    """Auto (-1) routes to the stratified sweep. It must pre-rank on the loaded
    snapshot's props — the SAME epoch as the circuit score — never the live
    sentinel. Capture the props object sweep_ancilla_layouts receives."""
    swept = {}

    def _fake_sweep(norb, backend, props, **kw):
        swept['props'] = props
        # Minimal well-formed result: no strata → the handler renders the
        # "no candidates" note without needing a real backend/coupling map.
        return {'strata': {}, 'vf2_truncated': False, 'per_stratum': {},
                'mapping_cap': None, 'k_max_effective': None}

    monkeypatch.setattr(
        "iqmapper.ancilla_sweep.sweep_ancilla_layouts", _fake_sweep)
    b, state, _ = _build(monkeypatch, snapshot=True, ancilla=-1)
    b.btn_lucj_run.click()
    pr = swept['props']
    assert pr is not state.initial_props
    # Live sentinel would be 0.99; the snapshot readout is 0.004.
    assert pr.readout_error(3) == _SNAP_RO[3]
    assert pr.gate_error('cz', (2, 3)) == _SNAP_CZ[2]
