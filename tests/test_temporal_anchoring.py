"""Temporal-anchoring audit fixes: the SELECTED calibration date
must govern all scoring in one search.

  * t1 — epoch consistency: when a past date is active, the sweep's pre-rank
    props are built from the SAME arrays the C1 circuit-score reads (the
    selected date's), not the live BackendProperties.
  * t2 — window anchor: volatility windows over [date-14d, date], not
    [now-14d, now]; sweep_with_stability forwards `today=`.
  * t3 — the scan header names the epoch mix when a past date is active.
"""
from types import SimpleNamespace

import numpy as np
import rustworkx

from iqmapper.stability_filter import (
    compute_qubit_volatility, sweep_with_stability,
)


# ── the volatility window anchors to the selected date (engine) ─────────

def _divergent_history(n_qubits=48):
    """30 daily snapshots. q3 spikes EARLY (07-05, inside [.. ,07-14]); q5
    spikes LATE (07-28, inside the now-anchored last-14 window). So the window
    the anchor picks decides which qubit reads volatile."""
    dates = [f"2026-07-{i + 1:02d}" for i in range(30)]
    hist = {}
    for d in dates:
        arr = [0.01] * n_qubits
        if d == '2026-07-05':
            arr[3] = 0.09
        if d == '2026-07-28':
            arr[5] = 0.09
        hist[d] = {'qubit_props': {'readout_error': arr}, 'gate_props': {}}
    return hist


def test_volatility_anchors_to_selected_date():
    hist = _divergent_history()
    early = compute_qubit_volatility(hist, 14, today='2026-07-14')
    live = compute_qubit_volatility(hist, 14, today=None)

    # Anchored to the past date: the EARLY spike (q3) is volatile, the late one
    # (q5) is outside the window and flat.
    assert early[3] > 0 and early[5] == 0
    # Now-anchored (default): the reverse.
    assert live[5] > 0 and live[3] == 0


def _two_block_backend():
    edges = []
    for base in (0, 24):
        top = [base + j for j in range(8)]
        bot = [base + 8 + j for j in range(8)]
        brg = [base + 16 + j for j in range(8)]
        for j in range(7):
            edges.append((top[j], top[j + 1]))
            edges.append((bot[j], bot[j + 1]))
        for j in range(8):
            edges.append((top[j], brg[j]))
            edges.append((brg[j], bot[j]))
    g = rustworkx.PyDiGraph()
    for i in range(48):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'], name='b')


class _Props:
    def readout_error(self, q):
        return 0.01

    def gate_error(self, name, edge):
        return 0.005


def test_sweep_with_stability_forwards_today():
    hist = _divergent_history()
    bk, pr = _two_block_backend(), _Props()
    early = sweep_with_stability(8, bk, pr, hist, window_days=14, threshold=1.0,
                                 min_bridges=1, today='2026-07-14')
    live = sweep_with_stability(8, bk, pr, hist, window_days=14, threshold=1.0,
                                min_bridges=1, today=None)
    # The anchor decides which qubit the filter prunes.
    assert 3 in early['stability']['excluded_final']
    assert 3 not in live['stability']['excluded_final']
    assert 5 in live['stability']['excluded_final']
    assert 5 not in early['stability']['excluded_final']


# ── UI harness for t1 / t3 ──────────────────────────────────────────────────

def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:                # pragma: no cover
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _canned():
    e = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
         'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.4,
         'additive_score': 0.5, 'lqs_circuit': 0.4, 'rank_in_stratum': 1,
         'source': 'derived'}
    return {'strata': {1: [e]}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 5,
            'n_candidates_scored': 5, 'excluded_qubits': [],
            'per_stratum': {1: {'n_direct': 0, 'n_derived': 1,
                                'truncated': False, 'shuffled': False}},
            'mapping_cap': 2000}


def _build(monkeypatch, *, past_date=None):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = SimpleNamespace(name='ibm_here')
    state.active_backend_name = "ibm_here"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
    state.available_backends = ["ibm_here", "ibm_other"]
    state.props_warning = None
    state.initial_props = SimpleNamespace(
        readout_error=lambda q: 0.99,          # LIVE sentinel: deliberately wrong
        gate_error=lambda n, e: 0.99)
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]
    # Selected-date arrays (what C1 reads) — distinct from initial_props.
    state.qubit_ids = [str(i) for i in range(8)]
    state.edges_list = [(i, i + 1) for i in range(7)]
    state.qubit_props_data = {'readout_error': np.array([0.001 * (i + 1)
                                                         for i in range(8)])}
    state.gate_props_data = {'cz_error': np.array([0.002 * (i + 1)
                                                   for i in range(7)])}
    b = MapperUIBuilder(state)
    b.build()
    # build() resets time_mode→'live'; apply the past-date selection AFTER it.
    if past_date:
        state.time_mode = 'point'
        state.point_timestamp = past_date
    return b, state


# ── pre-rank props == circuit-score arrays (one epoch) ──────────────────

def test_prerank_uses_selected_date_not_live_props(monkeypatch):
    b, state = _build(monkeypatch, past_date='2026-07-01 08:00:00')
    captured = {}
    monkeypatch.setattr(
        "iqmapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, bk, pr, **kw: captured.update(props=pr) or _canned())

    b.btn_lucj_run.click()
    pr = captured['props']
    # NOT the live BackendProperties (which returns the 0.99 sentinel)...
    assert pr is not state.initial_props
    assert abs(pr.readout_error(3) - 0.99) > 1e-6
    # ...it is the selected date's arrays — identical to what C1 reads.
    assert pr.readout_error(3) == state.qubit_props_data['readout_error'][3]
    assert pr.readout_error(7) == state.qubit_props_data['readout_error'][7]


# ── the scan header names the epoch mix on a past date ──────────────────

def test_scan_header_notes_epoch_mix_on_past_date(monkeypatch):
    b, state = _build(monkeypatch, past_date='2026-07-01 08:00:00')
    import iqmapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_other', 'family': 'Heron', 'eligible': True,
         'reason': None}])
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state, **k: (lambda backend: SimpleNamespace(
                            vintage='live · 09:30',
                            score_layout=lambda q: {'fidelity': 0.5})))
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned())

    b.btn_scan_qpus.click()
    html = "".join(c.value for c in b.scan_results_box.children
                   if isinstance(getattr(c, 'value', None), str))
    assert 'local backend scored on 2026-07-01' in html
    assert 'remote backends on live' in html


def test_scan_header_absent_in_live_mode(monkeypatch):
    b, state = _build(monkeypatch, past_date=None)     # live
    import iqmapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_other', 'family': 'Heron', 'eligible': True,
         'reason': None}])
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state, **k: (lambda backend: SimpleNamespace(
                            vintage='live · 09:30',
                            score_layout=lambda q: {'fidelity': 0.5})))
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned())

    b.btn_scan_qpus.click()
    html = "".join(c.value for c in b.scan_results_box.children
                   if isinstance(getattr(c, 'value', None), str))
    assert 'scored on' not in html
