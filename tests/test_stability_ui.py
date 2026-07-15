"""Headless tests for the stability-filter UI (Component 4b).

Offline: the data layer is stubbed and the stability/sweep engines are
monkeypatched with canned dicts. The background spawn + progress marshal
are injected synchronously.
"""
from types import SimpleNamespace

import numpy as np
import pytest


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:            # pragma: no cover
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _entry(prod=0.4, k=1):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': list(range(4, 4 + k)),
            'bridge_slots': tuple(range(0, 4 * k, 4)),
            'product_score': prod, 'additive_score': 0.5,
            'rank_in_stratum': 1, 'source': 'derived'}


def _canned_res(stability):
    return {'strata': {1: [_entry()]}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 5,
            'n_candidates_scored': 5, 'excluded_qubits': [1, 2, 3],
            'stability': stability}


def _detail_by_name(b, name):
    """Select the scan-selector row whose label contains `name` and return its
    rendered detail HTML (the comparison table is gone; per-backend stability /
    cap / calib now live on the detail's second line)."""
    from ipywidgets import Dropdown, HTML
    dd = next(w for w in b.scan_results_box.children if isinstance(w, Dropdown))
    dd.value = next(i for lbl, i in dd.options if name in lbl)
    ch = list(b.scan_results_box.children)
    di = ch.index(dd)
    return next(w.value for w in ch[di + 1:] if isinstance(w, HTML))


def _build(monkeypatch, *, history=False, spawn=_sync_spawn,
           marshal=lambda cb: cb()):
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', spawn)
    monkeypatch.setattr(au, 'marshal_to_main', marshal)
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)

    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder

    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
    state.available_backends = ["ibm_a", "ibm_b"]
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    if history:
        state.cal_history_dates = ['2026-07-01', '2026-07-02']
        state.cal_history = {
            d: {'qubit_props': {'readout_error': [0.01] * 8}, 'gate_props': {}}
            for d in state.cal_history_dates}
    else:
        state.cal_history_dates = ["_seed_"]   # skip bg auto-load; no dict
        state.cal_history = {}

    b = MapperUIBuilder(state)
    b.build()
    return b


# ── availability gating ─────────────────────────────────────────────────

def test_checkbox_disabled_without_history(monkeypatch):
    b = _build(monkeypatch, history=False)
    assert b.chk_stability_filter.disabled is True
    assert 'Fetch calibration history' in b.chk_stability_filter.tooltip


def test_checkbox_enabled_with_history(monkeypatch):
    b = _build(monkeypatch, history=True)
    assert b.chk_stability_filter.disabled is False
    assert b.chk_stability_filter.tooltip                      # non-empty


# ── filter ON routes to sweep_with_stability with the controls' args ────

def test_filter_on_calls_sweep_with_stability(monkeypatch):
    b = _build(monkeypatch, history=True)

    calls = {}
    def fake_sws(norb, backend, props, hist, **kw):
        calls['kw'] = kw
        return _canned_res({'available': True, 'relaxed': False,
                            'excluded_final': [1, 2, 3],
                            'window_days': kw.get('window_days')})
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability", fake_sws)
    plain = []
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: plain.append(1) or _canned_res(None))

    b.chk_stability_filter.value = True
    b.rb_stability_window_auto.value = 7
    b.ft_stability_threshold.value = 1.5
    b.btn_lucj_run.click()

    assert calls['kw']['window_days'] == 7
    assert calls['kw']['threshold'] == 1.5
    assert plain == []                                         # plain NOT called


# ── the three banner states ─────────────────────────────────────────────

def test_banner_states(monkeypatch):
    b = _build(monkeypatch, history=True)
    holder = {}
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: _canned_res(holder['stab']))
    b.chk_stability_filter.value = True

    holder['stab'] = {'available': True, 'relaxed': False,
                      'excluded_final': [1, 2, 3, 4], 'window_days': 14}
    b.btn_lucj_run.click()
    assert 'filtered 4 volatile qubits' in b.lbl_lucj_stability.value
    assert 'window 14d' in b.lbl_lucj_stability.value

    holder['stab'] = {'available': True, 'relaxed': True,
                      'n_readmitted': 2, 'n_excluded_initial': 6}
    b.btn_lucj_run.click()
    assert 'too strict' in b.lbl_lucj_stability.value
    assert 're-admitted 2 of 6' in b.lbl_lucj_stability.value

    holder['stab'] = {'available': False}
    b.btn_lucj_run.click()
    assert 'filter skipped' in b.lbl_lucj_stability.value


# ── scan rows carry stability badges ────────────────────────────────────

def test_scan_stability_badges(monkeypatch):
    b = _build(monkeypatch, history=True)
    b.chk_stability_filter.value = True        # enabled (history available)
    b.chk_scan_stability.value = True          # filter the scan too

    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': True, 'reason': None}])
    # ibm_a has CACHED history, ibm_b does not.
    from qpu_mapper import data
    monkeypatch.setattr(data, 'list_known_calibrations', lambda name, date=None:
                        [{'last_update': '2026-07-05 08:00:00'}] if name == 'ibm_a' else [])
    monkeypatch.setattr(data, '_load_calibration', lambda name, ts:
                        ({'readout_error': [0.01] * 8}, {'cz_error': [0.001] * 7}))
    # ibm_a → filtered (excluded_final); ibm_b → plain sweep.
    filt = dict(_canned_res({'available': True, 'relaxed': False,
                             'excluded_final': [3, 4], 'window_days': 14}))
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: filt)
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned_res(None))
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: SimpleNamespace(
                            vintage='live', score_layout=lambda q: {'fidelity': 0.5})))

    b.btn_scan_qpus.click()
    assert '🛡 −2 qubits' in _detail_by_name(b, 'ibm_a')       # cached → filtered
    assert 'n/a (no cached history)' in _detail_by_name(b, 'ibm_b')  # fresh → n/a


# ── filter OFF keeps the plain sweep path ───────────────────────────────

def test_filter_off_uses_plain_sweep(monkeypatch):
    b = _build(monkeypatch, history=True)

    plain = []
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: plain.append(1) or _canned_res(None))
    sws = []
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: sws.append(1) or _canned_res({'available': True}))

    assert b.chk_stability_filter.value is False               # default OFF
    b.btn_lucj_run.click()
    assert plain == [1]                                        # plain sweep used
    assert sws == []                                          # filter engine untouched
    assert b.lbl_lucj_stability.value == ""                    # no banner


# ── banner counts N volatile + M outlier from the metadata ──────────────

def test_banner_counts_volatile_and_outlier(monkeypatch):
    b = _build(monkeypatch, history=True)
    monkeypatch.setattr(
        "qpu_mapper.stability_filter.sweep_with_stability",
        lambda *a, **k: _canned_res({
            'available': True, 'relaxed': False,
            'excluded_final': [1, 2, 3, 4, 5],       # 5 total
            'n_floor_excluded': 2, 'floor_qubits': [4, 5],
            'window_days': 14}))
    b.chk_stability_filter.value = True
    b.btn_lucj_run.click()
    v = b.lbl_lucj_stability.value
    assert 'filtered 3 volatile + 2 outlier qubits' in v     # 5 − 2 floor = 3


def test_banner_no_outlier_suffix_when_floor_empty(monkeypatch):
    b = _build(monkeypatch, history=True)
    monkeypatch.setattr(
        "qpu_mapper.stability_filter.sweep_with_stability",
        lambda *a, **k: _canned_res({
            'available': True, 'relaxed': False,
            'excluded_final': [1, 2], 'n_floor_excluded': 0,
            'floor_qubits': [], 'window_days': 14}))
    b.chk_stability_filter.value = True
    b.btn_lucj_run.click()
    v = b.lbl_lucj_stability.value
    assert 'filtered 2 volatile qubits' in v and 'outlier' not in v


# ── _scan_filtered per-backend isolation + both badge paths ─────────────

def test_scan_filtered_per_backend_isolation(monkeypatch):
    b = _build(monkeypatch, history=True)
    b.chk_stability_filter.value = True
    b.chk_scan_stability.value = True
    # ibm_c (the MIDDLE target) blows up during history assembly; the loop
    # must survive so ibm_b (after it) still completes.
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_c', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': True, 'reason': None}])
    from qpu_mapper import data
    monkeypatch.setattr(data, 'list_known_calibrations', lambda name, date=None:
                        [{'last_update': '2026-07-05 08:00:00'}]
                        if name in ('ibm_a', 'ibm_c') else [])

    def _load(name, ts):
        if name == 'ibm_c':
            raise RuntimeError('cache read exploded')
        return ({'readout_error': [0.01] * 8}, {'cz_error': [0.001] * 7})
    monkeypatch.setattr(data, '_load_calibration', _load)

    filt = dict(_canned_res({'available': True, 'relaxed': False,
                             'excluded_final': [3, 4], 'window_days': 14}))
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: filt)
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned_res(None))
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: SimpleNamespace(
                            vintage='live', score_layout=lambda q: {'fidelity': 0.5})))

    b.btn_scan_qpus.click()
    # (a) isolation: ibm_c's raise became an error row, loop not killed.
    assert 'cache read exploded' in _detail_by_name(b, 'ibm_c')
    # (b) both badge paths: cached → filtered, fresh → n/a.
    assert '🛡 −2 qubits' in _detail_by_name(b, 'ibm_a')       # cached → filtered
    assert 'n/a (no cached history)' in _detail_by_name(b, 'ibm_b')  # fresh → n/a
