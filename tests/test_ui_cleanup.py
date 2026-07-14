"""Headless tests for the scan/sweep UI cleanup (C1 separation, C2 dismissal,
C3 staged banner, C4 labels)."""
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
    except Exception as e:
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _entry(prod, k):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': list(range(4, 4 + k)),
            'bridge_slots': tuple(range(0, 4 * k, 4)),
            'product_score': prod, 'additive_score': 0.4162,
            'rank_in_stratum': 1, 'source': 'derived'}


def _canned():
    return {'strata': {2: [_entry(0.30, 2)], 1: [_entry(0.40, 1)]},
            'k_max_effective': 2, 'min_bridges': 1, 'vf2_truncated': False,
            'n_raw_embeddings': 5, 'n_candidates_scored': 5,
            'excluded_qubits': []}


def _build(monkeypatch):
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)
    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "ibm_here"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: object())
    state.available_backends = ["ibm_here", "ibm_other"]
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]
    b = MapperUIBuilder(state)
    b.build()
    return b


# ── local block renders ABOVE the scan block ────────────────────────────

def test_local_block_above_scan(monkeypatch):
    b = _build(monkeypatch)
    view = b._build_auto_lucj_view()
    children = list(view.children)
    # The local sweep dropdown must precede the scan results container.
    assert children.index(b.dd_lucj_layout) < children.index(b.scan_results_box)
    # And the "This backend (<name>)" header exists, with the active name.
    from ipywidgets import HTML
    headers = [c.value for c in children if isinstance(c, HTML)]
    assert any('📍 This backend (ibm_here)' in h for h in headers)


# ── clear resets everything and re-arms the button ──────────────────────

def test_clear_scan_results(monkeypatch):
    b = _build(monkeypatch)
    import qpu_mapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_other', 'family': 'Heron', 'eligible': True,
         'reason': None}])
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {
        'results': [{'name': 'ibm_other', 'family': 'Heron',
                     'sweep': _canned(), 'best_lqs': 0.4, 'best_k': 1,
                     'error': None}],
        'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0})

    b.btn_scan_qpus.click()
    assert b.scan_results_box.children                 # populated
    assert b.btn_scan_qpus.disabled is False           # re-enabled on done

    b.btn_scan_clear.click()
    assert b.scan_results_box.children == ()           # table + dropdowns gone
    assert b.lbl_scan_progress.value == ""             # progress cleared
    assert b.btn_scan_qpus.disabled is False           # button armed


def test_new_scan_auto_clears(monkeypatch):
    b = _build(monkeypatch)
    import qpu_mapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_other', 'family': 'Heron', 'eligible': True,
         'reason': None}])
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {
        'results': [{'name': 'ibm_other', 'family': 'Heron',
                     'sweep': _canned(), 'best_lqs': 0.4, 'best_k': 1,
                     'error': None}], 'n_scanned': 1, 'n_failed': 0,
        'n_skipped': 0})
    b.btn_scan_qpus.click()
    first = list(b.scan_results_box.children)
    b.btn_scan_qpus.click()                            # a second scan
    # Fresh render, not appended onto the stale one.
    assert list(b.scan_results_box.children) != first + first


# ── staged banner appears on stage, cleared on connect ──────────────────

class _Stub:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self._handlers = []

    def observe(self, fn, names=None):
        self._handlers.append(fn)

    def on_click(self, fn):
        self._click = fn

    def fire(self, new):
        for fn in self._handlers:
            fn({'new': new})


def _inter_ctx():
    from ipywidgets import VBox
    state = SimpleNamespace(
        available_backends=[], active_backend_name='ibm_here',
        active_service=SimpleNamespace(
            backend=lambda n: (_ for _ in ()).throw(RuntimeError('stub'))))
    ctx = SimpleNamespace(
        state=state,
        chk_inter_qpu=_Stub(value=False),
        dd_backend_qpu2=_Stub(disabled=True, options=[], value=None),
        btn_connect_qpu2=_Stub(disabled=True),
        lbl_qpu2_status=_Stub(value=''),
        plot_w_slider=_Stub(value=1700), plot_h_slider=_Stub(value=1650),
        lbl_fig1=_Stub(layout=SimpleNamespace(display='none')),
        ui_sidebar=_Stub(children=[]), ui_sidebar_content=_Stub(children=[]),
        sidebar_body=_Stub(children=[]), fig_box=_Stub(children=[]),
        qpu2_sidebar_box=_Stub(children=[]), qpu2_fig_box=_Stub(children=[]),
        full_ui=_Stub(children=[]),
        handoff_cell=[None],
        spawn_bg=_sync_spawn,      # deterministic one-click auto-connect
    )
    return ctx, state


def test_staged_banner_then_cleared_on_connect():
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    ctx, state = _inter_ctx()
    setup_inter_qpu(ctx)
    stage = ctx.handoff_cell[0]
    assert stage is not None

    stage('ibm_remote', {'alpha': [0], 'beta': [1], 'ancillas': [2]})
    assert 'ibm_remote' in ctx.lbl_qpu2_status.value
    assert 'staged' in ctx.lbl_qpu2_status.value
    assert 'Connect QPU 2' in ctx.lbl_qpu2_status.value

    # Invoking the (stubbed) connect replaces the staged banner.
    ctx.btn_connect_qpu2._click(None)
    assert 'staged' not in ctx.lbl_qpu2_status.value


# ── label disambiguation in the detail + dropdown ───────────────────────

def test_label_disambiguation(monkeypatch):
    b = _build(monkeypatch)
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned())
    b.lucj_ancilla_dd.value = -1                       # sweep mode
    b.btn_lucj_run.click()

    # Dropdown labels keep the short "LQS" token.
    assert any('LQS' in opt[0] for opt in b.dd_lucj_layout.options)

    # Selecting an entry renders the disambiguated detail wording.
    b.dd_lucj_layout.value = 1                         # fire select
    detail = b.lbl_lucj_detail.value
    assert 'template fidelity (single-pass edges)' in detail
    assert 'internal ranking sum 0.4162' in detail
    assert '(add ' not in detail                       # old wording gone
