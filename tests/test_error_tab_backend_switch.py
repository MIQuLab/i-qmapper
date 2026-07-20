"""Error-tab header must follow a backend switch.

The header text comes from time_panel.describe_current_view(state) but only
re-renders when the panel's own observers fire — a backend switch never touches
them. The builder now KEEPS the time-panel API handle and the switch wiring
calls api.rebuild() (best-effort, never breaking a switch) so the header
re-renders for the new backend.
"""
import types
from types import SimpleNamespace

import pytest


# ── stubs shared by the switch-wiring tests ─────────────────────────────────

class _W:
    """Minimal widget stub: settable attrs + a no-op observe."""
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def observe(self, *a, **k):
        pass


def _data_stub():
    return types.SimpleNamespace(
        list_known_calibrations=lambda bn, d: [],
        scan_day_calibrations=lambda s, b, d, progress_callback=None: [],
        get_props_for_datetime=lambda s, b, t: (None, {}, {}),
        fetch_calibration_history=lambda s, a, b, progress_callback=None: 0,
        apply_history_snapshot=lambda s, ds: True,
        _extract_last_update=lambda p: None,
        _load_calibration=lambda bn, lu: None,
    )


def _export_stub():
    return types.SimpleNamespace(compute_error_delta=lambda a, b, c, d: ({}, {}))


# ── 1. header re-renders on api.rebuild() (the switch hook's effect) ─────────

def test_header_updates_after_backend_switch():
    from qpu_mapper.ui.time_panel import build_time_panel
    from qpu_mapper.state import SessionState

    state = SessionState()
    state.active_backend_name = 'ibm_old'
    state.time_mode = 'live'
    panel, api = build_time_panel(
        state=state, data_mod=_data_stub(), export_mod=_export_stub(),
        refresh_plot=lambda: None, load_device_graph=lambda: None,
        spawn_bg=lambda fn, on_done=None, on_error=None: (
            on_done(fn()) if on_done else fn()),
    )
    api.rebuild()
    assert 'ibm_old' in api._b.lbl_header.value

    # Post-switch condition (as the builder's reset produces it, then the
    # switch wiring calls api.rebuild()).
    state.active_backend_name = 'ibm_new'
    state.time_mode = 'live'
    state._live_qubit_props = None
    api.rebuild()

    assert 'ibm_new' in api._b.lbl_header.value
    assert 'ibm_old' not in api._b.lbl_header.value


# ── switch-wiring: hook fires on success (after load) + tolerates failure ────

def _switch_ctx(state, *, load_device_graph, refresh_time_panel):
    return SimpleNamespace(
        state=state,
        dd_backend_switch=_W(disabled=False, options=[], value=state.active_backend_name),
        lbl_switch_status=_W(value=''),
        lbl_calibration_date=_W(value=''),
        lbl_props_warning=_W(value=''),
        chk_keep_layout=_W(value=False),
        switching_backend=[False],
        load_device_graph=load_device_graph,
        refresh=lambda: None, get_export=lambda: None,
        out=SimpleNamespace(),
        refresh_time_panel=refresh_time_panel,
    )


def _fresh_state():
    from qpu_mapper.state import SessionState
    state = SessionState()
    state.active_backend_name = 'ibm_old'
    state.active_chip = 'Heron r1'
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
    state.available_backends = ['ibm_old', 'ibm_new']
    return state


def test_switch_invokes_time_panel_hook(monkeypatch):
    from qpu_mapper.ui.backend_switch import setup_backend_switch
    monkeypatch.setattr("qpu_mapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("qpu_mapper.constants.resolve_coords",
                        lambda b: ([], 'Heron r1'))

    events = []
    state = _fresh_state()
    ctx = _switch_ctx(
        state,
        load_device_graph=lambda force=False: events.append('load'),
        refresh_time_panel=lambda: events.append('hook'))
    setup_backend_switch(ctx)({'new': 'ibm_new'})

    assert events.count('hook') == 1                     # ran exactly once
    assert events.index('load') < events.index('hook')   # AFTER load_device_graph
    assert state.active_backend_name == 'ibm_new'        # switch succeeded

    # A raising hook must NOT break the switch.
    def _boom():
        raise RuntimeError('hook boom')
    state2 = _fresh_state()
    ctx2 = _switch_ctx(
        state2, load_device_graph=lambda force=False: None,
        refresh_time_panel=_boom)
    setup_backend_switch(ctx2)({'new': 'ibm_new'})
    assert state2.active_backend_name == 'ibm_new'       # switched despite hook error


def test_switch_rollback_invokes_hook(monkeypatch):
    from qpu_mapper.ui.backend_switch import setup_backend_switch
    monkeypatch.setattr("qpu_mapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("qpu_mapper.constants.resolve_coords",
                        lambda b: ([], 'Heron r1'))

    events = []

    def _load_fails(force=False):
        raise RuntimeError('graph load failed')
    state = _fresh_state()
    ctx = _switch_ctx(
        state, load_device_graph=_load_fails,
        refresh_time_panel=lambda: events.append('hook'))
    setup_backend_switch(ctx)({'new': 'ibm_new'})

    assert events.count('hook') == 1                     # hook ran on rollback too
    assert state.active_backend_name == 'ibm_old'        # state rolled back
