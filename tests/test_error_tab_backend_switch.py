"""Regression tests for the Error tab's "Showing:" header after a backend switch.

The bug: ui.builder discarded the time panel's `api` handle
(``self._time_panel, _ = _build_time_panel(...)``), so nothing could re-render
the panel from outside its own observers. A backend switch happens outside
those observers, so the header kept naming the PREVIOUS backend while the plot
below it already showed the new one.

The fix threads an optional ``refresh_time_panel`` hook through the
setup_backend_switch ctx and calls it on both the success and rollback paths.
These tests lock in all three halves:
  * the panel's api.rebuild() actually re-renders the header from live state;
  * the switch handler invokes the hook exactly once on success, and a hook
    that raises does not break the switch;
  * the rollback path invokes it too, so a failed switch doesn't strand the
    header on a backend that never finished loading.

Fully offline: no service, no network, no calibration fetch.
"""
from types import SimpleNamespace

import pytest

import iqmapper.ui.backend_switch as bs
import iqmapper.ui.time_panel as tp


# ═══════════════════════════════════════════════════════════════════════
# (a) api.rebuild() re-renders the header with the CURRENT backend name
# ═══════════════════════════════════════════════════════════════════════

def _build_panel(backend_name):
    """Build a real time panel in 'live' mode against a stub state."""
    from iqmapper.state import SessionState

    state = SessionState()
    state.active_backend_name = backend_name
    state.time_mode = 'live'

    panel, api = tp.build_time_panel(
        state=state,
        data_mod=SimpleNamespace(),
        export_mod=SimpleNamespace(),
        refresh_plot=lambda: None,
        load_device_graph=lambda *a, **k: None,
        spawn_bg=lambda fn, *a, **k: None,   # never run background work
    )
    return state, api


def _header_text(api):
    return api._b.lbl_header.value


def test_rebuild_updates_header_to_new_backend():
    """The header must follow state.active_backend_name across a switch."""
    state, api = _build_panel('ibm_old')
    assert 'ibm_old' in _header_text(api)

    # Simulate what a backend switch does to state, then refresh.
    state.active_backend_name = 'ibm_new'
    api.rebuild()

    header = _header_text(api)
    assert 'ibm_new' in header
    assert 'ibm_old' not in header


def test_status_text_follows_state_without_rebuild():
    """status_text() reads state directly — it is the pure describer, and is
    what the header renders. Guards against the describer regressing."""
    state, api = _build_panel('ibm_old')
    state.active_backend_name = 'ibm_new'
    assert api.status_text() == 'Live calibration — ibm_new'


# ═══════════════════════════════════════════════════════════════════════
# (b) + (c) setup_backend_switch invokes the hook
# ═══════════════════════════════════════════════════════════════════════

class _FakeWidget:
    """Minimal stand-in for the widgets the handler touches."""

    def __init__(self, value=None):
        self.value = value
        self.disabled = False
        self.options = ()
        self._handlers = []

    def observe(self, handler, names=None):
        self._handlers.append(handler)


def _switch_ctx(monkeypatch, *, load_device_graph, refresh_time_panel,
                available=('ibm_old', 'ibm_new')):
    """Wire a switch context whose every external dependency is stubbed."""
    # Lazy imports inside the handler → patch at the source modules.
    monkeypatch.setattr('iqmapper.connection._check_properties',
                        lambda backend, name: ({'props': True}, ''),
                        raising=False)
    monkeypatch.setattr('iqmapper.constants.resolve_coords',
                        lambda backend: ({0: (0, 0)}, 'Heron r3'),
                        raising=False)

    state = SimpleNamespace(
        active_service=SimpleNamespace(backend=lambda name: SimpleNamespace(name=name)),
        active_backend=SimpleNamespace(name='ibm_old'),
        active_backend_name='ibm_old',
        active_chip='Heron r3',
        initial_props={},
        props_warning='',
        active_coords={},
        qubit_ids=[0, 1],
        layouts_data={},
        available_backends=list(available),
    )
    dd = _FakeWidget('ibm_old')
    ctx = SimpleNamespace(
        state=state,
        dd_backend_switch=dd,
        lbl_switch_status=_FakeWidget(''),
        lbl_calibration_date=_FakeWidget(''),
        lbl_props_warning=_FakeWidget(''),
        chk_keep_layout=_FakeWidget(False),
        switching_backend=[False],
        load_device_graph=load_device_graph,
        refresh=lambda: None,
        get_export=lambda: None,
        out=SimpleNamespace(),
        refresh_time_panel=refresh_time_panel,
    )
    handler = bs.setup_backend_switch(ctx)
    return handler, state, ctx


def test_hook_invoked_once_on_successful_switch(monkeypatch):
    """Success path: the hook fires exactly once, after load_device_graph."""
    order = []
    handler, state, _ = _switch_ctx(
        monkeypatch,
        load_device_graph=lambda **kw: order.append('load'),
        refresh_time_panel=lambda: order.append('refresh'),
    )

    handler({'new': 'ibm_new'})

    assert order.count('refresh') == 1, order
    assert order == ['load', 'refresh'], "hook must run AFTER the graph reload"
    assert state.active_backend_name == 'ibm_new'


def test_raising_hook_does_not_break_the_switch(monkeypatch):
    """A refresh failure must never abort a switch that otherwise succeeded."""
    def _boom():
        raise RuntimeError("panel exploded")

    handler, state, ctx = _switch_ctx(
        monkeypatch,
        load_device_graph=lambda **kw: None,
        refresh_time_panel=_boom,
    )

    handler({'new': 'ibm_new'})   # must not raise

    assert state.active_backend_name == 'ibm_new'
    assert '✅' in ctx.lbl_switch_status.value
    assert ctx.switching_backend[0] is False


def test_hook_invoked_on_rollback_path(monkeypatch):
    """Failed switch: state rolls back, so the header must be refreshed too."""
    calls = []

    def _explode(**kw):
        raise RuntimeError("graph load failed")

    handler, state, ctx = _switch_ctx(
        monkeypatch,
        load_device_graph=_explode,
        refresh_time_panel=lambda: calls.append(state.active_backend_name),
    )

    handler({'new': 'ibm_new'})   # must not raise

    # Rolled back to the original backend...
    assert state.active_backend_name == 'ibm_old'
    # ...and the hook saw the rolled-back name, not the failed one.
    assert calls == ['ibm_old'], calls
    assert '❌' in ctx.lbl_switch_status.value


def test_builder_retains_time_api_and_wires_the_hook(monkeypatch):
    """The original bug lived here: the builder threw the api handle away.

    Locks both halves of the wiring — the handle is kept as `_time_api`, and
    the ctx passed to setup_backend_switch carries a working hook.
    """
    import numpy as np

    def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
        n = 5
        return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
                [str(i) for i in range(n)],
                np.array([0.01 * (i + 1) for i in range(n)]),
                np.array([0.001 * (i + 1) for i in range(n - 1)]))

    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)

    captured = {}
    real_setup = bs.setup_backend_switch
    monkeypatch.setattr(
        "iqmapper.ui.backend_switch.setup_backend_switch",
        lambda ctx: (captured.setdefault('ctx', ctx), real_setup(ctx))[1],
    )

    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder

    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "ibm_old"
    state.active_chip = "stub_chip"
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = None
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]

    builder = MapperUIBuilder(state)
    builder.build()

    # The handle survives construction (the regression).
    assert getattr(builder, '_time_api', None) is not None

    # And the switch wiring received a hook that actually drives it.
    hook = getattr(captured['ctx'], 'refresh_time_panel', None)
    assert callable(hook)

    state.active_backend_name = 'ibm_new'
    hook()
    assert 'ibm_new' in builder._time_api._b.lbl_header.value


def test_missing_hook_is_a_noop(monkeypatch):
    """Callers that never build a time panel omit the ctx key entirely."""
    handler, state, _ = _switch_ctx(
        monkeypatch,
        load_device_graph=lambda **kw: None,
        refresh_time_panel=None,
    )

    handler({'new': 'ibm_new'})   # must not raise

    assert state.active_backend_name == 'ibm_new'
