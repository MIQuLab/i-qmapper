"""Headless test for the QPU-1 switcher-label refresh (stale-title fix).

`ui.inter_qpu.setup_inter_qpu` returns a `refresh_qpu1_label` hook that
`ui.backend_switch` calls after a backend switch so the QPU-1/QPU-2
ToggleButtons stops showing the previous backend name. This drives the real
setup with SimpleNamespace/stub widgets (no kernel, no network), turns the
inter-QPU mode on to materialize the real ToggleButtons, then exercises the
hook directly.
"""
from types import SimpleNamespace

from iqpu_mapper.ui.inter_qpu import setup_inter_qpu


class _Stub:
    """Minimal widget stand-in: records .observe callbacks and exposes the
    handful of attributes the inter-QPU toggle handler touches."""

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


def _make_ctx():
    state = SimpleNamespace(available_backends=[],
                            active_backend_name='old_backend')
    ctx = SimpleNamespace(
        state=state,
        chk_inter_qpu=_Stub(value=False),
        dd_backend_qpu2=_Stub(disabled=True, options=[]),
        btn_connect_qpu2=_Stub(disabled=True),
        lbl_qpu2_status=_Stub(value=''),
        plot_w_slider=_Stub(value=1700),
        plot_h_slider=_Stub(value=1650),
        lbl_fig1=_Stub(layout=SimpleNamespace(display='none')),
        ui_sidebar=_Stub(children=[]),
        ui_sidebar_content=_Stub(children=[]),
        sidebar_body=_Stub(children=[]),
        fig_box=_Stub(children=[]),
        qpu2_sidebar_box=_Stub(children=[]),
        qpu2_fig_box=_Stub(children=[]),
        full_ui=_Stub(children=[]),
    )
    return ctx, state


def _open_inter_qpu(ctx):
    """Turn inter-QPU mode on and return the real ToggleButtons the handler
    created (mounted as ui_sidebar.children[0])."""
    ctx.chk_inter_qpu.fire(True)
    return ctx.ui_sidebar.children[0]


def test_refresh_updates_qpu1_label_only():
    ctx, state = _make_ctx()
    refresh = setup_inter_qpu(ctx)
    toggle = _open_inter_qpu(ctx)

    # Simulate the post-connect state: QPU-1 shows the (now old) backend,
    # QPU-2 shows the second backend, user is viewing QPU-2.
    toggle.options = [('📡 old', 'qpu1'), ('📡 qpu2', 'qpu2')]
    toggle.value = 'qpu2'

    # A backend switch changes the active name, then calls the hook.
    state.active_backend_name = 'ibm_new'
    refresh()

    opts = list(toggle.options)
    assert opts[0] == ('📡 ibm_new', 'qpu1')   # option[0] label refreshed
    assert opts[1] == ('📡 qpu2', 'qpu2')       # option[1] untouched
    assert toggle.value == 'qpu2'               # selection preserved


def test_refresh_noop_when_toggle_absent():
    ctx, state = _make_ctx()
    refresh = setup_inter_qpu(ctx)
    # Inter-QPU never opened -> _qpu_toggle is None -> must return silently.
    state.active_backend_name = 'ibm_new'
    refresh()  # no raise


def test_refresh_noop_on_single_option_toggle():
    ctx, state = _make_ctx()
    refresh = setup_inter_qpu(ctx)
    toggle = _open_inter_qpu(ctx)

    toggle.options = [('📡 solo', 'qpu1')]      # != 2 options -> guard trips
    toggle.value = 'qpu1'
    state.active_backend_name = 'ibm_new'
    refresh()  # no raise

    assert list(toggle.options) == [('📡 solo', 'qpu1')]  # unchanged
