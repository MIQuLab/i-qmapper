"""QPU-2 empty-map fix: resync the FigureWidget state into its synced traits
BEFORE the figure is first mounted via Box.children.

plotly 6.x snapshots _widget_data/_widget_layout only at construction and in
_repr_mimebundle_ (cell display); a figure mounted as a child widget passes
through neither, so the frontend model renders only the incremental messages it
managed to apply. In the live one-click handoff that lost the qubit node trace
→ "edges but no nodes". _resync_widget_state re-snapshots the trait state before
the mount so the view created by the mount renders the complete figure.
"""
from types import SimpleNamespace

import pytest


# ── unit: the helper snapshots data + layout onto the synced traits ─────────

def test_resync_helper_snapshots_state():
    pytest.importorskip('plotly')
    import plotly.graph_objects as go
    from iqmapper.ui.inter_qpu import _resync_widget_state

    fig = go.FigureWidget()
    fig.add_trace(go.Scatter(x=[1.0, 2.0, 3.0], y=[0.0, 0.0, 0.0],
                             text=['a', 'b', 'c'], mode='markers+text'))
    fig.add_shape(type='line', x0=0, y0=0, x1=1, y1=1)

    assert _resync_widget_state(fig) is True
    # The node trace data landed in the synced _widget_data trait…
    assert list(fig._widget_data[0]['x']) == [1.0, 2.0, 3.0]
    # …and the layout shapes landed in the synced _widget_layout trait.
    assert fig._widget_layout.get('shapes')


def test_resync_helper_never_raises():
    from iqmapper.ui.inter_qpu import _resync_widget_state
    # Non-figure object → the plotly-private attribute access fails; the guard
    # swallows it and returns False (never fatal to the connect).
    assert _resync_widget_state(object()) is False


# ── integration: the auto-connect resyncs BEFORE mounting the figure ────────

class _RecLabel:
    def __init__(self):
        self.history = []

    @property
    def value(self):
        return self.history[-1] if self.history else ''

    @value.setter
    def value(self, v):
        self.history.append(v)


class _Stub:
    def __init__(self, **attrs):
        self.__dict__.update(attrs)
        self._handlers = []

    def observe(self, fn, names=None):
        self._handlers.append(fn)

    def on_click(self, fn):
        self._click = fn


class _RecChildrenBox:
    """A container whose ``children`` setter records ('mount', tuple) events."""
    def __init__(self, events):
        self._events = events
        self._ch = []

    @property
    def children(self):
        return self._ch

    @children.setter
    def children(self, v):
        self._ch = v
        self._events.append(('mount', tuple(v)))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:                       # pragma: no cover
        if on_error:
            on_error(e)
        return
    if on_done:
        on_done(r)


_CAND = {'alpha': [0], 'beta': [1], 'ancillas': [2]}


def test_paint_resyncs_before_mount(monkeypatch):
    from iqmapper.ui.inter_qpu import setup_inter_qpu

    events = []

    def _rec_resync(fig):
        events.append(('resync', fig))
        return True
    monkeypatch.setattr("iqmapper.ui.inter_qpu._resync_widget_state",
                        _rec_resync)

    def _fake_build(s2, is_secondary=False, return_widgets=False):
        s2.qubit_ids = ['0', '1', '2']
        s2.edges_list = [(0, 1), (1, 2)]
        fig = SimpleNamespace(name='qpu2-fig')
        fig_box2 = SimpleNamespace(children=[SimpleNamespace(),
                                             SimpleNamespace(), fig])
        full_ui = SimpleNamespace(children=[SimpleNamespace(), fig_box2])
        w2 = SimpleNamespace(plot_w_slider=SimpleNamespace(value=0),
                             plot_h_slider=SimpleNamespace(value=0))
        return (full_ui, fig, w2) if return_widgets else (full_ui, fig)

    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("iqmapper.constants.resolve_coords",
                        lambda b: ([], 'chip'))
    monkeypatch.setattr("iqmapper.ui.build_mapper_ui", _fake_build)
    monkeypatch.setattr("iqmapper.plot.refresh_plot", lambda fig, st, w: None)

    state = SimpleNamespace(
        available_backends=['ibm_k'], active_backend_name='ibm_here',
        active_service=SimpleNamespace(backend=lambda n: SimpleNamespace(name=n)),
        active_ansatz='lucj', current_colorscale='Viridis')
    ctx = SimpleNamespace(
        state=state,
        chk_inter_qpu=_Stub(value=False),
        dd_backend_qpu2=_Stub(disabled=True, options=[], value=None),
        btn_connect_qpu2=_Stub(disabled=True),
        lbl_qpu2_status=_RecLabel(),
        plot_w_slider=_Stub(value=1700), plot_h_slider=_Stub(value=1650),
        lbl_fig1=_Stub(layout=SimpleNamespace(display='none')),
        ui_sidebar=_Stub(children=[]), ui_sidebar_content=_Stub(children=[]),
        sidebar_body=_Stub(children=[]), fig_box=_Stub(children=[]),
        qpu2_sidebar_box=_Stub(children=[]),
        qpu2_fig_box=_RecChildrenBox(events),
        full_ui=_Stub(children=[]),
        handoff_cell=[None],
        spawn_bg=_sync_spawn,
    )
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)               # scan pick → auto-connect

    resyncs = [e for e in events if e[0] == 'resync']
    assert len(resyncs) == 1                           # exactly one resync
    resync_fig = resyncs[0][1]
    # The mount that carries the figure (the toggle's empty init, if any, is
    # ignored) must contain the SAME figure object that was resynced…
    fig_mounts = [i for i, e in enumerate(events)
                  if e[0] == 'mount' and resync_fig in e[1]]
    assert len(fig_mounts) == 1
    # …and the resync must be ordered BEFORE that mount.
    assert events.index(resyncs[0]) < fig_mounts[0]
