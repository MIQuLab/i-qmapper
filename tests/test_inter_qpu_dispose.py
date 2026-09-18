"""Regression tests for the QPU-2 disposer leak on reconnect.

`_connect_qpu2_paint` installs a disposer (`_dispose2[0]`) that unobserves the
whole secondary widget tree so it can be garbage collected. The toggle-OFF
branch invokes it correctly, but a RECONNECT (QPU-2 already connected, user
picks a different backend) used to overwrite `_dispose2[0]` with the new
disposer WITHOUT invoking the previous one — so the prior UI + FigureWidget
stayed alive in ipywidgets' handler lists and memory grew on every reconnect.

Observing the fix
-----------------
`_walk_unobserve_all` is a closure defined inside `setup_inter_qpu`, not a
module-level function, so it cannot be monkeypatched directly. Instead the
fake `build_mapper_ui` returns a tree built from `_RecWidget` nodes that record
their own `unobserve_all()` calls. That is the boundary the disposer actually
acts on, so the assertions below exercise the real disposer end to end rather
than a stand-in: "tree N was disposed" is read off the very objects handed to
the connect that built them, which pins the identity the spec asks for.
"""
from types import SimpleNamespace


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


class _RecWidget:
    """Widget-shaped node that counts the disposer's unobserve_all() calls."""

    def __init__(self, children=()):
        self.children = list(children)
        self.unobserved = 0

    def unobserve_all(self):
        self.unobserved += 1


class _PoisonWidget(_RecWidget):
    """A node whose `children` access raises, so the failure escapes
    `_walk_unobserve_all` (which only guards the `unobserve_all()` call) and
    reaches the reconnect guard under test."""

    @property
    def children(self):
        raise RuntimeError("disposer boom")

    @children.setter
    def children(self, v):
        self._children = v


def _ctx(*, spawn):
    def _backend(n):
        return SimpleNamespace(name=n)

    state = SimpleNamespace(
        available_backends=['ibm_a', 'ibm_b'], active_backend_name='ibm_here',
        active_service=SimpleNamespace(backend=_backend),
        active_ansatz='lucj', current_colorscale='Viridis')
    return SimpleNamespace(
        state=state,
        chk_inter_qpu=_Stub(value=False),
        dd_backend_qpu2=_Stub(disabled=True, options=[], value=None),
        btn_connect_qpu2=_Stub(disabled=True),
        lbl_qpu2_status=_RecLabel(),
        plot_w_slider=_Stub(value=1700), plot_h_slider=_Stub(value=1650),
        lbl_fig1=_Stub(layout=SimpleNamespace(display='none')),
        ui_sidebar=_Stub(children=[]), ui_sidebar_content=_Stub(children=[]),
        sidebar_body=_Stub(children=[]), fig_box=_Stub(children=[]),
        qpu2_sidebar_box=_Stub(children=[]), qpu2_fig_box=_Stub(children=[]),
        full_ui=_Stub(children=[]),
        handoff_cell=[None],
        spawn_bg=spawn,
    )


def _sync_spawn(fn, on_done=None, on_error=None):
    """Inline stand-in for default_spawn_bg: fetch and completion both run
    here, so a connect is finished by the time the call returns."""
    try:
        result = fn()
    except Exception as e:                            # pragma: no cover
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(result)
    return None


_CAND = {'alpha': [0], 'beta': [1], 'ancillas': [2]}


def _install_fakes(monkeypatch, builds, *, sidebar_extra=()):
    """Patch the device fetch + UI build out. Every built tree is appended to
    `builds`, so a test can point at the exact objects a given connect made."""
    def _fake_build(s2, is_secondary=False, return_widgets=False):
        s2.qubit_ids = ['0', '1', '2']
        s2.edges_list = [(0, 1), (1, 2)]
        fig = SimpleNamespace()
        # Shape mirrors the real build: full_ui.children == [sidebar2, fig_box2]
        # and fig_box2.children[1] is the zoom bar.
        sidebar2 = _RecWidget(children=list(sidebar_extra))
        fig_box2 = _RecWidget(children=[_RecWidget(), _RecWidget(), fig])
        full_ui = _RecWidget(children=[sidebar2, fig_box2])
        w2 = SimpleNamespace(plot_w_slider=SimpleNamespace(value=0),
                             plot_h_slider=SimpleNamespace(value=0))
        builds.append(full_ui)
        return (full_ui, fig, w2) if return_widgets else (full_ui, fig)

    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("iqmapper.constants.resolve_coords",
                        lambda b: ([], 'chip_label'))
    monkeypatch.setattr("iqmapper.ui.build_mapper_ui", _fake_build)
    monkeypatch.setattr("iqmapper.plot.refresh_plot",
                        lambda fig, st, w: None)


def test_reconnect_disposes_previous_tree(monkeypatch):
    """Connecting QPU-2 to a second backend must dispose the FIRST tree."""
    from iqmapper.ui.inter_qpu import setup_inter_qpu

    builds = []
    _install_fakes(monkeypatch, builds)
    ctx = _ctx(spawn=_sync_spawn)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_a', _CAND)
    assert len(builds) == 1, "first connect did not build a QPU-2 UI"
    first_tree = builds[0]
    assert first_tree.unobserved == 0, "first connect disposed its own tree"

    ctx.handoff_cell[0]('ibm_b', _CAND)               # reconnect, new backend
    assert len(builds) == 2, "reconnect did not build a second QPU-2 UI"
    second_tree = builds[1]

    # Exactly one disposal, of the FIRST connect's tree (identity: the counter
    # lives on that very object), and the live tree is left untouched.
    assert first_tree.unobserved == 1, (
        f"previous QPU-2 tree was not disposed on reconnect "
        f"(unobserve_all calls: {first_tree.unobserved})")
    assert second_tree.unobserved == 0, "reconnect disposed the NEW tree"
    assert first_tree is not second_tree
    # The walk is recursive, so the children of the dead tree went too.
    assert all(c.unobserved == 1 for c in first_tree.children)
    assert 'Connected' in ctx.lbl_qpu2_status.value


def test_first_connect_does_not_dispose(monkeypatch):
    """With no previous tree the guard must not fire (nothing to dispose)."""
    from iqmapper.ui.inter_qpu import setup_inter_qpu

    builds = []
    _install_fakes(monkeypatch, builds)
    ctx = _ctx(spawn=_sync_spawn)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_a', _CAND)

    assert len(builds) == 1
    assert builds[0].unobserved == 0, "a first connect must dispose nothing"
    assert all(c.unobserved == 0 for c in builds[0].children)
    assert 'Connected' in ctx.lbl_qpu2_status.value


def test_raising_disposer_does_not_break_reconnect(monkeypatch, caplog):
    """A disposer that raises is logged, never fatal: the reconnect completes."""
    from iqmapper.ui.inter_qpu import setup_inter_qpu

    builds = []
    # The poison node sits inside sidebar2, which the paint path never walks —
    # only the disposer descends that far, so it breaks disposal and nothing else.
    _install_fakes(monkeypatch, builds, sidebar_extra=(_PoisonWidget(),))
    ctx = _ctx(spawn=_sync_spawn)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_a', _CAND)
    assert len(builds) == 1

    with caplog.at_level('ERROR', logger='iqmapper.ui.inter_qpu'):
        ctx.handoff_cell[0]('ibm_b', _CAND)           # disposer raises in here

    # The reconnect still completed: second tree built, figure mounted, status OK.
    assert len(builds) == 2, "a raising disposer aborted the reconnect"
    assert 'Connected' in ctx.lbl_qpu2_status.value, (
        f"reconnect did not report success: {ctx.lbl_qpu2_status.value!r}")
    assert ctx.qpu2_fig_box.children, "new QPU-2 figure was not mounted"
    # …and the failure was logged rather than swallowed or propagated.
    assert any('disposer raised on reconnect' in r.message
               for r in caplog.records), (
        f"disposer failure was not logged: {[r.message for r in caplog.records]}")
