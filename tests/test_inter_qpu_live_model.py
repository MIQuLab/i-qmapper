"""LIVE-model regression for the QPU-2 empty-map bug.

The one-click auto-connect must not hand the connect (build_mapper_ui + apply
candidate + refresh_plot + mounting the FigureWidget) to a background worker. A
plotly FigureWidget mutated off the kernel main thread sends its comm messages
on a non-kernel thread and the browser silently drops them: the QPU-2 map
paints nothing, with no error and no exception — invisible to a suite whose
`spawn_bg` runs inline.

This test reproduces the LIVE threading model deterministically: the injected
`spawn_bg` runs the worker fn on a real background thread and marshals the
completion callback back to the MAIN (test) thread, exactly like
`default_spawn_bg` does against a running kernel loop. It then records which
thread each FigureWidget-touching operation (`build_mapper_ui`, `refresh_plot`)
executes on and asserts it is the main thread.

Pre-fix: build + refresh run on the worker  -> assertion fails (bug present).
Post-fix: only the network fetch runs on the worker; build + refresh are
marshalled to main -> assertion passes.
"""
import queue
import threading
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


def _ctx(*, spawn):
    def _backend(n):
        return SimpleNamespace(name=n)

    state = SimpleNamespace(
        available_backends=['ibm_k'], active_backend_name='ibm_here',
        active_service=SimpleNamespace(backend=_backend),
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
        qpu2_sidebar_box=_Stub(children=[]), qpu2_fig_box=_Stub(children=[]),
        full_ui=_Stub(children=[]),
        handoff_cell=[None],
        spawn_bg=spawn,
    )
    return ctx


_CAND = {'alpha': [0], 'beta': [1], 'ancillas': [2]}


def test_auto_connect_paints_on_main_thread(monkeypatch):
    from iqmapper.ui.inter_qpu import setup_inter_qpu

    main_thread = threading.current_thread()
    build_threads = []
    refresh_threads = []

    def _fake_build(s2, is_secondary=False, return_widgets=False):
        build_threads.append(threading.current_thread())
        s2.qubit_ids = ['0', '1', '2']
        s2.edges_list = [(0, 1), (1, 2)]
        fig = SimpleNamespace()
        fig_box2 = SimpleNamespace(children=[SimpleNamespace(),
                                             SimpleNamespace(), fig])
        full_ui = SimpleNamespace(children=[SimpleNamespace(), fig_box2])
        w2 = SimpleNamespace(plot_w_slider=SimpleNamespace(value=0),
                             plot_h_slider=SimpleNamespace(value=0))
        return (full_ui, fig, w2) if return_widgets else (full_ui, fig)

    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("iqmapper.constants.resolve_coords",
                        lambda b: ([], 'chip_label'))
    monkeypatch.setattr("iqmapper.ui.build_mapper_ui", _fake_build)
    monkeypatch.setattr("iqmapper.plot.refresh_plot",
                        lambda fig, st, w: refresh_threads.append(
                            threading.current_thread()))

    # A spawn that mimics a LIVE kernel: worker fn on a background thread,
    # completion callback marshalled back to the main thread via a queue the
    # test drains. (default_spawn_bg does the same against a running loop.)
    marshal_q = queue.Queue()

    def live_spawn(fn, on_done=None, on_error=None):
        def _worker():
            try:
                r = fn()
            except Exception as e:                    # pragma: no cover
                if on_error:
                    marshal_q.put((on_error, e))
                return
            if on_done:
                marshal_q.put((on_done, r))
        t = threading.Thread(target=_worker)
        t.start()
        t.join()
        return t

    ctx = _ctx(spawn=live_spawn)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)               # scan pick -> auto-connect
    # Drain the marshalled completion(s) on the MAIN thread.
    while not marshal_q.empty():
        cb, arg = marshal_q.get()
        cb(arg)

    assert build_threads, "build_mapper_ui never ran"
    assert refresh_threads, "refresh_plot never ran"
    assert all(t is main_thread for t in build_threads), (
        f"build_mapper_ui ran off the main thread: {build_threads}")
    assert all(t is main_thread for t in refresh_threads), (
        f"refresh_plot ran off the main thread: {refresh_threads}")
    # And the connect still succeeded end to end.
    assert 'Connected' in ctx.lbl_qpu2_status.value
