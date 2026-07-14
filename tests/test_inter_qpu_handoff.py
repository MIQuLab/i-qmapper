"""One-click inter-QPU handoff: selecting a remote scan candidate stages
AND auto-connects QPU 2, with graceful fallback, race guards, and a same-backend
shortcut. Headless: the network-touching connect deps are stubbed.
"""
from types import SimpleNamespace

import pytest


# ── a label that records every value it is assigned (to watch transitions) ──

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


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:                       # pragma: no cover
        if on_error:
            on_error(e)
        return
    if on_done:
        on_done(r)


def _ctx(*, spawn, backend_calls, raise_connect=False):
    def _backend(n):
        backend_calls.append(n)
        if raise_connect:
            raise RuntimeError('network down')
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
    return ctx, state


def _stub_connect_deps(monkeypatch):
    """Stub the network/UI-building deps so a connect SUCCEEDS headlessly."""
    def _fake_build(s2, is_secondary=False, return_widgets=False):
        s2.qubit_ids = ['0', '1', '2']
        s2.edges_list = [(0, 1), (1, 2)]
        fig = SimpleNamespace()
        fig_box2 = SimpleNamespace(children=[SimpleNamespace(),
                                             SimpleNamespace(), fig])
        full_ui = SimpleNamespace(children=[SimpleNamespace(), fig_box2])
        w2 = SimpleNamespace(plot_w_slider=SimpleNamespace(value=0),
                             plot_h_slider=SimpleNamespace(value=0))
        return (full_ui, fig, w2) if return_widgets else (full_ui, fig)

    monkeypatch.setattr("qpu_mapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("qpu_mapper.constants.resolve_coords",
                        lambda b: ([], 'chip_label'))
    monkeypatch.setattr("qpu_mapper.ui.build_mapper_ui", _fake_build)


_CAND = {'alpha': [0], 'beta': [1], 'ancillas': [2]}


# ── selection triggers connect once; staged→connecting→connected ────────

def test_selection_auto_connects_once(monkeypatch):
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    _stub_connect_deps(monkeypatch)
    calls = []
    ctx, state = _ctx(spawn=_sync_spawn, backend_calls=calls)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)

    assert calls == ['ibm_k']                        # connect path run exactly once
    hist = ctx.lbl_qpu2_status.history
    assert any('staged' in h for h in hist)          # staged first
    assert any('onnecting' in h for h in hist)       # then connecting
    assert 'Connected' in hist[-1] and 'ibm_k' in hist[-1]   # cleared to connected
    assert ctx.btn_connect_qpu2.disabled is False    # trigger re-enabled


# ── connect raising → staged fallback + error, trigger re-enabled ───────

def test_connect_failure_falls_back_to_staged(monkeypatch):
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    _stub_connect_deps(monkeypatch)
    calls = []
    ctx, state = _ctx(spawn=_sync_spawn, backend_calls=calls, raise_connect=True)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)

    final = ctx.lbl_qpu2_status.value
    assert 'staged' in final and 'Connect QPU 2' in final     # back to staged
    assert 'network down' in final                            # esc'd error line
    assert ctx.btn_connect_qpu2.disabled is False             # re-enabled
    # The candidate is re-staged so one manual click can recover.
    # (a manual connect would read dd_backend_qpu2.value)
    assert ctx.dd_backend_qpu2.value == 'ibm_k'


# ── in-flight guard — a second selection does not double-connect ────────

def test_in_flight_guard_no_double_connect(monkeypatch):
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    _stub_connect_deps(monkeypatch)
    calls = []
    queued = []

    def _manual_spawn(fn, on_done=None, on_error=None):
        queued.append((fn, on_done, on_error))       # defer — stays in flight

    ctx, state = _ctx(spawn=_manual_spawn, backend_calls=calls)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)              # starts a connect (queued)
    ctx.handoff_cell[0]('ibm_k', _CAND)              # second pick while in flight

    assert len(queued) == 1                          # only ONE connect launched
    assert calls == []                               # nothing has run yet
    # Draining the single queued connect runs the path exactly once.
    fn, done, _ = queued[0]
    done(fn())
    assert calls == ['ibm_k']


# ── same backend already connected → apply + repaint, no reconnect ──────

def test_same_backend_applies_without_reconnect(monkeypatch):
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    _stub_connect_deps(monkeypatch)
    repaints = []
    monkeypatch.setattr("qpu_mapper.plot.refresh_plot",
                        lambda fig, st, w: repaints.append(st))
    calls = []
    ctx, state = _ctx(spawn=_sync_spawn, backend_calls=calls)
    setup_inter_qpu(ctx)

    ctx.handoff_cell[0]('ibm_k', _CAND)              # first: real connect
    assert calls == ['ibm_k']

    cand2 = {'alpha': [3], 'beta': [4], 'ancillas': [5]}
    ctx.handoff_cell[0]('ibm_k', cand2)              # same backend again

    assert calls == ['ibm_k']                        # NO reconnect (still one call)
    assert repaints                                  # repainted in place
    # The new candidate was applied to QPU-2's live state.
    d = repaints[-1].layouts_data[0]
    assert d['alpha'] == ['3'] and d['beta'] == ['4'] and d['ancilla'] == ['5']
    assert 'Applied to already-connected' in ctx.lbl_qpu2_status.value
