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


# ── batch handoff: a LIST of candidates applies to C1..Cn on QPU-2 ──────────

def test_batch_handoff_applies_all_circuits(monkeypatch):
    from qpu_mapper.ui.inter_qpu import setup_inter_qpu
    captured = {}

    def _fake_build(s2, is_secondary=False, return_widgets=False):
        captured['s2'] = s2
        s2.qubit_ids = [str(i) for i in range(8)]
        s2.edges_list = [(0, 1)]
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
                        lambda b: ([], 'chip'))
    monkeypatch.setattr("qpu_mapper.ui.build_mapper_ui", _fake_build)

    ctx, state = _ctx(spawn=_sync_spawn, backend_calls=[])
    setup_inter_qpu(ctx)
    batch = [{'alpha': [0], 'beta': [1], 'ancillas': [2]},
             {'alpha': [3], 'beta': [4], 'ancillas': [5]}]
    ctx.handoff_cell[0]('ibm_k', batch)              # LIST → batch handoff

    s2 = captured['s2']
    assert s2.layouts_data[0]['alpha'] == ['0'] and s2.layouts_data[0]['ancilla'] == ['2']
    assert s2.layouts_data[1]['alpha'] == ['3'] and s2.layouts_data[1]['ancilla'] == ['5']


# ── PAINT-PATH regression: the REAL build wipes layouts mid-connect ──────────
#    build_mapper_ui → _load_device_graph → state.clear_all_layouts(). A
#    candidate applied BEFORE the build was silently wiped and never painted.
#    This drives the REAL build (only the device/network deps are stubbed) and
#    proves the candidate both LANDS on QPU-2 state AND reaches the figure.

def _qubit_marker_colors(fig):
    """The per-qubit marker colour array from the figure (len == n_qubits)."""
    for tr in fig.data:
        mc = getattr(getattr(tr, 'marker', None), 'color', None)
        if isinstance(mc, (list, tuple)) and len(mc) == 8:
            return list(mc)
    return []


def test_first_connect_paints_candidate_real_build(monkeypatch):
    import numpy as np

    def _fake_device(state, vendor, chip, date, prefetched_props=None):
        n = 8
        return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
                [str(i) for i in range(n)],
                np.array([0.01 * (i + 1) for i in range(n)]),
                np.array([0.001 * (i + 1) for i in range(n - 1)]))

    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device)
    monkeypatch.setattr("qpu_mapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("qpu_mapper.constants.resolve_coords",
                        lambda b: ([(i, 0) for i in range(8)], 'chip'))
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())

    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = 'ibm_here'
    state.active_chip = 'chip'
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
    state.available_backends = ['ibm_k']
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = 'lucj'
    state.active_vendor = 'IBM'
    state.active_coords = [(i, 0) for i in range(8)]
    state.custom_colors = []
    state.cal_history_dates = ['_seed_']
    b = MapperUIBuilder(state)
    b.build()

    # Capture the connected QPU-2 state + figure by wrapping the REAL build.
    import qpu_mapper.ui as ui_pkg
    real = ui_pkg.build_mapper_ui
    cap = {}

    def _wrap(s2, is_secondary=False, return_widgets=False):
        res = real(s2, is_secondary=is_secondary, return_widgets=return_widgets)
        cap['s2'], cap['fig'] = s2, res[1]
        return res
    monkeypatch.setattr(ui_pkg, 'build_mapper_ui', _wrap)

    cand = {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': [4], 'lqs_circuit': 0.54}
    b._inter_qpu_handoff[0]('ibm_k', cand)

    # (a) candidate LANDED on QPU-2 state (survived clear_all_layouts()).
    d = cap['s2'].layouts_data[0]
    assert d['alpha'] == ['0', '1'] and d['beta'] == ['2', '3'] and d['ancilla'] == ['4']
    assert d['styles']                                    # per-qubit paint styles set
    # (b) figure RECEIVED the layout — the 5 candidate qubits are painted, not
    # the unpainted white; an untouched qubit stays white (specificity).
    colors = _qubit_marker_colors(cap['fig'])
    assert len(colors) == 8
    assert all(colors[i] != '#ffffff' for i in (0, 1, 2, 3, 4))   # painted
    assert colors[7] == '#ffffff'                                  # untouched
