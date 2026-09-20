"""UI tests for the 📡 Benchmark section.

Everything runs against a FAKE runner injected through the panel ctx, so no
test ever imports qiskit_experiments or reaches a network. The last test is
the important one: it pins the rule that the multi-QPU scan never benchmarks.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from iqmapper import benchmark as bench


# ── harness ─────────────────────────────────────────────────────────

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
        self._observers = []
        self.layout = SimpleNamespace(display='')

    def observe(self, fn, names=None):
        self._observers.append(fn)

    def on_click(self, fn):
        self._click = fn

    def click(self):
        self._click(self)


class _Dropdown(_Stub):
    """A stub whose .value assignment fires observers, as ipywidgets does."""

    def __init__(self, value=None, **attrs):
        super().__init__(**attrs)
        self.__dict__['value'] = value

    def __setattr__(self, name, val):
        fire = (name == 'value' and self.__dict__.get('value') != val
                and self.__dict__.get('_observers') is not None)
        object.__setattr__(self, name, val)
        if fire:
            for fn in list(self._observers):
                fn({'new': val})


def _line_edges(n):
    return [(i, i + 1) for i in range(n - 1)]


def _state(n_q=8):
    return SimpleNamespace(
        qubit_ids=[str(i) for i in range(n_q)],
        edges_list=_line_edges(n_q),
        layouts_data={},
        active_backend='backend-obj', active_backend_name='ibm_test',
        active_service=None,
        qubit_props_data={'readout_error': np.full(n_q, 0.05)},
        gate_props_data={'cz_error': np.full(n_q - 1, 0.01)},
        calibration_data_label='ibm_test — reported', delta_mode=False)


def _entry(alpha, beta, ancillas, lqs=0.5):
    return {'alpha': list(alpha), 'beta': list(beta),
            'ancillas': list(ancillas), 'lqs_circuit': lqs,
            'rank_in_stratum': 1}


def _ctx(*, state=None, candidates=None, runner=None, spawn=None,
         rescore=None, qpu2=None, inter_on=False, target=None):
    state = state or _state()
    return SimpleNamespace(
        state=state,
        dd_bench_target=_Dropdown(
            value=target or bench.TARGET_CURRENT,
            options=[(bench.TARGET_LABELS[m], m)
                     for m in (bench.TARGET_CURRENT, bench.TARGET_WHOLE,
                               bench.TARGET_CUSTOM)]),
        txt_bench_custom=_Dropdown(value=''),
        tb_bench_qpu=_Dropdown(value=1),
        dd_bench_level=_Dropdown(value=bench.LEVEL_QUICK_READOUT),
        lbl_bench_summary=_RecLabel(),
        chk_bench_ack=_Dropdown(value=False),
        btn_bench_run=_Stub(disabled=True),
        lbl_bench_status=_RecLabel(),
        chk_bench_absolute=_Dropdown(value=False),
        lbl_bench_consistency=_RecLabel(),
        lbl_bench_reuse=_RecLabel(),
        btn_bench_reuse=_Stub(disabled=True),
        lbl_lucj_notices=_RecLabel(),
        lucj_all_layouts=list(candidates or []),
        rescore_cell=[rescore],
        refresh=lambda: None,
        chk_inter_qpu=_Dropdown(value=inter_on),
        qpu2_handles_cell=[(lambda: qpu2)] if qpu2 is not None else [None],
        bench_spawn_bg=spawn,
        bench_runner_factory=(lambda _s: runner) if runner else None,
    )


class _MitigatorStub:
    """assignment_matrix is a METHOD, as the real mitigator's is."""

    def __init__(self, errors):
        self._e = dict(errors)

    def assignment_matrix(self, qubits=None):
        e = self._e[int(list(qubits)[0])]
        return np.array([[1 - e, e], [e, 1 - e]])


def _fake_runner(plan):
    """Plausible results for whatever the plan asked for."""
    out = {}
    for idx, spec in enumerate(plan.specs):
        if spec.kind == 'readout':
            # One mitigator for the whole target set, with assignment_matrix
            # as a METHOD — the real shape. Errors sit just above the
            # reported 0.05 so the run stays inside the consistency band and
            # is applied in absolute mode.
            out[idx] = [SimpleNamespace(
                name='Local Readout Mitigator',
                device_components=[f'Q{q}' for q in spec.qubits],
                value=_MitigatorStub({q: 0.06 for q in spec.qubits}))]
        elif spec.kind == 'rb2q':
            out[idx] = [SimpleNamespace(name='EPG', value=0.002)]
        else:
            out[idx] = []
    return out


def _have_qe(monkeypatch, present=True):
    """The panel disables itself without the extra; pretend it is installed."""
    monkeypatch.setattr(bench, 'have_qiskit_experiments', lambda: present)


def _setup(ctx):
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    return setup_benchmark_panel(ctx)


# ── 8a. gating on targets and acknowledgement ───────────────────────

def test_button_disabled_without_targets(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(candidates=[])
    _setup(ctx)
    assert ctx.btn_bench_run.disabled is True
    assert 'Targets: 0 qubits' in ctx.lbl_bench_summary.value


def test_summary_shows_cost_before_the_click(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], target=bench.TARGET_SWEEP)
    _setup(ctx)
    summary = ctx.lbl_bench_summary.value
    assert 'Targets: 4 qubits' in summary and '3 edges' in summary
    assert 'QPU (estimate)' in summary
    assert ctx.btn_bench_run.disabled is False, 'L1 on Open needs no ack'


def test_open_plan_requires_acknowledgement_for_l2_and_l3(monkeypatch,
                                                          tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], target=bench.TARGET_SWEEP)
    _setup(ctx)                                     # plan detection -> None = Open

    ctx.dd_bench_level.value = bench.LEVEL_TARGETED
    assert ctx.btn_bench_run.disabled is True, 'L2 under Open needs the ack'
    assert ctx.chk_bench_ack.layout.display == ''
    ctx.chk_bench_ack.value = True
    assert ctx.btn_bench_run.disabled is False

    ctx.chk_bench_ack.value = False
    ctx.dd_bench_level.value = bench.LEVEL_FULL
    assert ctx.btn_bench_run.disabled is True
    ctx.chk_bench_ack.value = True
    assert ctx.btn_bench_run.disabled is False


def test_run_refuses_when_acknowledgement_is_missing(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    called = []
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=lambda p: called.append(p) or _fake_runner(p))
    api = _setup(ctx)
    ctx.dd_bench_level.__dict__['value'] = bench.LEVEL_FULL   # no ack ticked

    api['run']()
    assert not called, 'the runner must not be reached without the ack'
    assert 'understand this uses QPU time' in ctx.lbl_bench_status.value


# ── 8b. a successful run ────────────────────────────────────────────

def test_successful_run_applies_measured_epoch_and_reranks(monkeypatch,
                                                           tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    reranked = []
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_fake_runner,
               rescore=lambda: reranked.append(1) or 4)
    api = _setup(ctx)

    api['run']()

    assert 'measured' in state.calibration_data_label
    assert 'L1' in state.calibration_data_label
    assert getattr(state, '_measured_epoch', None) is not None
    # readout was re-measured on the candidate qubits only
    epoch = state._measured_epoch
    assert list(np.where(epoch.measured_qubit_mask)[0]) == [0, 1, 2, 3]
    assert state.qubit_props_data['readout_error'][0] == pytest.approx(0.06)
    assert state.qubit_props_data['readout_error'][7] == 0.05  # unmeasured
    assert reranked, 'candidates must be re-ranked under the measured epoch'
    assert '✅' in ctx.lbl_bench_status.value
    assert '4 candidate(s) re-ranked' in ctx.lbl_bench_status.value
    # and it was persisted
    assert bench.list_measured_epochs('ibm_test')


def test_fit_failures_surface_in_the_notices_bar(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    def _bad_runner(plan):
        return {i: [] for i, _s in enumerate(plan.specs)}   # nothing usable

    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_bad_runner)
    api = _setup(ctx)
    api['run']()

    assert ctx.lbl_lucj_notices.value, 'fit failures must be visible'
    assert '⚠' in ctx.lbl_lucj_notices.value


def test_runner_failure_leaves_the_reported_epoch_untouched(monkeypatch,
                                                            tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    before_label = state.calibration_data_label
    before_readout = state.qubit_props_data['readout_error'].copy()

    def _boom(_plan):
        raise RuntimeError('device unavailable')

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_boom)
    api = _setup(ctx)
    api['run']()

    assert '❌' in ctx.lbl_bench_status.value
    assert 'device unavailable' in ctx.lbl_bench_status.value
    assert state.calibration_data_label == before_label
    np.testing.assert_array_equal(state.qubit_props_data['readout_error'],
                                  before_readout)
    assert getattr(state, '_measured_epoch', None) is None
    assert ctx.btn_bench_run.disabled is False, 'button must be re-enabled'


def test_missing_extra_disables_the_button_with_an_install_hint(monkeypatch,
                                                                tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch, present=False)
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], target=bench.TARGET_SWEEP)
    _setup(ctx)
    assert ctx.btn_bench_run.disabled is True
    assert 'i-qmapper[benchmark]' in ctx.lbl_bench_summary.value


# ── 8c. reuse ───────────────────────────────────────────────────────

def test_reuse_applies_a_saved_epoch_without_running(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    ran = []
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=lambda p: ran.append(p) or _fake_runner(p))
    api = _setup(ctx)

    api['run']()                                    # produce a saved epoch
    assert len(ran) == 1
    state.calibration_data_label = 'ibm_test — reported'   # pretend a reset
    api['refresh_summary']()
    assert 'Reuse (measured' in ctx.lbl_bench_reuse.value
    assert ctx.btn_bench_reuse.disabled is False

    api['reuse']()
    assert len(ran) == 1, 'reuse must not spend QPU time'
    assert 'measured' in state.calibration_data_label


# ── 9. the hard rule: scanning never benchmarks ─────────────────────

def test_scan_never_invokes_the_benchmark_engine(monkeypatch):
    """A scan sweeps dozens of backends; an implicit benchmark per backend
    would be catastrophic. Nothing on the scan path may touch `benchmark`."""
    import iqmapper.benchmark as bench_mod

    spied = []
    for name in ('run_plan', 'make_runtime_runner', 'apply_measured_epoch',
                 'build_plan', 'select_targets'):
        original = getattr(bench_mod, name)

        def _spy(*a, _n=name, _o=original, **k):
            spied.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(bench_mod, name, _spy)

    # Drive the real scan with its network layer faked out, exactly as
    # tests/test_scan_ui.py does.
    import numpy as _np
    monkeypatch.setattr(
        "iqmapper.data.generate_device_data",
        lambda s, v, c, d, prefetched_props=None: (
            list(range(8)), [0.0] * 8, _line_edges(8),
            [str(i) for i in range(8)],
            _np.array([0.01 * (i + 1) for i in range(8)]),
            _np.array([0.001 * (i + 1) for i in range(7)])))
    monkeypatch.setattr("iqmapper.qpu_scan.list_scan_targets",
                        lambda *a, **k: [])
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: None))

    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    st = SessionState()
    for key, val in dict(
            active_backend=object(), active_backend_name='ibm_here',
            active_chip='c', active_service=None, available_backends=[],
            props_warning=None, initial_props=object(), active_ansatz='lucj',
            active_vendor='IBM', active_coords=[], custom_colors=[],
            cal_history_dates=['_seed_']).items():
        setattr(st, key, val)
    builder = MapperUIBuilder(st)
    builder.build()
    spied.clear()                    # the panel plans once at build, legitimately

    builder.btn_scan_qpus.click()

    assert spied == [], f'the scan path reached benchmark: {spied}'


def test_qpu2_connect_never_invokes_the_benchmark_engine(monkeypatch):
    """Connecting a second QPU must not benchmark it either.

    The inter-QPU handoff is one click that builds a whole secondary UI; if
    benchmarking rode along with it, every scan pick would silently spend QPU
    time on a device the user only wanted to look at.
    """
    import iqmapper.benchmark as bench_mod

    spied = []
    for name in ('run_plan', 'make_runtime_runner', 'apply_measured_epoch',
                 'results_to_epoch', 'save_measured_epoch'):
        original = getattr(bench_mod, name)

        def _spy(*a, _n=name, _o=original, **k):
            spied.append(_n)
            return _o(*a, **k)

        monkeypatch.setattr(bench_mod, name, _spy)

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

    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("iqmapper.constants.resolve_coords",
                        lambda b: ([], 'chip_label'))
    monkeypatch.setattr("iqmapper.ui.build_mapper_ui", _fake_build)
    monkeypatch.setattr("iqmapper.plot.refresh_plot", lambda f, s, w: None)

    from iqmapper.ui.inter_qpu import setup_inter_qpu
    state = SimpleNamespace(
        available_backends=['ibm_k'], active_backend_name='ibm_here',
        active_service=SimpleNamespace(
            backend=lambda n: SimpleNamespace(name=n)),
        active_ansatz='lucj', current_colorscale='Viridis')
    handles_cell = [None]
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
        full_ui=_Stub(children=[]), handoff_cell=[None],
        qpu2_handles_cell=handles_cell,
        spawn_bg=lambda fn, on_done=None, on_error=None: on_done(fn()))
    setup_inter_qpu(ctx)
    spied.clear()

    ctx.handoff_cell[0]('ibm_k', {'alpha': [0], 'beta': [1], 'ancillas': [2]})

    assert 'Connected' in ctx.lbl_qpu2_status.value, 'the connect must work'
    assert spied == [], f'the QPU-2 connect path reached benchmark: {spied}'
    # ...and the accessor is published for the panel to use LATER, on demand.
    assert handles_cell[0] is not None
    assert handles_cell[0]().backend_name == 'ibm_k'


# ── v2: the QPU selector ────────────────────────────────────────────

def _qpu2_handles(n_q=6):
    """A stand-in for ui.inter_qpu's read-only QPU-2 accessor."""
    state = _state(n_q)
    state.active_backend_name = 'ibm_second'
    state.calibration_data_label = 'ibm_second — reported'
    state.layouts_data = {0: {'alpha': ['0', '1'], 'beta': ['2'],
                              'ancilla': [], 'other': []}}
    return SimpleNamespace(state=state, fig=None, widgets=None,
                           backend_name='ibm_second', connected=True)


def test_qpu_toggle_hidden_until_inter_qpu_is_on_and_qpu2_connected(
        monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    # inter-QPU off, no QPU 2
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])])
    api = _setup(ctx)
    assert ctx.tb_bench_qpu.layout.display == 'none'
    assert api['qpu2_available'] is not None and not api['qpu2_available']()

    # inter-QPU on but QPU 2 not connected
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], inter_on=True)
    _setup(ctx)
    assert ctx.tb_bench_qpu.layout.display == 'none'

    # both
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], inter_on=True,
               qpu2=_qpu2_handles())
    _setup(ctx)
    assert ctx.tb_bench_qpu.layout.display == ''


def test_qpu2_selection_applies_the_epoch_to_the_secondary_state_only(
        monkeypatch, tmp_path):
    """The whole point of the selector: nothing may leak into QPU 1."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    primary = _state()
    primary_label = primary.calibration_data_label
    primary_readout = primary.qubit_props_data['readout_error'].copy()
    handles = _qpu2_handles()

    reranked = []
    ctx = _ctx(state=primary, candidates=[_entry([0, 1], [2], [3])],
               inter_on=True, qpu2=handles, runner=_fake_runner,
               rescore=lambda: reranked.append(1) or 4)
    api = _setup(ctx)

    ctx.tb_bench_qpu.value = 2                       # act on QPU 2
    assert api['active_target']()[0] is handles.state
    api['run']()

    # QPU 2 got the epoch...
    assert 'measured' in handles.state.calibration_data_label
    assert getattr(handles.state, '_measured_epoch', None) is not None
    # ...and QPU 1 is untouched, label and arrays both.
    assert primary.calibration_data_label == primary_label
    assert getattr(primary, '_measured_epoch', None) is None
    np.testing.assert_array_equal(
        primary.qubit_props_data['readout_error'], primary_readout)
    # The Auto candidate list belongs to QPU 1 — not re-ranked on a QPU-2 run.
    assert not reranked, 'QPU-2 epoch must not re-rank QPU-1 candidates'
    # Saved under QPU 2's own backend name.
    assert bench.list_measured_epochs('ibm_second')
    assert not bench.list_measured_epochs('ibm_test')


def test_qpu2_targets_come_from_the_secondary_layout(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    handles = _qpu2_handles()
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])], inter_on=True,
               qpu2=handles)
    api = _setup(ctx)

    ctx.tb_bench_qpu.value = 2
    ctx.dd_bench_target.value = bench.TARGET_CURRENT
    targets, _plan = api['current_plan']()
    assert targets.qubits == [0, 1, 2], 'must read QPU 2 layouts_data'


# ── v2: whole-QPU messaging ─────────────────────────────────────────

def test_whole_qpu_level1_is_advertised_as_the_cheap_refresh(monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(target=bench.TARGET_WHOLE)
    _setup(ctx)
    assert 'cheap heatmap refresh' in ctx.lbl_bench_summary.value
    assert ctx.btn_bench_run.disabled is False


def test_whole_qpu_level3_always_demands_the_acknowledgement(monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(target=bench.TARGET_WHOLE)
    _setup(ctx)
    ctx.dd_bench_level.value = bench.LEVEL_FULL
    assert '4 min of QPU' in ctx.lbl_bench_summary.value
    assert ctx.btn_bench_run.disabled is True
    ctx.chk_bench_ack.value = True
    assert ctx.btn_bench_run.disabled is False


# ── v2.1: the quantum stack is imported on the MAIN thread ──────────

def test_runtime_is_preloaded_on_the_main_thread_before_the_worker(
        monkeypatch, tmp_path):
    """The import must not happen on the worker.

    Live regression: `make_runtime_runner` resolved qiskit_experiments — and
    transitively pandas — inside the background worker. Python publishes a
    module into sys.modules before executing it, so plotly's click handler,
    which probes sys.modules['pandas'] on the main thread, found a half-built
    module and raised "partially initialized module 'pandas'". Clicking the
    map during a benchmark was enough.
    """
    import threading

    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    bench._reset_runtime_cache()

    main_thread = threading.current_thread()
    resolved_on = []
    order = []

    def _recording_api():
        resolved_on.append(threading.current_thread())
        order.append('import')
        return _runtime_fakes()

    monkeypatch.setattr(bench, '_runtime_api', _recording_api)

    worker_thread = []

    def _threaded_spawn(fn, on_done=None, on_error=None):
        """A REAL background thread, as default_spawn_bg uses."""
        result = {}

        def _work():
            worker_thread.append(threading.current_thread())
            order.append('worker')
            try:
                result['value'] = fn()
            except Exception as exc:                  # pragma: no cover
                result['error'] = exc

        t = threading.Thread(target=_work)
        t.start()
        t.join()
        if 'error' in result:
            if on_error:
                on_error(result['error'])
        elif on_done:
            on_done(result['value'])

    # runner_factory=None -> the PRODUCTION path, which is the one that broke.
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, spawn=_threaded_spawn)
    api = _setup(ctx)
    try:
        api['run']()

        assert resolved_on, 'the runtime was never resolved'
        assert len(resolved_on) == 1, 'resolved more than once'
        assert resolved_on[0] is main_thread, (
            f'the quantum stack was imported on {resolved_on[0].name}, not '
            f'the main thread — this is the pandas race')
        assert worker_thread and worker_thread[0] is not main_thread, (
            'the test did not actually use a background thread')
        assert order == ['import', 'worker'], (
            f'the import must complete before the worker starts, got {order}')
        assert bench.runtime_ready() is True
    finally:
        bench._reset_runtime_cache()


def test_second_run_skips_the_loading_message(monkeypatch, tmp_path):
    """Preload is idempotent, so only the first run shows the banner."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    bench._reset_runtime_cache()
    monkeypatch.setattr(bench, '_runtime_api', _runtime_fakes)

    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP)
    api = _setup(ctx)
    try:
        api['run']()
        first = ' '.join(ctx.lbl_bench_status.history)
        assert 'Loading benchmark libraries' in first

        mark = len(ctx.lbl_bench_status.history)
        api['run']()
        second = ' '.join(ctx.lbl_bench_status.history[mark:])
        assert 'Loading benchmark libraries' not in second, (
            'the loading banner must only appear on the first use')
    finally:
        bench._reset_runtime_cache()


def test_missing_extra_at_run_time_is_reported_not_raised(monkeypatch,
                                                           tmp_path):
    """A preload failure must surface as a status, and re-enable the button."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    bench._reset_runtime_cache()

    def _boom():
        raise ImportError(bench._INSTALL_HINT)

    monkeypatch.setattr(bench, '_runtime_api', _boom)
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP)
    api = _setup(ctx)
    try:
        api['run']()
        assert '❌' in ctx.lbl_bench_status.value
        assert 'i-qmapper[benchmark]' in ctx.lbl_bench_status.value
        assert ctx.btn_bench_run.disabled is False
    finally:
        bench._reset_runtime_cache()


def _runtime_fakes():
    """Minimal stand-ins for the quantum stack, enough to complete a run."""
    class _Session:
        def __init__(self, **kw):
            self.kw = kw

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _Sampler:
        def __init__(self, **kw):
            self.kw = kw

    class _Exp:
        def __init__(self, *a, **k):
            pass

    class _Data:
        def __init__(self, n):
            self._n = n

        def block_for_results(self):
            pass

        def child_data(self, idx):
            return SimpleNamespace(analysis_results=lambda: [])

    class _Batch:
        def __init__(self, experiments, **kw):
            self._n = len(experiments)

        def run(self, **kw):
            return _Data(self._n)

    return SimpleNamespace(
        Session=_Session, SamplerV2=_Sampler, BatchExperiment=_Batch,
        ParallelExperiment=_Exp, LocalReadoutError=_Exp, StandardRB=_Exp,
        T1=_Exp, T2Hahn=_Exp)


# ── v3: no false success ────────────────────────────────────────────

def test_empty_epoch_is_not_applied_and_not_called_measured(monkeypatch,
                                                             tmp_path):
    """The exact 18 Sept failure: the run produced nothing and said ✅.

    IBM rejected the job, the parser found no analysis results, and the panel
    applied an epoch identical to the reported one labelled 'measured'.
    """
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    before_label = state.calibration_data_label
    before_readout = state.qubit_props_data['readout_error'].copy()

    def _nothing_runner(plan):
        return {i: [] for i, _s in enumerate(plan.specs)}

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_nothing_runner)
    api = _setup(ctx)
    api['run']()

    status = ctx.lbl_bench_status.value
    assert '✅' not in status, f'a run that measured nothing reported success: {status}'
    assert 'no usable measurements' in status
    # The reported epoch is untouched, and nothing claims to be measured.
    assert state.calibration_data_label == before_label
    assert 'measured' not in state.calibration_data_label
    assert getattr(state, '_measured_epoch', None) is None
    np.testing.assert_array_equal(
        state.qubit_props_data['readout_error'], before_readout)
    # ...and it is not offered for reuse.
    assert not bench.list_measured_epochs('ibm_test')
    # ...but the failure IS visible.
    assert ctx.lbl_lucj_notices.value


def test_partial_measurement_is_applied_but_labelled_partial(monkeypatch,
                                                              tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()

    def _half_runner(plan):
        out = {}
        for idx, spec in enumerate(plan.specs):
            if spec.kind == 'readout':
                # Only the first target qubit comes back.
                q = spec.qubits[0]
                out[idx] = [SimpleNamespace(
                    qubits=[q], device_components=[q],
                    value=SimpleNamespace(assignment_matrix=np.array(
                        [[0.99, 0.01], [0.01, 0.99]])))]
            else:
                out[idx] = []
        return out

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_half_runner)
    api = _setup(ctx)
    api['run']()

    status = ctx.lbl_bench_status.value
    assert '⚠' in status and 'partial' in status, status
    assert '1/4 qubits' in status, status
    assert 'measured' in state.calibration_data_label, 'a partial run applies'
    assert getattr(state, '_measured_epoch', None) is not None


def test_complete_measurement_reports_real_coverage(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_fake_runner)
    api = _setup(ctx)
    api['run']()

    status = ctx.lbl_bench_status.value
    assert '✅' in status, status
    assert '4/4 qubits' in status, status


# ── v3: a failed job says so ────────────────────────────────────────

def test_job_failure_surfaces_ibms_message_and_job_ids(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    before = state.calibration_data_label

    def _rejecting_runner(_plan):
        raise bench.BenchmarkRunError(
            'chunk 1: job error — The number of circuit shots 43433984 '
            'exceeds the system limit 10000000',
            job_ids=['d3adb33f'])

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_rejecting_runner)
    api = _setup(ctx)
    api['run']()

    status = ctx.lbl_bench_status.value
    assert '❌' in status
    assert 'exceeds the system limit' in status, status
    assert 'd3adb33f' in status, 'the job id must be reported for diagnosis'
    assert state.calibration_data_label == before
    assert getattr(state, '_measured_epoch', None) is None
    assert ctx.btn_bench_run.disabled is False


# ── v3: submittability gating ───────────────────────────────────────

def test_run_is_disabled_when_the_plan_cannot_fit_a_job(monkeypatch,
                                                         tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    monkeypatch.setattr(bench, 'plan_is_submittable',
                        lambda plan, shots=1000: (False, 'too many shots'))
    ctx = _ctx(candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP)
    _setup(ctx)
    assert ctx.btn_bench_run.disabled is True
    assert 'too many shots' in ctx.lbl_bench_summary.value


def test_summary_states_circuits_shots_and_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(target=bench.TARGET_WHOLE)
    _setup(ctx)
    summary = ctx.lbl_bench_summary.value
    assert 'circuits' in summary and 'shots' in summary and 'job' in summary


# ── v3.1: consistency check and relative mode in the panel ─────────

def _offset_runner(factor):
    """A runner whose readout comes back `factor` times the reported value."""
    def _run(plan):
        out = {}
        for idx, spec in enumerate(plan.specs):
            if spec.kind == 'readout':
                out[idx] = [SimpleNamespace(
                    name='Local Readout Mitigator',
                    device_components=[f'Q{q}' for q in spec.qubits],
                    value=_MitigatorStub({q: 0.05 * factor
                                          for q in spec.qubits}))]
            else:
                out[idx] = []
        return out
    return _run


def test_a_systematic_offset_triggers_relative_mode(monkeypatch, tmp_path):
    """The 20 Sept signature: everything uniformly worse than reported."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_offset_runner(3.0))
    api = _setup(ctx)
    api['run']()

    consistency = ctx.lbl_bench_consistency.value
    assert 'systematic offset' in consistency, consistency
    assert 'RELATIVE mode' in consistency
    assert 'readout_error' in consistency
    # Applied, and the label says which mode.
    assert 'measured' in state.calibration_data_label
    assert 'relative' in state.calibration_data_label
    # Rescaled back onto the reported median.
    epoch = state._measured_epoch
    mask = epoch.measured_qubit_mask
    assert np.median(epoch.qubit_props['readout_error'][mask]) == \
        pytest.approx(0.05, rel=1e-6)


def test_absolute_checkbox_opts_into_the_raw_values(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_offset_runner(3.0))
    ctx.chk_bench_absolute = _Dropdown(value=True)
    api = _setup(ctx)
    api['run']()

    epoch = state._measured_epoch
    mask = epoch.measured_qubit_mask
    assert np.median(epoch.qubit_props['readout_error'][mask]) == \
        pytest.approx(0.15, rel=1e-6), 'absolute mode must not rescale'
    assert 'relative' not in state.calibration_data_label


def test_an_in_band_run_stays_absolute(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()
    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_offset_runner(1.2))
    api = _setup(ctx)
    api['run']()

    consistency = ctx.lbl_bench_consistency.value
    assert 'systematic offset' not in consistency, consistency
    assert 'readout_error' in consistency
    assert 'relative' not in state.calibration_data_label
