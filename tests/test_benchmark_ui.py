"""UI tests for the 📡 Benchmark section.

Everything runs against a FAKE runner injected through the panel ctx, so no
test ever imports qiskit_experiments or reaches a network. The last test is
the important one: it pins the rule that the multi-QPU scan never benchmarks.
"""
import json
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
        lbl_bench_details=_RecLabel(),
        acc_bench_details=_Stub(selected_index=None),
        btn_bench_show_on_map=_Stub(description='show on map'),
        chk_err_threshold=_Dropdown(value=False),
        txt_g_thr_max=_Dropdown(value=''),
        dd_gate_prop=_Dropdown(value='cz_error', options=[('CZ', 'cz_error')]),
        lbl_bench_reuse=_RecLabel(),
        btn_bench_reuse=_Stub(disabled=True),
        btn_bench_reanalyze=_Stub(disabled=True),
        btn_bench_calcheck=_Stub(disabled=True),
        lbl_bench_calcheck=_RecLabel(),
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
    assert 'Nothing usable was measured' in status
    assert "IBM's reported calibration is still in use" in status
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
    assert 'Benchmark applied' in consistency, consistency
    assert 'constant factor' in consistency
    assert 'Readout error' in consistency
    for jargon in ('estimator mismatch', 'RELATIVE', 'systematic offset'):
        assert jargon not in consistency, consistency
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
    assert "in line with IBM's reported calibration" in consistency, \
        consistency
    assert 'constant factor' not in consistency
    assert 'relative' not in state.calibration_data_label


def test_analysis_callback_failures_are_reported_as_partial(monkeypatch,
                                                             tmp_path):
    """A DONE job whose fits all raised must not read as a clean success.

    On 20 Sept 13:08 both phases reported DONE, `errors()` was None, and the
    panel showed six bland "no results returned" notices with no hint that
    every two-qubit callback had thrown.
    """
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state()

    def _runner_with_analysis_failure(plan):
        out = {i: [] for i, _s in enumerate(plan.specs)}
        for idx, spec in enumerate(plan.specs):
            if spec.kind == 'readout':
                out[idx] = [SimpleNamespace(
                    name='Local Readout Mitigator',
                    device_components=[f'Q{q}' for q in spec.qubits],
                    value=_MitigatorStub({q: 0.06 for q in spec.qubits}))]
        out['_notices'] = [
            "phase 2: analysis failed: 6x AttributeError: 'float' object "
            "has no attribute 'device_components'"]
        return out

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP,
               runner=_runner_with_analysis_failure)
    api = _setup(ctx)
    api['run']()

    status = ctx.lbl_bench_status.value
    assert '✅' not in status, f'analysis failure reported as success: {status}'
    assert 'partial' in status
    assert 'analysis failed' in status, status
    assert 'device_components' in status
    assert 'analysis failed' in ctx.lbl_lucj_notices.value


# ── v3.2: reanalysis must replay the ORIGINAL plan ─────────────────

def _record_for(plan, when=1_700_000_000.0, jobs=None):
    """The _jobs.json payload a submission of `plan` would have written."""
    return {
        'when': when, 'level': plan.level, 'shots': plan.shots,
        'targets': {'qubits': list(plan.targets.qubits),
                    'edges': [list(e) for e in plan.targets.edges]},
        'specs': [{'kind': s.kind, 'qubits': list(s.qubits), 'layer': s.layer,
                   'params': {k: ([list(p) for p in v] if k == 'pairs'
                                  else v)
                              for k, v in s.params.items()}}
                  for s in plan.specs],
        'jobs': jobs or {'0': ['job-a'], '1': ['job-b']},
    }


def _readout_results(qubits, error=0.06):
    return [SimpleNamespace(
        name='Local Readout Mitigator',
        device_components=[f'Q{q}' for q in qubits],
        value=_MitigatorStub({q: error for q in qubits}))]


@pytest.mark.parametrize('selector_mode', [
    bench.TARGET_CURRENT, bench.TARGET_SWEEP, bench.TARGET_WHOLE,
    bench.TARGET_CUSTOM,
])
def test_reanalysis_ignores_the_selector_and_replays_the_run(
        monkeypatch, tmp_path, selector_mode):
    """Whatever the selector says, reanalysis must reproduce the run.

    Live bug: with the selector on "Current layout(s)" and nothing placed,
    the rebuilt plan had ZERO specs, so the panel returned (0, 0) with an
    empty notice list while the raw dump held every result.
    """
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(8)
    original = bench.build_plan(bench.LEVEL_QUICK_READOUT,
                                bench.Targets([0, 1, 2, 3], []), 8, coupling)
    record = _record_for(original)
    monkeypatch.setattr(bench, 'latest_job_record', lambda name: record)

    seen = {}

    def _fake_reanalyze(plan, job_ids, backend, service=None):
        seen['plan'] = plan
        raw = {}
        for idx, spec in enumerate(plan.specs):
            raw[idx] = (_readout_results(spec.qubits)
                        if spec.kind == 'readout' else [])
        return raw, []

    monkeypatch.setattr(bench, 'reanalyze_from_jobs', _fake_reanalyze)

    state = _state()
    # A runner_factory marks this as an injected-stack test, so the panel
    # skips the main-thread preload of the real quantum stack.
    ctx = _ctx(state=state, candidates=[], target=selector_mode,
               runner=lambda plan: {})
    api = _setup(ctx)
    api['reanalyze']()

    # The plan replayed is the recorded one, not the selector's.
    assert seen['plan'].specs, 'reanalysis rebuilt an empty plan'
    assert [s.kind for s in seen['plan'].specs] == \
        [s.kind for s in original.specs]
    assert seen['plan'].targets.qubits == [0, 1, 2, 3]

    epoch = getattr(state, '_measured_epoch', None)
    assert epoch is not None, ctx.lbl_bench_status.value
    assert epoch.measured_count == (4, 0), epoch.notices
    assert '✅' in ctx.lbl_bench_status.value


def test_reanalysis_without_a_plan_in_the_record_refuses_clearly(monkeypatch,
                                                                  tmp_path):
    """Old records predate the spec list; refuse rather than guess."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    monkeypatch.setattr(bench, 'latest_job_record',
                        lambda name: {'jobs': {'0': ['j']}, 'specs': []})
    called = []
    monkeypatch.setattr(bench, 'reanalyze_from_jobs',
                        lambda *a, **k: called.append(1) or ({}, []))

    ctx = _ctx(candidates=[], runner=lambda plan: {})
    api = _setup(ctx)
    api['reanalyze']()

    assert not called, 'must not reanalyse without the original plan'
    status = ctx.lbl_bench_status.value
    assert '❌' in status
    assert 'cannot be rebuilt' in status, status


def test_results_found_but_none_attributed_is_never_silent(monkeypatch,
                                                            tmp_path):
    """The mandatory invariant, on a deliberate target mismatch."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(8)
    plan = bench.build_plan(bench.LEVEL_QUICK_READOUT,
                            bench.Targets([0, 1], []), 8, coupling)
    qp = {'readout_error': np.full(8, 0.05)}
    gp = {'cz_error': np.full(7, 0.01)}

    # Results for qubits that are NOT in the plan's targets and not in range.
    raw = {0: _readout_results([90, 91])}
    epoch = bench.results_to_epoch(raw, plan, qp, gp, coupling)

    assert epoch.measured_count == (0, 0)
    invariant = [n for n in epoch.notices if '0 attributed to targets' in n]
    assert invariant, f'the invariant notice is missing: {epoch.notices}'
    assert '1 results found' in invariant[0], invariant
    assert 'edge orientation' in invariant[0] or 'matched' in invariant[0]


def test_an_empty_plan_is_refused_by_reanalysis():
    empty = bench.build_plan(bench.LEVEL_TARGETED, bench.Targets([], []), 8,
                             _line_edges(8))
    assert empty.specs == []
    with pytest.raises(bench.BenchmarkRunError, match='no experiments'):
        bench.reanalyze_from_jobs(empty, ['j'], object(), None)


def test_plan_from_record_round_trips():
    coupling = _line_edges(8)
    original = bench.build_plan(bench.LEVEL_TARGETED,
                                bench.Targets([0, 1, 2, 3], coupling[:3]), 8,
                                coupling)
    rebuilt = bench.plan_from_record(_record_for(original))

    assert rebuilt is not None
    assert rebuilt.level == original.level
    assert rebuilt.shots == original.shots
    assert [s.kind for s in rebuilt.specs] == [s.kind for s in original.specs]
    assert bench.plan_circuit_count(rebuilt) == \
        bench.plan_circuit_count(original)
    for a, b in zip(rebuilt.specs, original.specs):
        assert a.qubits == b.qubits
        assert a.params.get('pairs') == b.params.get('pairs')


# ── v3.3: the reanalysis must stay phase-aware and self-check ──────

def test_reanalysis_passes_job_ids_grouped_by_phase(monkeypatch, tmp_path):
    """Flattening the phases lost the 1Q subtraction silently.

    `reanalyze_from_jobs` only runs 1Q RB before the 2Q layers when it is
    given a list of lists. The panel flattened `record['jobs'].values()`, so
    both phases went into one batch, the 2Q analyses were built before any
    EPG existed, and the epoch carried un-subtracted values (median 4.05e-3
    instead of 2.96e-3) with nothing to indicate it.
    """
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(8)
    original = bench.build_plan(bench.LEVEL_TARGETED,
                                bench.Targets([0, 1, 2, 3], coupling[:3]), 8,
                                coupling)
    record = _record_for(original,
                         jobs={'0': ['phase1-job'], '1': ['phase2-job']})
    monkeypatch.setattr(bench, 'latest_job_record', lambda name: record)

    seen = {}

    def _fake_reanalyze(plan, job_ids, backend, service=None):
        seen['job_ids'] = job_ids
        raw = {i: (_readout_results(s.qubits) if s.kind == 'readout' else [])
               for i, s in enumerate(plan.specs)}
        return raw, []

    monkeypatch.setattr(bench, 'reanalyze_from_jobs', _fake_reanalyze)

    ctx = _ctx(candidates=[], runner=lambda plan: {})
    _setup(ctx)['reanalyze']()

    passed = seen['job_ids']
    assert isinstance(passed[0], list), (
        f'phases were flattened into a single list: {passed}')
    assert passed == [['phase1-job'], ['phase2-job']], passed


def test_reanalysis_runs_the_consistency_check_and_relative_mode(monkeypatch,
                                                                  tmp_path):
    """The reanalysis path never stashed the reported props, so the check
    returned early: no consistency line, and no rescaling despite a x3
    offset."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(8)
    original = bench.build_plan(bench.LEVEL_QUICK_READOUT,
                                bench.Targets([0, 1, 2, 3], []), 8, coupling)
    monkeypatch.setattr(bench, 'latest_job_record',
                        lambda name: _record_for(original))

    def _fake_reanalyze(plan, job_ids, backend, service=None):
        raw = {}
        for idx, spec in enumerate(plan.specs):
            raw[idx] = (_readout_results(spec.qubits, error=0.15)
                        if spec.kind == 'readout' else [])
        return raw, []

    monkeypatch.setattr(bench, 'reanalyze_from_jobs', _fake_reanalyze)

    state = _state()                      # reported readout_error == 0.05
    ctx = _ctx(state=state, candidates=[], runner=lambda plan: {})
    _setup(ctx)['reanalyze']()

    consistency = ctx.lbl_bench_consistency.value
    assert consistency, 'no consistency line was rendered after reanalysis'
    assert 'Benchmark applied' in consistency, consistency
    assert 'constant factor' in consistency

    epoch = state._measured_epoch
    assert epoch.relative is True
    mask = epoch.measured_qubit_mask
    assert np.median(epoch.qubit_props['readout_error'][mask]) == \
        pytest.approx(0.05, rel=1e-6), 'values were not rescaled'
    assert 'relative' in state.calibration_data_label


# ── v3.4: the UI path must equal the direct path ───────────────────

class _SpyAnalysis:
    """Records what the 2Q analysis was given."""

    def __init__(self, log):
        self._log = log
        self.options = {}

    def set_options(self, **kw):
        self.options.update(kw)
        if 'epg_1_qubit' in kw:
            self._log.append(list(kw['epg_1_qubit']))


def _spy_api(set_options_log, order_log):
    """A runtime API whose 2Q analyses record their epg_1_qubit input."""
    class _Exp:
        def __init__(self, *a, **k):
            self.analysis = _SpyAnalysis(set_options_log)

    class _RB1Q(_Exp):
        pass

    class _Data:
        job_ids = ['j']

        def __init__(self, kinds):
            self._kinds = kinds

        def status(self):
            return SimpleNamespace(name='DONE')

        def errors(self):
            return None

        def analysis_errors(self):
            return None

        def block_for_results(self):
            pass

        def child_data(self, idx=None):
            kind = self._kinds[idx if idx is not None else 0]
            results = []
            if kind == 'rb1q':
                results = [SimpleNamespace(name='EPG_sx',
                                           device_components=[f'Q{q}'],
                                           value=7e-4) for q in (0, 1)]
            return SimpleNamespace(
                analysis_results=lambda *a, **k: results,
                child_data=lambda *a, **k: [])

    class _Batch:
        def __init__(self, experiments, **kw):
            self._n = len(experiments)

        def run(self, **kw):
            return _Data(['rb1q'] * self._n)

    class _Session:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    return SimpleNamespace(
        Session=_Session, SamplerV2=lambda mode=None: None,
        BatchExperiment=_Batch, ParallelExperiment=_Exp,
        LocalReadoutError=_Exp, StandardRB=_RB1Q, T1=_Exp, T2Hahn=_Exp)


def test_phase_one_is_analysed_before_phase_two(monkeypatch):
    """Order is the whole mechanism: the 2Q analyses are CONSTRUCTED with the
    phase-1 EPGs, so phase 1 must be submitted and folded first."""
    coupling = _line_edges(8)
    plan = bench.build_plan(bench.LEVEL_TARGETED,
                            bench.Targets([0, 1], [(0, 1)]), 8, coupling)
    order = []
    set_log = []
    api = _spy_api(set_log, order)

    class _Batch(api.BatchExperiment):
        def __init__(self, experiments, **kw):
            order.append(len(experiments))
            super().__init__(experiments, **kw)

    api.BatchExperiment = _Batch
    monkeypatch.setattr(bench, '_runtime_api', lambda: api)
    bench._reset_runtime_cache()
    try:
        runner = bench.make_runtime_runner(object())
        runner(plan)
    finally:
        bench._reset_runtime_cache()

    assert len(order) == 2, f'expected two phases, got {order}'
    # Phase 1 carries readout + rb1q; phase 2 carries only the 2Q layers.
    n_2q = len([s for s in plan.specs if s.kind == 'rb2q_layer'])
    assert order[1] == n_2q, order
    assert set_log, 'the 2Q analysis was never given epg_1_qubit'


def test_ui_reanalysis_matches_direct_reanalysis(monkeypatch, tmp_path):
    """Same record, same jobs -> identical epoch, and the same EPG objects
    handed to set_options on the UI path."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(8)
    original = bench.build_plan(bench.LEVEL_TARGETED,
                                bench.Targets([0, 1], [(0, 1)]), 8, coupling)
    record = _record_for(original, jobs={'0': ['p1'], '1': ['p2']})

    ui_calls, direct_calls = [], []

    def _make_reanalyze(sink):
        def _rean(plan, job_ids, backend, service=None):
            sink.append({'plan_kinds': [s.kind for s in plan.specs],
                         'job_ids': job_ids})
            raw = {}
            for idx, spec in enumerate(plan.specs):
                raw[idx] = (_readout_results(spec.qubits)
                            if spec.kind == 'readout' else [])
            raw['_notices'] = ['1Q EPG subtraction: applied to 1 pairs']
            return raw, []
        return _rean

    # Direct path.
    monkeypatch.setattr(bench, 'reanalyze_from_jobs',
                        _make_reanalyze(direct_calls))
    raw_d, _ = bench.reanalyze_from_jobs(
        original, [['p1'], ['p2']], object(), None)
    qp = {'readout_error': np.full(8, 0.05)}
    gp = {'cz_error': np.full(7, 0.01)}
    direct_epoch = bench.results_to_epoch(raw_d, original, dict(qp), dict(gp),
                                          coupling, when=1.0)

    # UI path.
    monkeypatch.setattr(bench, 'latest_job_record', lambda name: record)
    monkeypatch.setattr(bench, 'reanalyze_from_jobs',
                        _make_reanalyze(ui_calls))
    state = _state()
    ctx = _ctx(state=state, candidates=[], runner=lambda plan: {})
    _setup(ctx)['reanalyze']()
    ui_epoch = state._measured_epoch

    assert ui_calls, 'the UI never reached reanalyze_from_jobs'
    assert ui_calls[0]['plan_kinds'] == direct_calls[0]['plan_kinds']
    assert ui_calls[0]['job_ids'] == direct_calls[0]['job_ids'] == \
        [['p1'], ['p2']]
    assert ui_epoch.measured_count == direct_epoch.measured_count
    np.testing.assert_allclose(ui_epoch.qubit_props['readout_error'],
                               direct_epoch.qubit_props['readout_error'])
    assert any('1Q EPG subtraction' in n for n in ui_epoch.notices), \
        ui_epoch.notices


# ── v3.6: the calibration check, wired ─────────────────────────────

def _calcheck_state(n_q=16):
    state = _state(n_q)
    state.layouts_data = {0: {'alpha': [str(q) for q in range(8)],
                              'beta': [str(q) for q in range(8, 14)],
                              'ancilla': [], 'other': []}}
    return state


def _calcheck_runner(par_factor=3.0, seq_factor=3.0):
    """Both arms measure; each arm's cz_error is `factor` x the reported."""
    calls = {'n': 0}

    def _run(plan):
        calls['n'] += 1
        factor = par_factor if calls['n'] == 1 else seq_factor
        out = {}
        for idx, spec in enumerate(plan.specs):
            if spec.kind == 'readout':
                out[idx] = _readout_results(spec.qubits)
            elif spec.kind == 'rb1q':
                out[idx] = [SimpleNamespace(name='EPG_sx',
                                            device_components=[f'Q{q}'],
                                            value=7e-4)
                            for q in spec.qubits]
            elif spec.kind == 'rb2q_layer':
                out[idx] = [
                    SimpleNamespace(name='EPG_cz',
                                    device_components=[f'Q{p[0]}', f'Q{p[1]}'],
                                    value=0.01 * factor)
                    for p in spec.params['pairs']]
            else:
                out[idx] = []
        return out

    _run.calls = calls
    return _run


def test_calibration_check_button_needs_six_edges(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    # Only 3 edges reachable -> disabled, and it says why.
    ctx = _ctx(state=_state(8), candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP)
    _setup(ctx)
    assert ctx.btn_bench_calcheck.disabled is True
    assert 'at least 6 target edges' in ctx.lbl_bench_calcheck.value

    # Whole QPU on a 16-qubit line -> 15 edges -> enabled, with a cost line.
    ctx = _ctx(state=_calcheck_state(), candidates=[],
               target=bench.TARGET_WHOLE)
    _setup(ctx)
    line = ctx.lbl_bench_calcheck.value
    assert '6 edges × 2 arms' in line, line
    assert 'circuits' in line and 'job(s)' in line and 'QPU (estimate)' in line
    # It spends QPU time, so the Open-plan acknowledgement gates it too.
    assert ctx.btn_bench_calcheck.disabled is True
    ctx.chk_bench_ack.value = True
    assert ctx.btn_bench_calcheck.disabled is False


def test_calibration_check_renders_the_verdict_table(monkeypatch, tmp_path):
    """Both arms run, the table renders, and NO epoch is applied."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _calcheck_state()
    before_label = state.calibration_data_label
    before_gate = state.gate_props_data['cz_error'].copy()

    runner = _calcheck_runner(par_factor=3.0, seq_factor=3.0)
    ctx = _ctx(state=state, candidates=[], target=bench.TARGET_WHOLE,
               runner=runner)
    api = _setup(ctx)
    ctx.chk_bench_ack.value = True            # Open plan: acknowledge first
    api['calibration_check']()

    assert runner.calls['n'] == 2, 'both arms must run'
    table_html = ctx.lbl_bench_calcheck.value
    assert 'Calibration check' in table_html
    assert 'parallel' in table_html and 'sequential' in table_html
    # Both arms offset the same way -> estimator, not crosstalk.
    assert 'estimator mismatch' in table_html, table_html

    # THE INVARIANT: a calibration check measures the instrument.
    assert getattr(state, '_measured_epoch', None) is None, \
        'the calibration check must not apply an epoch'
    assert state.calibration_data_label == before_label
    np.testing.assert_array_equal(state.gate_props_data['cz_error'],
                                  before_gate)
    assert 'unchanged' in ctx.lbl_bench_status.value


def test_calibration_check_distinguishes_crosstalk(monkeypatch, tmp_path):
    """Parallel offset, sequential clean -> crosstalk, not estimator bias."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(state=_calcheck_state(), candidates=[],
               target=bench.TARGET_WHOLE,
               runner=_calcheck_runner(par_factor=3.0, seq_factor=1.0))
    api = _setup(ctx)
    ctx.chk_bench_ack.value = True
    api['calibration_check']()
    assert 'crosstalk' in ctx.lbl_bench_calcheck.value, \
        ctx.lbl_bench_calcheck.value


def test_calibration_check_writes_its_record_kind(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    coupling = _line_edges(16)
    plan = bench.build_plan(bench.LEVEL_TARGETED,
                            bench.Targets(list(range(8)), coupling[:6]), 16,
                            coupling)
    when = 1_700_000_500.0
    path = bench.save_job_record(
        'ibm_test', when, plan, jobs={'0': ['a'], '1': ['b']},
        kind=bench.RECORD_KIND_CALIBRATION_CHECK,
        arms=['parallel', 'sequential'])
    assert path

    with open(path, encoding='utf-8') as fh:
        record = json.load(fh)
    assert record['kind'] == bench.RECORD_KIND_CALIBRATION_CHECK
    assert record['arms'] == ['parallel', 'sequential']

    back = bench.latest_job_record('ibm_test')
    assert back['kind'] == bench.RECORD_KIND_CALIBRATION_CHECK
    assert back['arms'] == ['parallel', 'sequential']


def test_reanalyze_routes_a_calibration_check_record(monkeypatch, tmp_path):
    """Reanalyze must reproduce the table and still apply no epoch."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)

    coupling = _line_edges(16)
    par = bench.build_plan(bench.LEVEL_TARGETED,
                           bench.Targets(list(range(8)), coupling[:6]), 16,
                           coupling)
    record = _record_for(par, jobs={'0': ['p1'], '1': ['p2']})
    record['kind'] = bench.RECORD_KIND_CALIBRATION_CHECK
    record['arms'] = ['parallel', 'sequential']
    monkeypatch.setattr(bench, 'latest_job_record', lambda name: record)

    runner = _calcheck_runner(par_factor=3.0, seq_factor=3.0)

    def _fake_rean(plan, job_ids, backend, service=None):
        return runner(plan), []

    monkeypatch.setattr(bench, 'reanalyze_from_jobs', _fake_rean)

    state = _calcheck_state()
    ctx = _ctx(state=state, candidates=[], runner=lambda plan: {})
    _setup(ctx)['reanalyze']()

    assert 'Calibration check' in ctx.lbl_bench_calcheck.value
    assert 'reanalysed (no QPU)' in ctx.lbl_bench_status.value
    assert getattr(state, '_measured_epoch', None) is None, \
        'reanalysing a calibration check must not apply an epoch'


def test_calibration_check_table_is_saved(tmp_path, monkeypatch):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    table = {'rows': [{'edge': (0, 1), 'reported': 1e-3, 'parallel': 3e-3,
                       'sequential': 3e-3}],
             'ratio_parallel': 3.0, 'ratio_sequential': 3.0,
             'verdict': 'estimator mismatch: both runs are offset'}
    path = bench.save_calibration_check('ibm_test', 1_700_000_600.0, table)
    assert path and path.endswith('_calcheck.json')
    with open(path, encoding='utf-8') as fh:
        back = json.load(fh)
    assert back['ratio_parallel'] == 3.0
    assert back['rows'][0]['edge'] == [0, 1]
    assert 'estimator' in back['verdict']


def _descend(node, seen=None):
    """Every widget reachable through .children, root included."""
    if seen is None:
        seen = []
    if any(n is node for n in seen):
        return seen
    seen.append(node)
    for child in getattr(node, 'children', ()) or ():
        _descend(child, seen)
    return seen


def _build_real_ui(monkeypatch):
    """A real MapperUIBuilder, for checks that need the actual widgets."""
    import numpy as _np
    monkeypatch.setattr(
        "iqmapper.data.generate_device_data",
        lambda s, v, c, d, prefetched_props=None: (
            list(range(8)), [0.0] * 8, [(i, i + 1) for i in range(7)],
            [str(i) for i in range(8)],
            _np.array([0.01 * (i + 1) for i in range(8)]),
            _np.array([0.001 * (i + 1) for i in range(7)])))
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
    return builder, st


# ── v3.7: the result copy a user can actually read ─────────────────

def _run_with_factors(monkeypatch, tmp_path, factors, n_q=8):
    """Drive a run whose measured/reported ratio is `factors` per property."""
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    state = _state(n_q)

    def _runner(plan):
        out = {}
        for idx, spec in enumerate(plan.specs):
            if spec.kind == 'readout':
                out[idx] = _readout_results(
                    spec.qubits, error=0.05 * factors.get('readout_error', 1))
            else:
                out[idx] = []
        return out

    ctx = _ctx(state=state, candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP, runner=_runner)
    api = _setup(ctx)
    api['run']()
    return ctx, state


def test_copy_all_in_band(monkeypatch, tmp_path):
    ctx, _ = _run_with_factors(monkeypatch, tmp_path,
                               {'readout_error': 1.2})
    line = ctx.lbl_bench_consistency.value
    assert '✅ Benchmark applied' in line
    assert "measured values are in line with IBM's reported calibration" \
        in line, line
    assert 'constant factor' not in line


def test_copy_names_in_band_and_rescaled_properties():
    """The exact parameterised sentence, on the real 20 Sept shape."""
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    ctx = _ctx()
    api = setup_benchmark_panel(ctx)

    stats = {
        'readout_error': {'ratio_of_medians': 1.55, 'n_worse_2x': 53,
                          'n': 156, 'median_reported': 3.4e-3},
        't1': {'ratio_of_medians': 0.70, 'n_worse_2x': 0, 'n': 156,
               'median_reported': 250.0},
        't2': {'ratio_of_medians': 0.24, 'n_worse_2x': 0, 'n': 156,
               'median_reported': 297.0},
        'cz_error': {'ratio_of_medians': 2.11, 'n_worse_2x': 170, 'n': 352,
                     'median_reported': 1.4e-3},
    }
    epoch = SimpleNamespace(planned_count=(156, 352),
                            measured_count=(156, 352))
    line = api['result_sentence'](epoch, stats, {'t2', 'cz_error'})

    assert 'Readout error and T1 differ from' in line, line
    assert 'real drift today' in line
    assert 'T2 and CZ error differ by a constant factor' in line, line
    assert '(×0.2, ×2.1)' in line, line
    assert "shown on IBM's scale with today's ranking preserved" in line
    for jargon in ('estimator mismatch', 'RELATIVE', 'systematic offset',
                   '⚠'):
        assert jargon not in line, f'{jargon!r} leaked into the sentence'


def test_copy_reports_unmeasured_elements():
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    api = setup_benchmark_panel(_ctx())
    stats = {'readout_error': {'ratio_of_medians': 1.1, 'n_worse_2x': 0,
                               'n': 120, 'median_reported': 3e-3}}
    epoch = SimpleNamespace(planned_count=(156, 352),
                            measured_count=(120, 340))
    line = api['result_sentence'](epoch, stats, set())
    assert '36 qubits and 12 edges could not be measured and keep their '\
           'reported values.' in line, line


def test_copy_for_an_empty_result(monkeypatch, tmp_path):
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    _have_qe(monkeypatch)
    ctx = _ctx(state=_state(), candidates=[_entry([0, 1], [2], [3])],
               target=bench.TARGET_SWEEP,
               runner=lambda plan: {i: [] for i, _ in enumerate(plan.specs)})
    _setup(ctx)['run']()
    status = ctx.lbl_bench_status.value
    assert '⚠ Nothing usable was measured' in status, status
    assert "IBM's reported calibration is still in use" in status


def test_no_developer_jargon_anywhere_after_a_run(monkeypatch, tmp_path):
    """The whole rendered panel, not just the one line."""
    ctx, _ = _run_with_factors(monkeypatch, tmp_path,
                               {'readout_error': 3.0})
    rendered = ' '.join([
        ctx.lbl_bench_consistency.value, ctx.lbl_bench_details.value,
        ctx.lbl_bench_status.value, ctx.lbl_bench_summary.value])
    for jargon in ('estimator mismatch', 'RELATIVE', 'systematic offset'):
        assert jargon not in rendered, f'{jargon!r} still shown to the user'


def test_details_table_rows_and_legend():
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    ctx = _ctx()
    api = setup_benchmark_panel(ctx)
    stats = {
        'readout_error': {'ratio_of_medians': 1.55, 'n_worse_2x': 53,
                          'n': 156, 'median_reported': 3.4e-3},
        't2': {'ratio_of_medians': 0.24, 'n_worse_2x': 0, 'n': 156,
               'median_reported': 297.0},
        'cz_error': {'ratio_of_medians': 2.11, 'n_worse_2x': 170, 'n': 352,
                     'median_reported': 1.4e-3},
    }
    api['render_details'](stats, {'t2', 'cz_error'}, {})
    html = ctx.lbl_bench_details.value

    assert 'Readout error' in html and '×1.55' in html
    assert ('53 of 156 qubits &gt;2× worse' in html
            or '53 of 156 qubits >2× worse' in html), html
    # rescaled rows carry the marker; in-band rows do not
    assert '×0.24 ⚠' in html and '×2.11 ⚠' in html, html
    assert '×1.55 ⚠' not in html, 'an in-band row was marked'
    # a rescaled property with no individual outliers shows a dash
    assert '—' in html
    assert 'rescaled (different estimator)' in html
    assert 'as measured' in html
    assert 'median aligned to IBM' in html, 'legend missing'


def test_details_accordion_is_collapsed_by_default(monkeypatch):
    b, _ = _build_real_ui(monkeypatch)
    assert b.acc_bench_details.selected_index is None
    assert b.acc_bench_details.get_title(0) == 'Details'
    inside = _descend(b.acc_bench_details)
    assert any(w is b.lbl_bench_details for w in inside)
    assert any(w is b.btn_bench_show_on_map for w in inside)


def test_show_on_map_sets_the_threshold(monkeypatch, tmp_path):
    """"170 edges" becomes something you can see."""
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    ctx = _ctx()
    api = setup_benchmark_panel(ctx)
    stats = {'cz_error': {'ratio_of_medians': 2.11, 'n_worse_2x': 170,
                          'n': 352, 'median_reported': 1.4e-3}}
    api['render_details'](stats, {'cz_error'}, {})

    assert ctx.btn_bench_show_on_map.layout.display == ''
    assert '170 edges' in ctx.btn_bench_show_on_map.description

    api['show_on_map']()
    assert ctx.chk_err_threshold.value is True
    assert ctx.dd_gate_prop.value == 'cz_error'
    assert float(ctx.txt_g_thr_max.value) == pytest.approx(2 * 1.4e-3)


def test_show_on_map_is_hidden_when_nothing_qualifies():
    from iqmapper.ui.benchmark_panel import setup_benchmark_panel
    ctx = _ctx()
    api = setup_benchmark_panel(ctx)
    api['render_details'](
        {'cz_error': {'ratio_of_medians': 1.1, 'n_worse_2x': 0, 'n': 352,
                      'median_reported': 1.4e-3}}, set(), {})
    assert ctx.btn_bench_show_on_map.layout.display == 'none'
    api['show_on_map']()                      # a no-op, not a crash
    assert ctx.chk_err_threshold.value is False
