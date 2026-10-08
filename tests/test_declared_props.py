"""Only the properties a backend declares are fetched; the rest are NaN with no
calls and no messages, every key is still stored, a real hole logs one line,
the property dropdowns list what has values, and a missing value reads
"no data".
"""
import logging
from types import SimpleNamespace

import numpy as np
import pytest

from iqmapper import data
from iqmapper.constants import (GATE_PROP_OPTIONS, QUBIT_PROP_OPTIONS,
                                fmt_e, fmt_ns, fmt_us)

fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')

# FakeNighthawk's own notice that its numbers are not typical Nighthawk values.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Properties of fake_nighthawk are not intended:UserWarning')

ALL_Q = {k for _, k in QUBIT_PROP_OPTIONS}
ALL_G = {k for _, k in GATE_PROP_OPTIONS}

#: What each fake declares in properties() (qiskit-ibm-runtime fake data).
UNDECLARED = {
    'FakeNighthawk': ({'rx_error', 'measure_2_error'}, {'rzz_error'}),
    'FakeMiami': ({'measure_2_error'}, {'rzz_error'}),
    'FakeFez': ({'measure_error', 'measure_2_error'}, set()),
}


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def collect():
    handler = _Collect()
    logger = logging.getLogger('iqmapper')
    old_level = logger.level
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    yield handler
    logger.removeHandler(handler)
    logger.setLevel(old_level)


def _backend(name):
    return getattr(fake_provider, name)()


def _edges(backend):
    return list(backend.coupling_map.get_edges())


# ── what is declared, what is fetched ───────────────────────────────

@pytest.mark.parametrize('name', sorted(UNDECLARED))
def test_declared_keys_match_the_backend(name):
    backend = _backend(name)
    q_keys, g_keys = data.declared_keys(backend.properties())
    no_q, no_g = UNDECLARED[name]
    assert q_keys == ALL_Q - no_q
    assert g_keys == ALL_G - no_g


@pytest.mark.parametrize('name', sorted(UNDECLARED))
def test_undeclared_keys_cost_no_calls_and_no_messages(name, collect,
                                                      monkeypatch):
    backend = _backend(name)
    props = backend.properties()
    asked = []
    real_q, real_g = data.fetch_qubit_prop, data.fetch_gate_prop
    monkeypatch.setattr(data, 'fetch_qubit_prop', lambda p, i, k: asked.append(
        k) or real_q(p, i, k))
    monkeypatch.setattr(data, 'fetch_gate_prop', lambda p, e, k: asked.append(
        k) or real_g(p, e, k))

    qp, gp = data.extract_props(props, backend.num_qubits, _edges(backend))

    no_q, no_g = UNDECLARED[name]
    assert not (set(asked) & (no_q | no_g)), 'an undeclared key was fetched'
    assert set(qp) == ALL_Q and set(gp) == ALL_G        # full key set kept
    for key in no_q:
        assert np.isnan(qp[key]).all() and len(qp[key]) == backend.num_qubits
    for key in no_g:
        assert np.isnan(gp[key]).all() and len(gp[key]) == len(_edges(backend))
    noisy = [r for r in collect.records if r.levelno >= logging.INFO]
    assert noisy == [], [r.getMessage() for r in noisy]


def test_the_app_load_of_a_nighthawk_logs_nothing(collect, monkeypatch,
                                                  tmp_path):
    from iqmapper.state import SessionState
    monkeypatch.setattr(data, '_CACHE_DIR', str(tmp_path))
    backend = _backend('FakeNighthawk')
    state = SessionState()
    state.active_backend, state.active_backend_name = backend, backend.name
    state.initial_props = backend.properties()
    data._generate_ibm(state, None)
    assert [r for r in collect.records if r.levelno >= logging.INFO
            and 'Cache' not in r.getMessage()] == []
    assert state.props_warning is None


# ── a real hole: one line for the whole key ─────────────────────────

class _MissingT2:
    """FakeFez properties with no T2 value for qubit 72."""

    def __init__(self, props):
        self._props = props

    def __getattr__(self, name):
        return getattr(self._props, name)

    def t2(self, qubit):
        if qubit == 72:
            raise KeyError('T2')
        return self._props.t2(qubit)


def test_a_hole_in_a_declared_property_logs_one_line(collect):
    backend = _backend('FakeFez')
    qp, _gp = data.extract_props(_MissingT2(backend.properties()),
                                 backend.num_qubits, _edges(backend))
    assert np.isnan(qp['t2'][72]) and np.isfinite(qp['t2'][71])
    lines = [r.getMessage() for r in collect.records
             if r.levelno >= logging.WARNING]
    assert lines == ['t2: 1 of 156 qubits without a value']


def test_props_without_gate_lists_fetch_every_key():
    """A props object that does not expose its gates (older objects, test
    stand-ins) is asked for every key, as before."""
    assert data.declared_keys(SimpleNamespace()) is None


# ── caches ──────────────────────────────────────────────────────────

def test_old_and_new_cache_files_load(monkeypatch, tmp_path):
    monkeypatch.setattr(data, '_CACHE_DIR', str(tmp_path))
    # Written the old way: every key fetched per element (NaN where absent).
    old_q = {k: np.full(4, 0.01) for k in ALL_Q}
    old_g = {k: np.full(3, 0.02) for k in ALL_G}
    data._save_calibration('ibm_old', '2026-01-01 00:00:00', old_q, old_g)
    assert data._load_calibration('ibm_old', '2026-01-01 00:00:00') is not None

    backend = _backend('FakeMiami')
    qp, gp = data.extract_props(backend.properties(), backend.num_qubits,
                                _edges(backend))
    data._save_calibration('fake_miami', '2026-01-02 00:00:00', qp, gp)
    cached = data._load_calibration('fake_miami', '2026-01-02 00:00:00')
    assert cached is not None
    assert set(cached[0]) == ALL_Q and set(cached[1]) == ALL_G
    assert np.isnan(cached[1]['rzz_error']).all()


# ── dropdowns ───────────────────────────────────────────────────────

def _build_on(monkeypatch, name, is_secondary=False):
    """A real MapperUIBuilder whose device data is the fake backend's own
    extraction (the same arrays a cached snapshot would hold)."""
    backend = _backend(name)
    n = backend.num_qubits
    edges = _edges(backend)
    qp, gp = data.extract_props(backend.properties(), n, edges)

    def _device(state, vendor, chip, date, prefetched_props=None):
        state.qubit_props_data, state.gate_props_data = qp, gp
        return (list(range(n)), [0.0] * n, edges, [str(i) for i in range(n)],
                qp['readout_error'], gp['cz_error'])

    monkeypatch.setattr('iqmapper.data.generate_device_data', _device)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    for key, val in dict(
            active_backend=object(), active_backend_name=backend.name,
            active_chip='stub_chip', active_service=None,
            available_backends=[], props_warning=None, initial_props=object(),
            active_ansatz='lucj', active_vendor='IBM',
            active_coords=[(i, 0) for i in range(n)], custom_colors=[],
            cal_history_dates=['_seed_']).items():
        setattr(state, key, val)
    b = MapperUIBuilder(state, is_secondary=is_secondary)
    b.build()
    b._refresh()
    return b


def _keys(dd):
    return {key for _label, key in dd.options}


@pytest.mark.parametrize('name', sorted(UNDECLARED))
@pytest.mark.parametrize('is_secondary', [False, True])
def test_property_dropdowns_list_what_the_backend_has(monkeypatch, name,
                                                      is_secondary):
    b = _build_on(monkeypatch, name, is_secondary)
    no_q, no_g = UNDECLARED[name]
    for attr in ('dd_qubit_prop', 'dd_rank_prop', 'dd_trend_prop',
                 'dd_stable_prop'):
        assert _keys(getattr(b, attr)) == ALL_Q - no_q, attr
    for attr in ('dd_gate_prop', 'dd_grank_prop', 'dd_gtrend_prop',
                 'dd_gstable_prop'):
        assert _keys(getattr(b, attr)) == ALL_G - no_g, attr
    assert b.dd_qubit_prop.value == 'readout_error'
    assert b.dd_gate_prop.value == 'cz_error'


def test_a_selected_property_that_disappears_falls_back(monkeypatch):
    b = _build_on(monkeypatch, 'FakeFez')
    b.dd_gate_prop.value = 'rzz_error'
    b.state.gate_props_data = dict(b.state.gate_props_data,
                                   rzz_error=np.full(len(b.state.edges_list),
                                                     np.nan))
    b._refresh()
    assert 'rzz_error' not in _keys(b.dd_gate_prop)
    assert b.dd_gate_prop.value == 'cz_error'


def test_nothing_loaded_keeps_the_full_list():
    from iqmapper.ui.prop_options import available_options
    assert available_options(QUBIT_PROP_OPTIONS, {}) == list(QUBIT_PROP_OPTIONS)
    assert available_options(GATE_PROP_OPTIONS, None) == list(GATE_PROP_OPTIONS)


# ── "no data" ───────────────────────────────────────────────────────

def test_a_missing_value_reads_no_data():
    assert fmt_e(np.nan) == 'no data'
    assert fmt_us(np.nan) == 'no data'
    assert fmt_ns(np.nan) == 'no data'
    assert fmt_e(1.23e-3) == '1.230E-3' and fmt_us(248.3) == '248.3 µs'


def test_the_heatmap_hover_says_no_data():
    pytest.importorskip('plotly')
    from iqmapper import plot
    from test_focus_isolate_boundary import _make_state, _wref
    state = _make_state(single=True)
    state.readout_errors = np.array([0.01, np.nan, 0.03, 0.04])
    state.qubit_props_data = {'readout_error': state.readout_errors}
    fig = plot.create_figure()
    plot.build_device_graph(fig, state, _wref(n_circuits=1))
    plot.refresh_plot(fig, state, _wref(n_circuits=1))
    hovers = [t for tr in fig.data for t in (getattr(tr, 'hovertext', None)
                                              or getattr(tr, 'text', None) or ())
              if isinstance(t, str) and 'Qubit 1<' in t]
    assert hovers and all('no data' in h and 'NAN' not in h for h in hovers)
