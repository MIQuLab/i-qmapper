"""LQS counts each physical coupler once.

CZ backends (Heron) list every coupler in both directions in their coupling
map, and `state.edges_list` carries both; ECR backends (Eagle) list one. The
score of a layout must not depend on that.
"""
import logging
from types import SimpleNamespace

import numpy as np
import pytest
import rustworkx

from iqmapper import evaluation as ev
from iqmapper.ancilla_sweep import sweep_ancilla_layouts
from iqmapper.export import scoring
from iqmapper.export.scoring import estimate_circuit_fidelity, unique_couplers
from iqmapper.props_shim import CachedPropsShim


def _heavy_hex(d=7, seed=5):
    graph = rustworkx.generators.heavy_hex_graph(d)
    rng = np.random.default_rng(seed)
    n = graph.num_nodes()
    undirected = [tuple(e) for e in graph.edge_list()]
    readout = 0.005 + 0.02 * rng.random(n)
    err = {frozenset(e): 0.002 + 0.01 * rng.random() for e in undirected}
    return n, undirected, readout, err


def _state(n, edges, readout, err):
    """The app's state shape: positional gate array over `edges`."""
    return SimpleNamespace(
        qubit_ids=[str(i) for i in range(n)], edges_list=list(edges),
        qubit_props_data={'readout_error': np.asarray(readout)},
        gate_props_data={'cz_error': np.array([err[frozenset(e)]
                                               for e in edges])})


def _both(undirected):
    return [e for u, v in undirected for e in ((u, v), (v, u))]


def _score(state, entry):
    layout = {'alpha': [str(q) for q in entry['alpha']],
              'beta': [str(q) for q in entry['beta']], 'other': [],
              'ancilla': [str(q) for q in entry['ancillas']]}
    fake = SimpleNamespace(layouts_data={0: layout}, **vars(state))
    return estimate_circuit_fidelity(fake, 0, 'lucj')


def _lucj_gates(entry):
    a, b, anc = entry['alpha'], entry['beta'], entry['ancillas']
    gates = ([(a[j], a[j + 1]) for j in range(len(a) - 1)]
             + [(b[j], b[j + 1]) for j in range(len(b) - 1)])
    for i, r in enumerate(anc):
        gates += [(a[4 * i], r), (r, b[4 * i])]
    return gates


def _backend(n, undirected):
    graph = rustworkx.PyDiGraph()
    for _ in range(n):
        graph.add_node(None)
    for u, v in undirected:
        graph.add_edge(u, v, None)
        graph.add_edge(v, u, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


class _Props:
    def __init__(self, readout, err):
        self.readout, self.err = readout, err

    def readout_error(self, q):
        return float(self.readout[q])

    def gate_error(self, name, edge):
        return self.err[frozenset(edge)]


@pytest.fixture(scope='module')
def layout_case():
    n, undirected, readout, err = _heavy_hex()
    res = sweep_ancilla_layouts(8, _backend(n, undirected),
                                _Props(readout, err), min_bridges=1, top_n=3)
    entry = res['strata'][1][0]
    return n, undirected, readout, err, entry


def test_both_directions_and_one_direction_score_the_same(layout_case):
    n, undirected, readout, err, entry = layout_case
    heron = _score(_state(n, _both(undirected), readout, err), entry)
    eagle = _score(_state(n, undirected, readout, err), entry)
    assert heron['n_gates'] == eagle['n_gates']
    assert heron['fidelity'] == pytest.approx(eagle['fidelity'], rel=1e-12)


def test_n_gates_is_the_lucj_gate_count(layout_case):
    n, undirected, readout, err, entry = layout_case
    res = _score(_state(n, _both(undirected), readout, err), entry)
    assert len(entry['ancillas']) == 1
    assert res['n_gates'] == len(_lucj_gates(entry)) == 2 * (8 - 1) + 2
    expected = np.prod([1 - readout[q] for q in
                        entry['alpha'] + entry['beta'] + entry['ancillas']])
    expected *= np.prod([1 - err[frozenset(g)] for g in _lucj_gates(entry)])
    assert res['fidelity'] == pytest.approx(expected, rel=1e-12)


def test_ui_scan_and_harness_agree(layout_case):
    n, undirected, readout, err, entry = layout_case
    edges = _both(undirected)
    state = _state(n, edges, readout, err)
    ui = _score(state, entry)['fidelity']
    shim = CachedPropsShim(
        {'readout_error': state.qubit_props_data['readout_error']},
        {'cz_error': state.gate_props_data['cz_error']}, edges, n_qubits=n)
    qubits = entry['alpha'] + entry['beta'] + entry['ancillas']
    scan = shim.score_layout(qubits)
    harness = ev._lqs_scorer(state.qubit_props_data, state.gate_props_data,
                             edges, state.qubit_ids)(entry)
    assert scan['n_gates'] == len(_lucj_gates(entry))
    assert scan['fidelity'] == pytest.approx(ui, rel=1e-12)
    assert harness == pytest.approx(ui, rel=1e-12)


def test_direction_mismatch_keeps_the_larger_error_and_logs_once(caplog):
    scoring._DIRECTION_MISMATCH_LOGGED.discard(frozenset(('7', '8')))
    edges = [('7', '8', 0.01), ('8', '7', 0.03), ('7', '8', 0.02)]
    with caplog.at_level(logging.INFO, logger='iqmapper.export.scoring'):
        out = unique_couplers(edges)
        unique_couplers(edges)
    assert out == [('7', '8', 0.03)]
    assert sum('different errors per direction' in r.message
               for r in caplog.records) == 1


def test_a_missing_direction_never_hides_a_present_one():
    nan = float('nan')
    assert unique_couplers([('0', '1', nan), ('1', '0', 0.02)]) == [
        ('0', '1', 0.02)]
    assert unique_couplers([('0', '1', 0.02), ('1', '0', None)]) == [
        ('0', '1', 0.02)]
    assert unique_couplers([('0', '1', 0.02), ('1', '2', 0.03)]) == [
        ('0', '1', 0.02), ('1', '2', 0.03)]


def test_a_real_heron_coupling_map_counts_each_coupler_once(tmp_path,
                                                             monkeypatch):
    """FakeFez through the app's own data path: both directions listed."""
    fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
    from iqmapper import data
    from iqmapper.state import SessionState
    monkeypatch.setattr(data, '_CACHE_DIR', str(tmp_path))
    backend = fake_provider.FakeFez()
    state = SessionState()
    state.active_backend = backend
    state.active_backend_name = backend.name
    state.initial_props = backend.properties()
    _x, _y, edges, q_ids, _ro, _cz = data._generate_ibm(state, None)
    state.edges_list, state.qubit_ids = edges, q_ids
    assert len({frozenset(e) for e in edges}) * 2 == len(edges)

    res = sweep_ancilla_layouts(8, backend, backend.properties(),
                                min_bridges=1, top_n=1)
    entry = res['strata'][1][0]
    scored = _score(SimpleNamespace(
        qubit_ids=state.qubit_ids, edges_list=state.edges_list,
        qubit_props_data=state.qubit_props_data,
        gate_props_data=state.gate_props_data), entry)
    assert scored['n_gates'] == len(_lucj_gates(entry)) == 16
