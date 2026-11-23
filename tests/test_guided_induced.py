"""The guided engine only produces induced embeddings of the LUCJ template.

A layout must not contain two qubits joined by a coupler the template does
not have: the LQS counts every coupler between layout qubits, so such a
layout would be scored on a gate the circuit never runs. VF2 searches induced
subgraphs, so this keeps both engines in the same space.

The check here reads the coupling graph directly instead of calling
`validate_layout`, so it does not test the validator with itself.
"""
from types import SimpleNamespace

import numpy as np
import pytest
import rustworkx

from iqmapper import guided_search as gs
from iqmapper.auto_lucj import _make_backend_cmap_pygraph
from iqmapper.guided_search import guided_sweep


def _backend_from(undirected):
    graph = rustworkx.PyDiGraph()
    n = 1 + max(max(e) for e in undirected)
    for _ in range(n):
        graph.add_node(None)
    for u, v in undirected:
        graph.add_edge(u, v, None)
        graph.add_edge(v, u, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


class _SeededProps:
    """Deterministic spread of readout and gate errors."""

    def __init__(self, n, seed):
        rng = np.random.default_rng(seed)
        self._r = 0.005 + 0.03 * rng.random(n)
        self._seed = seed

    def readout_error(self, q):
        return float(self._r[int(q)])

    def gate_error(self, _name, edge):
        u, v = sorted((int(edge[0]), int(edge[1])))
        rng = np.random.default_rng((self._seed, u, v))
        return 0.002 + 0.02 * float(rng.random())


def _extra_couplers(backend, entry, norb, k):
    """Couplers between layout qubits that are not template edges."""
    graph = _make_backend_cmap_pygraph(backend=backend)
    layout = list(entry['alpha']) + list(entry['beta']) + list(entry['ancillas'])
    template = {frozenset((layout[u], layout[v]))
                for u, v in gs.template_edges(norb, k)}
    qubits = set(layout)
    induced = {frozenset((u, v)) for u, v in graph.edge_list()
               if u in qubits and v in qubits}
    return sorted(tuple(sorted(e)) for e in induced - template)


def _entries(result):
    for k, entries in result['strata'].items():
        for entry in entries:
            yield k, entry


# ── growth and refinement never bring in an extra coupler ───────────

@pytest.mark.parametrize('d, norbs', [(5, (4, 6, 8)), (7, (8, 12))])
def test_no_guided_entry_has_a_non_template_coupler_on_heavy_hex(d, norbs):
    undirected = [tuple(e) for e in
                  rustworkx.generators.heavy_hex_graph(d).edge_list()]
    backend = _backend_from(undirected)
    props = _SeededProps(1 + max(max(e) for e in undirected), seed=d)
    for norb in norbs:
        res = guided_sweep(norb, backend, props, min_bridges=1, top_n=5,
                           circuit_score_fn=lambda e: e['product_score'])
        assert res['strata'], f'no strata for d={d} norb={norb}'
        for k, entry in _entries(res):
            assert _extra_couplers(backend, entry, norb, k) == [], (
                f'd={d} norb={norb} k={k}: {entry["alpha"]} '
                f'{entry["beta"]} {entry["ancillas"]}')


def test_no_guided_entry_has_a_non_template_coupler_on_fake_fez():
    fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
    backend = fake_provider.FakeFez()
    props = backend.properties()
    for norb in (8, 12, 16):
        res = guided_sweep(norb, backend, props, min_bridges=1, top_n=10,
                           circuit_score_fn=lambda e: e['product_score'])
        for k, entry in _entries(res):
            assert _extra_couplers(backend, entry, norb, k) == [], (
                f'norb={norb} k={k}: {entry["alpha"]} {entry["beta"]} '
                f'{entry["ancillas"]}')


# ── the validator ───────────────────────────────────────────────────

def test_the_fake_fez_hexagon_wrap_is_rejected():
    """norb=12, k=1: alpha runs round a hexagon, so its two ends 109 and 108
    sit on one coupler. Every template edge is present; only the extra
    coupler makes it invalid."""
    fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
    graph = _make_backend_cmap_pygraph(backend=fake_provider.FakeFez())
    adjacency = {q: frozenset(graph.neighbors(q)) for q in graph.node_indices()}
    alpha = [109, 110, 111, 98, 91, 90, 89, 88, 87, 97, 107, 108]
    beta = [129, 128, 127, 126, 125, 124, 123, 136, 143, 142, 141, 140]
    layout = alpha + beta + [118]
    norb, k = 12, 1

    assert all(layout[v] in adjacency[layout[u]]
               for u, v in gs.template_edges(norb, k))
    assert 108 in adjacency[109]
    assert gs.validate_layout(layout, norb, k, adjacency) is False


def _path_adjacency(edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return {q: frozenset(n) for q, n in adj.items()}


# norb=2, k=1: alpha = 0-1, beta = 2-3, bridge 4 joins 0 and 2.
_TEMPLATE = [(0, 1), (2, 3), (0, 4), (4, 2)]
_LAYOUT = [0, 1, 2, 3, 4]


def test_validate_layout_accepts_an_induced_embedding():
    assert gs.validate_layout(_LAYOUT, 2, 1, _path_adjacency(_TEMPLATE))


@pytest.mark.parametrize('extra, kind', [
    ((1, 3), 'alpha-beta direct'),
    ((0, 2), 'chain heads under the bridge'),
    ((4, 1), 'bridge to a non-bridge slot'),
])
def test_validate_layout_rejects_an_extra_coupler(extra, kind):
    adjacency = _path_adjacency(_TEMPLATE + [extra])
    assert gs.validate_layout(_LAYOUT, 2, 1, adjacency) is False, kind


def test_a_coupler_to_a_qubit_outside_the_layout_is_fine():
    adjacency = _path_adjacency(_TEMPLATE + [(1, 5), (4, 6)])
    assert gs.validate_layout(_LAYOUT, 2, 1, adjacency)


def test_seeds_never_pair_two_coupled_chain_heads():
    """A triangle 0-1-2: 1 and 2 both neighbour 0, but they are coupled to
    each other, so (bridge 0, heads 1 and 2) is not a valid seed."""
    backend = _backend_from([(0, 1), (0, 2), (1, 2), (0, 3)])
    graph = _make_backend_cmap_pygraph(backend=backend)
    engine = gs._GuidedEngine(1, graph, np.full(4, 0.01),
                              {frozenset(e): 0.01 for e in graph.edge_list()})
    for seed in engine.seeds(1):
        a, b = seed.alpha[0], seed.beta[0]
        assert b not in engine.adjacency[a], (seed.bridges, a, b)
    assert {(s.bridges[0], frozenset((s.alpha[0], s.beta[0])))
            for s in engine.seeds(1)} == {(0, frozenset((1, 3))),
                                          (0, frozenset((2, 3)))}
