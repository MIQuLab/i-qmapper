"""The fit test agrees with the sweep: a circuit fits a graph exactly when
`sweep_ancilla_layouts` returns at least one layout for it there."""
from types import SimpleNamespace

import pytest
import rustworkx

from iqmapper import fit_check
from iqmapper.ancilla_sweep import sweep_ancilla_layouts

from test_ancilla_sweep import FakeProps, make_backend


def _backend(undirected, n=None):
    graph = rustworkx.PyDiGraph()
    n = n if n is not None else 1 + max(max(e) for e in undirected)
    for _ in range(n):
        graph.add_node(None)
    for u, v in undirected:
        graph.add_edge(u, v, None)
        graph.add_edge(v, u, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


def ladder(width):
    """Top 0..w-1, bottom w..2w-1, rung qubit 2w+j joining column j."""
    edges = []
    for j in range(width - 1):
        edges += [(j, j + 1), (width + j, width + j + 1)]
    for j in range(width):
        edges += [(j, 2 * width + j), (2 * width + j, width + j)]
    return _backend(edges)


def _sweep_finds(norb, backend, exclude, min_bridges):
    try:
        res = sweep_ancilla_layouts(norb, backend, FakeProps(),
                                    exclude_qubits=exclude or None,
                                    min_bridges=min_bridges, top_n=1)
    except (ValueError, RuntimeError):
        return False
    return any(res['strata'].values())


CASES = [
    # (backend factory, norb, exclusions, min_bridges)
    (make_backend, 4, set(), None),
    (make_backend, 8, set(), None),
    (make_backend, 8, set(), 2),
    (make_backend, 8, {3}, None),                 # good block cut
    (make_backend, 8, {3, 27}, None),             # both blocks cut
    (make_backend, 6, {16}, 1),                   # a bridge qubit gone
    (make_backend, 9, set(), None),               # longer than a block
    (lambda: ladder(5), 3, set(), None),
    (lambda: ladder(5), 3, {12}, None),           # middle rung gone
    (lambda: ladder(5), 5, {10}, None),           # first rung gone
    (lambda: ladder(5), 5, {10}, 2),
    (lambda: ladder(5), 4, {2}, None),
]


@pytest.mark.parametrize('factory, norb, exclude, min_bridges', CASES)
def test_fits_agrees_with_the_sweep(factory, norb, exclude, min_bridges):
    backend = factory()
    graph = fit_check.remaining_graph(backend, exclude, [], 0)
    assert fit_check.fits(norb, graph, min_bridges) == _sweep_finds(
        norb, backend, exclude, min_bridges)


def test_both_outcomes_are_covered():
    outcomes = {_sweep_finds(n, f(), e, m) for f, n, e, m in CASES}
    assert outcomes == {True, False}


def test_a_placed_circuit_and_its_buffer_are_taken_out():
    backend = ladder(8)
    middle = {'alpha': [3, 4], 'beta': [11, 12], 'ancillas': [19]}
    left = {'alpha': [0, 1], 'beta': [8, 9], 'ancillas': [16]}
    assert not fit_check.fits(
        3, fit_check.remaining_graph(backend, None, [middle], 1))
    assert fit_check.fits(
        3, fit_check.remaining_graph(backend, None, [left], 1))


def test_later_fit_without_a_context_hides_nothing():
    check = fit_check.LaterFit({'circuits': [], '_context': {'stub': True}})
    assert check.usable is False
    assert check.later_circuits_fit(0, {'alpha': [0], 'beta': [1],
                                        'ancillas': []}) is True
