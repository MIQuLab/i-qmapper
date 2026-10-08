"""
fit_check.py — does a LUCJ circuit still fit? A yes/no without a search.

The multi-circuit UI lists an alternative layout for circuit i only when every
later circuit still fits after it. Answering that with full sweeps would cost
one search per alternative per later circuit; this module answers it with one
subgraph-isomorphism test each.

A circuit fits a graph exactly when the sweep would return at least one layout
for it: the LUCJ template with its minimum bridge count (`min_bridges`, or 1
when unset, the sweep's own floor) embeds in the graph. Fewer bridges is a
subgraph of more, so that one test decides it. The template comes from
`guided_search.template_edges`, the graph and the exclusions from the same
`auto_lucj` helpers `multi_sweep` uses, and the test is the induced
subgraph check `auto_lucj.create_lucj_zigzag_layout` runs.
"""
import rustworkx

from .auto_lucj import _exclude_with_buffer, _make_backend_cmap_pygraph
from .guided_search import template_edges
from .multi_sweep import _entry_qubits, _per_circuit


def template_graph(norb, n_bridges):
    """The LUCJ zigzag template with `n_bridges` bridges, as a PyGraph."""
    graph = rustworkx.PyGraph()
    graph.add_nodes_from(range(2 * norb + n_bridges))
    for u, v in template_edges(norb, n_bridges):
        graph.add_edge(u, v, None)
    return graph


def fits(norb, graph, min_bridges=None):
    """True when a `norb` circuit with at least `min_bridges` bridges (1 when
    None) embeds in `graph`."""
    n_bridges = 1 if min_bridges is None else int(min_bridges)
    if graph.num_nodes() < 2 * norb + n_bridges:
        return False
    return rustworkx.is_subgraph_isomorphic(graph,
                                            template_graph(norb, n_bridges))


def remaining_graph(backend, exclude_qubits, placed, buffer_qubits):
    """The coupling graph left after the hard exclusions and the circuits in
    `placed` (chosen entries, in placement order), each with its buffer —
    built as `multi_sweep` builds it."""
    remaining = _make_backend_cmap_pygraph(backend=backend)
    for idx in remaining.node_indices():
        remaining[idx] = idx
    stability = set(int(q) for q in (exclude_qubits or []))
    if stability:
        remaining, _ = _exclude_with_buffer(remaining, stability, 0)
    for entry in placed:
        remaining, _ = _exclude_with_buffer(
            remaining, set(_entry_qubits(entry)), buffer_qubits)
    return remaining


class LaterFit:
    """Answers "do circuits after i still fit if circuit i takes `entry`?"
    for one multi-circuit result. Results without a usable planning context
    (no backend) answer True, so nothing is hidden."""

    def __init__(self, result):
        self._ctx = (result or {}).get('_context') or {}
        self._circuits = list((result or {}).get('circuits') or [])
        self._cache = {}

    @property
    def usable(self):
        return self._ctx.get('backend') is not None and bool(
            self._ctx.get('norbs'))

    def later_circuits_fit(self, index, entry):
        if not self.usable:
            return True
        norbs = list(self._ctx['norbs'])
        later = range(index + 1, len(norbs))
        if not later:
            return True
        placed = [c['chosen'] for c in self._circuits[:index]] + [entry]
        key = tuple(frozenset(_entry_qubits(e)) for e in placed)
        if key not in self._cache:
            graph = remaining_graph(self._ctx['backend'],
                                    self._ctx.get('exclude_qubits'), placed,
                                    self._ctx.get('buffer_qubits', 1))
            self._cache[key] = all(
                fits(norbs[j], graph,
                     _per_circuit(self._ctx.get('min_bridges'), j))
                for j in later)
        return self._cache[key]
