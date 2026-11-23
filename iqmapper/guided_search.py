"""
guided_search.py — calibration-guided beam search for LUCJ layouts.

The existing engine (`ancilla_sweep.sweep_ancilla_layouts`) enumerates zigzag
embeddings with VF2, which is blind to calibration: it finds subgraphs that fit
the template and only then asks how good they are. On a large chip at low
bridge counts that enumeration yields far more mappings than the 2000 cap and
gets cut off, so the ranking sees only an arbitrary sample of the search
space. Quality survives because the downstream cascade, monotonicity anchors
and diversity quota are good — but generation is throwing away information it
already has.

This module grows layouts WITH the calibration in hand: at every step it
extends the chains onto the best available neighbours and keeps the best
`beam_width` partial states. It then polishes the finalists with a 1-opt local
search against the real objective (circuit-level LQS).

It is an alternative engine, not a replacement. `ancilla_sweep` stays the
default until `evaluation.compare_engines` says otherwise on real cached
calibration.


THE TEMPLATE (derived from auto_lucj.create_lucj_zigzag_layout)
---------------------------------------------------------------
Everything here must produce a valid embedding of the same graph the frozen
engine searches for, so the rules are restated explicitly:

* `create_linear_chains(norb)` makes two disjoint paths. Template node `j` is
  alpha orbital j (0 <= j < norb); template node `norb + j` is beta orbital j.
  Edges: (j, j+1) and (norb+j, norb+j+1) for j in 0..norb-2.
* Bridges sit at orbital slots `n` with `n % 4 == 0`, so slot 0 ALWAYS carries
  a bridge and the geometric ceiling is `ceil(norb / 4)`. Bridge i is template
  node `2*norb + i` and is adjacent to exactly two nodes: alpha slot `4*i` and
  beta slot `4*i`.
* A layout is the list `alpha(norb) + beta(norb) + bridges(k)`, and an
  embedding is valid iff it is injective, every template edge maps to a
  coupling edge, and no other coupling edge joins two of its qubits. That
  last rule makes the embedding induced, which is what
  `rustworkx.vf2_mapping(subgraph=True)` returns by default, so both engines
  search the same space. The LQS counts every coupler between layout qubits,
  so a non-template one would be scored as a gate the circuit never runs.

LUCJ PREFIX CONTRACT (from ancilla_sweep, and binding here)
    A k'-bridge variant keeps the bridges at the FIRST k' standard slots
    `(0, 4, ..., 4*(k'-1))` and drops the tail — one variant per k', never a
    scattered subset like {0, 8, 12}. A scattered set is a valid subgraph with
    a valid score but a semantically invalid ansatz layout. `bridge_slots` is
    therefore always `tuple(range(0, 4*k', 4))`, which is what certifies it.

CANONICALISATION
    `_deduplicate_layouts` treats a layout and its alpha/beta swap as the same
    thing; `_embedding_data` normalises by putting the chain with the smaller
    first element in alpha. Both are mirrored here (`_normalise_chains`), so
    the two engines agree on what counts as a duplicate.


SCORING
-------
Two scores, exactly as the frozen engine defines them:

* `additive_score` — sum of readout errors over used qubits plus 2Q errors
  over used edges. Grows with layout size, so it is comparable WITHIN a
  stratum only, never across.
* `product_score` — `prod(1 - readout) * prod(1 - gate_err)`. Adding a bridge
  multiplies by `(1-r)(1-e1)(1-e2) <= 1`, so it is comparable across strata.
  This is the beam's incremental objective, accumulated in log space to keep
  long chains numerically honest, and the pre-filter for the final window.

The FINAL ranking and the refinement objective are the circuit-level LQS
(`circuit_score_fn`), the same scorer the UI banner shows. The product score
is a proxy used to decide what to keep, not what to recommend.

Uncalibrated or NaN values take `ancilla_sweep._UNCALIBRATED_PENALTY`, via the
frozen module's own `_safe_readout` / `_safe_gate_error` — imported, not
reimplemented, so the two engines cannot drift apart on this.


WHAT THIS DOES NOT DO
---------------------
Beam search is a heuristic. It can miss the global optimum that exhaustive VF2
would find on a small device, which is precisely why the default stays VF2 and
why `evaluation.compare_engines` exists. On the fixtures in the test suite the
two agree; on a large chip VF2 cannot enumerate the space at all, so the
comparison there is between a guided sample and an arbitrary one.
"""
import logging
import math
import time
from collections import namedtuple

import numpy as np
import rustworkx

from .ancilla_sweep import (_UNCALIBRATED_PENALTY, _safe_gate_error,
                            _safe_readout, _select_with_diversity)
from .auto_lucj import (_get_two_q_gate_name, _make_backend_cmap_pygraph,
                        _remove_nodes_from_graph, create_lucj_zigzag_layout)

_log = logging.getLogger(__name__)

#: Beam width default. 200 partial states per growth step is ample on
#: heavy-hex, where a chain qubit has at most 3 candidate continuations.
DEFAULT_BEAM_WIDTH = 200

#: How many finalists per stratum get the 1-opt polish.
DEFAULT_REFINE_TOP = 5

#: Iteration ceiling for 1-opt, so a pathological calibration cannot spin.
DEFAULT_MAX_ITER = 20

#: log(0) guard: a completely dead qubit/edge scores as this, not -inf, so a
#: partial state carrying one can still be compared and discarded on merit.
_LOG_FLOOR = -30.0

Move = namedtuple('Move', 'kind index qubit')


# ── template helpers ────────────────────────────────────────────────

def bridge_slots_for(k):
    """The standard slots a k-bridge layout occupies. Prefix contract."""
    return tuple(range(0, 4 * k, 4))


def max_bridges_for(norb):
    """Geometric ceiling: one bridge every 4 orbitals, slot 0 included."""
    return (norb + 3) // 4


def template_edges(norb, k):
    """Every edge of the LUCJ template, as (template_node, template_node).

    Chain edges first, then the two edges of each bridge. Positions follow the
    layout convention: alpha = 0..norb-1, beta = norb..2norb-1, bridge i =
    2*norb + i.
    """
    edges = []
    for j in range(norb - 1):
        edges.append((j, j + 1))
        edges.append((norb + j, norb + j + 1))
    for i in range(k):
        slot = 4 * i
        edges.append((slot, 2 * norb + i))
        edges.append((2 * norb + i, norb + slot))
    return edges


def validate_layout(layout, norb, k, adjacency):
    """True iff `layout` is a valid induced embedding of the (norb, k) template.

    Checks injectivity, that every template edge lands on a coupling edge, and
    that every coupling edge between two layout qubits is a template edge.
    Used after every refinement move, because a move that produced an invalid
    layout would be worse than no refinement at all.
    """
    expected = 2 * norb + k
    if len(layout) != expected or any(q is None for q in layout):
        return False
    if len(set(layout)) != expected:
        return False
    edges = template_edges(norb, k)
    for u, v in edges:
        if layout[v] not in adjacency.get(layout[u], ()):
            return False
    allowed = {frozenset(e) for e in edges}
    position = {q: i for i, q in enumerate(layout)}
    for i, q in enumerate(layout):
        for n in adjacency.get(q, ()):
            j = position.get(n)
            if j is not None and frozenset((i, j)) not in allowed:
                return False
    return True


def _normalise_chains(alpha, beta):
    """Mirror `_embedding_data`: alpha is the chain with the smaller head."""
    if alpha and beta and alpha[0] > beta[0]:
        return list(beta), list(alpha)
    return list(alpha), list(beta)


# ── search state ────────────────────────────────────────────────────

class _Partial:
    """A partially grown layout: both chain prefixes plus its bridges.

    `depth` is how many orbital slots are filled (identical on both chains —
    they grow in lockstep so a bridge slot can always see both endpoints).
    """

    __slots__ = ('alpha', 'beta', 'bridges', 'used', 'log_score', 'add_score')

    def __init__(self, alpha, beta, bridges, used, log_score, add_score):
        self.alpha = alpha
        self.beta = beta
        self.bridges = bridges
        self.used = used
        self.log_score = log_score
        self.add_score = add_score

    @property
    def depth(self):
        return len(self.alpha)

    def key(self):
        """Canonical identity, collapsing the alpha/beta swap."""
        a, b = _normalise_chains(self.alpha, self.beta)
        return (tuple(a), tuple(b), tuple(self.bridges))

    def extend(self, a, b, bridge, d_log, d_add):
        used = self.used | {a, b}
        bridges = self.bridges
        if bridge is not None:
            used = used | {bridge}
            bridges = bridges + [bridge]
        return _Partial(self.alpha + [a], self.beta + [b], bridges, used,
                        self.log_score + d_log, self.add_score + d_add)


def _safe_log(value):
    """log of a fidelity factor, floored so a dead element stays comparable."""
    if value <= 0.0:
        return _LOG_FLOOR
    return max(_LOG_FLOOR, math.log(value))


# ── the engine ──────────────────────────────────────────────────────

class _GuidedEngine:
    """Holds the graph, the calibration and the growth rules for one run."""

    def __init__(self, norb, graph, readout, edge_err, qubit_weight=None):
        self.norb = norb
        self.graph = graph
        self.readout = readout
        self.edge_err = edge_err
        self.qubit_weight = qubit_weight
        self.expansions = 0
        self.adjacency = {}
        for node in graph.node_indices():
            self.adjacency[node] = frozenset(graph.neighbors(node))

    # ── per-element factors ─────────────────────────────────────
    def r_of(self, q):
        try:
            return float(self.readout[q])
        except (IndexError, TypeError):
            return _UNCALIBRATED_PENALTY

    def e_of(self, a, b):
        return self.edge_err.get(frozenset((a, b)), _UNCALIBRATED_PENALTY)

    def q_factor(self, q):
        """(1 - readout) for a qubit, times the optional weight hook."""
        factor = max(0.0, 1.0 - self.r_of(q))
        if self.qubit_weight is not None:
            try:
                weight = float(self.qubit_weight(q))
            except Exception:
                weight = 1.0
            factor *= max(0.0, min(1.0, weight))
        return factor

    def e_factor(self, a, b):
        return max(0.0, 1.0 - self.e_of(a, b))

    def couples_only_to(self, q, placed, allowed):
        """Is every coupler from `q` into `placed` one of the `allowed` ends?

        Growth calls this for each qubit it adds, against everything already
        placed, so every pair of layout qubits is checked once and the
        finished layout is an induced embedding.
        """
        return self.adjacency.get(q, frozenset()) & placed <= allowed

    # ── seeding ─────────────────────────────────────────────────
    def seeds(self, k_target):
        """Initial states at orbital slot 0.

        Slot 0 always carries a bridge when k >= 1 (n % 4 == 0 includes 0), so
        a seed is a (bridge, alpha0, beta0) triple with the bridge adjacent to
        both chain heads. With k == 0 there is no bridge and a seed is any
        ordered pair of distinct qubits — collapsed by canonicalisation. The
        two chain heads are never coupled to each other.
        """
        out = []
        # Iterate the ADJACENCY, not the graph: excluded qubits (and any
        # padding nodes used to keep physical ids as indices) are absent from
        # adjacency by construction, and must never seed a layout.
        if k_target >= 1:
            for r in sorted(self.adjacency):
                neighbours = sorted(self.adjacency[r])
                for i, a in enumerate(neighbours):
                    for b in neighbours[i + 1:]:
                        if b in self.adjacency.get(a, frozenset()):
                            continue
                        d_log = (_safe_log(self.q_factor(a))
                                 + _safe_log(self.q_factor(b))
                                 + _safe_log(self.q_factor(r))
                                 + _safe_log(self.e_factor(a, r))
                                 + _safe_log(self.e_factor(r, b)))
                        d_add = (self.r_of(a) + self.r_of(b) + self.r_of(r)
                                 + self.e_of(a, r) + self.e_of(r, b))
                        out.append(_Partial([a], [b], [r], {a, b, r},
                                            d_log, d_add))
        else:
            nodes = sorted(self.adjacency)
            for i, a in enumerate(nodes):
                for b in nodes[i + 1:]:
                    if b in self.adjacency.get(a, frozenset()):
                        continue
                    d_log = (_safe_log(self.q_factor(a))
                             + _safe_log(self.q_factor(b)))
                    d_add = self.r_of(a) + self.r_of(b)
                    out.append(_Partial([a], [b], [], {a, b}, d_log, d_add))
        return out

    # ── growth ──────────────────────────────────────────────────
    def _slot_has_bridge(self, j, k_target):
        """Does orbital slot j carry a bridge in a k_target layout?"""
        return j % 4 == 0 and (j // 4) < k_target

    def expand(self, state, k_target):
        """Every legal one-slot extension of `state`.

        A new qubit may couple only to its template neighbours among the
        qubits already placed: a chain qubit to its predecessor, a bridge to
        the two chain qubits it joins.
        """
        j = state.depth
        a_prev, b_prev = state.alpha[-1], state.beta[-1]
        want_bridge = self._slot_has_bridge(j, k_target)
        out = []
        for a in sorted(self.adjacency.get(a_prev, frozenset())):
            if a in state.used:
                continue
            if not self.couples_only_to(a, state.used, {a_prev}):
                continue
            for b in sorted(self.adjacency.get(b_prev, frozenset())):
                if b in state.used or b == a:
                    continue
                if not self.couples_only_to(b, state.used | {a}, {b_prev}):
                    continue
                self.expansions += 1
                d_log = (_safe_log(self.q_factor(a))
                         + _safe_log(self.q_factor(b))
                         + _safe_log(self.e_factor(a_prev, a))
                         + _safe_log(self.e_factor(b_prev, b)))
                d_add = (self.r_of(a) + self.r_of(b)
                         + self.e_of(a_prev, a) + self.e_of(b_prev, b))
                if not want_bridge:
                    out.append(state.extend(a, b, None, d_log, d_add))
                    continue
                common = (self.adjacency.get(a, frozenset())
                          & self.adjacency.get(b, frozenset()))
                for r in sorted(common):
                    if r in state.used or r in (a, b):
                        continue
                    if not self.couples_only_to(r, state.used | {a, b},
                                                {a, b}):
                        continue
                    b_log = (_safe_log(self.q_factor(r))
                             + _safe_log(self.e_factor(a, r))
                             + _safe_log(self.e_factor(r, b)))
                    b_add = self.r_of(r) + self.e_of(a, r) + self.e_of(r, b)
                    out.append(state.extend(a, b, r, d_log + b_log,
                                            d_add + b_add))
        return out

    def run_beam(self, k_target, beam_width):
        """Grow to full length, keeping the best `beam_width` states.

        Pruning only ever compares states of the same depth, so a long
        mediocre chain is never ranked against a short excellent one.
        """
        beam = self._prune(self.seeds(k_target), beam_width)
        for _j in range(1, self.norb):
            nxt = []
            for state in beam:
                nxt.extend(self.expand(state, k_target))
            if not nxt:
                return []
            beam = self._prune(nxt, beam_width)
        return beam

    @staticmethod
    def _prune(states, beam_width):
        """Top-`beam_width` by score, deduplicated, deterministic ties."""
        best = {}
        for state in states:
            key = state.key()
            current = best.get(key)
            if current is None or state.log_score > current.log_score:
                best[key] = state
        ranked = sorted(best.values(),
                        key=lambda s: (-s.log_score, s.key()))
        return ranked[:beam_width]


# ── entry construction (mirrors ancilla_sweep._add_variant) ─────────

def _entry_from_state(engine, alpha, beta, bridges, k):
    """Build a sweep-shaped entry for the first `k` bridges of a layout.

    Recomputes both scores from scratch over the kept elements rather than
    reusing the beam's incremental total, because a derived k' variant drops
    bridges and because the frozen engine's numbers are the reference — see
    test_guided_search.py::test_score_parity_with_the_frozen_sweep.
    """
    norb = engine.norb
    kept = list(bridges[:k])
    norm_alpha, norm_beta = _normalise_chains(alpha, beta)
    chain = norm_alpha + norm_beta

    chain_q = np.array(chain, dtype=int)
    readout_vals = np.array([engine.r_of(q) for q in chain_q], dtype=float)
    chain_edges = ([(norm_alpha[j], norm_alpha[j + 1]) for j in range(norb - 1)]
                   + [(norm_beta[j], norm_beta[j + 1]) for j in range(norb - 1)])
    gate_vals = np.array([engine.e_of(a, b) for a, b in chain_edges],
                         dtype=float)

    additive = float(readout_vals.sum() + gate_vals.sum())
    product = float(np.prod(np.clip(1.0 - readout_vals, 0.0, 1.0))
                    * np.prod(np.clip(1.0 - gate_vals, 0.0, 1.0)))

    for i, bphys in enumerate(kept):
        slot = 4 * i
        e1 = engine.e_of(norm_alpha[slot], bphys)
        e2 = engine.e_of(bphys, norm_beta[slot])
        r = engine.r_of(bphys)
        additive += r + e1 + e2
        product *= (max(0.0, 1.0 - r) * max(0.0, 1.0 - e1)
                    * max(0.0, 1.0 - e2))

    return {
        'layout': list(chain),
        'alpha': list(norm_alpha),
        'beta': list(norm_beta),
        'ancillas': kept,
        'bridge_slots': bridge_slots_for(k),
        'product_score': product,
        'additive_score': additive,
        'rank_in_stratum': 0,
        'source': 'guided',
        'chain_fs': frozenset(chain),
    }


# ── 1-opt local refinement ──────────────────────────────────────────

def _candidate_moves(engine, entry, k):
    """Every 1-opt neighbour of `entry`, as (label, new_layout_parts).

    Three move families. A move can bring in a coupler the template does not
    have, so `refine_entry` re-validates every one with `validate_layout`:

      bridge — swap a bridge for another common unused neighbour of the two
          chain qubits it joins;
      end    — swap a chain END qubit for another unused neighbour of its
          predecessor (both ends, both chains);
      shift  — move a whole chain one step along a parallel path.
    """
    norb = engine.norb
    alpha, beta = list(entry['alpha']), list(entry['beta'])
    bridges = list(entry['ancillas'])
    used = set(alpha) | set(beta) | set(bridges)
    adj = engine.adjacency
    out = []

    # (i) bridge replacement
    for i, bphys in enumerate(bridges):
        slot = 4 * i
        common = adj.get(alpha[slot], frozenset()) & adj.get(beta[slot],
                                                             frozenset())
        for r in sorted(common):
            if r in used or r == bphys:
                continue
            new_bridges = list(bridges)
            new_bridges[i] = r
            out.append(('bridge', alpha, beta, new_bridges))

    # (ii) chain-end replacement, both ends of both chains
    for chain_name, chain in (('alpha', alpha), ('beta', beta)):
        if norb < 2:
            break
        for end_idx, neighbour_idx in ((norb - 1, norb - 2), (0, 1)):
            # A head swap is only free when slot 0 carries no bridge; with a
            # bridge there the head is pinned by the bridge's two edges, and
            # the bridge move above already covers that region.
            if end_idx == 0 and k >= 1:
                continue
            anchor = chain[neighbour_idx]
            for q in sorted(adj.get(anchor, frozenset())):
                if q in used or q == chain[end_idx]:
                    continue
                new_chain = list(chain)
                new_chain[end_idx] = q
                if chain_name == 'alpha':
                    out.append(('end', new_chain, beta, bridges))
                else:
                    out.append(('end', alpha, new_chain, bridges))

    # (iii) whole-chain shift along a parallel path
    for chain_name, chain in (('alpha', alpha), ('beta', beta)):
        tail = chain[-1]
        for q in sorted(adj.get(tail, frozenset())):
            if q in used:
                continue
            shifted = chain[1:] + [q]
            if chain_name == 'alpha':
                out.append(('shift', shifted, beta, bridges))
            else:
                out.append(('shift', alpha, shifted, bridges))

    return out


def refine_entry(engine, entry, k, circuit_score_fn, max_iter=DEFAULT_MAX_ITER):
    """Hill-climb `entry` on LQS with 1-opt moves. Returns (entry, changed).

    Accepts a move only when the circuit-level score strictly improves, and
    only when the result is still a valid embedding. A scoring failure on a
    neighbour is treated as "no improvement" rather than an error, so a flaky
    scorer degrades the polish instead of breaking the sweep.
    """
    if circuit_score_fn is None:
        return entry, False

    current = entry
    best_score = current.get('lqs_circuit')
    if best_score is None:
        try:
            best_score = circuit_score_fn(current)
        except Exception:
            return entry, False
        current['lqs_circuit'] = best_score
    if best_score is None:
        return entry, False

    changed = False
    for _iteration in range(max_iter):
        improved = False
        for _kind, alpha, beta, bridges in _candidate_moves(engine, current, k):
            layout = list(alpha) + list(beta) + list(bridges)
            if not validate_layout(layout, engine.norb, k, engine.adjacency):
                continue
            cand = _entry_from_state(engine, alpha, beta, bridges, k)
            try:
                score = circuit_score_fn(cand)
            except Exception:
                continue
            if score is None or not (score > best_score):
                continue
            cand['lqs_circuit'] = score
            cand['source'] = current.get('source', 'guided')
            current, best_score = cand, score
            improved = changed = True
            break                              # first-improvement, restart
        if not improved:
            break
    return current, changed


def _layout_identity(entry):
    """One layout, however its chains are walked: all qubits + ancillas."""
    return (frozenset(entry['alpha']) | frozenset(entry['beta'])
            | frozenset(entry['ancillas']), frozenset(entry['ancillas']))


def _distinct_layouts(ranked):
    """`ranked` without repeated layouts, the first (best-ranked) copy kept."""
    seen, out = set(), []
    for entry in ranked:
        key = _layout_identity(entry)
        if key not in seen:
            seen.add(key)
            out.append(entry)
    return out


# ── public API ──────────────────────────────────────────────────────

def guided_sweep(norb, backend, props, *, min_bridges=None, top_n=3,
                 exclude_qubits=None, circuit_score_fn=None,
                 diversity_threshold=None, beam_width=DEFAULT_BEAM_WIDTH,
                 refine_top=DEFAULT_REFINE_TOP, qubit_weight=None,
                 max_iter=DEFAULT_MAX_ITER, seed=0) -> dict:
    """Calibration-guided beam search over LUCJ layouts.

    Returns the SAME dict shape as `ancilla_sweep.sweep_ancilla_layouts`, so
    every downstream consumer (strata display, best-per-k, diversity badges,
    notices, LQS re-scoring, benchmark re-ranking) works unchanged. The extra
    keys `engine`, `refined_count`, `beam_width` and `expansions` are additive.

    Args mirror the frozen sweep where they exist, plus:
        beam_width: partial states kept per growth step.
        refine_top: how many finalists per stratum get the 1-opt polish.
        qubit_weight: optional ``callable(q) -> float in (0, 1]`` folded into
            each qubit's fidelity factor — the hook for stability/T1/T2
            weighting in a later phase. Default None = no weighting.
        max_iter: 1-opt iteration ceiling.
        seed: reserved for deterministic tie-breaking; the search itself is
            already deterministic (sorted expansion, canonical tie-break).
    """
    started = time.time()

    full_graph = _make_backend_cmap_pygraph(backend=backend)
    excluded_sorted = sorted(int(q) for q in exclude_qubits) if exclude_qubits \
        else []
    if excluded_sorted:
        for i in full_graph.node_indices():
            full_graph[i] = i
        search_graph = _remove_nodes_from_graph(full_graph,
                                                set(excluded_sorted))
        phys = {i: search_graph[i] for i in search_graph.node_indices()}
    else:
        search_graph = full_graph
        phys = {i: i for i in search_graph.node_indices()}

    # Work in physical-qubit space throughout: rebuild the pruned graph with
    # physical ids as node indices so scores, adjacency and layouts all agree.
    work = rustworkx.PyGraph()
    id_map = {}
    for node in search_graph.node_indices():
        id_map[node] = work.add_node(phys[node])
    for u, v in search_graph.edge_list():
        work.add_edge(id_map[u], id_map[v], None)
    index_to_phys = {id_map[n]: phys[n] for n in search_graph.node_indices()}

    # k_max from the same template routine the frozen engine uses, so the two
    # agree on the geometric ceiling for this backend.
    _template, k_max_effective = create_lucj_zigzag_layout(
        num_orbitals=norb, backend_coupling_graph=search_graph,
        requested_bridges=-1)

    if min_bridges is None:
        min_bridges = min(3, k_max_effective)
    if not (1 <= min_bridges <= k_max_effective):
        raise ValueError(
            f"min_bridges={min_bridges} out of range; must satisfy "
            f"1 <= min_bridges <= k_max_effective ({k_max_effective}).")

    gate_name = _get_two_q_gate_name(backend)
    if gate_name is None:
        raise RuntimeError("Could not determine the backend 2Q gate name.")

    node_indices = list(full_graph.node_indices())
    max_id = max(node_indices) if node_indices else 0
    readout = np.full(max_id + 1, _UNCALIBRATED_PENALTY, dtype=float)
    for q in node_indices:
        readout[q] = _safe_readout(props, q)
    edge_err = {}
    for u, v in full_graph.edge_list():
        edge_err[frozenset((u, v))] = _safe_gate_error(props, gate_name, u, v)

    # The engine indexes adjacency by PHYSICAL id.
    phys_graph = rustworkx.PyGraph()
    seen_nodes = {}
    for idx in work.node_indices():
        q = index_to_phys[idx]
        seen_nodes[q] = q
    for q in sorted(seen_nodes):
        while len(phys_graph.node_indices()) <= q:
            phys_graph.add_node(None)
        phys_graph[q] = q
    for u, v in work.edge_list():
        phys_graph.add_edge(index_to_phys[u], index_to_phys[v], None)
    live = set(seen_nodes)

    engine = _GuidedEngine(norb, phys_graph, readout, edge_err, qubit_weight)
    # Restrict adjacency to live (non-excluded) qubits.
    engine.adjacency = {q: frozenset(n for n in phys_graph.neighbors(q)
                                     if n in live)
                        for q in live}

    strata_pool = {k: {} for k in range(k_max_effective, min_bridges - 1, -1)}
    per_stratum = {k: {'n_direct': 0, 'n_derived': 0, 'truncated': False,
                       'shuffled': False}
                   for k in range(k_max_effective, min_bridges - 1, -1)}
    n_candidates_scored = 0
    n_raw_embeddings = 0

    def _add(alpha, beta, bridges, k, source):
        entry = _entry_from_state(engine, alpha, beta, bridges, k)
        key = (entry['chain_fs'], frozenset(entry['ancillas']))
        if key in strata_pool[k]:
            return False
        entry['source'] = source
        strata_pool[k][key] = entry
        return True

    # ── generation: one beam at k_max, cascading down by the prefix
    # contract, plus a direct beam per lower k. The guided search does not
    # explode at low k the way VF2 enumeration does (the beam is bounded by
    # construction), so no cap is needed and every stratum gets a direct pass.
    for k_direct in range(k_max_effective, min_bridges - 1, -1):
        finals = engine.run_beam(k_direct, beam_width)
        n_raw_embeddings += len(finals)
        for state in finals:
            for k in range(k_direct, min_bridges - 1, -1):
                n_candidates_scored += 1
                source = 'guided' if k == k_direct else 'derived'
                if _add(state.alpha, state.beta, state.bridges, k, source):
                    bucket = 'n_direct' if k == k_direct else 'n_derived'
                    per_stratum[k][bucket] += 1

    if n_raw_embeddings == 0:
        raise RuntimeError(
            f"Guided search found no complete zigzag embedding for "
            f"norb={norb} on this backend.")

    # ── ranking, refinement, diversity — mirroring the frozen engine
    strata = {}
    higher_winners = []
    refined_count = 0
    refine_attempted = 0

    for k in range(k_max_effective, min_bridges - 1, -1):
        pool = strata_pool[k]
        by_product = sorted(pool.items(),
                            key=lambda kv: (-kv[1]['product_score'],
                                            tuple(kv[1]['layout'])))

        if circuit_score_fn is None:
            if diversity_threshold is None:
                entries = [e for _, e in by_product[:top_n]]
            else:
                entries = _select_with_diversity(
                    [e for _, e in by_product], top_n, diversity_threshold)
            for rank, entry in enumerate(entries, start=1):
                entry['rank_in_stratum'] = rank
            strata[k] = entries
            continue

        W = max(top_n * 5, 25)
        window_keys = [key for key, _ in by_product[:W]]
        seen = set(window_keys)
        for chain_fs, anc_full in higher_winners:
            if len(anc_full) < k:
                continue
            akey = (chain_fs, frozenset(anc_full[:k]))
            if akey in pool and akey not in seen:
                seen.add(akey)
                window_keys.append(akey)
        window = [pool[key] for key in window_keys]

        for entry in window:
            try:
                entry['lqs_circuit'] = circuit_score_fn(entry)
            except Exception as exc:
                _log.warning("Circuit scoring failed for a guided candidate "
                             "(k=%d): %s", k, exc)
                entry['lqs_circuit'] = None
        window.sort(
            key=lambda e: (e['lqs_circuit'] is not None,
                           e['lqs_circuit'] if e['lqs_circuit'] is not None
                           else 0.0,
                           e['product_score']),
            reverse=True)

        # 1-opt polish on the finalists, before the diversity slice so a
        # refined layout competes on its improved score.
        for pos in range(min(refine_top, len(window))):
            refine_attempted += 1
            refined, changed = refine_entry(engine, window[pos], k,
                                            circuit_score_fn, max_iter)
            if changed:
                refined['refined'] = True
                refined_count += 1
                window[pos] = refined
            else:
                window[pos].setdefault('refined', False)
        window.sort(
            key=lambda e: (e['lqs_circuit'] is not None,
                           e['lqs_circuit'] if e['lqs_circuit'] is not None
                           else 0.0,
                           e['product_score']),
            reverse=True)
        # A refined finalist can land on a layout another window entry
        # already holds (same qubits and ancillas, chains walked in another
        # order). Keep the higher-ranked copy so the freed slot goes to a
        # different layout.
        window = _distinct_layouts(window)

        if diversity_threshold is None:
            entries = window[:top_n]
        else:
            entries = _select_with_diversity(window, top_n,
                                             diversity_threshold)
        for rank, entry in enumerate(entries, start=1):
            entry['rank_in_stratum'] = rank
            entry.setdefault('refined', False)
        strata[k] = entries
        higher_winners.extend(
            (frozenset(e['layout']), tuple(e['ancillas'])) for e in entries)

    return {
        'strata': strata,
        'k_max_effective': k_max_effective,
        'min_bridges': min_bridges,
        'vf2_truncated': False,          # the beam is bounded, never truncated
        'n_raw_embeddings': n_raw_embeddings,
        'n_candidates_scored': n_candidates_scored,
        'excluded_qubits': excluded_sorted,
        'per_stratum': per_stratum,
        'mapping_cap': None,             # not a VF2 run; no cap applies
        'engine': 'guided',
        'beam_width': beam_width,
        'refined_count': refined_count,
        'refine_attempted': refine_attempted,
        'expansions': engine.expansions,
        'elapsed_s': time.time() - started,
    }
