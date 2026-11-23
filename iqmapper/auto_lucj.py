# Portions adapted from IBM code now incorporated in ffsim
# (ffsim.qiskit.generate_lucj_pass_manager),
# https://github.com/qiskit-community/ffsim
# (C) Copyright IBM.
#
# The adapted code is licensed under the Apache License, Version 2.0.
# You may obtain a copy of this license at
# http://www.apache.org/licenses/LICENSE-2.0.
#
# This file has been modified from the original IBM/ffsim code.
"""
auto_lucj.py — Automatic LUCJ zigzag layout on IBM heavy-hex backends.

Finds the optimal physical qubit placement for a LUCJ ansatz by:
  1. Building two linear chains (alpha/beta) + zigzag bridges
  2. Finding all isomorphic subgraph mappings on the backend coupling map
  3. Scoring each mapping by 2Q gate error + readout error
  4. Returning the best layout

Based on: Motta, Sung, Whaley, Head-Gordon, and Shee (2023)
"Bridging physical intuition and hardware efficiency for correlated
electronic states: the local unitary cluster Jastrow ansatz."

Dependencies: rustworkx (already a qiskit dependency).
"""

import logging
from typing import Sequence

import rustworkx
from rustworkx import NoEdgeBetweenNodes, PyGraph

_log = logging.getLogger(__name__)

IBM_TWO_Q_GATES = {"cx", "ecr", "cz"}


def _get_two_q_gate_name(backend):
    """Detect the 2Q gate name from backend basis gates.
    Returns gate name string or None if not found.
    """
    try:
        try:
            basis = backend.configuration().basis_gates
        except AttributeError:
            basis = list(backend.operation_names)
        matches = IBM_TWO_Q_GATES.intersection(basis)
        if not matches:
            _log.warning("No supported 2Q gate found in basis gates: %s", basis)
            return None
        return matches.pop()
    except Exception as exc:
        _log.warning("Could not determine 2Q gate: %s", exc)
        return None


# ═══════════════════════════════════════════════════════════════════════
# STEP 1: Build the two disconnected linear chains
# ═══════════════════════════════════════════════════════════════════════

def create_linear_chains(num_orbitals: int) -> PyGraph:
    """Create two disconnected linear chains of `num_orbitals` nodes each.

    Chain 1 (alpha): nodes 0 .. num_orbitals-1
    Chain 2 (beta):  nodes num_orbitals .. 2*num_orbitals-1
    """
    G = rustworkx.PyGraph()

    for n in range(num_orbitals):
        G.add_node(n)
    for n in range(num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    for n in range(num_orbitals, 2 * num_orbitals):
        G.add_node(n)
    for n in range(num_orbitals, 2 * num_orbitals - 1):
        G.add_edge(n, n + 1, None)

    return G


# ═══════════════════════════════════════════════════════════════════════
# STEP 2: Add zigzag bridges and verify subgraph isomorphism
# ═══════════════════════════════════════════════════════════════════════

def create_lucj_zigzag_layout(
    num_orbitals: int, backend_coupling_graph: PyGraph,
    requested_bridges: int = -1,
) -> tuple:
    """Create zigzag graph with bridges between alpha/beta chains.

    Bridges are added every 4 positions. If the resulting graph is not
    a subgraph of the backend coupling map, bridges are removed until
    it fits.

    Args:
        num_orbitals: Number of orbitals (alpha chain length).
        backend_coupling_graph: Backend topology as PyGraph.
        requested_bridges: -1 = auto (max then reduce), 0+ = exact number requested.
            If the exact count does not fit, falls back to the auto path
            (start from the maximum and reduce until isomorphic), which
            typically yields MORE bridges than requested, not fewer.

    Returns:
        (G_new, num_alpha_beta_qubits)

    Raises:
        RuntimeError: If even the bare two-chain graph cannot fit the backend.
    """
    isomorphic = False
    G = create_linear_chains(num_orbitals=num_orbitals)

    if requested_bridges >= 0:
        # User requested an exact count. The zigzag geometry only allows
        # bridges at positions n=0, 4, 8, ... so the absolute ceiling is
        # ceil(num_orbitals / 4). Without this cap, the original code
        # silently returned `min(requested, ceil(num_orbitals/4))` and
        # passed isomorphism — the user asked for 5 bridges with norb=4
        # and got 1, with no warning. Now we explicitly clamp and log.
        max_possible = (num_orbitals + 3) // 4
        effective = min(requested_bridges, max_possible)
        if effective < requested_bridges:
            _log.warning(
                "Requested %d bridges, but the zigzag geometry only "
                "allows %d for norb=%d (one bridge every 4 orbitals). "
                "Using %d.",
                requested_bridges, max_possible, num_orbitals, effective,
            )

        G_new = G.copy()
        num_alpha_beta_qubits = 0
        placed = 0
        for n in range(num_orbitals):
            if n % 4 == 0 and placed < effective:
                new_node = 2 * num_orbitals + num_alpha_beta_qubits
                G_new.add_node(new_node)
                G_new.add_edge(n, new_node, None)
                G_new.add_edge(new_node, n + num_orbitals, None)
                num_alpha_beta_qubits += 1
                placed += 1
        if rustworkx.is_subgraph_isomorphic(backend_coupling_graph, G_new):
            return G_new, num_alpha_beta_qubits
        # Even after the geometric clamp the layout did not fit on this
        # backend's topology. Fall through to the auto path below (same
        # code that runs when requested_bridges is left at its -1 default).
        # That path starts from num_orbitals bridges and reduces, so the
        # result is typically MORE bridges than the user asked for, not
        # fewer.
        _log.warning(
            "Requested %d bridges does not fit on this backend; falling "
            "back to auto (max-then-reduce). The result may use more "
            "than %d bridges.",
            effective, effective,
        )

    num_iters = num_orbitals
    while not isomorphic:
        if num_iters < 0:
            raise RuntimeError(
                f"Cannot fit LUCJ zigzag for norb={num_orbitals} "
                f"({2 * num_orbitals} qubits) on this backend topology. "
                f"The backend may not have a large enough connected subgraph."
            )
        G_new = G.copy()
        num_alpha_beta_qubits = 0
        for n in range(num_iters):
            if n % 4 == 0:
                new_node = 2 * num_orbitals + num_alpha_beta_qubits
                G_new.add_node(new_node)
                G_new.add_edge(n, new_node, None)
                G_new.add_edge(new_node, n + num_orbitals, None)
                num_alpha_beta_qubits += 1
        isomorphic = rustworkx.is_subgraph_isomorphic(
            backend_coupling_graph, G_new
        )
        num_iters -= 1

    return G_new, num_alpha_beta_qubits


# ═══════════════════════════════════════════════════════════════════════
# STEP 3: Score layouts by hardware error rates
# ═══════════════════════════════════════════════════════════════════════

def lightweight_layout_error_scoring(
    props,
    virtual_edges: Sequence,
    physical_layouts: Sequence,
    two_q_gate_name: str,
    t1_threshold_us: float = 50.0,
    t2_threshold_us: float = 30.0,
    coherence_penalty: float = 0.01,
    uncalibrated_penalty: float = 1.0,
) -> list:
    """Score each isomorphic layout by summing 2Q gate + readout + coherence errors.

    Args:
        props: BackendProperties object (already fetched, from state.initial_props
               or from data.py cache — NO API call here).
        virtual_edges: list of (u, v) edges in the zigzag graph.
        physical_layouts: list of layout mappings to score.
        two_q_gate_name: name of the 2Q gate ('cz', 'ecr', 'cx').
        t1_threshold_us: Qubits with T1 below this get penalized (microseconds).
        t2_threshold_us: Qubits with T2 below this get penalized (microseconds).
        coherence_penalty: Penalty added per qubit below threshold.
        uncalibrated_penalty: Finite score added per uncalibrated edge/qubit
                              (default 1.0 = maximum possible error, effectively
                              excludes the layout from the top candidates).

    Returns list of [layout, score] sorted ascending by score.

    NaN-safe: IBM BackendProperties.gate_error returns NaN for uncalibrated edges.
    Previously this poisoned the score (NaN + x = NaN) and sorted() became
    non-deterministic (IEEE 754: all NaN comparisons return False), sometimes
    placing the worst layout in rank 1. Now NaN values are replaced with
    uncalibrated_penalty and the count is logged at INFO level.

    Adapted from IBM code now incorporated in ffsim
    (ffsim.qiskit.generate_lucj_pass_manager). The cost function follows
    the mapomatic approach (Nation & Treinish, PRX Quantum 4, 010327,
    2023) — a first-order heuristic for relative layout
    ranking, NOT an absolute fidelity prediction. Known limitations: assumes
    independent errors (no crosstalk), each gate fires once per edge, ignores
    T2 decoherence over circuit duration. Suitable for ranking only.
    """
    if props is None:
        raise RuntimeError(
            "No calibration properties available for scoring. "
            "Disable 'Score layouts' or reconnect the backend."
        )

    import math  # stdlib, cheap

    scores = []
    total_uncalibrated = 0

    for layout in physical_layouts:
        total_2q_error = 0.0
        layout_uncalibrated = 0

        for edge in virtual_edges:
            physical_edge = (layout[edge[0]], layout[edge[1]])
            try:
                ge = props.gate_error(two_q_gate_name, physical_edge)
            except Exception:
                try:
                    ge = props.gate_error(two_q_gate_name, physical_edge[::-1])
                except Exception:
                    ge = float('nan')
            # NaN-safe: replace with penalty so sorted() stays deterministic
            if ge is None or (isinstance(ge, float) and math.isnan(ge)):
                ge = uncalibrated_penalty
                layout_uncalibrated += 1
            total_2q_error += ge

        total_measurement_error = 0.0
        coherence_error = 0.0
        for qubit in layout:
            try:
                meas_error = props.readout_error(qubit)
                if meas_error is None or (isinstance(meas_error, float)
                                          and math.isnan(meas_error)):
                    meas_error = uncalibrated_penalty
                total_measurement_error += meas_error
            except Exception:
                total_measurement_error += uncalibrated_penalty
            # T1/T2 coherence penalty
            try:
                t1 = props.t1(qubit) * 1e6  # seconds → microseconds
                if not math.isnan(t1) and t1 < t1_threshold_us:
                    coherence_error += coherence_penalty * (1 - t1 / t1_threshold_us)
            except Exception:
                _log.debug("T1 unavailable for qubit %s", qubit, exc_info=True)
            try:
                t2 = props.t2(qubit) * 1e6
                if not math.isnan(t2) and t2 < t2_threshold_us:
                    coherence_error += coherence_penalty * (1 - t2 / t2_threshold_us)
            except Exception:
                _log.debug("T2 unavailable for qubit %s", qubit, exc_info=True)

        score = total_2q_error + total_measurement_error + coherence_error
        # Defensive: even with the NaN guards above, don't let anything
        # unexpected poison sorted().
        if isinstance(score, float) and math.isnan(score):
            score = float('inf')
        scores.append([layout, float(score)])
        total_uncalibrated += layout_uncalibrated

    if total_uncalibrated > 0:
        _log.info(
            "Layout scoring: %d uncalibrated edges across %d layouts "
            "(penalty=%g applied per edge)",
            total_uncalibrated, len(physical_layouts), uncalibrated_penalty
        )

    return sorted(scores, key=lambda x: x[1])


# ═══════════════════════════════════════════════════════════════════════
# STEP 4: Build undirected backend coupling graph
# ═══════════════════════════════════════════════════════════════════════

def _make_backend_cmap_pygraph(backend) -> PyGraph:
    """Convert backend coupling map to an undirected PyGraph with no duplicate edges."""
    # .copy() first: backend.coupling_map.graph is a reference to the
    # backend's internal CouplingMap graph, so make_symmetric() would
    # mutate it in place (visible to transpile / later layout calls on
    # directional ECR backends). Operate on a private copy.
    graph = backend.coupling_map.graph.copy()
    if not graph.is_symmetric():
        graph.make_symmetric()
    backend_coupling_graph = graph.to_undirected()

    edge_list = backend_coupling_graph.edge_list()
    seen = set()
    for edge in edge_list:
        key = frozenset(edge)
        if key in seen:
            continue
        try:
            backend_coupling_graph.remove_edge(edge[0], edge[1])
            seen.add(key)
        except NoEdgeBetweenNodes:
            pass

    return backend_coupling_graph


# rustworkx.vf2_mapping is a generator that can yield tens of thousands
# of subgraph isomorphisms on large heavy-hex backends (Heron 156). The
# previous `list(generator)` materialized them all before scoring +
# dedup, peaking memory in the hundreds of MB and adding several
# seconds of wall time per call. This cap keeps the work bounded — the
# UI shows at most max_candidates=200 layouts after dedup, so even
# 5× headroom is generous.
_MAX_RAW_VF2_MAPPINGS = 2000


def _collect_vf2_mappings(graph, subgraph, cap=_MAX_RAW_VF2_MAPPINGS, *,
                          reservoir=None):
    """Collect at most `cap` isomorphic mappings from rustworkx.vf2_mapping.

    Returns the materialized list. Logs a warning when the cap was hit
    so we can tell the user (or future maintainers) that scoring may
    have missed better-ranked layouts beyond the cap.

    With `reservoir` (an `iqmapper.reservoir.DiverseReservoir`) the cap
    counts DISTINCT candidates instead of raw mappings: near-duplicates are
    offered, rejected and do not consume a slot. VF2 yields in graph
    traversal order, so a plain cap keeps one clustered corner of the chip;
    see reservoir.py for the measurement. `reservoir=None` is the historical
    path, byte-for-byte unchanged.
    """
    if reservoir is not None:
        for m in rustworkx.vf2_mapping(graph, subgraph, subgraph=True):
            reservoir.offer(m)
            # Stop on the OFFER budget, not on being full: the reservoir
            # fills before it sieves, so halting at `len >= cap` would make
            # it see nothing past the first `cap` mappings and behave
            # exactly like the plain cap it replaces.
            if reservoir.exhausted:
                _log.debug(
                    "VF2 cap reached: kept %d diverse mappings, skipped %d "
                    "near-duplicates", len(reservoir),
                    reservoir.stats['skipped_similar'])
                break
        return reservoir.mappings

    mappings = []
    truncated = False
    for m in rustworkx.vf2_mapping(graph, subgraph, subgraph=True):
        mappings.append(m)
        if len(mappings) >= cap:
            truncated = True
            break
    if truncated:
        _log.warning(
            "vf2_mapping yielded %d+ isomorphic subgraphs; truncating to "
            "%d to keep memory bounded. Layout scoring may not have seen "
            "every possible layout.",
            cap, cap,
        )
    return mappings


# ═══════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════

def get_all_zigzag_layouts(
    num_orbitals: int, backend, score: bool = True, props=None,
    requested_bridges: int = -1,
) -> tuple:
    """Generate ALL valid zigzag LUCJ layouts for a given backend.

    Args:
        num_orbitals: Number of spatial orbitals.
        backend: IBM Qiskit BackendV2 instance.
        score: If True, sort by error score (ascending). If False, unsorted.
        props: BackendProperties object (from state.initial_props or cache).
               If None and score=True, scoring is skipped with a warning.
        requested_bridges: -1 = auto, 0+ = exact number requested.

    Returns:
        (all_layouts, num_alpha_beta_qubits)
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    G, num_alpha_beta_qubits = create_lucj_zigzag_layout(
        num_orbitals=num_orbitals,
        backend_coupling_graph=backend_coupling_graph,
        requested_bridges=requested_bridges,
    )

    isomorphic_mappings = _collect_vf2_mappings(backend_coupling_graph, G)

    if not isomorphic_mappings:
        raise RuntimeError(
            f"No valid zigzag layout found for norb={num_orbitals} on this backend. "
            f"The backend may not have enough connected qubits."
        )

    edges = list(G.edge_list())

    # Each VF2 mapping should cover every slot 0..(2*norb + num_ab - 1)
    # exactly once. A partial mapping would leave `None` entries in the
    # layout list; downstream code (e.g. `layout[edge[0]]` in lightweight_layout_error_scoring,
    # scoring, the swap heuristic in `_deduplicate_layouts`) treats None
    # as a physical qubit index and crashes with TypeError. Validate
    # completeness up-front and skip any malformed mapping with a debug
    # log rather than letting the failure surface as an unrelated error
    # 200 lines deeper.
    expected_size = 2 * num_orbitals + num_alpha_beta_qubits
    raw_layouts = []
    for mapping in isomorphic_mappings:
        initial_layout = [None] * expected_size
        for key, value in mapping.items():
            initial_layout[value] = key
        if any(v is None for v in initial_layout):
            _log.debug(
                "Skipping incomplete VF2 mapping (expected %d slots, "
                "got %d): %r",
                expected_size, sum(v is not None for v in initial_layout),
                mapping)
            continue
        raw_layouts.append(initial_layout)

    # Score if requested
    scores_list = None
    if score:
        two_q_gate = _get_two_q_gate_name(backend)
        if two_q_gate and props is not None:
            try:
                scores_list = lightweight_layout_error_scoring(
                    props=props,
                    virtual_edges=edges,
                    physical_layouts=raw_layouts,
                    two_q_gate_name=two_q_gate,
                )
            except Exception as exc:
                _log.warning("Layout scoring failed (%s), returning unsorted", exc)
        elif props is None:
            _log.warning("No props provided — scoring skipped (pass props= to enable)")

    # Build result list
    all_layouts = _build_candidates(
        raw_layouts, scores_list, num_orbitals, num_alpha_beta_qubits)

    # Deduplicate: remove layouts that are alpha/beta swaps of each other
    all_layouts = _deduplicate_layouts(all_layouts)

    return all_layouts, num_alpha_beta_qubits


def _build_candidates(raw_layouts, scores_list, num_orbitals, num_bridges,
                      include_full=False):
    """Build candidate list from raw layouts and optional scores.

    Shared by get_all_zigzag_layouts and get_multi_circuit_layouts.
    """
    candidates = []
    if scores_list is not None:
        for rank, (layout_full, err_score) in enumerate(scores_list):
            lt = layout_full[:-num_bridges] if num_bridges > 0 else layout_full
            entry = {
                'layout': lt,
                'alpha': lt[:num_orbitals],
                'beta': lt[num_orbitals:],
                'score': err_score,
                'rank': rank + 1,
            }
            if include_full:
                entry['layout_full'] = layout_full
            candidates.append(entry)
    else:
        for rank, layout_full in enumerate(raw_layouts):
            lt = layout_full[:-num_bridges] if num_bridges > 0 else layout_full
            entry = {
                'layout': lt,
                'alpha': lt[:num_orbitals],
                'beta': lt[num_orbitals:],
                'score': None,
                'rank': rank + 1,
            }
            if include_full:
                entry['layout_full'] = layout_full
            candidates.append(entry)
    return candidates


def _deduplicate_layouts(layouts):
    """Remove duplicate layouts where alpha and beta are just swapped.

    Two layouts are equivalent if set(a['alpha']) == set(b['beta']) and
    set(a['beta']) == set(b['alpha']). We normalize by putting the set
    with the smallest min-element as alpha.

    Also removes exact duplicates (same qubit set, same assignment).
    """
    seen = set()
    deduped = []
    for entry in layouts:
        alpha_set = frozenset(entry['alpha'])
        beta_set = frozenset(entry['beta'])

        # Canonical key: deterministic order by smallest element in each set
        # (frozenset min/max uses subset ordering — broken for disjoint sets)
        if alpha_set and beta_set:
            if min(alpha_set) <= min(beta_set):
                canon = (alpha_set, beta_set)
            else:
                canon = (beta_set, alpha_set)
        else:
            canon = (alpha_set, beta_set)

        if canon in seen:
            continue
        seen.add(canon)

        # Normalize the entry itself: alpha gets the smaller-first-element chain
        if entry['alpha'] and entry['beta'] and entry['alpha'][0] > entry['beta'][0]:
            entry['alpha'], entry['beta'] = entry['beta'], entry['alpha']
            entry['layout'] = list(entry['alpha']) + list(entry['beta'])
            # The multi-circuit code path also carries `layout_full` —
            # the alpha+beta chain with the bridge qubits appended at
            # the end. After the swap, the bridge tail stays where it
            # was (bridges are tied to chain *positions*, not to the
            # alpha/beta label), so we rebuild `layout_full` as
            # `new_alpha + new_beta + old_bridges`. Without this, a
            # downstream call to reoptimize_after_change would pick up
            # the locked qubits from `cands[sel]['layout_full']` in the
            # OLD α/β order and produce inconsistent re-placements
            # against the user's visible swap.
            if 'layout_full' in entry:
                full = entry['layout_full']
                n_ab = len(entry['alpha']) + len(entry['beta'])
                bridges = list(full[n_ab:])
                entry['layout_full'] = (
                    list(entry['alpha']) + list(entry['beta']) + bridges)

        deduped.append(entry)

    # Re-rank
    for i, entry in enumerate(deduped):
        entry['rank'] = i + 1

    _log.info("Deduplication: %d -> %d layouts", len(layouts), len(deduped))
    return deduped


def get_zigzag_physical_layout(
    num_orbitals: int, backend, score_layouts: bool = True, props=None
) -> tuple:
    """Generate the optimal zigzag LUCJ layout for a given backend.

    Convenience wrapper around get_all_zigzag_layouts that returns only
    the best layout.

    Returns:
        (layout, num_alpha_beta_qubits) where layout is a list of
        2*num_orbitals physical qubit indices.
    """
    all_layouts, num_ab = get_all_zigzag_layouts(
        num_orbitals, backend, score=score_layouts, props=props)
    return all_layouts[0]['layout'], num_ab


# ═══════════════════════════════════════════════════════════════════════
# MULTI-CIRCUIT (INTRA-QPU) LAYOUT
# ═══════════════════════════════════════════════════════════════════════

def _remove_nodes_from_graph(graph: PyGraph, nodes_to_remove: set) -> PyGraph:
    """Create a copy of graph with specified nodes removed.

    Each node in the graph stores its original physical qubit index as data.
    This function preserves that mapping through rebuilds.

    NOTE on node indices: `set(range(len(graph)))` assumes node indices are
    densely packed 0..N-1. PyGraph guarantees this only on graphs built by
    sequential add_node calls; after any prior `remove_node` (or future
    refactor that uses it) indices become sparse and the `range(len)` set
    would silently exclude valid survivors. `graph.node_indices()` is the
    authoritative iterator regardless of density.
    """
    surviving = sorted(set(graph.node_indices()) - nodes_to_remove)
    if not surviving:
        return rustworkx.PyGraph()

    new_graph = rustworkx.PyGraph()
    old_to_new = {}
    for old_idx in surviving:
        phys_qubit = graph[old_idx]
        new_idx = new_graph.add_node(phys_qubit)
        old_to_new[old_idx] = new_idx

    for edge in graph.edge_list():
        u, v = edge
        if u in old_to_new and v in old_to_new:
            new_graph.add_edge(old_to_new[u], old_to_new[v], None)

    return new_graph


def _exclude_with_buffer(remaining, phys_qubits, buffer_qubits):
    """Compute qubits to exclude: phys_qubits + buffer hops on the graph.
    Returns (remaining_graph_after_removal, excluded_set).
    """
    exclude = set(phys_qubits)
    frontier = set(phys_qubits)
    for _ in range(buffer_qubits):
        next_frontier = set()
        for node_idx in remaining.node_indices():
            if remaining[node_idx] in frontier:
                for neighbor_idx in remaining.neighbors(node_idx):
                    neighbor_phys = remaining[neighbor_idx]
                    if neighbor_phys not in exclude:
                        next_frontier.add(neighbor_phys)
        exclude.update(next_frontier)
        frontier = next_frontier

    nodes_to_remove = set()
    for node_idx in remaining.node_indices():
        if remaining[node_idx] in exclude:
            nodes_to_remove.add(node_idx)

    return _remove_nodes_from_graph(remaining, nodes_to_remove), exclude


def get_multi_circuit_layouts(
    norb_list: list,
    backend,
    score: bool = True,
    buffer_qubits: int = 1,
    max_candidates: int = 200,
    props=None,
    requested_bridges: int = -1,
    requested_bridges_list: list = None,
) -> list:
    """Generate zigzag LUCJ layouts for multiple circuits on the same QPU.

    Strategy:
      1. For each circuit (in order), find ALL valid zigzag layouts on the
         remaining available qubits.
      2. Select the best, remove its qubits + buffer, repeat.
      3. Return all candidates per circuit for user inspection.

    Args:
        norb_list: List of norb values, one per sub-circuit.
        backend: IBM Qiskit BackendV2 instance.
        score: If True, sort candidates by error score.
        buffer_qubits: Idle-qubit hops between circuits.
        max_candidates: Cap per-circuit candidates to avoid memory issues.

    Returns:
        List of dicts, one per circuit:
        [
            {
                'circuit_idx': 0,
                'norb': 8,
                'selected': 0,              # index into all_candidates
                'num_bridges': 2,
                'all_candidates': [          # deduplicated, scored
                    {'layout': [...], 'alpha': [...], 'beta': [...],
                     'score': 0.034, 'rank': 1},
                    ...
                ],
            },
            ...
        ]
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    remaining = backend_coupling_graph.copy()
    for i in range(len(remaining)):
        remaining[i] = i

    results = []

    for circ_idx, norb in enumerate(norb_list):
        n_qubits_needed = 2 * norb
        real_nodes = list(remaining.node_indices())

        if len(real_nodes) < n_qubits_needed:
            raise RuntimeError(
                f"Circuit {circ_idx + 1} (norb={norb}) needs {n_qubits_needed} qubits "
                f"but only {len(real_nodes)} remain."
            )

        # Find zigzag layout on remaining graph
        try:
            rb = (requested_bridges_list[circ_idx]
                  if requested_bridges_list and circ_idx < len(requested_bridges_list)
                  else requested_bridges)
            G, num_ab = create_lucj_zigzag_layout(
                num_orbitals=norb,
                backend_coupling_graph=remaining,
                requested_bridges=rb,
            )
        except Exception:
            raise RuntimeError(
                f"Circuit {circ_idx + 1} (norb={norb}): no valid zigzag fits "
                f"in remaining {len(real_nodes)} qubits."
            )

        # Find isomorphic mappings (capped — see _collect_vf2_mappings)
        iso_mappings = _collect_vf2_mappings(remaining, G)
        if not iso_mappings:
            raise RuntimeError(
                f"Circuit {circ_idx + 1} (norb={norb}): no isomorphic subgraph found."
            )

        edges = list(G.edge_list())

        # Build layouts with original physical qubit indices. See the
        # completeness-check rationale in `get_all_zigzag_layouts`:
        # incomplete VF2 mappings leave `None` slots that crash
        # downstream as TypeError. Skip malformed mappings explicitly.
        expected_size = 2 * norb + num_ab
        layouts_physical = []
        for mapping in iso_mappings:
            layout_raw = [None] * expected_size
            for rem_idx, zig_idx in mapping.items():
                original_phys = remaining[rem_idx]
                layout_raw[zig_idx] = original_phys
            if any(v is None for v in layout_raw):
                _log.debug(
                    "Skipping incomplete multi-circuit mapping "
                    "(expected %d, got %d)",
                    expected_size, sum(v is not None for v in layout_raw))
                continue
            layouts_physical.append(layout_raw)

        # Score
        scored_list = None
        if score:
            two_q_gate = _get_two_q_gate_name(backend)
            if two_q_gate and props is not None:
                try:
                    scored_list = lightweight_layout_error_scoring(
                        props=props,
                        virtual_edges=edges,
                        physical_layouts=layouts_physical,
                        two_q_gate_name=two_q_gate,
                    )
                except Exception as exc:
                    _log.warning("Scoring failed for circuit %d: %s", circ_idx + 1, exc)

        # Build candidates list
        candidates = _build_candidates(
            layouts_physical, scored_list, norb, num_ab, include_full=True)

        # Deduplicate alpha/beta swaps
        candidates = _deduplicate_layouts(candidates)

        # Cap
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        results.append({
            'circuit_idx': circ_idx,
            'norb': norb,
            'selected': 0,
            'num_bridges': num_ab,
            'all_candidates': candidates,
        })

        # Remove best layout's qubits + buffer from remaining graph
        best_full = candidates[0].get('layout_full', candidates[0]['layout'])
        remaining, exclude = _exclude_with_buffer(
            remaining, set(best_full), buffer_qubits)

        _log.info(
            "Circuit %d: norb=%d, %d candidates (deduped), "
            "%d qubits excluded, %d remaining",
            circ_idx + 1, norb, len(candidates),
            len(exclude), len(remaining),
        )

    return results


# ═══════════════════════════════════════════════════════════════════════
# JOINT RE-OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════

def reoptimize_after_change(
    backend,
    norb_list: list,
    locked_qubits: dict,
    changed_idx: int,
    buffer_qubits: int = 1,
    max_candidates: int = 200,
    score: bool = True,
    props=None,
    requested_bridges: int = -1,
    requested_bridges_list: list = None,
) -> list:
    """Re-optimize circuits after the user changes one circuit's layout.

    When the user selects a different layout for circuit `changed_idx`,
    all circuits AFTER it (changed_idx+1, changed_idx+2, ...) need to be
    re-placed on the remaining qubits.

    Args:
        backend: IBM BackendV2.
        norb_list: Full list of norb values for all circuits.
        locked_qubits: Dict mapping circuit_idx → set of physical qubits
                       for circuits 0..changed_idx (the locked selections).
        changed_idx: The circuit that was just changed by the user.
        buffer_qubits: Idle-qubit hops between circuits.
        max_candidates: Cap per-circuit candidates.
        score: Whether to sort by error score.
        props: BackendProperties for scoring.

    Returns:
        List of result dicts for circuits changed_idx+1 onwards only.
        Same format as get_multi_circuit_layouts results.
    """
    backend_coupling_graph = _make_backend_cmap_pygraph(backend=backend)

    # Start with full graph, label nodes with physical qubit IDs
    remaining = backend_coupling_graph.copy()
    for i in range(len(remaining)):
        remaining[i] = i

    # Remove all locked circuits (0..changed_idx) from the graph
    for cidx in range(changed_idx + 1):
        phys = locked_qubits.get(cidx, set())
        if phys:
            remaining, _ = _exclude_with_buffer(remaining, phys, buffer_qubits)

    _log.info("Re-optimize: %d qubits remaining after excluding circuits 0..%d",
              len(remaining), changed_idx)

    # Now place circuits changed_idx+1 onwards on what's left
    results = []
    for circ_idx in range(changed_idx + 1, len(norb_list)):
        norb = norb_list[circ_idx]
        real_nodes = list(remaining.node_indices())

        if len(real_nodes) < 2 * norb:
            raise RuntimeError(
                f"Circuit {circ_idx + 1} (norb={norb}) needs {2 * norb} qubits "
                f"but only {len(real_nodes)} remain after re-optimization."
            )

        try:
            rb = (requested_bridges_list[circ_idx]
                  if requested_bridges_list and circ_idx < len(requested_bridges_list)
                  else requested_bridges)
            G, num_ab = create_lucj_zigzag_layout(
                num_orbitals=norb,
                backend_coupling_graph=remaining,
                requested_bridges=rb,
            )
        except Exception:
            raise RuntimeError(
                f"Circuit {circ_idx + 1} (norb={norb}): no valid zigzag fits "
                f"in remaining {len(real_nodes)} qubits after re-optimization."
            )

        edges = list(G.edge_list())
        isomorphic = _collect_vf2_mappings(remaining, G)

        if not isomorphic:
            raise RuntimeError(
                f"Circuit {circ_idx + 1}: no isomorphic mapping found "
                f"on remaining {len(real_nodes)} qubits."
            )

        # Map back to physical qubit IDs. See `get_all_zigzag_layouts`
        # for why incomplete mappings must be filtered (`None` slots
        # crash downstream as TypeError on layout[edge[0]] / scoring).
        expected_size = 2 * norb + num_ab
        layouts_physical = []
        for mapping in isomorphic:
            layout_raw = [None] * expected_size
            for key, value in mapping.items():
                phys_qubit = remaining[key]
                layout_raw[value] = phys_qubit
            if any(v is None for v in layout_raw):
                _log.debug(
                    "Skipping incomplete reopt mapping "
                    "(expected %d, got %d)",
                    expected_size, sum(v is not None for v in layout_raw))
                continue
            layouts_physical.append(layout_raw)

        # Score
        scored_list = None
        if score:
            two_q_gate = _get_two_q_gate_name(backend)
            if two_q_gate and props is not None:
                try:
                    scored_list = lightweight_layout_error_scoring(
                        props=props,
                        virtual_edges=edges,
                        physical_layouts=layouts_physical,
                        two_q_gate_name=two_q_gate,
                    )
                except Exception as exc:
                    _log.warning("Re-opt scoring failed for circuit %d: %s",
                                 circ_idx + 1, exc)

        candidates = _build_candidates(
            layouts_physical, scored_list, norb, num_ab, include_full=True)
        candidates = _deduplicate_layouts(candidates)
        if len(candidates) > max_candidates:
            candidates = candidates[:max_candidates]

        results.append({
            'circuit_idx': circ_idx,
            'norb': norb,
            'selected': 0,
            'num_bridges': num_ab,
            'all_candidates': candidates,
        })

        # Exclude best for next circuit
        best_full = candidates[0].get('layout_full', candidates[0]['layout'])
        remaining, _ = _exclude_with_buffer(
            remaining, set(best_full), buffer_qubits)

    return results