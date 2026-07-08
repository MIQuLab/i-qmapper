"""
export.data — Build the JSON-serialisable export payload for the
current state (LUCJ block / general block) and the ancilla-index
resolver shared by both code paths.
"""
import logging


_log = logging.getLogger(__name__)


def _safe_export_ids(id_list):
    """Convert qubit IDs for export.

    - Pure integer strings ("42") → int(42)
    - Non-numeric strings ("3,5", "q0") → kept as-is
    - Logs a warning if any non-numeric IDs are found so the user
      knows the export contains mixed types.
    """
    result = []
    non_numeric = []
    for x in sorted(set(id_list)):
        s = str(x)
        if s.lstrip('-').isdigit():
            result.append(int(s))
        else:
            result.append(s)
            non_numeric.append(s)
    if non_numeric:
        _log.warning(
            "Export contains non-numeric qubit IDs: %s. "
            "These are kept as strings in the JSON output. "
            "If you expected only integers, check your qubit ID format.",
            non_numeric[:10])
    return sorted(result, key=lambda x: (isinstance(x, str), x))


def _compute_ancilla_indices(state, alpha, beta, ancilla):
    """Label each ancilla with its LUCJ virtual-orbital indices (p, p).

    Convention used:
        indices[k] = [k * LUCJ_STEP, k * LUCJ_STEP]

    where k is the ancilla's position in the user-supplied ancilla list
    and LUCJ_STEP = 4 (the standard spacing in
    `alpha_beta_indices = [(p, p) for p in range(0, norb, 4)]`).

    Rationale: LUCJ virtual indices are a convention of the ansatz, not
    something we can reliably derive from the chip topology. The user
    decides the ansatz structure; the tool just labels the ancillas the
    user placed in the order they were placed. The topological role of
    the tool is to *verify* that each ancilla actually sits between one
    alpha qubit and one beta qubit on the chip (the minimum sanity
    check). Mismatches become warnings; the indices are still assigned
    so the user can continue editing without the export breaking.

    Earlier chain-walk approach failed on realistic LUCJ layouts: alpha/
    beta lists on heavy-hex chips are not a single simple path (numerical
    gaps split the induced subgraph into components), so the walk fell
    back to list-position indexing. Ordinal labelling here is topology-
    independent and always emits (0,0), (4,4), (8,8), ...

    Args:
        state:   SessionState (uses .edges_list and .qubit_ids for the
                 topology sanity check).
        alpha:   iterable of alpha qubit IDs (order does not affect the
                 indices assigned — only the sanity check uses alpha /
                 beta membership to confirm ancilla placement).
        beta:    iterable of beta qubit IDs.
        ancilla: iterable of ancilla qubit IDs. The ORDER of this list
                 determines the (p, p) labels: first ancilla -> (0, 0),
                 second -> (4, 4), and so on.

    Returns:
        (indices, warnings)
        - indices: list of [p, p] pairs, one per ancilla, in the same
                   order as the input.
        - warnings: list of English-language strings describing any
                    ancilla whose chip neighbors do not include exactly
                    one alpha qubit and exactly one beta qubit. Empty
                    list means the layout is topologically consistent.
    """
    LUCJ_STEP = 4  # Standard LUCJ ansatz spacing: range(0, norb, 4).

    alpha_set = set(str(q) for q in alpha)
    beta_set = set(str(q) for q in beta)

    # Build chip adjacency map for the sanity check.
    q_ids = state.qubit_ids
    adj = {}
    for edge in state.edges_list:
        u, v = edge[0], edge[1]
        if u < 0 or v < 0 or u >= len(q_ids) or v >= len(q_ids):
            continue
        u_id, v_id = q_ids[u], q_ids[v]
        adj.setdefault(u_id, set()).add(v_id)
        adj.setdefault(v_id, set()).add(u_id)

    indices = []
    warnings_list = []

    for k, anc in enumerate(ancilla):
        anc_str = str(anc)
        p = k * LUCJ_STEP
        neighbors = adj.get(anc_str, set())
        alpha_nbrs = [n for n in neighbors if n in alpha_set]
        beta_nbrs = [n for n in neighbors if n in beta_set]

        # Topology sanity check. A valid LUCJ ancilla touches exactly one
        # qubit of each chain on the chip. Anything else means the user
        # placed the ancilla in a location that doesn't match LUCJ's
        # bridge-qubit role.
        if len(alpha_nbrs) != 1 or len(beta_nbrs) != 1:
            warnings_list.append(
                f"Ancilla {anc_str} (position {k} in list, LUCJ p={p}): "
                f"expected exactly 1 alpha-chain neighbor and 1 beta-chain "
                f"neighbor on the chip; found {len(alpha_nbrs)} alpha and "
                f"{len(beta_nbrs)} beta. The ancilla may be placed outside "
                f"a bridge position — check the layout."
            )
        # Always emit the (p, p) label so export stays usable even when
        # a single ancilla is suspicious. The warning above is the signal.
        indices.append([p, p])

    return indices, warnings_list


_MERGED_FORMAT = 'Merged (Single List)'


def _build_lucj_block(state, d, anc, export_format):
    """Build the exported JSON block for one LUCJ circuit slot.

    Returns `(block, warnings)` where `block` is the dict that will land
    under `final_data[d['name']]` and `warnings` is the per-circuit
    ancilla-validity message list (empty when the layout passes the
    chip-topology sanity check).
    """
    a = _safe_export_ids(d['alpha'])
    b_l = _safe_export_ids(d['beta'])
    o = _safe_export_ids(d['other'])

    # Compute LUCJ virtual indices for each ancilla. The tool knows the
    # chip topology and the user's alpha/beta ordering, so it can figure
    # out which virtual orbital each ancilla sits next to. Valid LUCJ
    # => every ancilla has i == j. Mismatches surface as warnings.
    ancilla_indices, anc_warnings = [], []
    if anc:
        # Use `anc` (the sorted export-id list that becomes block["ancilla"])
        # so _lucj_ancilla_indices[k] lines up with block["ancilla"][k]. The
        # (p, p) labels are ordinal by convention; sorted order is both
        # consistent with the exported list and the more intuitive labelling.
        ancilla_indices, anc_warnings = _compute_ancilla_indices(
            state, d['alpha'], d['beta'], anc
        )

    if export_format == _MERGED_FORMAT:
        # Merged carries alpha + beta only, in α→β order (each group
        # already sorted within itself by _safe_export_ids above).
        # 'other' and 'ancilla' live in their own top-level fields so
        # the α/β order in 'qubits' is never reshuffled by their IDs.
        merged = a + b_l
        block = {
            "qubits": merged,
            "_lucj_partition": {'alpha_len': len(a), 'beta_len': len(b_l)},
        }
        if o:
            block["other"] = o
    else:
        block = {"alpha": a, "beta": b_l, "other": o, "ancilla": anc}

    if anc:
        # Split format already carries ancilla under its key; merged
        # adds it here. _lucj_ancilla_indices is attached in both shapes
        # since the reader uses it regardless of split-vs-merged.
        if export_format == _MERGED_FORMAT:
            block["ancilla"] = anc
        block["_lucj_ancilla_indices"] = ancilla_indices

    return block, anc_warnings


def _build_general_block(d, anc, export_format):
    """Build the exported JSON block for a general-ansatz circuit slot.

    No LUCJ partition / ancilla-index logic — just sorted qubit IDs and
    the optional ancilla list. Returns the block dict (or, in the
    legacy merged-no-ancilla shape, the bare list).
    """
    qbs = _safe_export_ids(d['qubits'])
    if export_format == _MERGED_FORMAT:
        merged = _safe_export_ids(list(d['qubits']))
        if anc:
            return {"qubits": merged, "ancilla": anc}
        # Legacy merged-no-ancilla shape: the value is the bare list, not
        # a dict. Preserved so old files that consume this format still parse.
        return merged
    return {"qubits": qbs, "ancilla": anc}


def get_export_data(state, mode, export_format, n_circ):
    """
    Build the export dict from current layouts_data.

    Args:
        state: SessionState
        mode: 'lucj' or 'general'
        export_format: 'Split (Alpha/Beta/...)' or 'Merged (Single List)'
        n_circ: number of active circuits

    Returns:
        dict ready for JSON serialization.
    """
    final_data = {}
    all_warnings = []  # aggregated per-circuit warnings for LUCJ ancilla validity
    for i in range(n_circ):
        d = state.layouts_data[i]
        anc = _safe_export_ids(d['ancilla'])

        if mode == 'lucj':
            block, anc_warnings = _build_lucj_block(state, d, anc, export_format)
            if anc_warnings:
                all_warnings.extend(
                    f"[{d['name']}] {w}" for w in anc_warnings
                )
            final_data[d['name']] = block
        else:
            final_data[d['name']] = _build_general_block(d, anc, export_format)

    # Add calibration metadata
    final_data['_metadata'] = {
        'backend': getattr(state, 'active_backend_name', None),
        'chip': getattr(state, 'active_chip', None),
        'calibration': getattr(state, 'calibration_data_label', None),
    }
    # Surface LUCJ ancilla-validity warnings at the top level so the reader
    # sees them immediately instead of having to cross-check indices by eye.
    # Empty / absent => layout is LUCJ-valid.
    if all_warnings:
        final_data['_warnings'] = all_warnings

    return final_data
