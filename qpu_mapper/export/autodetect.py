"""
export.autodetect — Split a flat qubit list into LUCJ alpha/beta
partitions when possible, and parse the textual `parse_auto_text`
payload (used by the Auto-map UI tab).

NOTE: when the LUCJ VF2 recovery functions land, they belong here too —
same domain, shared graph-component plumbing.
"""
import contextlib
import logging

from ..constants import MAX_CIRCUITS
from .layout_io import apply_layout_to_circuit

_log = logging.getLogger(__name__)


def _autodetect_lucj_partition(qubits, state):
    """Attempt to split a flat qubit list into alpha/beta by connected components.

    Strategy: build the subgraph induced by the selected qubits (only the
    chip edges that connect two qubits both present in the list). If this
    subgraph has exactly TWO connected components, return them as (alpha,
    beta). The component containing the smallest physical qubit ID becomes
    alpha — this matches the intuitive "topmost / leftmost chain is alpha"
    convention and is deterministic.

    When this works: the user pasted a list whose alpha and beta chains
    are topologically disjoint on the chip (no α-β CZ bridge qubits in
    the selection). This is the common case for circuits extracted from
    `transpiled.initial_layout` in ascending order — the original virtual
    index info is lost by the sort, but the chip topology lets us recover
    the partition.

    When this does NOT work: the selection contains one or more α-β bridge
    edges (induced subgraph is ONE component), or the selection has
    isolated qubits and fragments (>2 components). In those cases the
    caller should fall back to the midpoint split, which is the behavior
    users had before this autodetect existed.

    Args:
        qubits:  iterable of qubit IDs (int or str) — the full user list.
        state:   SessionState with .qubit_ids and .edges_list populated.

    Returns:
        (alpha, beta, ok) where alpha and beta are lists of string qubit IDs
        preserving the order they appeared in the input. `ok` is True iff
        CC gave exactly 2 components. When False, alpha and beta are left
        empty and the caller must use the midpoint fallback.
    """
    q_ids_list = state.qubit_ids
    if not q_ids_list:
        return [], [], False

    q_id_to_idx = {q_id: i for i, q_id in enumerate(q_ids_list)}

    # Keep only valid qubits and preserve their input order.
    valid_ordered = []
    valid_indices = set()
    for q in qubits:
        s = str(q)
        if s in q_id_to_idx:
            valid_ordered.append(s)
            valid_indices.add(q_id_to_idx[s])

    if len(valid_ordered) < 2:
        return [], [], False  # too small to meaningfully partition

    # Build adjacency for the induced subgraph. state.edges_list is a list
    # of (u, v) index pairs into state.qubit_ids.
    adj = {i: [] for i in valid_indices}
    for edge in state.edges_list:
        u, v = edge[0], edge[1]
        if u in valid_indices and v in valid_indices:
            adj[u].append(v)
            adj[v].append(u)

    # BFS to tag connected components. `component_of` maps qubit index ->
    # component id (0, 1, 2, ...).
    visited = set()
    component_of = {}
    comp_id = 0
    # Iterate in deterministic order (by qubit index) so the component
    # labelling is reproducible across runs.
    for start in sorted(valid_indices):
        if start in visited:
            continue
        # Iterative DFS to avoid stack depth issues on long chains.
        stack = [start]
        while stack:
            n = stack.pop()
            if n in visited:
                continue
            visited.add(n)
            component_of[n] = comp_id
            for nb in adj[n]:
                if nb not in visited:
                    stack.append(nb)
        comp_id += 1

    n_components = comp_id
    if n_components != 2:
        return [], [], False

    # Assign input qubits to their component, preserving input order within
    # each group.
    group0, group1 = [], []
    for q_str in valid_ordered:
        idx = q_id_to_idx[q_str]
        if component_of[idx] == 0:
            group0.append(q_str)
        else:
            group1.append(q_str)

    # Pick which component becomes alpha. Convention: component containing
    # the smallest integer qubit ID. Falls back to discovery order (group0)
    # if qubit IDs are not purely integer.
    def _min_int_id(group):
        ints = []
        for q in group:
            # Non-integer qubit IDs are skipped by design — caller then
            # falls back to discovery order (see comment above).
            with contextlib.suppress(ValueError):
                ints.append(int(q))
        return min(ints) if ints else None

    min0 = _min_int_id(group0)
    min1 = _min_int_id(group1)
    if min0 is not None and min1 is not None:
        if min0 <= min1:
            alpha, beta = group0, group1
        else:
            alpha, beta = group1, group0
    else:
        alpha, beta = group0, group1

    return alpha, beta, True


def parse_auto_text(state, text, mode):
    """Parse the Load & Map textarea into circuit layouts.

    One line per circuit. Each line is a comma-separated list of qubit IDs.

    In 'lucj' mode the tool first tries to auto-detect alpha / beta by
    running connected-components on the subgraph induced by the selected
    qubits (see _autodetect_lucj_partition). If the induced subgraph has
    exactly two components, those become alpha and beta, with the
    component containing the smallest qubit ID labelled alpha. Otherwise
    (one component = there is an alpha-beta bridge in the selection; or
    three+ components = ambiguous) the code falls back to the original
    midpoint split: first half of the list becomes alpha, second half
    becomes beta. The fallback is harmless because it matches the
    pre-autodetect behavior.

    In 'general' mode the whole line goes to 'qubits' and autodetect is
    not invoked (no alpha/beta distinction).

    Returns:
        (count, autodetect_stats) where count is the number of circuits
        mapped and autodetect_stats is a list of per-circuit dicts
        {'circuit': i, 'used_autodetect': bool, 'alpha_len': int,
        'beta_len': int} — useful for UI feedback.
    """
    lines = text.strip().split('\n')
    count = 0
    autodetect_stats = []
    for i, line in enumerate(lines):
        if i >= MAX_CIRCUITS:
            break
        parts = line.replace('[', '').replace(']', '').split(',')
        # Track items rejected by the int-conversion (negative numbers,
        # non-numeric tokens like 'q3', empty cells from trailing commas)
        # so the caller can surface what was dropped instead of silently
        # discarding it. isdigit() rejects negatives by design — qubit IDs
        # in this codebase are non-negative.
        qubits = []
        rejected = []
        for raw in parts:
            tok = raw.strip()
            if not tok:
                continue
            if tok.isdigit():
                qubits.append(int(tok))
            else:
                rejected.append(tok)
        if rejected:
            _log.warning(
                "parse_auto_text line %d: dropped %d non-numeric token(s): %s",
                i + 1, len(rejected), rejected[:10],
            )
        if not qubits:
            continue

        if mode == 'lucj':
            alpha, beta, ok = _autodetect_lucj_partition(qubits, state)
            if ok:
                merged = alpha + beta
                partition = {
                    'alpha_len': len(alpha),
                    'beta_len':  len(beta),
                    'other_len': 0,
                }
                apply_layout_to_circuit(
                    state, i, merged, mode, partition=partition
                )
                autodetect_stats.append({
                    'circuit': i,
                    'used_autodetect': True,
                    'alpha_len': len(alpha),
                    'beta_len':  len(beta),
                })
            else:
                # Midpoint fallback — preserves legacy behavior when
                # topology does not give a clean two-component split.
                apply_layout_to_circuit(state, i, qubits, mode)
                # apply_layout_to_circuit puts n_valid//2 in alpha and the
                # rest in beta — report those exact counts, not (mid, mid),
                # which was off by 1 for odd lengths.
                n_valid = len([q for q in qubits
                               if str(q) in set(state.qubit_ids)])
                mid = n_valid // 2
                autodetect_stats.append({
                    'circuit': i,
                    'used_autodetect': False,
                    'alpha_len': mid,
                    'beta_len':  n_valid - mid,
                })
        else:
            apply_layout_to_circuit(state, i, qubits, mode)
            autodetect_stats.append({
                'circuit': i,
                'used_autodetect': False,
                'alpha_len': 0,
                'beta_len':  0,
            })
        count += 1
    return count, autodetect_stats
