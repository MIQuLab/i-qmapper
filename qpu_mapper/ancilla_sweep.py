"""
ancilla_sweep.py — One-shot ancilla (bridge-count) sweep for LUCJ layouts.

Motivation
----------
A LUCJ zigzag layout places `k` "bridge" (ancilla) qubits between the alpha
and beta chains, one every four orbitals. The number of bridges is a knob:
more bridges give the ansatz more connectivity but consume more physical
qubits and add error-prone couplings, so the best `k` is device- and
region-dependent. The naive way to explore it is to run one VF2 subgraph
search per candidate `k`. That is wasteful and — worse — *misses layouts*:
each independent search re-solves a differently-shaped template and the
2000-mapping VF2 cap (see ``auto_lucj._MAX_RAW_VF2_MAPPINGS``) bites
differently at each `k`, so a good low-`k` placement that lives inside a
great high-`k` embedding is never enumerated.

Algorithm (bridge subtraction)
------------------------------
This module runs VF2 **once**, at the geometric maximum bridge count
``k_max = ceil(norb / 4)``, then derives every lower-`k` candidate by
*removing* bridges from each embedding:

1. Build the k_max template with
   :func:`auto_lucj.create_lucj_zigzag_layout` and collect its embeddings
   with :func:`auto_lucj._collect_vf2_mappings`.
2. For every embedding and every ``k'`` from ``k_max_effective`` down to
   ``min_bridges``, enumerate all size-``k'`` subsets of the bridges to keep.
   The chain qubits never move; only which ancillas (and their two coupling
   edges) are retained changes.
3. Deduplicate per stratum (alpha/beta swaps collapse, mirroring
   :func:`auto_lucj._deduplicate_layouts`), score, and keep the top-N per `k`.

Why this is strictly better than per-`k` search:

* The k_max template is the *most constrained* subgraph, so the VF2 cap
  removes the fewest useful embeddings there.
* Every k_max embedding yields all of its lower-`k` variants for free —
  including the "good-region k-1 layout" that an independent k-1 search,
  starting from the rigid every-4th-orbital template, never generates.
* Subtraction also unlocks *non-standard* bridge-position sets (e.g. keeping
  orbital slots ``{0, 8, 12}`` out of ``{0, 4, 8, 12}``) that the fixed
  template cannot express at all.

Scoring: product vs additive
-----------------------------
Each candidate carries two scores:

* ``additive_score`` — the legacy lightweight sum of readout errors over the
  used qubits plus 2Q gate errors over the used edges. It is a **sum**, so it
  grows with layout size: a k-bridge layout always has a larger additive sum
  than its (k-1)-bridge subset *even when the extra bridge is excellent*.
  Additive is therefore fine for ranking **within** a single stratum (fixed
  size) but must **never** be used to compare across strata.
* ``product_score`` — a multiplicative fidelity proxy,
  ``prod(1 - readout_err) * prod(1 - gate_err)`` over the same qubits and
  edges. Adding a bridge multiplies the score by that bridge's fidelity
  factor ``(1 - r)(1 - e1)(1 - e2) <= 1``, so product is directly comparable
  both within and across strata. **product_score is the ranking key.**

Caveats
-------
* The VF2 cap (``mapping_cap``, default 2000) can truncate the embedding set
  on large heavy-hex backends. When that happens ``vf2_truncated`` is True in
  the result and callers should surface it (the sweep may not have seen every
  possible layout).
* Like the underlying lightweight scorer, this is a first-order relative
  heuristic (independent-error assumption, one gate per edge, no crosstalk),
  suitable for ranking, not absolute fidelity prediction.
"""

import itertools
import logging

import numpy as np

from .auto_lucj import (
    _collect_vf2_mappings,
    _get_two_q_gate_name,
    _make_backend_cmap_pygraph,
    _MAX_RAW_VF2_MAPPINGS,
    _remove_nodes_from_graph,
    create_lucj_zigzag_layout,
)

_log = logging.getLogger(__name__)

# NaN / uncalibrated penalty, matching lightweight_layout_error_scoring.
# For the additive sum this adds 1.0 (max error); for the product it makes
# the fidelity factor (1 - 1.0) = 0, i.e. the candidate is effectively
# excluded from the top ranks.
_UNCALIBRATED_PENALTY = 1.0


def _safe_readout(props, qubit):
    """Readout error for `qubit`, NaN/None -> penalty (mirrors lightweight)."""
    try:
        val = props.readout_error(qubit)
    except Exception:
        return _UNCALIBRATED_PENALTY
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return _UNCALIBRATED_PENALTY
    return float(val)


def _safe_gate_error(props, gate_name, u, v):
    """2Q gate error for edge (u, v) with reverse fallback, NaN -> penalty."""
    for a, b in ((u, v), (v, u)):
        try:
            val = props.gate_error(gate_name, (a, b))
        except Exception:
            continue
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        return float(val)
    return _UNCALIBRATED_PENALTY


def sweep_ancilla_layouts(norb, backend, props, *,
                          min_bridges=None, top_n=3,
                          mapping_cap=None, exclude_qubits=None) -> dict:
    """One-shot ancilla sweep: VF2 at the geometric max bridge count,
    then derive every lower-k variant by bridge subtraction, score, and
    return the top_n candidates per ancilla count.

    Args:
        norb: Number of spatial orbitals (alpha chain length).
        backend: IBM Qiskit BackendV2-like object (needs ``coupling_map`` and
            a discoverable 2Q gate; see ``auto_lucj._get_two_q_gate_name``).
        props: BackendProperties-like object exposing ``readout_error(q)`` and
            ``gate_error(name, (u, v))``. No API calls are made here.
        min_bridges: Floor of the sweep (inclusive). Defaults to
            ``min(3, k_max_effective)``. Must satisfy
            ``1 <= min_bridges <= k_max_effective`` or ValueError is raised.
        top_n: Max candidates returned per ancilla count.
        mapping_cap: Override the VF2 mapping cap (default
            ``auto_lucj._MAX_RAW_VF2_MAPPINGS``). Mainly for tests.
        exclude_qubits: optional set of physical qubit ids to remove from the
            coupling graph BEFORE template creation (a hard search-space
            filter, e.g. volatile qubits from stability analysis). ``None`` or
            empty → byte-for-byte identical to the unfiltered sweep.

    Returns {
      'strata': {k: [entry, ...]},        # k descending; len<=top_n each
      'k_max_effective': int,             # what actually fit on the chip
      'min_bridges': int,                 # floor actually used
      'vf2_truncated': bool,              # 2000-cap hit (surface in UI!)
      'n_raw_embeddings': int,            # complete VF2 mappings used
      'n_candidates_scored': int,         # total variants scored (pre-dedupe)
      'excluded_qubits': [...],           # sorted list actually pruned (empty
                                          #   when exclude_qubits is None)
    }
    entry = {
      'layout': [...],                    # chain qubits (alpha+beta)
      'alpha': [...], 'beta': [...],
      'ancillas': [...],                  # physical bridge qubits kept
      'bridge_slots': (...),              # orbital slots kept, e.g. (0, 8, 12)
      'product_score': float,             # multiplicative, THE ranking key
      'additive_score': float,            # legacy lightweight, informational
      'rank_in_stratum': int,             # 1-based, by product_score desc
      'source': 'vf2' | 'derived',        # intact k_max vs subtraction variant
    }
    """
    # ---- 1. Build the backend graph and the k_max template ----------------
    # `full_graph` keeps physical-qubit ids as node INDICES (props precompute
    # relies on that). When excluding, we prune a SEPARATE `search_graph`:
    # _remove_nodes_from_graph reindexes, so we tag physical ids into node
    # data first and recover them via `_key_to_phys` when reading VF2 keys.
    full_graph = _make_backend_cmap_pygraph(backend=backend)
    excluded_sorted = sorted(int(q) for q in exclude_qubits) if exclude_qubits else []
    if excluded_sorted:
        for i in full_graph.node_indices():
            full_graph[i] = i
        search_graph = _remove_nodes_from_graph(full_graph, set(excluded_sorted))

        def _key_to_phys(k):
            return search_graph[k]
    else:
        search_graph = full_graph

        def _key_to_phys(k):
            return k

    k_max = (norb + 3) // 4  # ceil(norb / 4): the geometric bridge ceiling
    template, k_max_effective = create_lucj_zigzag_layout(
        num_orbitals=norb,
        backend_coupling_graph=search_graph,
        requested_bridges=k_max,
    )
    if k_max_effective < 1:
        raise RuntimeError(
            f"No ancilla bridges fit for norb={norb} on this backend "
            f"(k_max_effective={k_max_effective}); an ancilla sweep needs "
            f"at least one bridge."
        )

    # ---- 2. Resolve and validate min_bridges ------------------------------
    if min_bridges is None:
        min_bridges = min(3, k_max_effective)
    if not (1 <= min_bridges <= k_max_effective):
        raise ValueError(
            f"min_bridges={min_bridges} out of range; must satisfy "
            f"1 <= min_bridges <= k_max_effective ({k_max_effective})."
        )

    # ---- 3. Collect VF2 embeddings, detect truncation ---------------------
    cap = mapping_cap if mapping_cap is not None else _MAX_RAW_VF2_MAPPINGS
    mappings = _collect_vf2_mappings(search_graph, template, cap=cap)
    # The helper only *logs* truncation; detect it by the cap being reached.
    vf2_truncated = len(mappings) >= cap
    if vf2_truncated:
        _log.debug("Ancilla sweep: VF2 truncated at cap=%d", cap)

    # ---- 4. Materialize complete layouts (mirror get_all_zigzag_layouts) --
    expected_size = 2 * norb + k_max_effective
    layouts = []
    for mapping in mappings:
        layout = [None] * expected_size
        for key, value in mapping.items():
            layout[value] = _key_to_phys(key)
        if any(v is None for v in layout):
            _log.debug("Ancilla sweep: skipping incomplete VF2 mapping")
            continue
        layouts.append(layout)
    n_raw_embeddings = len(layouts)
    if not layouts:
        raise RuntimeError(
            f"No complete zigzag embedding found for norb={norb} on this "
            f"backend."
        )

    # ---- 5. Precompute props once (hoist all API-shaped lookups) ----------
    gate_name = _get_two_q_gate_name(backend)
    if gate_name is None:
        raise RuntimeError("Could not determine the backend 2Q gate name.")

    # Precompute over the FULL graph (indices == physical ids). Excluded
    # qubits/edges may be present here but never appear in any layout, so
    # their entries are harmless.
    node_indices = list(full_graph.node_indices())
    max_id = max(node_indices)
    # readout array indexed by physical qubit id; penalty for anything absent.
    readout = np.full(max_id + 1, _UNCALIBRATED_PENALTY, dtype=float)
    for q in node_indices:
        readout[q] = _safe_readout(props, q)
    # physical-edge -> gate error, keyed by frozenset so direction is moot.
    edge_err = {}
    for u, v in full_graph.edge_list():
        edge_err[frozenset((u, v))] = _safe_gate_error(props, gate_name, u, v)

    def _edge_e(a, b):
        return edge_err.get(frozenset((a, b)),
                            _safe_gate_error(props, gate_name, a, b))

    # ---- 6. Enumerate + score every variant, grouped by stratum ----------
    # strata_pool[k] : dict keyed by (frozenset(chain), frozenset(ancilla))
    #                  -> entry, giving per-stratum dedupe for free.
    strata_pool = {k: {} for k in range(k_max_effective, min_bridges - 1, -1)}
    n_candidates_scored = 0

    for layout in layouts:
        alpha = layout[:norb]
        beta = layout[norb:2 * norb]
        chain = layout[:2 * norb]
        chain_fs = frozenset(chain)

        # Chain aggregates are constant across every variant of this
        # embedding, so compute them once.
        chain_qubits = np.array(chain, dtype=int)
        chain_readout_sum = float(readout[chain_qubits].sum())
        chain_readout_prod = float(np.prod(np.clip(1.0 - readout[chain_qubits],
                                                   0.0, 1.0)))
        chain_edges = ([(alpha[j], alpha[j + 1]) for j in range(norb - 1)]
                       + [(beta[j], beta[j + 1]) for j in range(norb - 1)])
        chain_gate_errs = np.array([_edge_e(a, b) for a, b in chain_edges],
                                   dtype=float)
        chain_gate_sum = float(chain_gate_errs.sum())
        chain_gate_prod = float(np.prod(np.clip(1.0 - chain_gate_errs,
                                                0.0, 1.0)))

        # Per-bridge contributions for this embedding.
        bridges = []
        for i in range(k_max_effective):
            bphys = layout[2 * norb + i]
            e1 = _edge_e(alpha[4 * i], bphys)
            e2 = _edge_e(bphys, beta[4 * i])
            r = float(readout[bphys])
            add_i = r + e1 + e2
            prod_i = (max(0.0, 1.0 - r) * max(0.0, 1.0 - e1)
                      * max(0.0, 1.0 - e2))
            bridges.append({
                'phys': bphys,
                'slot': 4 * i,
                'add': add_i,
                'prod': prod_i,
            })

        # Alpha/beta swap normalization (mirror _deduplicate_layouts): the
        # representative alpha is the chain with the smaller first element.
        if alpha and beta and alpha[0] > beta[0]:
            norm_alpha, norm_beta = list(beta), list(alpha)
        else:
            norm_alpha, norm_beta = list(alpha), list(beta)
        norm_chain = norm_alpha + norm_beta

        for k in range(k_max_effective, min_bridges - 1, -1):
            for kept in itertools.combinations(range(k_max_effective), k):
                n_candidates_scored += 1
                ancillas = [bridges[i]['phys'] for i in kept]
                ancilla_fs = frozenset(ancillas)
                key = (chain_fs, ancilla_fs)
                if key in strata_pool[k]:
                    # Duplicate (e.g. alpha/beta swap): identical qubits and
                    # edges -> identical score, so keep the first.
                    continue
                additive = (chain_readout_sum + chain_gate_sum
                            + sum(bridges[i]['add'] for i in kept))
                product = chain_readout_prod * chain_gate_prod
                for i in kept:
                    product *= bridges[i]['prod']
                strata_pool[k][key] = {
                    'layout': list(norm_chain),
                    'alpha': list(norm_alpha),
                    'beta': list(norm_beta),
                    'ancillas': ancillas,
                    'bridge_slots': tuple(bridges[i]['slot'] for i in kept),
                    'product_score': product,
                    'additive_score': additive,
                    'rank_in_stratum': 0,  # filled after sort
                    'source': 'vf2' if k == k_max_effective else 'derived',
                }

    # ---- 7. Rank each stratum by product_score desc, slice top_n ---------
    strata = {}
    for k in range(k_max_effective, min_bridges - 1, -1):
        entries = sorted(strata_pool[k].values(),
                         key=lambda e: e['product_score'], reverse=True)
        entries = entries[:top_n]
        for rank, entry in enumerate(entries, start=1):
            entry['rank_in_stratum'] = rank
        strata[k] = entries
        _log.debug("Ancilla sweep stratum k=%d: %d candidates (top_n=%d)",
                   k, len(entries), top_n)

    return {
        'strata': strata,
        'k_max_effective': k_max_effective,
        'min_bridges': min_bridges,
        'vf2_truncated': vf2_truncated,
        'n_raw_embeddings': n_raw_embeddings,
        'n_candidates_scored': n_candidates_scored,
        'excluded_qubits': excluded_sorted,
    }
