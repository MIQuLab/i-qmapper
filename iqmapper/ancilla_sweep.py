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
* Every k_max embedding yields its lower-`k` variants for free — including the
  "good-region k-1 layout" that an independent k-1 search, starting from the
  rigid every-4th-orbital template, never generates.

LUCJ PREFIX CONTRACT (do not break)
-----------------------------------
A derived k' candidate keeps the bridges at the **first k' standard slots**
``(0, 4, ..., 4*(k'-1))`` and drops the tail — exactly ONE variant per k' per
embedding, byte-for-byte indistinguishable from a legacy exact-k' layout.

We deliberately do NOT enumerate all ``C(k_max, k')`` bridge subsets. A
scattered set like ``{0, 8, 12}`` is a *valid subgraph* with a *valid score*
but a **semantically invalid ansatz layout**: the LUCJ construction places
bridges at consecutive standard slots, and a downstream consumer (rendering,
codegen, exact-k' equivalence) assumes that. Keeping only the prefix restores
the contract — and, as a bonus, does strictly less work (one variant per k',
not a combinatorial fan-out). ``bridge_slots`` is retained precisely to
*certify* this: it is always ``tuple(range(0, 4*k', 4))``.

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

import hashlib
import logging
import random

import numpy as np
import rustworkx

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

# Sentinel for the `shuffle_seed` keyword: AUTO derives a deterministic
# per-stratum seed from (backend_name, norb, k). None disables the shuffle
# (natural node-index order == pre-shuffle behaviour, byte-compatible). An
# explicit int is combined with k so strata still sample differently yet
# reproducibly. See the per-stratum shuffled-search-graph construction below.
_AUTO_SHUFFLE = object()


def _stable_seed(*parts) -> int:
    """Process-independent 32-bit seed from the parts (Python's built-in
    hash() is salted per run and unusable here). Deterministic across runs
    and machines so a shuffled search is reproducible."""
    digest = hashlib.md5('|'.join(str(p) for p in parts).encode()).hexdigest()
    return int(digest[:8], 16)


def _entry_qubit_set(entry):
    """Full physical qubit footprint of a candidate: alpha ∪ beta ∪ ancillas."""
    return frozenset(entry['alpha']) | frozenset(entry['beta']) \
        | frozenset(entry['ancillas'])


def _jaccard(a, b):
    """|A ∩ B| / |A ∪ B| — 0 for two empty sets (no overlap to speak of)."""
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _select_with_diversity(ranked, top_n, threshold):
    """Quota-balanced top_n slice (user-approved option B).

    ``ranked`` is a best-first list of entries. The first ceil(top_n/2) slots
    are the best entries ABSOLUTELY (unchanged). Each remaining slot is filled
    with the best not-yet-picked entry whose Jaccard similarity
    (``_jaccard`` over the full qubit set) to EVERY already-selected entry is
    ``<= threshold`` — i.e. from a genuinely different chip region. If the
    scored window holds no such candidate, it falls back to the next best
    (similar) entry, so we never return fewer than ``min(top_n, len(ranked))``
    — diversity is a preference, not a constraint.

    Marks each returned entry: ``entry['diverse'] = True`` iff it was chosen
    for dissimilarity (the UI badges those), else ``False``.
    """
    n = min(top_n, len(ranked))
    if n <= 0:
        return []
    half = (top_n + 1) // 2                      # ceil(top_n / 2)
    selected = list(ranked[:half])
    for e in selected:
        e['diverse'] = False
    sel_sets = [_entry_qubit_set(e) for e in selected]
    remaining = list(ranked[half:])
    while len(selected) < n and remaining:
        pick = None
        for i, cand in enumerate(remaining):
            cs = _entry_qubit_set(cand)
            if all(_jaccard(cs, s) <= threshold for s in sel_sets):
                pick = i
                break
        if pick is None:                         # no dissimilar candidate left
            cand = remaining.pop(0)              # fall back to next best
            cand['diverse'] = False
        else:
            cand = remaining.pop(pick)
            cand['diverse'] = True
        selected.append(cand)
        sel_sets.append(_entry_qubit_set(cand))
    return selected


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
                          mapping_cap=None, exclude_qubits=None,
                          circuit_score_fn=None, hybrid=True,
                          shuffle_seed=_AUTO_SHUFFLE,
                          diversity_threshold=None,
                          diverse_cap=True) -> dict:
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
        circuit_score_fn: optional ``callable(entry) -> float`` computing a
            CIRCUIT-LEVEL fidelity for a candidate, using the SAME code path
            as the UI's Estimated Layout-Quality box
            (``export.scoring.estimate_circuit_fidelity``). It is state-bound
            (needs the app's dict-shaped props), so the UI injects it rather
            than the engine calling it directly. When provided, it is called
            ONLY on the top_n sliced entries per stratum (cheap) and its
            result is stored on ``entry['lqs_circuit']``; a scoring failure
            leaves ``lqs_circuit=None`` and never aborts the sweep. When
            ``None`` (default) the returned dict is byte-compatible with the
            pre-A2 shape — no ``lqs_circuit`` key is added.
        shuffle_seed: controls the node-INSERTION order of the graph handed to
            each VF2 call, so a truncated search samples the whole chip instead
            of one geographic corner (node-index prefix). ``AUTO`` (default)
            derives a deterministic per-stratum seed from
            ``(backend_name, norb, k)`` — reproducible across runs, and each
            stratum samples a different region. ``None`` keeps the natural
            order (byte-for-byte identical to the pre-shuffle sweep — the
            pinned compatibility path). An explicit ``int`` is combined with
            ``k`` (still per-stratum, still reproducible; for tests). Physical
            qubit ids ride in node data and are recovered order-independently
            via ``_key_to_phys`` — no relabel leakage.
        diversity_threshold: when a float in (0, 1), rebalance each stratum's
            top_n for REGION diversity. After the displayed-metric
            ranking, the first ceil(top_n/2) slots keep the best entries
            absolutely; each remaining slot takes the best entry whose Jaccard
            similarity — ``|A∩B|/|A∪B|`` over the full qubit set (alpha ∪ beta ∪
            ancillas) — to EVERY already-selected entry is ``<= threshold``
            (0.8 ⇒ >0.8 is "same region, variant"). If the scored window has no
            dissimilar candidate, it falls back to the next best (never fewer
            than ``min(top_n, pool)``). Diversity-selected entries are tagged
            ``entry['diverse'] = True``. ``None`` (default) → the exact top_n
            slice, byte-compatible (no ``diverse`` key added).
        hybrid: when True (default), each stratum k' < k_max_effective is the
            UNION of (a) the derived prefix candidates AND (b) a DIRECT VF2 at
            the k' template on the same (possibly pruned) graph. Derivation
            alone chains every low-k stratum to k_max's embeddings, so a clean
            chain that only the smaller k' template can reach is invisible to
            it; the direct pass recovers exactly those. One extra VF2 per
            stratum (the cost the old per-k flow already paid). hybrid=False →
            derivation-only, byte-compatible with the pre-hybrid sweep.

    Returns {
      'strata': {k: [entry, ...]},        # k descending; len<=top_n each
      'k_max_effective': int,             # what actually fit on the chip
      'min_bridges': int,                 # floor actually used
      'vf2_truncated': bool,              # 2000-cap hit (surface in UI!)
      'n_raw_embeddings': int,            # complete VF2 mappings used
      'n_candidates_scored': int,         # total variants scored (pre-dedupe)
      'excluded_qubits': [...],           # sorted list actually pruned (empty
                                          #   when exclude_qubits is None)
      'per_stratum': {k: {'n_direct', 'n_derived', 'truncated', 'shuffled'}},
                                          # per-k provenance; vf2_truncated is
                                          #   any(per-stratum truncated);
                                          #   'shuffled' = search order permuted
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

    # ---- 1b. Per-stratum shuffled search graph ----------------------------
    # The template (WHAT to search for) is always built on the natural
    # `search_graph`, so k_max_effective and every k' template are stable and
    # seed-independent. Only the TARGET graph handed to VF2 is permuted, which
    # merely changes the ORDER embeddings are yielded — so a cap truncates a
    # chip-wide SAMPLE, not one low-index corner. Node data carries the
    # physical id; recovery is order-independent.
    backend_name = str(getattr(backend, 'name', '') or '')

    def _seed_for(k):
        if shuffle_seed is None:
            return None
        if shuffle_seed is _AUTO_SHUFFLE:
            return _stable_seed(backend_name, norb, k)
        return _stable_seed(shuffle_seed, k)

    def _search_for(k):
        """(graph, key_to_phys, shuffled) for the VF2 call at stratum k."""
        seed = _seed_for(k)
        if seed is None:
            return search_graph, _key_to_phys, False
        order = list(search_graph.node_indices())
        random.Random(seed).shuffle(order)
        g = rustworkx.PyGraph()
        old_to_new = {}
        for old in order:
            old_to_new[old] = g.add_node(_key_to_phys(old))   # data = phys id
        for u, v in search_graph.edge_list():
            g.add_edge(old_to_new[u], old_to_new[v], None)
        return g, (lambda key: g[key]), True

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

    # ---- 3. Collect VF2 embeddings, detect truncation ---------------------
    cap = mapping_cap if mapping_cap is not None else _MAX_RAW_VF2_MAPPINGS

    reservoir_stats = {}

    def _make_reservoir(k, key_to_phys):
        """A DiverseReservoir scoring with the sweep's own product formula.

        The score closes over the `readout` / `_edge_e` hoisted just above,
        so enumeration and `_add_variant` cannot drift apart. Returns None
        when diverse_cap is off, which restores the historical path exactly.
        """
        if not diverse_cap:
            return None
        from .reservoir import DiverseReservoir
        size = 2 * norb + k

        def _to_layout(mapping):
            layout = [None] * size
            for key, value in mapping.items():
                if value < size:
                    layout[value] = key_to_phys(key)
            return layout

        def _score(layout):
            alpha = layout[:norb]
            beta = layout[norb:2 * norb]
            product = 1.0
            for q in layout[:2 * norb]:
                product *= max(0.0, 1.0 - float(readout[q]))
            for j in range(norb - 1):
                product *= max(0.0, 1.0 - _edge_e(alpha[j], alpha[j + 1]))
                product *= max(0.0, 1.0 - _edge_e(beta[j], beta[j + 1]))
            for i in range(k):
                bphys = layout[2 * norb + i]
                product *= (max(0.0, 1.0 - float(readout[bphys]))
                            * max(0.0, 1.0 - _edge_e(alpha[4 * i], bphys))
                            * max(0.0, 1.0 - _edge_e(bphys, beta[4 * i])))
            return product

        return DiverseReservoir(cap, _score, _to_layout)

    kmax_graph, kmax_key_to_phys, kmax_shuffled = _search_for(k_max_effective)
    _kmax_res = _make_reservoir(k_max_effective, kmax_key_to_phys)
    mappings = _collect_vf2_mappings(kmax_graph, template, cap=cap,
                                     reservoir=_kmax_res)
    if _kmax_res is not None:
        reservoir_stats[k_max_effective] = _kmax_res.stats
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
            layout[value] = kmax_key_to_phys(key)
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

    # ---- 6. Enumerate + score every variant, grouped by stratum ----------
    # strata_pool[k] : dict keyed by (frozenset(chain), frozenset(ancilla))
    #                  -> entry, giving per-stratum dedupe for free.
    strata_pool = {k: {} for k in range(k_max_effective, min_bridges - 1, -1)}
    per_stratum = {k: {'n_direct': 0, 'n_derived': 0, 'truncated': False,
                       'shuffled': False}
                   for k in range(k_max_effective, min_bridges - 1, -1)}
    n_candidates_scored = 0

    def _embedding_data(layout, n_bridges):
        """Per-embedding precompute: chain aggregates + per-bridge factors.
        Constant across every k' derived from this embedding."""
        alpha = layout[:norb]
        beta = layout[norb:2 * norb]
        chain = layout[:2 * norb]
        chain_qubits = np.array(chain, dtype=int)
        chain_edges = ([(alpha[j], alpha[j + 1]) for j in range(norb - 1)]
                       + [(beta[j], beta[j + 1]) for j in range(norb - 1)])
        chain_gate_errs = np.array([_edge_e(a, b) for a, b in chain_edges],
                                   dtype=float)
        bridges = []
        for i in range(n_bridges):
            bphys = layout[2 * norb + i]
            e1 = _edge_e(alpha[4 * i], bphys)
            e2 = _edge_e(bphys, beta[4 * i])
            r = float(readout[bphys])
            bridges.append({
                'phys': bphys, 'slot': 4 * i, 'add': r + e1 + e2,
                'prod': max(0.0, 1.0 - r) * max(0.0, 1.0 - e1) * max(0.0, 1.0 - e2),
            })
        # Alpha/beta swap normalization (mirror _deduplicate_layouts): the
        # representative alpha is the chain with the smaller first element.
        if alpha and beta and alpha[0] > beta[0]:
            norm_alpha, norm_beta = list(beta), list(alpha)
        else:
            norm_alpha, norm_beta = list(alpha), list(beta)
        return {
            'chain_fs': frozenset(chain),
            'norm_alpha': norm_alpha, 'norm_beta': norm_beta,
            'norm_chain': norm_alpha + norm_beta,
            'cr_sum': float(readout[chain_qubits].sum()),
            'cr_prod': float(np.prod(np.clip(1.0 - readout[chain_qubits], 0.0, 1.0))),
            'cg_sum': float(chain_gate_errs.sum()),
            'cg_prod': float(np.prod(np.clip(1.0 - chain_gate_errs, 0.0, 1.0))),
            'bridges': bridges,
        }

    def _add_variant(edata, k, source):
        """Score the first-k prefix of this embedding and add it to
        strata_pool[k]. Returns True iff a NEW (chain, ancilla) was inserted.

        LUCJ PREFIX CONTRACT: keep bridges at the first k standard slots
        (0, 4, ..., 4*(k-1)) — for a direct k' embedding that IS every bridge;
        for a derived one it drops the tail. Scattered subsets are invalid.
        """
        bridges = edata['bridges']
        kept = range(k)
        ancillas = [bridges[i]['phys'] for i in kept]
        key = (edata['chain_fs'], frozenset(ancillas))
        if key in strata_pool[k]:
            # Duplicate (alpha/beta swap, or a chain found both derived AND
            # direct): identical qubits/edges → identical score, keep first.
            return False
        additive = (edata['cr_sum'] + edata['cg_sum']
                    + sum(bridges[i]['add'] for i in kept))
        product = edata['cr_prod'] * edata['cg_prod']
        for i in kept:
            product *= bridges[i]['prod']
        strata_pool[k][key] = {
            'layout': list(edata['norm_chain']),
            'alpha': list(edata['norm_alpha']),
            'beta': list(edata['norm_beta']),
            'ancillas': ancillas,
            'bridge_slots': tuple(bridges[i]['slot'] for i in kept),
            'product_score': product,
            'additive_score': additive,
            'rank_in_stratum': 0,  # filled after sort
            'source': source,
        }
        return True

    # 6a. k_max embeddings → the k_max stratum (direct) + derived prefixes.
    per_stratum[k_max_effective]['truncated'] = vf2_truncated
    per_stratum[k_max_effective]['shuffled'] = kmax_shuffled
    for layout in layouts:
        edata = _embedding_data(layout, k_max_effective)
        for k in range(k_max_effective, min_bridges - 1, -1):
            n_candidates_scored += 1
            source = 'vf2' if k == k_max_effective else 'derived'
            if _add_variant(edata, k, source):
                bucket = 'n_direct' if k == k_max_effective else 'n_derived'
                per_stratum[k][bucket] += 1

    # 6b. HYBRID: a DIRECT VF2 at each lower-k template on the SAME (pruned)
    # graph — recovers clean chains derivation can't reach from k_max.
    if hybrid:
        for k in range(k_max_effective - 1, min_bridges - 1, -1):
            tmpl_k, nb_k = create_lucj_zigzag_layout(
                num_orbitals=norb, backend_coupling_graph=search_graph,
                requested_bridges=k)
            if nb_k != k:
                # The k' template auto-reduced to a different count — the
                # derived stratum already covers this k; skip the direct pass.
                _log.debug("Hybrid: k=%d template returned %d bridges; "
                           "skipping direct pass", k, nb_k)
                continue
            k_graph, k_key_to_phys, k_shuffled = _search_for(k)
            per_stratum[k]['shuffled'] = k_shuffled
            _res_k = _make_reservoir(nb_k, k_key_to_phys)
            maps_k = _collect_vf2_mappings(k_graph, tmpl_k, cap=cap,
                                           reservoir=_res_k)
            if _res_k is not None:
                reservoir_stats[k] = _res_k.stats
            if len(maps_k) >= cap:
                per_stratum[k]['truncated'] = True
            exp_k = 2 * norb + nb_k
            for mapping in maps_k:
                lay = [None] * exp_k
                for key, value in mapping.items():
                    lay[value] = k_key_to_phys(key)
                if any(v is None for v in lay):
                    continue
                edata = _embedding_data(lay, nb_k)
                # CASCADE: this direct k'-embedding seeds its OWN stratum
                # (source 'vf2') AND every stratum below it via the prefix rule
                # (source 'derived'). So a region that only a k'-search reaches
                # — never enumerated at k_max — still populates all lower
                # strata, and best-per-k stays monotone as k decreases.
                for kk in range(k, min_bridges - 1, -1):
                    n_candidates_scored += 1
                    src = 'vf2' if kk == k else 'derived'
                    if _add_variant(edata, kk, src):
                        bucket = 'n_direct' if kk == k else 'n_derived'
                        per_stratum[kk][bucket] += 1

    # Aggregate truncation for the existing banner (k_max OR any direct pass).
    vf2_truncated = any(m['truncated'] for m in per_stratum.values())

    # ---- 7. Slice each stratum by the DISPLAYED metric --------------------
    # product_score (template scale) is the cheap PRE-FILTER for the pool, but
    # the final slice must be chosen by the metric the UI shows (lqs_circuit).
    # The two scales diverge, so slicing to top_n by product_score BEFORE
    # circuit-scoring can drop the true circuit-best below the line unseen —
    # a monotonicity violation observed on hardware where a stratum's displayed
    # best fell as k decreased (e.g. "3->49.3% . 2->40.8%").
    #
    # Fix: circuit-score a WIDENED window = the top W by product_score
    # (W = max(top_n*5, 25)) PLUS, guaranteed regardless of rank, the
    # prefix-child of every higher stratum's already-selected winner (the
    # monotonicity anchors — the cascade already generated them, so they live
    # in this pool). Then rank the scored window by lqs_circuit (None sinks)
    # and slice to top_n. With no circuit scorer the product-sliced shape is
    # preserved byte-for-byte (the historical default, before circuit scoring).
    strata = {}
    # higher_winners: (chain_fs, ancillas) of every selected entry in a higher
    # stratum, so stratum k can pull in each one's k-prefix child as an anchor.
    higher_winners = []
    for k in range(k_max_effective, min_bridges - 1, -1):
        pool = strata_pool[k]
        by_product = sorted(pool.items(),
                            key=lambda kv: kv[1]['product_score'], reverse=True)

        if circuit_score_fn is None:
            # Byte-compatible product-only path: product slice, no lqs_circuit.
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
        # Guarantee anchors: the k-prefix child of each higher winner.
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
                # A scoring error must never abort the sweep.
                _log.warning("Circuit scoring failed for a candidate "
                             "(k=%d): %s", k, exc)
                entry['lqs_circuit'] = None
        # Rank by the DISPLAYED metric; None sinks; product_score breaks ties.
        window.sort(
            key=lambda e: (e['lqs_circuit'] is not None,
                           e['lqs_circuit'] if e['lqs_circuit'] is not None
                           else 0.0,
                           e['product_score']),
            reverse=True)
        # Diversity quota: reserve later slots for different-region
        # layouts (Jaccard <= threshold), applied AFTER the displayed-metric
        # ranking, BEFORE rank numbering. None → exact top-N slice.
        if diversity_threshold is None:
            entries = window[:top_n]
        else:
            entries = _select_with_diversity(window, top_n, diversity_threshold)
        for rank, entry in enumerate(entries, start=1):
            entry['rank_in_stratum'] = rank
        strata[k] = entries
        higher_winners.extend(
            (frozenset(e['layout']), tuple(e['ancillas'])) for e in entries)
        _log.debug("Ancilla sweep stratum k=%d: scored %d, kept top %d",
                   k, len(window), len(entries))

    # One notice per stratum whose cap was actually reached. The reservoir
    # only engages on the hybrid path (at k_max the enumeration is far below
    # the cap), so the notice names the hybrid pass explicitly.
    reservoir_notices = []
    if reservoir_stats:
        from .reservoir import summarise as _summarise_reservoir
        for k in sorted(reservoir_stats, reverse=True):
            stats_k = reservoir_stats[k]
            if stats_k.get('kept', 0) >= cap:
                note = _summarise_reservoir(
                    stats_k, k, hybrid=(k != k_max_effective))
                if note:
                    reservoir_notices.append(note)

    return {
        'strata': strata,
        'reservoir': reservoir_stats,
        'reservoir_notices': reservoir_notices,
        'k_max_effective': k_max_effective,
        'min_bridges': min_bridges,
        'vf2_truncated': vf2_truncated,
        'n_raw_embeddings': n_raw_embeddings,
        'n_candidates_scored': n_candidates_scored,
        'excluded_qubits': excluded_sorted,
        'per_stratum': per_stratum,
        # The ACTUAL VF2 cap, so the truncation banner can name it
        # ("truncated at 2000 mappings") instead of n_raw_embeddings, and
        # per_stratum[k]['truncated'] says WHICH strata hit it.
        'mapping_cap': cap,
    }
