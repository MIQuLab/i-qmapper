"""
engine_select.py — the hybrid search engine: guided below a bridge-count
threshold, exhaustive VF2 above it.

The two single-circuit engines fail in opposite places. With few bridges (low
k) the LUCJ template is loose, VF2 finds tens of thousands of embeddings on a
large heavy-hex and ranks only what fits under its mapping cap. With many
bridges (high k) the template is tight, the embeddings are few, and VF2 is
genuinely exhaustive — exactly where a beam search can only tie or lose. So
each stratum is taken from the engine that is reliable there:

    k <= guided_max_k   →  guided_search.guided_sweep
    k >  guided_max_k   →  ancilla_sweep.sweep_ancilla_layouts (floor k+1)

Both halves are the frozen engines, called unchanged; this module only splits
the k range and merges the two result dicts into the shared sweep shape.
"""
import time

from .ancilla_sweep import sweep_ancilla_layouts
from .auto_lucj import (_make_backend_cmap_pygraph, _remove_nodes_from_graph,
                        create_lucj_zigzag_layout)
from .guided_search import guided_sweep


def _k_range(norb, backend, exclude_qubits, min_bridges):
    """(k_max_effective, floor, excluded_sorted), computed as guided_sweep does.

    Same pruned graph, same `min(3, k_max)` default floor and the same
    ValueError text, so a bad floor reads identically whichever engine runs.
    """
    full_graph = _make_backend_cmap_pygraph(backend=backend)
    excluded_sorted = sorted(int(q) for q in exclude_qubits) if exclude_qubits \
        else []
    if excluded_sorted:
        for i in full_graph.node_indices():
            full_graph[i] = i
        search_graph = _remove_nodes_from_graph(full_graph,
                                                set(excluded_sorted))
    else:
        search_graph = full_graph
    _template, k_max_effective = create_lucj_zigzag_layout(
        num_orbitals=norb, backend_coupling_graph=search_graph,
        requested_bridges=-1)
    floor = min(3, k_max_effective) if min_bridges is None else min_bridges
    if not (1 <= floor <= k_max_effective):
        raise ValueError(
            f"min_bridges={floor} out of range; must satisfy "
            f"1 <= min_bridges <= k_max_effective ({k_max_effective}).")
    return k_max_effective, floor, excluded_sorted


def _tagged(per_stratum, ks, engine):
    """Copies of `per_stratum[k]` for k in `ks`, each tagged with its engine."""
    out = {}
    for k in ks:
        meta = dict(per_stratum.get(k, {}))
        meta['engine'] = engine
        out[k] = meta
    return out


def hybrid_sweep(norb, backend, props, *, guided_max_k, min_bridges=None,
                 top_n=3, exclude_qubits=None, circuit_score_fn=None,
                 diversity_threshold=None, vf2_kwargs=None,
                 guided_kwargs=None) -> dict:
    """Guided search for strata `k <= guided_max_k`, VF2 for the rest.

    Args:
        norb, backend, props, min_bridges, top_n, exclude_qubits,
            circuit_score_fn, diversity_threshold: as for both engines, and
            passed to both.
        guided_max_k: highest bridge count taken from the guided engine.
            Below the floor → pure VF2; at or above `k_max_effective` → pure
            guided. Required: `None` raises ValueError, the caller picks the
            threshold (see `evaluation.suggest_guided_max_k`).
        vf2_kwargs: extra keyword arguments for VF2 only (`hybrid`,
            `mapping_cap`, `shuffle_seed`, `diverse_cap`).
        guided_kwargs: extra keyword arguments for guided only (`beam_width`,
            `refine_top`, `qubit_weight`, `max_iter`, `seed`).

    `guided_sweep` has no upper-k parameter, so it runs from the floor to
    `k_max_effective` and its strata above `guided_max_k` are discarded. That
    is wasted work (one beam per discarded stratum), cheap next to the VF2
    half, and the price of leaving the frozen engine untouched. The guided
    strata are therefore exactly what `guided_sweep` alone would return,
    anchored on its own higher strata rather than on the VF2 winners.

    Returns the shared sweep dict shape (strata k-descending), plus
    `engine='hybrid'`, `guided_max_k`, and `per_stratum[k]['engine']` set to
    `'guided'` or `'vf2'`. Run-level fields come from the half that owns them:
    `vf2_truncated`, `reservoir`, `reservoir_notices` and `mapping_cap` from
    VF2; `refined_count`, `refine_attempted`, `beam_width` and `expansions`
    from guided; counts and `elapsed_s` are summed.
    """
    if guided_max_k is None:
        raise ValueError("guided_max_k is required for the hybrid engine.")
    started = time.time()
    k_max_effective, floor, excluded_sorted = _k_range(
        norb, backend, exclude_qubits, min_bridges)

    shared = dict(top_n=top_n, exclude_qubits=exclude_qubits,
                  circuit_score_fn=circuit_score_fn,
                  diversity_threshold=diversity_threshold)
    vf2_res = guided_res = None

    if guided_max_k < k_max_effective:
        vf2_floor = min_bridges if guided_max_k < floor else guided_max_k + 1
        vf2_res = sweep_ancilla_layouts(norb, backend, props,
                                        min_bridges=vf2_floor,
                                        **shared, **(vf2_kwargs or {}))
    if guided_max_k >= floor:
        guided_res = guided_sweep(norb, backend, props,
                                  min_bridges=min_bridges,
                                  **shared, **(guided_kwargs or {}))

    strata, per_stratum = {}, {}
    if vf2_res is not None:
        vf2_ks = sorted(vf2_res['strata'], reverse=True)
        strata.update((k, vf2_res['strata'][k]) for k in vf2_ks)
        per_stratum.update(_tagged(vf2_res['per_stratum'], vf2_ks, 'vf2'))
    if guided_res is not None:
        guided_ks = sorted((k for k in guided_res['strata']
                            if k <= guided_max_k), reverse=True)
        strata.update((k, guided_res['strata'][k]) for k in guided_ks)
        per_stratum.update(_tagged(guided_res['per_stratum'], guided_ks,
                                   'guided'))
    strata = {k: strata[k] for k in sorted(strata, reverse=True)}
    per_stratum = {k: per_stratum[k] for k in sorted(per_stratum,
                                                     reverse=True)}

    def _sum(key):
        return sum((r or {}).get(key) or 0 for r in (vf2_res, guided_res))

    vf2 = vf2_res or {}
    guided = guided_res or {}
    return {
        'strata': strata,
        'reservoir': vf2.get('reservoir', {}),
        'reservoir_notices': list(vf2.get('reservoir_notices', [])),
        'k_max_effective': k_max_effective,
        'min_bridges': floor,
        'vf2_truncated': bool(vf2.get('vf2_truncated', False)),
        'n_raw_embeddings': _sum('n_raw_embeddings'),
        'n_candidates_scored': _sum('n_candidates_scored'),
        'excluded_qubits': excluded_sorted,
        'per_stratum': per_stratum,
        'mapping_cap': vf2.get('mapping_cap'),
        'engine': 'hybrid',
        'guided_max_k': guided_max_k,
        'beam_width': guided.get('beam_width'),
        'refined_count': guided.get('refined_count', 0),
        'refine_attempted': guided.get('refine_attempted', 0),
        'expansions': guided.get('expansions', 0),
        'elapsed_s': time.time() - started,
    }
