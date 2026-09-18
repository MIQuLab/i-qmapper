"""
multi_sweep.py — Sequential greedy orchestrator for placing SEVERAL LUCJ
circuits on one QPU, each via the single-circuit ancilla sweep.

Placement is greedy and sequential, mirroring the legacy multi-circuit path
(``auto_lucj.get_multi_circuit_layouts``): circuit i is swept on the qubits
still free after circuits 0..i-1 have claimed their chosen layout's qubits
PLUS a coupling-graph buffer around each. The buffer semantics are inherited
verbatim from ``auto_lucj._exclude_with_buffer`` — "buffer_qubits=1" excludes
every 1-hop coupling neighbour of a placed qubit — so "Buffer qubits: N" means
exactly what it means in the legacy tool.

Nothing here edits the frozen engines: the single-circuit search is
``ancilla_sweep.sweep_ancilla_layouts`` and the buffer/exclusion math is
``auto_lucj._exclude_with_buffer`` / ``_make_backend_cmap_pygraph``, both
imported read-only.

Fit failures never raise: if circuit i cannot be placed on what remains, the
run stops there and returns the partial result with ``failed_at=i`` (the UI
decides how to message it).
"""

import logging

from .ancilla_sweep import sweep_ancilla_layouts, _AUTO_SHUFFLE
from .auto_lucj import _exclude_with_buffer, _make_backend_cmap_pygraph

_log = logging.getLogger(__name__)

AUTO = _AUTO_SHUFFLE   # re-export the shuffle sentinel for callers/tests


def _per_circuit(value, i):
    """Resolve a knob that may be a global scalar or a per-circuit list."""
    if isinstance(value, (list, tuple)):
        return value[i] if i < len(value) else None
    return value


def _entry_qubits(entry):
    """Full physical footprint of a chosen entry: alpha ∪ beta ∪ ancillas."""
    return list(entry['alpha']) + list(entry['beta']) + list(entry['ancillas'])


def _best_entry(res, has_scorer):
    """The circuit's chosen layout = the entry with the highest DISPLAYED
    metric across all strata (lqs_circuit when a circuit scorer is wired, else
    product_score). Falls back to product_score if every lqs_circuit is None
    (scoring failed for the whole sweep). Returns None if there are no
    entries at all."""
    def _scan(key):
        best, best_v = None, None
        for entries in res.get('strata', {}).values():
            for e in entries:
                v = key(e)
                if v is None:
                    continue
                if best is None or v > best_v:
                    best, best_v = e, v
        return best

    if has_scorer:
        best = _scan(lambda e: e.get('lqs_circuit'))
        if best is not None:
            return best
    return _scan(lambda e: e['product_score'])


def _circuit_record(i, norb_i, res_i, chosen):
    """Per-circuit result: the chosen layout PLUS the full strata (kept so a
    future override UX can substitute a different candidate) and metadata."""
    return {
        'circuit_idx': i,
        'norb': norb_i,
        'chosen': chosen,
        'strata': res_i['strata'],
        'k_max_effective': res_i['k_max_effective'],
        'min_bridges': res_i['min_bridges'],
        'vf2_truncated': res_i['vf2_truncated'],
        'per_stratum': res_i['per_stratum'],
        'mapping_cap': res_i.get('mapping_cap'),
        'excluded_qubits': res_i['excluded_qubits'],
    }


def _orchestrate(ctx, prefix=None):
    """Core sequential greedy pass.

    ``prefix`` (for :func:`replan_from`) is a list of ``(circuit_record,
    chosen_entry)`` for the first len(prefix) circuits whose selection is
    FIXED: they are not re-swept, but their chosen qubits still shape the
    exclusion set for everything after them. All later circuits are swept.
    """
    prefix = prefix or []
    norbs = ctx['norbs']
    buffer_qubits = ctx['buffer_qubits']
    has_scorer = ctx['circuit_score_fn'] is not None

    # Physical-id-tagged coupling graph, used ONLY for buffer expansion; the
    # sweep prunes its own graph from `exclude_qubits`.
    remaining = _make_backend_cmap_pygraph(backend=ctx['backend'])
    for idx in remaining.node_indices():
        remaining[idx] = idx

    stability = set(int(q) for q in (ctx['exclude_qubits'] or []))
    if stability:
        # buffer=0 → remove exactly the stability qubits (no expansion).
        remaining, _ = _exclude_with_buffer(remaining, stability, 0)
    excluded = set(stability)

    circuits, placed, buffers = [], [], []
    failed_at = None

    for i, norb_i in enumerate(norbs):
        if i < len(prefix):
            rec, chosen = prefix[i]
            rec = dict(rec)
            rec['chosen'] = chosen
        else:
            try:
                res_i = sweep_ancilla_layouts(
                    norb_i, ctx['backend'], ctx['props'],
                    exclude_qubits=(excluded or None),
                    min_bridges=_per_circuit(ctx['min_bridges'], i),
                    top_n=ctx['top_n'], mapping_cap=ctx['mapping_cap'],
                    hybrid=ctx['hybrid'],
                    diversity_threshold=ctx['diversity_threshold'],
                    circuit_score_fn=ctx['circuit_score_fn'],
                    shuffle_seed=ctx['shuffle_seed'])
            except Exception as exc:
                # A fit failure (or any sweep error) stops the run honestly.
                _log.info("Multi-sweep: circuit %d (norb=%d) did not fit: %s",
                          i, norb_i, exc)
                failed_at = i
                break
            chosen = _best_entry(res_i, has_scorer)
            if chosen is None:
                failed_at = i
                break
            rec = _circuit_record(i, norb_i, res_i, chosen)

        circuits.append(rec)
        chosen_full = _entry_qubits(chosen)
        placed.extend(chosen_full)
        remaining, new_excl = _exclude_with_buffer(
            remaining, set(chosen_full), buffer_qubits)
        buffers.extend(sorted(new_excl - set(chosen_full)))
        excluded |= new_excl

    chosen_list = [c['chosen'] for c in circuits]
    total_product = None
    if chosen_list:
        total_product = 1.0
        for c in chosen_list:
            total_product *= c['product_score']
    total_lqs = None
    if chosen_list and all(c.get('lqs_circuit') is not None for c in chosen_list):
        total_lqs = 1.0
        for c in chosen_list:
            total_lqs *= c['lqs_circuit']

    return {
        'circuits': circuits,
        'total_product': total_product,
        'total_lqs_circuit': total_lqs,
        'placed_qubits': placed,
        'buffer_qubits_used': buffers,
        'failed_at': failed_at,
        '_context': ctx,
    }


def sweep_multi_circuits(norbs, backend, props, *,
                         min_bridges=None, top_n=3, buffer_qubits=1,
                         exclude_qubits=None, mapping_cap=None, hybrid=True,
                         diversity_threshold=None, circuit_score_fn=None,
                         shuffle_seed=AUTO):
    """Place ``len(norbs)`` LUCJ circuits on one backend, greedily and in order.

    For each circuit i the search space is the chip minus: the stability
    pre-exclusions (``exclude_qubits``), every qubit already claimed by
    circuits 0..i-1, AND their ``buffer_qubits``-hop coupling neighbourhood
    (``auto_lucj._exclude_with_buffer`` semantics). The circuit's chosen layout
    is the best entry by the displayed metric (``lqs_circuit`` when
    ``circuit_score_fn`` is wired, else ``product_score``).

    Args:
        norbs: list of spatial-orbital counts, one per circuit.
        backend, props: as for :func:`ancilla_sweep.sweep_ancilla_layouts`.
        min_bridges: global floor, OR a per-circuit list (indexed by circuit).
        top_n, mapping_cap, hybrid, diversity_threshold, circuit_score_fn,
            shuffle_seed: forwarded unchanged to each single-circuit sweep.
        buffer_qubits: idle coupling-graph hops kept clear between circuits.
        exclude_qubits: physical ids barred from EVERY circuit (e.g. stability
            volatility/quality-floor pre-exclusions).

    Returns a dict::

        {
          'circuits': [ {circuit_idx, norb, chosen, strata, k_max_effective,
                         min_bridges, vf2_truncated, per_stratum, mapping_cap,
                         excluded_qubits} ... ],   # one per PLACED circuit
          'total_product': prod(chosen_i.product_score) or None,
          'total_lqs_circuit': prod(chosen_i.lqs_circuit) or None if any missing,
          'placed_qubits': [...],          # all chosen qubits, in placement order
          'buffer_qubits_used': [...],     # buffer qubits kept clear
          'failed_at': None | int,         # circuit i could not fit → stopped
          '_context': {...},               # inputs, for replan_from()
        }

    A circuit that cannot fit stops the run: earlier circuits are returned and
    ``failed_at`` is that index. This never raises for a fit failure.
    """
    ctx = {
        'norbs': list(norbs), 'backend': backend, 'props': props,
        'min_bridges': min_bridges, 'top_n': top_n,
        'buffer_qubits': buffer_qubits, 'exclude_qubits': exclude_qubits,
        'mapping_cap': mapping_cap, 'hybrid': hybrid,
        'diversity_threshold': diversity_threshold,
        'circuit_score_fn': circuit_score_fn, 'shuffle_seed': shuffle_seed,
    }
    return _orchestrate(ctx)


def replan_from(result, index, new_choice):
    """Override hook (mirrors ``auto_lucj.reoptimize_after_change``).

    Keep circuits ``< index`` with their current chosen layout, substitute
    ``new_choice`` (one of circuit ``index``'s strata entries) at ``index``,
    and re-sweep every circuit ``> index`` — their regions freed / re-excluded
    to match the new fixed prefix. Pure: reads ``result`` (and the ``_context``
    it carries), returns a fresh result dict of the same shape.
    """
    ctx = result['_context']
    circuits = result['circuits']
    if not (0 <= index < len(circuits)):
        raise IndexError(f"replan index {index} out of range "
                         f"(have {len(circuits)} placed circuits)")
    prefix = [(circuits[i], circuits[i]['chosen']) for i in range(index)]
    prefix.append((circuits[index], new_choice))
    return _orchestrate(ctx, prefix=prefix)
