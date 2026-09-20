"""
reservoir.py — make an enumeration cap count DISTINCT candidates.

The problem
-----------
`auto_lucj._collect_vf2_mappings` takes the first `cap` mappings rustworkx
yields and stops. VF2 walks the graph in a deterministic traversal order, so
those first mappings are spatially clustered: the same region of the chip with
one qubit shifted, over and over. Measured on a 193-qubit heavy-hex at
norb=8, k=1 (37,044 mappings available, cap 2000 = 5.4 %):

    kept set                mean pairwise Jaccard    qubits covered
    first 2000 (the cap)            0.161                109 / 193
    random 2000                     0.068                193 / 193
    all 37,044                      0.069                193 / 193

The cap's sample is 2.3x more self-similar than a representative one and
never sees 44 % of the chip. The downstream cascade, monotonicity anchors and
diversity quota are all good, but they can only choose among what enumeration
handed them.

Exact duplicates are NOT the problem: each scaffold appears exactly twice (the
alpha/beta mirror) and the sweep already collapses that. Near-duplicates are.

What this does
--------------
A `DiverseReservoir` sits between VF2 and the cap. Every mapping is offered to
it; the cap counts only what it KEEPS. A mapping that overlaps a kept one by
Jaccard >= 0.8 and does not beat it on the lightweight score is discarded
without consuming the cap; one that beats it takes its place. So the 2000
slots hold 2000 genuinely different candidates rather than 2000 views of the
same few regions.

WHERE IT ENGAGES
----------------
Only on the HYBRID path, and that is not an accident of implementation but a
consequence of the measurement above. At k_max the enumeration yields far
fewer mappings than the cap (908 on that device), so nothing is ever
discarded there and the reservoir is a no-op. The cap is only reached at low
k — k=1 and k=0 — which the sweep reaches through its hybrid per-stratum VF2
pass. With `hybrid=False` the low-k strata are derived from the k_max
survivors by bridge subtraction, no low-k enumeration happens, and the
reservoir never engages.

This also disposes of the obvious soundness worry. A reservoir sieving at
k_max on the k_max score could in principle discard a scaffold that is weak
with all bridges but excellent with one — and the cascade would then never
see it. It cannot happen, because at k_max the cap is not reached. Where the
reservoir does bite, the stratum is being enumerated directly and scored at
its own k.

THE SCORE IS THE SWEEP'S
------------------------
`score_fn` must be the same lightweight product the sweep computes in
`ancilla_sweep._add_variant`:

    product = prod(1 - readout[q]) over the 2*norb CHAIN qubits
            * prod(1 - cz_err[e]) over the 2*(norb-1) chain edges
            * prod over bridges of (1 - r)(1 - e1)(1 - e2)

Note the chain readout term covers the chain only — a bridge qubit's readout
enters through its own bridge factor, not through `cr_prod`. The sweep hoists
`readout` and `edge_err` (keyed by `frozenset`, so direction is moot) and the
reservoir is handed closures over those same arrays, so there is one formula
and it cannot drift. `tests/test_reservoir.py` pins the equality to 1e-12.

The circuit-level LQS is untouched: it remains the final ranking, applied
downstream to whatever this reservoir let through.
"""
import logging

_log = logging.getLogger(__name__)

#: Two layouts sharing at least this fraction of their qubits are "the same
#: region" for sieving purposes. 0.8 matches the sweep's own diversity quota
#: (`ancilla_sweep._select_with_diversity`), so enumeration and selection
#: agree on what counts as a different region.
DEFAULT_JACCARD = 0.8

#: Mappings considered per kept slot. The cap bounds what is HELD; this
#: bounds what is EXAMINED. Iterating VF2 is cheap (37k mappings in ~0.1 s on
#: a 193-qubit heavy-hex) — it is materialising and scoring them that the cap
#: was protecting against — so a generous factor buys most of the available
#: diversity at negligible cost.
OFFER_BUDGET_FACTOR = 20


class _Entry:
    __slots__ = ('mapping', 'layout', 'qubits', 'score', 'key')

    def __init__(self, mapping, layout, qubits, score, key):
        self.mapping = mapping
        self.layout = layout
        self.qubits = qubits
        self.score = score
        self.key = key


class DiverseReservoir:
    """Keeps up to `cap` distinct, non-dominated candidates.

    Args:
        cap: how many candidates to keep — what the caller's cap now counts.
        score_fn: ``callable(layout) -> float``, the lightweight product
            proxy. Higher is better.
        to_layout: ``callable(mapping) -> list[int] | None``, converting a VF2
            mapping to a physical layout. Return None for an incomplete
            mapping; it is counted as offered and otherwise ignored.
        jaccard: similarity threshold, or None to disable similarity
            filtering entirely (plain cap behaviour, for the parity path).
        key_fn: ``callable(layout) -> hashable`` for exact-duplicate
            detection. Defaults to `tuple(layout)` — the exact assignment.
            Deliberately NOT `frozenset(layout)`: that conflates the
            alpha/beta mirror, and the sweep must still see both members of
            every mirror pair (see `offer`).
    """

    def __init__(self, cap, score_fn, to_layout, *, jaccard=DEFAULT_JACCARD,
                 key_fn=None, offer_budget=None):
        self.cap = int(cap)
        # How many mappings to CONSIDER. Filling stops at `cap`, but the
        # reservoir must keep seeing candidates after that or replacement can
        # never happen and it degenerates to the plain cap. The cap bounds
        # memory (what we keep); this bounds work (what we look at).
        self.offer_budget = (int(offer_budget) if offer_budget is not None
                             else OFFER_BUDGET_FACTOR * int(cap))
        self._score_fn = score_fn
        self._to_layout = to_layout
        self._jaccard = jaccard
        self._key_fn = key_fn or (lambda layout: tuple(layout))
        self._entries = []
        self._by_qubit = {}                   # physical qubit -> {entry idx}
        self._keys = {}                       # exact key -> entry index
        self._stats = {'offered': 0, 'kept': 0, 'replaced': 0,
                       'skipped_similar': 0, 'skipped_duplicate': 0,
                       'incomplete': 0}

    # ── read-only surface ───────────────────────────────────────
    def __len__(self):
        return len(self._entries)

    @property
    def mappings(self):
        """Kept mappings, in the order they were kept."""
        return [e.mapping for e in self._entries]

    @property
    def layouts(self):
        return [e.layout for e in self._entries]

    @property
    def stats(self):
        return dict(self._stats)

    @property
    def exhausted(self):
        """True once the offer budget is spent — the caller should stop."""
        return self._stats['offered'] >= self.offer_budget

    # ── the sieve ───────────────────────────────────────────────
    def offer(self, mapping):
        """Consider one mapping. Returns what happened to it.

        One of 'kept', 'replaced', 'skipped_similar', 'skipped_duplicate' or
        'incomplete'. Only 'kept' grows `len(self)`.

        FILL FIRST, THEN SIEVE. Below the cap every mapping is kept
        unconditionally; the sieve engages only once the reservoir is full.
        That is not a refinement, it is required for correctness: the sweep
        observes the raw enumeration through `n_raw_embeddings` and through
        each entry's `source` ('vf2' vs 'derived'), and it relies on seeing
        BOTH members of every alpha/beta mirror pair — which have Jaccard
        1.0 and would be the first thing any similarity filter removed.
        `tests/test_ancilla_sweep.py::test_dedupe` pins exactly that. Sieving
        unconditionally also discards candidates for no gain when there is no
        scarcity, and the product score is only a proxy for the circuit-level
        LQS that ranks them later — so a near-duplicate dropped here can be
        one the real objective preferred.

        Once full, a candidate must earn its slot: it either beats the kept
        candidate it most resembles, or — being a genuinely new region — it
        beats the weakest thing currently held.
        """
        self._stats['offered'] += 1
        layout = self._to_layout(mapping)
        if not layout or any(q is None for q in layout):
            self._stats['incomplete'] += 1
            return 'incomplete'

        key = self._key_fn(layout)
        if key in self._keys:
            self._stats['skipped_duplicate'] += 1
            return 'skipped_duplicate'

        qubits = frozenset(layout)
        score = float(self._score_fn(layout))

        if len(self._entries) < self.cap:
            return self._keep(mapping, layout, qubits, score, key)

        if self._jaccard is None:
            self._stats['skipped_similar'] += 1
            return 'skipped_similar'

        rival_idx = self._most_similar(qubits)
        if rival_idx is not None:
            rival = self._entries[rival_idx]
            overlap = (len(qubits & rival.qubits)
                       / len(qubits | rival.qubits))
            if overlap >= self._jaccard:
                # Same region: keep only the better of the two, so a
                # near-duplicate can never occupy a slot it did not earn.
                if score > rival.score:
                    self._replace(rival_idx, mapping, layout, qubits, score,
                                  key)
                    return 'replaced'
                self._stats['skipped_similar'] += 1
                return 'skipped_similar'

        # A genuinely new region, and we are full. Displace the weakest
        # candidate held — this is what lets coverage keep growing after the
        # cap is reached, instead of freezing on whichever corner of the chip
        # VF2 happened to traverse first.
        weakest_idx = self._weakest()
        if weakest_idx is not None and score > self._entries[weakest_idx].score:
            self._replace(weakest_idx, mapping, layout, qubits, score, key)
            return 'replaced'
        self._stats['skipped_similar'] += 1
        return 'skipped_similar'

    # ── internals ───────────────────────────────────────────────
    def _most_similar(self, qubits):
        """Index of the kept entry sharing the most qubits, or None.

        Only entries sharing at least one qubit can reach the threshold, so
        the inverted index bounds the work to those — this is O(#kept
        sharing a qubit), not O(#kept).
        """
        counts = {}
        for q in qubits:
            for idx in self._by_qubit.get(q, ()):
                counts[idx] = counts.get(idx, 0) + 1
        if not counts:
            return None
        # Deterministic: most shared qubits, then earliest kept.
        return min(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]

    def _weakest(self):
        """Index of the lowest-scoring kept entry. Deterministic ties.

        Only consulted when the reservoir is full AND the candidate is a new
        region, which is the rare case — the common one is a near-duplicate,
        resolved by the inverted index above.
        """
        if not self._entries:
            return None
        return min(range(len(self._entries)),
                   key=lambda i: (self._entries[i].score, i))

    def _keep(self, mapping, layout, qubits, score, key):
        idx = len(self._entries)
        self._entries.append(_Entry(mapping, layout, qubits, score, key))
        for q in qubits:
            self._by_qubit.setdefault(q, set()).add(idx)
        self._keys[key] = idx
        self._stats['kept'] += 1
        return 'kept'

    def _replace(self, idx, mapping, layout, qubits, score, key):
        old = self._entries[idx]
        for q in old.qubits:
            bucket = self._by_qubit.get(q)
            if bucket is not None:
                bucket.discard(idx)
                if not bucket:
                    del self._by_qubit[q]
        self._keys.pop(old.key, None)

        self._entries[idx] = _Entry(mapping, layout, qubits, score, key)
        for q in qubits:
            self._by_qubit.setdefault(q, set()).add(idx)
        self._keys[key] = idx
        self._stats['replaced'] += 1

    def __repr__(self):                       # pragma: no cover - debug aid
        return (f'<DiverseReservoir {len(self)}/{self.cap} kept, '
                f"{self._stats['skipped_similar']} near-duplicates skipped>")


def summarise(stats, k=None, hybrid=True):
    """The one-line notice the sweep surfaces when a cap was reached.

    Names the hybrid stratum explicitly, because that is the only place the
    reservoir engages and a user comparing runs needs to know which pass the
    number belongs to.
    """
    if not stats or not stats.get('kept'):
        return None
    where = f'hybrid k={k}' if (hybrid and k is not None) else f'k={k}'
    return (f'{where}: cap reached — kept {stats["kept"]:,} diverse '
            f'candidates, skipped {stats["skipped_similar"]:,} '
            f'near-duplicates')
