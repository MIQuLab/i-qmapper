"""Tests for qpu_mapper.ancilla_sweep — the one-shot bridge-subtraction sweep.

Deterministic and fully offline (no qiskit, no network). A synthetic backend
is built from two disconnected width-8 heavy-hex-like "ladder" blocks:

    top chain    t0 - t1 - ... - t7          (8 qubits)
    bridges      | r0   | r1        | r7      (one bridge qubit per column)
    bottom chain b0 - b1 - ... - b7          (8 qubits)

Each bridge qubit rj connects tj and bj (a 3-node path tj-rj-bj), exactly the
shape auto_lucj's zigzag template expects. Two such blocks are built:

    * GOOD block  — physical qubit ids 0..23,  low error everywhere.
    * BAD  block  — physical qubit ids 24..47, high error everywhere.

For norb=8 the geometric ceiling is k_max = ceil(8/4) = 2 bridges (orbital
slots 0 and 4), so k_max_effective == 2 and a min_bridges=1 sweep produces two
strata (k=2 intact VF2 layouts, k=1 subtraction variants).
"""

import time

import pytest
import rustworkx

from types import SimpleNamespace

from qpu_mapper.ancilla_sweep import sweep_ancilla_layouts

GOOD_QUBITS = set(range(24))   # ids 0..23 are the low-error "good region"


# ── synthetic backend / props ──────────────────────────────────────────────

def _build_block(base, edges):
    """Append a width-8 ladder block based at physical id `base`.

    Ids: top base..base+7, bottom base+8..base+15, bridges base+16..base+23.
    """
    top = [base + j for j in range(8)]
    bot = [base + 8 + j for j in range(8)]
    brg = [base + 16 + j for j in range(8)]
    for j in range(7):
        edges.append((top[j], top[j + 1]))
        edges.append((bot[j], bot[j + 1]))
    for j in range(8):
        edges.append((top[j], brg[j]))
        edges.append((brg[j], bot[j]))


def make_backend():
    """SimpleNamespace exposing exactly what _make_backend_cmap_pygraph and
    _get_two_q_gate_name read: a directed rustworkx coupling graph and the
    backend operation names (with a 2Q 'cz')."""
    edges = []
    _build_block(0, edges)    # GOOD block  → ids 0..23
    _build_block(24, edges)   # BAD  block  → ids 24..47
    graph = rustworkx.PyDiGraph()
    for i in range(48):
        graph.add_node(i)
    for u, v in edges:
        graph.add_edge(u, v, None)
    coupling_map = SimpleNamespace(graph=graph)
    return SimpleNamespace(coupling_map=coupling_map,
                           operation_names=['cz', 'sx', 'rz', 'x'])


# Error parameters. GOOD block uses these small, IDENTICAL-per-role values so
# that the across-strata per-ancilla factor is exact and unambiguous.
R_CHAIN_GOOD, R_BRIDGE_GOOD = 0.001, 0.003
E_CHAIN_GOOD, E_BRIDGE_GOOD = 0.002, 0.004
R_CHAIN_BAD, R_BRIDGE_BAD = 0.05, 0.06
E_CHAIN_BAD, E_BRIDGE_BAD = 0.05, 0.07


class FakeProps:
    """BackendProperties stand-in: readout_error(q) and gate_error(name, (u,v))
    are all the sweep and its scoring read. Fully calibrated (no NaN)."""

    def readout_error(self, q):
        good = q in GOOD_QUBITS
        is_bridge = (q % 24) >= 16
        if is_bridge:
            return R_BRIDGE_GOOD if good else R_BRIDGE_BAD
        return R_CHAIN_GOOD if good else R_CHAIN_BAD

    def gate_error(self, name, edge):
        u, v = edge
        good = u in GOOD_QUBITS and v in GOOD_QUBITS
        is_bridge = (u % 24) >= 16 or (v % 24) >= 16
        if is_bridge:
            return E_BRIDGE_GOOD if good else E_BRIDGE_BAD
        return E_CHAIN_GOOD if good else E_CHAIN_BAD


ENTRY_FIELDS = {
    'layout', 'alpha', 'beta', 'ancillas', 'bridge_slots',
    'product_score', 'additive_score', 'rank_in_stratum', 'source',
}


def _dedupe_key(entry):
    return (frozenset(entry['alpha']) | frozenset(entry['beta']),
            frozenset(entry['ancillas']))


@pytest.fixture
def backend():
    return make_backend()


@pytest.fixture
def props():
    return FakeProps()


# ── t1: shape ──────────────────────────────────────────────────────────────

def test_shape(backend, props):
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=3)
    assert res['k_max_effective'] == 2
    assert res['min_bridges'] == 1

    # strata keys run k_max_effective..min_bridges, descending.
    assert list(res['strata'].keys()) == [2, 1]

    for k, entries in res['strata'].items():
        assert len(entries) <= 3                      # top_n cap
        assert entries, f"stratum k={k} unexpectedly empty"
        for e in entries:
            assert set(e) == ENTRY_FIELDS             # all documented fields
            assert len(e['ancillas']) == k            # ancilla count == k
            assert len(e['bridge_slots']) == k
            assert e['layout'] == list(e['alpha']) + list(e['beta'])
            assert len(e['alpha']) == 8 and len(e['beta']) == 8
        # rank_in_stratum is 1-based and dense.
        assert [e['rank_in_stratum'] for e in entries] == list(
            range(1, len(entries) + 1))


# ── t2: regression — the good-region k-1 layout is derived for free ─────────

def test_kminus1_variant_present(backend, props):
    """The user's observation: an independent per-k search misses the k-1
    layout that lives inside the best k-max embedding. With subtraction it is
    always in the k-1 pool on the same chain qubits."""
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=1000)
    top = res['strata'][2][0]                         # best intact k=2 layout
    assert top['source'] == 'vf2'
    chain = frozenset(top['alpha']) | frozenset(top['beta'])

    same_chain = [e for e in res['strata'][1]
                  if (frozenset(e['alpha']) | frozenset(e['beta'])) == chain]
    assert same_chain, "k-1 derived variant on the top chain is missing"
    # Everything in the k-1 stratum is a subtraction variant.
    assert all(e['source'] == 'derived' for e in same_chain)
    # The top layout's OWN k-1 variant (a strict subset of its two bridges)
    # must be present. Other same-chain entries may come from the reversed
    # embedding, which shares the chain qubits but keeps different bridges —
    # those are not subsets of top's ancillas, which is exactly why an
    # independent per-k search starting from the rigid template can miss the
    # good one that subtraction recovers here.
    top_ancillas = set(top['ancillas'])
    assert any(set(e['ancillas']) < top_ancillas for e in same_chain), (
        "the top layout's own k-1 subtraction variant is absent")


# ── t3: good region wins every stratum ──────────────────────────────────────

def test_good_region_wins(backend, props):
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=3)
    for k, entries in res['strata'].items():
        best = entries[0]
        used = set(best['alpha']) | set(best['beta']) | set(best['ancillas'])
        assert used <= GOOD_QUBITS, (
            f"stratum k={k} winner {sorted(used)} leaks outside good region")


# ── t4: size-bias guard (within-stratum agreement, across-stratum factor) ───

def test_size_bias_product_vs_additive(backend, props):
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=1000)

    # Within a stratum (fixed size), the product ranking agrees with the
    # additive ranking: entries are ordered by product desc, and in the
    # small-error regime prod(1-x) ~ 1 - sum(x), so additive is non-decreasing
    # down the list.
    for k, entries in res['strata'].items():
        prods = [e['product_score'] for e in entries]
        adds = [e['additive_score'] for e in entries]
        assert prods == sorted(prods, reverse=True)
        assert adds == sorted(adds), (
            f"stratum k={k}: additive not non-decreasing along product order")

    # ACROSS strata, additive is size-biased and must NOT rank: the k=1 best
    # has a SMALLER additive sum than the k=2 best (fewer terms) even though
    # both live in the same good region — proving additive can't compare sizes.
    k2, k1 = res['strata'][2][0], res['strata'][1][0]
    assert k1['additive_score'] < k2['additive_score']

    # product_score IS comparable across strata: adding the second bridge
    # multiplies the k=1 fidelity by exactly that bridge's factor
    # (1-r)(1-e1)(1-e2). Both good-region bridges are identical, so the ratio
    # is exact. Tolerance 1e-9 (float round-off only).
    per_ancilla = (1 - R_BRIDGE_GOOD) * (1 - E_BRIDGE_GOOD) ** 2
    ratio = k1['product_score'] / k2['product_score']
    assert ratio == pytest.approx(1.0 / per_ancilla, abs=1e-9)


# ── t5: dedupe ──────────────────────────────────────────────────────────────

def test_dedupe(backend, props):
    """Alpha/beta-swap embeddings collapse to a single entry per stratum."""
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=1000)

    for k, entries in res['strata'].items():
        keys = [_dedupe_key(e) for e in entries]
        assert len(keys) == len(set(keys)), (
            f"stratum k={k} has duplicate (chain, ancilla) keys")

    # A real collapse happened: VF2 found more raw embeddings than the k=2
    # stratum keeps, because each layout and its alpha/beta swap merge.
    assert res['n_raw_embeddings'] > len(res['strata'][2])


# ── t6: validation ──────────────────────────────────────────────────────────

def test_validation(backend, props):
    with pytest.raises(ValueError):
        sweep_ancilla_layouts(8, backend, props, min_bridges=0)
    with pytest.raises(ValueError):
        sweep_ancilla_layouts(8, backend, props, min_bridges=99)

    # min_bridges == k_max_effective → exactly one stratum, all intact 'vf2'.
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=2, top_n=1000)
    assert list(res['strata'].keys()) == [2]
    assert all(e['source'] == 'vf2' for e in res['strata'][2])


# ── t7: truncation ──────────────────────────────────────────────────────────

def test_truncation(backend, props):
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, mapping_cap=1)
    assert res['vf2_truncated'] is True
    assert res['n_raw_embeddings'] == 1
    # Still well-formed strata from the single embedding.
    assert list(res['strata'].keys()) == [2, 1]
    for k, entries in res['strata'].items():
        assert entries
        for e in entries:
            assert set(e) == ENTRY_FIELDS
            assert len(e['ancillas']) == k


# ── performance smoke (reported wall time) ──────────────────────────────────

def test_perf_smoke(backend, props):
    """Not a hard threshold — exercises the full sweep so pytest --durations
    reports its wall time. The vectorized precompute keeps this well under a
    second on the synthetic graph."""
    t0 = time.perf_counter()
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1, top_n=1000)
    elapsed = time.perf_counter() - t0
    assert res['strata']
    assert elapsed < 5.0, f"sweep too slow: {elapsed:.3f}s"
