"""Shuffled VF2 search order kills the geographic-prefix truncation bias.

Deterministic, offline, real VF2. Two identical blocks on one chip:
  * block A (ids 0..23)  — NOISY  (readout/gate error 0.06)
  * block B (ids 24..47) — CLEAN  (error 0.0005)
Both host the norb=8 zigzag. VF2 enumerates in node-index order, so with a
tiny mapping_cap the NATURAL search truncates inside block A and never scores
a block-B layout — the best it can return is noisy. A seeded shuffle permutes
the node insertion order, samples block B within the same cap, and returns the
clean layout. Physical ids ride in node data and are recovered order-
independently, so a shuffled result still names real chip qubits.
"""
from types import SimpleNamespace

import rustworkx

from iqmapper.ancilla_sweep import sweep_ancilla_layouts

NORB = 8
CAP = 2                      # smaller than block A's embedding count → truncates
CLEAN = set(range(24, 48))   # block B ids
SEED = 0                     # empirically samples block B (see probe)


def _block(base, edges):
    top = [base + j for j in range(8)]
    bot = [base + 8 + j for j in range(8)]
    brg = [base + 16 + j for j in range(8)]
    for j in range(7):
        edges.append((top[j], top[j + 1]))
        edges.append((bot[j], bot[j + 1]))
    for j in range(8):
        edges.append((top[j], brg[j]))
        edges.append((brg[j], bot[j]))


def _backend():
    edges = []
    _block(0, edges)     # A: noisy, low ids → enumerated first
    _block(24, edges)    # B: clean, high ids → beyond the cap in natural order
    g = rustworkx.PyDiGraph()
    for i in range(48):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'], name='chip')


class _Props:
    def readout_error(self, q):
        return 0.0005 if q in CLEAN else 0.06

    def gate_error(self, name, edge):
        u, v = edge
        return 0.0005 if (u in CLEAN and v in CLEAN) else 0.06


def _sweep(seed):
    return sweep_ancilla_layouts(NORB, _backend(), _Props(), min_bridges=1,
                                 top_n=10, mapping_cap=CAP, shuffle_seed=seed)


def _best(res, k):
    return max(res['strata'][k], key=lambda e: e['product_score'])


def _qubits(e):
    return set(e['alpha']) | set(e['beta']) | set(e['ancillas'])


# ── natural order misses the clean region; the shuffle finds it ─────────

def test_shuffle_reaches_a_region_natural_order_truncates_past():
    nat = _sweep(None)
    shuf = _sweep(SEED)

    nat_best = _best(nat, 1)
    shuf_best = _best(shuf, 1)

    # Natural order is trapped in the noisy block (truncated before block B).
    assert _qubits(nat_best).isdisjoint(CLEAN)
    assert nat['per_stratum'][1]['shuffled'] is False

    # The shuffle samples the clean block and returns a strictly better layout.
    assert _qubits(shuf_best) <= CLEAN
    assert shuf['per_stratum'][1]['shuffled'] is True
    assert shuf_best['product_score'] > nat_best['product_score'] + 1e-6


# ── seed=None is the pinned natural-order (byte-compat) path ────────────

def test_seed_none_is_natural_order_and_deterministic():
    a = _sweep(None)
    b = _sweep(None)
    # No shuffle anywhere, and the run is reproducible bit-for-bit.
    assert all(not m['shuffled'] for m in a['per_stratum'].values())
    for k in a['strata']:
        assert [e['product_score'] for e in a['strata'][k]] == \
               [e['product_score'] for e in b['strata'][k]]
        assert [e['ancillas'] for e in a['strata'][k]] == \
               [e['ancillas'] for e in b['strata'][k]]
    # And it is the NOISY solution — natural order never left block A.
    assert _qubits(_best(a, 1)).isdisjoint(CLEAN)


# ── same seed → identical results (reproducibility) ─────────────────────

def test_same_seed_is_reproducible():
    a = _sweep(SEED)
    b = _sweep(SEED)
    assert list(a['strata']) == list(b['strata'])
    for k in a['strata']:
        ea, eb = a['strata'][k], b['strata'][k]
        assert [e['product_score'] for e in ea] == [e['product_score'] for e in eb]
        assert [e['layout'] for e in ea] == [e['layout'] for e in eb]
        assert [e['ancillas'] for e in ea] == [e['ancillas'] for e in eb]


# ── shuffled results still name REAL chip ids (no relabel leakage) ──────

def test_shuffle_recovers_physical_ids():
    shuf = _sweep(SEED)
    chip = set(range(48))
    for entries in shuf['strata'].values():
        for e in entries:
            qs = e['alpha'] + e['beta'] + e['ancillas']
            assert all(q in chip for q in qs)          # real physical ids
            assert len(set(qs)) == len(qs)             # no qubit reused
            # chains are length norb; ancillas match the stratum's bridge count
            assert len(e['alpha']) == NORB and len(e['beta']) == NORB
            assert len(e['ancillas']) == len(e['bridge_slots'])
