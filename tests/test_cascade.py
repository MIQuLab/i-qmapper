"""K2 — the cascade invariant (every direct stratum seeds all lower strata).

Reproduces the live failure shape offline, deterministically, with a real VF2
run on a tiny two-region host:

  * POOR region P (low ids): a width-16 ladder with bridge columns
    {0,1,2,3,4,8}. It hosts the full k_max=3 template (bridges at cols 0,4,8),
    and — crucially — it yields MANY more k=1 embeddings than k=2, so a small
    mapping_cap fills up inside P (in node-index order) before the search ever
    reaches the good region.
  * GOOD region G (high ids): a width-12 ladder with bridge columns {0,4}. It
    hosts k=2 but NOT k=3 (no col-8 bridge), so k_max derivation can never
    reach it — only a direct k'=2 search does. Its qubits have near-zero error.

With mapping_cap=8 the k'=1 DIRECT search truncates inside P and never sees G,
yet G's k=1 layout is still delivered to the k=1 stratum — proving it arrived
by CASCADE from the k'=2 winner, not by a direct search. The best-per-k series
is then monotonically non-decreasing as k decreases, and the delivered prefix
child scores >= its direct parent (fewer error factors can only raise the
product).
"""
from types import SimpleNamespace

import rustworkx

from qpu_mapper.ancilla_sweep import sweep_ancilla_layouts

NORB = 12
GOOD = set(range(100, 126))          # region G qubit ids
CAP = 8                              # small enough to truncate the k=1 direct


def _ladder(edges, width_top, width_bot, brg_cols, base_top, base_bot, base_brg):
    tids = [base_top + j for j in range(width_top)]
    bids = [base_bot + j for j in range(width_bot)]
    for j in range(width_top - 1):
        edges.append((tids[j], tids[j + 1]))
    for j in range(width_bot - 1):
        edges.append((bids[j], bids[j + 1]))
    for i, c in enumerate(brg_cols):
        bq = base_brg + i
        edges.append((tids[c], bq))
        edges.append((bq, bids[c]))


def _make_backend():
    edges = []
    # POOR P (ids 0..47): many k=1 placements, few k=2, one k=3.
    _ladder(edges, 16, 16, [0, 1, 2, 3, 4, 8], 0, 16, 32)
    # GOOD G (ids 100..125): hosts k=2, never k=3; enumerated last.
    _ladder(edges, 12, 12, [0, 4], 100, 112, 124)
    g = rustworkx.PyDiGraph()
    for i in range(126):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'])


class _Props:
    """G is pristine; P is noisy — so any G layout dominates any P layout."""
    def readout_error(self, q):
        return 0.001 if q in GOOD else 0.05

    def gate_error(self, name, edge):
        u, v = edge
        return 0.001 if (u in GOOD and v in GOOD) else 0.05


def _run():
    return sweep_ancilla_layouts(NORB, _make_backend(), _Props(),
                                 min_bridges=1, top_n=50, mapping_cap=CAP)


def _in_good(e):
    return set(e['alpha']) | set(e['beta']) | set(e['ancillas']) <= GOOD


def _best_good(res, k):
    goods = [e for e in res['strata'].get(k, []) if _in_good(e)]
    return max(goods, key=lambda e: e['product_score'], default=None)


# ── the invariant: best-per-k is monotone non-decreasing as k decreases ─────

def test_best_per_k_is_monotone_non_decreasing():
    res = _run()
    ks = sorted(res['strata'], reverse=True)          # high k → low k
    bests = [max(e['product_score'] for e in res['strata'][k]) for k in ks]
    # k_max region is poor; the good region only appears at k<=2, so the
    # series must climb (or hold) as k drops. Never fall.
    for hi, lo in zip(bests, bests[1:]):
        assert lo >= hi - 1e-9, f"best-per-k fell: {bests} for k={ks}"
    # And concretely: the k_max stratum is stuck in the poor region.
    assert not _in_good(max(res['strata'][3], key=lambda e: e['product_score']))


# ── the cascade delivered G's k=1 prefix that no direct k=1 search could ────

def test_cascade_delivers_prefix_a_truncated_direct_search_missed():
    res = _run()
    # The k'=1 DIRECT search hit the cap inside the poor region.
    assert res['per_stratum'][1]['truncated'] is True

    k2 = _best_good(res, 2)
    k1 = _best_good(res, 1)
    assert k2 is not None, "k'=2 direct must have found the good region"
    assert k1 is not None, "the good k=1 prefix must be present"

    # It arrived by cascade (prefix-derived), not a direct VF2 hit.
    assert k1['source'] == 'derived'
    # It IS the prefix of the k=2 winner: same chains, ancillas truncated.
    assert k1['alpha'] == k2['alpha']
    assert k1['beta'] == k2['beta']
    assert k1['ancillas'] == k2['ancillas'][:len(k1['ancillas'])]


def test_prefix_child_scores_at_least_its_direct_parent():
    res = _run()
    k2, k1 = _best_good(res, 2), _best_good(res, 1)
    # The user's live proof: 67.8% child from a 67.2% parent — dropping one
    # bridge's error factors can only raise the product.
    assert k1['product_score'] >= k2['product_score'] - 1e-12


# ── K3: the actual cap is surfaced for an honest truncation banner ──────────

def test_result_exposes_mapping_cap_for_the_banner():
    res = _run()
    assert res['mapping_cap'] == CAP
    # per-stratum truncation flags say WHICH strata hit it (k=1 did, k=2 didn't).
    assert res['per_stratum'][1]['truncated'] is True
    assert res['per_stratum'][2]['truncated'] is False
