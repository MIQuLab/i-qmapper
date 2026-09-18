"""The cascade invariant (every direct stratum seeds all lower strata).

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

from iqmapper.ancilla_sweep import sweep_ancilla_layouts

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
    # shuffle_seed=None pins the NATURAL node order this fixture's truncation
    # story depends on (poor region enumerated first). The cascade is what's
    # under test here; the shuffle has its own suite (test_shuffle_search.py).
    return sweep_ancilla_layouts(NORB, _make_backend(), _Props(),
                                 min_bridges=1, top_n=50, mapping_cap=CAP,
                                 shuffle_seed=None)


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
    # Observed on hardware: a 3-bridge prefix scoring 67.8% from a 4-bridge
    # parent at 67.2% — dropping a bridge's error factors can only raise the
    # product, so a prefix child is never worse than its direct parent.
    assert k1['product_score'] >= k2['product_score'] - 1e-12


# ── the actual cap is surfaced for an honest truncation banner ──────────

def test_result_exposes_mapping_cap_for_the_banner():
    res = _run()
    assert res['mapping_cap'] == CAP
    # per-stratum truncation flags say WHICH strata hit it (k=1 did, k=2 didn't).
    assert res['per_stratum'][1]['truncated'] is True
    assert res['per_stratum'][2]['truncated'] is False


# ── monotonicity on the DISPLAYED metric (product ≠ circuit scale) ───────
#
# The live bug: strata are sliced to top_n by product_score BEFORE the UI's
# lqs_circuit is computed. Where the scales diverge, the true circuit-best
# falls below the product line unseen and the status line reads e.g.
# "3→49.3% · 2→40.8%" — non-monotone on the DISPLAYED metric.
#
# Fixture with the scales deliberately opposed:
#   * region D (low ids): width-16 ladder, bridges cols 0..7 → k=2 only, MANY
#     decoys, props error 0.001 → HIGH product_score (product's favourites).
#   * region G (high ids): width-12 ladder, bridges cols 0,4,8 → hosts k=3,
#     props error 0.02 → LOW product_score, but the canned circuit scorer rates
#     G's qubits ~1.0 and D's ~0.5 → G is the DISPLAYED-metric winner.
# The k=3 winner is in G; its k=2 prefix (the monotonicity anchor) has product
# BELOW every D decoy, so product-only slicing drops it — the widened-window slice must keep it in the
# scoring window and surface it once ranked by lqs_circuit.

_WG = set(range(100, 127))          # region G ids


def _w2_ladder(edges, w_top, w_bot, brg_cols, bt, bb, bg):
    t = [bt + j for j in range(w_top)]
    b = [bb + j for j in range(w_bot)]
    for j in range(w_top - 1):
        edges.append((t[j], t[j + 1]))
    for j in range(w_bot - 1):
        edges.append((b[j], b[j + 1]))
    for i, c in enumerate(brg_cols):
        edges.append((t[c], bg + i))
        edges.append((bg + i, b[c]))


def _w2_backend():
    edges = []
    _w2_ladder(edges, 16, 16, list(range(8)), 0, 16, 32)     # D: k=2 decoys
    _w2_ladder(edges, 12, 12, [0, 4, 8], 100, 112, 124)      # G: hosts k=3
    g = rustworkx.PyDiGraph()
    for i in range(127):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'], name='chip')


class _W2Props:                     # product_score PRE-FILTER favours D
    def readout_error(self, q):
        return 0.02 if q in _WG else 0.001

    def gate_error(self, name, edge):
        u, v = edge
        return 0.02 if (u in _WG and v in _WG) else 0.001


def _w2_circuit(entry):             # DISPLAYED metric favours G, fewer bridges higher
    qs = entry['alpha'] + entry['beta'] + entry['ancillas']
    f = 1.0
    for q in qs:
        f *= 0.999 if q in _WG else 0.5
    return f


def _w2_region(entry):
    return 'G' if set(entry['alpha'] + entry['beta'] + entry['ancillas']) <= _WG else 'D'


def _w2_run(**kw):
    return sweep_ancilla_layouts(NORB, _w2_backend(), _W2Props(), min_bridges=1,
                                 top_n=3, shuffle_seed=None, **kw)


def test_displayed_metric_best_per_k_is_monotone():
    res = _w2_run(circuit_score_fn=_w2_circuit)
    ks = sorted(res['strata'], reverse=True)
    bests = [res['strata'][k][0]['lqs_circuit'] for k in ks]
    for hi, lo in zip(bests, bests[1:]):
        assert lo >= hi - 1e-9, f"displayed-metric best fell: {bests} for {ks}"
    # Each stratum's displayed winner is region G (the circuit-good region).
    for k in ks:
        assert _w2_region(res['strata'][k][0]) == 'G'


def test_slice_contains_higher_winners_prefix_anchor():
    res = _w2_run(circuit_score_fn=_w2_circuit)
    k3 = res['strata'][3][0]
    assert _w2_region(k3) == 'G'
    chain, anc2 = frozenset(k3['layout']), frozenset(k3['ancillas'][:2])
    # The k=3 winner's k=2 prefix survives into the k=2 slice (anchor),
    # and it is the displayed winner there (circuit >= its k=3 parent).
    k2_top = res['strata'][2][0]
    assert frozenset(k2_top['layout']) == chain
    assert frozenset(k2_top['ancillas']) == anc2
    assert k2_top['lqs_circuit'] >= k3['lqs_circuit'] - 1e-9


def test_product_only_slice_would_have_hidden_the_anchor():
    # The divergence is real: without the circuit scorer, product-only slicing
    # fills the k=2 top_n with the low-error D decoys — the G anchor is nowhere
    # in the displayed slice. This is exactly what the widened window rescues.
    res = _w2_run()                                   # no circuit_score_fn
    regions = [_w2_region(e) for e in res['strata'][2]]
    assert regions == ['D', 'D', 'D']
    assert 'lqs_circuit' not in res['strata'][2][0]   # byte-compat shape intact
