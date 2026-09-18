"""Quota-balanced diverse top-N slicing.

Unit tests for the ranking-agnostic selector `_select_with_diversity` (the
quota math) plus an engine byte-compat pin for diversity_threshold=None.
"""
from types import SimpleNamespace

import rustworkx

from iqmapper.ancilla_sweep import (
    _jaccard, _select_with_diversity, sweep_ancilla_layouts,
)


def _mk(qubits):
    """A candidate entry with a given qubit footprint (split alpha/beta)."""
    q = list(qubits)
    h = len(q) // 2
    return {'alpha': q[:h], 'beta': q[h:], 'ancillas': []}


# ── Jaccard sanity ──────────────────────────────────────────────────────────

def test_jaccard_basics():
    assert _jaccard(frozenset(), frozenset()) == 0.0
    assert _jaccard(frozenset({1, 2}), frozenset({1, 2})) == 1.0
    assert _jaccard(frozenset({1, 2, 3}), frozenset({1, 2, 4})) == 0.5


# ── quota picks a different-region entry for the reserved slot ──────────

def test_quota_reserves_slot_for_diverse_region():
    e1 = _mk(range(10))                       # {0..9}
    e2 = _mk(list(range(9)) + [10])           # {0..8,10} — Jaccard(e1)=9/11≈.82
    e3 = _mk(list(range(9)) + [11])
    e4 = _mk(list(range(9)) + [12])
    e5 = _mk(range(20, 30))                   # {20..29} — disjoint
    out = _select_with_diversity([e1, e2, e3, e4, e5], top_n=3, threshold=0.8)

    assert out == [e1, e2, e5]                # slots 1,2 = best; slot 3 = elsewhere
    assert e5['diverse'] is True
    assert e1['diverse'] is False and e2['diverse'] is False


# ── all similar → fallback to plain top-3, nothing marked diverse ───────

def test_all_similar_falls_back_to_plain():
    es = [_mk(list(range(9)) + [10 + i]) for i in range(5)]   # all ≈ {0..8}+1
    out = _select_with_diversity(es, top_n=3, threshold=0.8)
    assert out == es[:3]                      # identical to plain slicing
    assert all(e['diverse'] is False for e in out)   # no genuine diversity pick


# ── threshold→0.99 makes ~everything "dissimilar" → plain slicing ───────

def test_high_threshold_behaves_like_plain_slice():
    e1 = _mk(range(10))
    e2 = _mk(list(range(9)) + [10])
    e3 = _mk(list(range(9)) + [11])
    e4 = _mk(range(20, 30))
    plain = [e1, e2, e3, e4][:3]
    out = _select_with_diversity([e1, e2, e3, e4], top_n=3, threshold=0.99)
    # Same ENTRIES as a plain top-3 slice (0.99 admits the next-best).
    assert out == plain


def test_never_returns_fewer_than_pool():
    e1, e2 = _mk(range(10)), _mk(range(20, 30))
    assert _select_with_diversity([e1, e2], top_n=5, threshold=0.8) == [e1, e2]
    assert _select_with_diversity([], top_n=3, threshold=0.8) == []


# ── diversity_threshold=None is byte-compatible (no 'diverse' key) ──────

def _make_backend():
    edges = []
    for base in (0, 24):
        top = [base + j for j in range(8)]
        bot = [base + 8 + j for j in range(8)]
        brg = [base + 16 + j for j in range(8)]
        for j in range(7):
            edges.append((top[j], top[j + 1]))
            edges.append((bot[j], bot[j + 1]))
        for j in range(8):
            edges.append((top[j], brg[j]))
            edges.append((brg[j], bot[j]))
    g = rustworkx.PyDiGraph()
    for i in range(48):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'], name='b')


class _Props:
    def readout_error(self, q):
        return 0.01

    def gate_error(self, name, edge):
        return 0.005


def _all_entries(res):
    return [e for entries in res['strata'].values() for e in entries]


def test_none_threshold_is_byte_compatible():
    bk, pr = _make_backend(), _Props()
    res = sweep_ancilla_layouts(8, bk, pr, min_bridges=1, top_n=3,
                                shuffle_seed=None, diversity_threshold=None)
    assert all('diverse' not in e for e in _all_entries(res))   # no key added


def test_threshold_set_marks_entries():
    bk, pr = _make_backend(), _Props()
    res = sweep_ancilla_layouts(8, bk, pr, min_bridges=1, top_n=3,
                                shuffle_seed=None, diversity_threshold=0.8)
    # Every returned entry now carries the (bool) diverse flag.
    entries = _all_entries(res)
    assert entries and all('diverse' in e for e in entries)
