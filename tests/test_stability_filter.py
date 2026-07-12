"""Tests for qpu_mapper.stability_filter — the stability-aware sweep.

Offline, deterministic. Two disconnected width-8 heavy-hex-like blocks
(the norb=8 host from the sweep tests): block A = ids 0..23, block B =
ids 24..47. Synthetic calibration history is crafted so specific qubits
spike (volatile). Real VF2 runs — the graphs are tiny — but there is no
qiskit dependency and no network.
"""
from types import SimpleNamespace

import rustworkx

from qpu_mapper.ancilla_sweep import sweep_ancilla_layouts
from qpu_mapper.stability_filter import (
    compute_qubit_volatility, select_excluded, sweep_with_stability,
)


# ── synthetic backend / props (mirrors the ancilla-sweep test host) ─────────

def _build_block(base, edges):
    top = [base + j for j in range(8)]
    bot = [base + 8 + j for j in range(8)]
    brg = [base + 16 + j for j in range(8)]
    for j in range(7):
        edges.append((top[j], top[j + 1]))
        edges.append((bot[j], bot[j + 1]))
    for j in range(8):
        edges.append((top[j], brg[j]))
        edges.append((brg[j], bot[j]))


def _make_backend():
    edges = []
    _build_block(0, edges)     # block A → ids 0..23
    _build_block(24, edges)    # block B → ids 24..47
    g = rustworkx.PyDiGraph()
    for i in range(48):
        g.add_node(i)
    for u, v in edges:
        g.add_edge(u, v, None)
    return SimpleNamespace(coupling_map=SimpleNamespace(graph=g),
                           operation_names=['cz', 'sx', 'rz', 'x'])


class _Props:
    def readout_error(self, q):
        return 0.01

    def gate_error(self, name, edge):
        return 0.005


def _history(spikes, n_dates=6, n_qubits=48):
    """History where every qubit is flat 0.01 except `spikes` {qid: value},
    which get a single high sample on the last date → a high volatility score."""
    dates = [f"2026-07-{i + 1:02d}" for i in range(n_dates)]
    hist = {}
    for di, d in enumerate(dates):
        arr = [0.01] * n_qubits
        if di == n_dates - 1:
            for q, val in spikes.items():
                arr[q] = val
        hist[d] = {'qubit_props': {'readout_error': arr}, 'gate_props': {}}
    return hist


def _all_qubits(res):
    used = set()
    for entries in res['strata'].values():
        for e in entries:
            used.update(e['alpha'])
            used.update(e['beta'])
            used.update(e['ancillas'])
    return used


# ── t1: volatility scoring ──────────────────────────────────────────────────

def test_volatility_orders_volatile_above_stable():
    # q3 spikes hard, q10 mildly, q5 flat.
    hist = _history({3: 0.06, 10: 0.02})
    vol = compute_qubit_volatility(hist, window_days=14)
    assert vol[3] > vol[10] > vol[5]
    assert vol[5] == 0.0                       # flat series → no volatility


# ── t2: select_excluded ─────────────────────────────────────────────────────

def test_select_excluded_threshold_and_order():
    vol = {3: 5.0, 2: 4.5, 10: 2.0, 5: 0.0, 7: 0.8}
    excluded = select_excluded(vol, threshold=1.0)
    assert excluded == [3, 2, 10]              # >1.0, most-volatile first
    assert 5 not in excluded and 7 not in excluded


# ── t3: filtered sweep never uses excluded qubits (A1 correctness) ──────────

def test_filtered_sweep_excludes_qubits():
    backend, props = _make_backend(), _Props()
    res = sweep_ancilla_layouts(8, backend, props, min_bridges=1,
                                exclude_qubits={4, 5})
    assert res['excluded_qubits'] == [4, 5]
    used = _all_qubits(res)
    assert 4 not in used and 5 not in used
    assert res['strata']                        # still found layouts (block B)


# ── t4: RELAX VALVE (the crux) ──────────────────────────────────────────────

def test_relax_valve_readmits_least_volatile():
    backend, props = _make_backend(), _Props()
    # Block A top row (0..3) + block B top row (24, 25) all volatile. Excluding
    # ALL breaks BOTH blocks; re-admitting the least-volatile (25 then 24)
    # restores block B while the most-volatile (3, block A) stays excluded.
    hist = _history({3: 0.06, 2: 0.055, 1: 0.05, 0: 0.045, 24: 0.03, 25: 0.025})
    res = sweep_with_stability(8, backend, props, hist,
                               window_days=14, threshold=1.0, min_bridges=1)

    st = res['stability']
    assert st['available'] is True
    assert st['relaxed'] is True
    assert st['n_readmitted'] > 0
    assert st['n_excluded_initial'] == 6
    assert 3 in st['excluded_final']            # most-volatile survived
    assert st['excluded_final'] == sorted(st['excluded_final'])
    assert res['strata']                        # sweep succeeded after relaxing
    # The pruned-out qubits never appear in a returned candidate.
    used = _all_qubits(res)
    assert used.isdisjoint(set(st['excluded_final']))


# ── t5: no history → plain sweep, stability unavailable ─────────────────────

def test_no_history_runs_plain_sweep():
    backend, props = _make_backend(), _Props()
    for empty in (None, {}):
        res = sweep_with_stability(8, backend, props, empty, min_bridges=1)
        assert res['stability'] == {'available': False}
        assert res['strata']


# ── t6: A1 non-regression — exclude_qubits=None path ────────────────────────

def test_a1_default_is_unfiltered():
    backend, props = _make_backend(), _Props()
    base = sweep_ancilla_layouts(8, backend, props, min_bridges=1)
    explicit_none = sweep_ancilla_layouts(8, backend, props, min_bridges=1,
                                          exclude_qubits=None)
    assert base['excluded_qubits'] == []
    assert explicit_none['excluded_qubits'] == []
    # Same strata shape (k range) with and without the (empty) exclusion.
    assert list(base['strata'].keys()) == list(explicit_none['strata'].keys())


# ── t7: metadata — sorted excluded_final, vf2_truncated propagated ──────────

def test_metadata_and_truncation_propagation():
    backend, props = _make_backend(), _Props()
    # A couple of volatile block-B bridge qubits (not chain-critical), plus a
    # cap of 1 so VF2 truncates — the flag must survive the orchestrator.
    hist = _history({40: 0.06, 41: 0.05})
    res = sweep_with_stability(8, backend, props, hist, window_days=14,
                               threshold=1.0, min_bridges=1, mapping_cap=1)
    assert res['vf2_truncated'] is True                 # propagated untouched
    st = res['stability']
    assert st['available'] is True
    assert st['excluded_final'] == sorted(st['excluded_final'])
    assert res['excluded_qubits'] == sorted(res['excluded_qubits'])
