"""Tests for iqmapper.props_shim — the cache-aware props adapter.

t1 (the crux): CachedPropsShim built from B0.a-shaped dicts answers
readout_error(q) and gate_error(name, pair) for both edge orders, and
raises a clear KeyError on any missing lookup. Plus a factory cache-hit
reconstruction test with fakes (offline, no qiskit, no network).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from iqmapper.props_shim import CachedPropsShim, make_scan_get_props


def _fixtures():
    # Mirrors data.get_props_for_datetime's shapes: per-qubit arrays and
    # per-edge arrays parallel to `edges`.
    edges = [(0, 1), (1, 2), (2, 3)]
    qubit_props = {
        'readout_error': np.array([0.010, 0.020, 0.030, 0.040]),
        't1': np.array([100.0, 110.0, 120.0, 130.0]),
    }
    gate_props = {
        'cz_error': np.array([0.001, 0.002, 0.003]),
        'gate_length': np.array([200.0, 210.0, 220.0]),
    }
    return qubit_props, gate_props, edges


# ── the crux ────────────────────────────────────────────────────────────

def test_readout_error_by_qubit():
    qp, gp, edges = _fixtures()
    shim = CachedPropsShim(qp, gp, edges)
    assert shim.readout_error(0) == pytest.approx(0.010)
    assert shim.readout_error(3) == pytest.approx(0.040)
    assert isinstance(shim.readout_error(1), float)


def test_gate_error_both_pair_orders():
    qp, gp, edges = _fixtures()
    shim = CachedPropsShim(qp, gp, edges)
    # (1,2) is index 1 → 0.002; reverse must give the same value.
    assert shim.gate_error('cz', (1, 2)) == pytest.approx(0.002)
    assert shim.gate_error('cz', (2, 1)) == pytest.approx(0.002)
    assert shim.gate_error('cz', (0, 1)) == pytest.approx(0.001)
    assert shim.gate_error('cz', (3, 2)) == pytest.approx(0.003)
    # gate_name is informational — a different name still resolves by edge.
    assert shim.gate_error('ecr', (0, 1)) == pytest.approx(0.001)


def test_missing_qubit_raises_keyerror():
    qp, gp, edges = _fixtures()
    shim = CachedPropsShim(qp, gp, edges)
    with pytest.raises(KeyError):
        shim.readout_error(999)


def test_missing_edge_raises_keyerror():
    qp, gp, edges = _fixtures()
    shim = CachedPropsShim(qp, gp, edges)
    with pytest.raises(KeyError):
        shim.gate_error('cz', (0, 3))     # not an edge in the map


def test_missing_prop_key_raises_on_construction():
    qp, gp, edges = _fixtures()
    with pytest.raises(KeyError):
        CachedPropsShim({'t1': qp['t1']}, gp, edges)          # no readout_error
    with pytest.raises(KeyError):
        CachedPropsShim(qp, {'gate_length': gp['gate_length']}, edges)  # no cz_error


def test_custom_gate_key():
    qp = {'readout_error': [0.01, 0.02]}
    gp = {'ecr_error': [0.005]}
    shim = CachedPropsShim(qp, gp, [(0, 1)], gate_key='ecr_error')
    assert shim.gate_error('ecr', (0, 1)) == pytest.approx(0.005)


# ── remote scoring is LIVE — never a cache read ────────────────

def _fake_remote_backend(name='ibm_remote', n=4):
    edges = [(0, 1), (1, 2), (2, 3)]

    class _Props:
        def readout_error(self, q):
            return 0.02
        def gate_error(self, gate_name, edge):
            return 0.003
    return SimpleNamespace(
        name=name, num_qubits=n,
        coupling_map=SimpleNamespace(get_edges=lambda: list(edges)),
        operation_names=['cz', 'sx', 'rz', 'x'],
        properties=lambda: _Props())


def test_remote_scoring_is_live_never_cache_read(monkeypatch):
    from datetime import datetime
    from iqmapper import data
    reads, saves, fetched = [], [], []
    monkeypatch.setattr(data, 'list_known_calibrations',
                        lambda *a, **k: reads.append('list') or [])
    monkeypatch.setattr(data, '_load_calibration',
                        lambda *a, **k: reads.append('load') or None)
    monkeypatch.setattr(data, '_save_calibration',
                        lambda *a, **k: saves.append(a))
    monkeypatch.setattr(data, '_extract_last_update',
                        lambda props: '2026-07-12 09:00:00')

    backend = _fake_remote_backend()
    inner = backend.properties
    backend.properties = lambda: fetched.append(1) or inner()

    state = SimpleNamespace(active_backend_name='current_other')
    props = make_scan_get_props(state, now=datetime(2026, 7, 12, 9, 30))(backend)

    assert fetched == [1]                 # LIVE fetch happened
    assert reads == []                    # ZERO cache-read for scoring
    assert saves                          # warming save still happened
    assert props.vintage == 'live · 09:30'  # live · <fetch time>
    assert props.readout_error(0) == pytest.approx(0.02)


# ── active backend scored on LIVE state props (cache NOT consulted) ──────

def test_active_backend_uses_state_props_not_cache(monkeypatch):
    qp = {'readout_error': np.array([0.01, 0.02, 0.03, 0.04])}
    gp = {'cz_error': np.array([0.001, 0.002, 0.003])}
    edges = [(0, 1), (1, 2), (2, 3)]
    backend = SimpleNamespace(
        name='ibm_active', num_qubits=4,
        coupling_map=SimpleNamespace(get_edges=lambda: list(edges)),
        operation_names=['cz', 'sx', 'rz', 'x'],
        properties=lambda: (_ for _ in ()).throw(
            AssertionError("must not fetch live for active backend")))

    from iqmapper import data
    cache_calls = []
    monkeypatch.setattr(data, 'list_known_calibrations',
                        lambda name, date=None: cache_calls.append(name) or [])
    monkeypatch.setattr(data, '_load_calibration',
                        lambda name, ts: cache_calls.append(('load', name)) or None)

    state = SimpleNamespace(
        active_backend_name='ibm_active',
        qubit_props_data=qp, gate_props_data=gp, edges_list=edges,
        qubit_ids=['0', '1', '2', '3'])
    props = make_scan_get_props(state)(backend)

    assert cache_calls == []                       # cache NEVER consulted
    assert props.vintage == 'live'

    # Score equals the C1 wrapper on the same fixture (identical data path).
    from iqmapper.export.scoring import estimate_circuit_fidelity
    st = SimpleNamespace(
        layouts_data={0: {'alpha': ['0', '1'], 'beta': ['2', '3'],
                          'other': [], 'ancilla': []}},
        qubit_props_data=qp, gate_props_data=gp,
        qubit_ids=['0', '1', '2', '3'], edges_list=edges)
    local = estimate_circuit_fidelity(st, 0, 'lucj')['fidelity']
    assert props.score_layout([0, 1, 2, 3])['fidelity'] == pytest.approx(local)


# ── a fetch failure raises (no cached fallback) ──────────────────────

def test_remote_fetch_failure_raises_no_cache_fallback(monkeypatch):
    from iqmapper import data
    # A cached snapshot EXISTS, but it must NOT be used as a fallback.
    monkeypatch.setattr(data, 'list_known_calibrations',
                        lambda *a, **k: [{'last_update': '2026-07-05 08:00:00'}])
    monkeypatch.setattr(data, '_load_calibration',
                        lambda *a, **k: ({'readout_error': [0.01, 0.02]},
                                         {'cz_error': [0.001]}))

    backend = _fake_remote_backend()
    backend.properties = lambda: (_ for _ in ()).throw(RuntimeError("network down"))
    state = SimpleNamespace(active_backend_name='current_other')
    with pytest.raises(RuntimeError):
        make_scan_get_props(state)(backend)      # no silent cached fallback


# ── the shim scores against ITS OWN edges, not the current state ─────

def test_shim_scores_against_own_edges():
    edges = [(0, 1), (1, 2), (2, 3)]
    # Backend X: coupler (0,1) is DEAD. Backend Y: (0,1) is healthy.
    x = CachedPropsShim({'readout_error': [0.001] * 4},
                        {'cz_error': [1.0, 0.001, 0.001]}, edges)
    y = CachedPropsShim({'readout_error': [0.001] * 4},
                        {'cz_error': [0.001, 0.001, 0.001]}, edges)
    # Candidate uses qubits 0,1,2 → only edges (0,1) and (1,2) count.
    fx = x.score_layout([0, 1, 2])['fidelity']
    fy = y.score_layout([0, 1, 2])['fidelity']
    assert fx == pytest.approx(0.0)          # X's dead (0,1) → zero
    assert fy > 0.9                          # Y's (0,1) healthy → high
    # Each shim used its OWN per-edge errors, not a shared/state edge list.


# ── the hard invariants reject a mispaired snapshot ──────────────────

def test_gate_length_mismatch_raises():
    with pytest.raises(ValueError):
        CachedPropsShim({'readout_error': [0.01] * 4},
                        {'cz_error': [0.001, 0.001]},        # 2 != 3 edges
                        [(0, 1), (1, 2), (2, 3)])


def test_qubit_length_mismatch_raises():
    with pytest.raises(ValueError):
        CachedPropsShim({'readout_error': [0.01, 0.02, 0.03]},  # 3 != 4 qubits
                        {'cz_error': [0.001]}, [(0, 1)], n_qubits=4)


# ── end-to-end pin — shim score == core hand-built from same snapshot ─

def test_shim_score_matches_core_handbuilt():
    from iqmapper.export.scoring import estimate_fidelity_core
    qp = {'readout_error': [0.01, 0.02, 0.03, 0.04]}
    gp = {'cz_error': [0.001, 0.002, 1.0]}       # (2,3) dead
    edges = [(0, 1), (1, 2), (2, 3)]
    cand = [0, 1, 2, 3]

    shim_score = CachedPropsShim(qp, gp, edges).score_layout(cand)['fidelity']

    readout_by_id = {q: qp['readout_error'][q] for q in cand}
    gate_edges = [(u, v, gp['cz_error'][i]) for i, (u, v) in enumerate(edges)]
    core = estimate_fidelity_core(cand, readout_by_id, gate_edges,
                                  all_readout=qp['readout_error'])['fidelity']

    assert shim_score == pytest.approx(core)
    assert shim_score == pytest.approx(0.0)      # dead coupler seen, not hidden


# ── SYM-1 t1: score_layout scores the FULL directed surface (== C1 core) ─────

def test_score_layout_full_directed_surface():
    from iqmapper.export.scoring import estimate_fidelity_core
    # Bidirectional coupling map + an INCIDENTAL edge (0,2) among the set
    # (not on the chain path) carrying a large error.
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    cz = [0.01, 0.01, 0.01, 0.01, 0.30, 0.30]
    qp = {'readout_error': [0.01, 0.01, 0.01]}
    gp = {'cz_error': cz}
    cand = [0, 1, 2]

    shim = CachedPropsShim(qp, gp, edges)
    sh = shim.score_layout(cand)

    core = estimate_fidelity_core(
        cand, {q: qp['readout_error'][q] for q in cand},
        [(u, v, cz[i]) for i, (u, v) in enumerate(edges)],
        all_readout=qp['readout_error'])

    assert sh['fidelity'] == pytest.approx(core['fidelity'])
    assert sh['n_gates'] == 6                      # all directed edges counted
    # The incidental (0,2) 0.30 factor IS included (both dirs) — not skipped.
    healthy_only = estimate_fidelity_core(
        cand, {q: 0.01 for q in cand},
        [(0, 1, 0.01), (1, 0, 0.01), (1, 2, 0.01), (2, 1, 0.01)],
        all_readout=qp['readout_error'])
    assert sh['fidelity'] < healthy_only['fidelity']


# ── SYM-3 t2: shim uses ITS OWN edges (no current-state leak) ────────────────

def test_shim_edges_are_its_own():
    edges = [(5, 6), (6, 5), (6, 7), (7, 6)]        # backend X's own topology
    shim = CachedPropsShim({'readout_error': [0.0] * 8},
                           {'cz_error': [0.1, 0.1, 0.2, 0.2]}, edges, n_qubits=8)
    assert shim._edges == [(5, 6), (6, 5), (6, 7), (7, 6)]   # X's edges verbatim
    # A mismatched-length snapshot for a different topology → ValueError.
    with pytest.raises(ValueError):
        CachedPropsShim({'readout_error': [0.0] * 8},
                        {'cz_error': [0.1, 0.1]}, edges, n_qubits=8)
