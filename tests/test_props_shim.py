"""Tests for qpu_mapper.props_shim — the cache-aware props adapter.

t1 (the crux): CachedPropsShim built from B0.a-shaped dicts answers
readout_error(q) and gate_error(name, pair) for both edge orders, and
raises a clear KeyError on any missing lookup. Plus a factory cache-hit
reconstruction test with fakes (offline, no qiskit, no network).
"""
from types import SimpleNamespace

import numpy as np
import pytest

from qpu_mapper.props_shim import CachedPropsShim, make_scan_get_props


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


# ── t1: the crux ────────────────────────────────────────────────────────────

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


# ── factory: cache-hit reconstruction (no live fetch) ───────────────────────

def test_factory_cache_hit(monkeypatch):
    qp, gp, edges = _fixtures()

    # Fake backend: coupling map + num_qubits + a 2Q gate + a name.
    backend = SimpleNamespace(
        name='ibm_fake',
        num_qubits=4,
        coupling_map=SimpleNamespace(get_edges=lambda: list(edges)),
        operation_names=['cz', 'sx', 'rz', 'x'],
    )

    from qpu_mapper import data
    monkeypatch.setattr(data, 'list_known_calibrations',
                        lambda name, day: [{'last_update': '2026-07-12 09:00:00'}])
    monkeypatch.setattr(data, '_load_calibration', lambda name, ts: (qp, gp))
    # properties() must NOT be called on a cache hit.
    backend.properties = lambda: (_ for _ in ()).throw(
        AssertionError("live fetch on cache hit"))

    state = SimpleNamespace(active_backend_name='current')
    get_props = make_scan_get_props(state, now='2026-07-12')
    props = get_props(backend)

    assert props.readout_error(2) == pytest.approx(0.030)
    assert props.gate_error('cz', (2, 1)) == pytest.approx(0.002)
