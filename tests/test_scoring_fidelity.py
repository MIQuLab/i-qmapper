"""Regression tests for estimate_circuit_fidelity.

A swallowed TypeError once made this function return None for every input.
The formula is now correct and there is no swallowing except; these tests
lock that in so a future rewrite of the fidelity path cannot silently
regress back to always-None.

Uses SimpleNamespace stand-ins for state — estimate_circuit_fidelity only
reads attributes, so a namespace is a faithful stand-in and avoids
depending on the real SessionState constructor."""
import numpy as np
from types import SimpleNamespace

from iqpu_mapper.export.scoring import estimate_circuit_fidelity


def _fake_state():
    # 3 device qubits; one CZ edge between device indices 0 and 1
    # (i.e. between qubit_ids 10 and 11). Layout uses qubits 10 and 11.
    return SimpleNamespace(
        qubit_ids=[10, 11, 12],
        edges_list=[(0, 1)],
        layouts_data={0: {'qubits': [10, 11]}},
        qubit_props_data={'readout_error': np.array([0.01, 0.02, 0.03])},
        gate_props_data={'cz_error': np.array([0.05])},
    )


def test_fidelity_is_a_real_number_not_none():
    """Regression: a swallowed TypeError once made fidelity always None."""
    result = estimate_circuit_fidelity(_fake_state(), 0, mode='general')
    assert result is not None, "fidelity must not be None for a valid layout+calibration"
    assert isinstance(result['fidelity'], float)
    assert 0.0 < result['fidelity'] <= 1.0
    # Pin the product-form value: readout (1-.01)(1-.02) × gate (1-.05)
    expected = (1 - 0.01) * (1 - 0.02) * (1 - 0.05)
    assert abs(result['fidelity'] - expected) < 1e-9
    assert abs(result['readout_fidelity'] - (1 - 0.01) * (1 - 0.02)) < 1e-9
    assert abs(result['gate_fidelity'] - (1 - 0.05)) < 1e-9


def test_fidelity_lucj_mode():
    """lucj mode reads alpha+beta+other+ancilla; still a real fidelity."""
    state = _fake_state()
    state.layouts_data = {0: {'alpha': [10], 'beta': [11], 'other': [], 'ancilla': []}}
    result = estimate_circuit_fidelity(state, 0, mode='lucj')
    assert result is not None
    assert 0.0 < result['fidelity'] <= 1.0


def test_fidelity_none_when_layout_empty():
    """Documented behaviour: no qubits in the layout → None (not a crash)."""
    state = _fake_state()
    state.layouts_data = {0: {'qubits': []}}
    assert estimate_circuit_fidelity(state, 0, mode='general') is None
