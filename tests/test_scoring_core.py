"""One scorer everywhere (estimate_fidelity_core).

Pure/offline. Pins C1 behaviour across the extract, checks the shim's
per-backend scorer, and proves scan == local on identical calibration.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from iqmapper.export.scoring import (
    estimate_circuit_fidelity, estimate_fidelity_core,
)
from iqmapper.props_shim import CachedPropsShim


# ── C1 pin — behaviour unchanged across the refactor-extract ────────────

def test_c1_pin_unchanged():
    qubit_ids = ['0', '1', '2', '3']
    readout = np.array([0.01, 0.02, 0.03, 0.04])
    cz = np.array([0.001, 0.002, 0.003])
    edges = [(0, 1), (1, 2), (2, 3)]
    state = SimpleNamespace(
        layouts_data={0: {'alpha': ['0', '1'], 'beta': ['2', '3'],
                          'other': [], 'ancilla': []}},
        qubit_props_data={'readout_error': readout},
        gate_props_data={'cz_error': cz},
        qubit_ids=qubit_ids, edges_list=edges)

    res = estimate_circuit_fidelity(state, 0, 'lucj')
    # Expected computed from the fixture with the documented formula — no
    # magic constant. All 4 qubits + all 3 edges are inside the layout.
    readout_f = float(np.prod(1 - readout))
    gate_f = float(np.prod(1 - cz))
    assert res['fidelity'] == pytest.approx(readout_f * gate_f)
    assert res['readout_fidelity'] == pytest.approx(readout_f)
    assert res['gate_fidelity'] == pytest.approx(gate_f)
    assert res['n_qubits'] == 4 and res['n_gates'] == 3

    # The pure core, called with the same lifted inputs, agrees exactly.
    readout_by_id = {q: readout[i] for i, q in enumerate(qubit_ids)}
    gate_edges = [(qubit_ids[u], qubit_ids[v], cz[i])
                  for i, (u, v) in enumerate(edges)]
    core = estimate_fidelity_core(['0', '1', '2', '3'], readout_by_id,
                                  gate_edges, all_readout=readout)
    assert core == res


# ── per-backend shim scorer == hand-computed product ────────────────────

def test_shim_score_hand_computed():
    qp = {'readout_error': np.array([0.01, 0.02, 0.03, 0.04])}
    gp = {'cz_error': np.array([0.001, 0.002, 0.003])}
    edges = [(0, 1), (1, 2), (2, 3)]
    shim = CachedPropsShim(qp, gp, edges)

    res = shim.score_layout([0, 1, 2, 3])
    readout_f = float(np.prod(1 - qp['readout_error']))
    gate_f = float(np.prod(1 - gp['cz_error']))
    assert res['fidelity'] == pytest.approx(readout_f * gate_f)
    assert res['n_qubits'] == 4 and res['n_gates'] == 3
    assert shim.score_layout([]) is None                    # empty → None


# ── scan row == local number on the SAME calibration (the screenshot) ──

def test_scan_equals_local_same_backend():
    qp = {'readout_error': np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06])}
    gp = {'cz_error': np.array([0.001, 0.002, 0.003, 0.004, 0.005])}
    edges = [(0, 1), (1, 2), (2, 3), (0, 4), (1, 5)]

    # Local (C1) via the state wrapper: candidate alpha=[0,1] beta=[2,3] anc=[4].
    qubit_ids = ['0', '1', '2', '3', '4', '5']
    state = SimpleNamespace(
        layouts_data={0: {'alpha': ['0', '1'], 'beta': ['2', '3'],
                          'other': [], 'ancilla': ['4']}},
        qubit_props_data=qp, gate_props_data=gp,
        qubit_ids=qubit_ids, edges_list=edges)
    local = estimate_circuit_fidelity(state, 0, 'lucj')['fidelity']

    # Scan (per-backend shim) on the SAME arrays, same candidate (physical ids).
    shim = CachedPropsShim(qp, gp, edges)
    scan = shim.score_layout([0, 1, 2, 3, 4])['fidelity']

    assert scan == pytest.approx(local)       # one scale, same number
