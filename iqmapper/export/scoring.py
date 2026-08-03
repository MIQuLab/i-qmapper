"""
export.scoring — Delta computation between calibration snapshots and
heuristic circuit-fidelity estimation. Pure numerical; no I/O.
"""
import numpy as np

from . import OUTLIER_MIN_SAMPLES, OUTLIER_SIGMA


# ═══════════════════════════════════════════════════════════════════════
# ERROR DELTA COMPUTATION
# ═══════════════════════════════════════════════════════════════════════

def compute_error_delta(qp_a, gp_a, qp_b, gp_b):
    """
    Compute Δ = props_B - props_A for all matching keys.

    Returns (delta_qubit_props, delta_gate_props) dicts of np.arrays.
    NaN in either input produces NaN in output.
    """
    delta_qp = {}
    if qp_a or qp_b:
        # Determine array length from whichever dict has data
        _qp_len = len(next(iter(qp_a.values()))) if qp_a else (
                   len(next(iter(qp_b.values()))) if qp_b else 0)
        for key in set(qp_a.keys()) | set(qp_b.keys()):
            a = qp_a.get(key, np.full(_qp_len, np.nan))
            b = qp_b.get(key, np.full(_qp_len, np.nan))
            delta_qp[key] = np.array(b, dtype=float) - np.array(a, dtype=float)

    delta_gp = {}
    if gp_a or gp_b:
        _gp_len = len(next(iter(gp_a.values()))) if gp_a else (
                   len(next(iter(gp_b.values()))) if gp_b else 0)
        for key in set(gp_a.keys()) | set(gp_b.keys()):
            a = gp_a.get(key, np.full(_gp_len, np.nan))
            b = gp_b.get(key, np.full(_gp_len, np.nan))
            delta_gp[key] = np.array(b, dtype=float) - np.array(a, dtype=float)

    return delta_qp, delta_gp
# ═══════════════════════════════════════════════════════════════════════
# ESTIMATED FIDELITY
# ═══════════════════════════════════════════════════════════════════════

def estimate_circuit_fidelity(state, circuit_idx, mode='lucj'):
    """Estimate a product-form fidelity upper bound from calibration data.

    Score formula:
        F ≈ ∏(1 - ε_readout_i) × ∏(1 - ε_gate_j)

    where i ranges over qubits in the layout and j over the CZ gates that
    connect them. This is the standard product-form fidelity bound used
    in noise-aware compilation literature, assuming independent errors.

    Note that ``mapomatic`` (Nation & Treinish, PRX Quantum 4, 010327,
    2023) uses an *additive* variant — Σ ε_i — for ranking candidate
    layouts. The two forms agree on relative ordering when individual
    errors are small (since log(1 - ε) ≈ -ε), but the product form here
    is reported as a fidelity-like quantity in [0, 1] for user
    interpretability. It is NOT a circuit-fidelity prediction in the
    physical sense — it is a relative-ranking heuristic intended for
    post-selection reporting of a chosen layout.

    Known limitations (cite explicitly when publishing):

      1. Assumes independent errors. Does not model crosstalk between
         simultaneous two-qubit gates or correlated readout errors.
      2. Counts each edge once. Overestimates fidelity for deep circuits
         that fire the same CZ pair multiple times (e.g. LUCJ ansatzen
         with large bond-dimension parameters). A tighter upper bound
         would multiply ε_gate by the per-edge CZ count.
      3. Ignores dephasing over circuit duration. A circuit whose depth ×
         mean gate time approaches T2 will measure far below the score
         reported here; the caller should separately check
         ``state.qubit_props_data['t2']`` vs expected circuit duration.
      4. Ignores state-preparation errors (SPAM). Only measurement errors
         are included via ε_readout.

    For absolute fidelity prediction on real hardware, run the circuit
    with randomized benchmarking — this heuristic is only suitable for
    RELATIVE comparison between candidate layouts on the same device
    at the same calibration epoch.

    Args:
        state: SessionState with ``qubit_props_data`` and ``gate_props_data``
               populated (i.e. calibration data loaded).
        circuit_idx: index into ``state.layouts_data``.
        mode: 'lucj' (use alpha+beta+other) or 'general' (use qubits).

    Returns:
        dict with keys:
          - fidelity         (float, 0..1) — product score
          - readout_fidelity (float) — qubit readout contribution alone
          - gate_fidelity    (float) — CZ gate contribution alone
          - n_qubits         (int)   — number of qubits included
          - n_gates          (int)   — number of CZ edges included
          - worst_qubit      (id, error)   or (None, 0) if empty
          - worst_gate       (edge, error) or (None, 0) if empty
          - outliers         list[(id, error)] — qubits > 3σ from device mean
        Returns None if no qubits are in the layout or calibration is missing.

    References:
        - mapomatic: https://github.com/qiskit-community/mapomatic
        - Nation, P. D. & Treinish, M. "Suppressing quantum circuit errors
          due to system variability." PRX Quantum 4, 010327 (2023).
    """
    d = state.layouts_data.get(circuit_idx, {})
    if mode == 'lucj':
        qubit_ids = d.get('alpha', []) + d.get('beta', []) + d.get('other', []) + d.get('ancilla', [])
    else:
        qubit_ids = d.get('qubits', [])

    if not qubit_ids or not state.qubit_props_data:
        return None

    q_id_map = {q: i for i, q in enumerate(state.qubit_ids)} if state.qubit_ids else {}

    # Readout fidelity
    re_arr = state.qubit_props_data.get('readout_error', np.array([]))
    readout_f = 1.0
    worst_q = (None, 0)
    n_q = 0
    for qid in qubit_ids:
        idx = q_id_map.get(qid)
        if idx is not None and idx < len(re_arr):
            err = re_arr[idx]
            if not np.isnan(err):
                readout_f *= (1 - err)
                n_q += 1
                if err > worst_q[1]:
                    worst_q = (qid, float(err))

    # Gate fidelity (CZ gates between layout qubits)
    cz_arr = state.gate_props_data.get('cz_error', np.array([]))
    qubit_set = set(qubit_ids)
    gate_f = 1.0
    worst_g = (None, 0)
    n_g = 0
    for i, edge in enumerate(state.edges_list or []):
        u_id, v_id = state.qubit_ids[edge[0]], state.qubit_ids[edge[1]]
        if u_id in qubit_set and v_id in qubit_set:
            if i < len(cz_arr):
                err = cz_arr[i]
                if not np.isnan(err):
                    # Multiplicative fidelity accumulation, same form as
                    # readout_f above: each valid gate contributes a
                    # (1 - err) factor to the product.
                    gate_f *= (1 - err)
                    n_g += 1
                    if err > worst_g[1]:
                        worst_g = ((u_id, v_id), float(err))

    total_f = readout_f * gate_f

    # Outlier detection: flag qubits whose readout error sits more than
    # OUTLIER_SIGMA standard deviations above the device mean.
    outliers = []
    if len(re_arr) > OUTLIER_MIN_SAMPLES:
        valid = re_arr[~np.isnan(re_arr)]
        if len(valid) > 0:
            mean, std = np.mean(valid), np.std(valid)
            threshold = mean + OUTLIER_SIGMA * std
            for qid in qubit_ids:
                idx = q_id_map.get(qid)
                if idx is not None and idx < len(re_arr):
                    if not np.isnan(re_arr[idx]) and re_arr[idx] > threshold:
                        outliers.append((qid, float(re_arr[idx])))

    return {
        'fidelity': float(total_f),
        'readout_fidelity': float(readout_f),
        'gate_fidelity': float(gate_f),
        'n_qubits': n_q,
        'n_gates': n_g,
        'worst_qubit': worst_q,
        'worst_gate': worst_g,
        'outliers': outliers,
    }
