import numpy as np
import pytest

from qpu_mapper.export.scoring import estimate_circuit_fidelity, compute_error_delta


def test_fidelity_product_and_counts(make_state, make_slot):
    st = make_state(
        qubit_ids=["0", "1", "2", "3"],
        edges_list=[(0, 1), (1, 2), (2, 3)],
        qubit_props={"readout_error": np.array([0.01, 0.02, 0.03, 0.04])},
        gate_props={"cz_error": np.array([0.001, 0.002, 0.003])},
        layouts={0: make_slot(alpha=["0", "1"], beta=["2", "3"])},
    )
    r = estimate_circuit_fidelity(st, 0, mode="lucj")
    assert r is not None                      # regression: gate_f *= (1-err) must not TypeError->None
    assert r["n_qubits"] == 4
    assert r["n_gates"] == 3
    assert r["readout_fidelity"] == pytest.approx(0.99 * 0.98 * 0.97 * 0.96)
    assert r["gate_fidelity"] == pytest.approx(0.999 * 0.998 * 0.997)
    assert r["gate_fidelity"] < 1.0           # proves gate errors were actually multiplied in
    assert r["fidelity"] == pytest.approx(r["readout_fidelity"] * r["gate_fidelity"])
    assert r["worst_qubit"] == ("3", pytest.approx(0.04))
    assert r["worst_gate"] == (("2", "3"), pytest.approx(0.003))
    assert r["outliers"] == []                # <10 samples -> outlier detection skipped


def test_fidelity_none_when_empty_or_no_calibration(make_state, make_slot):
    assert estimate_circuit_fidelity(make_state(layouts={0: make_slot()}), 0, "lucj") is None
    st = make_state(qubit_props={}, layouts={0: make_slot(alpha=["0"])})
    assert estimate_circuit_fidelity(st, 0, "lucj") is None


def test_fidelity_skips_nan_readout(make_state, make_slot):
    st = make_state(
        qubit_ids=["0", "1"],
        edges_list=[(0, 1)],
        qubit_props={"readout_error": np.array([np.nan, 0.05])},
        gate_props={"cz_error": np.array([0.01])},
        layouts={0: make_slot(alpha=["0"], beta=["1"])},
    )
    r = estimate_circuit_fidelity(st, 0, "lucj")
    assert r["n_qubits"] == 1                  # the NaN-readout qubit is excluded from the product
    assert r["readout_fidelity"] == pytest.approx(0.95)


def test_compute_error_delta_basic_and_nan_fill():
    qp_a = {"readout_error": np.array([0.01, 0.02])}
    qp_b = {"readout_error": np.array([0.03, 0.02])}
    dqp, _dgp = compute_error_delta(qp_a, {}, qp_b, {})
    assert dqp["readout_error"] == pytest.approx([0.02, 0.0])
    # a key present only in B -> A side filled with NaN -> delta is NaN
    qp_b2 = {"readout_error": np.array([0.03, 0.04]), "t1": np.array([100.0, 200.0])}
    dqp2, _ = compute_error_delta(qp_a, {}, qp_b2, {})
    assert np.isnan(dqp2["t1"]).all()
