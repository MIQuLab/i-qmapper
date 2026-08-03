import numpy as np
import pytest

from iqmapper.export.analysis import (
    get_connectivity_scores, get_connectivity_ranking_html, find_stable_qubits,
)


def test_connectivity_scores_numeric(make_state):
    st = make_state(
        qubit_ids=["0", "1", "2"],
        edges_list=[(0, 1), (1, 2)],
        gate_props={"cz_error": np.array([0.01, 0.03])},
    )
    by_id = {s["qubit"]: s for s in get_connectivity_scores(st)}
    assert by_id["1"]["n_connections"] == 2                 # qubit 1 touches both edges
    assert by_id["1"]["avg_gate_error"] == pytest.approx(0.02)
    assert by_id["0"]["n_connections"] == 1


def test_connectivity_empty_when_no_data(make_state):
    assert get_connectivity_scores(make_state(edges_list=[], gate_props={})) == []


def test_connectivity_nan_safe(make_state):
    st = make_state(qubit_ids=["0", "1"], edges_list=[(0, 1)],
                    gate_props={"cz_error": np.array([np.nan])})
    assert all(s["n_connections"] == 0 for s in get_connectivity_scores(st))


def test_connectivity_ranking_html_renders(make_state):
    st = make_state(qubit_ids=["0", "1", "2"], edges_list=[(0, 1), (1, 2)],
                    gate_props={"cz_error": np.array([0.01, 0.03])})
    html = get_connectivity_ranking_html(st, top_n=20, ascending=True)
    assert isinstance(html, str)
    assert "Q1" in html


def test_find_stable_qubits_lower_is_better(make_state):
    st = make_state(
        qubit_ids=["0", "1"],
        cal_history_dates=["d1", "d2"],
        cal_history={
            "d1": {"qubit_props": {"readout_error": np.array([0.01, 0.20])}},
            "d2": {"qubit_props": {"readout_error": np.array([0.02, 0.25])}},
        },
    )
    ids = {r["qubit"] for r in find_stable_qubits(
        st, prop_key="readout_error", max_error=0.05, min_days_ratio=1.0)}
    assert "0" in ids        # always <= 0.05
    assert "1" not in ids    # always > 0.05
