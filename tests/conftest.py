"""Pytest config: make the iqmapper package importable from tests."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# ...and the tests directory itself, so a test module can reuse another's
# fixture (test_guided_search cross-checks against test_ancilla_sweep's
# GOOD/BAD-block backend rather than cloning it and letting the two drift).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# Fixtures for iqmapper.export pure-function tests
from types import SimpleNamespace

import pytest


def _make_slot(name="C1", alpha=None, beta=None, other=None, qubits=None,
               ancilla=None, defaults=None):
    """One entry for state.layouts_data[i] - only the keys the export
    functions read, plus a populated `defaults` for the apply path."""
    return {
        "name": name,
        "alpha": list(alpha or []),
        "beta": list(beta or []),
        "other": list(other or []),
        "qubits": list(qubits or []),
        "ancilla": list(ancilla or []),
        "styles": {},
        "defaults": defaults if defaults is not None else {
            "fill": "#FF0000", "fill_beta": "#00FF00",
            "fill_other": "#F39C12", "fill_anc": "#95A5A6",
            "line": "#000000",
        },
    }


def _make_state(qubit_ids=None, edges_list=None, qubit_props=None,
                gate_props=None, layouts=None, backend_name="fake_backend",
                chip="fake_chip", cal_label="2026-01-01",
                cal_history=None, cal_history_dates=None):
    """SimpleNamespace standing in for SessionState. The export functions
    read only these attributes, so a namespace is a faithful stand-in and
    avoids depending on SessionState's real constructor."""
    return SimpleNamespace(
        qubit_ids=list(qubit_ids) if qubit_ids is not None else ["0", "1", "2", "3"],
        edges_list=list(edges_list) if edges_list is not None else [(0, 1), (1, 2), (2, 3)],
        qubit_props_data=qubit_props if qubit_props is not None else {},
        gate_props_data=gate_props if gate_props is not None else {},
        layouts_data=layouts if layouts is not None else {},
        active_backend_name=backend_name,
        active_chip=chip,
        calibration_data_label=cal_label,
        cal_history=cal_history if cal_history is not None else {},
        cal_history_dates=cal_history_dates if cal_history_dates is not None else [],
    )


@pytest.fixture
def make_slot():
    return _make_slot


@pytest.fixture
def make_state():
    return _make_state


@pytest.fixture(autouse=True)
def _file_log_in_tmp(tmp_path, monkeypatch):
    """Keep the app's log file out of the user's log directory and detach it
    after every test. A test that starts the UI (Refresh rebuilds through
    `launch_mapper`) would otherwise write the real log and leave the handler
    attached, with propagation off, for every test after it."""
    from iqmapper import log_setup
    monkeypatch.setattr(log_setup, 'default_log_dir',
                        lambda: str(tmp_path / 'logs'))
    yield
    log_setup.remove_file_log()
