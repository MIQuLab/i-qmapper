"""F1 guard: refresh_plot must never leave the qubit node trace with positions
but empty colour/text ('edges but no nodes'). It (a) rewrites x/y from the SAME
state it paints, and (b) refuses to empty a populated trace when the state has
no qubit_ids — logging loudly instead.
"""
import logging
from types import SimpleNamespace

import numpy as np


def _fake_device(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _build(monkeypatch):
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]
    b = MapperUIBuilder(state)
    b.build()
    return b


def _qubit_trace(fig):
    from iqmapper.plot import _META_QUBITS
    return next(t for t in fig.data if getattr(t, 'meta', None) == _META_QUBITS)


def test_refresh_preserves_nodes_when_qubit_ids_empty(monkeypatch, caplog):
    b = _build(monkeypatch)
    from iqmapper import plot

    qt = _qubit_trace(b.fig)
    assert len(qt.x) == 8 and len(qt.marker.color) == 8      # nodes present

    # Emptying qubit_ids while edges remain used to wipe the node colour/text
    # (positions kept) → invisible nodes, edges drawn. The guard must refuse.
    b.state.qubit_ids = []
    with caplog.at_level(logging.WARNING, logger='iqmapper.plot'):
        plot.refresh_plot(b.fig, b.state, b.widgets_ref)

    assert len(qt.x) == 8                                     # positions intact
    assert isinstance(qt.marker.color, (list, tuple)) and len(qt.marker.color) == 8
    assert qt.text is not None and len(qt.text) == 8         # labels intact
    assert any('edges-but-no-nodes' in r.message for r in caplog.records)


def test_refresh_rewrites_node_positions_from_state(monkeypatch):
    b = _build(monkeypatch)
    from iqmapper import plot

    # Move the device coordinates, then refresh: the node trace x/y must follow
    # the state (same source), so it can never diverge from what is painted.
    b.state.x_coord = [v + 100.0 for v in b.state.x_coord]
    plot.refresh_plot(b.fig, b.state, b.widgets_ref)

    qt = _qubit_trace(b.fig)
    assert list(qt.x) == [float(v) for v in b.state.x_coord]
    assert len(qt.x) == len(qt.marker.color) == len(b.state.qubit_ids)
