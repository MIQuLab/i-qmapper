"""Headless tests for MapperUIBuilder construction + handler wiring.

Builds the real builder with generate_device_data stubbed (deterministic
5-qubit line device, no network, no kernel) and a placeholder backend
object — the backend identity doesn't matter once generate_device_data
is monkeypatched. The smoke test fires every button + nav/view observer
the way ipywidgets would and re-raises only NameError/AttributeError —
the bug class the closure->method refactor risked (and that bit
ui.time_panel).
"""
import numpy as np
import pytest


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    """Deterministic 5-qubit line device - replaces the network/data layer.

    Matches the 6-tuple generate_device_data returns and that
    _load_device_graph unpacks: (x, y, edges, qubit_ids, readout, gate).
    """
    n = 5
    x = list(range(n))
    y = [0.0] * n
    edges = [(i, i + 1) for i in range(n - 1)]
    qids = [str(i) for i in range(n)]
    readout = np.array([0.01 * (i + 1) for i in range(n)])
    gate = np.array([0.001 * (i + 1) for i in range(len(edges))])
    return x, y, edges, qids, readout, gate


def _values(widget):
    """Option values from a Dropdown/ToggleButtons, whether options are
    plain strings or (label, value) tuples."""
    out = []
    for opt in widget.options:
        out.append(opt[1] if isinstance(opt, tuple) else opt)
    return out


@pytest.fixture
def dummy_builder(monkeypatch):
    # Isolate the build from the data layer: deterministic, offline.
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)

    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder

    state = SessionState()
    state.active_backend = object()                    # non-None placeholder
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = None
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]               # non-empty -> skip the bg auto-load thread

    builder = MapperUIBuilder(state)
    builder.build()                                    # full pipeline, headless
    return builder


def test_builder_constructs_headless(dummy_builder):
    b = dummy_builder
    assert b.state.qubit_ids == ["0", "1", "2", "3", "4"]   # device loaded via the stub
    assert len(b.fig.data) >= 4                             # expected plot traces present


def test_handler_smoke_no_bare_name_errors(dummy_builder):
    b = dummy_builder
    errors = []

    def fire(label, fn):
        try:
            fn()
        except (NameError, AttributeError) as e:            # the bug class -> record (fail)
            errors.append(f"{label}: {type(e).__name__}: {e}")
        except Exception:
            pass                                            # other errors aren't what we're hunting

    # button handlers (real on_click flow)
    for name in ["btn_undo", "btn_redo", "btn_clear_all", "btn_refresh", "btn_auto"]:
        btn = getattr(b, name, None)
        if btn is not None:
            fire(name, btn.click)

    # every nav tab - exercises _update_view + all five _build_*_view methods + the Glitch-2 guard
    for val in _values(b.nav_bar):
        fire(f"nav={val}", lambda v=val: setattr(b.nav_bar, "value", v))

    # every render mode
    for val in _values(b.view_mode):
        fire(f"view={val}", lambda v=val: setattr(b.view_mode, "value", v))

    assert not errors, "bare-name/attribute errors in handlers:\n" + "\n".join(errors)


def test_on_clear_all_empties_layouts(dummy_builder):
    from qpu_mapper.export.layout_io import apply_layout_to_circuit
    b = dummy_builder
    apply_layout_to_circuit(b.state, 0, [0, 1, 2], "general")
    assert b.state.layouts_data[0]["qubits"]                # populated before
    b.btn_clear_all.click()
    assert not b.state.layouts_data[0]["qubits"]            # cleared after


def test_on_auto_map_populates_layout(dummy_builder):
    b = dummy_builder
    b.dd_mode.value = "general"
    b.txt_auto.value = "0,1,2,3"
    b.btn_auto.click()
    assert b.state.layouts_data[0]["qubits"] == ["0", "1", "2", "3"]
