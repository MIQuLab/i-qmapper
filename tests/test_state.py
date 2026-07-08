"""Smoke tests for SessionState undo/redo and layout management."""
import numpy as np
import pytest

from qpu_mapper.state import SessionState


def test_fresh_state_has_default_circuits():
    s = SessionState()
    assert len(s.layouts_data) == 4  # MAX_CIRCUITS
    for i in range(4):
        assert s.layouts_data[i]['alpha'] == []
        assert s.layouts_data[i]['beta'] == []


def test_clear_all_layouts_resets():
    s = SessionState()
    s.layouts_data[0]['alpha'] = [0, 1]
    s.layouts_data[0]['beta'] = [2, 3]
    s.clear_all_layouts()
    assert s.layouts_data[0]['alpha'] == []
    assert s.layouts_data[0]['beta'] == []


def test_delta_mode_default_is_false():
    """delta_mode is the runtime flag; default must be False."""
    s = SessionState()
    assert s.delta_mode is False
    assert s.delta_qubit_props == {}


def test_history_depth_capped():
    """undo stack must not grow beyond _max_history."""
    s = SessionState()
    for _ in range(100):
        s.push_undo()
    assert len(s._undo_stack) <= s._max_history


# ═══════════════════════════════════════════════════════════════════════
# Undo / redo round-trip
# ═══════════════════════════════════════════════════════════════════════

def test_undo_restores_prior_layout_and_redo_reapplies():
    """push_undo → mutate → undo restores; redo re-applies."""
    s = SessionState()
    s.push_undo()
    s.layouts_data[0]['alpha'] = [1, 2, 3]

    ok = s.undo()
    assert ok is True
    assert s.layouts_data[0]['alpha'] == []   # baseline restored

    ok = s.redo()
    assert ok is True
    assert s.layouts_data[0]['alpha'] == [1, 2, 3]


def test_new_push_undo_clears_redo_stack():
    """After a fresh push_undo, any pending redo history is dropped —
    the standard editor invariant so a new edit forks the timeline."""
    s = SessionState()
    s.push_undo()
    s.layouts_data[0]['alpha'] = [1]
    s.undo()
    assert s.has_redo is True   # redo is available
    s.push_undo()               # new edit — must clear the redo stack
    assert s.has_redo is False


def test_history_never_exceeds_max():
    """Cap holds even with well over _max_history pushes."""
    s = SessionState()
    for _ in range(s._max_history + 20):
        s.push_undo()
    assert len(s._undo_stack) <= s._max_history


# ═══════════════════════════════════════════════════════════════════════
# Zoom
# ═══════════════════════════════════════════════════════════════════════

def test_zoom_set_get_reset_roundtrip():
    s = SessionState()
    s.set_zoom(-1.0, 2.0, -3.0, 4.0)
    assert s.get_zoom_range() == (-1.0, 2.0, -3.0, 4.0)
    s.reset_zoom()
    # With no zoom set AND no coords, get_zoom_range returns None.
    assert s.get_zoom_range() is None


# ═══════════════════════════════════════════════════════════════════════
# Session round-trip (numpy arrays included)
# ═══════════════════════════════════════════════════════════════════════

def test_session_roundtrip_preserves_layouts_and_numpy_arrays():
    s = SessionState()
    # Layout mutation across two circuits.
    s.layouts_data[0]['alpha'] = ['q0', 'q1']
    s.layouts_data[0]['beta'] = ['q2']
    s.layouts_data[1]['ancilla'] = ['q3']
    # Numpy arrays that must survive serialization.
    s.x_coord = np.array([0.0, 1.5, 2.75])
    s.y_coord = np.array([-1.0, 0.0, 1.0])
    s.qubit_ids = ['q0', 'q1', 'q2']
    s.qubit_props_data = {'readout_error': np.array([0.01, 0.02, 0.03])}

    payload = s.to_session_dict()
    s2 = SessionState()
    s2.from_session_dict(payload)

    assert s2.layouts_data[0]['alpha'] == ['q0', 'q1']
    assert s2.layouts_data[0]['beta'] == ['q2']
    assert s2.layouts_data[1]['ancilla'] == ['q3']
    # Numpy arrays round-trip as numpy arrays (from_session_dict wraps them).
    assert isinstance(s2.x_coord, np.ndarray)
    assert isinstance(s2.qubit_props_data['readout_error'], np.ndarray)
    np.testing.assert_array_equal(s2.x_coord, np.array([0.0, 1.5, 2.75]))
    np.testing.assert_array_equal(
        s2.qubit_props_data['readout_error'], np.array([0.01, 0.02, 0.03]))


# ═══════════════════════════════════════════════════════════════════════
# Version validation
# ═══════════════════════════════════════════════════════════════════════

def test_from_session_dict_rejects_missing_version():
    s = SessionState()
    with pytest.raises(ValueError, match="_version"):
        s.from_session_dict({})


def test_from_session_dict_rejects_future_version():
    s = SessionState()
    with pytest.raises(ValueError, match="newer than this"):
        s.from_session_dict({'_version': SessionState._MAX_SUPPORTED_VERSION + 1})


def test_from_session_dict_rejects_legacy_version():
    s = SessionState()
    with pytest.raises(ValueError, match="too old"):
        s.from_session_dict({'_version': 2})


# ═══════════════════════════════════════════════════════════════════════
# Regression — legend default in serialized session
# ═══════════════════════════════════════════════════════════════════════

def test_to_session_dict_legend_enabled_default_is_false():
    """A fresh SessionState must serialize legend_config.enabled = False.
    Previously the .get(..., True) fallback in to_session_dict inverted
    the default, so an untouched session file claimed the legend was on."""
    s = SessionState()
    payload = s.to_session_dict()
    assert payload['legend_config']['enabled'] is False


# ═══════════════════════════════════════════════════════════════════════
# Style dict round-trip
# ═══════════════════════════════════════════════════════════════════════

def test_style_dict_roundtrip_preserves_colorscale_and_per_circuit_colors():
    s = SessionState()
    s.current_colorscale = 'Blue→Purple (IBM)'
    s.layouts_data[0]['color'] = '#123456'
    s.layouts_data[1]['color'] = '#abcdef'

    payload = s.to_style_dict()
    s2 = SessionState()
    s2.from_style_dict(payload)

    assert s2.current_colorscale == 'Blue→Purple (IBM)'
    assert s2.layouts_data[0]['color'] == '#123456'
    assert s2.layouts_data[1]['color'] == '#abcdef'
