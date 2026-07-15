"""Unified Import/Export tab (two subpanels) + collapsible JSON preview."""
from types import SimpleNamespace

import numpy as np
from ipywidgets import Accordion, Tab


def _fake_device(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _build(monkeypatch):
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device)
    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "ibm_here"
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
    return b, state


def _walk(node):
    yield node
    for ch in getattr(node, 'children', []):
        yield from _walk(ch)


def _contains(root, target):
    return any(w is target for w in _walk(root))


def _find(root, cls):
    return next((w for w in _walk(root) if isinstance(w, cls)), None)


# ── t3: unified Import/Export tab — both subpanels, both paths intact ───────

def test_unified_io_tab(monkeypatch):
    b, state = _build(monkeypatch)
    exp_calls = []
    real = type(b)._get_export
    monkeypatch.setattr(b, '_get_export', lambda: exp_calls.append(1) or real(b))

    b.nav_bar.value = 'io'
    view = b.content_box.children[0]
    assert isinstance(view, Tab) and len(view.children) == 2
    assert view.get_title(0) == '📤 Export'
    assert view.get_title(1) == '📥 Import'
    assert exp_calls                                       # export path ran on open

    export_panel, import_panel = view.children[0], view.children[1]
    assert _contains(export_panel, b.btn_save_json)        # export control present
    assert _contains(import_panel, b.btn_unified_upload)   # import control present

    # Import path still wired: firing the upload handler reaches its reader.
    up = []
    monkeypatch.setattr("qpu_mapper.ui.builder._read_upload",
                        lambda w: up.append(1) or None)
    b._on_unified_upload({'new': [object()]})
    assert up


def test_old_nav_tabs_removed(monkeypatch):
    b, state = _build(monkeypatch)
    vals = [v for _, v in b.nav_bar.options]
    assert 'io' in vals
    assert 'export' not in vals and 'auto' not in vals      # merged away


# ── t4: JSON preview collapsed by default; expand refreshes + persists ──────

def test_json_preview_collapsible(monkeypatch):
    b, state = _build(monkeypatch)
    b.nav_bar.value = 'io'
    export_panel = b.content_box.children[0].children[0]
    acc = _find(export_panel, Accordion)
    assert acc is not None
    assert acc.selected_index is None                       # collapsed by default
    assert b._json_preview_open is False

    exp_calls = []
    real = type(b)._get_export
    monkeypatch.setattr(b, '_get_export', lambda: exp_calls.append(1) or real(b))
    acc.selected_index = 0                                   # expand
    assert b._json_preview_open is True
    assert exp_calls                                        # content refreshed on expand

    # Choice persists: a rebuilt Export subpanel keeps the preview open.
    io2 = b._build_import_export_view()
    acc2 = _find(io2.children[0], Accordion)
    assert acc2.selected_index == 0
