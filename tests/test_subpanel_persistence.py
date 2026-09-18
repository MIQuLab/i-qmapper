"""SUBPANEL SELECTION MUST SURVIVE REBUILDS.

Every structural refresh rebuilds a nav view from scratch, minting a fresh Tab
whose selected_index defaults to 0. Firing Generate Multi-Circuit from the Multi
subpanel used to bounce the user back to Single mid-operation. These tests pin
that the Auto / I/O / Error subtabs keep their selection across rebuilds.
"""
from types import SimpleNamespace

import numpy as np
from ipywidgets import Dropdown


def _fake_device(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:                       # pragma: no cover
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _mentry(anc):
    return {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': list(anc),
            'bridge_slots': tuple(range(0, 4 * len(anc), 4)),
            'product_score': 0.60, 'additive_score': 0.5, 'lqs_circuit': 0.60,
            'rank_in_stratum': 1, 'source': 'vf2'}


def _mcircuit(i, anc):
    ch = _mentry(anc)
    return {'circuit_idx': i, 'norb': 8, 'chosen': ch, 'strata': {1: [ch]},
            'k_max_effective': 1, 'min_bridges': 1, 'vf2_truncated': False,
            'per_stratum': {1: {'n_direct': 0, 'n_derived': 1,
                                'truncated': False, 'shuffled': False}},
            'mapping_cap': 2000, 'excluded_qubits': []}


def _mresult():
    return {'circuits': [_mcircuit(0, [4]), _mcircuit(1, [6])],
            'total_product': 0.36, 'total_lqs_circuit': 0.36,
            'placed_qubits': [], 'buffer_qubits_used': [], 'failed_at': None,
            '_context': {'stub': True}}


def _build(monkeypatch):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()              # no .coords → Auto allowed
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


# ── t1: the user's exact sequence — Generate Multi stays on Multi ───────────

def test_multi_generate_keeps_multi_subpanel(monkeypatch):
    b, state = _build(monkeypatch)
    import iqmapper.multi_sweep as ms
    monkeypatch.setattr(ms, 'sweep_multi_circuits', lambda *a, **k: _mresult())

    b.intra_qpu_mode.value = True                # reveal the Multi subpanel
    b.nav_bar.value = 'auto_lucj'                # on the Auto tab
    assert len(b._auto_lucj_tabs.children) == 2
    b._auto_lucj_tabs.selected_index = 1         # user picks Multi-Circuit

    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()               # ← the action that bounced

    # Still on Multi (the fresh Tab from the rebuild carried the index over).
    assert b._auto_lucj_tabs.selected_index == 1
    # Results rendered in place.
    assert 'circuit(s) placed' in b.lbl_lucj_multi.value
    assert state.layouts_data[0]['ancilla'] == ['4']    # batch applied
    assert state.layouts_data[1]['ancilla'] == ['6']


def test_num_circuits_change_keeps_multi_subpanel(monkeypatch):
    # The num_circuits observer fires _update_view during the multi flow; a
    # bare structural rebuild must also preserve the selection.
    b, state = _build(monkeypatch)
    b.intra_qpu_mode.value = True
    b.nav_bar.value = 'auto_lucj'
    b._auto_lucj_tabs.selected_index = 1

    b.num_circuits.value = 3                      # structural change → rebuild
    assert b._auto_lucj_tabs.selected_index == 1


# ── t2: fresh reveal defaults to Single; rebuilds while ON preserve ─────────

def test_fresh_reveal_single_then_preserve(monkeypatch):
    b, state = _build(monkeypatch)

    b.intra_qpu_mode.value = False
    b._build_auto_lucj_view()
    assert len(b._auto_lucj_tabs.children) == 1
    assert b._auto_lucj_tabs.selected_index == 0

    b.intra_qpu_mode.value = True                # fresh reveal of Multi
    b._build_auto_lucj_view()
    assert len(b._auto_lucj_tabs.children) == 2
    assert b._auto_lucj_tabs.selected_index == 0   # defaults to Single

    b._auto_lucj_tabs.selected_index = 1           # user picks Multi
    b._build_auto_lucj_view()                       # subsequent rebuild
    assert b._auto_lucj_tabs.selected_index == 1   # preserved

    # Turning it OFF clamps past-the-end index back to the only subpanel.
    b.intra_qpu_mode.value = False
    b._build_auto_lucj_view()
    assert len(b._auto_lucj_tabs.children) == 1
    assert b._auto_lucj_tabs.selected_index == 0


# ── t3: same-class pins for the I/O and Error tabs ─────────────────────────

def test_io_subtab_survives_rebuild(monkeypatch):
    b, state = _build(monkeypatch)
    b.nav_bar.value = 'io'
    assert len(b._io_tabs.children) == 2
    b._io_tabs.selected_index = 1                # Export subpanel (index 1)
    b._update_view()                             # a refresh while on I/O
    assert b._io_tabs.selected_index == 1         # still Export, not Import (index 0)


def test_io_import_first_paste_first(monkeypatch):
    from ipywidgets import HTML
    b, state = _build(monkeypatch)
    b.nav_bar.value = 'io'
    # I/O opens on Import (index 0); Export is the second sub-tab.
    assert b._io_tabs.get_title(0) == '📥 Import'
    assert b._io_tabs.get_title(1) == '📤 Export'
    # Import subpanel leads with the paste-qubit-list block (renamed, no "Or").
    import_panel = b._io_tabs.children[0]
    first_html = next(c for c in import_panel.children if isinstance(c, HTML))
    assert 'Paste qubit list' in first_html.value
    assert 'Or paste' not in first_html.value


def test_error_subtab_survives_rebuild(monkeypatch):
    b, state = _build(monkeypatch)
    b.nav_bar.value = 'error_inspect'
    assert len(b._err_tabs.children) == 2
    b._err_tabs.selected_index = 1               # Analysis subtab
    b._update_view()                             # a refresh while on Error
    assert b._err_tabs.selected_index == 1        # still Analysis


# ══ X2b — no visible rebuild on Multi Generate ═════════════════════════════

def _on_multi(b):
    """Put the UI on the Auto tab with the Multi subpanel revealed + selected."""
    b.intra_qpu_mode.value = True
    b.nav_bar.value = 'auto_lucj'
    b._auto_lucj_tabs.selected_index = 1
    return b._auto_lucj_tabs


# ── W3 t1: the Tab widget instance is UNCHANGED after Generate (no rebuild) ──

def test_multi_generate_no_rebuild(monkeypatch):
    b, state = _build(monkeypatch)
    import iqmapper.multi_sweep as ms
    monkeypatch.setattr(ms, 'sweep_multi_circuits', lambda *a, **k: _mresult())
    tab_before = _on_multi(b)
    content_before = b.content_box.children[0]

    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    # Same widget instances — the visible panel was updated in place, not rebuilt.
    assert b._auto_lucj_tabs is tab_before
    assert b.content_box.children[0] is content_before
    assert b._auto_lucj_tabs.selected_index == 1          # still on Multi
    assert 'circuit(s) placed' in b.lbl_lucj_multi.value  # results rendered


# ── W3 t2: the targeted widgets ARE refreshed (no rebuild ≠ no update) ───────

def test_multi_generate_targeted_widgets_refreshed(monkeypatch):
    b, state = _build(monkeypatch)
    import iqmapper.multi_sweep as ms
    monkeypatch.setattr(ms, 'sweep_multi_circuits', lambda *a, **k: _mresult())
    _on_multi(b)

    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    # Per-circuit override dropdowns reflect the 2-circuit result (in place).
    dds = [c for c in b.lucj_multi_detail_box.children if isinstance(c, Dropdown)]
    assert len(dds) == 2
    # ancilla-tool options + manual per-circuit tabs re-seeded for the new count
    # (the targeted _build_dynamic_ui, done WITHOUT a content rebuild).
    anc_labels = [lbl for lbl, _ in b.ancilla_tool.options]
    assert 'Add Ancilla C1' in anc_labels and 'Add Ancilla C2' in anc_labels
    assert len(b.manual_tabs.children) == 2


# ── W3 t3: a checkbox toggle STILL rebuilds legitimately (X behavior intact) ─

def test_checkbox_toggle_still_rebuilds(monkeypatch):
    b, state = _build(monkeypatch)
    b.intra_qpu_mode.value = True
    b.nav_bar.value = 'auto_lucj'
    tab_before = b._auto_lucj_tabs

    b.intra_qpu_mode.value = False               # legitimate structural change
    assert b._auto_lucj_tabs is not tab_before    # a real rebuild happened
    assert len(b._auto_lucj_tabs.children) == 1   # Multi subpanel dropped


# ── W3 t4: the legacy multi Generate path gets the same no-rebuild treatment ─

def test_legacy_multi_generate_no_rebuild(monkeypatch):
    b, state = _build(monkeypatch)
    monkeypatch.setattr(
        "iqmapper.auto_lucj.get_multi_circuit_layouts",
        lambda **k: [
            {'circuit_idx': i, 'norb': 8, 'num_bridges': 0,
             'all_candidates': [{'layout': [0, 1, 2, 3], 'alpha': [0, 1],
                                 'beta': [2, 3], 'score': 0.03, 'rank': 1,
                                 'layout_full': [0, 1, 2, 3, 4]}]}
            for i in range(2)])
    tab_before = _on_multi(b)

    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi.click()                     # legacy (non-sweep) multi

    assert b._auto_lucj_tabs is tab_before        # no rebuild here either
    assert b._auto_lucj_tabs.selected_index == 1
    assert 'circuits placed' in b.lbl_lucj_multi.value
