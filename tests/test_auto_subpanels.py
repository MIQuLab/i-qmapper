"""Auto-tab Single/Multi subpanels, candidate-detail declutter, and the
export-is-auto-current pin.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ipywidgets import Button


def _fake_device(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _entry(anc, rank):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': list(anc), 'bridge_slots': (0,), 'product_score': 0.4,
            'additive_score': 0.5, 'lqs_circuit': 0.4, 'rank_in_stratum': rank,
            'source': 'derived'}


def _canned():
    e = [_entry([4], 1), _entry([5], 2)]
    return {'strata': {1: e}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 5,
            'n_candidates_scored': 5, 'excluded_qubits': [],
            'per_stratum': {1: {'n_direct': 0, 'n_derived': 2,
                                'truncated': False, 'shuffled': False}},
            'mapping_cap': 2000}


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


# ── t1: subpanels follow the Multi-Programming enable ───────────────────────

def test_subpanels_follow_multi_programming(monkeypatch):
    b, state = _build(monkeypatch)

    b.intra_qpu_mode.value = False
    b._build_auto_lucj_view()
    assert len(b._auto_lucj_tabs.children) == 1              # Single only
    assert b._auto_lucj_tabs.get_title(0) == '🔬 Single'
    # Multi widgets are not in the single panel.
    assert b.btn_scan_batch not in b._auto_single_panel.children

    b.intra_qpu_mode.value = True
    b._build_auto_lucj_view()
    assert len(b._auto_lucj_tabs.children) == 2              # Single + Multi
    assert b._auto_lucj_tabs.get_title(1) == '🔗 Multi'
    assert b.btn_scan_batch in b._auto_multi_panel.children
    assert b.btn_lucj_multi_sweep in b._auto_multi_panel.children


# ── R3 t1: exactly TWO buttons in the Multi subpanel — no legacy node ────────

def test_multi_subpanel_has_two_buttons_no_legacy(monkeypatch):
    b, state = _build(monkeypatch)
    b.intra_qpu_mode.value = True
    b._build_auto_lucj_view()

    buttons = [w for w in b._auto_multi_panel.children if isinstance(w, Button)]
    assert len(buttons) == 2                                 # generate + batch
    assert set(buttons) == {b.btn_lucj_multi_sweep, b.btn_scan_batch}
    assert b.btn_lucj_multi not in b._auto_multi_panel.children   # legacy gone
    # The sweep button is the single generate action — no "(sweep)" qualifier.
    assert b.btn_lucj_multi_sweep.description == '🧺 Generate Multi-Circuit'
    # The ancilla field speaks one (sweep) language.
    assert b.lucj_multi_ancillas.tooltip == 'min ancillas per circuit (blank = auto)'
    assert 'Legacy' not in (b.lucj_multi_ancillas.tooltip or '')


# ── R3 t2: the legacy handler + engine stay defined (source pin) ────────────

def test_legacy_handler_and_engine_still_defined():
    panel = Path('qpu_mapper/ui/lucj_panel.py').read_text()
    assert 'def _on_lucj_multi(' in panel                    # handler retained
    assert 'get_multi_circuit_layouts' in panel              # engine path retained
    # And the widget object is still wired (clickable) for the pinned tests.
    from qpu_mapper.ui.builder import MapperUIBuilder
    assert hasattr(MapperUIBuilder, '_build_auto_lucj_view')


# ── t3: candidate detail shows counts, never qubit index lists ──────────────

def test_detail_panel_counts_not_qubit_lists(monkeypatch):
    b, state = _build(monkeypatch)
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _canned())
    b.lucj_ancilla_dd.value = -1                             # sweep mode
    b.btn_lucj_run.click()
    b.dd_lucj_layout.value = 1                               # change → fire select

    v = b.lbl_lucj_detail.value
    assert 'α=2' in v and 'β=2' in v and 'Anc=1' in v        # compact counts
    assert 'α: ' not in v and 'β: ' not in v                 # no index dumps
    assert '0, 1' not in v and 'ancillas:' not in v          # no id lists


# ── t5: an applied batch (C1..Cn) is auto-reflected in the export ──────────

def test_export_reads_applied_batch():
    from qpu_mapper.export.codegen import get_qiskit_layout_code
    state = SimpleNamespace(layouts_data={
        0: {'alpha': ['0', '1'], 'beta': ['2', '3'], 'other': [], 'name': 'C1'},
        1: {'alpha': ['4', '5'], 'beta': ['6', '7'], 'other': [], 'name': 'C2'},
    })
    # n_circ = 2 (what the export computes when intra-QPU is on with 2 circuits).
    code = get_qiskit_layout_code(state, 'lucj', 2)
    assert 'C1' in code and 'C2' in code                      # both circuits
    assert '0' in code and '7' in code                        # each circuit's qubits
    # Circuit 1's physical qubits appear in its layout dict.
    assert '{0: 0, 1: 1, 2: 2, 3: 3}' in code
    assert '{0: 4, 1: 5, 2: 6, 3: 7}' in code
