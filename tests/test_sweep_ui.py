"""Headless tests for the Auto-tab ancilla-sweep UI wiring (Component 1b+2).

Builds the real MapperUIBuilder with the data layer stubbed (deterministic
8-qubit line device, offline) and monkeypatches the engine entry points so
NO real VF2 runs in the UI tests. Exercises: sweep-mode dropdown population +
label format, legacy-mode routing, apply-on-select (chains + kept ancillas),
the truncation banner, and engine-ValueError surfacing.
"""
import numpy as np
import pytest


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    """Deterministic 8-qubit line device (enough qubits for a k=3 candidate)."""
    n = 8
    x = list(range(n))
    y = [0.0] * n
    edges = [(i, i + 1) for i in range(n - 1)]
    qids = [str(i) for i in range(n)]
    readout = np.array([0.01 * (i + 1) for i in range(n)])
    gate = np.array([0.001 * (i + 1) for i in range(len(edges))])
    return x, y, edges, qids, readout, gate


@pytest.fixture
def builder(monkeypatch):
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)
    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder

    state = SessionState()
    state.active_backend = object()          # non-None, and NO .coords attr
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = None
    state.available_backends = []
    state.props_warning = None
    state.initial_props = object()           # opaque; engine is monkeypatched
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]     # skip the bg auto-load thread

    b = MapperUIBuilder(state)
    b.build()
    return b


def _labels(dd):
    return [opt[0] if isinstance(opt, tuple) else opt for opt in dd.options]


def _fake_sweep_result(vf2_truncated=False):
    """Crafted stratified result: k_max=3, strata 3/2/1, one entry each.
    product_score chosen so LQS renders as 20.0 / 30.0 / 40.0%."""
    def entry(anc, prod, add, src):
        return {
            'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': anc, 'bridge_slots': tuple(range(0, 4 * len(anc), 4)),
            'product_score': prod, 'additive_score': add,
            # engine now attaches circuit-level LQS; equal to product here
            # so the rendered percentage is unchanged in these UI assertions.
            'lqs_circuit': prod,
            'rank_in_stratum': 1, 'source': src,
        }
    return {
        'strata': {
            3: [entry([4, 5, 6], 0.20, 1.0, 'vf2')],
            2: [entry([4, 5], 0.30, 0.8, 'derived')],
            1: [entry([4], 0.40, 0.5, 'derived')],
        },
        'k_max_effective': 3, 'min_bridges': 1,
        'vf2_truncated': vf2_truncated,
        'n_raw_embeddings': 214, 'n_candidates_scored': 1891,
        # real engine metadata: which strata hit the cap, and the cap.
        'per_stratum': {
            3: {'n_direct': 5, 'n_derived': 0, 'truncated': vf2_truncated},
            2: {'n_direct': 0, 'n_derived': 5, 'truncated': False},
            1: {'n_direct': 0, 'n_derived': 5, 'truncated': False},
        },
        'mapping_cap': 2000,
    }


# ── label/status formatters prefer circuit-level LQS ─────────────────

def test_formatters_use_lqs_circuit():
    from qpu_mapper.ui.lucj_panel import _entry_lqs_pct, _format_best_per_k

    circ = {'product_score': 0.30, 'lqs_circuit': 0.42, 'ancillas': [4]}
    pct, is_tmpl = _entry_lqs_pct(circ)
    assert pct == '42.0%' and is_tmpl is False          # circuit LQS, no flag

    tmpl = {'product_score': 0.30, 'ancillas': [4]}      # no lqs_circuit
    pct2, is_tmpl2 = _entry_lqs_pct(tmpl)
    assert pct2 == '30.0%' and is_tmpl2 is True          # falls back, flagged

    strata = {2: [{'product_score': 0.3, 'lqs_circuit': 0.5}],
              1: [{'product_score': 0.4, 'lqs_circuit': 0.6}]}
    assert _format_best_per_k(strata) == '2→50.0% · 1→60.0%'
    assert _format_best_per_k({1: [{'product_score': 0.4}]}) == '1→40.0%'


# ── sweep mode default → stratified dropdown, exact label format ────────

def test_sweep_mode_populates_stratified_dropdown(builder, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: (calls.append((a, k)) or _fake_sweep_result()))

    b = builder
    assert b.lucj_ancilla_dd.value == -1          # 'Auto' is the default
    b.btn_lucj_run.click()

    assert len(calls) == 1                          # sweep path taken
    assert _labels(b.dd_lucj_layout) == [
        "k=3 · #1 · LQS 20.0%",
        "k=2 · #1 · LQS 30.0%",
        "k=1 · #1 · LQS 40.0%",
    ]
    # k descending, values are flat indices.
    assert [opt[1] for opt in b.dd_lucj_layout.options] == [0, 1, 2]


# ── the "Show top" field controls entries-per-stratum in the dropdown ───

def _fake_sweep_topn(*a, **k):
    """Echo the requested top_n: k_max=3, each stratum gets top_n entries."""
    n = k['top_n']

    def entry(kk, rank):
        anc = list(range(4, 4 + kk))
        return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
                'ancillas': anc, 'bridge_slots': tuple(range(0, 4 * kk, 4)),
                'product_score': 0.3, 'additive_score': 0.5, 'lqs_circuit': 0.3,
                'rank_in_stratum': rank, 'source': 'derived'}
    return {
        'strata': {kk: [entry(kk, r + 1) for r in range(n)]
                   for kk in (3, 2, 1)},
        'k_max_effective': 3, 'min_bridges': 1, 'vf2_truncated': False,
        'n_raw_embeddings': 50, 'n_candidates_scored': 90,
        'per_stratum': {kk: {'n_direct': 0, 'n_derived': n, 'truncated': False}
                        for kk in (3, 2, 1)},
        'mapping_cap': 2000,
    }


def test_top_n_field_controls_dropdown_entries(builder, monkeypatch):
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        _fake_sweep_topn)
    b = builder
    b.lucj_top_n.value = 7                            # Show top = 7
    b.btn_lucj_run.click()

    labels = _labels(b.dd_lucj_layout)
    for kk in (3, 2, 1):
        assert sum(lbl.startswith(f"k={kk} · ") for lbl in labels) == 7
    assert len(labels) == 21                          # 3 strata × 7


# ── legacy mode → single-search path, sweep NOT called ──────────────────

def test_legacy_mode_skips_sweep(builder, monkeypatch):
    sweep_calls, legacy_calls = [], []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: sweep_calls.append(1) or _fake_sweep_result())

    def fake_legacy(*a, **k):
        legacy_calls.append((a, k))
        return ([{'alpha': [0, 1], 'beta': [2, 3], 'score': 0.05, 'rank': 1}], 2)
    monkeypatch.setattr(
        "qpu_mapper.auto_lucj.get_all_zigzag_layouts", fake_legacy)

    b = builder
    b.lucj_ancilla_dd.value = 2                     # exact count → legacy
    b.btn_lucj_run.click()

    assert sweep_calls == []                         # sweep NOT called
    assert len(legacy_calls) == 1                    # legacy called
    # requested_bridges passed through as the exact count.
    assert legacy_calls[0][1]['requested_bridges'] == 2


# ── apply a k=2 derived entry → chains + its 2 ancillas placed ──────────

def test_apply_derived_candidate_places_chains_and_ancillas(builder, monkeypatch):
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: _fake_sweep_result())

    b = builder
    b.btn_lucj_run.click()                           # applies flat[0] (k=3)

    # Select the k=2 derived entry (flat index 1).
    b.dd_lucj_layout.value = 1

    d = b.state.layouts_data[0]
    assert d['alpha'] == ['0', '1']
    assert d['beta'] == ['2', '3']
    assert d['ancilla'] == ['4', '5']                # exactly its 2 kept bridges
    # Consistent role lengths (the metadata downstream derives from).
    assert (len(d['alpha']), len(d['beta']), len(d['ancilla'])) == (2, 2, 2)
    assert d['other'] == [] and d['qubits'] == []
    # Every placed qubit is styled.
    for q in ['0', '1', '2', '3', '4', '5']:
        assert q in d['styles']


# ── truncation banner visibility ────────────────────────────────────────

def test_truncation_banner(builder, monkeypatch):
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: _fake_sweep_result(vf2_truncated=True))
    b = builder
    b.btn_lucj_run.click()
    # names the REAL cap and the truncated stratum — not n_raw_embeddings.
    assert "truncated at 2,000 mappings for k=3" in b.lbl_lucj_trunc.value
    assert "214" not in b.lbl_lucj_trunc.value          # old embedding-count gone
    assert b.lbl_lucj_trunc.value != ""

    # False → hidden (empty).
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: _fake_sweep_result(vf2_truncated=False))
    b.btn_lucj_run.click()
    assert b.lbl_lucj_trunc.value == ""


# ── engine ValueError surfaces as a friendly status, no traceback ───────

def test_min_bridges_validation_surfaces_as_status(builder, monkeypatch):
    def boom(*a, **k):
        raise ValueError("min_bridges=99 out of range; must satisfy "
                         "1 <= min_bridges <= k_max_effective (3).")
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts", boom)

    b = builder
    b.lucj_min_ancilla.value = 99
    b.btn_lucj_run.click()                           # must NOT raise

    assert "Error" in b.lbl_lucj_status.value
    assert "min_bridges=99 out of range" in b.lbl_lucj_status.value
    # UI stays usable: the button is re-enabled after the failure.
    assert b.btn_lucj_run.disabled is False


# ── the deep-search checkbox wires hybrid= and the per-stratum mode label ────────

def test_deep_search_checkbox_wires_hybrid(builder, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: (calls.append(k) or _fake_sweep_result()))
    b = builder
    b.chk_deep_search.value = False
    b.btn_lucj_run.click()
    assert calls[-1]['hybrid'] is False
    b.chk_deep_search.value = True
    b.btn_lucj_run.click()
    assert calls[-1]['hybrid'] is True


def test_sweep_mode_label():
    from qpu_mapper.ui.lucj_panel import _sweep_mode_label
    hy = {'k_max_effective': 3,
          'per_stratum': {3: {'n_direct': 5, 'n_derived': 0},
                          2: {'n_direct': 2, 'n_derived': 3},
                          1: {'n_direct': 0, 'n_derived': 4}}}
    assert _sweep_mode_label(hy) == 'direct+derived'      # k=2 got direct hits
    dv = {'k_max_effective': 3,
          'per_stratum': {3: {'n_direct': 5, 'n_derived': 0},
                          2: {'n_direct': 0, 'n_derived': 3},
                          1: {'n_direct': 0, 'n_derived': 4}}}
    assert _sweep_mode_label(dv) == 'derived-only'


# ── diverse entries get a "· alt region" dropdown-label suffix ──────────

def test_diverse_entry_label_suffix(builder, monkeypatch):
    def _fake(*a, **k):
        def e(anc, prod, diverse):
            return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
                    'ancillas': anc, 'bridge_slots': tuple(range(0, 4 * len(anc), 4)),
                    'product_score': prod, 'additive_score': 0.5,
                    'lqs_circuit': prod, 'rank_in_stratum': 1, 'source': 'derived',
                    'diverse': diverse}
        return {
            'strata': {2: [dict(e([4, 5], 0.30, False), rank_in_stratum=1),
                           dict(e([6, 7], 0.28, True), rank_in_stratum=2)]},
            'k_max_effective': 2, 'min_bridges': 2, 'vf2_truncated': False,
            'n_raw_embeddings': 5, 'n_candidates_scored': 5,
            'per_stratum': {2: {'n_direct': 0, 'n_derived': 2,
                                'truncated': False, 'shuffled': False}},
            'mapping_cap': 2000,
        }
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts", _fake)
    b = builder
    b.btn_lucj_run.click()
    labels = _labels(b.dd_lucj_layout)
    assert any('· alt region' in lab for lab in labels)          # the diverse one
    assert sum('· alt region' in lab for lab in labels) == 1     # only it
