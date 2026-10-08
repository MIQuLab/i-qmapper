"""Auto (LUCJ) panel usability polish.

Three UI-only changes, none of which touch the sweep engines:

  * the expert knobs (search cap + deep search) live in a collapsed
    "Advanced" accordion instead of the primary ancilla-dropdown -> Run flow;
  * "Prefer diverse regions" ships ON, because diversity only shapes the
    second half of each stratum's list (ancilla_sweep._select_with_diversity
    keeps the first ceil(top_n/2) slots as the absolute best);
  * the legacy "Score layouts" checkbox is hidden in Auto/sweep mode rather
    than shown greyed out, since Auto is the default the user lands on.
"""
import logging

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
    # build() returns the assembled tree rather than storing it; keep it so
    # tests can walk the real sidebar/figure widget graph.
    b._test_full_ui, _fig = b.build()
    b.chk_guided_search.value = False    # stubbed exhaustive engine only
    return b, state


def _descendants(node, seen=None):
    """Every widget reachable through .children, root included."""
    if seen is None:
        seen = []
    if any(n is node for n in seen):
        return seen
    seen.append(node)
    for c in getattr(node, 'children', ()) or ():
        _descendants(c, seen)
    return seen


def _empty_sweep():
    return {'strata': {}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 0,
            'n_candidates_scored': 0, 'excluded_qubits': [],
            'per_stratum': {}, 'mapping_cap': None}


# ── t1: the expert knobs are collapsed out of the primary flow ──────────────

def test_search_cap_and_deep_search_live_in_advanced_accordion(monkeypatch):
    b, _ = _build(monkeypatch)

    acc = b.lucj_advanced_acc
    assert acc.get_title(0) == '⚙️ Advanced'
    assert acc.selected_index is None, "Advanced must render collapsed"

    inside = _descendants(acc)
    assert any(w is b.lucj_search_cap for w in inside), "cap not in Advanced"
    assert any(w is b.chk_deep_search for w in inside), "deep search not in Advanced"

    # The Auto controls box mounts the accordion, not the two widgets directly.
    b._build_auto_lucj_view()
    single = b._auto_single_panel
    assert any(c is acc for c in single.children), "Advanced not mounted"
    assert not any(c is b.lucj_search_cap for c in single.children)
    assert not any(c is b.chk_deep_search for c in single.children)
    # The primary flow is untouched.
    assert any(c is b.lucj_ancilla_dd for c in single.children)
    assert any(c is b.btn_lucj_run for c in single.children)
    # Defaults unchanged by the move.
    assert b.lucj_search_cap.value == 2000
    assert b.chk_deep_search.value is True


# ── t2: diversity ships ON and still reaches the engine both ways ───────────

def test_diverse_default_on(monkeypatch):
    b, _ = _build(monkeypatch)
    assert b.chk_lucj_diverse.value is True, "diversity must default ON"

    seen = {}
    monkeypatch.setattr(
        "iqmapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, bk, pr, **k: (seen.update(k), _empty_sweep())[1])

    b.lucj_ancilla_dd.value = -1                      # sweep mode
    b.btn_lucj_run.click()
    assert seen['diversity_threshold'] == 0.8, (
        f"default-on diversity did not reach the engine: {seen.get('diversity_threshold')!r}")

    seen.clear()
    b.chk_lucj_diverse.value = False
    b.btn_lucj_run.click()
    assert seen['diversity_threshold'] is None, (
        f"unticking diversity did not disable it: {seen.get('diversity_threshold')!r}")


# ── t3: the score checkbox belongs to legacy mode only ──────────────────────

def test_score_checkbox_hidden_in_auto_visible_in_legacy(monkeypatch):
    b, _ = _build(monkeypatch)
    chk = b.lucj_chk_score

    # Auto (-1) is the default the panel opens on.
    assert b.lucj_ancilla_dd.value == -1
    assert chk.layout.display == 'none', "score checkbox visible in Auto mode"

    b.lucj_ancilla_dd.value = 1                       # explicit count -> legacy
    assert chk.layout.display == ''
    assert chk.disabled is False
    assert chk.tooltip == 'Score layouts (pick lowest error)'

    b.lucj_ancilla_dd.value = -1                      # back to Auto
    assert chk.layout.display == 'none'
    assert chk.disabled is True

    # Hidden, not removed: the legacy run path still reads .value.
    b._build_auto_lucj_view()
    assert any(w is chk for w in _descendants(b._auto_single_panel)), (
        "the checkbox must stay in the tree, only hidden")


# ── t4: the hidden checkbox must not drag other widgets down with it ────────

def test_score_checkbox_owns_its_layout(monkeypatch):
    """Regression: `W` is ONE shared Layout instance handed to ~86 widgets, so
    toggling `.layout.display` on a widget built with `layout=W` hid every one
    of them (the I/O paste box, the Run button, ...). The score checkbox must
    own its Layout so the sweep-mode toggle is local to it."""
    b, _ = _build(monkeypatch)
    chk = b.lucj_chk_score

    assert chk.layout is not b.txt_auto.layout, (
        "score checkbox shares its Layout instance with txt_auto — "
        "hiding it would hide every widget built with W")

    b.lucj_ancilla_dd.value = -1                      # Auto -> checkbox hidden
    assert chk.layout.display == 'none'
    # …and nothing else moved.
    assert b.txt_auto.layout.display != 'none'
    assert b.btn_auto.layout.display != 'none'
    assert b.btn_lucj_run.layout.display != 'none'
    assert b.chk_lucj_diverse.layout.display != 'none'

    # No other widget in the tree shares that Layout object.
    b._build_auto_lucj_view()
    sharers = [w for w in _descendants(b._auto_single_panel)
               if getattr(w, 'layout', None) is chk.layout]
    assert sharers == [chk], f"Layout shared with {len(sharers) - 1} other widget(s)"

    # The visual properties W supplied are preserved on the private copy.
    assert (chk.layout.width, chk.layout.margin) == \
           (b.chk_lucj_diverse.layout.width, b.chk_lucj_diverse.layout.margin)


# ── t5: benchmark lives in the sidebar, not the Auto tab ───────────────────

def _sidebar_accordion(b):
    """The numbered settings accordion from the assembled sidebar."""
    from ipywidgets import Accordion
    for w in _descendants(b._test_full_ui):
        if isinstance(w, Accordion) and w.get_title(0) == '1. Hardware':
            return w
    raise AssertionError('numbered settings accordion not found')


def test_benchmark_is_sidebar_section_5_not_an_auto_tab_block(monkeypatch):
    """Benchmarking is a property of the BACKEND, so it belongs beside
    Hardware — it has to serve hand-placed layouts, every circuit slot, the
    whole QPU and the inter-QPU second panel, none of which the Auto tab
    covers."""
    b, _ = _build(monkeypatch)

    acc = _sidebar_accordion(b)
    assert acc.get_title(3) == '4. Multi-Programming'
    assert acc.get_title(4) == '5. Benchmark', 'section 5 missing'
    assert any(w is b.bench_box for w in _descendants(acc.children[4])), \
        'the benchmark controls are not inside section 5'

    # ...and gone from the Auto tab.
    b._build_auto_lucj_view()
    auto = _descendants(b._auto_single_panel)
    for widget, name in ((b.bench_box, 'bench_box'),
                         (b.btn_bench_run, 'btn_bench_run'),
                         (b.dd_bench_target, 'dd_bench_target')):
        assert not any(w is widget for w in auto), \
            f'{name} still mounted in the Auto tab'


def test_benchmark_widgets_do_not_share_a_mutated_layout(monkeypatch):
    """Same bug class as the score checkbox: `W` is ONE shared Layout, so a
    widget whose .layout.display is toggled at runtime must own its own."""
    b, _ = _build(monkeypatch)
    mutated = [('txt_bench_custom', b.txt_bench_custom),
               ('tb_bench_qpu', b.tb_bench_qpu),
               ('chk_bench_ack', b.chk_bench_ack),
               ('btn_bench_reuse', b.btn_bench_reuse)]

    everything = _descendants(b._test_full_ui)
    for name, widget in mutated:
        assert widget.layout is not b.txt_auto.layout, (
            f'{name} shares the shared W Layout — toggling its display '
            f'would hide every widget built with W')
        sharers = [w for w in everything
                   if getattr(w, 'layout', None) is widget.layout]
        assert len(set(id(w) for w in sharers)) <= 1, (
            f'{name} shares its Layout with {len(sharers) - 1} other widget(s)')

    # Hidden-by-default widgets really do start hidden.
    assert b.txt_bench_custom.layout.display == 'none'
    assert b.tb_bench_qpu.layout.display == 'none'


# ── t7: the search (engine widgets: see test_search_ui.py) ─────────────────

def test_exhaustive_still_routes_to_the_frozen_sweep(monkeypatch):
    b, _ = _build(monkeypatch)
    called = []
    monkeypatch.setattr(
        'iqmapper.ancilla_sweep.sweep_ancilla_layouts',
        lambda norb, backend, props, **k: called.append(k) or {
            'strata': {}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 0,
            'n_candidates_scored': 0, 'excluded_qubits': [],
            'per_stratum': {}, 'mapping_cap': 2000})

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()
    assert called, 'the frozen sweep was not used for Exhaustive'
    assert 'mapping_cap' in called[0], 'VF2 must still receive its cap'


def test_guided_results_render_through_the_existing_display(monkeypatch):
    """The display code must not be able to tell the engines apart."""
    import sys
    sys.path.insert(0, 'tests')
    from test_ancilla_sweep import FakeProps, make_backend
    from iqmapper.guided_search import guided_sweep

    b, _ = _build(monkeypatch)
    res = guided_sweep(4, make_backend(), FakeProps(), min_bridges=1, top_n=3)

    from iqmapper.ui.lucj_panel import _format_best_per_k
    best = _format_best_per_k(res['strata'])
    assert best, 'best-per-k line empty for a guided result'


# ── t8: diverse enumeration ────────────────────────────────────────────────

def test_diverse_enumeration_checkbox_defaults_on(monkeypatch):
    b, _ = _build(monkeypatch)
    assert b.chk_diverse_cap.value is True
    assert any(w is b.chk_diverse_cap
               for w in _descendants(b.lucj_advanced_acc)), \
        'the checkbox is not in the Advanced accordion'
    tip = b.chk_diverse_cap.tooltip
    assert 'distinct candidates' in tip, tip
    assert 'README: Diverse enumeration' in tip, tip
    assert len(tip) <= 90, f'tooltip is {len(tip)} chars'


def test_diverse_enumeration_reaches_the_sweep(monkeypatch):
    b, _ = _build(monkeypatch)
    seen = {}
    monkeypatch.setattr(
        'iqmapper.ancilla_sweep.sweep_ancilla_layouts',
        lambda norb, backend, props, **k: seen.update(k) or {
            'strata': {}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': False, 'n_raw_embeddings': 0,
            'n_candidates_scored': 0, 'excluded_qubits': [],
            'per_stratum': {}, 'mapping_cap': 2000,
            'reservoir': {}, 'reservoir_notices': []})

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()
    assert seen.get('diverse_cap') is True

    seen.clear()
    b.chk_diverse_cap.value = False
    b.btn_lucj_run.click()
    assert seen.get('diverse_cap') is False


class _Collect(logging.Handler):
    def __init__(self):
        super().__init__(level=0)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def test_the_cap_notice_goes_to_the_log_not_the_ui(monkeypatch):
    """The file handler sits on the package logger, so a handler there sees
    what the log file gets (the sweep's notice capture stops propagation)."""
    import logging
    b, _ = _build(monkeypatch)
    entry = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
             'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.9,
             'additive_score': 0.1, 'lqs_circuit': 0.8, 'rank_in_stratum': 1,
             'source': 'vf2'}
    note = ('hybrid k=1: cap reached — kept 2,000 diverse candidates, '
            'skipped 34,812 near-duplicates')
    monkeypatch.setattr(
        'iqmapper.ancilla_sweep.sweep_ancilla_layouts',
        lambda norb, backend, props, **k: {
            'strata': {1: [entry]}, 'k_max_effective': 1, 'min_bridges': 1,
            'vf2_truncated': True, 'n_raw_embeddings': 2000,
            'n_candidates_scored': 2000, 'excluded_qubits': [],
            'per_stratum': {1: {'n_direct': 2000, 'n_derived': 0,
                                'truncated': True, 'shuffled': False}},
            'mapping_cap': 2000,
            'reservoir': {1: {'kept': 2000, 'skipped_similar': 34812}},
            'reservoir_notices': [note]})

    b.lucj_ancilla_dd.value = -1
    pkg = logging.getLogger('iqmapper')
    collect = _Collect()
    monkeypatch.setattr(pkg, 'level', logging.INFO)
    pkg.addHandler(collect)
    try:
        b.btn_lucj_run.click()
    finally:
        pkg.removeHandler(collect)
    status = b.lbl_lucj_status.value
    assert 'cap reached' not in status and 'near-duplicates' not in status
    assert 'cap reached' not in b.lbl_lucj_notices.value
    assert status.endswith('✅ 1 layout found · ' + status.split(' · ', 1)[1])
    assert any('hybrid k=1: cap reached' in m for m in collect.messages)


# ── t9: one cap notice per stratum, not two ────────────────────────────────

_CAP_ENTRY = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
              'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.9,
              'additive_score': 0.1, 'lqs_circuit': 0.8,
              'rank_in_stratum': 1, 'source': 'vf2'}


def _capped_result(*, reservoir, notices):
    """A sweep result whose k=3 stratum hit the cap."""
    return {
        'strata': {3: [dict(_CAP_ENTRY)]}, 'k_max_effective': 3,
        'min_bridges': 3, 'vf2_truncated': True,
        'n_raw_embeddings': 2000, 'n_candidates_scored': 2000,
        'excluded_qubits': [],
        'per_stratum': {3: {'n_direct': 2000, 'n_derived': 0,
                            'truncated': True, 'shuffled': False}},
        'mapping_cap': 2000,
        'reservoir': reservoir, 'reservoir_notices': notices,
    }


def test_reservoir_replaces_the_truncation_warning(monkeypatch):
    """A stratum the reservoir covered gets no ⚠️ banner: its cap was
    reached by design, and the cap line goes to the log file."""
    b, _ = _build(monkeypatch)
    note = ('hybrid k=3: cap reached — kept 2,000 diverse candidates, '
            'skipped 2,498 near-duplicates (of 4,498 seen)')
    monkeypatch.setattr(
        'iqmapper.ancilla_sweep.sweep_ancilla_layouts',
        lambda norb, backend, props, **k: _capped_result(
            reservoir={3: {'kept': 2000, 'skipped_similar': 2498,
                           'offered': 4498}},
            notices=[note]))

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()

    assert b.lbl_lucj_trunc.value == '', (
        f'the ⚠️ banner was not suppressed: {b.lbl_lucj_trunc.value}')
    status = b.lbl_lucj_status.value
    assert 'cap reached' not in status, status
    assert 'non-exhaustive' not in status


def test_the_warning_survives_when_diverse_cap_is_off(monkeypatch):
    """With no reservoir there is nothing else saying the cap was hit."""
    b, _ = _build(monkeypatch)
    monkeypatch.setattr(
        'iqmapper.ancilla_sweep.sweep_ancilla_layouts',
        lambda norb, backend, props, **k: _capped_result(reservoir={},
                                                         notices=[]))

    b.chk_diverse_cap.value = False
    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()

    banner = b.lbl_lucj_trunc.value
    assert '⚠️ direct search truncated at' in banner, banner
    assert '2,000 mappings for k=3' in banner, banner
    assert 'non-exhaustive' in banner


def test_a_stratum_the_reservoir_missed_still_warns(monkeypatch):
    """Mixed case: k=3 covered by the reservoir, k=2 truncated without it."""
    b, _ = _build(monkeypatch)
    result = _capped_result(
        reservoir={3: {'kept': 2000, 'skipped_similar': 10, 'offered': 2010}},
        notices=['hybrid k=3: cap reached — kept 2,000 diverse candidates, '
                 'skipped 10 near-duplicates (of 2,010 seen)'])
    result['per_stratum'][2] = {'n_direct': 2000, 'n_derived': 0,
                                'truncated': True, 'shuffled': False}
    result['strata'][2] = [dict(_CAP_ENTRY)]
    monkeypatch.setattr('iqmapper.ancilla_sweep.sweep_ancilla_layouts',
                        lambda norb, backend, props, **k: result)

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()

    banner = b.lbl_lucj_trunc.value
    assert 'k=2' in banner, banner
    assert 'k=3' not in banner, 'the reservoir-covered stratum was re-warned'
