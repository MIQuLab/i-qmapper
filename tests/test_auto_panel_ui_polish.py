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
