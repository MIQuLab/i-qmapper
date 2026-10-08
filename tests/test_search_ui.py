"""The search controls: one "Guided search" checkbox (default on) instead of
an engine selector, the combined search behind every Auto run, the source of
each candidate in its label, and no "alt region" suffix.

Headless, offline; the combined search is stubbed where only the UI is under
test.
"""
import inspect

import pytest

from test_auto_panel_ui_polish import _descendants, _fake_device


def _build(monkeypatch):
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    for key, val in dict(
            active_backend=object(), active_backend_name='ibm_here',
            active_chip='stub_chip', active_service=None,
            available_backends=[], props_warning=None,
            initial_props=object(), active_ansatz='lucj',
            active_vendor='IBM', active_coords=[], custom_colors=[],
            cal_history_dates=['_seed_']).items():
        setattr(state, key, val)
    b = MapperUIBuilder(state)
    b._test_full_ui, _fig = b.build()
    return b


def _entry(ancillas, lqs, source, rank=1):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': ancillas, 'bridge_slots': (0,), 'product_score': lqs,
            'additive_score': 0.1, 'lqs_circuit': lqs, 'rank_in_stratum': rank,
            'source': source, 'diverse': source == 'guided'}


def _result(strata, guided_winners=()):
    return {'strata': strata, 'k_max_effective': max(strata),
            'min_bridges': min(strata), 'vf2_truncated': False,
            'n_raw_embeddings': 3, 'n_candidates_scored': 3,
            'excluded_qubits': [],
            'per_stratum': {k: {'n_direct': 1, 'n_derived': 0,
                                'truncated': False, 'shuffled': False}
                            for k in strata},
            'mapping_cap': 2000, 'reservoir': {}, 'reservoir_notices': [],
            'engine': 'combined', 'use_guided': True,
            'guided_winners': list(guided_winners), 'guided_error': None}


def _stub_combined(monkeypatch, result):
    seen = []
    monkeypatch.setattr('iqmapper.engine_select.combined_sweep',
                        lambda norb, backend, props, **k:
                        seen.append(k) or result)
    return seen


def _run(b):
    b.lucj_ancilla_dd.value = -1                    # sweep mode
    b.btn_lucj_run.click()


# ── the checkbox ────────────────────────────────────────────────────

def test_guided_search_is_one_checkbox_on_by_default(monkeypatch):
    b = _build(monkeypatch)
    chk = b.chk_guided_search
    assert chk.value is True and chk.description == 'Guided search'
    assert 'README: Search engine' in chk.tooltip and len(chk.tooltip) <= 90
    assert any(w is chk for w in _descendants(b.lucj_advanced_acc))


def test_the_engine_dropdown_and_threshold_are_gone(monkeypatch):
    b = _build(monkeypatch)
    assert not hasattr(b, 'dd_search_engine')
    assert not hasattr(b, 'int_guided_max_k')
    descriptions = [getattr(w, 'description', '') or ''
                    for w in _descendants(b._test_full_ui)]
    assert not any('Search engine' in d for d in descriptions)
    assert not any('Guided up to' in d for d in descriptions)


def test_the_checkboxes_own_their_layouts(monkeypatch):
    b = _build(monkeypatch)
    toggled = (b.chk_guided_search, b.bench_guided_box, b.chk_fast_multi,
               b.joint_params_box)
    shared = (b.txt_auto.layout, b.chk_diverse_cap.layout,
              b.chk_deep_search.layout, b.lucj_buffer_slider.layout)
    for w in toggled:
        assert all(w.layout is not lay for lay in shared)
    assert len({id(w.layout) for w in toggled}) == len(toggled)


def test_beam_parameters_show_only_while_guided_is_on(monkeypatch):
    b = _build(monkeypatch)
    assert b.bench_guided_box.layout.display == ''
    b.chk_guided_search.value = False
    assert b.bench_guided_box.layout.display == 'none'
    assert b.txt_auto.layout.display != 'none'
    b.chk_guided_search.value = True
    assert b.bench_guided_box.layout.display == ''


# ── every Auto run is the combined search ──────────────────────────

@pytest.mark.parametrize('guided', [True, False])
def test_the_run_calls_the_combined_search(monkeypatch, guided):
    b = _build(monkeypatch)
    seen = _stub_combined(monkeypatch, _result({1: [_entry([4], 0.8,
                                                           'exhaustive')]}))
    b.chk_guided_search.value = guided
    b.int_beam_width.value = 64
    b.int_refine_top.value = 2
    _run(b)
    assert seen, 'combined_sweep was not called'
    k = seen[0]
    assert k['use_guided'] is guided
    assert k['beam_width'] == 64 and k['refine_top'] == 2
    assert {'mapping_cap', 'hybrid', 'diverse_cap', 'top_n',
            'circuit_score_fn'} <= set(k)


# ── status line ─────────────────────────────────────────────────────

def test_status_is_one_short_line(monkeypatch):
    b = _build(monkeypatch)
    _stub_combined(monkeypatch, _result(
        {2: [_entry([4, 5], 0.7, 'guided')], 1: [_entry([4], 0.8, 'guided')]},
        guided_winners=[2, 1]))
    _run(b)
    status = b.lbl_lucj_status.value
    assert '✅ 2 layouts found · scored on live · ' in status, status
    for gone in ('guided', 'embeddings', 'best per k', 'direct+derived',
                 'capped', 'Sweep:'):
        assert gone not in status, gone


# ── labels ──────────────────────────────────────────────────────────

def test_each_candidate_label_shows_its_source(monkeypatch):
    b = _build(monkeypatch)
    _stub_combined(monkeypatch, _result({1: [
        _entry([4], 0.8, 'guided', rank=1),
        _entry([5], 0.7, 'both', rank=2),
        _entry([6], 0.6, 'exhaustive', rank=3)]}, guided_winners=[1]))
    _run(b)
    labels = [label for label, _ in b.dd_lucj_layout.options]
    assert labels == ['k=1 · #1 · LQS 80.0% · G',
                      'k=1 · #2 · LQS 70.0% · E+G',
                      'k=1 · #3 · LQS 60.0% · E']


def test_no_alt_region_suffix_anywhere(monkeypatch):
    from iqmapper.ui import lucj_panel
    assert 'alt region' not in inspect.getsource(lucj_panel)
    b = _build(monkeypatch)
    _stub_combined(monkeypatch, _result({1: [
        _entry([4], 0.8, 'guided'), _entry([5], 0.7, 'exhaustive', rank=2)]}))
    _run(b)
    assert not any('alt region' in label
                   for label, _ in b.dd_lucj_layout.options)
