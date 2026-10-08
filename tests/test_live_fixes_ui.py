"""UI fixes from the 0.1.0 live tests: technical notices kept out of the UI,
complete batches ranked first, the Plans menu and compatible alternatives,
and joint placement by default.

Headless, offline; engines are stubbed with canned results except where the
fit check needs a real coupling graph.
"""
import logging
from types import SimpleNamespace

import numpy as np
from ipywidgets import Dropdown, HTML

from test_fit_check import ladder
from test_multi_sweep_ui import (_build as _build_multi, _circuit, _dropdowns,
                                 _entry, _patch_engine, _plans, _result)
from test_search_ui import _build as _build_auto


# ── A2: two technical notices stay out of the UI ────────────────────

def test_bridge_fallback_and_cap_line_never_reach_the_ui(monkeypatch):
    b = _build_auto(monkeypatch)
    entry = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
             'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.9,
             'additive_score': 0.1, 'lqs_circuit': 0.8, 'rank_in_stratum': 1,
             'source': 'exhaustive'}

    def _sweep(*a, **k):
        logging.getLogger('iqmapper.auto_lucj').warning(
            "Requested %d bridges does not fit on this backend; falling "
            "back to auto (max-then-reduce). The result may use more "
            "than %d bridges.", 6, 6)
        logging.getLogger('iqmapper.ancilla_sweep').warning(
            "VF2 truncating to %d mappings", 2000)
        return {'strata': {1: [entry]}, 'k_max_effective': 1,
                'min_bridges': 1, 'vf2_truncated': False,
                'n_raw_embeddings': 1, 'n_candidates_scored': 1,
                'excluded_qubits': [], 'per_stratum': {}, 'mapping_cap': 2000,
                'reservoir': {1: {'kept': 2000, 'skipped_similar': 9}},
                'reservoir_notices': [
                    'hybrid k=1: cap reached — kept 2,000 diverse '
                    'candidates, skipped 9 near-duplicates (of 2,009 seen)'],
                'engine': 'combined', 'guided_winners': []}

    monkeypatch.setattr('iqmapper.engine_select.combined_sweep', _sweep)
    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()
    shown = b.lbl_lucj_status.value + b.lbl_lucj_notices.value
    assert 'does not fit on this backend' not in shown
    assert 'cap reached' not in shown and 'near-duplicates' not in shown
    assert 'VF2 truncating to 2000 mappings' in b.lbl_lucj_notices.value
    n = len(b.dd_lucj_layout.options)
    assert f'✅ {n} layout found · scored on live' in b.lbl_lucj_status.value


# ── A4: complete batches rank first; partial ones carry no % ────────

def _row(name, total, placed, n=2):
    circuits = [{'circuit_idx': j, 'norb': 8,
                 'chosen': {'alpha': [0, 1], 'beta': [2, 3],
                            'ancillas': [4 + j], 'product_score': total,
                            'lqs_circuit': total},
                 'strata': {1: []}, 'per_stratum': {}} for j in range(placed)]
    sweep = {'circuits': circuits, 'total_product': total,
             'total_lqs_circuit': total, 'placed_qubits': [],
             'buffer_qubits_used': [],
             'failed_at': None if placed == n else placed, '_context': {},
             'strata': {0: [{'product_score': total}]}}
    return {'name': name, 'family': 'Heron', 'sweep': sweep,
            'best_lqs': total, 'best_k': 0, 'error': None}


def test_a_complete_batch_outranks_a_partial_one(monkeypatch):
    from test_floor_batch_ui import _build as _build_batch
    from iqmapper import qpu_scan as qs
    b, _state = _build_batch(monkeypatch)
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [])
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {'results': [
        _row('ibm_pittsburgh', 0.759, 1),             # scan_qpus put it first
        _row('ibm_fez', 0.48, 2),
        {'name': 'ibm_bad', 'family': 'Heron', 'sweep': None,
         'best_lqs': None, 'best_k': None, 'error': 'boom'}],
        'n_scanned': 2, 'n_failed': 1, 'n_skipped': 0})
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_scan_batch.click()

    dd = next(c for c in b.scan_batch_box.children
              if isinstance(c, Dropdown))
    labels = [label for label, _ in dd.options]
    assert labels == ['ibm_fez — batch 48.0%',
                      'ibm_pittsburgh — 1 of 2 circuits fit',
                      'ibm_bad — ✗ error']


# ── A5: compatible alternatives only; the Plans menu ────────────────

def _blocking_result():
    """norbs [2, 3] on an 8-wide ladder, buffer 1. C1 on the left leaves room
    for C2; C1 in the middle cuts the ladder into two 2-wide pieces."""
    backend = ladder(8)
    left = _entry([16], 0.60, lqs=0.60, rank=1)
    left.update(alpha=[0, 1], beta=[8, 9])
    middle = _entry([19], 0.59, lqs=0.59, rank=2)
    middle.update(alpha=[3, 4], beta=[11, 12])
    right = _entry([22], 0.58, lqs=0.58, rank=3)
    right.update(alpha=[6, 7], beta=[14, 15])
    c2 = _entry([19], 0.50, lqs=0.50, rank=1)
    c2.update(alpha=[3, 4, 5], beta=[11, 12, 13])
    circuits = [_circuit(0, left, {1: [left, middle, right]}),
                _circuit(1, c2, {1: [c2]})]
    ctx = {'backend': backend, 'norbs': [2, 3], 'buffer_qubits': 1,
           'exclude_qubits': None, 'min_bridges': None}
    return {'circuits': circuits, 'total_product': 0.30,
            'total_lqs_circuit': 0.30, 'placed_qubits': [],
            'buffer_qubits_used': [], 'failed_at': None,
            '_context': ctx}


def test_an_alternative_that_blocks_a_later_circuit_is_hidden(monkeypatch):
    b, _state = _build_multi(monkeypatch)
    _patch_engine(monkeypatch, _blocking_result())
    b.lucj_multi_norbs.value = '2, 3'
    b.btn_lucj_multi_sweep.click()

    c1, c2 = _dropdowns(b.lucj_multi_detail_box)
    assert [label.split(' · ')[1] for label, _ in c1.options] == ['#1', '#3']
    lines = [c.value for c in b.lucj_multi_detail_box.children
             if isinstance(c, HTML)]
    assert any('1 alternative hidden: the other circuits would not fit' in v
               for v in lines)
    assert len(c2.options) == 1                   # the last circuit: no filter


def test_selecting_a_plan_applies_every_circuit(monkeypatch):
    b, state = _build_multi(monkeypatch)
    result, _ = _result()
    p2_c0 = _entry([5], 0.55, lqs=0.70, rank=2)
    p2_c1 = _entry([7], 0.57, lqs=0.40, rank=2)
    plan_2 = {'circuits': [_circuit(0, p2_c0, {1: [p2_c0]}),
                           _circuit(1, p2_c1, {1: [p2_c1]})],
              'total_product': 0.55 * 0.57, 'total_lqs_circuit': 0.28,
              'placed_qubits': [], 'buffer_qubits_used': [],
              'boundary_couplers': [], 'failed_at': None}
    plan_1 = {k: result[k] for k in ('circuits', 'total_product',
                                     'total_lqs_circuit', 'placed_qubits',
                                     'buffer_qubits_used', 'failed_at')}
    result['plans'] = [plan_1, plan_2]
    _patch_engine(monkeypatch, result)
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    plans = _plans(b.lucj_multi_detail_box)
    assert [label for label, _ in plans.options] == [
        'Plan 1 · total 35.7% · C1 61.2% + C2 58.4%',
        'Plan 2 · total 28.0% · C1 70.0% + C2 40.0%']
    plans.value = 1
    assert state.layouts_data[0]['ancilla'] == ['5']
    assert state.layouts_data[1]['ancilla'] == ['7']
    assert _plans(b.lucj_multi_detail_box).value == 1        # stays selected
    c1, c2 = _dropdowns(b.lucj_multi_detail_box)
    assert 'LQS 70.0%' in c1.options[0][0] and 'LQS 40.0%' in c2.options[0][0]


# ── A6: joint by default, B=3 and M=3 ───────────────────────────────

def test_the_default_run_plans_jointly_with_light_params(monkeypatch):
    import iqmapper.joint_plan as jp
    b, _state = _build_multi(monkeypatch)
    b.chk_fast_multi.value = False                # the shipped default
    result, _ = _result()
    seen = {}
    monkeypatch.setattr(jp, 'plan_multi_circuits',
                        lambda norbs, backend, props, **k:
                        seen.update(k) or dict(result, planner='joint'))
    _patch_engine(monkeypatch, None)              # sequential must not run
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    assert (seen['beam_width'], seen['branch']) == (3, 3)


def test_fast_is_off_by_default_and_owns_its_layout(monkeypatch):
    from test_search_ui import _build
    b = _build(monkeypatch)
    assert b.chk_fast_multi.value is False
    assert b.chk_fast_multi.layout is not b.txt_auto.layout
    assert b.chk_fast_multi.layout is not b.joint_params_box.layout
    assert b.multi_advanced_acc.layout is not b.txt_auto.layout


# ── A7: the "(this QPU)" row needs a complete local result ──────────

def test_a_partial_local_result_gives_no_reference_row(monkeypatch):
    from test_batch_scan_active import ACTIVE, _selector, _texts, _wire
    from test_floor_batch_ui import _build as _build_batch
    import iqmapper.multi_sweep as ms
    b, state = _build_batch(monkeypatch)
    _wire(monkeypatch, b, state)
    real = ms.sweep_multi_circuits

    def _partial(norbs, backend, props, **k):
        res = real(norbs, backend, props, **k)
        if getattr(backend, 'name', ACTIVE) == ACTIVE:
            res = dict(res, circuits=res['circuits'][:1], failed_at=1)
        return res

    monkeypatch.setattr(ms, 'sweep_multi_circuits', _partial)
    b.btn_lucj_multi_sweep.click()
    b.btn_scan_batch.click()
    dd, _btn = _selector(b)
    assert not any(label.startswith(ACTIVE) for label, _ in dd.options)
    assert any('not included' in t for t in _texts(b))
