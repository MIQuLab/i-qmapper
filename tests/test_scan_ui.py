"""Headless tests for the multi-QPU scan UI (Component 3b) + M1/M2.

Offline: the data layer is stubbed, and qpu_scan.scan_qpus / list_scan_targets
are monkeypatched with canned results (no VF2, no network). The background
spawn and the progress marshal primitive are injected synchronously so the
handler runs deterministically.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from qpu_mapper.ui.lucj_panel import _format_sweep_status


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:            # pragma: no cover - defensive
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _sweep(prod_hi, prod_lo):
    def entry(prod, k):
        return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
                'ancillas': list(range(4, 4 + k)),
                'bridge_slots': tuple(range(0, 4 * k, 4)),
                'product_score': prod, 'additive_score': 1.0,
                'rank_in_stratum': 1, 'source': 'vf2'}
    return {'strata': {2: [entry(prod_lo, 2)], 1: [entry(prod_hi, 1)]},
            'k_max_effective': 2, 'min_bridges': 1, 'vf2_truncated': False,
            'n_raw_embeddings': 10, 'n_candidates_scored': 20}


def _fake_scan_result(error='fetch failed'):
    return {
        'results': [
            {'name': 'ibm_b', 'family': 'Heron', 'sweep': _sweep(0.60, 0.50),
             'best_lqs': 0.60, 'best_k': 1, 'error': None},
            {'name': 'ibm_a', 'family': 'Heron', 'sweep': _sweep(0.40, 0.30),
             'best_lqs': 0.40, 'best_k': 1, 'error': None},
            {'name': 'ibm_c', 'family': 'Nighthawk', 'sweep': None,
             'best_lqs': None, 'best_k': None, 'error': error},
        ],
        'n_scanned': 2, 'n_failed': 1, 'n_skipped': 0,
    }


def _targets():
    return [
        {'name': 'ibm_b', 'family': 'Heron', 'n_qubits': 156,
         'eligible': True, 'reason': None},
        {'name': 'ibm_a', 'family': 'Heron', 'n_qubits': 156,
         'eligible': True, 'reason': None},
        {'name': 'ibm_c', 'family': 'Nighthawk', 'n_qubits': 120,
         'eligible': True, 'reason': None},
    ]


def _build(monkeypatch, spawn=_sync_spawn, marshal=None):
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', spawn)
    if marshal is not None:
        monkeypatch.setattr(au, 'marshal_to_main', marshal)
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)

    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder

    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: object())
    state.available_backends = ["ibm_a", "ibm_b", "ibm_c"]
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]

    b = MapperUIBuilder(state)
    b.build()
    return b


def _patch_scan(monkeypatch, result=None, capture_progress=False):
    import qpu_mapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: _targets())
    res = result if result is not None else _fake_scan_result()

    def fake_scan(targets, norb, get_backend, get_props, **kw):
        cb = kw.get('progress_cb')
        if capture_progress and cb:
            for r in res['results']:
                cb(r['name'], 'start', '')
                cb(r['name'], 'ok' if r['error'] is None else 'error',
                   r['error'] or '')
        return res
    monkeypatch.setattr(qs, 'scan_qpus', fake_scan)


def _dropdowns(children):
    from ipywidgets import Dropdown
    return [c for c in children if isinstance(c, Dropdown)]


# ── t2: scan happy path — table sorted, dropdowns, button re-enabled ────────

def test_scan_happy_path(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch)
    b.btn_scan_qpus.click()

    children = b.scan_results_box.children
    table_html = children[0].value
    assert table_html.index('ibm_b') < table_html.index('ibm_a')  # sorted
    assert '60.0%' in table_html and '40.0%' in table_html
    # One stratified dropdown per successful QPU.
    assert len(_dropdowns(children)) == 2
    # Button re-enabled after completion.
    assert b.btn_scan_qpus.disabled is False


# ── t3: progress marshaling — no direct-from-thread widget writes ───────────

def test_scan_progress_marshaled(monkeypatch):
    captured = []
    b = _build(monkeypatch, marshal=lambda cb: captured.append(cb))
    _patch_scan(monkeypatch, capture_progress=True)

    b.btn_scan_qpus.click()
    # Progress callbacks were marshaled, NOT applied directly: the progress
    # label is still empty even though progress_cb fired during the scan.
    assert b.lbl_scan_progress.value == ""
    assert captured, "progress updates should have been marshaled"

    # Draining the marshaled callbacks updates the label on the main loop.
    for cb in captured:
        cb()
    assert 'ibm_b' in b.lbl_scan_progress.value
    assert '✓' in b.lbl_scan_progress.value


# ── t4: failed-backend row greyed with esc()'d error ────────────────────────

def test_scan_failed_row_escaped(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch, result=_fake_scan_result(error='boom <x>'))
    b.btn_scan_qpus.click()

    table_html = b.scan_results_box.children[0].value
    assert 'color:#999' in table_html          # greyed
    assert 'boom &lt;x&gt;' in table_html       # esc()'d error, not raw '<x>'
    assert '<x>' not in table_html


# ── t7: handoff — selecting a remote candidate stages it ────────────────────

def test_scan_handoff_on_select(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch)

    staged = []
    b._inter_qpu_handoff[0] = lambda name, cand: staged.append((name, cand))

    b.btn_scan_qpus.click()
    dds = _dropdowns(b.scan_results_box.children)
    dds[0].value = 0                           # pick ibm_b's top candidate

    assert len(staged) == 1
    name, cand = staged[0]
    assert name == 'ibm_b'
    # flat index 0 = top stratum (k descending) = the k=2 entry.
    assert cand['ancillas'] == [4, 5]


# ── t5: M1 — sweep mode greys out the score checkbox ────────────────────────

def test_m1_score_checkbox_toggle(monkeypatch):
    b = _build(monkeypatch)
    # Default: Ancilla='Auto' (-1) → sweep mode → score checkbox disabled.
    assert b.lucj_ancilla_dd.value == -1
    assert b.lucj_chk_score.disabled is True

    b.lucj_ancilla_dd.value = 2                 # exact count → legacy mode
    assert b.lucj_chk_score.disabled is False

    b.lucj_ancilla_dd.value = -1                # back to sweep
    assert b.lucj_chk_score.disabled is True


# ── t6: M2 — per-stratum status line format ─────────────────────────────────

def test_m2_status_line_format():
    res = _sweep(0.40, 0.30)                    # strata 2→30%, 1→40%
    res['k_max_effective'] = 2
    res['min_bridges'] = 1
    res['n_raw_embeddings'] = 214
    line = _format_sweep_status(res)
    assert line == "Sweep: k=2..1 · 214 embeddings · best per k: 2→30.0% · 1→40.0%"
