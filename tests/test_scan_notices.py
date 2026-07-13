"""Headless tests for in-UI scan notices (N1 cap note, N2 capture, N3 no-leak).

Offline: data layer stubbed, scan monkeypatched, spawn+marshal synchronous.
"""
import logging
from types import SimpleNamespace

import numpy as np


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _sweep(k_max_effective=1):
    entry = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
             'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.4,
             'additive_score': 0.5, 'rank_in_stratum': 1, 'source': 'vf2'}
    return {'strata': {1: [entry]}, 'k_max_effective': k_max_effective,
            'min_bridges': 1, 'vf2_truncated': False, 'n_raw_embeddings': 5,
            'n_candidates_scored': 5}


def _build(monkeypatch):
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)
    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: object())
    state.available_backends = ["ibm_a"]
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


def _patch_targets(monkeypatch, names=('ibm_a',)):
    import qpu_mapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in names])


def _all_html(box):
    from ipywidgets import HTML
    return " ".join(c.value for c in box.children if isinstance(c, HTML))


# ── t1: k-capped row note from metadata ─────────────────────────────────────

def test_k_capped_row_note(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import qpu_mapper.qpu_scan as qs

    def scan_capped(*a, **k):
        return {'results': [{'name': 'ibm_a', 'family': 'Heron',
                             'sweep': _sweep(k_max_effective=2),
                             'best_lqs': 0.5, 'best_k': 2, 'error': None}],
                'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0}
    monkeypatch.setattr(qs, 'scan_qpus', scan_capped)

    b.lucj_norb_slider.value = 12                 # request = ceil(12/4) = 3
    b.btn_scan_qpus.click()
    assert 'k capped at 2 (requested 3)' in _all_html(b.scan_results_box)


def test_k_not_capped_no_note(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import qpu_mapper.qpu_scan as qs

    def scan_ok(*a, **k):
        return {'results': [{'name': 'ibm_a', 'family': 'Heron',
                             'sweep': _sweep(k_max_effective=3),
                             'best_lqs': 0.5, 'best_k': 3, 'error': None}],
                'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0}
    monkeypatch.setattr(qs, 'scan_qpus', scan_ok)

    b.lucj_norb_slider.value = 12                 # request = 3, k_eff = 3 → equal
    b.btn_scan_qpus.click()
    assert 'k capped' not in _all_html(b.scan_results_box)


# ── t2: warning capture → grouped, deduplicated notices ─────────────────────

def test_notices_capture_and_dedup(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import qpu_mapper.qpu_scan as qs
    lg = logging.getLogger('qpu_mapper')

    def scan_warns(targets, norb, gb, gp, **kw):
        cb = kw['progress_cb']
        cb('ibm_a', 'start', '')
        lg.warning("stale cache housekeeping")
        lg.warning("stale cache housekeeping")
        lg.warning("other notice")
        cb('ibm_a', 'ok', '')
        return {'results': [{'name': 'ibm_a', 'family': 'Heron',
                             'sweep': _sweep(), 'best_lqs': 0.5, 'best_k': 1,
                             'error': None}],
                'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0}
    monkeypatch.setattr(qs, 'scan_qpus', scan_warns)

    b.btn_scan_qpus.click()
    html = _all_html(b.scan_results_box)
    assert '⚠ notices (3)' in html                # 3 total records
    assert 'stale cache housekeeping ×2' in html   # deduped with count
    assert 'other notice' in html                  # the singleton (no ×)
    assert 'other notice ×' not in html
    assert 'ibm_a' in html                         # grouped under the backend


# ── t3: finally-restoration (done AND error paths) ──────────────────────────

def test_logger_restored_after_scan(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import qpu_mapper.qpu_scan as qs
    lg = logging.getLogger('qpu_mapper')
    h0, p0 = list(lg.handlers), lg.propagate

    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {
        'results': [{'name': 'ibm_a', 'family': 'Heron', 'sweep': _sweep(),
                     'best_lqs': 0.5, 'best_k': 1, 'error': None}],
        'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0})
    b.btn_scan_qpus.click()
    assert list(lg.handlers) == h0
    assert lg.propagate == p0

    def boom(*a, **k):
        raise RuntimeError("scan blew up")
    monkeypatch.setattr(qs, 'scan_qpus', boom)
    b.btn_scan_qpus.click()                        # error path
    assert list(lg.handlers) == h0                 # still restored
    assert lg.propagate == p0


# ── t4: empty notices → no section ──────────────────────────────────────────

def test_no_notices_section_when_empty(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import qpu_mapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {
        'results': [{'name': 'ibm_a', 'family': 'Heron', 'sweep': _sweep(),
                     'best_lqs': 0.5, 'best_k': 1, 'error': None}],
        'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0})
    b.btn_scan_qpus.click()
    assert '⚠ notices' not in _all_html(b.scan_results_box)
