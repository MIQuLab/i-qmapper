"""Headless tests for in-UI scan notices (cap note, capture, no-leak).

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
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
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
    import iqmapper.qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in names])


def _patch_engine(monkeypatch, sweep_fn):
    """Drive _scan_backends: per-target sweep + a stub props factory."""
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts", sweep_fn)
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: SimpleNamespace(
                            vintage='live', score_layout=lambda q: {'fidelity': 0.5})))


def _walk(node):
    yield node
    for ch in getattr(node, 'children', []):
        yield from _walk(ch)


def _all_html(box):
    # Recurse: the comparison table now lives inside a toggled VBox (behind the
    # '📊 Show comparison table' checkbox), so k-cap notes are nested.
    from ipywidgets import HTML
    return " ".join(w.value for w in _walk(box) if isinstance(w, HTML))


# ── k-capped row note from metadata ─────────────────────────────────────

def test_k_capped_row_note(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    _patch_engine(monkeypatch, lambda *a, **k: _sweep(k_max_effective=2))

    b.lucj_norb_slider.value = 12                 # request = ceil(12/4) = 3
    b.btn_scan_qpus.click()
    assert 'k capped at 2 (requested 3)' in _all_html(b.scan_results_box)


def test_k_not_capped_no_note(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    _patch_engine(monkeypatch, lambda *a, **k: _sweep(k_max_effective=3))

    b.lucj_norb_slider.value = 12                 # request = 3, k_eff = 3 → equal
    b.btn_scan_qpus.click()
    assert 'k capped' not in _all_html(b.scan_results_box)


# ── warning capture → grouped, deduplicated notices ─────────────────────

def test_notices_capture_and_dedup(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    import iqmapper.qpu_scan as qs
    lg = logging.getLogger('iqmapper')

    def sweep_warns(norb, backend, props, **kw):
        # Runs while _scan['current'] == 'ibm_a' (set by the 'start' event).
        lg.warning("stale cache housekeeping")
        lg.warning("stale cache housekeeping")
        lg.warning("other notice")
        return _sweep()
    _patch_engine(monkeypatch, sweep_warns)

    b.btn_scan_qpus.click()
    html = _all_html(b.scan_results_box)
    assert 'notices' in html and '>3</span>' in html   # loud header + count badge
    assert 'stale cache housekeeping ×2' in html   # deduped with count
    assert 'other notice' in html                  # the singleton (no ×)
    assert 'other notice ×' not in html
    assert 'ibm_a' in html                         # grouped under the backend


# ── finally-restoration (done AND error paths) ──────────────────────────

def test_logger_restored_after_scan(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    lg = logging.getLogger('iqmapper')
    h0, p0 = list(lg.handlers), lg.propagate

    _patch_engine(monkeypatch, lambda *a, **k: _sweep())
    b.btn_scan_qpus.click()                        # normal path
    assert list(lg.handlers) == h0
    assert lg.propagate == p0

    # A malformed sweep makes _work raise (best_lqs loop is outside the
    # per-target isolation) → the on_error/finally path must still restore.
    _patch_engine(monkeypatch, lambda *a, **k: {'strata': None})
    b.btn_scan_qpus.click()
    assert list(lg.handlers) == h0                 # still restored
    assert lg.propagate == p0


# ── empty notices → no section ──────────────────────────────────────────

def test_no_notices_section_when_empty(monkeypatch):
    b = _build(monkeypatch)
    _patch_targets(monkeypatch)
    _patch_engine(monkeypatch, lambda *a, **k: _sweep())
    b.btn_scan_qpus.click()
    assert 'notices' not in _all_html(b.scan_results_box)
