"""The one-line scan-targets summary (eligible names + short skip reasons)."""
from types import SimpleNamespace

import numpy as np


def _fake_device(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _noop_spawn(fn, on_done=None, on_error=None):
    # Do NOT run the scan work — leave the rendered Targets line in place.
    return None


def _build(monkeypatch, active='ibm_boston'):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _noop_spawn)
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device)
    monkeypatch.setattr("iqmapper.qpu_scan._detect_family", lambda b: 'Heron')
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = active
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
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


def _targets_html(b):
    return b.scan_results_box.children[0].value


# ── t1: mixed fixture → exact compact line ──────────────────────────────────

def test_targets_line_compact_mixed(monkeypatch):
    b, state = _build(monkeypatch, active='ibm_boston')
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_pittsburgh', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_fez', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_marrakesh', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_kingston', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_boston', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_miami', 'family': 'Nighthawk', 'eligible': False,
         'reason': 'different family (Nighthawk)'}])
    b.btn_scan_qpus.click()

    html = _targets_html(b)
    # ibm_boston is the active backend → marked 'current' by the handler.
    assert ('<b>Targets:</b> ibm_pittsburgh · ibm_fez · ibm_marrakesh · '
            'ibm_kingston (4) — skipped: ibm_boston (current), '
            'ibm_miami (Nighthawk)') in html
    # Full reasons in the tooltip.
    assert "title='" in html
    assert 'ibm_miami: different family (Nighthawk)' in html
    assert 'ibm_boston: current backend' in html


def test_short_reasons_mapped(monkeypatch):
    b, state = _build(monkeypatch, active='none')
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_ok', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_small', 'family': 'Heron', 'eligible': False,
         'reason': 'needs >= 17 qubits, has 12'},
        {'name': 'ibm_unk', 'family': None, 'eligible': False,
         'reason': 'family unknown'},
        {'name': 'ibm_other', 'family': 'Flamingo', 'eligible': False,
         'reason': 'different family (Flamingo)'}])
    b.btn_scan_qpus.click()

    html = _targets_html(b)
    assert 'ibm_ok (1)' in html
    assert 'ibm_small (too small)' in html
    assert 'ibm_unk (unknown family)' in html
    assert 'ibm_other (Flamingo)' in html


# ── t2: no eligible targets → 'none' form ──────────────────────────────────

def test_targets_line_none(monkeypatch):
    b, state = _build(monkeypatch, active='none')
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Nighthawk', 'eligible': False,
         'reason': 'different family (Nighthawk)'},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': False,
         'reason': 'needs >= 17 qubits, has 12'}])
    b.btn_scan_qpus.click()

    html = _targets_html(b)
    assert '<b>Targets:</b> none — ibm_a (Nighthawk), ibm_b (too small)' in html
    assert 'skipped:' not in html                     # 'none' form omits the word


# ── t3: esc() on a hostile backend name ─────────────────────────────────────

def test_targets_line_escapes_name(monkeypatch):
    b, state = _build(monkeypatch, active='none')
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': '<script>x</script>', 'family': 'Heron', 'eligible': True,
         'reason': None}])
    b.btn_scan_qpus.click()

    html = _targets_html(b)
    assert '&lt;script&gt;' in html and '<script>x</script>' not in html
