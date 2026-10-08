"""NOTICE CAPTURE FOR ALL SWEEP PATHS.

The scan already captured iqmapper WARNING+ into its loud '⚠ notices' bar via
a scoped handler. These tests pin the SAME capture for the LOCAL sweep and the
MULTI sweep + replan: warnings render into the respective notices widget,
deduplicate (message ×count), and NEVER leak to the root logger (propagate is
restored in finally — proven on the error path).
"""
import logging
from types import SimpleNamespace

import numpy as np

_QLOG = 'iqmapper'


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


def _entry(prod, k):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': list(range(4, 4 + k)),
            'bridge_slots': tuple(range(0, 4 * k, 4)),
            'product_score': prod, 'additive_score': 1.0, 'lqs_circuit': None,
            'rank_in_stratum': 1, 'source': 'vf2'}


def _sweep():
    return {'strata': {2: [_entry(0.50, 2)], 1: [_entry(0.60, 1)]},
            'k_max_effective': 2, 'min_bridges': 1, 'vf2_truncated': False,
            'n_raw_embeddings': 10, 'n_candidates_scored': 20, 'mapping_cap': 2000,
            'per_stratum': {2: {'n_direct': 1, 'n_derived': 0, 'truncated': False},
                            1: {'n_direct': 1, 'n_derived': 1, 'truncated': False}}}


def _multi_result():
    ch = _entry(0.60, 1)
    return {'circuits': [{'circuit_idx': 0, 'chosen': ch, 'strata': {1: [ch]},
                          'per_stratum': {1: {'truncated': False}}}],
            'total_lqs_circuit': None, 'total_product': 0.60, 'failed_at': None}


def _build(monkeypatch):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "ibm_here"
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
    b.chk_guided_search.value = False    # stubbed exhaustive engine only
    b.chk_fast_multi.value = True     # the stubbed engine is the sequential one
    return b


class _RootSpy(logging.Handler):
    """Records everything that reaches the ROOT logger — nothing captured
    during a sweep should show up here (propagate=False)."""
    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture_handlers():
    from iqmapper.ui.lucj_panel import _ScanNoticeCapture
    return [h for h in logging.getLogger(_QLOG).handlers
            if isinstance(h, _ScanNoticeCapture)]


# ── t1: local sweep warning → notices rendered, nothing hits the root ───────

def test_local_sweep_notice_captured_not_leaked(monkeypatch):
    b = _build(monkeypatch)
    spy = _RootSpy()
    root = logging.getLogger()
    root.addHandler(spy)
    lg = logging.getLogger(_QLOG)
    try:
        def warn_sweep(norb, backend, props, **kw):
            lg.warning("Requested 6 bridges does not fit; falling back")
            return _sweep()
        monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                            warn_sweep)

        b.lucj_ancilla_dd.value = -1             # Auto → stratified sweep path
        b.btn_lucj_run.click()

        notices = b.lbl_lucj_notices.value
        assert 'notices' in notices               # loud bar rendered
        assert 'Requested 6 bridges does not fit' in notices
        assert 'background:#fff3cd' in notices      # same amber Y-pass styling
        # Not leaked under the cell: the root logger never saw it.
        assert not any('Requested 6 bridges' in m for m in spy.messages)
        # Handler detached after the sweep (restoration).
        assert _capture_handlers() == []
    finally:
        root.removeHandler(spy)


# ── t2: duplicate messages deduplicate with ×count ──────────────────────────

def test_local_sweep_dedup_count(monkeypatch):
    b = _build(monkeypatch)
    lg = logging.getLogger(_QLOG)

    def warn_sweep(norb, backend, props, **kw):
        lg.warning("vf2 truncating to 2000")
        lg.warning("vf2 truncating to 2000")
        lg.warning("bridge fallback")
        return _sweep()
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts", warn_sweep)

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()

    notices = b.lbl_lucj_notices.value
    assert 'vf2 truncating to 2000 ×2' in notices    # deduped with count
    assert 'bridge fallback' in notices               # singleton, no ×
    assert 'bridge fallback ×' not in notices
    assert '>3</span>' in notices                     # badge = 3 total records


# ── t3 is the existing scan suite (unchanged) — see test_scan_notices.py ────


# ── t4: restoration on error — BOTH paths (try/finally proof) ───────────────

def test_local_sweep_restores_on_error(monkeypatch):
    b = _build(monkeypatch)
    lg = logging.getLogger(_QLOG)
    assert lg.propagate is True                      # baseline

    def boom_sweep(norb, backend, props, **kw):
        lg.warning("partial notice before the crash")
        raise RuntimeError("sweep exploded")
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts", boom_sweep)

    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()                            # handler catches → error banner

    assert '❌' in b.lbl_lucj_status.value            # error surfaced
    assert lg.propagate is True                       # restored despite raise
    assert _capture_handlers() == []                  # handler detached


def test_multi_sweep_restores_on_error(monkeypatch):
    b = _build(monkeypatch)
    lg = logging.getLogger(_QLOG)
    assert lg.propagate is True

    def boom_multi(*a, **k):
        lg.warning("multi partial notice")
        raise RuntimeError("multi exploded")
    monkeypatch.setattr("iqmapper.multi_sweep.sweep_multi_circuits", boom_multi)

    b.lucj_multi_norbs.value = "8"
    b.btn_lucj_multi_sweep.click()

    assert '❌' in b.lbl_lucj_multi.value              # error surfaced
    assert lg.propagate is True                        # restored despite raise
    assert _capture_handlers() == []


# ── multi sweep happy path → notices rendered into the multi widget ─────────

def test_multi_sweep_notice_captured(monkeypatch):
    b = _build(monkeypatch)
    lg = logging.getLogger(_QLOG)

    def warn_multi(*a, **k):
        lg.warning("circuit 2 search truncated")
        lg.warning("circuit 2 search truncated")
        return _multi_result()
    monkeypatch.setattr("iqmapper.multi_sweep.sweep_multi_circuits", warn_multi)

    b.lucj_multi_norbs.value = "8"
    b.btn_lucj_multi_sweep.click()

    notices = b.lbl_lucj_multi_notices.value
    assert 'circuit 2 search truncated ×2' in notices
    assert 'background:#fff3cd' in notices
    assert _capture_handlers() == []
