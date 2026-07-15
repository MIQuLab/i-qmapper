"""Headless tests for the multi-QPU scan UI.

Offline: the scan runs through the single handler-side path (_scan_backends),
so we monkeypatch the per-target sweep (ancilla_sweep.sweep_ancilla_layouts),
list_scan_targets, and the props factory (make_scan_get_props → a stub with
score_layout + vintage). The spawn and progress marshal are injected
synchronously so the handler runs deterministically.
"""
from types import SimpleNamespace

import numpy as np

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
            'n_raw_embeddings': 10, 'n_candidates_scored': 20,
            'per_stratum': {2: {'n_direct': 1, 'n_derived': 0, 'truncated': False},
                            1: {'n_direct': 1, 'n_derived': 1, 'truncated': False}}}


class _StubProps:
    """Per-backend props stand-in: score_layout returns a fixed fidelity;
    carries a vintage. score_layout raises when fid is None."""
    def __init__(self, fid=0.5, vintage='live'):
        self._fid = fid
        self.vintage = vintage

    def score_layout(self, qubits):
        if self._fid is None:
            raise RuntimeError("no circuit score")
        return {'fidelity': self._fid}


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
    state.active_backend_name = "stub"          # NOT one of _targets → no ref row
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
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


def _patch_scan(monkeypatch, sweeps=None, props=None, targets=None):
    """Drive _scan_backends: canned per-target sweep + stub props factory."""
    import qpu_mapper.qpu_scan as qs
    import qpu_mapper.ancilla_sweep as asw
    import qpu_mapper.props_shim as ps
    monkeypatch.setattr(qs, 'list_scan_targets',
                        lambda *a, **k: targets or _targets())
    sweeps = sweeps or {'ibm_b': _sweep(0.60, 0.50), 'ibm_a': _sweep(0.40, 0.30),
                        'ibm_c': RuntimeError('fetch failed')}

    def fake_sweep(norb, backend, propsobj, **kw):
        s = sweeps[backend.name]
        if isinstance(s, Exception):
            raise s
        return s
    monkeypatch.setattr(asw, 'sweep_ancilla_layouts', fake_sweep)
    props = props or {'ibm_b': _StubProps(0.60), 'ibm_a': _StubProps(0.40),
                      'ibm_c': _StubProps(0.0)}
    monkeypatch.setattr(ps, 'make_scan_get_props',
                        lambda state: (lambda backend: props[backend.name]))


def _dropdowns(children):
    from ipywidgets import Dropdown
    return [c for c in children if isinstance(c, Dropdown)]


def _walk(node):
    yield node
    for ch in getattr(node, 'children', []):
        yield from _walk(ch)


def _selector(b):
    from ipywidgets import Dropdown
    return next(w for w in b.scan_results_box.children if isinstance(w, Dropdown))


def _labels(b):
    """The scan selector's dropdown option labels (name — best%)."""
    return [lbl for lbl, _ in _selector(b).options]


def _detail(b, i):
    """Select row i in the selector and return its rendered detail HTML."""
    from ipywidgets import HTML
    dd = _selector(b)
    dd.value = i
    ch = list(b.scan_results_box.children)
    di = ch.index(dd)
    return next(w.value for w in ch[di + 1:] if isinstance(w, HTML))


def _detail_by_name(b, name):
    dd = _selector(b)
    idx = next(i for lbl, i in dd.options if name in lbl)
    return _detail(b, idx)


# ── happy path — table sorted, dropdowns, button re-enabled ─────────────────

def test_scan_happy_path(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch)
    b.btn_scan_qpus.click()

    children = b.scan_results_box.children
    # The per-QPU blocks are ONE selector dropdown, ranked (best first).
    assert len(_dropdowns(children)) == 1
    labels = _labels(b)
    assert labels[0].startswith('ibm_b') and 'ibm_a' in labels[1]   # sorted
    assert '60.0%' in labels[0] and '40.0%' in labels[1]            # best% in label
    assert b.btn_scan_qpus.disabled is False


# ── progress marshaling — no direct-from-thread widget writes ───────────────

def test_scan_progress_marshaled(monkeypatch):
    captured = []
    b = _build(monkeypatch, marshal=lambda cb: captured.append(cb))
    _patch_scan(monkeypatch)

    b.btn_scan_qpus.click()
    assert b.lbl_scan_progress.value == ""          # marshaled, not direct
    assert captured
    for cb in captured:
        cb()
    assert 'ibm_b' in b.lbl_scan_progress.value
    assert '✓' in b.lbl_scan_progress.value


# ── failed-backend row greyed with esc()'d error ────────────────────────────

def test_scan_failed_row_escaped(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch, sweeps={
        'ibm_b': _sweep(0.60, 0.50), 'ibm_a': _sweep(0.40, 0.30),
        'ibm_c': RuntimeError('boom <x>')})
    b.btn_scan_qpus.click()

    # The errored backend is a selectable row; its label + detail name the
    # esc()'d error (no raw angle brackets leak into HTML).
    assert any('ibm_c' in lbl and '✗ error' in lbl for lbl in _labels(b))
    detail = _detail_by_name(b, 'ibm_c')
    assert 'boom &lt;x&gt;' in detail
    assert '<x>' not in detail


# ── handoff — selecting a remote candidate stages it ────────────────────────

def test_scan_handoff_on_select(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch)
    staged = []
    b._inter_qpu_handoff[0] = lambda name, cand: staged.append((name, cand))

    b.btn_scan_qpus.click()
    # Select the top QPU in the selector, then click its "open on QPU 2" button.
    from ipywidgets import Button, HBox
    sel = _dropdowns(b.scan_results_box.children)[0]
    sel.value = 0                                   # ibm_b (top-ranked)
    btns = [c for w in b.scan_results_box.children
            for c in (w.children if isinstance(w, HBox) else [w])
            if isinstance(c, Button)]
    btns[0].click()

    assert len(staged) == 1
    name, cand = staged[0]
    assert name == 'ibm_b'
    assert cand['ancillas'] == [4, 5]              # the QPU's best (k=2) candidate


# ── sweep mode greys the score checkbox ────────────────────────────────

def test_m1_score_checkbox_toggle(monkeypatch):
    b = _build(monkeypatch)
    assert b.lucj_ancilla_dd.value == -1
    assert b.lucj_chk_score.disabled is True
    b.lucj_ancilla_dd.value = 2
    assert b.lucj_chk_score.disabled is False
    b.lucj_ancilla_dd.value = -1
    assert b.lucj_chk_score.disabled is True


# ── per-stratum status line format ─────────────────────────────────────

def test_m2_status_line_format():
    res = _sweep(0.40, 0.30)
    res['k_max_effective'] = 2
    res['min_bridges'] = 1
    res['n_raw_embeddings'] = 214
    line = _format_sweep_status(res)
    assert line == "Sweep: k=2..1 · 214 embeddings · best per k: 2→30.0% · 1→40.0%"


# ── circuit-LQS annotation (success → number, failure → template) ───────

def test_scan_circuit_lqs_annotated(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch)                        # stub props score → circuit LQS
    b.btn_scan_qpus.click()
    assert not any('(template)' in lbl for lbl in _labels(b))


def test_scan_template_suffix_only_on_failure(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    # Props whose score_layout raises → lqs_circuit stays None → template.
    _patch_scan(monkeypatch, props={'ibm_b': _StubProps(None), 'ibm_a': _StubProps(None),
                                    'ibm_c': _StubProps(None)})
    b.btn_scan_qpus.click()
    assert any('(template)' in lbl for lbl in _labels(b))


# ── vintage renders 'live · <fetch time>' (no stale badge) ───────────

def test_scan_live_vintage(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    _patch_scan(monkeypatch, props={'ibm_b': _StubProps(0.6, 'live · 09:30'),
                                    'ibm_a': _StubProps(0.4, 'live · 09:30'),
                                    'ibm_c': _StubProps(0.0, 'live · 09:30')})
    b.btn_scan_qpus.click()
    detail = _detail(b, 0)                      # calib vintage on detail line 2
    assert 'live · 09:30' in detail
    assert 'stale calib' not in detail          # stale-badge path removed


# ── a fetch failure → honest error row, others complete ──────────────

def test_scan_fetch_failure_error_row(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _sweep(0.6, 0.5))

    def get_props(state):
        def gp(backend):
            if backend.name == 'ibm_a':
                raise RuntimeError("network down fetching calibration")
            return _StubProps(0.6, 'live · 09:30')
        return gp
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props", get_props)

    b.btn_scan_qpus.click()
    assert 'network down' in _detail_by_name(b, 'ibm_a')   # honest error row
    assert 'live · 09:30' in _detail_by_name(b, 'ibm_b')   # ibm_b completed


# ── stability filter still reads the cache for HISTORY ────────────────

def test_stability_history_reads_cache(monkeypatch):
    from qpu_mapper.state import SessionState
    b = _build(monkeypatch, marshal=lambda cb: cb())
    # Enable the filter (needs local history available) + the scan sub-filter.
    b.state.cal_history_dates = ['2026-07-01']
    b.state.cal_history = {'2026-07-01': {'qubit_props': {'readout_error': [0.01] * 8},
                                          'gate_props': {}}}
    b._refresh_stability_filter_availability()
    b.chk_stability_filter.value = True
    b.chk_scan_stability.value = True

    from qpu_mapper import data
    loads = []
    monkeypatch.setattr(data, 'list_known_calibrations',
                        lambda name, date=None: [{'last_update': '2026-07-05 08:00:00'}])
    monkeypatch.setattr(data, '_load_calibration',
                        lambda name, ts: loads.append(name) or
                        ({'readout_error': [0.01] * 8}, {'cz_error': [0.001] * 7}))
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("qpu_mapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: {**_sweep(0.6, 0.5), 'stability': {'available': True}})
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: _StubProps(0.6, 'live · 09:30')))

    b.btn_scan_qpus.click()
    assert loads                                # cache loader used for HISTORY


# ── Regression: local sweep and scan must receive IDENTICAL sweep params ────
#    (any divergence silently skews cross-QPU rankings against the local one).

def test_local_and_scan_identical_sweep_kwargs(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    calls = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, bk, pr, **kw: calls.append(kw) or _sweep(0.6, 0.5))
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets",
                        lambda *a, **k: _targets())
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: _StubProps(0.6)))

    b.chk_deep_search.value = False                 # a non-default value
    b.lucj_min_ancilla.value = 2
    b.lucj_top_n.value = 7                           # non-default top_n
    b.lucj_search_cap.value = 5000                   # non-default cap
    b.chk_lucj_diverse.value = True                 # non-default diversity

    b.btn_lucj_run.click()                          # local sweep
    local_kw = calls[-1]
    calls.clear()
    b.btn_scan_qpus.click()                         # scan (per-target)
    scan_kw = calls[0]

    keys = ('min_bridges', 'top_n', 'hybrid', 'mapping_cap', 'diversity_threshold')
    assert {k: local_kw.get(k) for k in keys} == {k: scan_kw.get(k) for k in keys}
    assert scan_kw['hybrid'] is False               # honored, not the engine default
    # Show-top, Search-cap, diversity reach BOTH paths — parity.
    assert local_kw['top_n'] == 7 and scan_kw['top_n'] == 7
    assert local_kw['mapping_cap'] == 5000 and scan_kw['mapping_cap'] == 5000
    assert local_kw['diversity_threshold'] == 0.8 and scan_kw['diversity_threshold'] == 0.8


# ── active backend skipped-with-reference (never re-searched) ───────────

def test_active_backend_skipped_with_reference(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    swept = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, bk, pr, **kw: swept.append(bk.name) or _sweep(0.6, 0.5))
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'stub', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: _StubProps(0.6)))

    b.btn_scan_qpus.click()
    assert 'stub' not in swept                       # active never swept
    # (The removed comparison table used to render a greyed '📍 <active>'
    # reference row; that row is gone with the table. The skip behaviour — the
    # active backend is never re-swept — is what still matters and is pinned.)
    assert 'stub' not in [lbl.split(' —')[0] for lbl in _labels(b)]  # not a row


# ── scan honors the hybrid checkbox ─────────────────────────────────────

def test_scan_honors_hybrid_checkbox(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    calls = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, bk, pr, **kw: calls.append(kw) or _sweep(0.6, 0.5))
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets",
                        lambda *a, **k: _targets())
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: _StubProps(0.6)))

    b.chk_deep_search.value = True
    b.btn_scan_qpus.click()
    assert calls and all(kw['hybrid'] is True for kw in calls)


# ── SYM-2 t3/t4: None vs 0.0 distinguishable; summary = max over non-None ────

def _scan_html(b):
    from ipywidgets import HTML
    return " ".join(w.value for w in _walk(b.scan_results_box)
                    if isinstance(w, HTML))


def _patch_one_target(monkeypatch, sweep, props_obj):
    monkeypatch.setattr("qpu_mapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: sweep)
    monkeypatch.setattr("qpu_mapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: props_obj))


def test_annotation_failure_is_none_not_zero(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())

    class _RaiseProps:
        vintage = 'live'
        def score_layout(self, q):
            raise RuntimeError("edge cz absent from snapshot")
    _patch_one_target(monkeypatch, _sweep(0.60, 0.50), _RaiseProps())

    b.btn_scan_qpus.click()
    assert any('(template)' in lbl for lbl in _labels(b))   # None → template
    assert 'circuit score unavailable' in _scan_html(b)     # named in notices


def test_genuine_zero_renders_as_zero(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())

    class _ZeroProps:
        vintage = 'live'
        def score_layout(self, q):
            return {'fidelity': 0.0}                 # a real dead-edge zero
    _patch_one_target(monkeypatch, _sweep(0.60, 0.50), _ZeroProps())

    b.btn_scan_qpus.click()
    # Genuine zero shows as 0.0% in the per-k detail; not flagged (template).
    assert '0.0%' in _detail(b, 0)
    assert not any('(template)' in lbl for lbl in _labels(b))


def test_summary_best_is_max_over_non_none(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())

    class _VaryProps:
        vintage = 'live'
        def score_layout(self, q):
            # k=2 entry has 6 qubits → 0.0 (dead edge); k=1 has 5 → 0.62.
            return {'fidelity': 0.0 if len(q) >= 6 else 0.62}
    _patch_one_target(monkeypatch, _sweep(0.60, 0.50), _VaryProps())

    b.btn_scan_qpus.click()
    label = _labels(b)[0]
    assert '62.0%' in label                          # summary/label = max, NOT
    assert '— 0.0%' not in label                     # the k_max 0.0 wall
