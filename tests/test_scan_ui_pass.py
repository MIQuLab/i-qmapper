"""Scan results: the selector label and the detail quote one metric (max
lqs_circuit, never max product_score), no comparison table, loud notices
header.
"""
import re
from types import SimpleNamespace

import numpy as np
from ipywidgets import HTML, Dropdown


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


def _entry(prod, k):
    return {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
            'ancillas': list(range(4, 4 + k)),
            'bridge_slots': tuple(range(0, 4 * k, 4)),
            'product_score': prod, 'additive_score': 1.0,
            'rank_in_stratum': 1, 'source': 'vf2'}


def _sweep(prod_hi, prod_lo):
    return {'strata': {2: [_entry(prod_lo, 2)], 1: [_entry(prod_hi, 1)]},
            'k_max_effective': 2, 'min_bridges': 1, 'vf2_truncated': False,
            'n_raw_embeddings': 10, 'n_candidates_scored': 20,
            'per_stratum': {2: {'n_direct': 1, 'n_derived': 0, 'truncated': False},
                            1: {'n_direct': 1, 'n_derived': 1, 'truncated': False}}}


class _FixedProps:
    """score_layout returns a FIXED fidelity (the circuit LQS), independent of
    the entry's product_score — so we can drive product_score != lqs_circuit and
    prove the label follows lqs_circuit, not the product heuristic."""
    def __init__(self, fid, vintage='live'):
        self._fid = fid
        self.vintage = vintage

    def score_layout(self, qubits):
        if self._fid is None:
            raise RuntimeError("no circuit score")
        return {'fidelity': self._fid}


def _build(monkeypatch, spawn=_sync_spawn, marshal=lambda cb: cb()):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', spawn)
    monkeypatch.setattr(au, 'marshal_to_main', marshal)
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"          # NOT a target → no ref row
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


def _patch_one(monkeypatch, sweep, props_obj, name='ibm_a'):
    monkeypatch.setattr("iqmapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': name, 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: sweep)
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: props_obj))


def _walk(node):
    yield node
    for ch in getattr(node, 'children', []):
        yield from _walk(ch)


def _has_table(b):
    """True if a comparison-table node (its 'best LQS' header) is anywhere in
    the scan results tree — it must NOT be, post-removal."""
    return any(isinstance(w, HTML) and 'best LQS' in w.value
               for w in _walk(b.scan_results_box))


def _selector_dd(b):
    return next(w for w in b.scan_results_box.children if isinstance(w, Dropdown))


def _detail(b, i):
    """Select row i and return its rendered detail HTML."""
    dd = _selector_dd(b)
    dd.value = i
    ch = list(b.scan_results_box.children)
    di = ch.index(dd)
    return next(w.value for w in ch[di + 1:] if isinstance(w, HTML))


def _pct(s):
    m = re.search(r'(\d+\.\d)%', s)
    return m.group(1) if m else None


# ── selector label quotes max lqs_circuit (== the detail metric) ────────────

def test_label_quotes_lqs_circuit(monkeypatch):
    b = _build(monkeypatch)
    # product_score peaks at 0.73 (would drive a product-based label to 73.0%),
    # but the circuit LQS is 0.653 → label + detail read 65.3%.
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(0.653))
    b.btn_scan_qpus.click()

    label = _selector_dd(b).options[0][0]         # top row's label
    assert _pct(label) == '65.3'                   # follows lqs_circuit, NOT 73.0
    assert '73.0%' not in label
    assert '65.3%' in _detail(b, 0)                # detail per-k agrees


def test_label_template_fallback(monkeypatch):
    b = _build(monkeypatch)
    # No circuit score (score_layout raises) → label falls back to product best
    # with the '(template)' flag.
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(None))
    b.btn_scan_qpus.click()

    label = _selector_dd(b).options[0][0]
    assert '(template)' in label
    assert _pct(label) == '73.0'


# ── detail line 2 carries cap + calib; omits absent segments ────────────────

def test_detail_line2_cap_and_calib(monkeypatch):
    b = _build(monkeypatch)
    b.lucj_norb_slider.value = 20                 # request k=5; sweep caps at 2
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(0.653, 'live · 21:11'))
    b.btn_scan_qpus.click()

    d = _detail(b, 0)
    assert 'k capped at 2 (requested 5)' in d      # cap segment rescued
    assert 'live · 21:11' in d                      # calib segment rescued
    assert '🛡' not in d                             # no stability filter → omitted


def test_detail_line2_omits_absent_cap(monkeypatch):
    b = _build(monkeypatch)
    b.lucj_norb_slider.value = 8                  # request k=2 == k_max → no cap
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(0.653, 'live · 09:00'))
    b.btn_scan_qpus.click()

    d = _detail(b, 0)
    assert 'capped' not in d                         # not capped → omitted
    assert 'live · 09:00' in d                        # calib still present


# ── NO comparison table + NO toggle checkbox in the results tree ────────────

def test_no_table_no_checkbox(monkeypatch):
    b = _build(monkeypatch)
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(0.653))
    b.btn_scan_qpus.click()

    from ipywidgets import Checkbox
    assert not _has_table(b)                          # table node gone
    # No "Show comparison table" checkbox anywhere in the results tree…
    assert not any(isinstance(w, Checkbox)
                   and 'comparison table' in (w.description or '')
                   for w in _walk(b.scan_results_box))
    # …and the widget/attribute is removed from the builder entirely.
    assert not hasattr(b, 'chk_show_scan_table')
    # The selector is the only results UI.
    assert any(isinstance(w, Dropdown) for w in b.scan_results_box.children)


# ── selector + open still drives handoff (no table involved) ────────────────

def test_selector_open_drives_handoff_no_table(monkeypatch):
    b = _build(monkeypatch)
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(0.653))
    staged = []
    b._inter_qpu_handoff[0] = (lambda name, cand, report=None:
                               staged.append((name, cand)))
    b.btn_scan_qpus.click()

    assert not _has_table(b)
    from ipywidgets import Button, HBox
    _selector_dd(b).value = 0
    btns = [c for w in b.scan_results_box.children
            for c in (w.children if isinstance(w, HBox) else [w])
            if isinstance(c, Button)]
    btns[0].click()

    assert len(staged) == 1
    name, cand = staged[0]
    assert name == 'ibm_a'
    # circuit LQS ties across strata → first-scored (k=2) entry wins.
    assert cand['ancillas'] == [4, 5]


# ── notices collapsed header carries the LOUD style + count ─────────────────

def test_notices_header_loud_with_count(monkeypatch):
    b = _build(monkeypatch)
    # score_layout raises → per-candidate 'circuit score unavailable' notices.
    _patch_one(monkeypatch, _sweep(0.73, 0.60), _FixedProps(None))
    b.btn_scan_qpus.click()

    notices = next(w.value for w in _walk(b.scan_results_box)
                   if isinstance(w, HTML) and '<details' in w.value
                   and 'notices' in w.value)
    assert 'background:#fff3cd' in notices             # amber warning bar
    assert 'font-weight:bold' in notices               # prominent header
    assert 'font-size:14px' in notices                 # larger than the 12px body
    assert 'border-radius:10px' in notices             # count badge
    assert re.search(r'>\s*\d+\s*</span>', notices)     # a numeric count in the badge
