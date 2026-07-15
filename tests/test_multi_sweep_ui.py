"""Headless tests for the sweep-based Multi-Circuit UI (auto-best + override).

Offline: real MapperUIBuilder over a stubbed 8-qubit device; the engine
(multi_sweep.sweep_multi_circuits / replan_from) is monkeypatched with canned
dicts, and the background spawn is injected (sync or a manual queue).
"""
from types import SimpleNamespace

import numpy as np
from ipywidgets import Dropdown


def _fake_device_data(state, vendor, chip, date, prefetched_props=None):
    n = 8
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01 * (i + 1) for i in range(n)]),
            np.array([0.001 * (i + 1) for i in range(n - 1)]))


def _sync_spawn(fn, on_done=None, on_error=None):
    try:
        r = fn()
    except Exception as e:                       # pragma: no cover
        if on_error:
            on_error(e)
        return None
    if on_done:
        on_done(r)
    return None


def _entry(anc, prod, lqs=None, rank=1):
    e = {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': list(anc),
         'bridge_slots': tuple(range(0, 4 * len(anc), 4)),
         'product_score': prod, 'additive_score': 0.5,
         'rank_in_stratum': rank, 'source': 'vf2'}
    if lqs is not None:
        e['lqs_circuit'] = lqs
    return e


def _circuit(i, chosen, strata, truncated=False):
    return {'circuit_idx': i, 'norb': 8, 'chosen': chosen, 'strata': strata,
            'k_max_effective': 2, 'min_bridges': 1, 'vf2_truncated': truncated,
            'per_stratum': {1: {'n_direct': 0, 'n_derived': len(strata.get(1, [])),
                                'truncated': truncated, 'shuffled': False}},
            'mapping_cap': 2000, 'excluded_qubits': []}


def _result(failed_at=None):
    """2-circuit canned result; circuit 0 has a #2 alternative for override."""
    c0_chosen = _entry([4], 0.60, lqs=0.612, rank=1)
    c0_alt = _entry([5], 0.55, lqs=0.500, rank=2)
    c1_chosen = _entry([6], 0.58, lqs=0.584, rank=1)
    circuits = [_circuit(0, c0_chosen, {1: [c0_chosen, c0_alt]})]
    total_lqs = 0.612
    total_prod = 0.60
    if failed_at is None:
        circuits.append(_circuit(1, c1_chosen, {1: [c1_chosen]}))
        total_lqs = 0.612 * 0.584
        total_prod = 0.60 * 0.58
    return {'circuits': circuits, 'total_product': total_prod,
            'total_lqs_circuit': total_lqs, 'placed_qubits': [],
            'buffer_qubits_used': [], 'failed_at': failed_at,
            '_context': {'stub': True}}, c0_alt


def _build(monkeypatch, spawn=_sync_spawn):
    import qpu_mapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', spawn)
    monkeypatch.setattr("qpu_mapper.data.generate_device_data", _fake_device_data)
    from qpu_mapper.state import SessionState
    from qpu_mapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()                 # no .coords → Auto allowed
    state.active_backend_name = "stub"
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
    b.build()
    return b, state


def _patch_engine(monkeypatch, result):
    import qpu_mapper.multi_sweep as ms
    seen = {}
    monkeypatch.setattr(ms, 'sweep_multi_circuits',
                        lambda *a, **k: seen.update(sweep_kwargs=k, sweep_args=a) or result)
    return seen


def _dropdowns(box):
    return [c for c in box.children if isinstance(c, Dropdown)]


# ── happy path — both applied, summary exact, dropdowns collapsed ────────

def test_multi_sweep_happy_path(monkeypatch):
    b, state = _build(monkeypatch)
    result, _ = _result()
    _patch_engine(monkeypatch, result)

    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    # Both circuits applied to state (α/β/ancilla painted).
    assert state.layouts_data[0]['alpha'] == ['0', '1']
    assert state.layouts_data[0]['ancilla'] == ['4']
    assert state.layouts_data[1]['ancilla'] == ['6']
    # Summary format exact.
    v = b.lbl_lucj_multi.value
    assert 'C1: k=1 · 61.2%' in v
    assert 'C2: k=1 · 58.4%' in v
    assert 'batch: 35.7%' in v                       # 0.612*0.584 = 0.3574
    # One collapsed dropdown per circuit, showing the chosen entry's label.
    dds = _dropdowns(b.lucj_multi_detail_box)
    assert len(dds) == 2
    assert dds[0].value == 0                          # chosen (#1) selected
    assert 'k=1 · #1 · LQS 61.2%' in dds[0].options[0][0]
    assert 'k=1 · #2 · LQS 50.0%' in dds[0].options[1][0]
    assert b.btn_lucj_multi_sweep.disabled is False   # re-armed


# ── override dropdown ranks best→worst by LQS, not by stratum order ─────────

def test_override_dropdown_sorted_best_first(monkeypatch):
    import re
    b, state = _build(monkeypatch)
    # k=2 stratum has a LOWER LQS than k=1 — stratified order (k desc) would put
    # the worse candidate first. The dropdown must instead read best→worst.
    e_k2 = _entry([4, 5], 0.40, lqs=0.40, rank=1)
    e_k1a = _entry([4], 0.66, lqs=0.65, rank=1)          # the chosen best
    e_k1b = _entry([5], 0.55, lqs=0.55, rank=2)
    c0 = _circuit(0, e_k1a, {2: [e_k2], 1: [e_k1a, e_k1b]})
    result = {'circuits': [c0], 'total_product': 0.65,
              'total_lqs_circuit': 0.65, 'placed_qubits': [],
              'buffer_qubits_used': [], 'failed_at': None, '_context': {}}
    _patch_engine(monkeypatch, result)

    b.lucj_multi_norbs.value = '8'
    b.btn_lucj_multi_sweep.click()

    dds = _dropdowns(b.lucj_multi_detail_box)
    assert len(dds) == 1
    labels = [o[0] for o in dds[0].options]
    pcts = [float(re.search(r'LQS ([\d.]+)%', l).group(1)) for l in labels]
    assert pcts == [65.0, 55.0, 40.0]                    # descending, best first
    assert dds[0].value == 0                             # chosen best preselected
    assert 'LQS 65.0%' in labels[0]


def test_override_template_entries_sink(monkeypatch):
    b, state = _build(monkeypatch)
    # A scored entry (lqs_circuit) must rank ABOVE a template one (no
    # lqs_circuit) even if the template's product_score is higher.
    scored = _entry([4], 0.30, lqs=0.30, rank=1)         # scored 30%
    templ = _entry([5], 0.90, rank=2)                    # template, no lqs
    c0 = _circuit(0, scored, {1: [templ, scored]})
    result = {'circuits': [c0], 'total_product': 0.30,
              'total_lqs_circuit': 0.30, 'placed_qubits': [],
              'buffer_qubits_used': [], 'failed_at': None, '_context': {}}
    _patch_engine(monkeypatch, result)

    b.lucj_multi_norbs.value = '8'
    b.btn_lucj_multi_sweep.click()

    labels = [o[0] for o in _dropdowns(b.lucj_multi_detail_box)[0].options]
    assert 'template' in labels[1].lower()               # template sinks below
    assert 'template' not in labels[0].lower()           # scored ranks first


# ── override — replan_from(result, 0, alt) re-applies + updates summary ──

def test_multi_sweep_override(monkeypatch):
    b, state = _build(monkeypatch)
    result, c0_alt = _result()
    _patch_engine(monkeypatch, result)
    import qpu_mapper.multi_sweep as ms
    calls = []
    # replan returns a new result where C0 = the alt and C1 re-swept.
    new_c0 = _circuit(0, c0_alt, {1: [c0_alt]})
    new_c1 = _circuit(1, _entry([7], 0.57, lqs=0.570), {1: [_entry([7], 0.57, lqs=0.570)]})
    new_result = {'circuits': [new_c0, new_c1], 'total_product': 0.55 * 0.57,
                  'total_lqs_circuit': 0.500 * 0.570, 'placed_qubits': [],
                  'buffer_qubits_used': [], 'failed_at': None,
                  '_context': {'stub': True}}
    monkeypatch.setattr(ms, 'replan_from',
                        lambda res, i, choice: calls.append((i, choice)) or new_result)

    b.btn_lucj_multi_sweep.click()
    dds = _dropdowns(b.lucj_multi_detail_box)
    dds[0].value = 1                                  # pick C0's #2 (the alt)

    assert len(calls) == 1
    assert calls[0][0] == 0 and calls[0][1] is c0_alt   # replan_from(result, 0, alt)
    # Re-applied: C0 now the alt (ancilla 5), C1 re-swept (ancilla 7).
    assert state.layouts_data[0]['ancilla'] == ['5']
    assert state.layouts_data[1]['ancilla'] == ['7']
    assert 'C2: k=1 · 57.0%' in b.lbl_lucj_multi.value  # summary refreshed


# ── failed_at — banner + partial application ────────────────────────────

def test_multi_sweep_failed_at(monkeypatch):
    b, state = _build(monkeypatch)
    result, _ = _result(failed_at=1)                 # circuit 2 doesn't fit
    _patch_engine(monkeypatch, result)

    b.btn_lucj_multi_sweep.click()
    v = b.lbl_lucj_multi.value
    assert 'circuit 2 does not fit' in v
    assert state.layouts_data[0]['ancilla'] == ['4']  # C1 still applied
    assert len(_dropdowns(b.lucj_multi_detail_box)) == 1   # only the placed one


# ── in-flight guard — a second override during replan → single call ─────

def test_multi_sweep_override_in_flight_guard(monkeypatch):
    queued = []

    def qspawn(fn, on_done=None, on_error=None):
        queued.append((fn, on_done, on_error))

    b, state = _build(monkeypatch, spawn=qspawn)
    result, _ = _result()
    _patch_engine(monkeypatch, result)
    import qpu_mapper.multi_sweep as ms
    monkeypatch.setattr(ms, 'replan_from', lambda res, i, choice: result)

    b.btn_lucj_multi_sweep.click()
    # Drain the initial sweep so the dropdowns exist.
    fn, done, _ = queued.pop(0)
    done(fn())
    dds = _dropdowns(b.lucj_multi_detail_box)

    dds[0].value = 1                                  # override → queues a replan
    assert len(queued) == 1
    dds[0].value = 0                                  # second, while in flight
    assert len(queued) == 1                           # guarded — no second spawn


# ── kwargs parity — multi-sweep gets the same knobs as the local sweep ──

def test_multi_sweep_kwargs_parity(monkeypatch):
    b, state = _build(monkeypatch)
    result, _ = _result()
    seen = _patch_engine(monkeypatch, result)
    local = {}
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda norb, bk, pr, **k: local.update(k) or result['circuits'][0]
                        and {'strata': {}, 'k_max_effective': 1, 'min_bridges': 1,
                             'vf2_truncated': False, 'n_raw_embeddings': 0,
                             'n_candidates_scored': 0, 'excluded_qubits': [],
                             'per_stratum': {}, 'mapping_cap': None})

    b.lucj_top_n.value = 7
    b.lucj_search_cap.value = 5000
    b.chk_deep_search.value = False
    b.chk_lucj_diverse.value = True

    b.lucj_ancilla_dd.value = -1                      # sweep mode
    b.btn_lucj_run.click()                            # local sweep
    b.btn_lucj_multi_sweep.click()                    # multi sweep

    mk = seen['sweep_kwargs']
    for key in ('top_n', 'hybrid', 'mapping_cap', 'diversity_threshold'):
        assert mk[key] == local[key], key
    assert mk['top_n'] == 7 and mk['mapping_cap'] == 5000
    assert mk['hybrid'] is False and mk['diversity_threshold'] == 0.8


# ── legacy Generate untouched — still calls the old engine ──────────────

def test_legacy_multi_button_unchanged(monkeypatch):
    b, state = _build(monkeypatch)
    calls = []
    monkeypatch.setattr("qpu_mapper.auto_lucj.get_multi_circuit_layouts",
                        lambda **k: calls.append(k) or [
                            {'circuit_idx': 0, 'norb': 8, 'num_bridges': 2,
                             'all_candidates': [{'layout': [0, 1, 2, 3],
                                                 'alpha': [0, 1], 'beta': [2, 3],
                                                 'score': 0.03, 'rank': 1,
                                                 'layout_full': [0, 1, 2, 3, 4]}]}])
    ms_calls = []
    import qpu_mapper.multi_sweep as ms
    monkeypatch.setattr(ms, 'sweep_multi_circuits',
                        lambda *a, **k: ms_calls.append(1) or _result()[0])

    b.lucj_multi_norbs.value = '8'
    b.btn_lucj_multi.click()                          # LEGACY button

    assert len(calls) == 1                            # old engine called
    assert ms_calls == []                             # sweep engine NOT called
