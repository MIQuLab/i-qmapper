"""Batch scan and the active backend: QPU 1 is never re-searched by the batch
scan. A local multi-circuit result for the same batch appears as a reference
row "<name> (this QPU)" at its rank, never openable; without one, a short line
says this QPU is not included.

Headless: a real MapperUIBuilder; the multi-circuit planner is stubbed, so
every search, local or remote, is recorded by backend.
"""
from types import SimpleNamespace

import numpy as np
from ipywidgets import Button, Dropdown, HBox, HTML

from test_floor_batch_ui import _build

ACTIVE = 'stub'                       # test_floor_batch_ui's active backend
TOTALS = {ACTIVE: 0.5, 'ibm_a': 0.6, 'ibm_b': 0.4}


def _result(backend, norbs, total):
    circuits = [{'circuit_idx': j, 'norb': n,
                 'chosen': {'alpha': [0, 1], 'beta': [2, 3],
                            'ancillas': [4 + j], 'product_score': total,
                            'lqs_circuit': total},
                 'strata': {1: []}, 'per_stratum': {}}
                for j, n in enumerate(norbs)]
    return {'circuits': circuits, 'total_product': total,
            'total_lqs_circuit': total, 'placed_qubits': [],
            'buffer_qubits_used': [], 'failed_at': None,
            '_context': {'backend': backend, 'norbs': list(norbs)}}


def _wire(monkeypatch, b, state):
    """Stub planner, targets, backends and props; return the search log."""
    from iqmapper import multi_sweep as ms
    from iqmapper import qpu_scan as qs
    log = {'searched': [], 'fetched': []}

    def _plan(norbs, backend, props, **_kw):
        name = getattr(backend, 'name', ACTIVE)
        log['searched'].append(name)
        return _result(backend, norbs, TOTALS[name])

    def _backend(name):
        log['fetched'].append(name)
        return SimpleNamespace(name=name)

    monkeypatch.setattr(ms, 'sweep_multi_circuits', _plan)
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in (ACTIVE, 'ibm_a', 'ibm_b')])
    monkeypatch.setattr('iqmapper.props_shim.make_scan_get_props',
                        lambda state: (lambda backend: SimpleNamespace(
                            _readout=np.full(16, 0.01), n_qubits=16,
                            vintage='live',
                            score_layout=lambda q: {'fidelity': 0.5})))
    state.active_service = SimpleNamespace(backend=_backend)
    b.lucj_multi_norbs.value = '4, 4'
    return log


def _selector(b):
    kids = list(b.scan_batch_box.children)
    dd = next(c for c in kids if isinstance(c, Dropdown))
    btn = next(c for w in kids if isinstance(w, HBox)
               for c in w.children if isinstance(c, Button))
    return dd, btn


def _texts(b):
    return [c.value for c in b.scan_batch_box.children if isinstance(c, HTML)]


# ── the active backend is never swept by the batch scan ─────────────

def test_the_batch_scan_never_searches_the_active_backend(monkeypatch):
    b, state = _build(monkeypatch)
    log = _wire(monkeypatch, b, state)
    b.btn_scan_batch.click()
    assert ACTIVE not in log['fetched']
    assert log['searched'] == ['ibm_a', 'ibm_b']


# ── reference row only with a local result ──────────────────────────

def test_without_a_local_result_there_is_no_row_and_one_short_line(
        monkeypatch):
    b, state = _build(monkeypatch)
    _wire(monkeypatch, b, state)
    b.btn_scan_batch.click()
    dd, _btn = _selector(b)
    labels = [label for label, _ in dd.options]
    assert labels[0].startswith('ibm_a') and labels[1].startswith('ibm_b')
    assert not any(label.startswith(ACTIVE) for label in labels)
    assert any('this QPU not included — run Generate Multi-Circuit to '
               'compare' in t for t in _texts(b))


def test_a_local_result_is_a_reference_row_at_its_rank(monkeypatch):
    b, state = _build(monkeypatch)
    log = _wire(monkeypatch, b, state)
    b.btn_lucj_multi_sweep.click()                # local result, batch 50%
    assert log['searched'] == [ACTIVE]
    b.btn_scan_batch.click()
    assert log['searched'] == [ACTIVE, 'ibm_a', 'ibm_b']   # no re-search

    dd, btn = _selector(b)
    labels = [label for label, _ in dd.options]
    assert labels[1] == f'{ACTIVE} (this QPU) — batch 50.0%'
    assert labels[0].startswith('ibm_a') and labels[2].startswith('ibm_b')
    assert not any('not included' in t for t in _texts(b))


def test_the_reference_row_cannot_be_opened(monkeypatch):
    b, state = _build(monkeypatch)
    _wire(monkeypatch, b, state)
    staged = []
    b._inter_qpu_handoff[0] = (lambda name, payload, report=None:
                               staged.append(name))
    b.btn_lucj_multi_sweep.click()
    b.btn_scan_batch.click()
    dd, btn = _selector(b)

    dd.value = 1                                  # the reference row
    assert btn.disabled is True and btn.tooltip == 'already QPU 1'
    btn.click()
    assert staged == []

    dd.value = 0                                  # a real remote batch
    assert btn.disabled is False
    btn.click()
    assert staged == ['ibm_a']


def test_a_local_result_for_another_batch_is_not_a_reference(monkeypatch):
    b, state = _build(monkeypatch)
    log = _wire(monkeypatch, b, state)
    b.lucj_multi_norbs.value = '4, 6'
    b.btn_lucj_multi_sweep.click()
    b.lucj_multi_norbs.value = '4, 4'
    b.btn_scan_batch.click()
    assert log['searched'] == [ACTIVE, 'ibm_a', 'ibm_b']
    dd, _btn = _selector(b)
    assert not any(label.startswith(ACTIVE) for label, _ in dd.options)
    assert any('not included' in t for t in _texts(b))
