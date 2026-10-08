"""A backend the map cannot open (no coordinate layout for its chip) is listed
as "<name> — not supported" at the bottom of both multi-QPU scans: never
searched, never ranked, never openable.
"""
from types import SimpleNamespace

from ipywidgets import Button, HBox

from test_scan_ui import (_build, _detail_by_name, _labels, _patch_scan,
                          _sweep, _StubProps)

HERON_156 = {'family': 'Heron', 'revision': '2'}
HERON_R1_133 = {'family': 'Heron', 'revision': '1'}


def _backends():
    """ibm_a / ibm_b: 156-qubit Heron (supported); ibm_torino: 133-qubit
    Heron r1 (no map layout)."""
    return {
        'ibm_a': SimpleNamespace(name='ibm_a', num_qubits=156,
                                 processor_type=HERON_156),
        'ibm_b': SimpleNamespace(name='ibm_b', num_qubits=156,
                                 processor_type=HERON_156),
        'ibm_torino': SimpleNamespace(name='ibm_torino', num_qubits=133,
                                      processor_type=HERON_R1_133),
    }


def _targets():
    return [{'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
            for n in ('ibm_torino', 'ibm_a', 'ibm_b')]


def _open_button(box):
    return next(c for w in box.children if isinstance(w, HBox)
                for c in w.children if isinstance(c, Button))


# ── single-circuit scan ─────────────────────────────────────────────

def test_single_scan_lists_an_unsupported_chip_last_and_never_searches_it(
        monkeypatch):
    b = _build(monkeypatch)
    backends = _backends()
    b.state.active_service = SimpleNamespace(backend=lambda n: backends[n])
    searched = []
    sweeps = {'ibm_a': _sweep(0.40, 0.30), 'ibm_b': _sweep(0.60, 0.50),
              'ibm_torino': _sweep(0.99, 0.99)}
    _patch_scan(monkeypatch, sweeps=sweeps, targets=_targets(),
                props={'ibm_a': _StubProps(0.40), 'ibm_b': _StubProps(0.60),
                       'ibm_torino': _StubProps(0.99)})
    import iqmapper.ancilla_sweep as asw
    real = asw.sweep_ancilla_layouts
    monkeypatch.setattr(asw, 'sweep_ancilla_layouts',
                        lambda norb, backend, props, **k:
                        searched.append(backend.name) or real(
                            norb, backend, props, **k))
    b.btn_scan_qpus.click()

    labels = _labels(b)
    assert labels[-1] == 'ibm_torino — not supported'
    assert labels[0].startswith('ibm_b') and labels[1].startswith('ibm_a')
    assert 'ibm_torino' not in searched
    assert 'supported: 156-qubit Heron, 120-qubit Nighthawk' in \
        _detail_by_name(b, 'ibm_torino')
    button = _open_button(b.scan_results_box)
    assert button.disabled is True
    assert button.tooltip == 'not supported by this app'


# ── batch scan ──────────────────────────────────────────────────────

def test_batch_scan_lists_an_unsupported_chip_last_and_never_searches_it(
        monkeypatch):
    from test_batch_scan_active import _selector, _wire
    from test_floor_batch_ui import _build as _build_batch
    import iqmapper.qpu_scan as qs
    b, state = _build_batch(monkeypatch)
    log = _wire(monkeypatch, b, state)
    backends = _backends()
    state.active_service = SimpleNamespace(
        backend=lambda n: log['fetched'].append(n) or backends[n])
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in ('ibm_torino', 'ibm_a', 'ibm_b')])
    from test_batch_scan_active import TOTALS
    monkeypatch.setitem(TOTALS, 'ibm_torino', 0.99)
    b.btn_scan_batch.click()

    dd, button = _selector(b)
    labels = [label for label, _ in dd.options]
    assert labels[0].startswith('ibm_a') and labels[1].startswith('ibm_b')
    assert labels[-1] == 'ibm_torino — not supported'
    assert 'ibm_torino' not in log['searched']
    dd.value = len(labels) - 1
    assert button.disabled is True
    assert button.tooltip == 'not supported by this app'

