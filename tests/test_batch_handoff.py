"""Opening a scan pick on QPU 2: every circuit of a batch is drawn, the circuit
count follows the payload, and every outcome is reported next to the button
that was clicked.

Headless: a real MapperUIBuilder with the real inter-QPU wiring; network,
device data and coordinates are stubbed, as in test_inter_qpu_handoff.
"""
import logging
from types import SimpleNamespace

import numpy as np
from ipywidgets import Button, Dropdown, HBox, HTML

from test_floor_batch_ui import _ladder_backend, _sync_spawn

N = 48


def _build(monkeypatch):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr(au, 'marshal_to_main', lambda cb: cb())
    edges = list(_ladder_backend('x').coupling_map.graph.edge_list())

    def _fake_device(state, vendor, chip, date, prefetched_props=None):
        return (list(range(N)), [0.0] * N, edges, [str(i) for i in range(N)],
                np.full(N, 0.01), np.full(len(edges), 0.002))

    monkeypatch.setattr('iqmapper.data.generate_device_data', _fake_device)
    monkeypatch.setattr('iqmapper.connection._check_properties',
                        lambda b, n: (object(), None))
    monkeypatch.setattr('iqmapper.constants.resolve_coords',
                        lambda b: ([(i, 0) for i in range(N)], 'chip'))

    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    backends = {n: _ladder_backend(n) for n in ('ibm_a', 'ibm_b')}
    state = SessionState()
    for key, val in dict(
            active_backend=object(), active_backend_name='ibm_here',
            active_chip='chip',
            active_service=SimpleNamespace(backend=lambda n: backends[n]),
            available_backends=['ibm_a', 'ibm_b'], props_warning=None,
            initial_props=object(), active_ansatz='lucj', active_vendor='IBM',
            active_coords=[(i, 0) for i in range(N)], custom_colors=[],
            cal_history_dates=['_seed_']).items():
        setattr(state, key, val)
    b = MapperUIBuilder(state)
    b.build()

    import iqmapper.ui as ui_pkg
    real_build = ui_pkg.build_mapper_ui
    qpu2 = {}

    def _capture(s2, is_secondary=False, return_widgets=False):
        res = real_build(s2, is_secondary=is_secondary,
                         return_widgets=return_widgets)
        qpu2.update(state=s2, fig=res[1], widgets=res[2])
        return res

    monkeypatch.setattr(ui_pkg, 'build_mapper_ui', _capture)
    return b, backends, qpu2


def _entry(base):
    return {'alpha': [base, base + 1], 'beta': [base + 8, base + 9],
            'ancillas': [base + 16]}


def _qubits(entry):
    return set(entry['alpha']) | set(entry['beta']) | set(entry['ancillas'])


def _painted(qpu2):
    trace = next(t for t in qpu2['fig'].data
                 if 'markers' in (getattr(t, 'mode', '') or '')
                 and t.x is not None and len(t.x) == N)
    return {i for i, c in enumerate(trace.marker.color) if c != '#ffffff'}


def _drawn(qpu2):
    w = qpu2['widgets']
    return w.num_circuits.value if w.intra_qpu_mode.value else 1


def _selector(box):
    kids = list(box.children)
    row = next(c for c in kids if isinstance(c, HBox))
    button = next(c for c in row.children if isinstance(c, Button))
    status = next(c for c in row.children if isinstance(c, HTML))
    return next(c for c in kids if isinstance(c, Dropdown)), button, status


def _batch_scan(monkeypatch, b, backends, norbs='4, 4'):
    from iqmapper import qpu_scan as qs
    from iqmapper.props_shim import CachedPropsShim

    def _get_props(backend):
        edges = list(backend.coupling_map.graph.edge_list())
        return CachedPropsShim({'readout_error': np.full(N, 0.01)},
                               {'cz_error': np.full(len(edges), 0.002)},
                               edges, n_qubits=N, vintage='live')

    monkeypatch.setattr('iqmapper.props_shim.make_scan_get_props',
                        lambda state: _get_props)
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in backends])
    b.intra_qpu_mode.value = True
    b.lucj_multi_norbs.value = norbs
    b.btn_scan_batch.click()


# ── every circuit of a batch is drawn ───────────────────────────────

def test_a_scanned_batch_opens_with_all_its_circuits(monkeypatch):
    b, backends, qpu2 = _build(monkeypatch)
    _batch_scan(monkeypatch, b, backends)
    _dd, button, status = _selector(b.scan_batch_box)
    button.click()

    s2 = qpu2['state']
    assert _drawn(qpu2) == 2
    second = {int(q) for role in ('alpha', 'beta', 'ancilla')
              for q in s2.layouts_data[1][role]}
    assert second and second <= _painted(qpu2), 'C2 is not on the map'
    assert 'open on QPU 2 · 2 circuit(s)' in status.value


def test_a_batch_on_an_already_connected_backend_draws_all_circuits(
        monkeypatch):
    b, _backends, qpu2 = _build(monkeypatch)
    hook = b._inter_qpu_handoff[0]
    hook('ibm_a', _entry(0))                      # single open first
    before = _painted(qpu2)
    seen = []
    hook('ibm_a', [_entry(0), _entry(24)], report=lambda *m: seen.append(m))
    assert _drawn(qpu2) == 2
    assert _qubits(_entry(24)) <= _painted(qpu2)
    assert _painted(qpu2) != before
    assert seen[-1] == ('ok', 'applied to ibm_a on QPU 2 · 2 circuit(s)')


# ── the circuit count follows the payload, both ways ────────────────

def test_a_smaller_batch_drops_the_extra_circuits(monkeypatch):
    b, _backends, qpu2 = _build(monkeypatch)
    hook = b._inter_qpu_handoff[0]
    hook('ibm_a', [_entry(0), _entry(3), _entry(24)])
    assert _drawn(qpu2) == 3
    hook('ibm_a', [_entry(0), _entry(24)])
    assert _drawn(qpu2) == 2
    s2 = qpu2['state']
    assert all(not s2.layouts_data[2][role]
               for role in ('alpha', 'beta', 'ancilla'))
    assert not (_qubits(_entry(3)) - _qubits(_entry(0))) & _painted(qpu2)


def test_a_single_pick_after_a_batch_goes_back_to_one_circuit(monkeypatch):
    b, _backends, qpu2 = _build(monkeypatch)
    hook = b._inter_qpu_handoff[0]
    hook('ibm_a', [_entry(0), _entry(24)])
    assert _drawn(qpu2) == 2
    hook('ibm_a', _entry(5))
    assert _drawn(qpu2) == 1
    assert _painted(qpu2) == _qubits(_entry(5))
    assert not qpu2['state'].layouts_data[1]['alpha']


# ── outcomes are reported where the user clicked ───────────────────

def test_a_failing_hook_is_reported_next_to_the_button_and_logged(
        monkeypatch, caplog):
    b, backends, _qpu2 = _build(monkeypatch)
    _batch_scan(monkeypatch, b, backends)

    def _boom(name, payload, report=None):
        raise RuntimeError('panel gone')

    b._inter_qpu_handoff[0] = _boom
    _dd, button, status = _selector(b.scan_batch_box)
    with caplog.at_level(logging.WARNING, logger='iqmapper.ui.lucj_panel'):
        button.click()
    assert 'could not open ibm_a on QPU 2: panel gone' in status.value
    assert any('panel gone' in r.getMessage() for r in caplog.records)


def test_a_missing_hook_is_reported_next_to_the_button(monkeypatch):
    b, backends, _qpu2 = _build(monkeypatch)
    _batch_scan(monkeypatch, b, backends)
    b._inter_qpu_handoff[0] = None
    _dd, button, status = _selector(b.scan_batch_box)
    button.click()
    assert 'QPU 2 panel unavailable here' in status.value
    assert b.lbl_scan_progress.value == ''


def test_a_backend_missing_from_the_qpu2_list_is_logged_and_reported(
        monkeypatch, caplog):
    b, backends, qpu2 = _build(monkeypatch)
    backends['ibm_new'] = _ladder_backend('ibm_new')   # not in the dropdown
    seen = []
    with caplog.at_level(logging.WARNING, logger='iqmapper.ui.inter_qpu'):
        b._inter_qpu_handoff[0]('ibm_new', _entry(0),
                                report=lambda *m: seen.append(m))
    assert any(level == 'warn' and 'not in the QPU 2 list' in text
               for level, text in seen)
    assert any('ibm_new' in r.getMessage() for r in caplog.records)
    assert seen[-1][0] == 'ok'                  # the connect still went ahead
    assert _qubits(_entry(0)) <= _painted(qpu2)
