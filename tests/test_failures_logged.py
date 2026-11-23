"""Failures that do not stop the action still leave a log line.

A QPU-1 label hook that raises during a backend switch is logged at debug
with its traceback. A scan backend whose properties cannot be loaded for the
circuit-level scores is named in a warning, which the scan shows as a notice.
"""
import logging

from test_error_tab_backend_switch import _fresh_state, _switch_ctx
from test_scan_ui import _build, _labels, _StubProps, _sweep, _walk


def test_a_failing_qpu1_label_hook_is_logged_and_the_switch_completes(
        monkeypatch, caplog):
    from iqmapper.ui.backend_switch import setup_backend_switch
    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))
    monkeypatch.setattr("iqmapper.constants.resolve_coords",
                        lambda b: ([], 'Heron r1'))

    def _boom():
        raise RuntimeError('label boom')
    state = _fresh_state()
    ctx = _switch_ctx(state, load_device_graph=lambda force=False: None,
                      refresh_time_panel=None)
    ctx.refresh_qpu1_label = [_boom]
    with caplog.at_level(logging.DEBUG, logger='iqmapper.ui.backend_switch'):
        setup_backend_switch(ctx)({'new': 'ibm_new'})

    assert state.active_backend_name == 'ibm_new'
    failed = [r for r in caplog.records
              if r.getMessage() == 'QPU-1 toggle label refresh failed']
    assert failed and failed[0].exc_info[1].args == ('label boom',)


def test_a_scan_backend_without_properties_for_scoring_is_named(monkeypatch):
    b = _build(monkeypatch, marshal=lambda cb: cb())
    monkeypatch.setattr("iqmapper.qpu_scan.list_scan_targets", lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: _sweep(0.6, 0.5))
    calls = {'ibm_a': 0}

    def get_props(state):
        def gp(backend):
            if backend.name == 'ibm_a':
                calls['ibm_a'] += 1
                if calls['ibm_a'] > 1:          # the search got them; scoring not
                    raise RuntimeError('props gone')
            return _StubProps(0.6, 'live · 09:30')
        return gp
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props", get_props)

    b.btn_scan_qpus.click()

    assert 'ibm_a' in ' '.join(_labels(b))     # still listed
    notices = ' '.join(getattr(w, 'value', '') or ''
                       for w in _walk(b.scan_results_box))
    assert 'circuit scores unavailable for ibm_a' in notices
    assert 'props gone' in notices
