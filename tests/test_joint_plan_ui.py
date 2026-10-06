"""The Planner selector: Sequential (default, unchanged) vs Joint.

Headless, offline. Planner engines are monkeypatched with canned results
except in the batch-scan test, which runs the real joint planner on two stub
backends to prove the per-backend scorer survives the dispatch.
"""
import numpy as np
import pytest

from test_multi_sweep_ui import _build, _dropdowns, _patch_engine, _result

SEQ_MULTI_KWARGS = {'min_bridges', 'top_n', 'buffer_qubits', 'exclude_qubits',
                    'mapping_cap', 'hybrid', 'diversity_threshold',
                    'circuit_score_fn'}


def _joint(result, **extra):
    res = dict(result, planner='joint', beam_width=4, branch=3,
               boundary_penalty=0.0, boundary_couplers=[], greedy_total=0.30,
               joint_total=0.3574, greedy_placed=len(result['circuits']),
               plan_source='beam', n_searches=7)
    res.update(extra)
    return res


def _patch_joint(monkeypatch, result):
    import iqmapper.joint_plan as jp
    seen = {}

    def _spy(norbs, backend, props, **kwargs):
        seen.update(norbs=norbs, kwargs=kwargs)
        return result

    monkeypatch.setattr(jp, 'plan_multi_circuits', _spy)
    return seen


# ── widgets ─────────────────────────────────────────────────────────

def test_planner_defaults_to_sequential_with_joint_params_hidden(monkeypatch):
    b, _ = _build(monkeypatch)
    assert b.dd_planner.value == 'sequential'
    assert [v for _l, v in b.dd_planner.options] == ['sequential', 'joint']
    assert b.int_plans_kept.value == 4
    assert b.int_cands_per_circuit.value == 3
    assert b.ft_boundary_penalty.value == 0.0
    assert b.joint_params_box.layout.display == 'none'
    b.intra_qpu_mode.value = True
    b._build_auto_lucj_view()
    for w in (b.dd_planner, b.joint_params_box):
        assert any(c is w for c in b._auto_multi_panel.children)


def test_planner_widgets_own_their_layouts(monkeypatch):
    b, _ = _build(monkeypatch)
    widgets = (b.dd_planner, b.joint_params_box, b.int_plans_kept,
               b.int_cands_per_circuit, b.ft_boundary_penalty)
    for w in widgets:
        assert w.layout is not b.txt_auto.layout
        assert w.layout is not b.lucj_buffer_slider.layout
    layouts = [w.layout for w in widgets]
    assert len({id(lay) for lay in layouts}) == len(layouts)


def test_joint_reveals_its_params(monkeypatch):
    b, _ = _build(monkeypatch)
    b.dd_planner.value = 'joint'
    assert b.joint_params_box.layout.display == ''
    assert b.txt_auto.layout.display != 'none'
    b.dd_planner.value = 'sequential'
    assert b.joint_params_box.layout.display == 'none'


# ── sequential is the call it always was ────────────────────────────

def test_sequential_makes_the_same_call_as_before(monkeypatch):
    b, state = _build(monkeypatch)
    result, _ = _result()
    seen = _patch_engine(monkeypatch, result)
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    assert set(seen['sweep_kwargs']) == SEQ_MULTI_KWARGS
    assert seen['sweep_args'][0] == [8, 8]
    assert seen['sweep_args'][1] is state.active_backend
    assert 'joint plan' not in b.lbl_lucj_multi.value


def test_sequential_replan_still_uses_multi_sweep(monkeypatch):
    import iqmapper.multi_sweep as ms
    b, _ = _build(monkeypatch)
    result, alt = _result()
    _patch_engine(monkeypatch, result)
    called = []
    monkeypatch.setattr(ms, 'replan_from',
                        lambda r, i, c: called.append((i, c)) or result)
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    _dropdowns(b.lucj_multi_detail_box)[0].value = 1
    assert called and called[0][1] is alt


# ── joint ───────────────────────────────────────────────────────────

def test_joint_dispatches_with_its_params_and_reports_the_gain(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    seen = _patch_joint(monkeypatch, _joint(result))
    _patch_engine(monkeypatch, None)            # must not be called
    b.dd_planner.value = 'joint'
    b.int_plans_kept.value = 6
    b.int_cands_per_circuit.value = 2
    b.ft_boundary_penalty.value = 0.25
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    kw = seen['kwargs']
    assert (kw['beam_width'], kw['branch'], kw['boundary_penalty']) == (
        6, 2, 0.25)
    assert callable(kw['search_fn']) and callable(kw['circuit_score_fn'])
    status = b.lbl_lucj_multi.value
    assert 'joint plan · +19.1% vs sequential · 7 searches' in status, status
    assert 'shared couplers' not in status
    assert 'C1: k=1 · 61.2%' in status           # the usual summary stays


def test_joint_that_places_more_circuits_says_so(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    _patch_joint(monkeypatch, _joint(result, greedy_placed=1,
                                     greedy_total=0.9))
    b.dd_planner.value = 'joint'
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    status = b.lbl_lucj_multi.value
    assert 'joint plan · places 2 vs 1 sequential · 7 searches' in status
    assert '%' not in status.split('joint plan')[1]


def test_joint_names_shared_couplers(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    _patch_joint(monkeypatch, _joint(result, boundary_couplers=[(47, 48)]))
    b.dd_planner.value = 'joint'
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    assert 'shared couplers: 47–48' in b.lbl_lucj_multi.value


def test_joint_search_fn_honours_the_selected_engine(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    seen = _patch_joint(monkeypatch, _joint(result))
    b.dd_planner.value = 'joint'
    b.dd_search_engine.value = 'hybrid'
    b.int_guided_max_k.value = 2
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    hybrid_calls = []
    monkeypatch.setattr('iqmapper.engine_select.hybrid_sweep',
                        lambda norb, backend, props, **k:
                        hybrid_calls.append((norb, k)) or {'strata': {}})
    seen['kwargs']['search_fn'](8, exclude_qubits={3}, min_bridges=1)
    norb, k = hybrid_calls[0]
    assert norb == 8 and k['guided_max_k'] == 2
    assert k['exclude_qubits'] == {3} and k['min_bridges'] == 1
    assert 'mapping_cap' in k['vf2_kwargs']


def test_joint_pending_line_shows_the_worst_case_count(monkeypatch):
    queued = []
    b, _ = _build(monkeypatch, spawn=lambda fn, on_done=None, on_error=None:
                  queued.append(fn))
    b.dd_planner.value = 'joint'
    b.int_plans_kept.value = 4
    b.lucj_multi_norbs.value = '8, 8, 8'
    b.btn_lucj_multi_sweep.click()
    assert queued, 'the run was not started'
    assert 'joint plan · up to 10 searches' in b.lbl_lucj_multi.value


def test_joint_replan_uses_the_joint_planner(monkeypatch):
    import iqmapper.joint_plan as jp
    import iqmapper.multi_sweep as ms
    b, _ = _build(monkeypatch)
    result, alt = _result()
    joint = _joint(result)
    _patch_joint(monkeypatch, joint)
    called = []
    monkeypatch.setattr(jp, 'replan_from',
                        lambda r, i, c: called.append((r, i, c)) or joint)
    monkeypatch.setattr(ms, 'replan_from', lambda *a: pytest.fail(
        'sequential replan used for a joint result'))
    b.dd_planner.value = 'joint'
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    _dropdowns(b.lucj_multi_detail_box)[0].value = 1
    assert called and called[0][0] is joint and called[0][2] is alt


# ── batch scan ──────────────────────────────────────────────────────

def test_batch_sequential_call_is_unchanged(monkeypatch):
    from types import SimpleNamespace
    from test_floor_batch_ui import _build as _build_batch
    from iqmapper import multi_sweep as ms
    from iqmapper import qpu_scan as qs
    b, _ = _build_batch(monkeypatch)
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: SimpleNamespace(
                            _readout=np.array([0.01] * 16), n_qubits=16,
                            vintage='live',
                            score_layout=lambda q: {'fidelity': 0.5})))
    seen = {}
    monkeypatch.setattr(ms, 'sweep_multi_circuits',
                        lambda *a, **k: seen.update(kwargs=k) or {
                            'circuits': [], 'total_product': None,
                            'total_lqs_circuit': None, 'placed_qubits': [],
                            'buffer_qubits_used': [], 'failed_at': 0,
                            '_context': {}})
    b.lucj_multi_norbs.value = '8, 6'
    b.btn_scan_batch.click()
    assert set(seen['kwargs']) == SEQ_MULTI_KWARGS


def test_batch_joint_scores_each_backend_with_its_own_props(monkeypatch):
    """Joint batch: real planner, per-backend scorer, through the helper."""
    from types import SimpleNamespace
    from test_floor_batch_ui import _build as _build_batch, _ladder_backend
    from iqmapper import qpu_scan as qs
    from iqmapper.props_shim import CachedPropsShim
    b, state = _build_batch(monkeypatch)
    backends = {n: _ladder_backend(n) for n in ('ibm_a', 'ibm_b')}
    gate_err = {'ibm_a': 0.002, 'ibm_b': 0.03}
    shims = {}

    def _get_props(backend):
        edges = list(backend.coupling_map.graph.edge_list())
        shim = CachedPropsShim(
            {'readout_error': np.array([0.01] * 48)},
            {'cz_error': np.array([gate_err[backend.name]] * len(edges))},
            edges, n_qubits=48, vintage='live')
        shims[backend.name] = shim
        return shim

    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: _get_props)
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': n, 'family': 'Heron', 'eligible': True, 'reason': None}
        for n in backends])
    state.active_service = SimpleNamespace(backend=lambda n: backends[n])
    captured = {}
    real_scan = qs.scan_qpus
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: captured.setdefault(
        'res', real_scan(*a, **k)))
    b.dd_planner.value = 'joint'
    b.lucj_multi_norbs.value = '4, 4'
    b.btn_scan_batch.click()

    for row in captured['res']['results']:
        sweep = row['sweep']
        assert sweep['planner'] == 'joint' and sweep['failed_at'] is None
        expect = 1.0
        for c in sweep['circuits']:
            ch = c['chosen']
            qubits = list(ch['alpha']) + list(ch['beta']) + list(ch['ancillas'])
            own = shims[row['name']].score_layout(qubits)['fidelity']
            assert ch['lqs_circuit'] == own
            expect *= own
        assert sweep['total_lqs_circuit'] == expect
    assert 'joint plan · up to' not in b.lbl_lucj_multi.value
