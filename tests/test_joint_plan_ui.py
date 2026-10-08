"""The multi-circuit planner: Joint by default, Sequential with "Fast".

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

def _fast(b):
    b.chk_fast_multi.value = True
    return b


def test_planner_defaults_to_joint_with_light_params(monkeypatch):
    b, _ = _build(monkeypatch)
    b.chk_fast_multi.value = False                # undo the fixture's Fast
    assert b.chk_fast_multi.description == 'Fast: one circuit at a time'
    assert not hasattr(b, 'chk_joint_plan') and not hasattr(b, 'dd_planner')
    assert b.int_plans_kept.value == 3
    assert b.int_cands_per_circuit.value == 3
    assert b.ft_boundary_penalty.value == 0.0
    assert b.joint_params_box.layout.display == ''
    b.intra_qpu_mode.value = True
    b._build_auto_lucj_view()
    assert any(c is b.multi_advanced_acc
               for c in b._auto_multi_panel.children)
    inside = b.multi_advanced_acc.children[0].children
    assert inside[0] is b.chk_fast_multi and inside[1] is b.joint_params_box


def test_planner_widgets_own_their_layouts(monkeypatch):
    b, _ = _build(monkeypatch)
    widgets = (b.chk_fast_multi, b.joint_params_box, b.int_plans_kept,
               b.int_cands_per_circuit, b.ft_boundary_penalty)
    for w in widgets:
        assert w.layout is not b.txt_auto.layout
        assert w.layout is not b.lucj_buffer_slider.layout
    layouts = [w.layout for w in widgets]
    assert len({id(lay) for lay in layouts}) == len(layouts)


def test_fast_hides_the_joint_params(monkeypatch):
    b, _ = _build(monkeypatch)
    b.chk_fast_multi.value = False
    assert b.joint_params_box.layout.display == ''
    b.chk_fast_multi.value = True
    assert b.joint_params_box.layout.display == 'none'
    assert b.txt_auto.layout.display != 'none'
    b.chk_fast_multi.value = False
    assert b.joint_params_box.layout.display == ''


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
    b.chk_fast_multi.value = False
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
    assert '✅ 2 circuits placed · +19.1% vs one-by-one' in status, status
    assert 'shared couplers' not in status and 'searches' not in status


def test_joint_with_no_gain_says_only_how_many_were_placed(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    _patch_joint(monkeypatch, _joint(result, joint_total=0.30))
    b.chk_fast_multi.value = False
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    status = b.lbl_lucj_multi.value
    assert '✅ 2 circuits placed</div>' in status, status


def test_joint_that_places_more_circuits_says_so(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    _patch_joint(monkeypatch, _joint(result, greedy_placed=1,
                                     greedy_total=0.9))
    b.chk_fast_multi.value = False
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    status = b.lbl_lucj_multi.value
    assert '✅ 2 circuits placed · places 2 vs 1 one-by-one' in status
    assert '%' not in status.split('placed')[1]


def test_joint_names_shared_couplers(monkeypatch):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    _patch_joint(monkeypatch, _joint(result, boundary_couplers=[(47, 48)]))
    b.chk_fast_multi.value = False
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()
    assert 'shared couplers: 47–48' in b.lbl_lucj_multi.value


@pytest.mark.parametrize('guided', [True, False])
def test_joint_searches_with_the_combined_search(monkeypatch, guided):
    b, _ = _build(monkeypatch)
    result, _ = _result()
    seen = _patch_joint(monkeypatch, _joint(result))
    b.chk_fast_multi.value = False
    b.chk_guided_search.value = guided
    b.int_beam_width.value = 64
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_lucj_multi_sweep.click()

    calls = []
    monkeypatch.setattr('iqmapper.engine_select.combined_sweep',
                        lambda norb, backend, props, **k:
                        calls.append((norb, k)) or {'strata': {}})
    seen['kwargs']['search_fn'](8, exclude_qubits={3}, min_bridges=1)
    norb, k = calls[0]
    assert norb == 8 and k['use_guided'] is guided
    assert k['exclude_qubits'] == {3} and k['min_bridges'] == 1
    assert k['beam_width'] == 64 and 'mapping_cap' in k


def test_joint_pending_line_shows_the_worst_case_count(monkeypatch):
    queued = []
    b, _ = _build(monkeypatch, spawn=lambda fn, on_done=None, on_error=None:
                  queued.append(fn))
    b.chk_fast_multi.value = False
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
    b.chk_fast_multi.value = False
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


def test_batch_scan_places_one_by_one_even_when_fast_is_off(monkeypatch):
    """The batch scan keeps the sequential plan on every backend (real
    planner, per-backend scorer, through the helper)."""
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
    b.chk_fast_multi.value = False
    b.chk_guided_search.value = True             # both engines, real ones
    b.lucj_multi_norbs.value = '4, 4'
    b.btn_scan_batch.click()

    for row in captured['res']['results']:
        sweep = row['sweep']
        assert sweep.get('planner') != 'joint' and sweep['failed_at'] is None
        expect = 1.0
        for c in sweep['circuits']:
            ch = c['chosen']
            qubits = list(ch['alpha']) + list(ch['beta']) + list(ch['ancillas'])
            own = shims[row['name']].score_layout(qubits)['fidelity']
            assert ch['lqs_circuit'] == own
            expect *= own
        assert sweep['total_lqs_circuit'] == expect
    assert 'joint plan · up to' not in b.lbl_lucj_multi.value
