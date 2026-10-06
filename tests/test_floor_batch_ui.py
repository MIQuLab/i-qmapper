"""Always-on quality floor (every path) + multi-circuit batch scan (UI).

Headless: real MapperUIBuilder over a 16-qubit device (the floor needs
> OUTLIER_MIN_SAMPLES qubits to judge); engines monkeypatched.
"""
from types import SimpleNamespace

import numpy as np
from ipywidgets import Button


def _fake_device_16(state, vendor, chip, date, prefetched_props=None):
    n = 16
    return (list(range(n)), [0.0] * n, [(i, i + 1) for i in range(n - 1)],
            [str(i) for i in range(n)],
            np.array([0.01] * n), np.array([0.001] * (n - 1)))


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


def _canned(**extra):
    e = {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
         'ancillas': [4], 'bridge_slots': (0,), 'product_score': 0.4,
         'additive_score': 0.5, 'lqs_circuit': 0.4, 'rank_in_stratum': 1,
         'source': 'derived'}
    res = {'strata': {1: [e]}, 'k_max_effective': 1, 'min_bridges': 1,
           'vf2_truncated': False, 'n_raw_embeddings': 5,
           'n_candidates_scored': 5, 'excluded_qubits': [],
           'per_stratum': {1: {'n_direct': 0, 'n_derived': 1,
                               'truncated': False, 'shuffled': False}},
           'mapping_cap': 2000}
    res.update(extra)
    return res


def _build(monkeypatch, spawn=_sync_spawn):
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', spawn)
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_16)
    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder
    state = SessionState()
    state.active_backend = object()
    state.active_backend_name = "stub"
    state.active_chip = "stub_chip"
    state.active_service = SimpleNamespace(backend=lambda n: SimpleNamespace(name=n))
    state.available_backends = ["ibm_a", "ibm_b"]
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = "lucj"
    state.active_vendor = "IBM"
    state.active_coords = []
    state.custom_colors = []
    state.cal_history_dates = ["_seed_"]
    b = MapperUIBuilder(state)
    b.build()
    # current readout with one outlier at qubit 5 (16 samples → floor can judge).
    ro = np.array([0.01] * 16)
    ro[5] = 0.5
    state.qubit_props_data = {'readout_error': ro}
    state.gate_props_data = {'cz_error': np.array([0.001] * 15)}
    return b, state


# ── unfiltered single sweep excludes the outlier by default ──────────

def test_floor_excludes_outlier_by_default(monkeypatch):
    b, state = _build(monkeypatch)
    cap = {}
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda norb, bk, pr, **k: cap.update(k) or _canned())
    b.lucj_ancilla_dd.value = -1                  # sweep mode
    b.btn_lucj_run.click()
    assert 5 in (cap.get('exclude_qubits') or set())      # floor removed q5
    assert 'floor excluded 1 outlier' in b.lbl_lucj_stability.value


# ── σ=0 → floor off ──────────────────────────────────────────────────

def test_floor_off_at_sigma_zero(monkeypatch):
    b, state = _build(monkeypatch)
    cap = {}
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda norb, bk, pr, **k: cap.update(k) or _canned())
    b.ft_stability_floor.value = 0                # off
    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()
    assert cap.get('exclude_qubits') is None
    assert 'floor excluded' not in b.lbl_lucj_stability.value


# ── filter ON + floor → floor via quality_sigma, not double-excluded ─

def test_floor_and_filter_no_double(monkeypatch):
    b, state = _build(monkeypatch, spawn=_sync_spawn)
    state.cal_history = {'2026-07-01': {'qubit_props': {'readout_error': [0.01] * 16},
                                        'gate_props': {}}}
    sws = {}
    monkeypatch.setattr("iqmapper.stability_filter.sweep_with_stability",
                        lambda *a, **k: sws.update(k) or _canned(
                            stability={'available': True, 'relaxed': False,
                                       'excluded_final': [5], 'window_days': 14,
                                       'n_floor_excluded': 1, 'floor_qubits': [5]}))
    plain = []
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda *a, **k: plain.append(k) or _canned())
    b.chk_stability_filter.disabled = False          # history available
    b.chk_stability_filter.value = True
    b.lucj_ancilla_dd.value = -1
    b.btn_lucj_run.click()
    # Filtered path used; the floor arrives through quality_sigma (not a second
    # exclude_qubits pass), so the plain sweep is not called.
    assert sws.get('quality_sigma') == 3.0
    assert plain == []


# ── scan floor is per-backend (outlier only in one) ──────────────────

def test_scan_floor_per_backend(monkeypatch):
    b, state = _build(monkeypatch)
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None},
        {'name': 'ibm_b', 'family': 'Heron', 'eligible': True, 'reason': None}])
    # ibm_a props have an outlier at q5; ibm_b is clean.
    ro_a = np.array([0.01] * 16); ro_a[5] = 0.5
    props = {'ibm_a': SimpleNamespace(_readout=ro_a, n_qubits=16, vintage='live',
                                      score_layout=lambda q: {'fidelity': 0.5}),
             'ibm_b': SimpleNamespace(_readout=np.array([0.01] * 16), n_qubits=16,
                                      vintage='live',
                                      score_layout=lambda q: {'fidelity': 0.5})}
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: props[backend.name]))
    seen = {}
    monkeypatch.setattr("iqmapper.ancilla_sweep.sweep_ancilla_layouts",
                        lambda norb, bk, pr, **k: seen.setdefault(bk.name, k) or _canned())
    b.btn_scan_qpus.click()
    assert 5 in (seen['ibm_a'].get('exclude_qubits') or set())
    assert not (seen['ibm_b'].get('exclude_qubits') or set())


# ── legacy single/multi buttons untouched by the floor ───────────────

def test_legacy_paths_untouched(monkeypatch):
    b, state = _build(monkeypatch)
    single = []
    monkeypatch.setattr("iqmapper.auto_lucj.get_all_zigzag_layouts",
                        lambda **k: single.append(k) or ([{'rank': 1, 'score': 0.1,
                                                           'alpha': [0, 1], 'beta': [2, 3]}], 1))
    multi = []
    monkeypatch.setattr("iqmapper.auto_lucj.get_multi_circuit_layouts",
                        lambda **k: multi.append(k) or [
                            {'circuit_idx': 0, 'norb': 8, 'num_bridges': 2,
                             'all_candidates': [{'layout': [0, 1, 2, 3],
                                                 'alpha': [0, 1], 'beta': [2, 3],
                                                 'score': 0.03, 'rank': 1,
                                                 'layout_full': [0, 1, 2, 3, 4]}]}])
    b.lucj_ancilla_dd.value = 2                    # legacy single (not sweep)
    b.btn_lucj_run.click()
    b.lucj_multi_norbs.value = '8'
    b.btn_lucj_multi.click()                       # legacy multi
    assert len(single) == 1                        # legacy engines called…
    assert len(multi) == 1
    # …and neither legacy engine accepts/forwards a floor exclude set.
    assert 'exclude_qubits' not in single[0]
    assert 'exclude_qubits' not in multi[0]


# ── batch scan ──────────────────────────────────────────────────────────

def _batch_result_row(name, total_lqs, ks, failed_at=None):
    circuits = [{'circuit_idx': j, 'norb': 8,
                 'chosen': {'alpha': [0, 1], 'beta': [2, 3], 'ancillas': [4 + j],
                            'product_score': 0.5, 'lqs_circuit': 0.5},
                 'strata': {1: []}, 'per_stratum': {}} for j in range(ks)]
    sweep = {'circuits': circuits, 'total_product': 0.5, 'total_lqs_circuit': total_lqs,
             'placed_qubits': [], 'buffer_qubits_used': [], 'failed_at': failed_at,
             '_context': {}, 'strata': {0: [{'product_score': total_lqs}]}}
    return {'name': name, 'family': 'Heron', 'sweep': sweep,
            'best_lqs': total_lqs, 'best_k': 0, 'error': None}


def test_batch_scan_rows_sorted_and_failed_labeled(monkeypatch):
    b, state = _build(monkeypatch)
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [])
    # scan_qpus returns a ready-ranked batch result (ibm_hi first).
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {'results': [
        _batch_result_row('ibm_hi', 0.61, 2),
        _batch_result_row('ibm_lo', 0.40, 2, failed_at=1)],
        'n_scanned': 2, 'n_failed': 0, 'n_skipped': 0})
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_scan_batch.click()

    # ONE selector dropdown (no table), labels sorted by batch LQS + one button.
    from ipywidgets import Dropdown, HTML, HBox
    kids = list(b.scan_batch_box.children)
    dd = next(c for c in kids if isinstance(c, Dropdown))
    det = next(c for c in kids if isinstance(c, HTML))
    btn = next(c for w in kids for c in
               (w.children if isinstance(w, HBox) else [w]) if isinstance(c, Button))

    labels = [o[0] for o in dd.options]
    assert not any('<table' in lab for lab in labels)
    assert labels[0].startswith('ibm_hi') and 'ibm_lo' in labels[1]  # sorted
    assert '61.0%' in labels[0] and '40.0%' in labels[1]
    # Selecting the failed batch: detail names the failure, open disabled + tip.
    dd.value = 1
    assert 'C2: does not fit' in det.value and '✗' not in det.value
    assert btn.disabled is True and 'does not fully fit' in btn.tooltip
    # A fitting batch is openable.
    dd.value = 0
    assert btn.disabled is False


def test_batch_scan_sweep_fn_gets_multi_knobs(monkeypatch):
    b, state = _build(monkeypatch)
    from iqmapper import qpu_scan as qs
    from iqmapper import multi_sweep as ms
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [
        {'name': 'ibm_a', 'family': 'Heron', 'eligible': True, 'reason': None}])
    monkeypatch.setattr("iqmapper.props_shim.make_scan_get_props",
                        lambda state: (lambda backend: SimpleNamespace(
                            _readout=np.array([0.01] * 16), n_qubits=16,
                            vintage='live', score_layout=lambda q: {'fidelity': 0.5})))
    seen = {}
    monkeypatch.setattr(ms, 'sweep_multi_circuits',
                        lambda *a, **k: seen.update(kwargs=k, norbs=a[0]) or {
                            'circuits': [], 'total_product': None,
                            'total_lqs_circuit': None, 'placed_qubits': [],
                            'buffer_qubits_used': [], 'failed_at': 0, '_context': {}})
    b.lucj_top_n.value = 7
    b.lucj_search_cap.value = 5000
    b.chk_deep_search.value = False
    b.chk_lucj_diverse.value = False                 # non-default (default is ON)
    b.lucj_multi_norbs.value = '8, 6'
    b.btn_scan_batch.click()

    assert seen['norbs'] == [8, 6]
    k = seen['kwargs']
    assert k['top_n'] == 7 and k['mapping_cap'] == 5000
    assert k['hybrid'] is False and k['diversity_threshold'] is None


# ── batch handoff fires from the selector's open button (right payload) ──────

def test_batch_selector_handoff_payload(monkeypatch):
    b, state = _build(monkeypatch)
    from iqmapper import qpu_scan as qs
    monkeypatch.setattr(qs, 'list_scan_targets', lambda *a, **k: [])
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: {'results': [
        _batch_result_row('ibm_hi', 0.61, 2)],
        'n_scanned': 1, 'n_failed': 0, 'n_skipped': 0})
    staged = []
    b._inter_qpu_handoff[0] = (lambda name, payload, report=None:
                               staged.append((name, payload)))
    b.lucj_multi_norbs.value = '8, 8'
    b.btn_scan_batch.click()

    from ipywidgets import Dropdown, HBox
    kids = list(b.scan_batch_box.children)
    btn = next(c for w in kids for c in
               (w.children if isinstance(w, HBox) else [w]) if isinstance(c, Button))
    btn.click()

    assert len(staged) == 1
    name, payload = staged[0]
    assert name == 'ibm_hi'
    assert isinstance(payload, list) and len(payload) == 2      # both circuits
    assert payload[0]['ancillas'] == [4] and payload[1]['ancillas'] == [5]


# ── batch scan scores each backend on ITS OWN calibration ───────────────

def _ladder_backend(name):
    """Two disjoint width-8 ladder blocks (48 qubits), as in test_multi_sweep."""
    import rustworkx
    edges = []
    for base in (0, 24):
        top = [base + j for j in range(8)]
        bot = [base + 8 + j for j in range(8)]
        brg = [base + 16 + j for j in range(8)]
        for j in range(7):
            edges += [(top[j], top[j + 1]), (bot[j], bot[j + 1])]
        for j in range(8):
            edges += [(top[j], brg[j]), (brg[j], bot[j])]
    graph = rustworkx.PyDiGraph()
    for i in range(48):
        graph.add_node(i)
    for u, v in edges:
        graph.add_edge(u, v, None)
    return SimpleNamespace(name=name, coupling_map=SimpleNamespace(graph=graph),
                           operation_names=['cz', 'sx', 'rz', 'x'])


def test_batch_scan_scores_each_backend_with_its_own_props(monkeypatch):
    """Same qubit indices, different gate errors: each backend's batch metric
    must come from that backend's props, not from the active backend's."""
    from iqmapper import qpu_scan as qs
    from iqmapper.props_shim import CachedPropsShim
    b, state = _build(monkeypatch)
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
    b.lucj_multi_norbs.value = '4, 4'
    captured = {}
    real_scan = qs.scan_qpus
    monkeypatch.setattr(qs, 'scan_qpus', lambda *a, **k: captured.setdefault(
        'res', real_scan(*a, **k)))
    b.btn_scan_batch.click()

    rows = {r['name']: r for r in captured['res']['results']}
    metrics = {}
    for name, row in rows.items():
        sweep = row['sweep']
        assert sweep['failed_at'] is None and len(sweep['circuits']) == 2
        expect = 1.0
        for c in sweep['circuits']:
            ch = c['chosen']
            qubits = list(ch['alpha']) + list(ch['beta']) + list(ch['ancillas'])
            own = shims[name].score_layout(qubits)['fidelity']
            assert ch['lqs_circuit'] == own
            expect *= own
        assert sweep['total_lqs_circuit'] == expect
        metrics[name] = sweep['total_lqs_circuit']
    assert metrics['ibm_a'] > metrics['ibm_b']
