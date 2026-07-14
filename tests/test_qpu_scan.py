"""Tests for qpu_mapper.qpu_scan — the multi-QPU scan engine.

Offline, deterministic, no qiskit, no network. Fake backends are
SimpleNamespaces exposing exactly what family/size detection reads
(``name`` / ``num_qubits`` / ``processor_type``); the ancilla sweep is
monkeypatched with canned dicts (real VF2 has its own tests).
"""
from types import SimpleNamespace

import pytest

from qpu_mapper.qpu_scan import list_scan_targets, scan_qpus


def _mk_backend(name, family, n_qubits):
    proc = {'family': family, 'revision': '1'} if family else None
    return SimpleNamespace(name=name, processor_type=proc, num_qubits=n_qubits)


def _canned_sweep(strata, **meta):
    base = {
        'strata': strata,
        'k_max_effective': max(strata) if strata else 0,
        'min_bridges': min(strata) if strata else 0,
        'vf2_truncated': False,
        'n_raw_embeddings': 1,
        'n_candidates_scored': 1,
    }
    base.update(meta)
    return base


def _entry(prod, k):
    """One sweep candidate with `k` ancillas and the given product_score."""
    return {
        'layout': [0, 1], 'alpha': [0], 'beta': [1],
        'ancillas': list(range(2, 2 + k)),
        'bridge_slots': tuple(range(0, 4 * k, 4)),
        'product_score': prod, 'additive_score': 1.0,
        'rank_in_stratum': 1, 'source': 'vf2',
    }


# ── list_scan_targets — flags and reasons ───────────────────────────────

def test_list_scan_targets_flags_and_reasons():
    service = SimpleNamespace(backends=lambda: [
        _mk_backend('ibm_a', 'Heron', 156),      # same family, OK
        _mk_backend('ibm_b', 'Heron', 156),      # same family, OK
        _mk_backend('ibm_c', 'Nighthawk', 120),  # other family
        _mk_backend('ibm_d', 'Heron', 6),        # same family, too small
        _mk_backend('ibm_e', None, 200),         # unknown family
    ])
    # norb=4 → floor = 2*4+1 = 9 qubits.
    targets = list_scan_targets(service, 4, same_family_as='Heron r3')
    by = {t['name']: t for t in targets}

    assert by['ibm_a']['eligible'] and by['ibm_a']['reason'] is None
    assert by['ibm_b']['eligible']
    assert by['ibm_a']['family'] == 'Heron'

    assert not by['ibm_c']['eligible']
    assert by['ibm_c']['reason'] == 'different family (Nighthawk)'

    assert not by['ibm_d']['eligible']
    assert by['ibm_d']['reason'] == 'needs >= 9 qubits, has 6'

    assert not by['ibm_e']['eligible']
    assert by['ibm_e']['reason'] == 'family unknown'

    # include_other_families flips the family-filtered one to eligible,
    # keeping its label; the too-small one stays ineligible (size gate).
    widened = {t['name']: t for t in list_scan_targets(
        service, 4, same_family_as='Heron r3', include_other_families=True)}
    assert widened['ibm_c']['eligible']
    assert widened['ibm_c']['family'] == 'Nighthawk'
    assert widened['ibm_e']['eligible']          # unknown family now included
    assert not widened['ibm_d']['eligible']      # still too small


# ── scan happy path — ranking + progress order ──────────────────────────

def test_scan_happy_path_sorted_and_progress(monkeypatch):
    scores = {'ibm_a': 0.30, 'ibm_b': 0.50}

    def fake_sweep(norb, backend, props, **kw):
        return _canned_sweep({2: [_entry(scores[backend.name], 2)]})
    monkeypatch.setattr("qpu_mapper.ancilla_sweep.sweep_ancilla_layouts", fake_sweep)

    backends = {'ibm_a': _mk_backend('ibm_a', 'Heron', 156),
                'ibm_b': _mk_backend('ibm_b', 'Heron', 156)}
    targets = [{'name': 'ibm_a', 'family': 'Heron', 'eligible': True},
               {'name': 'ibm_b', 'family': 'Heron', 'eligible': True}]

    events = []
    out = scan_qpus(
        targets, 4,
        get_backend=lambda n: backends[n],
        get_props=lambda b: object(),
        progress_cb=lambda name, status, detail='': events.append((name, status)))

    # Sorted by best_lqs desc → ibm_b (0.50) before ibm_a (0.30).
    assert [r['name'] for r in out['results']] == ['ibm_b', 'ibm_a']
    assert out['n_scanned'] == 2 and out['n_failed'] == 0 and out['n_skipped'] == 0
    # progress: start→ok per backend, in INPUT order (not sorted).
    assert events == [
        ('ibm_a', 'start'), ('ibm_a', 'ok'),
        ('ibm_b', 'start'), ('ibm_b', 'ok'),
    ]


# ── isolation — one backend failing doesn't kill the scan ───────────────

def test_scan_isolation_on_props_failure(monkeypatch):
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, backend, props, **kw: _canned_sweep({2: [_entry(0.4, 2)]}))

    backends = {'ibm_a': _mk_backend('ibm_a', 'Heron', 156),
                'ibm_b': _mk_backend('ibm_b', 'Heron', 156)}

    def get_props(backend):
        if backend.name == 'ibm_a':
            raise RuntimeError('calibration fetch boom')
        return object()

    targets = [{'name': 'ibm_a', 'family': 'Heron', 'eligible': True},
               {'name': 'ibm_b', 'family': 'Heron', 'eligible': True}]

    out = scan_qpus(targets, 4, get_backend=lambda n: backends[n],
                    get_props=get_props)

    assert out['n_failed'] == 1 and out['n_scanned'] == 1
    by = {r['name']: r for r in out['results']}
    assert by['ibm_a']['error'] and 'boom' in by['ibm_a']['error']
    assert by['ibm_a']['sweep'] is None
    assert by['ibm_b']['error'] is None and by['ibm_b']['best_lqs'] == 0.4
    # Failed backend sorts to the end.
    assert out['results'][-1]['name'] == 'ibm_a'


# ── best_lqs / best_k extraction ────────────────────────────────────────

def test_best_lqs_extraction(monkeypatch):
    # Max product_score is the k=1 entry (0.90); k=3 top is 0.20.
    strata = {
        3: [_entry(0.20, 3)],
        2: [_entry(0.55, 2)],
        1: [_entry(0.90, 1)],
    }
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, backend, props, **kw: _canned_sweep(strata))

    out = scan_qpus(
        [{'name': 'ibm_a', 'family': 'Heron', 'eligible': True}], 4,
        get_backend=lambda n: _mk_backend(n, 'Heron', 156),
        get_props=lambda b: object())

    r = out['results'][0]
    assert r['best_lqs'] == 0.90
    assert r['best_k'] == 1


# ── empty eligible set ──────────────────────────────────────────────────

def test_scan_empty_targets():
    out = scan_qpus([], 4, get_backend=lambda n: None, get_props=lambda b: None)
    assert out['results'] == []
    assert out['n_scanned'] == 0 and out['n_failed'] == 0 and out['n_skipped'] == 0


def test_scan_skips_ineligible_targets(monkeypatch):
    """Ineligible targets are skipped (never swept), counted in n_skipped."""
    called = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda *a, **k: called.append(1) or _canned_sweep({2: [_entry(0.4, 2)]}))

    targets = [{'name': 'ibm_x', 'family': 'Nighthawk', 'eligible': False,
                'reason': 'different family (Nighthawk)'}]
    out = scan_qpus(targets, 4, get_backend=lambda n: None, get_props=lambda b: None)

    assert called == []                          # sweep never invoked
    assert out['n_skipped'] == 1 and out['n_scanned'] == 0
    assert out['results'][0]['sweep'] is None


# ── sweep_fn injection hook — the ONE loop, a pluggable sweep ────────

def test_sweep_fn_injected_per_target_with_isolation():
    """t-new-1: sweep_fn is called per eligible target with the forwarded
    kwargs, and the shared loop keeps per-backend isolation (one raise → error
    row, the others complete)."""
    calls = []

    def fake_sweep(norb, backend, props, *, min_bridges=None, top_n=3):
        calls.append((backend.name, min_bridges, top_n))
        if backend.name == 'bad':
            raise RuntimeError('boom')
        return _canned_sweep({1: [{'product_score': 0.5}]})

    backends = {n: SimpleNamespace(name=n) for n in ('good', 'bad', 'good2')}
    targets = [{'name': n, 'family': 'Heron', 'eligible': True}
               for n in ('good', 'bad', 'good2')]
    out = scan_qpus(targets, 6, get_backend=lambda n: backends[n],
                    get_props=lambda b: object(),
                    min_bridges=2, top_n=5, sweep_fn=fake_sweep)

    assert len(calls) == 3                                   # per eligible target
    assert all(mb == 2 and tn == 5 for _, mb, tn in calls)  # kwargs forwarded
    rows = {r['name']: r for r in out['results']}
    assert rows['bad']['error'] == 'boom' and rows['bad']['sweep'] is None
    assert rows['good']['sweep'] is not None                # isolation held
    assert rows['good2']['sweep'] is not None
    assert out['n_failed'] == 1 and out['n_scanned'] == 2


def test_sweep_fn_none_uses_sweep_ancilla_layouts(monkeypatch):
    """sweep_fn=None (default) is byte-compatible: the built-in sweep runs."""
    seen = []
    monkeypatch.setattr(
        "qpu_mapper.ancilla_sweep.sweep_ancilla_layouts",
        lambda norb, b, p, **k: seen.append(k) or _canned_sweep({1: [{'product_score': 0.4}]}))
    out = scan_qpus([{'name': 'a', 'family': 'H', 'eligible': True}], 6,
                    get_backend=lambda n: SimpleNamespace(name=n),
                    get_props=lambda b: object(), min_bridges=1, top_n=3)
    assert seen == [{'min_bridges': 1, 'top_n': 3}]          # exactly today's call
    assert out['results'][0]['best_lqs'] == 0.4


def test_filtered_and_unfiltered_same_row_schema():
    """t-new-3: injecting a stability-carrying sweep_fn does not change the
    result schema — the stability block rides inside `sweep`, the rows stay
    identical to the plain path."""
    def plain(norb, backend, props, *, min_bridges=None, top_n=3):
        return _canned_sweep({1: [{'product_score': 0.5}]})

    def filtered(norb, backend, props, *, min_bridges=None, top_n=3):
        s = _canned_sweep({1: [{'product_score': 0.5}]})
        s['stability'] = {'available': True, 'excluded_final': [1]}
        return s

    tgts = [{'name': 'a', 'family': 'Heron', 'eligible': True}]
    bk = lambda n: SimpleNamespace(name=n)
    out_u = scan_qpus(tgts, 6, bk, lambda b: object(), sweep_fn=plain)
    out_f = scan_qpus(tgts, 6, bk, lambda b: object(), sweep_fn=filtered)

    assert set(out_u) == set(out_f) == {
        'results', 'n_scanned', 'n_failed', 'n_skipped'}
    ku = set(out_u['results'][0]); kf = set(out_f['results'][0])
    assert ku == kf == {'name', 'family', 'sweep', 'best_lqs', 'best_k', 'error'}
