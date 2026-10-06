"""Tests for the offline engine-comparison harness.

Everything runs against a temporary cache directory holding synthetic
calibrations. No network, no real cache, no QPU.
"""
import json
import os

import numpy as np
import pytest

from iqmapper import evaluation as ev


# ── a synthetic cache ───────────────────────────────────────────────

def _ladder_edges(n_rungs):
    """Two rails plus rungs: wide enough for norb=4 with a bridge."""
    edges = []
    for i in range(n_rungs - 1):
        edges.append((i, i + 1))                       # rail A
        edges.append((n_rungs + i, n_rungs + i + 1))   # rail B
    for i in range(n_rungs):
        edges.append((i, 2 * n_rungs + i))             # A -> rung
        edges.append((2 * n_rungs + i, n_rungs + i))   # rung -> B
    return edges


def _write_cache(tmp_path, backend, stamps, n_rungs=8):
    """Write index + calibration files the way data._save_calibration does."""
    from iqmapper import data as _data

    old = _data._CACHE_DIR
    _data._CACHE_DIR = str(tmp_path)
    try:
        # _load_calibration deletes any snapshot missing a key from the
        # canonical option lists, so write the full set, not just the two the
        # engines read.
        from iqmapper.constants import GATE_PROP_OPTIONS, QUBIT_PROP_OPTIONS

        edges = _ladder_edges(n_rungs)
        n_q = 3 * n_rungs
        for i, stamp in enumerate(stamps):
            rng = np.random.default_rng(i)
            qubit_props = {pk: np.full(n_q, 0.001)
                           for _label, pk in QUBIT_PROP_OPTIONS}
            qubit_props['readout_error'] = 0.01 + 0.01 * rng.random(n_q)
            gate_props = {pk: np.full(len(edges), 0.001)
                          for _label, pk in GATE_PROP_OPTIONS}
            gate_props['cz_error'] = 0.005 + 0.01 * rng.random(len(edges))
            _data._save_calibration(backend, stamp, qubit_props, gate_props)
        return edges
    finally:
        _data._CACHE_DIR = old


def _write_session(session_dir, backend, edges, n_qubits):
    os.makedirs(session_dir, exist_ok=True)
    payload = {
        'active_backend_name': backend,
        'qubit_ids': [str(i) for i in range(n_qubits)],
        'edges_list': [list(e) for e in edges],
    }
    with open(os.path.join(session_dir, 'sess.json'), 'w',
              encoding='utf-8') as fh:
        json.dump(payload, fh)


# ── 8. the harness ──────────────────────────────────────────────────

def test_compare_engines_returns_rows_for_each_date_and_norb(tmp_path):
    backend = 'ibm_fixture'
    stamps = ['2026-09-01T00:00:00Z', '2026-09-02T00:00:00Z']
    edges = _write_cache(tmp_path, backend, stamps)

    report = ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path, coupling=edges,
                                beam_width=50)

    assert not report.errors, report.errors
    assert report.rows, 'no comparison rows produced'
    assert {r.date for r in report.rows} == set(stamps)
    assert {r.norb for r in report.rows} == {4}
    for row in report.rows:
        assert row.vf2_lqs is not None and row.guided_lqs is not None
        assert row.delta is not None
        assert row.vf2_time >= 0 and row.guided_time >= 0
        assert row.vf2_scaffolds > 0 and row.guided_scaffolds > 0


def test_report_renders_markdown_with_a_summary(tmp_path):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    report = ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path, coupling=edges,
                                beam_width=50)

    text = report.to_markdown()
    assert text.startswith('# Engine comparison - ibm_fixture')
    assert 'guided >= VF2 in' in text
    assert '| date | norb | k |' in text
    assert text.count('\n|') >= 2 + len(report.rows)


def test_missing_coupling_is_reported_not_guessed(tmp_path):
    """The cache has no topology; inventing one would produce fake numbers."""
    backend = 'ibm_fixture'
    _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])

    report = ev.compare_engines(backend, norbs=(4,), cache_dir=tmp_path,
                                session_dir=str(tmp_path / 'no_sessions'))
    assert report.rows == []
    assert report.errors and 'No coupling map' in report.errors[0]
    assert 'coupling=' in report.errors[0], 'the error must say how to fix it'


def test_coupling_is_recovered_from_a_saved_session(tmp_path):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    session_dir = tmp_path / 'sessions'
    _write_session(str(session_dir), backend, edges, 24)

    report = ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path,
                                session_dir=str(session_dir), beam_width=50)
    assert not report.errors, report.errors
    assert report.rows


def test_coupling_from_sessions_ignores_other_backends(tmp_path):
    session_dir = tmp_path / 'sessions'
    _write_session(str(session_dir), 'someone_else', [(0, 1)], 2)
    assert ev.coupling_from_sessions('ibm_fixture',
                                     str(session_dir)) is None
    assert ev.coupling_from_sessions('someone_else',
                                     str(session_dir)) == [(0, 1)]


def test_engine_failures_are_recorded_not_raised(tmp_path):
    """A norb too large for the device must be a row-less error, not a crash."""
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'],
                         n_rungs=4)
    report = ev.compare_engines(backend, norbs=(64,), cache_dir=tmp_path,
                                coupling=edges, min_bridges=1)
    assert report.rows == []
    assert report.errors
    assert any('norb=64' in e for e in report.errors)


def test_limit_caps_the_number_of_calibrations(tmp_path):
    backend = 'ibm_fixture'
    stamps = [f'2026-09-0{i}T00:00:00Z' for i in range(1, 5)]
    edges = _write_cache(tmp_path, backend, stamps)
    report = ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path, coupling=edges,
                                beam_width=50, limit=2)
    assert len({r.date for r in report.rows}) <= 2


def test_summary_handles_an_empty_report():
    report = ev.Report('nothing', [])
    assert report.summary() == 'No comparable cases.'
    assert 'nothing' in report.to_markdown()


def test_cached_props_reads_arrays_and_penalises_gaps():
    props = ev._CachedProps(
        {'readout_error': [0.01, 0.02]},
        {'cz_error': [0.03]},
        [(0, 1)])
    assert props.readout_error(0) == pytest.approx(0.01)
    assert props.gate_error('cz', (1, 0)) == pytest.approx(0.03), \
        'edge lookup must be direction-agnostic'
    assert np.isnan(props.readout_error(99))
    assert np.isnan(props.gate_error('cz', (0, 99)))


def test_cli_runs_and_writes_a_report(tmp_path, capsys, monkeypatch):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    session_dir = tmp_path / 'sessions'
    _write_session(str(session_dir), backend, edges, 24)
    out_dir = tmp_path / 'out'

    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    code = ev.main([backend, '--norb', '4', '--top-n', '2',
                    '--min-bridges', '1', '--beam-width', '50',
                    '--session-dir', str(session_dir),
                    '--out-dir', str(out_dir)])

    assert code == 0
    printed = capsys.readouterr().out
    assert '# Engine comparison' in printed
    written = list(out_dir.glob('eval_ibm_fixture_*.md'))
    assert written, 'the CLI did not write a report'
    assert '# Engine comparison' in written[0].read_text(encoding='utf-8')


# ── the real-LQS scorer ─────────────────────────────────────────────

def _entry(alpha, beta, ancillas):
    return {'alpha': list(alpha), 'beta': list(beta),
            'ancillas': list(ancillas), 'product_score': 0.5}


def _snapshot(n_rungs=8, seed=0, gate_scale=1.0):
    edges = _ladder_edges(n_rungs)
    n_q = 3 * n_rungs
    rng = np.random.default_rng(seed)
    qubit_props = {'readout_error': 0.01 + 0.01 * rng.random(n_q)}
    gate_props = {'cz_error': gate_scale * (0.005 + 0.01
                                            * rng.random(len(edges)))}
    return qubit_props, gate_props, edges, [str(i) for i in range(n_q)]


def test_lqs_scorer_matches_the_app_path():
    """Same value as estimate_circuit_fidelity on a real SessionState, and
    on the scratch state the panel's _circuit_score builds."""
    from types import SimpleNamespace
    from iqmapper.export.scoring import estimate_circuit_fidelity
    from iqmapper.state import SessionState

    qubit_props, gate_props, edges, ids = _snapshot()
    entry = _entry([0, 1, 2, 3], [8, 9, 10, 11], [16])
    ours = ev._lqs_scorer(qubit_props, gate_props, edges, ids)(entry)

    layout = {'alpha': ['0', '1', '2', '3'], 'beta': ['8', '9', '10', '11'],
              'other': [], 'ancilla': ['16']}
    fake = SimpleNamespace(layouts_data={0: layout},
                           qubit_props_data=qubit_props,
                           gate_props_data=gate_props, qubit_ids=ids,
                           edges_list=[tuple(e) for e in edges])
    panel = estimate_circuit_fidelity(fake, 0, 'lucj')['fidelity']

    state = SessionState()
    state.qubit_ids = list(ids)
    state.edges_list = [tuple(e) for e in edges]
    state.qubit_props_data = {k: np.asarray(v) for k, v in qubit_props.items()}
    state.gate_props_data = {k: np.asarray(v) for k, v in gate_props.items()}
    state.layouts_data = {0: dict(layout)}
    real = estimate_circuit_fidelity(state, 0, 'lucj')['fidelity']

    assert ours == pytest.approx(panel, rel=1e-12)
    assert ours == pytest.approx(real, rel=1e-12)
    assert 0.0 < ours < entry['product_score'] * 2


def test_lqs_scorer_is_rebuilt_for_every_snapshot(tmp_path, monkeypatch):
    backend = 'ibm_fixture'
    stamps = ['2026-09-01T00:00:00Z', '2026-09-02T00:00:00Z']
    edges = _write_cache(tmp_path, backend, stamps)
    built = []
    real = ev._lqs_scorer

    def _spy(qubit_props, gate_props, edges_, ids):
        fn = real(qubit_props, gate_props, edges_, ids)
        built.append(fn)
        return fn

    monkeypatch.setattr(ev, '_lqs_scorer', _spy)
    report = ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path, coupling=edges,
                                beam_width=50, scorer='lqs')
    assert not report.errors, report.errors
    assert len(built) == len(stamps)
    entry = _entry([0, 1, 2, 3], [8, 9, 10, 11], [16])
    assert built[0](entry) != built[1](entry)     # different gate errors
    assert report.metric == 'lqs'
    assert 'Metric: lqs\n' in report.to_markdown()


def test_lqs_scorer_takes_qubit_ids_from_the_session(tmp_path, monkeypatch):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    session_dir = tmp_path / 'sessions'
    _write_session(str(session_dir), backend, edges, 24)
    seen = []
    real = ev._lqs_scorer
    monkeypatch.setattr(ev, '_lqs_scorer', lambda q, g, e, ids: (
        seen.append((list(e), list(ids))) or real(q, g, e, ids)))
    ev.compare_engines(backend, norbs=(4,), top_n=2, min_bridges=1,
                       cache_dir=tmp_path, session_dir=str(session_dir),
                       beam_width=50, scorer='lqs')
    assert seen == [([tuple(e) for e in edges],
                     [str(i) for i in range(24)])]


def _numbers(report):
    return [(r.date, r.norb, r.k, r.vf2_lqs, r.guided_lqs, r.vf2_scaffolds,
             r.guided_scaffolds, r.vf2_capped, r.refined)
            for r in report.rows]


def test_proxy_and_an_explicit_scorer_reproduce_the_default(tmp_path):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    kw = dict(norbs=(4,), top_n=2, min_bridges=1, cache_dir=tmp_path,
              coupling=edges, beam_width=50)
    default = ev.compare_engines(backend, **kw)
    proxy = ev.compare_engines(backend, scorer='proxy', **kw)
    explicit = ev.compare_engines(backend, scorer='lqs',
                                  circuit_score_fn=ev._product_scorer(), **kw)
    assert _numbers(default) == _numbers(proxy) == _numbers(explicit)
    assert default.metric == proxy.metric == 'lqs_proxy'
    assert explicit.metric == 'custom'


def test_an_unknown_scorer_is_refused(tmp_path):
    with pytest.raises(ValueError, match='scorer'):
        ev.compare_engines('ibm_fixture', scorer='fidelity',
                           cache_dir=tmp_path, coupling=[(0, 1)])


# ── the threshold suggestion ────────────────────────────────────────

def _row(norb, k, vf2, guided, date='d1'):
    return ev.Row(date, norb, k, vf2, guided, 0.0, 0.0, 1, 1, False, 0)


def test_suggest_guided_max_k():
    rows = [
        # norb 8: guided holds at k=1 and 2, loses at 3 → t = 2
        _row(8, 1, 0.50, 0.52), _row(8, 1, 0.50, 0.49, 'd2'),
        _row(8, 1, 0.50, 0.51, 'd3'),
        _row(8, 2, 0.40, 0.40), _row(8, 3, 0.30, 0.25),
        # norb 12: loses at the floor → None
        _row(12, 1, 0.50, 0.45), _row(12, 2, 0.40, 0.45),
        # norb 16: 0.4 pt behind at k=1 — a tolerance of 0.5 pt lets it pass
        _row(16, 1, 0.500, 0.496), _row(16, 2, 0.4, None),
    ]
    report = ev.Report('ibm_x', rows)
    assert ev.suggest_guided_max_k(report) == {8: 2, 12: None, 16: None}
    assert ev.suggest_guided_max_k(report, tol=0.5) == {8: 2, 12: None,
                                                         16: 1}
    assert ev.suggest_guided_max_k(report, tol=6.0)[8] == 3
    text = ev.format_suggestion({8: 2, 12: None})
    assert '- norb 8: 2' in text
    assert '- norb 12: none (guided loses at the floor)' in text


# ── the hybrid column ───────────────────────────────────────────────

def test_hybrid_column_is_added_and_matches_its_halves(tmp_path):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    report = ev.compare_engines(backend, norbs=(5,), top_n=2, min_bridges=1,
                                cache_dir=tmp_path, coupling=edges,
                                beam_width=50, hybrid_ks=(1,))
    assert not report.errors, report.errors
    assert {r.k for r in report.rows} == {1, 2}
    for r in report.rows:
        expected = r.guided_lqs if r.k <= 1 else r.vf2_lqs
        assert r.hybrid_lqs == {1: expected}
    text = report.to_markdown()
    header = next(line for line in text.splitlines()
                  if line.startswith('| date'))
    assert header.endswith(' refined | hybrid k<=1 |')


def test_no_hybrid_leaves_the_table_unchanged():
    rows = [_row(8, 1, 0.5, 0.52)]
    report = ev.Report('ibm_x', rows)
    assert report.to_markdown() == ev.Report('ibm_x', rows,
                                             hybrid_ks=()).to_markdown()
    assert report.to_markdown() == (
        '# Engine comparison - ibm_x\n'
        '\n'
        'Metric: lqs_proxy\n'
        '\n'
        f'{report.summary()}\n'
        '\n'
        '| date | norb | k | VF2 | guided | dLQS (pt) | VF2 s | guided s |'
        ' VF2 scaffolds | guided scaffolds | VF2 capped | refined |\n'
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:--:|---:|\n'
        '| d1 | 8 | 1 | 0.5000 | 0.5200 | +2.00 | 0.00 | 0.00 | 1 | 1 |'
        ' no | 0 |\n')


def test_cli_prints_the_suggestion_and_takes_the_new_flags(
        tmp_path, capsys, monkeypatch):
    backend = 'ibm_fixture'
    edges = _write_cache(tmp_path, backend, ['2026-09-01T00:00:00Z'])
    session_dir = tmp_path / 'sessions'
    _write_session(str(session_dir), backend, edges, 24)
    monkeypatch.setattr('iqmapper.data._CACHE_DIR', str(tmp_path))
    code = ev.main([backend, '--norb', '5', '--top-n', '2',
                    '--min-bridges', '1', '--beam-width', '50',
                    '--session-dir', str(session_dir),
                    '--out-dir', str(tmp_path / 'out'),
                    '--hybrid-k', '1', '--tol', '0.5'])
    assert code == 0
    printed = capsys.readouterr().out
    assert 'Metric: lqs\n' in printed                 # CLI default
    assert 'hybrid k<=1' in printed
    assert 'Suggested "Guided up to k":' in printed
    assert '- norb 5:' in printed
