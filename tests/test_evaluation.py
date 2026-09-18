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
