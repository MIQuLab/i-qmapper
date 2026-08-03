"""
tests/test_mapper.py — Automated tests for I-QMapper.

Covers the highest-value targets:
1. Cache save/load round-trip
2. _safe_path rejects traversal
3. Session save/load round-trip
4. LUCJ dedup correctness
5. Scoring with mock backend
6. build_static_figure with mock state
7. Export metadata
8. _sanitize_ts consistency
9. _extract_last_update UTC normalization

Run: pytest tests/test_mapper.py -v
"""
import pytest
import numpy as np
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from iqmapper.paths import safe_path


# ═══════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════

@pytest.fixture
def tmp_cache_dir(tmp_path):
    """Temporary cache directory, cleaned up after test."""
    d = tmp_path / ".qpu_calibration_cache"
    d.mkdir()
    return d


@pytest.fixture
def fake_props():
    """Fake qubit/gate property arrays (like a 10-qubit device).

    Generates a value for EVERY key listed in QUBIT_PROP_OPTIONS /
    GATE_PROP_OPTIONS so that the round-trip through _save_calibration
    → _load_calibration doesn't trip the stale-cache rejection
    (iqmapper.data._load_calibration removes any cache file that
    doesn't carry every expected property key). The previous version
    only populated 3 qubit + 2 gate keys and `test_save_and_load`
    therefore broke as soon as the cache validator started enforcing
    the full key set.
    """
    from iqmapper.constants import QUBIT_PROP_OPTIONS, GATE_PROP_OPTIONS
    n_q, n_e = 10, 12
    rng = np.random.default_rng(42)
    # Reasonable per-key ranges; values don't matter for I/O round-trip,
    # only that they're floats. The (low, high) tuples are plausible
    # placeholders, not tied to any real backend.
    _qr = {
        'readout_error': (0.001, 0.05),
        't1': (50, 300),
        't2': (30, 200),
        'prob_meas0_prep1': (0.004, 0.2),
        'prob_meas1_prep0': (0.004, 0.2),
        'readout_length': (400, 1200),
        'id_error': (1e-4, 5e-3),
        'sq_gate_length': (20, 80),
        'rx_error': (1e-4, 5e-3),
        'sx_error': (1e-4, 5e-3),
        'rz_error': (1e-6, 1e-4),
        'x_error': (1e-4, 5e-3),
        'measure_error': (0.004, 0.3),
        'measure_2_error': (0.004, 0.3),
    }
    _gr = {
        'cz_error': (0.001, 0.05),
        'rzz_error': (6e-4, 0.15),
        'gate_length': (200, 600),
    }
    qp = {pk: rng.uniform(*_qr.get(pk, (1e-4, 5e-3)), n_q)
          for _, pk in QUBIT_PROP_OPTIONS}
    gp = {pk: rng.uniform(*_gr.get(pk, (6e-4, 0.15)), n_e)
          for _, pk in GATE_PROP_OPTIONS}
    return qp, gp, n_q, n_e


@pytest.fixture
def mock_state():
    """Minimal mock SessionState for testing."""
    state = MagicMock()
    state.active_backend_name = 'test_backend'
    state.active_chip = 'Heron r3'
    state.calibration_data_label = 'test — 2026-01-20 19:18:17'
    state.current_colorscale = 'Viridis'
    state.qubit_ids = [str(i) for i in range(10)]
    state.edges_list = [(i, i+1) for i in range(9)] + [(0, 5), (3, 8), (4, 9)]
    state.x_coord = np.arange(10, dtype=float)
    state.y_coord = np.zeros(10, dtype=float)
    state.readout_errors = np.random.uniform(0.001, 0.05, 10)
    state.gate_errors = np.random.uniform(0.001, 0.05, 12)
    state.qubit_props_data = {'readout_error': state.readout_errors}
    state.gate_props_data = {'cz_error': state.gate_errors}
    state.delta_mode = False
    state.delta_qubit_props = None
    state.delta_gate_props = None
    state.layouts_data = {
        0: {'alpha': ['0', '1'], 'beta': ['2', '3'], 'other': [],
            'qubits': [], 'ancilla': [], 'styles': {}, 'name': 'C1'}
    }
    state.custom_colors = []
    return state


# ═══════════════════════════════════════════════════════════════════════
# 1. CACHE ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════

class TestCacheSystem:
    """Test calibration cache save/load/index."""

    def test_save_and_load(self, tmp_cache_dir, fake_props):
        """Save calibration, load it back, verify data is identical."""
        from iqmapper.data import (
            _save_calibration, _load_calibration, _cache_dirs_created,
        )
        qp, gp, _, _ = fake_props
        backend = 'test_backend'
        ts = '2026-01-20 19:18:17'

        # Patch cache dir
        with patch('iqmapper.data._CACHE_DIR', str(tmp_cache_dir)):
            _cache_dirs_created.clear()
            _save_calibration(backend, ts, qp, gp)
            result = _load_calibration(backend, ts)

        assert result is not None
        loaded_qp, loaded_gp = result
        for k in qp:
            np.testing.assert_array_almost_equal(loaded_qp[k], qp[k])
        for k in gp:
            np.testing.assert_array_almost_equal(loaded_gp[k], gp[k])

    def test_load_nonexistent(self, tmp_cache_dir):
        """Loading a non-existent calibration returns None."""
        from iqmapper.data import _load_calibration, _cache_dirs_created
        with patch('iqmapper.data._CACHE_DIR', str(tmp_cache_dir)):
            _cache_dirs_created.clear()
            result = _load_calibration('fake_backend', '2099-01-01 00:00:00')
        assert result is None

    def test_index_populated(self, tmp_cache_dir, fake_props):
        """Saving a calibration adds it to the index."""
        from iqmapper.data import (
            _save_calibration, _load_index, _cache_dirs_created,
        )
        qp, gp, _, _ = fake_props
        with patch('iqmapper.data._CACHE_DIR', str(tmp_cache_dir)):
            _cache_dirs_created.clear()
            _save_calibration('test_be', '2026-01-20 10:00:00', qp, gp)
            _save_calibration('test_be', '2026-01-20 18:00:00', qp, gp)
            idx = _load_index('test_be')
        assert len(idx) == 2
        timestamps = [e['last_update'] for e in idx]
        assert '2026-01-20 10:00:00' in timestamps
        assert '2026-01-20 18:00:00' in timestamps

    def test_list_known_calibrations_by_date(self, tmp_cache_dir, fake_props):
        """list_known_calibrations filters by date."""
        from iqmapper.data import (
            _save_calibration, list_known_calibrations, _cache_dirs_created,
        )
        from datetime import date
        qp, gp, _, _ = fake_props
        with patch('iqmapper.data._CACHE_DIR', str(tmp_cache_dir)):
            _cache_dirs_created.clear()
            _save_calibration('be', '2026-01-20 10:00:00', qp, gp)
            _save_calibration('be', '2026-01-20 18:00:00', qp, gp)
            _save_calibration('be', '2026-01-21 08:00:00', qp, gp)

            jan20 = list_known_calibrations('be', date(2026, 1, 20))
            jan21 = list_known_calibrations('be', date(2026, 1, 21))
            all_cals = list_known_calibrations('be')

        assert len(jan20) == 2
        assert len(jan21) == 1
        assert len(all_cals) == 3


# ═══════════════════════════════════════════════════════════════════════
# 2. _safe_path
# ═══════════════════════════════════════════════════════════════════════

class TestSafePath:
    """Test path traversal prevention against the REAL safe_path
    (iqmapper.paths.safe_path), not a buggy inline reimplementation.

    The previous version of this class had an inline `_safe_path` that
    used `str.startswith` — the same bypass-prone check that paths.py
    explicitly documents as INCORRECT in its module docstring. The tests
    were therefore green against a buggy copy while leaving the real
    function untested. Full coverage of the new guards (null byte,
    component-level '..', 255-byte limit, symlink containment) lives in
    `tests/test_paths.py`; this class keeps the original 6 tests so the
    classic traversal cases stay covered from this entry point too.
    """

    def test_simple_name(self):
        result = safe_path('my_layout.json')
        assert result.endswith('my_layout.json')

    def test_rejects_dotdot(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('../../etc/passwd')

    def test_rejects_absolute(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('/tmp/evil.json')

    def test_rejects_tilde(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('~/.bashrc')

    def test_rejects_empty(self):
        with pytest.raises(ValueError, match="Empty"):
            safe_path('')

    def test_subdirectory_ok(self):
        result = safe_path('output/my_layout.json')
        assert 'output' in result


# ═══════════════════════════════════════════════════════════════════════
# 3. TIMEZONE / SANITIZE
# ═══════════════════════════════════════════════════════════════════════

class TestTimezone:
    """Test UTC normalization and sanitization."""

    def test_sanitize_ts_basic(self):
        from iqmapper.data import _sanitize_ts
        assert _sanitize_ts('2026-01-20 14:18:17') == '2026-01-20T14-18-17'

    def test_sanitize_ts_none(self):
        from iqmapper.data import _sanitize_ts
        assert _sanitize_ts(None) == 'unknown'

    def test_extract_last_update_converts_to_utc(self):
        from iqmapper.data import _extract_last_update
        # Simulate IBM returning Eastern time (-05:00)
        eastern = timezone(timedelta(hours=-5))
        mock_props = MagicMock()
        mock_props.last_update_date = datetime(2026, 1, 20, 14, 18, 17, tzinfo=eastern)

        result = _extract_last_update(mock_props)
        # Should be UTC: 14:18 + 5h = 19:18
        assert result == '2026-01-20 19:18:17'

    def test_extract_last_update_naive_passthrough(self):
        from iqmapper.data import _extract_last_update
        mock_props = MagicMock()
        mock_props.last_update_date = datetime(2026, 1, 20, 14, 18, 17)
        result = _extract_last_update(mock_props)
        assert result == '2026-01-20 14:18:17'

    def test_same_moment_different_tz_same_key(self):
        """Same calibration reported in different timezones → same cache key."""
        from iqmapper.data import _extract_last_update, _sanitize_ts
        eastern = timezone(timedelta(hours=-5))
        utc = timezone.utc

        props_east = MagicMock()
        props_east.last_update_date = datetime(2026, 1, 20, 14, 18, 17, tzinfo=eastern)
        props_utc = MagicMock()
        props_utc.last_update_date = datetime(2026, 1, 20, 19, 18, 17, tzinfo=utc)

        key_east = _sanitize_ts(_extract_last_update(props_east))
        key_utc = _sanitize_ts(_extract_last_update(props_utc))
        assert key_east == key_utc


# ═══════════════════════════════════════════════════════════════════════
# 4. LUCJ DEDUP
# ═══════════════════════════════════════════════════════════════════════

class TestLUCJDedup:
    """Test that alpha/beta swap deduplication works."""

    def test_dedup_removes_swaps(self):
        from iqmapper.auto_lucj import _deduplicate_layouts
        layouts = [
            {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
             'score': 0.01, 'rank': 1},
            {'layout': [2, 3, 0, 1], 'alpha': [2, 3], 'beta': [0, 1],
             'score': 0.01, 'rank': 2},  # swap of #1
            {'layout': [4, 5, 6, 7], 'alpha': [4, 5], 'beta': [6, 7],
             'score': 0.02, 'rank': 3},
        ]
        result = _deduplicate_layouts(layouts)
        assert len(result) == 2  # one swap removed

    def test_dedup_keeps_different(self):
        from iqmapper.auto_lucj import _deduplicate_layouts
        layouts = [
            {'layout': [0, 1, 2, 3], 'alpha': [0, 1], 'beta': [2, 3],
             'score': 0.01, 'rank': 1},
            {'layout': [4, 5, 6, 7], 'alpha': [4, 5], 'beta': [6, 7],
             'score': 0.02, 'rank': 2},
        ]
        result = _deduplicate_layouts(layouts)
        assert len(result) == 2  # both kept


# ═══════════════════════════════════════════════════════════════════════
# 5. SCORING
# ═══════════════════════════════════════════════════════════════════════

class TestScoring:
    """Test layout scoring with mock props."""

    def test_scoring_returns_sorted(self):
        from iqmapper.auto_lucj import lightweight_layout_error_scoring
        mock_props = MagicMock()
        # Layout 1: edges (0,1) and (2,3) — low error
        # Layout 2: edges (4,5) and (6,7) — high error
        def gate_error(gate, qubits):
            if set(qubits) in [{0, 1}, {2, 3}]:
                return 0.001
            return 0.05

        def readout_error(qubit):
            return 0.01

        mock_props.gate_error = gate_error
        mock_props.readout_error = readout_error

        layouts = [
            [4, 5, 6, 7],  # high error edges
            [0, 1, 2, 3],  # low error edges
        ]
        edges = [(0, 1), (2, 3)]

        result = lightweight_layout_error_scoring(
            mock_props, edges, layouts, 'cz')

        # Best (lowest score) should be first
        assert result[0][0] == [0, 1, 2, 3]
        assert result[0][1] < result[1][1]

    def test_scoring_none_props_raises(self):
        from iqmapper.auto_lucj import lightweight_layout_error_scoring
        with pytest.raises(RuntimeError, match="No calibration"):
            lightweight_layout_error_scoring(None, [], [], 'cz')


# ═══════════════════════════════════════════════════════════════════════
# 6. EXPORT METADATA
# ═══════════════════════════════════════════════════════════════════════

class TestExport:
    """Test export includes metadata."""

    def test_metadata_present(self, mock_state):
        from iqmapper.export import get_export_data
        result = get_export_data(mock_state, 'lucj', 'Split (Alpha/Beta/...)', 1)
        assert '_metadata' in result
        assert result['_metadata']['backend'] == 'test_backend'
        assert result['_metadata']['chip'] == 'Heron r3'


# ═══════════════════════════════════════════════════════════════════════
# 7. _build_candidates HELPER
# ═══════════════════════════════════════════════════════════════════════

class TestBuildCandidates:
    """Test the _build_candidates helper."""

    def test_with_scores(self):
        from iqmapper.auto_lucj import _build_candidates
        scores_list = [
            ([10, 11, 12, 13, 99], 0.03),
            ([20, 21, 22, 23, 99], 0.05),
        ]
        result = _build_candidates([], scores_list, 2, 1)
        assert len(result) == 2
        assert result[0]['score'] == 0.03
        assert result[0]['alpha'] == [10, 11]
        assert result[0]['beta'] == [12, 13]
        assert len(result[0]['layout']) == 4  # bridges trimmed

    def test_without_scores(self):
        from iqmapper.auto_lucj import _build_candidates
        raw = [[10, 11, 12, 13, 99], [20, 21, 22, 23, 99]]
        result = _build_candidates(raw, None, 2, 1)
        assert len(result) == 2
        assert result[0]['score'] is None
        assert result[0]['rank'] == 1


# ═══════════════════════════════════════════════════════════════════════
# 9. NaN AUDIT
# ═══════════════════════════════════════════════════════════════════════

class TestNaNAudit:
    """Test NaN fraction auditing."""

    def test_no_warnings_clean_data(self):
        from iqmapper.data import _audit_nan_fraction
        qp = {'readout_error': np.array([0.01, 0.02, 0.03])}
        gp = {'cz_error': np.array([0.001, 0.002])}
        warnings = _audit_nan_fraction(qp, gp)
        assert len(warnings) == 0

    def test_warns_partial_nan(self):
        from iqmapper.data import _audit_nan_fraction
        qp = {'readout_error': np.array([0.01, np.nan, np.nan, np.nan])}  # 75% NaN
        gp = {}
        warnings = _audit_nan_fraction(qp, gp, threshold=0.10)
        assert len(warnings) == 1
        assert 'readout_error' in warnings[0]

    def test_silent_all_nan(self):
        """100% NaN = unsupported property, not a warning."""
        from iqmapper.data import _audit_nan_fraction
        qp = {'rx_error': np.array([np.nan, np.nan, np.nan])}
        gp = {}
        warnings = _audit_nan_fraction(qp, gp)
        assert len(warnings) == 0
