"""Unit tests for qpu_mapper.stability.analyze_property_stability.

Covers seven cases:
  1. Error kind, happy path (stable series)
  2. Time kind, happy path (stable series)
  3. NaN handling (NaN samples are skipped)
  4. All-equal values (zero variance → not unstable)
  5. Single spike on a long series (not flagged, < min_bad_days)
  6. Repeated spike (≥ min_bad_days → flagged)
  7. Insufficient data (< 5 valid values → not flagged)
"""
import numpy as np
import pytest

from qpu_mapper.stability import analyze_property_stability


def test_error_happy_path_stable_series():
    """A stable error series sits well below 2× median → not flagged."""
    # 14 days of readout_error around 1e-3 with tiny FP noise.
    rng = np.random.default_rng(seed=42)
    values = 1.0e-3 + rng.normal(0, 5e-5, size=14)
    result = analyze_property_stability(values, kind='error')
    assert result['unstable'] is False
    assert result['total_days'] == 14
    assert result['score'] < 0.5            # max/median - 1 ≪ 1
    assert result['bad_days'] == 0
    assert result['median'] > 0


def test_time_happy_path_stable_series():
    """A stable T1 series sits well above 0.5× median → not flagged."""
    rng = np.random.default_rng(seed=7)
    # T1 around 150 µs, ±5 µs noise. Min is well above 75 µs (0.5×).
    values = 150.0 + rng.normal(0, 5, size=14)
    result = analyze_property_stability(values, kind='time')
    assert result['unstable'] is False
    assert result['total_days'] == 14
    assert result['bad_days'] == 0
    assert result['min'] > 0.5 * result['median']


def test_nan_handling_skips_nan_samples():
    """NaNs do not count toward total_days, score, or bad_days."""
    # 14 entries total; 4 are NaN. Valid 10 values are all stable.
    values = np.array([1e-3] * 10 + [np.nan] * 4, dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['total_days'] == 10
    assert result['bad_days'] == 0
    assert result['unstable'] is False
    # median computed only on valid values
    assert result['median'] == pytest.approx(1e-3)


def test_all_equal_values_zero_variance_not_unstable():
    """A perfectly flat series has score == 0 and is never unstable."""
    values = np.full(14, 5e-4, dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['unstable'] is False
    assert result['score'] == pytest.approx(0.0)
    assert result['bad_days'] == 0
    assert result['median'] == pytest.approx(5e-4)
    assert result['max'] == result['min'] == result['median']


def test_single_spike_not_flagged():
    """1 bad day on a 14-day window is below the 2-day threshold."""
    # 13 normal samples + 1 spike at 5× median.
    values = np.array([1e-3] * 13 + [5e-3], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['bad_days'] == 1
    assert result['unstable'] is False
    # The worst-day pointer still resolves to the spike position so the UI
    # can highlight it even when the series doesn't trip the flag.
    assert result['worst_day_idx'] == 13
    assert result['max'] == pytest.approx(5e-3)


def test_repeated_spike_flagged_unstable():
    """3 bad days on a 14-day window crosses the threshold."""
    # 11 normal + 3 spikes at 4× median.
    values = np.array([1e-3] * 11 + [4e-3, 4e-3, 4e-3], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['bad_days'] == 3
    assert result['unstable'] is True
    assert result['score'] == pytest.approx(3.0)   # max/median - 1 = 4 - 1


def test_insufficient_data_below_min_samples():
    """< 5 valid samples returns the degenerate "no judgment" shape."""
    values = np.array([1e-3, 2e-3, np.nan, 1.5e-3], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['unstable'] is False
    assert result['total_days'] == 3
    assert result['bad_days'] == 0
    assert result['score'] == 0.0
    assert result['worst_day_idx'] == -1


# ── Extras that catch sharp edges I'd worry about ────────────────────

def test_time_kind_repeated_dip_flagged():
    """Mirror of test_repeated_spike for the 'time' branch (T1 collapse)."""
    # 11 normal T1=150 µs + 3 dips to 60 µs (below 0.5×median=75).
    values = np.array([150.0] * 11 + [60.0, 60.0, 60.0], dtype=float)
    result = analyze_property_stability(values, kind='time')
    assert result['bad_days'] == 3
    assert result['unstable'] is True
    assert result['score'] == pytest.approx(1.0 - 60.0 / 150.0)


def test_zero_median_degenerate():
    """A series whose median is exactly zero can't form ratios → not flagged."""
    values = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.1], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['unstable'] is False
    assert result['median'] == pytest.approx(0.0)
    assert result['score'] == 0.0


def test_short_window_lower_threshold():
    """At total ≤ 14, only 2 bad days are needed (boundary case)."""
    # Use 7 days: 5 normal + 2 bad. min_bad_days(7)=2 → flagged.
    values = np.array([1e-3] * 5 + [3e-3, 3e-3], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['bad_days'] == 2
    assert result['unstable'] is True


def test_long_window_higher_threshold():
    """At total > 14, 2 bad days are NOT enough (need 3)."""
    # 30 days: 28 normal + 2 bad. min_bad_days(30)=3 → not flagged.
    values = np.array([1e-3] * 28 + [3e-3, 3e-3], dtype=float)
    result = analyze_property_stability(values, kind='error')
    assert result['bad_days'] == 2
    assert result['unstable'] is False


def test_unknown_kind_raises():
    """Misuse of the API surfaces a clear error rather than silently mis-analyzing."""
    with pytest.raises(ValueError, match='kind'):
        analyze_property_stability(np.array([1.0, 2.0, 3.0]), kind='bogus')
