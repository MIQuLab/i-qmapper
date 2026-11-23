"""Regression tests for the pure status-describers in iqmapper.ui.time_panel.

Historical crash: state.cal_history_dates is rebuilt by multiple writers
(range fetch, intraday scan, 14-day auto-load) while state.range_index /
state.intraday_index are only reset by their own writer. Stale indices
then crashed the pure describers (dates[state.range_index]) from inside
ipywidgets observers, surfacing as an IndexError traceback through
Widget.set_state.

The fix is defensive clamps in the readers plus a range_index=0 reset in
the writers we control. These tests lock in both:
  * stale-index cases must NOT raise and must return a sane string
    pointing to the LAST available entry (documented "show the latest we
    have" semantic);
  * happy-path output must be byte-identical to before the guards.
"""
from types import SimpleNamespace

import iqmapper.ui.time_panel as tp


def _range_state(dates, range_index):
    """Minimal fake carrying only the fields describe_current_view reads
    in the range branch."""
    return SimpleNamespace(
        active_backend_name='ibm_fake',
        time_mode='range',
        cal_history_dates=list(dates),
        range_index=range_index,
        # unused in the range branch, filled to prevent AttributeError on
        # other branches the function does NOT enter:
        point_timestamp=None,
        intraday_date=None,
        intraday_timestamps=[],
        intraday_index=0,
    )


def _intraday_state(timestamps, intraday_index, intraday_date='2026-01-15'):
    return SimpleNamespace(
        active_backend_name='ibm_fake',
        time_mode='intraday',
        intraday_date=intraday_date,
        intraday_timestamps=list(timestamps),
        intraday_index=intraday_index,
        # unused fields:
        point_timestamp=None,
        cal_history_dates=[],
        range_index=0,
    )


def _ref_state(mode, *, time_mode, cal_history_dates=(), range_index=0,
               intraday_timestamps=(), intraday_index=0):
    """Fake for describe_current_reference."""
    return SimpleNamespace(
        delta_ref_mode=mode,
        ref_timestamp=None,
        time_mode=time_mode,
        cal_history_dates=list(cal_history_dates),
        range_index=range_index,
        intraday_timestamps=list(intraday_timestamps),
        intraday_index=intraday_index,
    )


# ═══════════════════════════════════════════════════════════════════════
# Stale-index regressions — the four crash scenarios
# ═══════════════════════════════════════════════════════════════════════

def test_describe_current_view_range_stale_index_does_not_raise():
    """Range: range_index=25 vs. 2 dates. Old code: IndexError. New code:
    clamps to the last date and reports counter [2/2]."""
    state = _range_state(['2026-01-01', '2026-01-02'], range_index=25)
    out = tp.describe_current_view(state)
    assert out == "Range: 2026-01-02 [2/2]"


def test_describe_current_view_intraday_stale_index_does_not_raise():
    """Intraday: intraday_index=7 vs. 1 timestamp. Old code: IndexError.
    New code: clamps to the last ts and reports [1/1]."""
    state = _intraday_state(['t1'], intraday_index=7,
                            intraday_date='2026-01-15')
    out = tp.describe_current_view(state)
    assert out == "Intraday (2026-01-15): t1 UTC [1/1]"


def test_describe_current_reference_previous_range_stale_index_does_not_raise():
    """previous + range_index=25 + 2 dates: the `range_index - 1 = 24`
    guard fires and we fall through to the sequence-first return."""
    state = _ref_state('previous', time_mode='range',
                       cal_history_dates=['2026-01-01', '2026-01-02'],
                       range_index=25)
    out = tp.describe_current_reference(state)
    assert out == "previous in sequence (at first position — no delta)"


def test_describe_current_reference_previous_intraday_stale_index_does_not_raise():
    """Symmetric guard on the intraday 'previous' branch."""
    state = _ref_state('previous', time_mode='intraday',
                       intraday_timestamps=['t1'],
                       intraday_index=25)
    out = tp.describe_current_reference(state)
    assert out == "previous in sequence (at first position — no delta)"


# ═══════════════════════════════════════════════════════════════════════
# Happy paths — the guards must NOT alter valid-index output
# ═══════════════════════════════════════════════════════════════════════

def test_describe_current_view_range_valid_index_unchanged():
    state = _range_state(['2026-01-01', '2026-01-02', '2026-01-03'],
                         range_index=1)
    assert tp.describe_current_view(state) == "Range: 2026-01-02 [2/3]"


def test_describe_current_view_intraday_valid_index_unchanged():
    state = _intraday_state(['t1', 't2', 't3'], intraday_index=2,
                            intraday_date='2026-01-15')
    assert (tp.describe_current_view(state)
            == "Intraday (2026-01-15): t3 UTC [3/3]")


def test_describe_current_reference_previous_valid_range_index_unchanged():
    """Valid previous: range_index=2 with 3 dates → 'previous: <dates[1]>'."""
    state = _ref_state('previous', time_mode='range',
                       cal_history_dates=['2026-01-01', '2026-01-02', '2026-01-03'],
                       range_index=2)
    assert (tp.describe_current_reference(state)
            == "previous: 2026-01-02")


def test_describe_current_reference_previous_valid_intraday_index_unchanged():
    state = _ref_state('previous', time_mode='intraday',
                       intraday_timestamps=['t1', 't2', 't3'],
                       intraday_index=2)
    assert tp.describe_current_reference(state) == "previous: t2"
