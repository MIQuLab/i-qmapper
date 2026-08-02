"""Regression tests for iqpu_mapper.async_util.default_spawn_bg.

Historical bug: the on_error path inside _worker scheduled
`lambda: _safe(on_error, exc)`. Python 3 unbinds `exc` when the except
block exits — with a DEFERRED scheduler (tornado add_callback,
asyncio call_soon_threadsafe), the lambda runs later and raises
NameError, silently swallowed inside `_safe`; the user's on_error is
never invoked. Inline execution masks the bug because the lambda runs
before `exc` is cleared. The fix binds `exc` as a default argument:
`lambda err=exc: _safe(on_error, err)`.

The tests below drive the DEFERRED path explicitly — the queued-callback
harness reproduces the failure mode observed with real tornado/asyncio
schedulers, without depending on either.
"""
import pytest

import iqpu_mapper.async_util as async_util


@pytest.fixture
def _scheduler_isolation():
    """Save and restore the module-level _MAIN_LOOP_SCHEDULER around the
    test so different scheduler configurations don't leak between tests."""
    saved = async_util._MAIN_LOOP_SCHEDULER
    try:
        yield
    finally:
        async_util._MAIN_LOOP_SCHEDULER = saved


def _join(thread):
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "worker thread did not finish"


def test_on_error_receives_exception_with_deferred_scheduler(_scheduler_isolation):
    """Deferred scheduler regression: with the OLD code the lambda would
    resolve `exc` after the except block cleared it, NameError'd inside
    _safe, and `capture` would never be called."""
    queued = []
    async_util._MAIN_LOOP_SCHEDULER = queued.append  # defer — don't run

    captured = []

    def _boom():
        raise ValueError("kaboom")

    def _on_error(e):
        captured.append(e)

    t = async_util.default_spawn_bg(_boom, on_error=_on_error)
    _join(t)                             # except block has exited by now
    assert queued, "on_error callback should have been queued"
    for cb in queued:                    # drain the deferred queue
        cb()

    assert len(captured) == 1, "on_error must run exactly once"
    assert isinstance(captured[0], ValueError)
    assert str(captured[0]) == "kaboom"


def test_on_done_receives_result_with_deferred_scheduler(_scheduler_isolation):
    """Sanity: the success path (already-correct in the old code)
    keeps working under the same deferred scheduler."""
    queued = []
    async_util._MAIN_LOOP_SCHEDULER = queued.append

    sentinel = object()
    captured = []

    def _ok():
        return sentinel

    def _on_done(r):
        captured.append(r)

    t = async_util.default_spawn_bg(_ok, on_done=_on_done)
    _join(t)
    for cb in queued:
        cb()

    assert captured == [sentinel]


def test_inline_fallback_scheduler(_scheduler_isolation):
    """No-event-loop path: the scheduler runs the callback synchronously
    on the worker thread. on_error must still fire."""
    async_util._MAIN_LOOP_SCHEDULER = lambda cb: cb()

    captured = []

    def _boom():
        raise RuntimeError("inline-boom")

    def _on_error(e):
        captured.append(e)

    t = async_util.default_spawn_bg(_boom, on_error=_on_error)
    _join(t)

    assert len(captured) == 1
    assert isinstance(captured[0], RuntimeError)
    assert str(captured[0]) == "inline-boom"


def test_callback_exception_is_contained(_scheduler_isolation):
    """_safe wraps the user callback so an exception it raises does not
    escape into the scheduler / worker. The worker must finish cleanly."""
    queued = []
    async_util._MAIN_LOOP_SCHEDULER = queued.append

    def _ok():
        return 1

    def _on_done_that_raises(_r):
        raise RuntimeError("callback failure — must be contained by _safe")

    t = async_util.default_spawn_bg(_ok, on_done=_on_done_that_raises)
    _join(t)                             # worker thread finished normally
    # Draining the callback must not propagate the exception either — _safe
    # logs it and returns.
    for cb in queued:
        cb()
