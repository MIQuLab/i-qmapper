"""Regression tests for qpu_mapper.async_util.default_spawn_bg.

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

import qpu_mapper.async_util as async_util


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


# ── Zero-warnings guard: the scheduler resolver must never trip tornado's
#    "There is no current event loop" DeprecationWarning in a headless
#    context (no running loop) — the regression that put 1 warning back in
#    the suite after the calibration-epoch merge. ───────────────────────────

def _install_warning_tornado(monkeypatch):
    """Inject a fake tornado.ioloop whose IOLoop.current() emits the exact
    deprecation, so we can assert our resolver does NOT reach it with no
    running loop — independent of whether real tornado is installed."""
    import sys
    import types
    import warnings as _w

    class _IOLoop:
        @staticmethod
        def current():
            # Faithful to real tornado: the deprecation fires ONLY when there
            # is no running loop (IOLoop.current() → asyncio.get_event_loop()).
            import asyncio as _a
            try:
                _a.get_running_loop()
            except RuntimeError:
                _w.warn("There is no current event loop", DeprecationWarning)
            return types.SimpleNamespace(add_callback=lambda cb: cb())

    tornado = types.ModuleType("tornado")
    ioloop = types.ModuleType("tornado.ioloop")
    ioloop.IOLoop = _IOLoop
    tornado.ioloop = ioloop
    monkeypatch.setitem(sys.modules, "tornado", tornado)
    monkeypatch.setitem(sys.modules, "tornado.ioloop", ioloop)


def test_resolver_no_deprecation_without_running_loop(monkeypatch):
    import warnings
    import asyncio
    _install_warning_tornado(monkeypatch)
    # Guarantee no current/running loop in this thread.
    monkeypatch.setattr(asyncio, "get_running_loop",
                        lambda: (_ for _ in ()).throw(RuntimeError("no loop")))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        sched = async_util._resolve_main_loop_scheduler()
    assert [str(x.message) for x in rec] == []      # tornado never touched
    ran = []
    sched(lambda: ran.append(1))                    # inline fallback runs cb
    assert ran == [1]


def test_resolver_uses_loop_scheduler_when_running(monkeypatch):
    """With a running loop the tornado path IS taken (real tornado does not
    warn in that case). Proves the fix didn't disable marshaling."""
    import asyncio
    _install_warning_tornado(monkeypatch)

    async def _main():
        return async_util._resolve_main_loop_scheduler()

    sched = asyncio.run(_main())
    # Not the inline fallback: the tornado IOLoop scheduler was selected.
    delivered = []
    sched(lambda: delivered.append(1))
    assert delivered == [1]
