"""
async_util.py — Background-threading helper shared across modules.

`default_spawn_bg` runs a function on a daemon thread and marshals its
completion callback back onto the kernel's main event loop. Extracted from
ui.time_panel.py so non-UI callers (connection, ui.history, ui/builder) can
use it without importing a UI panel module. Imports only stdlib, so it can
never participate in an import cycle with the rest of the package.
"""
import logging
import threading

_log = logging.getLogger(__name__)


# Cached scheduler — resolved once on the kernel's main thread (the thread
# that first imports a module which imports this one, i.e. during
# build_mapper_ui). Resolving per-job did identical work on every call:
# import tornado, fetch IOLoop.current(), allocate a lambda. `_MAIN_LOOP_
# SCHEDULER` may be None if the import-time resolver failed; default_spawn_bg
# re-tries lazily on first call so a kernel that started its event loop AFTER
# import still works.
_MAIN_LOOP_SCHEDULER = None


def _resolve_main_loop_scheduler():
    """Return a callable that schedules a zero-arg function on the main
    event loop, so background workers can hand UI mutations back to the
    thread that owns the kernel.

    Why we need this: ipywidgets value assignments are advertised as
    thread-safe, but plotly.FigureWidget.batch_update() — invoked by
    refresh_plot via on_done — has produced rendering glitches when
    called from worker threads (cached geometry, ghost edges). Marshaling
    the callback onto the main loop guarantees the figure mutates on the
    same thread the kernel runs on.

    Resolution order:
      1. tornado IOLoop (classic notebook + most JupyterLab kernels)
      2. asyncio event loop (asyncio-native kernels)
      3. inline fallback (run on the worker thread, original behaviour)
    """
    try:
        from tornado.ioloop import IOLoop
        loop = IOLoop.current()
        return lambda cb: loop.add_callback(cb)
    except Exception:
        _log.debug("tornado IOLoop unavailable", exc_info=True)
    try:
        import asyncio
        try:
            aloop = asyncio.get_running_loop()
        except RuntimeError:
            aloop = None
        if aloop is not None:
            return lambda cb: aloop.call_soon_threadsafe(cb)
    except Exception:
        _log.debug("asyncio event loop unavailable", exc_info=True)
    # No event loop available — run inline on the worker thread.
    return lambda cb: cb()


def marshal_to_main(cb):
    """Schedule a zero-arg callback ``cb`` onto the kernel main loop.

    The incremental-update counterpart to :func:`default_spawn_bg` (which
    only marshals completion). Use it to push progress updates from a worker
    thread to widgets without ever touching them off-thread. Reuses the same
    cached scheduler. Safe to call from any thread; on a kernel with no event
    loop it falls back to running ``cb`` inline.
    """
    global _MAIN_LOOP_SCHEDULER
    if _MAIN_LOOP_SCHEDULER is None:
        _MAIN_LOOP_SCHEDULER = _resolve_main_loop_scheduler()
    _MAIN_LOOP_SCHEDULER(cb)


def default_spawn_bg(fn, on_done=None, on_error=None):
    """Run `fn()` in a daemon thread and marshal the completion callback
    (`on_done` or `on_error`) onto the main event loop.

    Callers can substitute their own spawn_bg (e.g. queue + Play-timer)
    if they need finer control over scheduling."""
    global _MAIN_LOOP_SCHEDULER
    if _MAIN_LOOP_SCHEDULER is None:
        _MAIN_LOOP_SCHEDULER = _resolve_main_loop_scheduler()
    schedule = _MAIN_LOOP_SCHEDULER

    def _safe(cb, arg):
        try:
            cb(arg)
        except Exception:
            _log.exception("Callback raised")

    def _worker():
        try:
            result = fn()
        except Exception as exc:
            _log.exception("Background op raised")
            if on_error:
                schedule(lambda err=exc: _safe(on_error, err))
            return
        if on_done:
            schedule(lambda: _safe(on_done, result))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    return t
