"""
I-QMapper — Quantum circuit layout visualization tool.

Usage:
    from iqmapper import run
    run()                # sidebar UI

Architecture:
    connection.py shows auth UI → user clicks Connect → on_connect callback
    fires → builds a config dict → launch_mapper(config) creates SessionState
    and displays the main UI.
"""
from .log_setup import install_file_log, install_null_handler

install_null_handler()     # records never print under a notebook cell

from .connection import _make_connection_ui  # noqa: E402
from .mapper import launch_mapper            # noqa: E402


def run():
    """Launch the I-QMapper UI (sidebar layout)."""
    # Capture the kernel main loop HERE — run() is called from a notebook cell,
    # which is GUARANTEED to be the main thread with the kernel loop up. This is
    # the earliest, most reliable capture point; later build() calls are
    # capture-once + main-only, so they never overwrite it (e.g. the secondary
    # build re-entering build() must not clobber this with a worker/None value).
    from .async_util import init_main_loop
    init_main_loop()
    install_file_log()

    def _on_connect(vendor, backend_name, backend, chip,
                    props, props_warning, ansatz,
                    service, available_backends):
        launch_mapper({
            'vendor':             vendor,
            'backend':            backend,
            'backend_name':       backend_name,
            'chip':               chip,
            'service':            service,
            'available_backends': available_backends,
            'props_warning':      props_warning,
            'initial_props':      props,
            'ansatz':             ansatz,
        })

    _make_connection_ui(_on_connect)


__all__ = ["run", "launch_mapper"]
