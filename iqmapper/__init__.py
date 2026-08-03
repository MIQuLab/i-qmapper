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
from .connection import _make_connection_ui
from .mapper import launch_mapper


def run():
    """Launch the I-QMapper UI (sidebar layout)."""
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
