"""
mapper.py — Thin orchestrator for QPU Mapper.

Receives connection parameters as a dict from __init__.py,
builds SessionState, and launches the UI.

All actual logic lives in state.py, data.py, plot.py, the ui/ package, and the export/ package.
"""
from .state import SessionState
from . import ui as ui_mod


def launch_mapper(config):
    """
    Initialize SessionState from connection config and launch the UI.

    Args:
        config: dict with keys:
            backend, backend_name, chip, service, available_backends,
            props_warning, initial_props, ansatz, vendor
    """
    from IPython.display import display, clear_output
    from .constants import load_custom_colors, resolve_coords

    backend = config.get('backend')
    vendor  = config.get('vendor', '')

    # ── Resolve coordinates (single call, was duplicated before) ──
    # resolve_coords only works for IBM backends (reads processor_type).
    # Offline / session-load has no backend; coords are populated later.
    #
    # Match prefix, not substring: `'ibm' in vendor.lower()` would also
    # fire on labels like "non-IBM-clone". startswith('ibm') covers
    # "IBM Quantum", "IBM" (inter-QPU), and future "IBM Cloud" cleanly.
    is_ibm = backend is not None and vendor.lower().startswith('ibm')

    if is_ibm:
        coords, chip_label = resolve_coords(backend)
    else:
        # No backend (offline / session-load): ui.py will populate
        # coords later via _load_device_graph or session restore.
        coords = []
        chip_label = config.get('chip', 'Unknown')

    # ── Build SessionState ────────────────────────────────────────
    state = SessionState()
    state.active_backend      = backend
    state.active_backend_name = config.get('backend_name')
    state.active_chip         = chip_label
    state.active_service      = config.get('service')
    state.available_backends  = config.get('available_backends', [])
    state.props_warning       = config.get('props_warning')
    state.initial_props       = config.get('initial_props')
    state.active_ansatz       = config.get('ansatz', 'lucj')
    state.active_vendor       = vendor
    state.active_coords       = coords
    state.custom_colors       = load_custom_colors()

    # ── Build and display UI ─────────────────────────────────────
    full_ui, fig = ui_mod.build_mapper_ui(state)
    clear_output(wait=True)
    display(full_ui)
