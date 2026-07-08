"""
export package — Export (JSON, PNG), import (.npy, .json, .qpy, .qasm),
                 session save/restore, style presets, and delta computation.

All file I/O for layout data. No widgets — receives state and returns data.
Sub-modules are organised by responsibility; this __init__ re-exports the
public API so `from qpu_mapper import export as export_mod; export_mod.foo(...)`
resolves unchanged.

Shared constants live here because they are consumed by more than one
sub-module — submodules import them via `from . import <name>`.
"""

# Sigma threshold for "this qubit is an outlier" in
# estimate_circuit_fidelity. 3σ matches the convention used elsewhere
# in the IBM Quantum stack for calibration anomaly detection; raise
# this if the device shows wide natural variation, lower it for tighter
# selection.
OUTLIER_SIGMA = 3.0

# Minimum number of valid (non-NaN) readout samples before outlier
# detection runs. Below this the std estimate is too noisy to threshold.
OUTLIER_MIN_SAMPLES = 10

# Defensive size cap applied to all binary/JSON imports to mitigate
# denial-of-service via crafted files that declare enormous arrays in
# their header. The cap is intentionally conservative — real layouts
# and sessions are typically <1 MB; 50 MB leaves comfortable headroom.
#
# Why each entry point needs this independently:
#   - QPY: qpy.load() walks a binary format; a crafted header can drive
#     it into excessive allocations even before exception is raised.
#   - NPY: np.load with allow_pickle=False is safe against RCE but a
#     declared shape like (10**12,) makes it allocate ~8 TB before
#     OOM. The size cap stops it earlier.
#   - JSON (session / style preset): json.loads materialises the whole
#     parse tree in memory before validation — a 5 GB JSON ate the
#     kernel before the first key was inspected in earlier testing.
MAX_IMPORT_BYTES = 50 * 1024 * 1024  # 50 MB


# ─────────────────────────────────────────────────────────────────────
# Public API — re-exported from sub-modules so `export.<name>` resolves
# unchanged for every existing caller. Add new names to __all__ below
# when surfacing them.
# ─────────────────────────────────────────────────────────────────────
from .analysis import (  # noqa: E402  (late import: sub-modules read MAX_IMPORT_BYTES / OUTLIER_* above)
    get_qubit_ranking_html,
    get_gate_ranking_html,
    get_qubit_trend_data,
    get_gate_trend_data,
    get_connectivity_scores,
    get_connectivity_ranking_html,
    find_stable_qubits,
    get_stable_qubits_html,
    find_stable_gates,
    get_stable_gates_html,
)
from .autodetect import (  # noqa: E402
    parse_auto_text,
)
from .codegen import (  # noqa: E402
    get_qiskit_layout_code,
)
from .data import (  # noqa: E402
    get_export_data,
)
from .layout_io import (  # noqa: E402
    apply_layout_to_circuit,
    import_json_layout,
    import_npy_layout,
    import_layout_file,
)
from .scoring import (  # noqa: E402
    compute_error_delta,
    estimate_circuit_fidelity,
)
from .session import (  # noqa: E402
    save_session,
    save_session_to_path,
    load_session,
    list_saved_sessions,
    load_session_by_name,
    save_style_preset,
    save_style_to_path,
    load_style_preset,
    list_saved_styles,
    load_style_by_name,
)

__all__ = [
    # data
    'get_export_data',
    # layout_io
    'apply_layout_to_circuit',
    'import_json_layout',
    'import_npy_layout',
    'import_layout_file',
    # autodetect
    'parse_auto_text',
    # codegen
    'get_qiskit_layout_code',
    # scoring
    'compute_error_delta',
    'estimate_circuit_fidelity',
    # session
    'save_session',
    'save_session_to_path',
    'load_session',
    'list_saved_sessions',
    'load_session_by_name',
    'save_style_preset',
    'save_style_to_path',
    'load_style_preset',
    'list_saved_styles',
    'load_style_by_name',
    # analysis
    'get_qubit_ranking_html',
    'get_gate_ranking_html',
    'get_qubit_trend_data',
    'get_gate_trend_data',
    'get_connectivity_scores',
    'get_connectivity_ranking_html',
    'find_stable_qubits',
    'get_stable_qubits_html',
    'find_stable_gates',
    'get_stable_gates_html',
]
