"""
widgets_ref.py — Typed reference to UI widgets shared between modules
(extracted from constants.py to keep that module pure).
"""
from dataclasses import dataclass
from typing import Any


@dataclass
class WidgetsRef:
    """Typed reference to UI widgets, shared between ui.py and plot.py.
    Dataclass (not a dict) for IDE autocomplete (``w.view_mode``) and
    fail-loud typos (AttributeError, not silent KeyError at runtime).
    """
    view_mode: Any = None
    size_slider: Any = None
    gate_size_slider: Any = None
    gate_hl_w_slider: Any = None
    chk_link_gates: Any = None
    show_numbers: Any = None
    font_size_slider: Any = None
    chk_line_only: Any = None
    chk_err_threshold: Any = None
    chk_focus_selected: Any = None
    chk_isolate: Any = None
    txt_q_thr_min: Any = None
    txt_q_thr_max: Any = None
    txt_g_thr_min: Any = None
    txt_g_thr_max: Any = None
    plot_w_slider: Any = None
    plot_h_slider: Any = None
    intra_qpu_mode: Any = None
    num_circuits: Any = None
    dd_qubit_prop: Any = None
    dd_gate_prop: Any = None
    pp_bg_fill: Any = None
    pp_bg_line: Any = None
    pp_gate_col: Any = None
    global_text_color: Any = None
