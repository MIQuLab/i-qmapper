"""
ui/builder.py — MapperUIBuilder class.

Constructs the main I-QMapper UI. Public entry is the module-level
``build_mapper_ui(state, is_secondary=False) -> (ui_hbox, fig)`` (kept
for back-compat); it instantiates this class and calls ``build()``.
Shared state (``self.fig``, ``widgets_ref``, ``self._ext_fns``) lives on
the instance rather than being smuggled through closures.

All ipywidgets live here; business logic delegates to data.py / plot.py
/ export.py via the shared SessionState.
"""
import logging
import os
from types import SimpleNamespace as _NS
import numpy as np
from ipywidgets import (
    Output, VBox, HBox, RadioButtons, Label, ColorPicker,
    IntSlider, FloatSlider, BoundedIntText, BoundedFloatText, Button, Layout,
    Checkbox, Tab, HTML, Dropdown,
    Textarea, Text, ToggleButtons, Accordion, DatePicker, FileUpload,
    IntProgress,
)

from ..constants import (
    MAX_CIRCUITS, COLORSCALE_OPTIONS, QUBIT_PROP_OPTIONS, GATE_PROP_OPTIONS,
    STANDARD_COLORS, hex_to_rgba, cat_paws_html,
    save_custom_colors, WidgetsRef,
)
from ..html_safe import esc
from ..paths import safe_path as _safe_path_shared
from ..paths import resolve_user_save_path as _user_save_path
from .._upload import read_upload as _read_upload
from .._sidebar_css import SIDEBAR_CSS
from .. import data as data_mod
from .. import plot as plot_mod
from .. import export as export_mod
from .time_panel import build_time_panel as _build_time_panel
from .pretty_json import pretty_json_html as _pretty_json_html
from .color_picker import PowerPointColorPicker


_log = logging.getLogger(__name__)


# PowerPointColorPicker, _read_upload, and SIDEBAR_CSS were moved out
# of this file (to ui.color_picker.py / _upload.py / _sidebar_css.py
# respectively) for shareability and diff readability.


class MapperUIBuilder:
    """Orchestrates construction of the I-QMapper UI.

    Usage::

        builder = MapperUIBuilder(state, is_secondary=False)
        ui_hbox, fig = builder.build()
    """

    # Role-label → group-key map, shared by group-style helpers and the
    # _mk_* per-circuit factories. Class-level so all instances share
    # one object.
    _SEL_TO_GROUP = {
        'Select Alpha': 'alpha', 'Select Beta': 'beta',
        'Select Ancilla': 'ancilla', 'Select Other': 'other',
        'Select Qubits': 'qubits', 'Remove': None,
    }

    def __init__(self, state, is_secondary: bool = False):
        self.state = state
        self.is_secondary = is_secondary

        # Outputs populated by ``build()`` so callers can introspect a
        # built panel (handy for tests and future inter-QPU wiring).
        self.fig = None
        self.widgets_ref = None
        self.ui_hbox = None

        # Reentrancy / cross-module guards for self._load_device_graph.
        # * self._loading_device — clean bool, only mutated inside
        #   _load_device_graph (True at entry, False in finally).
        # * self._switching_backend — kept as a [bool] single-element list
        #   because ui.backend_switch.setup_backend_switch (different
        #   module) mutates the SAME list object via `switching_backend[0]
        #   = True/False`. The asymmetric shape matches each flag's
        #   mutator contract — do not "fix" it.
        self._loading_device = False
        self._switching_backend = [False]

    def build(self, return_widgets=False):
        """Run the full UI construction pipeline.

        Returns ``(ui_hbox, fig)`` by default; with ``return_widgets=True``
        also returns ``widgets_ref`` (the secondary-QPU path uses this to
        reach the Fig Width/Height sliders directly instead of walking the
        tree by description). ``self.widgets_ref`` is always populated.
        """
        # Capture the kernel main loop + main-thread identity HERE — build()
        # runs on the main thread inside the running loop (live cell execution),
        # which is the only place asyncio.get_running_loop() succeeds. Workers
        # then schedule their paint onto THIS captured loop instead of resolving
        # (and inline-running) on themselves. Idempotent; no-op headless.
        from ..async_util import init_main_loop
        init_main_loop()
        ui_hbox, _ = self._run_pipeline()
        self.ui_hbox = ui_hbox
        if return_widgets:
            return ui_hbox, self.fig, self.widgets_ref
        return ui_hbox, self.fig

    def _run_pipeline(self):
        """Build pipeline: widget construction → ctx setup → wiring → assemble."""
        state = self.state
        is_secondary = self.is_secondary
        W = Layout(width='100%', margin='2px 0px', min_width='0px')
        self.out = Output(layout=Layout(max_height='200px', overflow='auto', width='100%'))
        self.lbl_notify = HTML("")  # empty placeholder banner above Output

        # ── Create figure ────────────────────────────────────────────
        self.fig = plot_mod.create_figure()

        # Inter-QPU holders (QPU2 state / fig / sidebar / zoom bar) now
        # live inside ui.inter_qpu.setup_inter_qpu — nothing else reads them.

        # ── Core widgets ─────────────────────────────────────────────
        self._build_core_widgets(W, state)

        # ── Inter-QPU mode widgets ───────────────────────────────────
        self._build_inter_qpu_widgets(W)

        # ── Property dropdowns ───────────────────────────────────────
        self._build_property_dropdowns(W)

        # ── Error threshold filter ───────────────────────────────────
        self._build_threshold_widgets(W)

        # ── Error map panel ──────────────────────────────────────────
        self.colorscale_preview.value = self._make_cs_preview()

        # ── Color pickers ────────────────────────────────────────────
        self._build_color_pickers(state)

        # ── File widgets ─────────────────────────────────────────────
        self._build_file_widgets(W)

        # ── Session save/load ────────────────────────────────────────
        self._build_session_io_widgets(W)

        # ── Style preset save/load ───────────────────────────────────
        self._build_style_preset_widgets(W)

        # ── Export inputs (JSON / PNG / preview) ─────────────────────
        self._build_export_widgets(W)

        # ── LQS banner (Estimated Layout-Quality, above inspector cards) ──
        self._build_fidelity_widget()

        # ── Inspector + Stability widgets (rendered below the LQS banner) ──
        self._build_inspector_widgets(W)

        # ── Section "card" containers ────────────────────────────────────
        self._build_section_cards()

        # ── Qubit + Gate Ranking ──────────────────────────────────────
        self._build_ranking_widgets(W)

        # ── Trend Charts (Qubit + Gate) ──────────────────────────────
        self._build_trend_widgets(W)

        # ── Qubit Inspector ──────────────────────────────────────────
        self._build_qubit_inspector_widgets()

        # ── Stable Finder (Qubit + Gate) ─────────────────────────────
        self._build_stable_finder_widgets(W)

        # ── Calibration history widgets ──────────────────────────────
        self._build_calibration_history_widgets(W)

        # ── Time-lapse export options ─────────────────────────────────
        self._build_timelapse_widgets(W)

        # ── Auto-tab input ───────────────────────────────────────────
        self._build_auto_input_widgets(W)

        # ── Auto-LUCJ widgets ────────────────────────────────────────
        self._build_auto_lucj_widgets(W)

        # ── Legend controls ──────────────────────────────────────────
        self._build_legend_widgets(W)

        # ── Undo / redo ──────────────────────────────────────────────
        self._build_undo_redo_widgets()

        # ── Backend switcher ─────────────────────────────────────────
        self._build_backend_switcher_widgets(W)

        # ── Navigation ───────────────────────────────────────────────
        self._build_navigation_widgets()

        # ── Zoom buttons ─────────────────────────────────────────────
        self._build_zoom_widgets()

        # ── Build widgets_ref for plot.py ────────────────────────────
        self._build_widgets_ref()

        # ═══════════════════════════════════════════════════════════════
        # CORE CALLBACKS
        # ═══════════════════════════════════════════════════════════════

        # Path sanitisation lives in paths.py (module-level import).
        # `_safe_path` is a thin local alias kept so inline callers keep
        # their existing shape. Uses os.path.commonpath so '../alice_evil'
        # traversal is correctly rejected when cwd is '/home/alice' — the
        # previous str.startswith implementation was vulnerable to that.
        _safe_path = _safe_path_shared

        # Dispatch dict for helpers wired later by setup_analysis_callbacks;
        # used by self._refresh / self._get_export / self._update_view.
        self._ext_fns = {
            'update_fidelity': lambda: None,
            'update_ranking': lambda c=None: None,
            'update_grank': lambda c=None: None,
        }

        # ── Colorscale ───────────────────────────────────────────────
        self.dd_colorscale.observe(self._on_colorscale_change, 'value')

        # ── Property dropdowns ───────────────────────────────────────
        self.dd_qubit_prop.observe(lambda c: self._refresh(), 'value')
        self.dd_gate_prop.observe(lambda c: self._refresh(), 'value')

        # ── Display widgets ──────────────────────────────────────────
        for w in [self.view_mode, self.size_slider, self.gate_size_slider, self.gate_hl_w_slider, self.chk_link_gates, self.show_numbers, self.font_size_slider, self.chk_line_only, self.chk_err_threshold, self.chk_focus_selected, self.chk_isolate]:
            w.observe(self._refresh, 'value')

        self.global_text_color.observe(self._on_global_text_color, 'value')
        self.txt_q_thr_min.observe(self._on_threshold_text_change, 'value')
        self.txt_q_thr_max.observe(self._on_threshold_text_change, 'value')
        self.txt_g_thr_min.observe(self._on_threshold_text_change, 'value')
        self.txt_g_thr_max.observe(self._on_threshold_text_change, 'value')
        self.plot_w_slider.observe(self._update_fig_layout, 'value')
        self.plot_h_slider.observe(self._update_fig_layout, 'value')

        # ── Zoom ─────────────────────────────────────────────────────
        self.btn_zoom_in.on_click(lambda b: plot_mod.do_zoom(self.fig, state, 0.65))
        self.btn_zoom_out.on_click(lambda b: plot_mod.do_zoom(self.fig, state, 1.5))
        self.btn_zoom_rst.on_click(lambda b: plot_mod.reset_zoom(self.fig, state))

        # ── Undo / redo ──────────────────────────────────────────────
        self.btn_undo.on_click(self._on_undo)
        self.btn_redo.on_click(self._on_redo)

        # ── Backend switch ───────────────────────────────────────────
        # Reentrancy guard shared with self._load_device_graph: the
        # method reads self._switching_backend[0] to refuse observer-driven
        # reloads mid-switch, and ui.backend_switch mutates the SAME list
        # via switching_backend[0] = True/False. The list shape IS the
        # cross-module API — do not "fix" it to a bool.
        # Shared single-element cell holding ui.inter_qpu's refresh hook.
        # setup_inter_qpu runs LATER (after assembly), so we pass the cell
        # now and it is populated below — backend_switch reads cell[0] at
        # call time to relabel the QPU-1 switcher after a backend switch.
        self._refresh_qpu1_label = [None]
        from .backend_switch import setup_backend_switch
        setup_backend_switch(_NS(
            state=state,
            dd_backend_switch=self.dd_backend_switch,
            lbl_switch_status=self.lbl_switch_status,
            lbl_calibration_date=self.lbl_calibration_date,
            lbl_props_warning=self.lbl_props_warning,
            chk_keep_layout=self.chk_keep_layout,
            switching_backend=self._switching_backend,
            load_device_graph=self._load_device_graph,
            refresh=self._refresh, get_export=self._get_export,
            out=self.out,
            refresh_qpu1_label=self._refresh_qpu1_label,
            refresh_time_panel=lambda: self._time_api.rebuild(),
        ))

        # ── Inter-QPU mode (toggle + connect QPU2) ───────────────────
        # Wiring lives in ui.inter_qpu.setup_inter_qpu — invoked AFTER
        # assembly because its toggle mutates ui_sidebar / fig_box /
        # full_ui / qpu2_* children. Here we just allocate the swap container.
        self._sidebar_body = VBox()  # swappable sidebar body (QPU1 / QPU2)

            # ── Device reload triggers ───────────────────────────────────
        self.dp_date.observe(self._load_device_graph, 'value')
        self.txt_hw_hour.observe(self._load_device_graph, 'value')
        # vendor / chip not observed — both derive from state
        # (active_vendor / active_chip), set by mapper.py at launch and
        # by _on_backend_switch on subsequent backend changes.

        # ── File handlers ────────────────────────────────────────────
        # `_read_upload` bridges ipywidgets v7 tuple vs. v8 dict shape.
        self.btn_unified_upload.observe(self._on_unified_upload, names='value')
        self.btn_json_import.observe(self._on_json_import, names='value')

        # Session save/load + style preset wiring is further down (after
        # the analysis callbacks block) — order preserved for diff stability.

        self.btn_clear_all.on_click(self._on_clear_all)
        self.btn_refresh.on_click(self._on_refresh)
        self.btn_auto.on_click(self._on_auto_map)

        # ── Auto-LUCJ callbacks (extracted to ui.lucj_panel.py) ──────
        from .lucj_panel import setup_lucj_callbacks
        from ..async_util import default_spawn_bg, marshal_to_main
        # Shared cell: ui.inter_qpu populates [0] with its handoff hook
        # (reveal QPU-2 + prefill dropdown + stage candidate). Same
        # deferred-wiring pattern as the QPU-1 title-refresh cell.
        self._inter_qpu_handoff = [None]
        # Shared cell: ui.lucj_panel populates [0] with the candidate
        # re-ranking hook, which ui.benchmark_panel calls after installing a
        # measured epoch. Same deferred-wiring pattern as the handoff cell.
        self._bench_rescore = [None]
        # Shared cell: ui.inter_qpu publishes a read-only accessor for QPU 2's
        # live state/figure here, so the benchmark panel — built before
        # setup_inter_qpu runs — can reach QPU 2 once it connects.
        self._qpu2_handles = [None]
        _lucj_ctx = _NS(
            state=state, out=self.out, W=W,
            lucj_norb_slider=self.lucj_norb_slider, lucj_ancilla_dd=self.lucj_ancilla_dd,
            lucj_min_ancilla=self.lucj_min_ancilla, lucj_top_n=self.lucj_top_n,
            lucj_search_cap=self.lucj_search_cap,
            lucj_chk_score=self.lucj_chk_score, chk_deep_search=self.chk_deep_search,
            chk_lucj_diverse=self.chk_lucj_diverse,
            chk_diverse_cap=self.chk_diverse_cap,
            dd_search_engine=self.dd_search_engine,
            int_beam_width=self.int_beam_width,
            int_refine_top=self.int_refine_top,
            guided_params_box=self.bench_guided_box,
            btn_lucj_run=self.btn_lucj_run,
            lbl_lucj_status=self.lbl_lucj_status, lbl_lucj_trunc=self.lbl_lucj_trunc,
            lbl_lucj_notices=self.lbl_lucj_notices,
            dd_lucj_layout=self.dd_lucj_layout,
            lbl_lucj_detail=self.lbl_lucj_detail,
            lucj_multi_norbs=self.lucj_multi_norbs, lucj_buffer_slider=self.lucj_buffer_slider,
            lucj_multi_ancillas=self.lucj_multi_ancillas,
            btn_lucj_multi=self.btn_lucj_multi,
            btn_lucj_multi_sweep=self.btn_lucj_multi_sweep,
            btn_scan_batch=self.btn_scan_batch,
            scan_batch_box=self.scan_batch_box,
            lbl_lucj_multi=self.lbl_lucj_multi,
            lbl_lucj_multi_notices=self.lbl_lucj_multi_notices,
            lucj_multi_detail_box=self.lucj_multi_detail_box,
            num_circuits=self.num_circuits, intra_qpu_mode=self.intra_qpu_mode, dd_mode=self.dd_mode,
            set_num_circuits=self._set_num_circuits_silent,
            lucj_all_layouts=self._lucj_all_layouts,
            rescore_cell=self._bench_rescore,
            lucj_multi_results=self._lucj_multi_results,
            lucj_multi_norbs_list=self._lucj_multi_norbs,
            lucj_multi_ancillas_list=self._lucj_multi_ancillas,
            lucj_multi_dropdowns=self._lucj_multi_dropdowns,
            refresh=self._refresh, get_export=self._get_export, update_view=lambda: self._update_view(),
            chk_line_only=self.chk_line_only,
            # Multi-QPU scan.
            btn_scan_qpus=self.btn_scan_qpus,
            chk_scan_other_families=self.chk_scan_other_families,
            btn_scan_clear=self.btn_scan_clear,
            lbl_scan_progress=self.lbl_scan_progress,
            scan_results_box=self.scan_results_box,
            scan_spawn_bg=default_spawn_bg,
            scan_marshal=marshal_to_main,
            inter_qpu_handoff=self._inter_qpu_handoff,
            # Stability filter.
            chk_stability_filter=self.chk_stability_filter,
            rb_stability_window=self.rb_stability_window_auto,
            ft_stability_threshold=self.ft_stability_threshold,
            ft_stability_floor=self.ft_stability_floor,
            stability_ctrl_box=self.stability_ctrl_box,
            lbl_lucj_stability=self.lbl_lucj_stability,
            chk_scan_stability=self.chk_scan_stability,
        )
        setup_lucj_callbacks(_lucj_ctx)
        self._refresh_stability_filter_availability()

        # ── 📡 Benchmark section (ui.benchmark_panel) ────────────────
        # After setup_lucj_callbacks, so the re-ranking hook cell is filled
        # by the time the panel could call it. Opt-in only: nothing here
        # runs unless the user clicks Run benchmark.
        from .benchmark_panel import setup_benchmark_panel
        _bench_ctx = _NS(
            state=state,
            dd_bench_target=self.dd_bench_target,
            txt_bench_custom=self.txt_bench_custom,
            tb_bench_qpu=self.tb_bench_qpu,
            dd_bench_level=self.dd_bench_level,
            lbl_bench_summary=self.lbl_bench_summary,
            chk_bench_ack=self.chk_bench_ack,
            btn_bench_run=self.btn_bench_run,
            lbl_bench_status=self.lbl_bench_status,
            chk_bench_absolute=self.chk_bench_absolute,
            # Read-only from the panel except for the three writes the
            # "show on map" link performs (enable the filter, set the gate
            # max, select the 2Q property).
            chk_err_threshold=self.chk_err_threshold,
            txt_g_thr_max=self.txt_g_thr_max,
            dd_gate_prop=self.dd_gate_prop,
            lbl_bench_consistency=self.lbl_bench_consistency,
            lbl_bench_details=self.lbl_bench_details,
            acc_bench_details=self.acc_bench_details,
            btn_bench_show_on_map=self.btn_bench_show_on_map,
            lbl_bench_reuse=self.lbl_bench_reuse,
            btn_bench_reuse=self.btn_bench_reuse,
            btn_bench_reanalyze=self.btn_bench_reanalyze,
            btn_bench_calcheck=self.btn_bench_calcheck,
            lbl_bench_calcheck=self.lbl_bench_calcheck,
            lbl_lucj_notices=self.lbl_lucj_notices,
            lucj_all_layouts=self._lucj_all_layouts,
            rescore_cell=self._bench_rescore,
            refresh=self._refresh,
            chk_inter_qpu=self.chk_inter_qpu,
            qpu2_handles_cell=self._qpu2_handles,
            bench_spawn_bg=default_spawn_bg,
            bench_runner_factory=None,      # production: built at click time
        )
        self._bench_api = setup_benchmark_panel(_bench_ctx)

        # ── PNG save popup ───────────────────────────────────────────
        # ui.png_export builds the popup + sliders + checkbox and wires
        # btn_save_img.on_click. We capture png_popup so the export tab
        # can mount it; the slider / checkbox refs are unneeded (discard
        # with `_`).
        from .png_export import setup_png_export
        self.png_popup, _, _ = setup_png_export(_NS(
            state=state, widgets_ref=self.widgets_ref,
            fname_img=self.fname_img, btn_save_img=self.btn_save_img,
            W=W,
            plot_mod=plot_mod, safe_path=_safe_path, user_save_path=_user_save_path, save_dir=self._save_dir, out=self.out,
            lbl_png_status=self.lbl_png_status,
            fig=self.fig,               # live figure — copy its visible colorbars into the PNG
        ))


        # ── History + GIF callbacks (extracted to ui.history.py) ──────
        from .history import setup_history_callbacks
        _hist_ctx = _NS(
            state=state, out=self.out, W=W, widgets_ref=self.widgets_ref,
            data_mod=data_mod, plot_mod=plot_mod, export_mod=export_mod,
            hist_start_date=self.hist_start_date, hist_end_date=self.hist_end_date,
            btn_fetch_hist=self.btn_fetch_hist, hist_progress=self.hist_progress,
            hist_status=self.hist_status, hist_slider_box=self.hist_slider_box,
            btn_export_gif=self.btn_export_gif, fname_img=self.fname_img,
            # Time-lapse format selector + fps slider + filename. Optional in
            # ui.history.py via getattr(); now wired so user can pick GIF/MP4
            # and FPS before clicking 🎬.
            fname_timelapse=self.fname_timelapse,
            dd_timelapse_format=self.dd_timelapse_format,
            slider_timelapse_fps=self.slider_timelapse_fps,
            refresh=self._refresh, safe_path=_safe_path, user_save_path=_user_save_path, save_dir=self._save_dir,
            user_save_path_in_dir=self._user_save_path_in_save_dir,
            lbl_timelapse_status=self.lbl_timelapse_status,
        )
        _hist_fns = setup_history_callbacks(_hist_ctx)

        # ── Time / Delta panel (built ONCE, mounted in error_inspect) ──
        # Built at builder scope so the panel's slider state (intraday
        # timestamps, range) persists across nav / num_circuits / mode
        # changes — otherwise every _update_view rebuild would discard a
        # fetched scan. The returned `api` handle is KEPT (self._time_api):
        # a backend switch resets the panel's time-state but never fires its
        # own observers, so the switch wiring calls api.rebuild() to re-render
        # the header + sub-panels for the new backend.
        self._time_panel, self._time_api = _build_time_panel(
            state=state,
            data_mod=data_mod,
            export_mod=export_mod,
            refresh_plot=self._refresh,
            load_device_graph=self._load_device_graph,
            spawn_bg=None,                              # default daemon-thread runner
            on_gif_export=_hist_fns.get('on_export_gif'),
            on_history_loaded=self._enable_export_gif,
        )

        # ── Analysis callbacks (extracted to ui.analysis_panel.py) ────
        from .analysis_panel import setup_analysis_callbacks
        _analysis_ctx = _NS(
            state=state, out=self.out, export_mod=export_mod,
            fname_json=self.fname_json, btn_save_json=self.btn_save_json,
            lbl_json_status=self.lbl_json_status,
            export_format=self.export_format,
            lbl_fidelity=self.lbl_fidelity,
            dd_mode=self.dd_mode, num_circuits=self.num_circuits, intra_qpu_mode=self.intra_qpu_mode,
            txt_inspect_qubit=self.txt_inspect_qubit, btn_inspect=self.btn_inspect,
            lbl_inspect_result=self.lbl_inspect_result,
            dd_rank_prop=self.dd_rank_prop, dd_rank_order=self.dd_rank_order,
            txt_rank_filter=self.txt_rank_filter, rank_table_html=self.rank_table_html,
            dd_grank_prop=self.dd_grank_prop, dd_grank_order=self.dd_grank_order,
            txt_grank_filter=self.txt_grank_filter, grank_table_html=self.grank_table_html,
            txt_trend_qubits=self.txt_trend_qubits, dd_trend_prop=self.dd_trend_prop,
            btn_trend_plot=self.btn_trend_plot, trend_chart_box=self.trend_chart_box,
            txt_trend_gates=self.txt_trend_gates, dd_gtrend_prop=self.dd_gtrend_prop,
            btn_gtrend_plot=self.btn_gtrend_plot, gtrend_chart_box=self.gtrend_chart_box,
            dd_stable_prop=self.dd_stable_prop, txt_stable_threshold=self.txt_stable_threshold,
            slider_stable_ratio=self.slider_stable_ratio, btn_stable_find=self.btn_stable_find,
            stable_html=self.stable_html,
            dd_gstable_prop=self.dd_gstable_prop, txt_gstable_threshold=self.txt_gstable_threshold,
            slider_gstable_ratio=self.slider_gstable_ratio, btn_gstable_find=self.btn_gstable_find,
            gstable_html=self.gstable_html,
            # Section A — current calibration inspector (opt-in)
            chk_inspect_enable=self.chk_inspect_enable,
            inspector_body_box=self.inspector_body_box,
            dd_inspect_qubit=self.dd_inspect_qubit,
            lbl_inspect_qubit_metrics=self.lbl_inspect_qubit_metrics,
            dd_inspect_gate=self.dd_inspect_gate,
            lbl_inspect_gate_metrics=self.lbl_inspect_gate_metrics,
            # Section B — stability check (opt-in via chk_stability_enable)
            chk_stability_enable=self.chk_stability_enable,
            stability_body_box=self.stability_body_box,
            rb_stability_window=self.rb_stability_window,
            dd_unstable_qubit=self.dd_unstable_qubit,
            lbl_unstable_qubit_metrics=self.lbl_unstable_qubit_metrics,
            dd_unstable_gate=self.dd_unstable_gate,
            lbl_unstable_gate_metrics=self.lbl_unstable_gate_metrics,
            lbl_stability_note=self.lbl_stability_note,
            stability_section_box=self.stability_section_box,
            QUBIT_PROP_OPTIONS=QUBIT_PROP_OPTIONS,
            GATE_PROP_OPTIONS=GATE_PROP_OPTIONS,
            safe_path=_safe_path, user_save_path=_user_save_path, save_dir=self._save_dir, get_export=self._get_export,
            user_save_path_in_dir=self._user_save_path_in_save_dir,
        )
        _analysis_result = setup_analysis_callbacks(_analysis_ctx)
        self._ext_fns.update(_analysis_result)

        # ── Session save/load + style preset save/load ───────────────
        # All four button observers + collect/apply helpers live in
        # ui.session_io.setup_session_io.
        from .session_io import setup_session_io
        setup_session_io(_NS(
            state=state, widgets_ref=self.widgets_ref,
            dd_colorscale=self.dd_colorscale, dd_mode=self.dd_mode, export_format=self.export_format,
            pp_bg_fill=self.pp_bg_fill, pp_bg_line=self.pp_bg_line, pp_gate_col=self.pp_gate_col,
            global_text_color=self.global_text_color,
            txt_session_path=self.txt_session_path,
            btn_save_session=self.btn_save_session,
            btn_load_session=self.btn_load_session,
            lbl_session_status=self.lbl_session_status,
            lbl_session_save_status=self.lbl_session_save_status,
            lbl_style_status=self.lbl_style_status,
            txt_style_path=self.txt_style_path,
            btn_save_style=self.btn_save_style,
            btn_load_style=self.btn_load_style,
            fig=self.fig, on_plot_click=self._on_plot_click,
            export_mod=export_mod, plot_mod=plot_mod,
            safe_path=_safe_path, user_save_path=_user_save_path, save_dir=self._save_dir, out=self.out,
            refresh=self._refresh, get_export=self._get_export,
            resync_group_controls=self._resync_group_controls_after_load,
            flush_controls_to_groups=self._flush_controls_to_groups,
        ))

        # ── Nav observers ────────────────────────────────────────────
        self.nav_bar.observe(self._update_view, 'value')
        self.intra_qpu_mode.observe(self._toggle_intra_qpu, 'value')
        self.num_circuits.observe(self._update_view, 'value')
        self.dd_mode.observe(self._update_view, 'value')

        # ── Assemble layout ───────────────────────────────────────────

        (full_ui, ui_sidebar, ui_sidebar_content,
         fig_box, _qpu2_sidebar_box, _qpu2_fig_box) = self._assemble_full_ui()

        # ── Post-assembly side-effects ───────────────────────────────
        # Show calibration warning
        if state.props_warning:
            with self.out:
                print(f"⚠️  {state.props_warning}")

        # Initial load
        self._load_device_graph()
        self._update_view()

        # ── Auto-load of recent calibration history (Stability inspector) ──
        # Daemon thread keeps the kernel responsive while walking cache /
        # fetching missing days; Section-B carries a "Loading…" indicator
        # until done. auto_load_recent_history gatekeeps:
        #   • 'noop' if state.cal_history_dates already populated;
        #   • 'cache' when local cache has ≥5 snapshots (no IBM call);
        #   • 'fetch' otherwise — same routine Multi-day uses.
        # On done, _enable_export_gif triggers _update_fidelity and
        # Section B re-renders.
        if not state.cal_history_dates:
            # Loading indicator BEFORE spawning the thread.
            self.lbl_stability_note.value = (
                "<span style='color:#1a5276;font-size:16px;'>"
                "Loading calibration history (last 14 days)... ⏳</span>"
            )

            from ..async_util import default_spawn_bg
            default_spawn_bg(self._auto_load_work, on_done=self._auto_load_done,
                             on_error=self._auto_load_err)

        # ── Inter-QPU wiring (toggle + connect) ───────────────────────
        # Must run AFTER assembly: setup_inter_qpu's toggle handler
        # mutates ui_sidebar / fig_box / _qpu2_*_box / full_ui children.
        # Skipped on secondary instances — they don't host their own
        # tertiary, and chk_inter_qpu / dd_backend_qpu2 / etc. are no
        # longer mounted in the sidebar.
        if not is_secondary:
            from .inter_qpu import setup_inter_qpu
            self._refresh_qpu1_label[0] = setup_inter_qpu(_NS(
                data_mod=data_mod, state=state,
                chk_inter_qpu=self.chk_inter_qpu,
                dd_backend_qpu2=self.dd_backend_qpu2,
                btn_connect_qpu2=self.btn_connect_qpu2,
                lbl_qpu2_status=self.lbl_qpu2_status,
                plot_w_slider=self.plot_w_slider, plot_h_slider=self.plot_h_slider,
                lbl_fig1=self.lbl_fig1,
                ui_sidebar=ui_sidebar,
                ui_sidebar_content=ui_sidebar_content,
                sidebar_body=self._sidebar_body,
                fig_box=fig_box,
                qpu2_sidebar_box=_qpu2_sidebar_box,
                qpu2_fig_box=_qpu2_fig_box,
                full_ui=full_ui,
                handoff_cell=self._inter_qpu_handoff,
                qpu2_handles_cell=self._qpu2_handles,
            ))

        return full_ui, self.fig

    def _build_widgets_ref(self):
        """Bundle the already-built display widgets into a ``WidgetsRef``.
        Must run AFTER each widget is on ``self.*`` via its ``_build_*``.
        """
        self.widgets_ref = WidgetsRef(
            view_mode=self.view_mode,
            size_slider=self.size_slider,
            gate_size_slider=self.gate_size_slider,
            gate_hl_w_slider=self.gate_hl_w_slider,
            chk_link_gates=self.chk_link_gates,
            show_numbers=self.show_numbers,
            font_size_slider=self.font_size_slider,
            chk_line_only=self.chk_line_only,
            chk_err_threshold=self.chk_err_threshold,
            chk_focus_selected=self.chk_focus_selected,
            chk_isolate=self.chk_isolate,
            txt_q_thr_min=self.txt_q_thr_min,
            txt_q_thr_max=self.txt_q_thr_max,
            txt_g_thr_min=self.txt_g_thr_min,
            txt_g_thr_max=self.txt_g_thr_max,
            plot_w_slider=self.plot_w_slider,
            plot_h_slider=self.plot_h_slider,
            intra_qpu_mode=self.intra_qpu_mode,
            num_circuits=self.num_circuits,
            dd_qubit_prop=self.dd_qubit_prop,
            dd_gate_prop=self.dd_gate_prop,
            pp_bg_fill=self.pp_bg_fill,
            pp_bg_line=self.pp_bg_line,
            pp_gate_col=self.pp_gate_col,
            global_text_color=self.global_text_color,
        )

    def _build_color_pickers(self, state):
        """Construct the three PowerPointColorPicker widgets on ``self.*``.

        ``state`` is passed so the pickers can read
        ``state.custom_colors`` for their custom-swatch row.
        """
        self.pp_bg_fill = PowerPointColorPicker(
            state, '#ffffff', 'Background Fill:',
            lambda c: self._refresh(),
        )
        self.pp_bg_line = PowerPointColorPicker(
            state, '#dee2e6', 'Background Line:',
            lambda c: self._refresh(),
        )
        self.pp_gate_col = PowerPointColorPicker(
            state, '#e9ecef', 'Gate Default Color:',
            lambda c: self._refresh(),
        )

    def _build_file_widgets(self, W):
        """Construct the four file-action widgets in the sidebar.

        Pure widget construction — callbacks wired in ``_run_pipeline``.
        ``btn_refresh`` is also mutated in-place by ``_on_refresh`` to
        show the "⏳ Refreshing..." transient label.
        """
        self.btn_unified_upload = FileUpload(
            accept='.npy,.json,.qpy,.qasm', multiple=False,
            description='📂 Upload Layout', layout=W,
        )
        self.btn_json_import = FileUpload(
            accept='.json', multiple=False,
            description='📂 Import JSON', layout=W,
        )
        self.btn_clear_all = Button(
            description='⚠️ Clear All Layouts',
            button_style='danger', layout=W,
        )
        self.btn_refresh = Button(
            description='🔄 Refresh (Fresh Start)',
            button_style='warning', layout=W,
        )

    def _build_threshold_widgets(self, W):
        """Construct the error-threshold filter widgets.

        ``chk_err_threshold`` (enable), four ``txt_{q,g}_thr_{min,max}``
        bounds (floats parsed from Text), and the passive
        ``threshold_status`` HTML banner driven by
        ``_update_threshold_status``. The five user-input widgets feed
        ``_build_widgets_ref``; ``threshold_status`` is sidebar-only.
        """
        self.chk_err_threshold = Checkbox(
            value=False, description='Enable Threshold', layout=W,
        )
        self.chk_focus_selected = Checkbox(
            value=False, description='Focus: selected layout only', layout=W,
        )
        self.chk_isolate = Checkbox(
            value=False, description='Isolate layout (heatmap only)', layout=W,
        )
        self.txt_q_thr_min = Text(
            value='', placeholder='min (e.g. 1e-3)',
            layout=Layout(width='150px'),
        )
        self.txt_q_thr_max = Text(
            value='', placeholder='max (e.g. 1e-1)',
            layout=Layout(width='150px'),
        )
        self.txt_g_thr_min = Text(
            value='', placeholder='min (e.g. 1e-3)',
            layout=Layout(width='150px'),
        )
        self.txt_g_thr_max = Text(
            value='', placeholder='max (e.g. 1e-1)',
            layout=Layout(width='150px'),
        )
        self.threshold_status = HTML("")

    def _build_inter_qpu_widgets(self, W):
        """Construct the five Inter-QPU mode widgets.

        ``chk_inter_qpu`` and ``chk_keep_layout`` are real Checkboxes on
        BOTH primary and secondary — the first because ``_refresh``
        reads its value to toggle ``lbl_fig1`` visibility, the second
        because "keep layout on backend switch" applies on the secondary
        too.

        On the secondary (``is_secondary=True``) the three QPU-2
        controls are cheap stubs — never mounted in a secondary sidebar
        (no tertiary panel) and never wired. On the primary they are
        full controls, disabled until ``chk_inter_qpu`` is ticked.
        """
        self.chk_inter_qpu = Checkbox(
            value=False, description='🔗 ENABLE INTER-QPU', layout=W,
        )
        self.chk_keep_layout = Checkbox(
            value=False, description='🔒 Keep layout on backend switch',
            layout=W,
        )
        if self.is_secondary:
            # Stubs: never displayed, never observed.
            self.dd_backend_qpu2 = Dropdown(options=[])
            self.btn_connect_qpu2 = Button()
            self.lbl_qpu2_status = HTML("")
        else:
            self.dd_backend_qpu2 = Dropdown(
                options=[], description='QPU 2:', layout=W, disabled=True,
            )
            self.btn_connect_qpu2 = Button(
                description='🔌 Connect QPU 2', button_style='info',
                layout=W, disabled=True,
            )
            self.lbl_qpu2_status = HTML("")

    def _build_fidelity_widget(self):
        """LQS (Estimated Layout-Quality) HTML banner; content driven by
        ``setup_analysis_callbacks`` via ``_ext_fns['update_fidelity']``.
        """
        self.lbl_fidelity = HTML("")

    def _build_core_widgets(self, W, state):
        """Construct the 16 core widgets that drive the main plot.

        Mode / calibration / display:
          * ``dd_mode`` — Chemistry (LUCJ) vs General; seeds from
            ``state.active_ansatz`` set by mapper.py at connect.
          * ``dp_date`` — DatePicker for the calibration day.
          * ``txt_hw_hour`` — optional ``HH:MM`` intraday selector
            (``continuous_update=False`` so the API call only fires
            on Enter / blur, not per-keystroke).
          * ``view_mode`` — Design vs Error Map.

        Sizing / styling:
          * ``size_slider`` — qubit dot size.
          * ``gate_size_slider`` / ``gate_hl_w_slider`` — BoundedIntText
            (small ranges fit in the sidebar without slider clipping).
          * ``chk_link_gates`` / ``show_numbers`` / ``font_size_slider``.
          * ``global_text_color`` — text-overlay colour.
          * ``chk_line_only`` — line-only-no-fill toggle.
          * ``plot_w_slider`` / ``plot_h_slider`` — FigureWidget pixels.

        Multi-programming:
          * ``intra_qpu_mode`` — toggle.
          * ``num_circuits`` — 1..MAX_CIRCUITS, disabled until
            ``intra_qpu_mode`` is ticked.

        Pure construction — observers are wired in ``_run_pipeline``.
        """
        self.dd_mode = Dropdown(
            options=[('Chemistry (LUCJ)', 'lucj'), ('General Use', 'general')],
            value=state.active_ansatz or 'lucj', description='Layout:', layout=W,
        )
        self.dp_date = DatePicker(description='Calib Date:', layout=W)
        self.txt_hw_hour = Text(
            value='', placeholder='HH:MM (optional)',
            description='Hour:', layout=Layout(width='200px'),
            style={'description_width': '50px'},
            continuous_update=False,
        )

        self.view_mode = Dropdown(
            options=[('🎨 Design Mode', 'design'), ('🔥 Error Map', 'error')],
            value='design', layout=W,
        )
        self.size_slider = IntSlider(
            value=45, min=10, max=60, description='Qubit Size', layout=W,
        )
        # Small ranges (1-15, 1-20): BoundedIntText (~80px) instead of a
        # 400px slider whose right-side number was clipping in the sidebar.
        self.gate_size_slider = BoundedIntText(
            value=5, min=1, max=15, description='Gate Width',
            layout=Layout(width='200px'),
            style={'description_width': '110px'},
        )
        self.gate_hl_w_slider = BoundedIntText(
            value=6, min=1, max=20, description='Gate Width',
            layout=Layout(width='200px'),
            style={'description_width': '90px'},
        )
        self.chk_link_gates = Checkbox(
            value=True, description='🔗 Auto-Color Gates',
            layout=W, tooltip='',
        )
        self.show_numbers = Checkbox(
            value=True, description='Show Qubit IDs', layout=W, tooltip='',
        )
        self.font_size_slider = IntSlider(
            value=22, min=4, max=30, description='Font Size', layout=W,
        )
        self.global_text_color = ColorPicker(
            value='#000000', description='Text Color:',
            layout=Layout(width='200px'),
        )
        self.chk_line_only = Checkbox(
            value=False, description='🔲 No Fill',
            layout=W, tooltip='Draw gate highlights as outlines only (no fill)',
        )
        self.plot_w_slider = IntSlider(
            value=1700, min=600, max=4000, step=50,
            description='Fig Width', layout=W,
        )
        self.plot_h_slider = IntSlider(
            value=1650, min=600, max=4000, step=50,
            description='Fig Height', layout=W,
        )

        self.intra_qpu_mode = Checkbox(
            value=False, description='ENABLE INTRA-QPU', layout=W,
        )
        # Discrete 1-4 choice → numeric input is more direct than a slider.
        self.num_circuits = BoundedIntText(
            value=2, min=1, max=MAX_CIRCUITS,
            description='Num Sub-Circs',
            layout=Layout(width='220px'),
            style={'description_width': '120px'},
            disabled=True,
        )

    def _build_navigation_widgets(self):
        """Construct the four nav widgets at the top of the panel.

          * ``manual_tabs`` — empty Tab(), populated by
            ``_build_dynamic_ui`` with one tab per active circuit.
          * ``ancilla_tool`` — RadioButtons mapping label
            ("Add Ancilla C1") → circuit-index int (or None for View
            Only). (label, value) tuples let the click handler read
            ``.value`` directly — no UI-text parsing.
          * ``nav_bar`` — ToggleButtons over manual / auto / auto_lucj /
            export / error_inspect. 120px button width pins visual
            consistency across emoji widths.
          * ``content_box`` — empty VBox; ``.children`` swapped by
            ``_update_view`` on nav switch.
        """
        self.manual_tabs = Tab()
        self.ancilla_tool = RadioButtons(
            options=[('View Only', None)], value=None,
            description='Map Click:',
        )
        self.nav_bar = ToggleButtons(
            options=[
                ('✏️ Layout', 'manual'),
                ('🤖 Auto', 'auto_lucj'),
                ('📁 I/O', 'io'), ('🔍 Error', 'error_inspect'),
            ],
            value='manual', style={'button_width': '120px'},
        )
        self.content_box = VBox()

    def _build_inspector_widgets(self, W):
        """Construct the 15 LQS Inspector + Stability widgets.

        Two clusters mounted into the section cards from
        ``_build_section_cards``; observers wired in
        ``setup_analysis_callbacks``.

        Section A (🔍 Current calibration inspector, opt-in):
          chk_inspect_enable, dd_inspect_{qubit,gate} +
          lbl_inspect_{qubit,gate}_metrics, inspector_body_box (hidden
          by default; checkbox observer flips ``display='flex'``).

        Section B (⚠️ Stability check, opt-in): chk_stability_enable
          (14-day history auto-loads regardless so opting in later
          shows results immediately), rb_stability_window (7/14/30,
          default 14), dd_unstable_{qubit,gate} + metrics,
          lbl_stability_note, stability_{section,body}_box (same
          hidden/flip-on-checkbox semantics as Section A).
        """
        # Inspector cluster ───────────────────────────────────────
        self.chk_inspect_enable = Checkbox(
            value=False,
            description='🔍 Enable Current calibration inspector',
            indent=False, layout=W,
        )
        self.dd_inspect_qubit = Dropdown(
            options=[], description='Top qubits:', layout=W,
            style={'description_width': '90px'}, disabled=True,
        )
        self.lbl_inspect_qubit_metrics = HTML("")
        self.dd_inspect_gate = Dropdown(
            options=[], description='Top gates:', layout=W,
            style={'description_width': '90px'}, disabled=True,
        )
        self.lbl_inspect_gate_metrics = HTML("")
        # Hidden by default; the checkbox observer (wired in
        # ui.analysis_panel.setup_analysis_callbacks) flips display='flex'.
        self.inspector_body_box = VBox(
            [self.dd_inspect_qubit, self.lbl_inspect_qubit_metrics,
             self.dd_inspect_gate, self.lbl_inspect_gate_metrics],
            layout=Layout(display='none'),
        )

        # Stability cluster ───────────────────────────────────────
        self.chk_stability_enable = Checkbox(
            value=False,
            description='⚠️ Enable Stability check',
            indent=False, layout=W,
        )
        self.rb_stability_window = RadioButtons(
            options=[('7 days', 7), ('14 days', 14), ('30 days', 30)],
            value=14,
            description='Window:',
            layout=W,
            style={'description_width': '90px'},
        )
        self.dd_unstable_qubit = Dropdown(
            options=[], description='Unstable Q:', layout=W,
            style={'description_width': '90px'}, disabled=True,
        )
        self.lbl_unstable_qubit_metrics = HTML("")
        self.dd_unstable_gate = Dropdown(
            options=[], description='Unstable G:', layout=W,
            style={'description_width': '90px'}, disabled=True,
        )
        self.lbl_unstable_gate_metrics = HTML("")
        self.lbl_stability_note = HTML("")
        self.stability_section_box = VBox([
            self.rb_stability_window,
            self.dd_unstable_qubit, self.lbl_unstable_qubit_metrics,
            self.dd_unstable_gate, self.lbl_unstable_gate_metrics,
        ])
        # Wrap the note + section_box in a body container hidden by default;
        # the checkbox observer (in ui.analysis_panel) flips display='flex'.
        self.stability_body_box = VBox(
            [self.lbl_stability_note, self.stability_section_box],
            layout=Layout(display='none'),
        )

    def _build_ranking_widgets(self, W):
        """Construct the Qubit Ranking + Gate Ranking widgets.

        Two parallel 4-widget clusters wrapped in a single
        ``self._rank_accordion`` (collapsed by default).

        Qubit cluster (``dd_rank_prop``, ``dd_rank_order``,
        ``txt_rank_filter``, ``rank_table_html``):
          - ``Rank by:`` — property dropdown (default ``readout_error``).
          - ``Order:`` — ``Best first`` / ``Worst first``, encoded as
            ``True`` / ``False`` so the observer can pass it straight
            to ``sorted(..., reverse=not v)`` with no parsing.
          - ``Filter:`` Text — substring filter on qubit IDs.
          - HTML output table.

        Gate cluster — same shape with ``g``-prefixed names; uses
        ``GATE_PROP_OPTIONS`` (default ``cz_error``) and an edge-syntax
        filter placeholder ``(1,2)``.

        Observers wired in ``setup_analysis_callbacks``.
        """
        # Qubit Ranking cluster ───────────────────────────────────────
        self.dd_rank_prop = Dropdown(
            options=QUBIT_PROP_OPTIONS, value='readout_error',
            description='Rank by:', layout=W,
        )
        self.dd_rank_order = Dropdown(
            options=[('Best first', True), ('Worst first', False)],
            value=True, description='Order:',
            layout=Layout(width='200px'),
        )
        self.txt_rank_filter = Text(
            value='', placeholder='Filter (e.g. 47, 100)',
            description='Filter:', layout=W,
            style={'description_width': '50px'},
        )
        self.rank_table_html = HTML("")

        # Gate Ranking cluster ────────────────────────────────────────
        self.dd_grank_prop = Dropdown(
            options=GATE_PROP_OPTIONS, value='cz_error',
            description='Rank by:', layout=W,
        )
        self.dd_grank_order = Dropdown(
            options=[('Best first', True), ('Worst first', False)],
            value=True, description='Order:',
            layout=Layout(width='200px'),
        )
        self.txt_grank_filter = Text(
            value='', placeholder='Filter: qubit or edge e.g. (1,2)',
            description='Filter:', layout=W,
            style={'description_width': '50px'},
        )
        self.grank_table_html = HTML("")

        # Accordion wrapper that mounts both clusters as collapsed
        # sections in the sidebar Analysis tab.
        self._rank_accordion = Accordion(children=[
            VBox([
                HBox([self.dd_rank_prop, self.dd_rank_order]),
                self.txt_rank_filter,
                self.rank_table_html,
            ]),
            VBox([
                HBox([self.dd_grank_prop, self.dd_grank_order]),
                self.txt_grank_filter,
                self.grank_table_html,
            ]),
        ], selected_index=None)
        self._rank_accordion.set_title(0, '🏆 Qubit Ranking')
        self._rank_accordion.set_title(1, '🔗 Gate Ranking')

    def _build_trend_widgets(self, W):
        """Construct the Qubit Trend + Gate Trend chart widgets.

        Two parallel 4-widget clusters (qubit / gate) driving the
        time-series chart in the Analysis tab. Each: a comma-separated
        ID list Text input + property Dropdown + Plot Button + VBox
        chart container (filled with a Plotly FigureWidget by the
        observer on click).

        Both clusters require calibration history; the Plot button
        surfaces a "history required" hint when history is empty.
        Observers wired in ``setup_analysis_callbacks``.
        """
        # Qubit Trend cluster ─────────────────────────────────────────
        self.txt_trend_qubits = Text(
            value='', placeholder='e.g. 0, 5, 47, 100',
            description='Track qubits:', layout=W,
            style={'description_width': '80px'},
        )
        self.dd_trend_prop = Dropdown(
            options=QUBIT_PROP_OPTIONS, value='readout_error',
            description='Property:', layout=W,
        )
        self.btn_trend_plot = Button(
            description='📈 Plot Trend',
            button_style='info', layout=W,
        )
        self.trend_chart_box = VBox()

        # Gate Trend cluster ──────────────────────────────────────────
        self.txt_trend_gates = Text(
            value='', placeholder='e.g. (0,1), (5,6), (47,48)',
            description='Track gates:', layout=W,
            style={'description_width': '80px'},
        )
        self.dd_gtrend_prop = Dropdown(
            options=GATE_PROP_OPTIONS, value='cz_error',
            description='Property:', layout=W,
        )
        self.btn_gtrend_plot = Button(
            description='📈 Plot Gate Trend',
            button_style='info', layout=W,
        )
        self.gtrend_chart_box = VBox()

    def _build_stable_finder_widgets(self, W):
        """Construct the Stable Qubits + Stable Gates finder widgets.

        Two parallel 5-widget clusters. Each: property Dropdown,
        scientific-notation threshold Text, ``Min stab %`` IntSlider
        (0–100, default 80 — fraction of fetched snapshots that must
        satisfy the threshold), Find Button, HTML output table.

        The Text input's ``description`` flips dynamically between
        ``Max err`` (error rates / durations) and ``Min value``
        (coherence times T1/T2) — wired by ``_on_stable_prop_change``
        in ``ui.analysis_panel.py``. Default thresholds differ:
        ``5e-2`` (qubit, readout) vs ``1e-2`` (gate — noisier).
        """
        # Qubit Stable Finder cluster ─────────────────────────────────
        self.txt_stable_threshold = Text(
            value='5e-2', placeholder='e.g. 1e-3',
            description='Max err:', layout=Layout(width='200px'),
            style={'description_width': '70px'},
        )
        self.slider_stable_ratio = IntSlider(
            value=80, min=0, max=100, step=1,
            description='Min stab %:',
            layout=Layout(width='280px'),
            style={'description_width': 'initial'},
        )
        self.dd_stable_prop = Dropdown(
            options=QUBIT_PROP_OPTIONS, value='readout_error',
            description='Property:', layout=W,
        )
        self.btn_stable_find = Button(
            description='🔎 Find Stable Qubits',
            button_style='success', layout=W,
        )
        self.stable_html = HTML("")

        # Gate Stable Finder cluster ──────────────────────────────────
        self.dd_gstable_prop = Dropdown(
            options=GATE_PROP_OPTIONS, value='cz_error',
            description='Property:', layout=W,
        )
        self.txt_gstable_threshold = Text(
            value='1e-2', placeholder='e.g. 1e-3',
            description='Max err:', layout=Layout(width='200px'),
            style={'description_width': '70px'},
        )
        self.slider_gstable_ratio = IntSlider(
            value=80, min=0, max=100, step=1,
            description='Min stab %:',
            layout=Layout(width='280px'),
            style={'description_width': 'initial'},
        )
        self.btn_gstable_find = Button(
            description='🔎 Find Stable Gates',
            button_style='success', layout=W,
        )
        self.gstable_html = HTML("")

    def _build_qubit_inspector_widgets(self):
        """Construct the 3 Qubit Inspector widgets (Analysis tab).

          * ``self.txt_inspect_qubit`` — Text input (260px, label-width
            80px to fit 3-digit IDs on 156-qubit chips like Heron).
          * ``self.btn_inspect`` — ``🔍 Inspect`` Button.
          * ``self.lbl_inspect_result`` — HTML card populated by the
            click handler with qubit properties, connected gates, and
            (if calibration history is loaded) a mini trend chart.

        Click handler wired in ``setup_analysis_callbacks``.
        """
        # Qubit Inspector cluster ─────────────────────────────────────
        self.txt_inspect_qubit = Text(
            value='', placeholder='e.g. 5',
            description='Qubit:',
            layout=Layout(width='260px'),
            style={'description_width': '80px'},
        )
        self.btn_inspect = Button(
            description='🔍 Inspect', button_style='info',
            layout=Layout(width='140px'),
        )
        self.lbl_inspect_result = HTML("")

    def _build_calibration_history_widgets(self, W):
        """Construct the Multi-day calibration history fetch widgets.

        Six fetch widgets (``hist_start_date`` / ``hist_end_date``
        DatePickers, ``btn_fetch_hist``, ``hist_progress`` IntProgress
        hidden by default, ``hist_status`` HTML, ``hist_slider_box``
        VBox populated by ``ui.history`` after a successful fetch) plus
        ``self.btn_export_gif`` — the time-lapse trigger, constructed
        here (not in ``_build_timelapse_widgets``) because its
        disabled→enabled transition is driven by the same success path
        that populates the history (``ui.history.py`` flips
        ``disabled = False`` when ``state.cal_history_dates`` becomes
        non-empty). Mounted in the Export tab.

        Observer wiring lives in ``setup_history_callbacks``.
        """
        self.hist_start_date = DatePicker(description='From:', layout=W)
        self.hist_end_date = DatePicker(description='To:', layout=W)
        self.btn_fetch_hist = Button(
            description='📡 Fetch History',
            button_style='info', layout=W,
        )
        self.hist_progress = IntProgress(
            value=0, min=0, max=1, description='Loading:',
            layout=Layout(width='100%', display='none'),
        )
        self.hist_status = HTML("")
        # History player — populated dynamically by ui.history after
        # a successful fetch. Hidden until then.
        self.hist_slider_box = VBox(layout=Layout(display='none'))
        # Time-lapse export trigger — disabled until calibration
        # history is fetched (see method docstring for why this lives
        # in the history block).
        self.btn_export_gif = Button(
            description='🎬 Export Time-Lapse',
            button_style='warning',
            disabled=True,
            tooltip='Fetch calibration history first (Error tab → Multi-day)',
            layout=Layout(width='100%'),
        )

    def _build_timelapse_widgets(self, W):
        """Construct the three time-lapse export option widgets.

        Filename Text + format Dropdown (GIF / MP4, default GIF) + FPS
        IntSlider (1..10, default 2 — values above ~10 make per-frame
        charts illegible). Read by ``ui.history._on_export_gif`` via
        the shared ``_hist_ctx``; mounted in the Export tab BEFORE the
        ``🎬 Export Time-Lapse`` button so users pick options before
        clicking.
        """
        self.fname_timelapse = Text(
            value='timelapse', description='File name:',
            placeholder='filename or full path — default: current folder',
            layout=W,
            style={'description_width': 'initial'},
        )
        self.dd_timelapse_format = Dropdown(
            options=[('🎞️ GIF (animated, lighter)', 'gif'),
                     ('🎬 MP4 (smoother, smaller)', 'mp4')],
            value='gif', description='Format:', layout=W,
            style={'description_width': 'initial'},
        )
        self.slider_timelapse_fps = IntSlider(
            value=2, min=1, max=10, step=1, description='FPS:', layout=W,
            style={'description_width': 'initial'},
        )

    def _build_auto_lucj_widgets(self, W):
        """Construct the Auto-LUCJ layout-engine widgets (3 clusters).

        Single Circuit: lucj_norb_slider (BoundedIntText 2..80 — users
        target specific Hamiltonians, prefer numeric input over a
        slider), lucj_ancilla_dd (Auto=-1 or 0..5), lucj_chk_score
        (rank by error), btn_lucj_run, lbl_lucj_status.

        Inspect Layouts: dd_lucj_layout (ranked selector, disabled until
        populated — also drives cascading re-optimization for
        multi-circuit) + lbl_lucj_detail.

        Multi-Circuit: lucj_multi_norbs (Text '8, 8' default),
        lucj_buffer_slider (0..5 buffer qubits), btn_lucj_multi,
        lbl_lucj_multi, lucj_multi_detail_box (VBox populated per
        packed circuit).

        Mutable backing stores on ``self.*`` (mutated by
        ``setup_lucj_callbacks`` via ``.append()`` / ``.clear()`` —
        identity must remain stable across the builder's lifetime):
        ``_lucj_all_layouts``, ``_lucj_multi_results``,
        ``_lucj_multi_norbs`` (for cascading re-optimization on dropdown
        change), ``_lucj_multi_dropdowns``
        (``circuit_idx → (dropdown, label, candidates)``).

        Click handlers + cascading observer in
        ``setup_lucj_callbacks``. ``_load_device_graph`` resets the
        containers on backend switch.
        """
        # ── Single Circuit cluster ───────────────────────────────
        # Num Orbitals 2-80 wide range but the user wants a precise value
        # (e.g. 12 for a specific Hamiltonian) — numeric input, not a slider.
        self.lucj_norb_slider = BoundedIntText(
            value=4, min=2, max=80,
            description='Num Orbitals:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'})
        # 'Auto' runs the stratified ancilla sweep (default). Picking an
        # explicit 0..5 runs the legacy single-count search unchanged.
        self.lucj_ancilla_dd = Dropdown(
            options=[('Auto (sweep all counts)', -1), ('0 (no ancillas)', 0),
                     ('1 ancilla', 1), ('2 ancillas', 2), ('3 ancillas', 3),
                     ('4 ancillas', 4), ('5 ancillas', 5)],
            value=-1, description='Ancilla:', layout=W,
            style={'description_width': 'initial'},
            tooltip='Fix the ancilla count (legacy single search); '
                    'Auto runs the stratified sweep')
        # Sweep-only floor. 0 = auto → engine default min(3, k_max). Only
        # consulted when the Ancilla dropdown is on 'Auto'.
        self.lucj_min_ancilla = BoundedIntText(
            value=0, min=0, max=20,
            description='Min ancillas:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'},
            tooltip='Sweep all counts from max down to this floor '
                    '(0 = auto)')
        # How many candidates to keep/show per ancilla count. Wired to the
        # sweep's top_n through _sweep_kwargs_from_ui, so the scan inherits it.
        self.lucj_top_n = BoundedIntText(
            value=3, min=1, max=25,
            description='Show top:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'},
            tooltip='candidates shown per ancilla count')
        # Advanced: the VF2 mapping cap. Raise to trade time for
        # exhaustiveness (default matches auto_lucj._MAX_RAW_VF2_MAPPINGS).
        self.lucj_search_cap = BoundedIntText(
            value=2000, min=100, max=20000, step=500,
            description='Search cap:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'},
            tooltip='Max VF2 embeddings enumerated per search (default 2000). '
                    'When the cap truncates, the notices bar says so; raise it '
                    'for a more exhaustive (slower) search.')
        self.lucj_chk_score = Checkbox(
            value=True,
            description='Score layouts (pick lowest error)',
            # Its OWN Layout, not the shared `W` instance: lucj_panel toggles
            # .layout.display on this widget to hide it in sweep mode, and a
            # shared Layout would hide every other widget built with W.
            layout=Layout(width=W.width, margin=W.margin,
                          min_width=W.min_width))
        self.chk_deep_search = Checkbox(
            value=True,
            description='Deep search per ancilla count',
            layout=W,
            tooltip='adds a direct search per k — complete but slower; '
                    'off = fast derivation from k_max only')
        # Region diversity in the top-N slice. Default ON: it never costs
        # quality (ancilla_sweep._select_with_diversity keeps the first
        # ceil(top_n/2) slots as the absolute best) and it widens the choice.
        self.chk_lucj_diverse = Checkbox(
            value=True,
            description='Prefer diverse regions',
            layout=W,
            tooltip="Second half of each stratum's list favors layouts on "
                    "different chip regions (Jaccard ≤ 0.8). The top half "
                    "is always the absolute best — diversity never "
                    "demotes it.")
        # ── 📡 Benchmark widgets (mounted as sidebar section 5) ───
        # Never automatic: see iqmapper/benchmark.py. The target dropdown,
        # level dropdown, cost line and acknowledgement all exist so the QPU
        # cost is on screen before the button can be clicked.
        #
        # Benchmarking is a property of the BACKEND, not of the auto search,
        # so this lives in the sidebar rather than in the Auto tab: it has to
        # serve hand-placed layouts, every circuit slot, the whole QPU, and
        # the inter-QPU second panel.
        from .. import benchmark as _bench
        # Its OWN Layout: benchmark_panel toggles .layout.display on this to
        # hide the field unless Custom is selected, and the shared `W`
        # instance would hide every widget built with W.
        _own = lambda: Layout(width=W.width, margin=W.margin,
                              min_width=W.min_width)
        self.dd_bench_target = Dropdown(
            options=[(_bench.TARGET_LABELS[_bench.TARGET_CURRENT],
                      _bench.TARGET_CURRENT)],
            value=_bench.TARGET_CURRENT,
            description='Targets:',
            layout=Layout(width='280px'),
            style={'description_width': '110px'},
            tooltip='What to measure: the placed layout(s), the last sweep\'s '
                    'top candidates, the whole QPU, or an explicit qubit list.')
        _custom_layout = _own()
        _custom_layout.display = 'none'     # revealed only for Targets=Custom
        self.txt_bench_custom = Text(
            value='', description='Qubits:', placeholder='0, 1, 2, 5, 8',
            layout=_custom_layout, style={'description_width': '110px'})
        _qpu_layout = _own()
        _qpu_layout.display = 'none'        # revealed only when QPU 2 is up
        self.tb_bench_qpu = ToggleButtons(
            options=[('QPU 1', 1), ('QPU 2', 2)], value=1,
            layout=_qpu_layout)
        self.dd_bench_level = Dropdown(
            options=_bench.LEVEL_OPTIONS,
            value=_bench.LEVEL_QUICK_READOUT,
            description='Level:',
            layout=Layout(width='280px'),
            style={'description_width': '110px'},
            tooltip='1 = readout (2 circuits, cost does not grow with qubit '
                    'count); 2 = adds 2Q RB on the target edges; '
                    '3 = the full-chip tutorial suite (~4 min).')
        self.lbl_bench_summary = HTML("")
        self.chk_bench_ack = Checkbox(
            value=False,
            description='I understand this uses QPU time',
            layout=_own())
        self.btn_bench_run = Button(
            description='📡 Run benchmark',
            button_style='warning', layout=W, disabled=True)
        self.btn_bench_calcheck = Button(
            description='🔬 Calibration check',
            button_style='info', layout=W, disabled=True,
            tooltip='Measure a few edges twice — all pairs together, then one '
                    'at a time — to tell an estimator bias (both arms offset) '
                    'from crosstalk (only the parallel arm offset). Does not '
                    'change the calibration in use.')
        self.lbl_bench_calcheck = HTML("")
        self.lbl_bench_reuse = HTML("")
        self.btn_bench_reuse = Button(
            description='Reuse', button_style='info', disabled=True,
            layout=Layout(width='120px', margin='2px 0px'))
        _reanalyze_layout = Layout(width='200px', margin='2px 0px')
        _reanalyze_layout.display = 'none'   # only when a job record exists
        self.btn_bench_reanalyze = Button(
            description='Reanalyze last run', button_style='info',
            disabled=True, layout=_reanalyze_layout,
            tooltip='Re-run the analysis on the last submission\'s jobs. '
                    'Analysis is client-side, so this costs no QPU time.')
        self.chk_bench_absolute = Checkbox(
            value=False,
            description='Apply absolute values',
            layout=_own(),
            tooltip='Off (default): when the measured values show a '
                    'systematic offset against the reported calibration, '
                    'they are rescaled to the reported median so only the '
                    'relative differences are applied. On: use the raw '
                    'measured numbers.')
        self.lbl_bench_consistency = HTML("")
        self.lbl_bench_details = HTML("")
        _map_btn_layout = Layout(width='220px', margin='4px 0px')
        _map_btn_layout.display = 'none'      # shown only when edges qualify
        self.btn_bench_show_on_map = Button(
            description='show on map', button_style='',
            layout=_map_btn_layout,
            tooltip="Filter the Error tab to the edges more than twice "
                    "IBM's reported median, so the count above becomes "
                    "visible on the device map.")
        self.acc_bench_details = Accordion(
            children=[VBox([self.lbl_bench_details,
                            self.btn_bench_show_on_map])],
            layout=W)
        self.acc_bench_details.set_title(0, 'Details')
        self.acc_bench_details.selected_index = None   # collapsed by default
        self.lbl_bench_status = HTML("")
        self.bench_box = VBox([
            HTML("<div style='font-size:14px;color:#6c757d;"
                 "margin-bottom:4px;'>Re-measure the live device and re-rank "
                 "against what the hardware is doing now. Opt-in: nothing "
                 "runs until you click.</div>"),
            self.tb_bench_qpu,
            self.dd_bench_target,
            self.txt_bench_custom,
            self.dd_bench_level,
            self.lbl_bench_summary,
            self.chk_bench_ack,
            self.chk_bench_absolute,
            self.btn_bench_run,
            self.btn_bench_calcheck,
            self.lbl_bench_consistency,
            self.acc_bench_details,
            self.lbl_bench_calcheck,
            HBox([self.lbl_bench_reuse, self.btn_bench_reuse,
                  self.btn_bench_reanalyze],
                 layout=Layout(align_items='center')),
            self.lbl_bench_status,
        ], layout=W)

        _diverse_layout = Layout(width=W.width, margin=W.margin,
                                 min_width=W.min_width)
        self.chk_diverse_cap = Checkbox(
            value=True,
            description='Diverse enumeration',
            layout=_diverse_layout,
            tooltip='Make the search cap count DISTINCT candidates: during '
                    'enumeration a layout that overlaps a kept one by 80% '
                    'and does not score better is discarded without using a '
                    'slot. Engages only where the cap is actually reached — '
                    'the low-bridge (hybrid) passes; at the maximum bridge '
                    'count the search finds fewer layouts than the cap and '
                    'nothing is filtered. Off = the historical behaviour.')

        # ── Search engine (Advanced) ─────────────────────────────
        # Default stays Exhaustive (VF2) until iqmapper.evaluation says the
        # guided beam wins on real cached calibration — see the harness.
        self.dd_search_engine = Dropdown(
            options=[('Exhaustive (VF2)', 'vf2'), ('Guided (beam)', 'guided')],
            value='vf2',
            description='Search engine:',
            layout=Layout(width='280px'),
            style={'description_width': '110px'},
            tooltip='Exhaustive enumerates zigzag embeddings with VF2 and is '
                    'calibration-blind (capped at the search cap). Guided '
                    'grows chains using the calibration and polishes the '
                    'finalists; it is a heuristic, so it stays opt-in.')
        _guided_layout = Layout(width=W.width, margin=W.margin,
                                min_width=W.min_width)
        _guided_layout.display = 'none'      # shown only for Guided
        self.bench_guided_box = VBox([], layout=_guided_layout)
        self.int_beam_width = BoundedIntText(
            value=200, min=10, max=5000, step=50,
            description='Beam width:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'},
            tooltip='partial layouts kept at each growth step')
        self.int_refine_top = BoundedIntText(
            value=5, min=0, max=50,
            description='Refine top:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'},
            tooltip='how many finalists per ancilla count get the 1-opt polish')
        self.bench_guided_box.children = [self.int_beam_width,
                                          self.int_refine_top]

        # Expert knobs, collapsed out of the primary flow (ancilla dropdown
        # -> Run). The defaults are right for nearly everyone, and the notices
        # bar already reports when the cap truncated a search.
        self.lucj_advanced_acc = Accordion(
            children=[VBox([self.lucj_search_cap, self.chk_deep_search,
                            self.chk_diverse_cap,
                            self.dd_search_engine, self.bench_guided_box])],
            layout=W)
        self.lucj_advanced_acc.set_title(0, '⚙️ Advanced')
        self.lucj_advanced_acc.selected_index = None   # collapsed on open
        self.btn_lucj_run = Button(
            description='🤖 Generate LUCJ Layout',
            button_style='success', layout=W)
        self.lbl_lucj_status = HTML("")
        # Sweep truncation banner (VF2 cap hit); empty/hidden otherwise.
        self.lbl_lucj_trunc = HTML("")
        # Captured WARNING+ from the local sweep (loud notices bar); empty when
        # the sweep emitted nothing.
        self.lbl_lucj_notices = HTML("")

        # ── Stability filter (opt-in) ────────────────────────────
        self.chk_stability_filter = Checkbox(
            value=False, description='🛡 Stability filter', layout=W,
            tooltip='Exclude volatile qubits (from calibration history) '
                    'before the sweep')
        self.rb_stability_window_auto = RadioButtons(
            options=[('7 days', 7), ('14 days', 14), ('30 days', 30)],
            value=14, description='Window:', layout=W,
            style={'description_width': '70px'})
        self.ft_stability_threshold = BoundedFloatText(
            value=0.5, min=0.0, max=100.0, step=0.1,
            description='Threshold:', layout=Layout(width='230px'),
            style={'description_width': '90px'},
            tooltip='Volatility score = max/median − 1; higher threshold = '
                    'more tolerant (fewer qubits excluded)')
        # Quality floor: always-on factory guard — exclude qubits whose CURRENT
        # readout error is a device outlier (> σ above the mean), from EVERY
        # search path (single / multi / scan), independent of the volatility
        # filter. Defined here but mounted in the always-visible Auto section.
        self.ft_stability_floor = BoundedFloatText(
            value=3.0, min=0.0, max=20.0, step=0.5,
            description='Quality floor:', layout=Layout(width='230px'),
            style={'description_width': '110px'},
            tooltip='Always on: exclude qubits whose CURRENT readout error is '
                    'more than this many σ above the device mean (0 = off). '
                    'Applies to single, multi and scan searches.')
        # Window + threshold row, revealed only when the volatility filter is on.
        self.stability_ctrl_box = VBox(
            [self.rb_stability_window_auto, self.ft_stability_threshold])
        self.stability_ctrl_box.layout.display = 'none'
        self.lbl_lucj_stability = HTML("")
        # Scan sub-option: filter each scanned backend with CACHED history only.
        self.chk_scan_stability = Checkbox(
            value=False,
            description='apply stability filter (cached history only)',
            layout=W,
            tooltip='Filter each scanned backend using its CACHED history '
                    'only — no remote history fetch in this version')
        self.btn_scan_clear = Button(
            description='✕ Clear results', button_style='',
            layout=Layout(width='130px'),
            tooltip='Dismiss the scan results table, dropdowns and notices')

        # ── Multi-QPU scan cluster ───────────────────────────────
        self.btn_scan_qpus = Button(
            description='🌐 Scan all QPUs', button_style='info', layout=W,
            tooltip='Run the ancilla sweep on every compatible account '
                    'backend (same norb / min-ancillas as above)')
        self.chk_scan_other_families = Checkbox(
            value=False, description='include other families', layout=W,
            tooltip='Scan every topology family, not just the current one '
                    '(cross-family zigzag comparison is apples-to-oranges)')
        self.lbl_scan_progress = HTML("")
        self.scan_results_box = VBox([])

        # ── Inspect Layouts cluster ──────────────────────────────
        self.dd_lucj_layout = Dropdown(
            options=[], description='Layout:', layout=W,
            style={'description_width': '50px'}, disabled=True)
        self.lbl_lucj_detail = HTML("")

        # Mutable container for all computed single-circuit layouts.
        self._lucj_all_layouts = []
        self._lucj_multi_results = []
        self._lucj_multi_norbs = []
        self._lucj_multi_ancillas = []
        self._lucj_multi_dropdowns = {}

        # ── Multi-Circuit (intra-QPU multi-programming) cluster ──
        self.lucj_multi_norbs = Text(
            value='8, 8', description='norb per circuit:',
            placeholder='e.g. 8, 8 or 10, 10, 6',
            layout=W, style={'description_width': 'initial'})
        # Per-circuit MIN ancillas (bridge floor) for the sweep, parallel to
        # norb. Blank → auto per circuit; otherwise one value per circuit.
        self.lucj_multi_ancillas = Text(
            value='', description='min ancillas per circuit:',
            placeholder='blank = auto; or e.g. 2, 2',
            layout=W, style={'description_width': 'initial'},
            tooltip='min ancillas per circuit (blank = auto)')
        # Buffer qubits: tiny range 0-5, numeric input is more direct.
        self.lucj_buffer_slider = BoundedIntText(
            value=1, min=0, max=5,
            description='Buffer qubits:',
            layout=Layout(width='220px'),
            style={'description_width': '110px'})
        # Legacy multi-circuit button — no longer shown in the Multi subpanel
        # (the sweep is the single generate action). The widget + its handler
        # (_on_lucj_multi) and engine (get_multi_circuit_layouts) stay wired and
        # test-pinned; it is simply omitted from the layout below.
        self.btn_lucj_multi = Button(
            description='🤖 Generate Multi-Circuit',
            button_style='success', layout=W)
        # Sweep-based multi-circuit (the ONLY multi generate action in the UI):
        # greedy auto-best per circuit with a per-circuit override dropdown.
        self.btn_lucj_multi_sweep = Button(
            description='🧺 Generate Multi-Circuit',
            button_style='info', layout=W)
        # Batch scan: run the whole multi-circuit batch on every compatible QPU.
        self.btn_scan_batch = Button(
            description='🌐 Scan batch on all QPUs',
            button_style='info', layout=W,
            tooltip='Place the full multi-circuit batch on every eligible '
                    'backend and rank by batch fidelity')
        self.scan_batch_box = VBox([])
        self.lbl_lucj_multi = HTML("")
        # Captured WARNING+ from the multi-circuit sweep + replan (loud notices
        # bar); empty when nothing was emitted.
        self.lbl_lucj_multi_notices = HTML("")
        # Populated dynamically with per-circuit dropdowns when the
        # user clicks btn_lucj_multi.
        self.lucj_multi_detail_box = VBox()

    # Roles offered by the legend panel — must match plot._LEGEND_ROLE_ORDER.
    _LEGEND_ROLES = ('alpha', 'beta', 'other', 'qubits', 'ancilla', 'unselected')

    def _build_legend_widgets(self, W):
        """Construct the Legend control group (master + per-role rows).
        Observers write back into ``state.legend_config`` and call
        ``_refresh`` so the plot updates live. Per-row visibility flips
        with ``dd_mode`` (LUCJ hides 'qubits'; general hides α/β/other).
        """
        cfg = self.state.legend_config
        self.chk_legend_enabled = Checkbox(
            value=bool(cfg.get('enabled', True)),
            description='Show legend', layout=W,
        )
        self.legend_chk = {}
        self.legend_lbl = {}
        self.legend_rows = {}
        for role in self._LEGEND_ROLES:
            role_cfg = cfg['roles'].get(role, {})
            chk = Checkbox(
                value=bool(role_cfg.get('show', False)),
                description='', indent=False,
                layout=Layout(width='30px'),
            )
            lbl = Text(
                value=str(role_cfg.get('label', role)),
                description=role + ':',
                layout=Layout(width='220px'),
                style={'description_width': '80px'},
            )
            self.legend_chk[role] = chk
            self.legend_lbl[role] = lbl
            self.legend_rows[role] = HBox([chk, lbl], layout=Layout(margin='1px 0'))

        # ── Observers ────────────────────────────────────────────────
        def _on_master(change):
            self.state.legend_config['enabled'] = bool(change['new'])
            self._refresh()
        self.chk_legend_enabled.observe(_on_master, 'value')

        def _make_chk_obs(role):
            def _on(change):
                self.state.legend_config['roles'][role]['show'] = bool(change['new'])
                self._refresh()
            return _on

        def _make_lbl_obs(role):
            def _on(change):
                self.state.legend_config['roles'][role]['label'] = str(change['new'])
                self._refresh()
            return _on

        for role in self._LEGEND_ROLES:
            self.legend_chk[role].observe(_make_chk_obs(role), 'value')
            self.legend_lbl[role].observe(_make_lbl_obs(role), 'value')

    def _legend_panel(self):
        """Return the VBox mounted in the manual view. Row visibility
        depends on the current ansatz: LUCJ hides 'qubits';
        general hides 'alpha'/'beta'/'other'. 'ancilla' and 'unselected'
        are visible in both modes."""
        is_lucj = self.dd_mode.value == 'lucj'
        visible_by_mode = {
            'alpha':      is_lucj,
            'beta':       is_lucj,
            'other':      is_lucj,
            'qubits':     not is_lucj,
            'ancilla':    True,
            'unselected': True,
        }
        for role in self._LEGEND_ROLES:
            self.legend_rows[role].layout.display = (
                'flex' if visible_by_mode[role] else 'none'
            )
        rows = [self.legend_rows[r] for r in self._LEGEND_ROLES]
        controls = VBox([self.chk_legend_enabled, *rows])
        # Collapse into an Accordion so the controls don't clutter the
        # sidebar by default. Accordion provides its own header so the
        # inline "🏷️ Legend" HTML title is no longer needed.
        accordion = Accordion(children=[controls])
        accordion.set_title(0, '🏷️ Legend')
        accordion.selected_index = None   # collapsed on open
        return accordion

    def _build_undo_redo_widgets(self):
        """Layout undo/redo buttons (both disabled at build — stacks
        empty). ``_update_undo_btns`` (called by ``_refresh``) keeps
        them in sync with ``state.has_undo`` / ``state.has_redo``.
        Click handlers wired in ``_run_pipeline``.
        """
        btn_layout = Layout(width='100px', height='30px')
        self.btn_undo = Button(
            description='↩ Undo', button_style='',
            layout=btn_layout, disabled=True)
        self.btn_redo = Button(
            description='↪ Redo', button_style='',
            layout=btn_layout, disabled=True)

    def _build_backend_switcher_widgets(self, W):
        """Construct the backend switcher dropdown + 3 status banners.

          * ``dd_backend_switch`` — backends-of-current-vendor selector;
            options populated by ``setup_backend_switch`` on first
            refresh, shows ``'(loading...)'`` until then. Change
            triggers a full calibration refetch.
          * ``lbl_switch_status`` — switch-in-progress / error messages.
          * ``lbl_calibration_date`` — date + optional hour banner;
            updated by ``self._refresh``.
          * ``lbl_props_warning`` — prominent banner for missing/stale
            backend props; also updated by ``self._refresh``.
        """
        self.dd_backend_switch = Dropdown(
            options=['(loading...)'], value='(loading...)',
            description='Backend:', layout=W, disabled=True)
        self.lbl_switch_status = HTML("")
        self.lbl_calibration_date = HTML("")
        self.lbl_props_warning = HTML("")

    def _build_zoom_widgets(self):
        """Three 60×32px zoom buttons: btn_zoom_in (do_zoom 0.65 —
        factor<1 shrinks range = zooms in), btn_zoom_out (1.5),
        btn_zoom_rst (reset_zoom, full extents). Click handlers wired
        as lambdas in ``_run_pipeline``.
        """
        zoom_layout = Layout(width='60px', height='32px')
        self.btn_zoom_in = Button(
            description='🔍+', tooltip='Zoom in', layout=zoom_layout)
        self.btn_zoom_out = Button(
            description='🔍-', tooltip='Zoom out', layout=zoom_layout)
        self.btn_zoom_rst = Button(
            description='⊙', tooltip='Reset zoom', layout=zoom_layout)

    def _build_property_dropdowns(self, W):
        """Construct property selector dropdowns + colorscale preview.

        Five widgets — three of them control the Error map
        rendering (which qubit / gate calibration property is shown,
        and the palette), one drives the JSON / GIF export layout,
        and one is a passive HTML swatch that previews the active
        colorscale.

          * ``self.export_format`` — Dropdown over the two layout
            shapes used by the JSON / Qiskit export (Split per group
            vs Merged single list). Read by ``self._get_export`` and by
            ``export_mod.get_export_data``. No ``description=`` so
            the white control flushes against the left edge of the
            Export panel, matching PNG / Session / Style controls.
          * ``self.dd_qubit_prop`` — Dropdown over ``QUBIT_PROP_OPTIONS``,
            default ``'readout_error'``. Drives the qubit-side heatmap
            in the Error tab. Read by ``_update_threshold_status`` and
            by ``plot_mod.refresh_plot`` via the widgets_ref bundle.
          * ``self.dd_gate_prop`` — Dropdown over ``GATE_PROP_OPTIONS``,
            default ``'cz_error'``. Same shape as ``dd_qubit_prop`` but
            for the gate-side heatmap.
          * ``self.dd_colorscale`` — Dropdown over
            ``COLORSCALE_OPTIONS.keys()``, default ``'Blue→Purple
            (IBM)'``. Read by ``_on_colorscale_change`` and by
            ``state.current_colorscale`` mirror.
          * ``self.colorscale_preview`` — HTML widget. Initial value
            empty; populated by ``_make_cs_preview()`` immediately
            below the Error map panel section to show a small
            gradient swatch matching ``dd_colorscale.value``.

        Observers wired further down: ``_on_colorscale_change`` on
        ``dd_colorscale``, ``self._refresh`` on ``dd_qubit_prop`` and
        ``dd_gate_prop``. ``export_format`` has no observer — it's
        read on-demand by ``self._get_export`` and by the WidgetsRef-
        independent ``setup_session_io`` / ``setup_analysis_callbacks``
        ctx kwargs.
        """
        self.export_format = Dropdown(
            options=['Split (Alpha/Beta/...)', 'Merged (Single List)'],
            value='Merged (Single List)', layout=W,
        )
        self.dd_qubit_prop = Dropdown(
            options=QUBIT_PROP_OPTIONS, value='readout_error',
            description='Qubit Prop:', layout=W)
        self.dd_gate_prop = Dropdown(
            options=GATE_PROP_OPTIONS, value='cz_error',
            description='Gate Prop:', layout=W)
        self.dd_colorscale = Dropdown(
            options=list(COLORSCALE_OPTIONS.keys()),
            value='Blue→Purple (IBM)',
            description='Color Scale:', layout=W)
        self.colorscale_preview = HTML()

    def _build_session_io_widgets(self, W):
        """Construct the 4 session save/load widgets.

        Save side mounted in Export tab, load side in Auto tab. All
        callback wiring lives in ``setup_session_io``.
        ``txt_session_path`` uses no ``description=`` so the input
        flushes against the left edge of the panel (aligned with the
        💾 / 🎨 section emoji).
        """
        self.txt_session_path = Text(
            value='./my_session.qpusession.json', layout=W,
            placeholder='filename or full path — default: current folder')
        self.btn_save_session = Button(
            description='📤 Export Session',
            button_style='primary', layout=W)
        self.btn_load_session = FileUpload(
            accept='.json', multiple=False,
            description='📂 Load Session', layout=W)
        self.lbl_session_status = HTML("")
        self.lbl_session_save_status = HTML("")
        self.lbl_json_status = HTML("")
        self.lbl_png_status = HTML("")
        self.lbl_style_status = HTML("")
        self.lbl_timelapse_status = HTML("")

    def _build_auto_input_widgets(self, W):
        """Construct the General-mode auto-mapping widgets.

        Mounted in the Auto tab. ``txt_auto`` is a **Textarea** (not
        Text — multi-line, one line per circuit); ``btn_auto`` uses
        warning style to flag that each click pushes undo and replaces
        the previous layout. Click handler in ``_on_auto_map``.
        """
        self.txt_auto = Textarea(
            placeholder='Line 1 -> C1...\nLine 2 -> C2...',
            description='Input:', layout=W)
        self.btn_auto = Button(
            description='🚀 Load & Map',
            button_style='warning', layout=W)

    def _build_style_preset_widgets(self, W):
        """Construct the 3 style-preset save/load widgets.

        Save side in Export tab, load side in Auto tab. All callback
        wiring lives in ``setup_session_io`` alongside the session
        save/load handlers.
        """
        self.txt_style_path = Text(
            value='./my_style.qpustyle.json', layout=W,
            placeholder='filename or full path — default: current folder')
        self.btn_save_style = Button(
            description='📤 Export Style',
            button_style='primary', layout=W)
        self.btn_load_style = FileUpload(
            accept='.json', multiple=False,
            description='📂 Load Style', layout=W)

    def _build_export_widgets(self, W):
        """Construct the JSON/PNG export inputs + live JSON preview.

        Mounted in the Export tab. The PNG popup VBox is built later
        by ``setup_png_export(...)`` and stored on ``self.png_popup``;
        ``btn_save_img`` is passed into that setup call so its click
        opens the popup. ``json_preview`` is an HTML widget (not
        Textarea — flat JSON is unreadable for LUCJ layouts with
        alpha/beta/ancilla blocks) populated by ``self._get_export``
        via ``_pretty_json_html``.
        """
        self.fname_json = Text(
            value='layout_data', layout=W,
            placeholder='filename or full path — default: current folder')
        self.btn_save_json = Button(
            description='📤 Export JSON',
            button_style='primary', layout=W)
        self.fname_img = Text(
            value='layout_map', layout=W,
            placeholder='filename or full path — default: current folder')
        self.btn_save_img = Button(
            description='📤 Export PNG (HQ)',
            button_style='primary', layout=W)
        self.json_preview = HTML("")
        # JSON preview starts collapsed; the open/closed choice persists for
        # the session (the Export subpanel is rebuilt on every nav switch).
        self._json_preview_open = False

        # Shared "Save folder" for every export (PNG / JSON / session /
        # style / GIF/MP4). Plain Text field — user types a path
        # (absolute / `~` / relative). Default = CWD. A bare filename
        # typed into a name field below resolves against this folder.
        self.save_folder = Text(
            value=os.getcwd(),
            description='Save folder:',
            placeholder='folder to save into (e.g. ~/Desktop)',
            layout=Layout(width='99%'),
            style={'description_width': '110px'},
        )

    def _save_dir(self):
        """Current export folder. Reads the text field; empty → CWD.
        ``~`` is expanded; the resolver creates parent dirs as needed."""
        val = (self.save_folder.value or '').strip()
        return os.path.expanduser(val) if val else os.getcwd()

    def _user_save_path_in_save_dir(self, p, ext=''):
        """Resolve a user-typed export path against the session save dir.
        Single shared binding — panels must use this instead of re-deriving
        it locally (local copies have historically diverged)."""
        return _user_save_path(p, ext, default_dir=self._save_dir())

    def _build_manual_view(self):
        """Build the 'Layout' nav tab view: the per-circuit manual editor
        (``self.manual_tabs``) with the GLOBAL Circuit Drawing controls and
        the legend stacked below it. Circuit Drawing applies to ALL circuits,
        so it sits full-width BELOW the per-circuit tabs — not in a column
        beside a single tab — so extra circuit panels (multi-programming)
        won't collide with it.
        """
        return VBox([
            self.manual_tabs,
            HTML("<hr style='margin:6px 0;'>"),
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;"
                 "margin:2px 0;'>🖌️ Circuit Drawing</div>"),
            HBox([self.gate_hl_w_slider, self.chk_line_only],
                 layout=Layout(align_items='center', flex_flow='row wrap')),
            HTML("<hr style='margin:6px 0;'>"),
            self._legend_panel(),
        ])

    def _build_auto_view(self):
        """Build the 'Load'/Import nav tab view (General Ansatz mode): four
        load-from-outside sections — paste auto-map, layout-file upload,
        session load, style preset load. In LUCJ mode the same nav slot
        selects ``_build_auto_lucj_view`` via
        ``nav_bar.value == 'auto_lucj'``.
        """
        return VBox([
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;margin:3px 0;'>✏️ Paste qubit list</div>"),
            self.txt_auto, self.btn_auto,
            HTML("<hr style='margin:4px 0;'>"),
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;margin:3px 0;'>📂 Load layout file</div>"),
            self.btn_unified_upload,
            HTML("<hr style='margin:6px 0; border-color:#dee2e6;'>"),
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;margin:3px 0;'>💾 Load Session</div>"),
            self.btn_load_session,
            self.lbl_session_status,
            HTML("<hr style='margin:4px 0; border-color:#eee;'>"),
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;margin:3px 0;'>🎨 Load Style Preset</div>"),
            self.btn_load_style,
        ])

    def _refresh_stability_filter_availability(self):
        """Enable the 🛡 Stability filter only when calibration history is
        loaded; otherwise disable it with a hint. Re-evaluated each time the
        Auto view is (re)built, so it flips on after history is fetched."""
        state = self.state
        available = bool(getattr(state, 'cal_history', None)
                         and getattr(state, 'cal_history_dates', None))
        self.chk_stability_filter.disabled = not available
        if available:
            self.chk_stability_filter.tooltip = (
                'Exclude volatile qubits (from calibration history) before '
                'the sweep')
        else:
            self.chk_stability_filter.tooltip = (
                'Fetch calibration history first (Stability Check / Multi-day)')

    @staticmethod
    def _restore_tab_index(prev, tabs):
        """Carry a Tab's ``selected_index`` across a from-scratch rebuild.

        Every structural refresh (num_circuits / dd_mode / intra_qpu_mode
        observers, ``_toggle_intra_qpu``, an explicit ``update_view`` from the
        multi Generate flow) funnels through the ``_build_*_view`` methods,
        each of which mints a FRESH ``Tab`` whose ``selected_index`` defaults to
        0 — bouncing the user off whatever subpanel they were on (e.g. Multi →
        Single mid-Generate). Restoring the previous index at this single choke
        point fixes every current and future trigger. Clamped to the new child
        count so a shrink (e.g. intra_qpu OFF drops the Multi subpanel) can't
        point past the end. First build (prev is None) keeps the default 0."""
        prev_idx = getattr(prev, 'selected_index', None)
        n = len(tabs.children)
        if prev_idx is not None and n:
            tabs.selected_index = min(prev_idx, n - 1)
        return tabs

    def _build_auto_lucj_view(self):
        """Build the 'Auto' nav tab view (LUCJ chemistry mode).

        Three clusters (Single Circuit / Inspect Layouts /
        Multi-Circuit) built from ``_build_auto_lucj_widgets``.

        ``self.lucj_ancilla_dd`` is intentionally NOT remounted in
        the Multi-Circuit section: ipywidgets enforces single
        parenthood per widget. The Single Circuit dropdown sets the
        default ancilla policy; the Multi-Circuit section adds
        ``lucj_multi_ancillas`` to override it per sub-circuit (blank
        falls back to the dropdown).
        """
        self._refresh_stability_filter_availability()
        prev_tabs = getattr(self, '_auto_lucj_tabs', None)
        name = esc(str(getattr(self.state, 'active_backend_name', '')
                       or 'current backend'))
        # Subpanel 1 — "Single" (single-circuit): everything except the
        # Multi-Circuit section (single-circuit controls + the local results +
        # the scan block). Mirrors the Error tab's Tab-based subpanel switcher.
        single_panel = VBox([
            self.lucj_norb_slider,
            self.lucj_ancilla_dd,
            self.lucj_min_ancilla,
            self.lucj_top_n,
            self.lucj_chk_score,
            self.chk_lucj_diverse,
            self.ft_stability_floor,
            self.chk_stability_filter,
            self.stability_ctrl_box,
            self.lbl_lucj_stability,
            self.btn_lucj_run,
            self.lbl_lucj_trunc,
            self.lbl_lucj_status,
            self.lbl_lucj_notices,
            self.lucj_advanced_acc,
            HTML("<hr style='margin:8px 0; border-color:#2980B9;'>"),
            HTML(f"<div style='font-size:14px;font-weight:bold;color:#1a5276;"
                 f"margin:3px 0;'>📍 This backend ({name})</div>"),
            self.dd_lucj_layout,
            self.lbl_lucj_detail,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            HBox([
                HTML("<div style='font-size:14px;font-weight:bold;"
                     "color:#1a5276;margin:3px 8px 3px 0;'>🌐 Scan All "
                     "QPUs (other backends)</div>"),
                self.btn_scan_clear,
            ], layout=Layout(align_items='center')),
            self.btn_scan_qpus,
            self.chk_scan_other_families,
            self.chk_scan_stability,
            self.lbl_scan_progress,
            self.scan_results_box,
        ], layout=Layout(padding='6px'))
        # Subpanel 2 — "Multi" (multi-circuit): revealed only while Multi-
        # Programming (intra-QPU) is enabled; the enable checkbox rebuilds it.
        multi_panel = VBox([
            self.lucj_multi_norbs,
            self.lucj_multi_ancillas,
            self.lucj_buffer_slider,
            # Legacy btn_lucj_multi intentionally omitted — the sweep button is
            # the single generate action. Handler/engine remain wired + pinned.
            self.btn_lucj_multi_sweep,
            self.btn_scan_batch,
            self.lbl_lucj_multi,
            self.lbl_lucj_multi_notices,
            self.lucj_multi_detail_box,
            self.scan_batch_box,
        ], layout=Layout(padding='6px'))
        self._auto_single_panel = single_panel
        self._auto_multi_panel = multi_panel
        if self.intra_qpu_mode.value:
            tabs = Tab(children=[single_panel, multi_panel])
            tabs.set_title(0, '🔬 Single')
            tabs.set_title(1, '🔗 Multi')
        else:
            tabs = Tab(children=[single_panel])
            tabs.set_title(0, '🔬 Single')
        # Keep the user on the subpanel they were viewing across rebuilds.
        self._auto_lucj_tabs = self._restore_tab_index(prev_tabs, tabs)
        return VBox([
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;"
                 "margin:4px 0;'>🤖 Auto-LUCJ Layout Engine</div>"),
            HTML("<hr style='margin:6px 0; border-color:#dee2e6;'>"),
            tabs,
        ], layout=Layout(padding='6px'))

    def _on_json_preview_toggle(self, change):
        """Persist the JSON-preview open/closed choice; refresh on expand."""
        self._json_preview_open = change['new'] is not None
        if self._json_preview_open:
            self._get_export()

    def _build_import_export_view(self):
        """The unified '📁 I/O' nav tab: an Import subpanel (the former '📂 Load'
        view) first, then an Export subpanel, switched via a Tab — the same
        pattern as the Error / Auto tabs. Opens on Import by default."""
        prev_tabs = getattr(self, '_io_tabs', None)
        io_tabs = Tab(children=[self._build_auto_view(),
                                self._build_export_view()])
        io_tabs.set_title(0, '📥 Import')
        io_tabs.set_title(1, '📤 Export')
        # Same class of bug as the Auto tab: a refresh while the user is on a
        # non-default subpanel would otherwise snap back to index 0.
        self._io_tabs = self._restore_tab_index(prev_tabs, io_tabs)
        return io_tabs

    def _build_export_view(self):
        """Build the Export subpanel: JSON preview (collapsible), Export Layout
        (JSON), Export PNG, Export Session, Export Style Preset, Export
        Time-Lapse.
        """
        preview = Accordion(children=[self.json_preview])
        preview.set_title(0, 'JSON preview')
        preview.selected_index = 0 if self._json_preview_open else None
        preview.observe(self._on_json_preview_toggle, names='selected_index')
        return VBox([
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📁 Save folder (applies to every export below)</div>"),
            self.save_folder,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            preview,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📤 Export Layout</div>"),
            self.export_format,
            HBox([self.fname_json, self.btn_save_json]),
            self.lbl_json_status,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📤 Export PNG</div>"),
            HBox([self.fname_img, self.btn_save_img]),
            self.png_popup,
            self.lbl_png_status,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📤 Export Session</div>"),
            HBox([self.txt_session_path, self.btn_save_session]),
            self.lbl_session_save_status,
            HTML("<hr style='margin:4px 0; border-color:#eee;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📤 Export Style Preset</div>"),
            HBox([self.txt_style_path, self.btn_save_style]),
            self.lbl_style_status,
            HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📤 Export Time-Lapse</div>"),
            HTML("<div style='font-size:13px;color:#1a5276;font-weight:bold;margin:4px 0 2px 0;'>🎬 Time-Lapse output</div>"),
            self.fname_timelapse,
            self.dd_timelapse_format,
            self.slider_timelapse_fps,
            self.btn_export_gif,
            self.lbl_timelapse_status,
        ])

    def _build_error_view(self):
        """Build the 'Error' nav tab view (calibration map + analysis).

        Returns a ``Tab`` widget with two sub-tabs:

          * **Calibration** — Property & Scale row, Threshold
            Filter, and the Calibration section anchored on the hoisted
            ``self._time_panel`` (whose slider state survives nav /
            num_circuits / mode changes).
          * **Analysis** — collapsible Qubit Inspector accordion,
            Ranking accordion, Qubit + Gate Trend rows, Stable
            Qubits + Stable Gates rows.

        Containers rebuilt on every nav switch — cheap (no widget
        allocation), and keeps the Qubit Inspector accordion in its
        collapsed default state on re-entry.
        """
        _err_tab_map = VBox([
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📡 Property & Scale</div>"),
            HBox([self.dd_qubit_prop, self.dd_gate_prop]),
            HBox([self.dd_colorscale, self.colorscale_preview]),
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>🔍 Focus</div>"),
            self.chk_focus_selected,
            self.chk_isolate,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>🎯 Threshold Filter</div>"),
            self.chk_err_threshold,
            HTML("<div style='font-size:14px;color:#666;margin:1px 0;'>Qubit range (dim outside):</div>"),
            HBox([Label('Min', layout=Layout(width='30px')), self.txt_q_thr_min,
                  Label('Max', layout=Layout(width='30px')), self.txt_q_thr_max]),
            HTML("<div style='font-size:14px;color:#666;margin:1px 0;'>Gate range (dim outside):</div>"),
            HBox([Label('Min', layout=Layout(width='30px')), self.txt_g_thr_min,
                  Label('Max', layout=Layout(width='30px')), self.txt_g_thr_max]),
            self.threshold_status,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>⏱️ Calibration</div>"),
            self._time_panel,
            HTML("<hr style='margin:6px 0; border-color:#dee2e6;'>"),
            self.lbl_calibration_date,
            self.lbl_props_warning,
        ])

        # Qubit inspector in a collapsible Accordion (collapsed by
        # default, selected_index=None) so the analysis tab isn't
        # dominated by it when unused. Result label lives inside the
        # Accordion so closing it also hides stale output.
        _inspector_body = VBox([
            HBox([self.txt_inspect_qubit, self.btn_inspect]),
            self.lbl_inspect_result,
        ])
        _inspector_acc = Accordion(children=[_inspector_body])
        _inspector_acc.set_title(0, '🔍 Qubit Inspector')
        _inspector_acc.selected_index = None   # collapsed on open

        _err_tab_analysis = VBox([
            _inspector_acc,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            self._rank_accordion,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📈 Qubit Trend (requires history)</div>"),
            HBox([self.txt_trend_qubits, self.dd_trend_prop]),
            self.btn_trend_plot,
            self.trend_chart_box,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>📈 Gate Trend (requires history)</div>"),
            HBox([self.txt_trend_gates, self.dd_gtrend_prop]),
            self.btn_gtrend_plot,
            self.gtrend_chart_box,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>🔎 Stable Qubits (requires history)</div>"),
            HTML("<div style='font-size:14px;color:#6c757d;margin-bottom:4px;'>"
                 "Qubits consistently below a threshold across all fetched dates.</div>"),
            HBox([self.dd_stable_prop, self.txt_stable_threshold, self.slider_stable_ratio]),
            self.btn_stable_find,
            self.stable_html,
            HTML("<hr style='margin:6px 0; border-color:#bce8f1;'>"),
            HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>🔗 Stable Gates (requires history)</div>"),
            HTML("<div style='font-size:14px;color:#6c757d;margin-bottom:4px;'>"
                 "Gates consistently below a threshold across all fetched dates.</div>"),
            HBox([self.dd_gstable_prop, self.txt_gstable_threshold, self.slider_gstable_ratio]),
            self.btn_gstable_find,
            self.gstable_html,
        ])

        _err_tabs = Tab(children=[_err_tab_map, _err_tab_analysis])
        _err_tabs.set_title(0, 'Calibration')
        _err_tabs.set_title(1, 'Analysis')
        # Preserve the Calibration/Analysis subtab across rebuilds (same class
        # of bug as the Auto/I/O tabs).
        self._err_tabs = self._restore_tab_index(
            getattr(self, '_err_tabs', None), _err_tabs)
        return _err_tabs

    def _refresh(self, *args):
        """Central refresh — plot + counters + LQS banner + warnings.

        Wired as an ``observe`` callback on every widget that changes
        visible state. ``*args`` absorbs the ipywidgets change dict;
        the body re-reads inputs from ``self.*`` / ``state``.

        Order matters:
          1. refresh_plot (FigureWidget redraw)
          2. _update_all_counters (per-circuit α/β/anc)
          3. _update_undo_btns
          4. _update_threshold_status (banner when filtering is on)
          5. _ext_fns['update_fidelity'] — wrapped in try/except +
             ``_log.exception``; the LQS estimator has historically
             masked downstream bugs (e.g. the ``gate_f (1-err)``
             TypeError silently swallowed), so we surface at WARNING.
          6. calibration-date banner (green pill)
          7. props-warning banner (red, only when set)
          8. QPU 1 header label (inter-QPU mode only)
        """
        plot_mod.refresh_plot(self.fig, self.state, self.widgets_ref)
        self._update_all_counters()
        self._update_undo_btns()
        self._update_threshold_status()
        try:
            self._ext_fns["update_fidelity"]()
        except Exception:
            # Was a silent debug log — bumped to WARNING because it used
            # to hide real bugs in estimate_circuit_fidelity.
            _log.exception("update_fidelity failed during _refresh")
        if self.state.calibration_data_label:
            self.lbl_calibration_date.value = (
                f"<div style='font-size:15px;background:#eaf2e3;padding:5px 8px;"
                f"border-radius:4px;border:1px solid #b5d89c;margin:3px 0;'>"
                f"📅 <b>Data:</b> {esc(self.state.calibration_data_label)}</div>"
            )
        else:
            self.lbl_calibration_date.value = ""
        # ── Prominent warning banner for calibration issues ──────
        if self.state.props_warning:
            self.lbl_props_warning.value = (
                f"<div style='font-size:14px;color:#721c24;background:#f8d7da;padding:10px 12px;"
                f"border-radius:6px;border:2px solid #f5c6cb;margin:4px 0;font-weight:bold;'>"
                f"⚠️ {esc(self.state.props_warning)}</div>"
            )
        else:
            self.lbl_props_warning.value = ""
        # Update QPU 1 label (only visible in inter-QPU mode)
        if self.chk_inter_qpu.value:
            self.lbl_fig1.value = (
                f"<div style='font-size:14px;font-weight:bold;color:#fff;background:#2C3E50;"
                f"padding:3px 10px;border-radius:4px;text-align:center;margin-bottom:2px;'>"
                f"📡 {esc(self.state.active_backend_name or 'QPU 1')}</div>"
            )

    def _get_export(self):
        """Refresh the JSON-layout preview HTML.

        Called by every layout-mutating callback and on Export-tab
        entry. Returns ``ed`` (export data) so callers can chain — most
        ignore it. Also refreshes the LQS banner via
        ``_ext_fns['update_fidelity']`` (layout change can shift LQS).
        """
        n_circ = self.num_circuits.value if self.intra_qpu_mode.value else 1
        ed = export_mod.get_export_data(
            self.state, self.dd_mode.value, self.export_format.value, n_circ)
        self.json_preview.value = _pretty_json_html(ed)
        # Same WARNING-bump rationale as in _refresh.
        try:
            self._ext_fns["update_fidelity"]()
        except Exception:
            _log.exception("update_fidelity failed during _get_export")
        return ed

    def _update_undo_btns(self):
        """Sync Undo/Redo button enabled state with the state stacks.

        Called from ``self._refresh`` and from the click handlers
        themselves so the buttons grey out the moment a stack is
        empty.
        """
        self.btn_undo.disabled = not self.state.has_undo
        self.btn_redo.disabled = not self.state.has_redo

    def _update_threshold_status(self):
        """Parse threshold inputs, format the status banner, dim out-of-range qubits/gates.

        Only acts when ``self.chk_err_threshold`` is on AND the view
        mode is ``'error'``; otherwise clears the banner and returns.

        The bare ``except Exception`` is intentional: ipywidgets calls
        this on every keystroke, so partial / unparseable input must
        silently clear the banner (a noisy log would spam the kernel
        log; the plot just stays at full colours).

        Local helper ``_count_pass`` is a vectorised NumPy count of
        values in range (ignoring NaNs); used twice (qubits + gates)
        so kept as a method-local def to avoid duplicating ~10 lines
        of NumPy boilerplate.
        """
        if not self.chk_err_threshold.value or self.view_mode.value != 'error':
            self.threshold_status.value = ""
            return
        try:
            q_prop_key = self.dd_qubit_prop.value
            g_prop_key = self.dd_gate_prop.value
            # Always look up by the user-selected property key. The previous
            # fallback to readout_errors / gate_errors when *_props_data was
            # empty silently showed the wrong stats (e.g. T1 selected but
            # readout-error counts displayed). Empty arrays are handled
            # cleanly downstream by len(valid_*) == 0.
            q_arr = np.array(self.state.qubit_props_data.get(q_prop_key, []), dtype=float)
            g_arr = np.array(self.state.gate_props_data.get(g_prop_key, []), dtype=float)

            def _count_pass(arr, lo, hi):
                """Vectorized count of values in arr that satisfy
                lo <= v <= hi, ignoring NaNs. Replaces a Python for-loop
                that ran on every threshold-text keystroke."""
                if arr.size == 0:
                    return 0, arr
                valid = arr[~np.isnan(arr)]
                if valid.size == 0:
                    return 0, valid
                mask = np.ones(valid.shape, dtype=bool)
                if lo is not None:
                    mask &= (valid >= lo)
                if hi is not None:
                    mask &= (valid <= hi)
                return int(mask.sum()), valid

            # Parse qubit thresholds
            q_lo = float(self.txt_q_thr_min.value) if self.txt_q_thr_min.value.strip() else None
            q_hi = float(self.txt_q_thr_max.value) if self.txt_q_thr_max.value.strip() else None
            n_q_pass, valid_q = _count_pass(q_arr, q_lo, q_hi)
            lo_s = f"{q_lo:.1E}" if q_lo is not None else "—"
            hi_s = f"{q_hi:.1E}" if q_hi is not None else "—"
            q_info = f"Qubits [{lo_s}, {hi_s}]: <b>{n_q_pass}</b>/{len(valid_q)}"

            # Parse gate thresholds
            g_lo = float(self.txt_g_thr_min.value) if self.txt_g_thr_min.value.strip() else None
            g_hi = float(self.txt_g_thr_max.value) if self.txt_g_thr_max.value.strip() else None
            n_g_pass, valid_g = _count_pass(g_arr, g_lo, g_hi)
            lo_s = f"{g_lo:.1E}" if g_lo is not None else "—"
            hi_s = f"{g_hi:.1E}" if g_hi is not None else "—"
            g_info = f"Gates [{lo_s}, {hi_s}]: <b>{n_g_pass}</b>/{len(valid_g)}"

            self.threshold_status.value = (
                f"<div style='font-size:14px;color:#555;background:#fff;padding:4px 6px;"
                f"border-radius:3px;margin:3px 0;border:1px solid #ddd;'>"
                f"🎯 {q_info} · {g_info}</div>"
            )
        except Exception:
            self.threshold_status.value = ""

    def _update_all_counters(self):
        """Refresh the per-circuit counter HTML banners.

        Iterates ``self.state.layouts_data[i]`` for ``i`` in
        ``range(MAX_CIRCUITS)`` and updates the ``counter_html``
        widget (state-resident, built per-circuit by
        ``self._init_circuit_slot``). Format:
          * LUCJ — ``α=A β=B Anc=C Other=O | Tot=A+B`` (Tot excludes
            Anc/Other; only α+β counts for the LUCJ parameter count).
          * General — ``Qubits=Q Anc=A``.

        Skips slots without ``counter_html`` so partial init in
        ``_build_dynamic_ui`` doesn't KeyError.
        """
        for i in range(MAX_CIRCUITS):
            if i in self.state.layouts_data and 'counter_html' in self.state.layouts_data[i]:
                d = self.state.layouts_data[i]
                a, b_cnt = len(d['alpha']), len(d['beta'])
                anc, o = len(d['ancilla']), len(d['other'])
                if self.dd_mode.value == 'lucj':
                    tot = a + b_cnt
                    d['counter_html'].value = (
                        f"<div style='font-size:17px;background:#e8f4f8;padding:8px 10px;border-radius:5px;"
                        f"border:1px solid #bce8f1;margin-bottom:6px;'>"
                        f"<b>📊</b> α=<b>{a}</b> β=<b>{b_cnt}</b> Anc=<b>{anc}</b> Other=<b>{o}</b>"
                        f"&nbsp;|&nbsp;<span style='color:#E74C3C;font-weight:bold;'>Tot={tot}</span></div>"
                    )
                else:
                    q_cnt = len(d['qubits'])
                    d['counter_html'].value = (
                        f"<div style='font-size:17px;background:#e8f4f8;padding:8px 10px;border-radius:5px;"
                        f"border:1px solid #bce8f1;margin-bottom:6px;'>"
                        f"<b>📊</b> Qubits=<b>{q_cnt}</b> Anc=<b>{anc}</b></div>"
                    )

    def _on_plot_click(self, trace, points, selector):
        """Handle Plotly click events on the FigureWidget qubit dots.

        Wired in ``self._load_device_graph`` (and re-wired after every
        graph rebuild) onto the qubits trace, looked up by its stable
        ``.meta`` (``plot._META_QUBITS``) so trace reorders can't misroute
        it. (``.uid`` would be regenerated by FigureWidget; ``.meta`` is
        preserved.)
        Signature follows the Plotly ``on_click`` contract —
        ``(trace, points, selector)`` positional. ``selector`` is
        always ``None`` for click events (only populated for box/lasso
        selection), but stays in the signature to honour the contract
        — do NOT remove it.

        Branches on ``self.ancilla_tool.value``:

          * **Ancilla tool active** — toggle the clicked qubit's
            membership in ``state.layouts_data[c_idx]['ancilla']``:
            add (with standard gray fill) if absent, remove (+ pop
            style) if present.
          * **Ancilla tool inactive** — look up the visible manual
            tab (``self.manual_tabs.selected_index``) to pick the
            destination circuit ``idx``. **Critical guard**: reject
            clicks targeting circuits not currently visible
            (``idx >= n_visible``) — without it a stale
            ``selected_index`` from a previous multi-circuit session
            would write into a hidden layout slot, and the styling
            would only become visible the next time the user enabled
            multi-circuit mode. Then:
              - Role ``'Remove'``: scrub the qubit from every role
                list and pop its style. Idempotent.
              - Else: parse the trailing word of the selector
                (``'Select Alpha'`` → ``'alpha'``). Toggle off if
                already in that role; otherwise scrub from every
                other role first, then assign with the active
                per-circuit style.

        ``state.push_undo`` is called BEFORE any mutation so each
        click is independently revertible. On success calls
        ``self._refresh`` + ``self._get_export``.
        """
        if not points.point_inds:
            return
        q_id = self.state.qubit_ids[points.point_inds[0]]

        self.state.push_undo()

        if self.ancilla_tool.value is not None:
            # ancilla_tool.value is the circuit index (0-based int) thanks
            # to the (label, value) options below. None means "View Only".
            c_idx = self.ancilla_tool.value
            if c_idx not in self.state.layouts_data:
                return
            d = self.state.layouts_data[c_idx]
            if q_id not in d['ancilla']:
                d['ancilla'].append(q_id)
                rgba = hex_to_rgba('#95A5A6', 1.0)
                d['styles'][q_id] = {'fill': rgba, 'line': '#000000', 'width': 2, 'text_color': '#000000'}
            else:
                d['ancilla'].remove(q_id)
                d['styles'].pop(q_id, None)
        else:
            idx = self.manual_tabs.selected_index
            if idx is None:
                return
            # Reject clicks targeting circuits not currently visible in the
            # active mode. Without this, a stale selected_index from a
            # previous multi-circuit session (e.g. user had 4 circuits, then
            # switched to single-circuit) would still write into
            # state.layouts_data[idx] for a circuit hidden from the UI —
            # the qubit would be styled invisibly until the user re-enables
            # multi-circuit mode.
            n_visible = self.num_circuits.value if self.intra_qpu_mode.value else 1
            if idx >= n_visible:
                return
            d = self.state.layouts_data[idx]

            if d['sel'].value != 'Remove':
                tag = d['sel'].value.split(' ')[-1].lower()
                if q_id in d[tag]:
                    d[tag].remove(q_id)
                    d['styles'].pop(q_id, None)
                else:
                    for g in ['alpha', 'beta', 'other', 'qubits', 'ancilla']:
                        if q_id in d[g]:
                            d[g].remove(q_id)
                    fill_val = 'none' if self.chk_line_only.value else hex_to_rgba(d['current_fill'], d['sl_o'].value)
                    d['styles'][q_id] = {'fill': fill_val, 'line': d['current_line'], 'width': d['sl_w'].value, 'text_color': d['current_text_color']}
                    d[tag].append(q_id)
            else:
                for g in ['alpha', 'beta', 'other', 'qubits', 'ancilla']:
                    if q_id in d[g]:
                        d[g].remove(q_id)
                d['styles'].pop(q_id, None)

        self._refresh()
        self._get_export()

    def _on_undo(self, b):
        """Pop the last layout from ``state.undo_stack`` and refresh.

        ``state.undo()`` is the gate — returns ``False`` on an empty
        stack (no-op).
        """
        if self.state.undo():
            self._refresh()
            self._get_export()

    def _on_redo(self, b):
        """Symmetric to :meth:`_on_undo` — pop from ``state.redo_stack``."""
        if self.state.redo():
            self._refresh()
            self._get_export()

    def _on_clear_all(self, b):
        """Wipe every per-circuit layout (``state.clear_all_layouts``).

        Pushes undo first so the destructive click is revertible.
        """
        self.state.push_undo()
        self.state.clear_all_layouts()
        self._refresh()
        self._get_export()
        with self.out:
            print("🗑️ All layouts cleared.")

    def _on_auto_map(self, b):
        """Apply the auto-map text input to the active circuits.

        Reads ``self.txt_auto`` (one circuit per line) and delegates
        to ``export_mod.parse_auto_text`` → ``(count, autodetect_stats)``.

        Per-circuit α/β breakdown is printed only in LUCJ mode
        (general mode has no alpha/beta concept).
        """
        try:
            self.state.push_undo()
            # parse_auto_text now returns (count, autodetect_stats). The stats
            # let us tell the user whether alpha/beta were topologically
            # auto-detected or fell back to midpoint split.
            count, stats = export_mod.parse_auto_text(
                self.state, self.txt_auto.value, self.dd_mode.value
            )
            self._refresh()
            self._get_export()
            with self.out:
                print(f"✅ Auto-Layout applied! ({count} circuit(s))")
                # Per-circuit breakdown. Only meaningful for LUCJ mode.
                if self.dd_mode.value == 'lucj':
                    for s in stats:
                        tag = ('topology auto-detected'
                               if s.get('used_autodetect')
                               else 'midpoint split (fallback)')
                        print(
                            f"   • C{s['circuit'] + 1}: "
                            f"α={s['alpha_len']}, β={s['beta_len']} — {tag}"
                        )
        except Exception as e:
            with self.out:
                print(f"❌ Error Auto-Layout: {e}")

    def _on_unified_upload(self, change):
        """Handle ``self.btn_unified_upload`` FileUpload changes.

        Decodes the payload via ``_read_upload`` (bridges ipywidgets
        v7 tuple vs. v8 dict differences) then dispatches to
        ``export_mod.import_layout_file`` (auto-detects .npy / .json /
        .qpy / .qasm). Pushes undo BEFORE mutation so bad uploads
        are revertible.

        The trailing ``self.btn_unified_upload.value = ()`` reset is
        load-bearing — ipywidgets debounces identical ``value``
        events, so without the reset re-uploading the same file
        would silently no-op.
        """
        if not change['new']:
            return
        try:
            content, filename = _read_upload(change['new'])
            n_circ = self.num_circuits.value if self.intra_qpu_mode.value else 1
            self.state.push_undo()
            fmt, detail = export_mod.import_layout_file(
                self.state, content, filename, self.dd_mode.value, n_circ)
            self._refresh()
            self._get_export()
            with self.out:
                print(f"✅ [{fmt}] {detail}")
            self.btn_unified_upload.value = ()
        except Exception as e:
            with self.out:
                print(f"❌ Upload error: {e}")

    def _on_json_import(self, change):
        """Handle ``self.btn_json_import`` FileUpload — JSON-only path.

        Symmetric to :meth:`_on_unified_upload` but goes through
        ``import_json_layout`` (expects a pre-parsed layout dict —
        qubits per role, no auto-detect). Returns the list of
        layout names imported. Same FileUpload reset-after-success
        debounce gotcha as ``_on_unified_upload``.
        """
        if not change['new']:
            return
        try:
            content, _ = _read_upload(change['new'])
            self.state.push_undo()
            names = export_mod.import_json_layout(
                self.state, content, self.dd_mode.value)
            self._refresh()
            self._get_export()
            with self.out:
                print(f"✅ JSON layout imported: {names}")
            self.btn_json_import.value = ()
        except Exception as e:
            with self.out:
                print(f"❌ Error import JSON: {e}")

    def _on_refresh(self, b):
        """Fresh-start refresh: tear down the current UI and rebuild from scratch.

        Wipes every user customisation in widget state (color
        pickers, sliders, checkboxes, layouts, delta/ref calibration,
        intra-QPU mode) and re-fetches today's calibration via the
        normal connect pipeline. The persistent custom-color palette
        (``state.custom_colors``, saved to disk) survives because
        ``launch_mapper`` reloads it on the rebuilt session.

        Rebuilding via ``launch_mapper`` is the single source of
        truth — earlier surgical resets that preserved layouts/display
        prefs were a recurring source of stale-UI bugs from diverging
        widget vs. state defaults.

        The rebuild runs inside a nested ``_do_rebuild()`` (captures
        the ephemeral ``config`` dict from this method scope) and is
        async-marshalled via ``_resolve_main_loop_scheduler``: running
        ``launch_mapper`` inline before ipywidgets flushes the click
        event causes a visible swatch-row reflow on the first colour
        pick after Refresh (old widget tree stays momentarily attached
        and races the new one). Same pattern used by
        ``connection.on_connect``.
        """
        self.btn_refresh.disabled = True
        self.btn_refresh.description = '⏳ Refreshing...'

        # Invalidate today's disk cache so the rebuilt graph fetches
        # live calibration from the API rather than reading back what
        # we just had. Best-effort: a failure here doesn't block the
        # rebuild because the API call below may still return fresh data.
        if self.state.active_backend_name:
            try:
                from datetime import datetime as _dt, timezone as _tz
                today = _dt.now(_tz.utc).date()
                data_mod.invalidate_cache_for_date(
                    self.state.active_backend_name, today)
            except Exception as exc:
                _log.warning(
                    "Cache invalidation failed during refresh: %s", exc)

        # Snapshot the live connection so we can hand it back to
        # launch_mapper. `initial_props=None` forces the rebuilt UI's
        # _load_device_graph to fetch fresh calibration through the
        # standard pipeline (same path as the original post-connect load).
        config = {
            'vendor':             self.state.active_vendor,
            'backend':            self.state.active_backend,
            'backend_name':       self.state.active_backend_name,
            'chip':               self.state.active_chip,
            'service':            self.state.active_service,
            'available_backends': list(self.state.available_backends or []),
            'props_warning':      None,
            'initial_props':      None,
            'ansatz':             self.state.active_ansatz,
        }

        def _do_rebuild():
            try:
                # Deferred import to avoid the ui→mapper circular at module load.
                from ..mapper import launch_mapper
                launch_mapper(config)
            except Exception as e:
                # If the rebuild raised before launch_mapper's clear_output
                # ran, the old UI is still visible and we can recover. If
                # it raised after, the user sees an empty cell — log so
                # the failure isn't completely silent.
                _log.exception("Refresh rebuild failed")
                try:
                    self.btn_refresh.disabled = False
                    self.btn_refresh.description = '🔄 Refresh (Fresh Start)'
                    with self.out:
                        print(f"❌ Refresh failed: {e}")
                except Exception:
                    pass

        # Marshal the rebuild onto the main event loop instead of running
        # it inline inside this button callback. See the method docstring
        # for the reasoning around the visible swatch-row reflow that
        # this defer eliminates.
        try:
            from ..async_util import _resolve_main_loop_scheduler
            schedule = _resolve_main_loop_scheduler()
            schedule(_do_rebuild)
        except Exception:
            _log.exception(
                "main-loop scheduler unavailable; running rebuild inline")
            _do_rebuild()

    def _load_device_graph(self, change=None, force=False):
        """Reload device topology from the current calibration / backend.

        Called at initial bootstrap, on ``dp_date`` / ``txt_hw_hour``
        observers, by ``ui.backend_switch`` (with ``force=True``), and
        by ``ui.inter_qpu`` panel setup.

        Args:
            change: ipywidget observer payload, ignored.
            force: bypass the reentrancy guards. Used by callers
                (e.g. ``_on_backend_switch`` in ``ui.backend_switch``)
                that hold their own guard and need this routine as the
                final step of their own operation.

        Reentrancy — two flags with asymmetric shapes (do not "fix"):
          * ``_loading_device`` (clean bool) — this method's own guard.
          * ``_switching_backend`` (``[bool]`` list) — ``ui.backend_switch``
            mutates the SAME object via ``[0] = True/False``.

        Body order: reset zoom + parse date/hour (invalid hour surfaces
        to ``self.out`` instead of silently loading midnight UTC) →
        fetch device data → clear LUCJ state → clear delta/reference
        calibration, time_mode='live' → build_device_graph → rebind
        click handler (guard ``len(fig.data) >= 4`` so a future trace
        reorder in ``plot.create_figure`` degrades to "no handler"
        instead of IndexError) → ``_refresh()``.
        """
        if not force and (self._loading_device or self._switching_backend[0]):
            return
        self._loading_device = True
        try:
            self.state.reset_zoom()
            vendor = getattr(self.state, 'active_vendor', '') or ''
            date_val = self.dp_date.value
            # Hour-parse failure falls back to date-only and surfaces
            # the error in self.out (see docstring).
            if date_val and self.txt_hw_hour.value.strip():
                from datetime import datetime as _dt
                raw_hour = self.txt_hw_hour.value.strip()
                parts = raw_hour.replace('.', ':').split(':')
                try:
                    h = int(parts[0])
                    m = int(parts[1]) if len(parts) > 1 else 0
                    if not (0 <= h < 24 and 0 <= m < 60):
                        raise ValueError(
                            f"Hour out of range: '{raw_hour}' "
                            f"(expected HH in 0-23 and MM in 0-59)")
                    date_val = _dt(date_val.year, date_val.month, date_val.day, h, m)
                except (ValueError, IndexError) as _hour_exc:
                    with self.out:
                        print(f"⚠️ Ignoring invalid hour '{raw_hour}': "
                              f"{_hour_exc} — using date only.")

            result = data_mod.generate_device_data(
                self.state, vendor, self.state.active_chip, date_val,
                prefetched_props=self.state.initial_props)
            self.state.x_coord, self.state.y_coord, self.state.edges_list = result[0], result[1], result[2]
            self.state.qubit_ids, self.state.readout_errors, self.state.gate_errors = result[3], result[4], result[5]

            self.state.clear_all_layouts()

            # Clear LUCJ state from the previous backend — without this,
            # the ranking dropdowns can point at qubits that no longer
            # exist after Refresh, silently corrupting layout selection.
            self._lucj_all_layouts.clear()
            self.dd_lucj_layout.options = []
            self.dd_lucj_layout.disabled = True
            self.lbl_lucj_status.value = ""
            self.lbl_lucj_notices.value = ""
            self.lbl_lucj_detail.value = ""
            self._lucj_multi_results.clear()
            self._lucj_multi_norbs.clear()
            self._lucj_multi_dropdowns.clear()
            self.lbl_lucj_multi.value = ""
            self.lbl_lucj_multi_notices.value = ""
            self.lucj_multi_detail_box.children = ()

            # {} (not None) for delta props — plot.py readers guard on
            # truthiness, but None would break any future caller that
            # does `delta_qubit_props.items()` outside the guard.
            self.state.delta_mode = False
            self.state.delta_qubit_props = {}
            self.state.delta_gate_props = {}

            # Reset reference calibration (single source: state.ref_*).
            self.state.ref_qubit_props = None
            self.state.ref_gate_props = None
            self.state.ref_timestamp = None
            self.state.delta_enabled = False
            # Reset time mode + saved live-cal cache; ui.time_panel
            # reads these on rebuild() and reflects them in its widgets.
            self.state.time_mode = 'live'
            self.state._live_qubit_props = None
            self.state._live_gate_props = None
            self.state._live_cal_label = None

            plot_mod.build_device_graph(self.fig, self.state, self.widgets_ref)

            # Re-bind click handler on the qubit trace via its stable
            # .meta (never by index — adding/removing legend traces would
            # otherwise misroute clicks to a non-qubit trace). We used
            # .uid first, but FigureWidget regenerates .uid; .meta is
            # preserved.
            q_trace = plot_mod._trace_by_meta(self.fig, plot_mod._META_QUBITS)
            if q_trace is not None:
                q_trace.on_click(self._on_plot_click)
            else:
                _log.warning(
                    "Device graph has no qubit trace; click handler not bound "
                    "(%d traces)", len(self.fig.data))
            self._refresh()
        finally:
            self._loading_device = False

    def _update_view(self, change=None):
        """Switch the visible content panel based on ``self.nav_bar.value``.

        Wired as an ``observe`` callback on ``nav_bar`` /
        ``num_circuits`` / ``dd_mode`` and called from
        ``_toggle_intra_qpu`` and bootstrap.

        Three responsibilities, in order:
          1. **Nav-bar option refresh** — show 🤖 Auto-LUCJ only when
             ``dd_mode.value == 'lucj'``.
          2. **Content-box dispatch** — call one of the 5
             ``self._build_X_view()`` methods, mount on
             ``content_box.children``. Side effects: Error tab
             auto-switches ``view_mode`` to ``'error'`` and fires
             ``update_ranking`` / ``update_grank``; Export tab calls
             ``self._get_export`` + ``update_fidelity``.
          3. **Ancilla tool refresh** — repopulate options for
             current ``num_circuits`` / ``intra_qpu_mode`` and hide
             outside manual mode.

        The unobserve→reseed→observe dance silences the observer
        while we replace options so we don't recurse into ourselves
        via the change event. The try/finally re-attaches the
        observer even if ipywidgets type-checks fail on the re-seed
        — without it a transient bug would silently leave the nav
        bar inert for the rest of the session.

        Bound-method ``__eq__`` returns ``True`` for
        same-method-same-instance, so traitlets' ``unobserve`` finds
        the registered callback even though ``self._update_view``
        creates a fresh bound-method object on each access.
        """
        # Update nav bar: hide Auto-LUCJ in general mode
        is_lucj = self.dd_mode.value == 'lucj'
        if is_lucj:
            new_opts = [('✏️ Layout', 'manual'), ('🤖 Auto', 'auto_lucj'),
                        ('📁 I/O', 'io'),
                        ('🔍 Error', 'error_inspect')]
        else:
            new_opts = [('✏️ Layout', 'manual'), ('📁 I/O', 'io'),
                        ('🔍 Error', 'error_inspect')]
        current_vals = [v for _, v in self.nav_bar.options]
        new_vals = [v for _, v in new_opts]
        if current_vals != new_vals:
            old_val = self.nav_bar.value
            # See method docstring for the unobserve/observe rationale.
            self.nav_bar.unobserve(self._update_view, 'value')
            try:
                self.nav_bar.options = new_opts
                self.nav_bar.value = old_val if old_val in new_vals else 'manual'
            finally:
                self.nav_bar.observe(self._update_view, 'value')

        mode = self.nav_bar.value
        # Glitch 2 fix — only rebuild the per-circuit manual UI when it will
        # actually be shown (mode == 'manual'), or when the trigger was a
        # structural change rather than a plain nav switch. num_circuits /
        # dd_mode changes arrive with a non-nav owner, and _toggle_intra_qpu
        # / bootstrap call with change=None — all of those still rebuild so
        # the manual view is correct the next time it opens. A bare nav to
        # auto/export/error used to reallocate every per-circuit tab VBox for
        # a view that doesn't display them (flicker + wasted work); skip it.
        _nav_only = change is not None and getattr(change, 'owner', None) is self.nav_bar
        if mode == 'manual' or not _nav_only:
            self._build_dynamic_ui()
        # Auto-switch to error view when entering Error tab
        if mode == 'error_inspect' and self.view_mode.value != 'error':
            self.view_mode.value = 'error'
        if mode == 'error_inspect':
            self._ext_fns["update_ranking"]()
            self._ext_fns["update_grank"]()
        if mode == 'manual':
            self.content_box.children = [self._build_manual_view()]
        elif mode == 'auto_lucj':
            self.content_box.children = [self._build_auto_lucj_view()]
        elif mode == 'io':
            # Unified Import / Export tab (Export + former Load subpanels).
            self._get_export()
            self._ext_fns["update_fidelity"]()
            self.content_box.children = [self._build_import_export_view()]
        elif mode == 'error_inspect':
            self.content_box.children = [self._build_error_view()]

        n_circ = self.num_circuits.value if self.intra_qpu_mode.value else 1
        self.ancilla_tool.options = [('View Only', None)] + [
            (f"Add Ancilla C{i+1}", i) for i in range(n_circ)
        ]
        self.ancilla_tool.value = None
        # Only show ancilla tool in manual mode — irrelevant for auto/export/error
        self.ancilla_tool.layout.display = 'block' if self.nav_bar.value == 'manual' else 'none'

    def _make_cs_preview(self):
        """Build a 14px-tall horizontal-gradient HTML swatch for the active colorscale."""
        cs = COLORSCALE_OPTIONS[self.dd_colorscale.value]
        r0, g0, b0 = cs['low']
        r1, g1, b1 = cs['high']
        stops = [f"rgb({int(r0+(r1-r0)*t/4)},{int(g0+(g1-g0)*t/4)},{int(b0+(b1-b0)*t/4)}) {t*25}%"
                 for t in range(5)]
        return (f"<div style='width:100%;height:14px;border-radius:3px;margin:2px 0;"
                f"background:linear-gradient(to right,{','.join(stops)});border:1px solid #ccc;'></div>")

    def _on_colorscale_change(self, change):
        """Observer for ``dd_colorscale`` — mirror to state, rebuild swatch, refresh."""
        self.state.current_colorscale = change['new']
        self.colorscale_preview.value = self._make_cs_preview()
        self._refresh()

    def _on_global_text_color(self, change):
        """Observer for ``global_text_color`` — cascade text color to all circuits.

        Intentionally **destructive** (a "set all", not a "set
        default") — pushes undo first so individually-set text colors
        can be recovered. Writes the new colour into every per-slot
        default, ColorPicker, per-qubit style, and per-group memory.
        """
        self.state.push_undo()
        new_color = change['new']
        for i in range(MAX_CIRCUITS):
            d = self.state.layouts_data[i]
            d['current_text_color'] = new_color
            if 'cp_text_color' in d:
                d['cp_text_color'].value = new_color
            for s in d['styles'].values():
                s['text_color'] = new_color
            for gs in d.get('group_styles', {}).values():
                gs['text_color'] = new_color
        self._refresh()

    def _on_threshold_text_change(self, change):
        """Observer for the 4 threshold min/max Text inputs.

        The cheap ``chk_err_threshold`` guard skips refresh when
        filtering is off, so typing in the boxes while disabled
        doesn't redraw the plot on every keystroke.
        """
        if self.chk_err_threshold.value:
            self._refresh()

    def _update_fig_layout(self, *args):
        """Push new width/height to the FigureWidget via Plotly ``update_layout``."""
        self.fig.update_layout(width=self.plot_w_slider.value, height=self.plot_h_slider.value)

    def _enable_export_gif(self):
        """Enable ``self.btn_export_gif`` once history is populated.

        Fired by ``ui.time_panel`` after a successful fetch and by
        ``self._auto_load_done``. Also triggers ``update_fidelity``
        so the Stability Check section refreshes — without it the
        section would stay on the "no history fetched" hint (state
        mutation isn't observable by ipywidgets).
        """
        self.btn_export_gif.disabled = False
        self.btn_export_gif.tooltip = ''
        try:
            self._ext_fns["update_fidelity"]()
        except Exception:
            _log.exception(
                "update_fidelity failed after history load")

    def _set_num_circuits_silent(self, n):
        """Set ``num_circuits`` WITHOUT the full ``_update_view`` rebuild its
        observer would fire — used when a multi-circuit sweep/legacy apply
        changes the count as a RESULT of its own action.

        The visible Multi subpanel is already updated in place (its summary and
        per-circuit override dropdowns are persistent widgets refreshed by the
        caller), so rebuilding ``content_box`` would only repaint an identical
        panel — the flicker X2b targets. A count change genuinely needs just the
        OFF-SCREEN refresh: the per-circuit manual UI + the ancilla-tool
        options — exactly what ``_build_dynamic_ui`` does (it never touches
        ``content_box``). The nav options don't depend on ``num_circuits`` (only
        on ``dd_mode``), so they need no refresh here.

        The unobserve→set→observe idiom (same as ``_update_view``'s nav-bar
        reseed) silences only THIS assignment; the ``_restore_tab_index`` helper
        still guards every legitimate full rebuild (checkbox toggles, nav
        switches)."""
        self.num_circuits.unobserve(self._update_view, 'value')
        try:
            self.num_circuits.value = n
        finally:
            self.num_circuits.observe(self._update_view, 'value')
        self._build_dynamic_ui()

    def _toggle_intra_qpu(self, change):
        """Observer for ``intra_qpu_mode`` — enable/disable num_circuits + refresh view."""
        self.num_circuits.disabled = not change['new']
        self._update_view()

    def _auto_load_work(self):
        """Background-thread worker — 14-day post-connect history auto-load.

        Calls ``data_mod.auto_load_recent_history``; returns
        ``(n_loaded, source)`` where ``source`` is ``'noop'`` /
        ``'cache'`` / ``'fetch'``.
        """
        return data_mod.auto_load_recent_history(
            self.state, days=14, min_cached_threshold=5)

    def _auto_load_done(self, result):
        """on_done callback for the history auto-load thread.

        Delegates to ``self._enable_export_gif`` on success
        (cascades into ``update_fidelity`` to repopulate Stability
        Check) or restores the "switch to Multi-day…" hint on
        zero snapshots (e.g. offline mode).
        """
        n_loaded, source = result
        if n_loaded > 0:
            self._enable_export_gif()
        else:
            self.lbl_stability_note.value = (
                "<span style='color:#888;font-size:16px;'>"
                "no history fetched — switch to Multi-day mode and refetch "
                "to enable stability check</span>"
            )

    def _auto_load_err(self, exc):
        """on_error callback — log + surface a manual-Multi-day hint."""
        _log.exception("Auto-load of calibration history failed")
        self.lbl_stability_note.value = (
            f"<span style='color:#e74c3c;font-size:16px;'>"
            f"Auto-load failed: {type(exc).__name__} — open Multi-day "
            f"mode to fetch manually.</span>"
        )

    # ── Group-style helpers (per-circuit-slot widget memory) ─────────

    def _update_manual_swatches(self, idx):
        """Rebuild the per-circuit fill/line swatch buttons from STANDARD_COLORS + state.custom_colors."""
        d = self.state.layouts_data[idx]
        all_colors = STANDARD_COLORS + self.state.custom_colors
        d['swatches_fill'].children = [d['create_fill_swatch'](c) for c in all_colors]
        d['swatches_line'].children = [d['create_line_swatch'](c) for c in all_colors]

    def _sync_manual_swatch_borders(self, idx):
        """Flip ONLY the border of the already-built per-circuit swatches to
        reflect the active fill/line colour — no teardown/rebuild. Each swatch
        owns its Layout, so a border write re-renders just that one button; the
        no-op guard limits writes to the swatches that actually changed."""
        d = self.state.layouts_data[idx]
        for box, current in ((d['swatches_fill'], d.get('current_fill')),
                             (d['swatches_line'], d.get('current_line'))):
            for btn in box.children:
                selected = getattr(btn, '_swatch_hex', None) == current
                new_border = ('2px solid #2980B9' if selected
                              else '2px solid #bdc3c7')
                if btn.layout.border != new_border:
                    btn.layout.border = new_border

    def _update_manual_preview(self, idx):
        """Redraw the per-circuit preview dot — what the user sees before Apply commits to selected qubits."""
        d = self.state.layouts_data[idx]
        rgba = hex_to_rgba(d['current_fill'], d['sl_o'].value)
        d['preview'].value = (
            f"<div style='width:35px;height:35px;border-radius:50%;background:{rgba};"
            f"border:{d['sl_w'].value}px solid {d['current_line']}; margin:5px 0;'></div>"
        )

    def _active_group(self, sel_value):
        """Map a role-selector label to its group key (None for 'Remove'/unknown)."""
        return self._SEL_TO_GROUP.get(sel_value)

    def _ensure_group_styles(self, d):
        """Lazy-init the per-slot ``d['group_styles']`` dict (idempotent).

        Each circuit remembers per-role (alpha/beta/other/qubits/
        ancilla) style settings so switching back to an edited role
        restores its appearance. Seeded from ``d['defaults']``.
        """
        if 'group_styles' not in d:
            defs = d.get('defaults', {})
            d['group_styles'] = {
                'alpha':   {'fill': defs.get('fill', '#636EFA'),      'line': defs.get('line', '#000'), 'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
                'beta':    {'fill': defs.get('fill_beta', '#3498DB'), 'line': defs.get('line', '#000'), 'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
                'other':   {'fill': defs.get('fill_other', '#F39C12'),'line': defs.get('line', '#000'), 'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
                'qubits':  {'fill': defs.get('fill', '#636EFA'),      'line': defs.get('line', '#000'), 'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
                'ancilla': {'fill': defs.get('fill_anc', '#95A5A6'),  'line': '#000000',                'opacity': 1.0, 'width': 2, 'text_color': '#000000'},
            }

    def _save_controls_to_group(self, idx, group=None):
        """Snapshot the active widget state into ``d['group_styles'][group]``.

        ``group=None`` defaults to the active role. An explicit
        ``group`` is used by ``self._mk_sel_obs`` to save the OLD
        role's state before loading the NEW role's on selector
        change. No-ops if the slot isn't initialised or role is
        ``'Remove'``.
        """
        d = self.state.layouts_data[idx]
        self._ensure_group_styles(d)
        if group is None:
            group = self._active_group(d['sel'].value) if 'sel' in d else None
        if group is None or 'current_fill' not in d:
            return
        gs = d['group_styles'].get(group)
        if gs is None:
            return
        gs['fill'] = d['current_fill']
        gs['line'] = d['current_line']
        gs['opacity'] = d['sl_o'].value
        gs['width'] = d['sl_w'].value
        gs['text_color'] = d.get('current_text_color', '#000000')

    def _load_group_to_controls(self, idx, group_name):
        """Inverse of ``_save_controls_to_group`` — hydrate widgets from
        group memory. Writes into both the per-slot ``current_*`` keys
        AND the widget values (color pickers, sliders), then refreshes
        swatches + preview dot so the whole control bar snaps to the
        new role's appearance.
        """
        d = self.state.layouts_data[idx]
        self._ensure_group_styles(d)
        gs = d['group_styles'].get(group_name)
        if gs is None or 'sl_o' not in d:
            return
        d['current_fill'] = gs['fill']
        d['current_line'] = gs['line']
        d['current_text_color'] = gs.get('text_color', '#000000')
        d['cp_fill_advanced'].value = gs['fill']
        d['cp_line_advanced'].value = gs['line']
        d['sl_o'].value = gs.get('opacity', 1.0)
        d['sl_w'].value = gs.get('width', 2)
        d['cp_text_color'].value = gs.get('text_color', '#000000')
        # Role switch is a SELECTION change, not a palette change — flip
        # borders in place. cp_* assigns above no-op through their guarded
        # observers (current_* is set before each value), so this is the
        # single sync point.
        self._sync_manual_swatch_borders(idx)
        self._update_manual_preview(idx)

    def _resync_group_controls_after_load(self):
        """After a session load, hydrate every visible slot's controls +
        preview from its restored ``group_styles``.

        ``from_session_dict`` updates the data in place but leaves
        ``current_*`` / colour-pickers / preview stale, and the per-role
        sync otherwise only fires on a manual role switch — so without
        this a loaded session's per-role colours never reach the editing
        controls or the preview dot.
        """
        n = self.num_circuits.value if self.intra_qpu_mode.value else 1
        for idx in range(n):
            d = self.state.layouts_data.get(idx)
            if not d or 'sel' not in d or 'cp_line_advanced' not in d:
                continue
            self._ensure_group_styles(d)
            g = self._active_group(d['sel'].value)
            if g:
                self._load_group_to_controls(idx, g)
        # Legend widgets: hydrate from restored state.legend_config without
        # re-firing their observers (we'd double-refresh otherwise).
        if hasattr(self, 'chk_legend_enabled'):
            cfg = self.state.legend_config
            self.chk_legend_enabled.value = bool(cfg.get('enabled', True))
            for role in self._LEGEND_ROLES:
                role_cfg = cfg.get('roles', {}).get(role, {})
                if role in self.legend_chk:
                    self.legend_chk[role].value = bool(role_cfg.get('show', False))
                if role in self.legend_lbl:
                    self.legend_lbl[role].value = str(role_cfg.get('label', role))

    def _flush_controls_to_groups(self):
        """Commit each visible slot's active-role control state into its
        ``group_styles`` before serialising a session. ``_save_controls_to_group``
        otherwise only runs on Apply / role-switch, so a colour set on the
        active role but not yet Applied would be lost on save.
        """
        n = self.num_circuits.value if self.intra_qpu_mode.value else 1
        for idx in range(n):
            d = self.state.layouts_data.get(idx)
            if not d or 'sel' not in d:
                continue
            self._save_controls_to_group(idx)   # group=None -> active role

    def _update_preview_label(self, idx):
        """Update the 'Preview: <Role>' label (or bare 'Preview:' if no role active)."""
        d = self.state.layouts_data[idx]
        group = self._active_group(d['sel'].value) if 'sel' in d else None
        if 'preview_label' in d and group:
            d['preview_label'].value = (
                f"<span style='font-size:15px;font-weight:bold;color:#2980B9;'>"
                f"Preview: {group.title()}</span>")
        elif 'preview_label' in d:
            d['preview_label'].value = "<span style='font-size:15px;'>Preview:</span>"

    def _build_dynamic_ui(self):
        """Build per-circuit UI for intra-QPU multi-programming mode.

        Called from ``self._update_view`` on nav switch and on
        ``num_circuits`` / ``intra_qpu_mode`` / ``dd_mode`` change.

        **Idempotent** — re-running after slots are allocated just
        refreshes visible children + tab titles + counter HTMLs
        without reallocating widgets. The ``if 'sel' not in d:``
        guard handles first-time allocation per slot:
          * **First-time**: call ``self._init_circuit_slot(i, opts)``
            (allocates ~15 widgets into ``state.layouts_data[i]`` +
            wires callbacks), then hydrate active role's controls
            from group memory.
          * **Re-render**: just re-seed the selector options (slot
            may have been built in a different mode — LUCJ vs
            General — so options must match current ``dd_mode``).

        Role-selector options: LUCJ gets α/β/Anc/Other/Remove,
        General gets Qubits/Anc/Remove.
        ``counter_html`` is lazy-allocated here (not in
        ``_init_circuit_slot``) because the dynamic UI may need it
        before any user click.

        After the loop: assign ``self.manual_tabs.children``, label
        tabs ``C1`` / ``C2`` / …, force a counter refresh (slot
        dict may carry stale counts from a previous nav switch),
        and refresh ancilla-tool options for the new circuit count.
        """
        mode = self.dd_mode.value
        n_circ = self.num_circuits.value if self.intra_qpu_mode.value else 1

        tab_children = []
        for i in range(n_circ):
            d = self.state.layouts_data[i]
            opts = (
                ['Select Alpha', 'Select Beta', 'Select Ancilla', 'Select Other', 'Remove']
                if mode == 'lucj' else
                ['Select Qubits', 'Select Ancilla', 'Remove']
            )

            if 'counter_html' not in d:
                d['counter_html'] = HTML()

            if 'sel' not in d:
                # First-time allocation for this slot — widgets land in
                # state.layouts_data[i]; see _init_circuit_slot.
                self._init_circuit_slot(i, opts)
                _init_g = self._active_group(d['sel'].value)
                if _init_g:
                    self._load_group_to_controls(i, _init_g)
            else:
                d['sel'].options = opts
                if d['sel'].value not in opts:
                    d['sel'].value = 'Remove'

            self._update_manual_swatches(i)
            self._update_manual_preview(i)
            self._update_preview_label(i)

            tab_children.append(VBox([
                d['counter_html'], d['sel'],
                HTML("<hr style='margin:4px 0;'>"),
                Label("Fill Color:", style={'font_size': '11px', 'font_weight': 'bold'}),
                d['swatches_fill'], d['btn_fill_toggle'], d['fill_advanced_box'],
                Label("Line Color:", style={'font_size': '11px', 'font_weight': 'bold'}),
                d['swatches_line'], d['btn_line_toggle'], d['line_advanced_box'],
                HTML("<hr style='margin:4px 0;'>"),
                d['sl_o'], d['sl_w'],
                HBox([d.get('preview_label', Label("Preview:")), d['preview']]),
                HBox([d['btn_upd'], d['btn_reset_circ']],
                     layout=Layout(justify_content='space-between')),
            ], layout=Layout(padding='6px')))

        self.manual_tabs.children = tab_children
        for i in range(n_circ):
            self.manual_tabs.set_title(i, f"C{i+1}")

        self._update_all_counters()

        self.ancilla_tool.options = [('View Only', None)] + [
            (f"Add Ancilla C{i+1}", i) for i in range(n_circ)
        ]

    def _init_circuit_slot(self, idx, opts):
        """Allocate the per-slot widgets + wire callbacks for circuit ``idx``.

        Only invoked by ``self._build_dynamic_ui`` when ``'sel' not
        in d`` — the idempotency guard lives at the call site, not
        here.

        ``opts`` is the RadioButtons option list for the current
        mode (LUCJ vs General); passed in so the widget is created
        with the right options ON FIRST CONSTRUCTION rather than
        created empty and reassigned (which would needlessly fire
        the value observer during init).

        Allocates ~20 widgets into ``self.state.layouts_data[idx]``
        (role selector, fill/line pickers + swatches + advanced
        boxes, opacity, line width, text colour, preview, Apply/
        Reset buttons) and wires the 13 per-slot callbacks via
        ``self._mk_*`` / ``self._make_on_*`` factories.
        """
        W = Layout(width='100%', margin='2px 0px', min_width='0px')
        d = self.state.layouts_data[idx]
        d['sel'] = RadioButtons(options=opts, layout=W)
        d['swatches_fill'] = HBox(
            layout=Layout(flex_flow='row wrap', width='100%'))
        d['cp_fill_advanced'] = ColorPicker(
            value=d['defaults']['fill'], description='',
            layout=Layout(width='70px'))
        d['btn_fill_add'] = Button(
            description='Add', button_style='success',
            layout=Layout(width='50px'))
        d['btn_fill_toggle'] = Button(
            description='+ More', button_style='info',
            layout=Layout(width='100%', margin='2px 0'), icon='plus')
        d['fill_advanced_box'] = HBox(
            [Label('Custom:', layout=Layout(width='50px')),
             d['cp_fill_advanced'], d['btn_fill_add']],
            layout=Layout(margin='3px 0', display='none'),
        )
        d['swatches_line'] = HBox(
            layout=Layout(flex_flow='row wrap', width='100%'))
        d['cp_line_advanced'] = ColorPicker(
            value=d['defaults']['line'], description='',
            layout=Layout(width='70px'))
        d['btn_line_add'] = Button(
            description='Add', button_style='success',
            layout=Layout(width='50px'))
        d['btn_line_toggle'] = Button(
            description='+ More', button_style='info',
            layout=Layout(width='100%', margin='2px 0'), icon='plus')
        d['line_advanced_box'] = HBox(
            [Label('Custom:', layout=Layout(width='50px')),
             d['cp_line_advanced'], d['btn_line_add']],
            layout=Layout(margin='3px 0', display='none'),
        )
        d['current_fill'] = d['defaults']['fill']
        d['current_line'] = d['defaults']['line']
        d['current_text_color'] = '#000000'
        d['cp_text_color'] = ColorPicker(
            value='#000000', description='', layout=Layout(width='70px'))
        d['sl_o'] = FloatSlider(
            value=1.0, min=0, max=1, step=0.05,
            description='Opacity:', layout=Layout(width='98%'))
        # Width 1-10: numeric input over a slider in the narrow sidebar.
        d['sl_w'] = BoundedIntText(
            value=2, min=1, max=10, description='Width:',
            layout=Layout(width='180px'),
            style={'description_width': '70px'})
        d['preview'] = HTML()
        d['preview_label'] = HTML(
            "<span style='font-size:15px;font-weight:bold;color:#2980B9;'>"
            "Preview: Alpha</span>")
        d['btn_upd'] = Button(
            description='✅ Apply to Selection',
            button_style='info', layout=W)
        d['btn_reset_circ'] = Button(
            description=f'🗑️ Reset C{idx + 1}', button_style='warning',
            layout=Layout(width='100%', margin='2px 0'))

        # Swatch builders used by _update_manual_swatches per colour.
        d['create_fill_swatch'] = lambda color: self._mk_fill_swatch(idx, color)
        d['create_line_swatch'] = lambda color: self._mk_line_swatch(idx, color)

        # Wire per-slot observers. Each factory takes idx and returns
        # the inner closure that's actually registered.
        d['btn_fill_toggle'].on_click(self._mk_fill_toggle(idx))
        d['btn_line_toggle'].on_click(self._mk_line_toggle(idx))
        d['btn_fill_add'].on_click(self._mk_fill_add(idx))
        d['btn_line_add'].on_click(self._mk_line_add(idx))
        d['cp_fill_advanced'].observe(self._make_on_fill_change(idx), 'value')
        d['cp_line_advanced'].observe(self._make_on_line_change(idx), 'value')
        d['sl_o'].observe(self._make_on_preview_only(idx), 'value')
        d['sl_w'].observe(self._make_on_preview_only(idx), 'value')
        d['cp_text_color'].observe(self._make_on_text_color_change(idx), 'value')
        d['sel'].observe(self._mk_sel_obs(idx), names='value')
        d['btn_upd'].on_click(self._mk_on_apply(idx))
        d['btn_reset_circ'].on_click(self._mk_on_reset_circ(idx))

    # ── Per-circuit-slot factory methods ─────────────────────────────
    # Used by _init_circuit_slot. Each takes ``idx`` and either returns a
    # Button eagerly (the two _mk_*_swatch) or returns an inner closure
    # (the other 11). The inner closures MUST stay nested defs (not bound
    # methods) — each ``self._mk_X(idx)`` must produce a NEW closure
    # bound to that idx; a bound ``self._on_X(idx, change)`` wouldn't fit
    # ipywidgets' single-arg observer contract.

    def _mk_fill_swatch(self, idx, color):
        """Build one fill-colour swatch button for circuit ``idx``.
        Eager factory — returns the Button directly. The 2px highlight
        border marks the active fill so the user sees which is selected.
        """
        btn = Button(
            layout=Layout(width='24px', height='24px', margin='1px'),
            tooltip=color)
        btn.style.button_color = color
        btn._swatch_hex = color   # canonical key for _sync_manual_swatch_borders
        # Constant 2px border (only colour changes on selection) so the
        # swatch never resizes — no horizontal reflow of neighbours.
        btn.layout.border = (
            '2px solid #2980B9'
            if color == self.state.layouts_data[idx]['current_fill']
            else '2px solid #bdc3c7'
        )

        def on_click(b):
            # Set current_fill BEFORE the picker value — the picker's
            # observer guard then sees new == current_fill and no-ops,
            # avoiding the old double-rebuild. Borders flip in place.
            self.state.layouts_data[idx]['current_fill'] = color
            self.state.layouts_data[idx]['cp_fill_advanced'].value = color
            self._sync_manual_swatch_borders(idx)
            self._update_manual_preview(idx)

        btn.on_click(on_click)
        return btn

    def _mk_line_swatch(self, idx, color):
        """Symmetric to :meth:`_mk_fill_swatch` for the line side."""
        btn = Button(
            layout=Layout(width='24px', height='24px', margin='1px'),
            tooltip=color)
        btn.style.button_color = color
        btn._swatch_hex = color   # canonical key for _sync_manual_swatch_borders
        btn.layout.border = (
            '2px solid #2980B9'
            if color == self.state.layouts_data[idx]['current_line']
            else '2px solid #bdc3c7'
        )

        def on_click(b):
            self.state.layouts_data[idx]['current_line'] = color
            self.state.layouts_data[idx]['cp_line_advanced'].value = color
            self._sync_manual_swatch_borders(idx)
            self._update_manual_preview(idx)

        btn.on_click(on_click)
        return btn

    def _mk_fill_toggle(self, idx):
        """on_click handler — toggle the per-slot 'fill +More' advanced box."""
        def toggle(b):
            dd = self.state.layouts_data[idx]
            if dd['fill_advanced_box'].layout.display == 'none':
                dd['fill_advanced_box'].layout.display = 'flex'
                dd['btn_fill_toggle'].description = '- Hide'
            else:
                dd['fill_advanced_box'].layout.display = 'none'
                dd['btn_fill_toggle'].description = '+ More'

        return toggle

    def _mk_line_toggle(self, idx):
        """Symmetric to :meth:`_mk_fill_toggle` for the line side."""
        def toggle(b):
            dd = self.state.layouts_data[idx]
            if dd['line_advanced_box'].layout.display == 'none':
                dd['line_advanced_box'].layout.display = 'flex'
                dd['btn_line_toggle'].description = '- Hide'
            else:
                dd['line_advanced_box'].layout.display = 'none'
                dd['btn_line_toggle'].description = '+ More'

        return toggle

    def _mk_fill_add(self, idx):
        """on_click handler — append the picker colour to the global custom palette + activate it.

        Persists ``state.custom_colors`` to disk via
        ``save_custom_colors``.
        """
        def add(b):
            dd = self.state.layouts_data[idx]
            nc = dd['cp_fill_advanced'].value
            if nc not in STANDARD_COLORS and nc not in self.state.custom_colors:
                self.state.custom_colors.append(nc)
                save_custom_colors(self.state.custom_colors)
            dd['current_fill'] = nc
            self._update_manual_swatches(idx)
            self._update_manual_preview(idx)

        return add

    def _mk_line_add(self, idx):
        """Symmetric to :meth:`_mk_fill_add` for the line side."""
        def add(b):
            dd = self.state.layouts_data[idx]
            nc = dd['cp_line_advanced'].value
            if nc not in STANDARD_COLORS and nc not in self.state.custom_colors:
                self.state.custom_colors.append(nc)
                save_custom_colors(self.state.custom_colors)
            dd['current_line'] = nc
            self._update_manual_swatches(idx)
            self._update_manual_preview(idx)

        return add

    def _make_on_fill_change(self, idx):
        """Observer for the per-slot fill ColorPicker — mirror to current_fill + refresh."""
        def _on_change(change):
            d = self.state.layouts_data[idx]
            # Re-entry guard: a swatch click sets current_fill first, then the
            # picker value — so by the time we fire here new == current_fill and
            # we no-op. Real work happens only when the change came from the
            # +More picker directly (current_fill still the old value).
            if change['new'] == d.get('current_fill'):
                return
            d['current_fill'] = change['new']
            self._sync_manual_swatch_borders(idx)
            self._update_manual_preview(idx)

        return _on_change

    def _make_on_line_change(self, idx):
        """Symmetric to :meth:`_make_on_fill_change` for the line side."""
        def _on_change(change):
            d = self.state.layouts_data[idx]
            if change['new'] == d.get('current_line'):
                return
            d['current_line'] = change['new']
            self._sync_manual_swatch_borders(idx)
            self._update_manual_preview(idx)

        return _on_change

    def _make_on_preview_only(self, idx):
        """Observer for ``sl_o`` / ``sl_w`` — redraws the preview only.

        Both sliders affect ONLY the preview-dot rendering; per-qubit
        styles are written by ``_mk_on_apply`` on Apply click. So
        this observer just calls ``_update_manual_preview`` and bails.
        """
        def _on_change(change):
            self._update_manual_preview(idx)

        return _on_change

    def _make_on_text_color_change(self, idx):
        """Observer for the per-slot text-colour picker — stores value, no refresh.

        Text color only matters when applied via ``_mk_on_apply`` or
        cascaded globally via ``self._on_global_text_color``.
        """
        def _on_change(change):
            self.state.layouts_data[idx]['current_text_color'] = change['new']

        return _on_change

    def _mk_sel_obs(self, idx):
        """Observer for the per-slot role selector — save OLD role + load NEW.

        Save-then-load ensures the previous role's tweaks survive
        the swap. ``change.get('old', '')`` defaults to '' on the
        first-ever fire; ``_active_group`` returns ``None`` for
        ``'Remove'`` / unknown labels, gating the save/load calls.
        """
        def _on_switch(change):
            old_g = self._active_group(change.get('old', ''))
            new_g = self._active_group(change['new'])
            if old_g:
                self._save_controls_to_group(idx, group=old_g)
            if new_g:
                self._load_group_to_controls(idx, new_g)
            self._update_preview_label(idx)

        return _on_switch

    def _mk_on_apply(self, idx):
        """on_click handler — commit the active style to every qubit in the active role.

        Respects ``self.chk_line_only`` (when ticked, every qubit
        gets ``fill='none'`` so only outlines show). Snapshots the
        active widget state into group memory via
        ``self._save_controls_to_group`` and refreshes. ``Remove``
        is a no-op.
        """
        def _on_upd(b):
            tag = self.state.layouts_data[idx]['sel'].value.split(' ')[-1].lower()
            if tag == 'remove':
                return
            dd = self.state.layouts_data[idx]
            fill_val = ('none' if self.chk_line_only.value
                        else hex_to_rgba(dd['current_fill'], dd['sl_o'].value))
            for q in dd[tag]:
                dd['styles'][q] = {
                    'fill': fill_val, 'line': dd['current_line'],
                    'width': dd['sl_w'].value,
                    'text_color': dd['current_text_color'],
                }
            self._save_controls_to_group(idx)
            self._refresh()

        return _on_upd

    def _mk_on_reset_circ(self, idx):
        """on_click handler — clear every qubit assignment for circuit ``idx`` (destructive, undo-pushed)."""
        def _on_reset(b):
            self.state.push_undo()
            self.state.clear_circuit(idx)
            self._refresh()
            self._get_export()

        return _on_reset

    def _build_section_cards(self):
        """Construct the two LQS-sibling card containers.

          * ``self.inspector_card`` — Section A
            (🔍 Current calibration inspector): opt-in checkbox +
            body VBox.
          * ``self.stability_card`` — Section B (⚠️ Stability check):
            same shape.

        Styled to match the LQS banner so the three render as
        stacked sibling cards. MUST run after
        ``_build_inspector_widgets``.
        """
        card_layout = Layout(
            border='1px solid #dee2e6',
            padding='10px 12px', margin='6px 0',
        )
        header_style = (
            'font-size:18px;font-weight:bold;color:#1a5276;'
            'margin:0 0 6px 0;line-height:1.4;'
        )
        self.inspector_card = VBox([
            HTML(f"<div style='{header_style}'>"
                 f"🔍 Current calibration inspector</div>"),
            self.chk_inspect_enable,
            self.inspector_body_box,
        ], layout=card_layout)
        self.stability_card = VBox([
            HTML(f"<div style='{header_style}'>"
                 f"⚠️ Stability check</div>"),
            self.chk_stability_enable,
            self.stability_body_box,
        ], layout=card_layout)

    # ── Final view assembly ──────────────────────────────────────────
    # Three methods that build the complete UI VBox/HBox tree. Called
    # from the tail of `_run_pipeline` after every `_build_*` method
    # has populated the corresponding self.* widget. Pure assembly —
    # no state mutation, no observer wiring, no callbacks fired.
    #
    # `self._sidebar_body` is the swap container whose `.children` is
    # mutated by `ui.inter_qpu.setup_inter_qpu` when the user toggles
    # inter-QPU mode — filled with `[ui_sidebar_content]` here and
    # later swapped to `[qpu2_sidebar_content]` when QPU 2 connects.

    def _assemble_sidebar(self):
        """Build the QPU 1 sidebar VBox.

        Composes the 4-section Accordion (Hardware / Display /
        Colors / Multi-Programming — the last branches on
        ``self.is_secondary`` since a secondary panel cannot host
        its own tertiary), title + CSS + divider, undo/redo +
        clear-all/refresh rows, LQS banner, inspector + stability
        cards, nav bar, ancilla tool, content box, Output log.

        Populates ``self._sidebar_body.children = [ui_sidebar_content]``
        — the same instance is wrapped in the 720px outer
        ``ui_sidebar`` VBox here. Returns ``(ui_sidebar,
        ui_sidebar_content)``: both needed by ``setup_inter_qpu``.
        """
        # Accordion settings — inter-QPU controls live in Hardware section
        acc_settings = Accordion(children=[
            VBox([
                self.dd_backend_switch, self.lbl_switch_status,
                HBox([self.dp_date, self.txt_hw_hour]),
                self.lbl_calibration_date,
                self.lbl_props_warning,
                self.chk_keep_layout,
            ]),
            VBox([self.view_mode, self.size_slider, self.gate_size_slider, self.show_numbers, self.font_size_slider,
                  self.global_text_color,
                  self.chk_link_gates,
                  self.plot_w_slider, self.plot_h_slider]),
            VBox([self.pp_bg_fill.widget, self.pp_bg_line.widget, self.pp_gate_col.widget]),
            # 4th section: intra-QPU controls always shown; inter-QPU
            # sub-block is a static notice on the secondary panel (can't
            # host its own tertiary), full controls on the primary.
            VBox([
                HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>"
                     "🔀 Multi-Programming (multiple circuits on same device)</div>"),
                HTML("<div style='font-size:14px;color:#6c757d;margin-bottom:4px;'>"
                     "Run multiple sub-circuits in parallel on one QPU.</div>"),
                HBox([self.intra_qpu_mode, self.num_circuits]),
                HTML("<hr style='margin:6px 0; border-color:#dee2e6;'>"),
                HTML("<div style='font-size:15px;font-weight:bold;color:#1a5276;margin:4px 0;'>"
                     "🔗 Inter-QPU (circuits across different devices)</div>"),
            ] + (
                [HTML("<div style='font-size:14px;color:#6c757d;margin-bottom:4px;'>"
                      "<i>This is the secondary panel — Inter-QPU is controlled "
                      "from QPU 1's sidebar.</i></div>")]
                if self.is_secondary else
                [HTML("<div style='font-size:14px;color:#6c757d;margin-bottom:4px;'>"
                      "Compare or distribute circuits across two QPUs.</div>"),
                 self.chk_inter_qpu,
                 self.dd_backend_qpu2, self.btn_connect_qpu2, self.lbl_qpu2_status]
            )),
            # 5th section: benchmarking. A property of the backend, so it
            # belongs beside Hardware rather than inside the Auto search.
            VBox([
                HTML("<div style='font-size:15px;font-weight:bold;"
                     "color:#1a5276;margin:4px 0;'>📡 Real-time benchmarking"
                     "</div>"),
                self.bench_box,
            ]),
        ])
        acc_settings.set_title(0, '1. Hardware')
        acc_settings.set_title(1, '2. Display')
        acc_settings.set_title(2, '3. Colors')
        acc_settings.set_title(3, '4. Multi-Programming')
        acc_settings.set_title(4, '5. Benchmark')

        sidebar_title = HTML(
            f"<h3 style='margin:0 0 6px 0; color:#2C3E50;'>🎛️ QPU Map Editor</h3>"
            f"<div style='font-size:15px; color:#6c757d; margin-bottom:8px;'>{cat_paws_html()}</div>"
        )
        sidebar_hr = HTML("<hr style='margin:8px 0; border-color:#dee2e6;'>")

        # Stylesheet scoped to .iqmapper-sidebar (add_class below) so
        # it doesn't leak. See _sidebar_css.py for the per-rule rationale.
        _sidebar_css = HTML(SIDEBAR_CSS)

        # QPU 1 sidebar content
        ui_sidebar_content = VBox([
            _sidebar_css,
            sidebar_title,
            acc_settings,
            sidebar_hr,
            HBox([self.btn_undo, self.btn_redo], layout=Layout(margin='2px 0', justify_content='flex-start')),
            HBox([self.btn_clear_all, self.btn_refresh], layout=Layout(margin='2px 0')),
            self.lbl_fidelity,
            self.inspector_card,
            self.stability_card,
            self.nav_bar,
            self.ancilla_tool,
            self.content_box,
            self.lbl_notify,
            self.out,
        ], layout=Layout(padding='10px'))

        # Tag container so the scoped CSS selectors above apply.
        ui_sidebar_content.add_class('iqmapper-sidebar')

        # Populate the swappable sidebar body now that ui_sidebar_content exists
        self._sidebar_body.children = [ui_sidebar_content]

        # QPU 1 layout — flat HBox, exactly like the working version
        ui_sidebar = VBox([
            self._sidebar_body,
        ], layout=Layout(width='720px', padding='5px',
                         border_right='2px solid #ddd'))

        return ui_sidebar, ui_sidebar_content

    def _assemble_figure_box(self):
        """Build the figure-side VBox: ``[lbl_fig1, zoom_bar, self.fig]``.

        Creates ``self.lbl_fig1`` — the green "📡 <backend>" pill
        above the FigureWidget when inter-QPU mode is active
        (hidden by default; ``self._refresh`` toggles visibility
        based on ``self.chk_inter_qpu.value``).
        """
        self.lbl_fig1 = HTML(
            f"<div style='font-size:14px;font-weight:bold;color:#fff;background:#2C3E50;"
            f"padding:3px 10px;border-radius:4px;text-align:center;margin-bottom:2px;'>"
            f"📡 {esc(self.state.active_backend_name or 'QPU 1')}</div>",
            layout=Layout(display='none'),
        )

        zoom_bar = HBox(
            [self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_rst,
             HTML("<span style='font-size:14px;color:#888;margin-left:6px;'>Zoom QPU</span>")],
            layout=Layout(margin='4px 0 0 4px'),
        )

        fig_box = VBox([self.lbl_fig1, zoom_bar, self.fig])
        return fig_box

    def _assemble_full_ui(self):
        """Top-level assembly: sidebar + figure-box + QPU 2 placeholders.

        Returns the 6-tuple ``(full_ui, ui_sidebar,
        ui_sidebar_content, fig_box, qpu2_sidebar_box,
        qpu2_fig_box)`` — all six are needed by
        ``setup_inter_qpu(_NS(...))`` in ``_run_pipeline``.
        Returning a tuple (vs. promoting all 6 to ``self.*``) avoids
        bloating the instance with transient build artifacts.

        The QPU 2 placeholder VBoxes are empty until QPU 2 connects;
        the sidebar box is pre-tagged with ``.iqmapper-sidebar``
        so the scoped CSS applies on append (otherwise QPU 2 widgets
        would render at the default ipywidgets font size and look
        inconsistent with the QPU 1 panel).
        """
        ui_sidebar, ui_sidebar_content = self._assemble_sidebar()
        fig_box = self._assemble_figure_box()

        qpu2_sidebar_box = VBox(layout=Layout(width='720px', padding='5px',
                                              border_right='2px solid #ddd'))
        qpu2_sidebar_box.add_class('iqmapper-sidebar')
        qpu2_fig_box = VBox()

        # ui_sidebar & fig_box NEVER move — QPU2 boxes are appended/removed.
        full_ui = HBox([ui_sidebar, fig_box],
                       layout=Layout(width='100%', align_items='flex-start'))

        return full_ui, ui_sidebar, ui_sidebar_content, fig_box, qpu2_sidebar_box, qpu2_fig_box


def build_mapper_ui(state, is_secondary=False, return_widgets=False):
    """Backward-compatible entry point.

    The 2-tuple ``(ui_hbox, fig)`` return is preserved by default, so every
    existing caller (``mapper.py``, notebooks, tests) keeps working unchanged.
    Pass ``return_widgets=True`` to also get the ``WidgetsRef`` as a third
    element — used by ``ui.inter_qpu`` to drive QPU2's fig-size sliders.
    Internally delegates to :class:`MapperUIBuilder`.
    """
    return MapperUIBuilder(state, is_secondary).build(return_widgets=return_widgets)
