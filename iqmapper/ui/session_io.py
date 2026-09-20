"""
ui.session_io.py — session and style-preset save/load wiring.

Hosts the (collect → save) and (load → apply) round trips for both
`.qpusession.json` and `.qpustyle.json`, the "click again to overwrite"
gate on existing files, the diagnostic banner, and the post-load graph
rebuild. The ctx SimpleNamespace mirrors the pattern used by the other
setup_*_callbacks modules (ui.lucj_panel, ui.history, ui.analysis_panel).
"""
import os

from ..html_safe import esc
from .._upload import read_upload
from .status import status_html


def setup_session_io(ctx):
    """Wire session-save / session-load / style-save / style-load callbacks.

    Required ctx fields:
      state, widgets_ref           — SessionState and the WidgetsRef bundle
      dd_colorscale, dd_mode, export_format
                                   — dropdowns not present in WidgetsRef
      pp_bg_fill, pp_bg_line, pp_gate_col, global_text_color
                                   — colour widgets (need .set_color() / .value)
      txt_session_path, btn_save_session, btn_load_session, lbl_session_status
      txt_style_path,   btn_save_style,   btn_load_style
      fig, on_plot_click            — figure to rebuild on session load
      export_mod, plot_mod          — modules for save_*/load_* and graph rebuild
      safe_path, out, refresh, get_export
                                   — shared helpers from build_mapper_ui

    Returns a dict of helpers (currently empty) — kept for symmetry with
    the other setup_*_callbacks() so future helpers can be exposed without
    touching the call site.
    """
    state = ctx.state
    w = ctx.widgets_ref
    dd_colorscale = ctx.dd_colorscale
    dd_mode = ctx.dd_mode
    export_format = ctx.export_format
    pp_bg_fill = ctx.pp_bg_fill
    pp_bg_line = ctx.pp_bg_line
    pp_gate_col = ctx.pp_gate_col
    global_text_color = ctx.global_text_color
    txt_session_path = ctx.txt_session_path
    btn_save_session = ctx.btn_save_session
    flush_controls_to_groups = ctx.flush_controls_to_groups
    btn_load_session = ctx.btn_load_session
    resync_group_controls = ctx.resync_group_controls
    lbl_session_status = ctx.lbl_session_status
    lbl_session_save_status = ctx.lbl_session_save_status
    txt_style_path = ctx.txt_style_path
    btn_save_style = ctx.btn_save_style
    btn_load_style = ctx.btn_load_style
    lbl_style_status = ctx.lbl_style_status
    fig = ctx.fig
    on_plot_click = ctx.on_plot_click
    export_mod = ctx.export_mod
    plot_mod = ctx.plot_mod
    user_save_path = ctx.user_save_path
    save_dir = ctx.save_dir
    out = ctx.out
    refresh = ctx.refresh
    get_export = ctx.get_export

    # ── Helpers ────────────────────────────────────────────────────

    def _collect_widget_values():
        """Snapshot every widget value the session JSON should persist.

        Includes display widgets (sliders, checkboxes, dropdowns), the
        ansatz mode, threshold filter, export format, and the colour
        pickers. Calibration data and layouts are persisted by
        SessionState.to_session_dict — this dict only holds UI widgets.
        """
        return {
            'view_mode': w.view_mode.value,
            'size_slider': w.size_slider.value,
            'gate_size_slider': w.gate_size_slider.value,
            'gate_hl_w_slider': w.gate_hl_w_slider.value,
            'font_size_slider': w.font_size_slider.value,
            'show_numbers': w.show_numbers.value,
            'chk_link_gates': w.chk_link_gates.value,
            'chk_line_only': w.chk_line_only.value,
            'plot_w_slider': w.plot_w_slider.value,
            'plot_h_slider': w.plot_h_slider.value,
            'dd_qubit_prop': w.dd_qubit_prop.value,
            'dd_gate_prop': w.dd_gate_prop.value,
            'dd_mode': dd_mode.value,
            'intra_qpu_mode': w.intra_qpu_mode.value,
            'num_circuits': w.num_circuits.value,
            'bg_fill': pp_bg_fill.value,
            'bg_line': pp_bg_line.value,
            'gate_col': pp_gate_col.value,
            'global_text_color': global_text_color.value,
            # Threshold settings
            'chk_err_threshold': w.chk_err_threshold.value,
            'txt_q_thr_min': w.txt_q_thr_min.value,
            'txt_q_thr_max': w.txt_q_thr_max.value,
            'txt_g_thr_min': w.txt_g_thr_min.value,
            'txt_g_thr_max': w.txt_g_thr_max.value,
            # Export format
            'export_format': export_format.value,
        }

    def _apply_widget_values(wv):
        """Restore widget values from a session/style dict, accumulating
        any setter failures into a single warning instead of aborting."""
        if not wv:
            return
        failed = []
        # Trait widgets — all use .value
        for attr, widget in [
            ('view_mode', w.view_mode),
            ('size_slider', w.size_slider),
            ('gate_size_slider', w.gate_size_slider),
            ('gate_hl_w_slider', w.gate_hl_w_slider),
            ('font_size_slider', w.font_size_slider),
            ('show_numbers', w.show_numbers),
            ('chk_link_gates', w.chk_link_gates),
            ('chk_line_only', w.chk_line_only),
            ('plot_w_slider', w.plot_w_slider),
            ('plot_h_slider', w.plot_h_slider),
            ('dd_qubit_prop', w.dd_qubit_prop),
            ('dd_gate_prop', w.dd_gate_prop),
            ('dd_mode', dd_mode),
            ('intra_qpu_mode', w.intra_qpu_mode),
            ('num_circuits', w.num_circuits),
            ('chk_err_threshold', w.chk_err_threshold),
            ('export_format', export_format),
        ]:
            if attr in wv:
                try:
                    widget.value = wv[attr]
                except Exception as e:
                    failed.append(f"{attr}: {e}")
        # Threshold text fields — values may be int/float in older sessions
        for attr, widget in [
            ('txt_q_thr_min', w.txt_q_thr_min),
            ('txt_q_thr_max', w.txt_q_thr_max),
            ('txt_g_thr_min', w.txt_g_thr_min),
            ('txt_g_thr_max', w.txt_g_thr_max),
        ]:
            if attr in wv:
                try:
                    widget.value = str(wv[attr])
                except Exception as e:
                    failed.append(f"{attr}: {e}")
        # Color pickers — set via dedicated method that updates the swatch
        # row and triggers the on_change callback.
        if 'bg_fill' in wv:
            pp_bg_fill.set_color(wv['bg_fill'])
        if 'bg_line' in wv:
            pp_bg_line.set_color(wv['bg_line'])
        if 'gate_col' in wv:
            pp_gate_col.set_color(wv['gate_col'])
        if 'global_text_color' in wv:
            try:
                global_text_color.value = wv['global_text_color']
            except Exception as e:
                failed.append(f"global_text_color: {e}")
        if failed:
            with out:
                print(f"⚠️ Session restore: {len(failed)} setting(s) "
                      f"could not be applied: {', '.join(failed)}")

    # ── Session save/load ──────────────────────────────────────────
    # "Click again to overwrite" gate: first click on an existing file
    # raises the warning banner and arms this flag; the next click goes
    # through. Reset whenever the path text changes.
    _session_overwrite_pending = [False]

    def _on_save_session(b):
        try:
            raw = txt_session_path.value.strip()
            if not raw.endswith('.json'):
                raw += '.qpusession.json'
                txt_session_path.value = raw
            path = user_save_path(raw, '.qpusession.json', default_dir=save_dir())
        except ValueError as e:
            lbl_session_save_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )
            return

        if os.path.exists(path) and not _session_overwrite_pending[0]:
            _session_overwrite_pending[0] = True
            # `path` is sanitised by safe_path (CWD-relative, no traversal),
            # but its basename can still contain user-supplied characters
            # — escape before interpolating into HTML.
            lbl_session_save_status.value = (
                f"<div style='font-size:15px;color:#e67e22;background:#fef9e7;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #f5cba7;"
                f"margin:3px 0;'>⚠️ <b>{esc(os.path.basename(path))}</b> "
                f"already exists. Click Save again to overwrite.</div>"
            )
            return

        _session_overwrite_pending[0] = False
        try:
            flush_controls_to_groups()   # commit the active role's in-progress edit so it's saved
            export_mod.save_session_to_path(
                state, path, widget_values=_collect_widget_values())
            lbl_session_save_status.value = (
                f"<div style='font-size:15px;color:#1e8449;background:#eafaf1;"
                f"padding:4px 8px;border-radius:4px;border:1px solid #abebc6;"
                f"margin:3px 0;'>✅ Session saved: <b>{esc(path)}</b></div>"
            )
        except Exception as e:
            lbl_session_save_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )

    txt_session_path.observe(
        lambda c: _session_overwrite_pending.__setitem__(0, False), 'value')
    btn_save_session.on_click(_on_save_session)

    def _on_load_session(change):
        if not change['new']:
            return
        try:
            # _upload.read_upload normalises ipywidgets v7 (dict-of-dicts)
            # and v8 (tuple-of-dicts) payloads and raises ValueError on
            # empty/malformed input so the user sees a clear message instead
            # of a downstream IndexError.
            content, _filename = read_upload(change['new'])
            backend_name, wv = export_mod.load_session(state, content)
            # load_session already set state.current_colorscale; sync the
            # dropdown widget so the user sees the selection.
            dd_colorscale.value = state.current_colorscale
            _apply_widget_values(wv)
            resync_group_controls()      # pull restored per-role colours into the Layout controls + preview
            # Rebuild graph from loaded data (no API call needed).
            plot_mod.build_device_graph(fig, state, w)
            # Look up the qubit trace by its stable .meta — same robust
            # binding as builder.py's connect path. .uid is regenerated
            # by FigureWidget, so .meta is the stable identifier.
            q_trace = plot_mod._trace_by_meta(fig, plot_mod._META_QUBITS)
            if q_trace is not None:
                q_trace.on_click(on_plot_click)
            refresh()
            get_export()
            reconnect_msg = ""
            if state.active_backend is None:
                reconnect_msg = (
                    f"<br><div style='font-size:15px;color:#856404;"
                    f"background:#fff3cd;padding:6px 10px;border-radius:5px;"
                    f"border:1px solid #ffc107;margin:4px 0;'>"
                    f"⚠️ <b>Offline session: no backend connected.</b> "
                    f"Live data needs a connection "
                    f"· README: Offline mode<br>"
                    f"To reconnect: select <b>'{esc(backend_name or 'your backend')}'</b> "
                    f"from the Backend dropdown above.</div>"
                )
            elif backend_name and backend_name != state.active_backend_name:
                reconnect_msg = (
                    f"<br><span style='font-size:15px;color:#e67e22;'>"
                    f"⚠️ Session was saved from <b>'{esc(backend_name)}'</b> "
                    f"but current backend is "
                    f"<b>'{esc(state.active_backend_name)}'</b>. "
                    f"Calibration data is from the session file.</span>"
                )
            lbl_session_status.value = (
                f"<span style='font-size:15px;color:#27ae60;'>"
                f"✅ Session loaded{reconnect_msg}</span>"
            )
            btn_load_session.value = ()
        except Exception as e:
            lbl_session_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ {esc(str(e))}</span>"
            )

    btn_load_session.observe(_on_load_session, names='value')

    # ── Style preset save/load ─────────────────────────────────────
    _style_overwrite_pending = [False]

    def _on_save_style(b):
        try:
            raw = txt_style_path.value.strip()
            if not raw.endswith('.json'):
                raw += '.qpustyle.json'
                txt_style_path.value = raw
            path = user_save_path(raw, '.qpustyle.json', default_dir=save_dir())
        except ValueError as e:
            lbl_style_status.value = status_html('error', str(e))
            return

        if os.path.exists(path) and not _style_overwrite_pending[0]:
            _style_overwrite_pending[0] = True
            base = os.path.basename(path)
            lbl_style_status.value = status_html(
                'warn', f"'{base}' already exists. Click Save again to overwrite.",
                bold=base)
            return

        _style_overwrite_pending[0] = False
        try:
            export_mod.save_style_to_path(
                state, path, widget_values=_collect_widget_values())
            lbl_style_status.value = status_html(
                'ok', f"Style preset saved: {path}", bold=path)
        except Exception as e:
            lbl_style_status.value = status_html('error', f"Style save error: {e}")

    txt_style_path.observe(
        lambda c: _style_overwrite_pending.__setitem__(0, False), 'value')
    btn_save_style.on_click(_on_save_style)

    def _on_load_style(change):
        if not change['new']:
            return
        try:
            content, _filename = read_upload(change['new'])
            wv = export_mod.load_style_preset(state, content)
            dd_colorscale.value = state.current_colorscale
            _apply_widget_values(wv)
            refresh()
            with out:
                print("✅ Style preset loaded")
            btn_load_style.value = ()
        except Exception as e:
            with out:
                print(f"❌ Style load error: {e}")

    btn_load_style.observe(_on_load_style, names='value')

    return {
        'collect_widget_values': _collect_widget_values,
        'apply_widget_values': _apply_widget_values,
    }
