"""
ui.history.py — history fetch, slider, and time-lapse export callbacks.

Reads reference calibration from state.ref_qubit_props / .ref_gate_props /
.ref_timestamp / .delta_enabled — these are always present on SessionState
(v5+). Earlier these were added lazily by ui.time_panel, causing
AttributeError if history was fetched before entering error_inspect; now
centralised in SessionState.__init__.
"""
import logging
import os

import numpy as np
from ipywidgets import SelectionSlider, Play, IntSlider, HBox, HTML

from ..html_safe import esc
from .status import status_html

_log = logging.getLogger(__name__)

# Per-frame display time in the exported GIF (seconds). 0.5s ≈ 2 fps —
# readable date overlay without a 30-day history running a full minute.
# imageio applies this to *every* frame; no per-frame variation here.
GIF_FRAME_DURATION_S = 0.5


def setup_history_callbacks(ctx):
    """Wire history/GIF callbacks. Returns dict of helper functions."""

    state = ctx.state
    out = ctx.out
    W = ctx.W
    widgets_ref = ctx.widgets_ref
    data_mod = ctx.data_mod
    plot_mod = ctx.plot_mod
    export_mod = ctx.export_mod

    hist_start_date = ctx.hist_start_date
    hist_end_date = ctx.hist_end_date
    btn_fetch_hist = ctx.btn_fetch_hist
    hist_progress = ctx.hist_progress
    hist_status = ctx.hist_status
    hist_slider_box = ctx.hist_slider_box
    btn_export_gif = ctx.btn_export_gif
    fname_img = ctx.fname_img
    # Time-lapse export controls (new in v6+). Optional on ctx for backward
    # compat: if absent, fall back to the legacy fname_img-based name and
    # GIF format with the constant frame duration.
    fname_timelapse = getattr(ctx, 'fname_timelapse', None)
    dd_timelapse_format = getattr(ctx, 'dd_timelapse_format', None)
    slider_timelapse_fps = getattr(ctx, 'slider_timelapse_fps', None)
    lbl_timelapse_status = getattr(ctx, 'lbl_timelapse_status', None)

    def _refresh():
        ctx.refresh()

    _user_save_path = ctx.user_save_path_in_dir

    def _hist_known_cals_html(date_str):
        """Return HTML snippet showing known calibration hours for a date."""
        try:
            from datetime import date as _date
            d = _date.fromisoformat(date_str)
            cals = data_mod.list_known_calibrations(state.active_backend_name, d)
            if cals:
                hours = [esc(c.get('last_update', '')[11:16]) for c in cals]
                return f" <span style='color:#2471a3;font-size:13px;'>(cals: {', '.join(hours)} UTC)</span>"
        except Exception:
            # UI returns an empty snippet (unchanged) but log so the two
            # distinct failure modes — bad date_str (UX bug) vs. backend
            # lookup error (ops bug) — don't vanish silently.
            _log.exception(
                "list_known_calibrations lookup failed for date=%r", date_str)
        return ""

    def _on_fetch_history(b):
        start = hist_start_date.value
        end = hist_end_date.value
        if start is None or end is None:
            with out:
                print("❌ Select both start and end dates.")
            return
        if start >= end:
            with out:
                print("❌ Start must be before end.")
            return
        # Reject future dates BEFORE the IBM Runtime worker — otherwise
        # the retry loop spins for tens of seconds while the button shows
        # "⏳ Fetching..." with no signal to the user. Calibration data
        # only exists for past + today.
        from datetime import date as _date
        today = _date.today()
        for label, d in (('Start', start), ('End', end)):
            d_date = d.date() if hasattr(d, 'date') and callable(d.date) else d
            if d_date > today:
                with out:
                    print(f"❌ {label} {d_date.isoformat()} is in the future — "
                          f"IBM Quantum has no calibration past today. "
                          f"Pick {today.isoformat()} or earlier.")
                return
        btn_fetch_hist.disabled = True
        btn_fetch_hist.description = '⏳ Fetching...'
        hist_progress.layout.display = 'block'
        hist_progress.max = 1
        hist_progress.value = 0

        # Runs on the worker thread. ipywidgets value assignment is
        # advertised thread-safe; no plotly touched here.
        def _progress(idx, total, date_str):
            hist_progress.max = total
            hist_progress.value = idx + 1
            hist_status.value = (
                f"<span style='font-size:15px;color:#888;'>"
                f"Fetching {esc(date_str)}... ({idx+1}/{total})</span>")

        def _work():
            return data_mod.fetch_calibration_history(
                state, start, end, progress_callback=_progress)

        def _done(n):
            hist_progress.layout.display = 'none'
            btn_fetch_hist.disabled = False
            btn_fetch_hist.description = '📡 Fetch History'

            if n == 0:
                hist_status.value = "<span style='font-size:15px;color:#e74c3c;'>No data found.</span>"
                hist_slider_box.layout.display = 'none'
                return

            hist_status.value = f"<span style='font-size:15px;color:#27ae60;'>✅ {esc(n)} snapshots loaded</span>"
            state.range_index = 0   # list was rebuilt — old index is stale
            dates = state.cal_history_dates

            sel_slider = SelectionSlider(
                options=dates, value=dates[0], description='Date:',
                layout=W, style={'description_width': 'initial'},
                continuous_update=False)
            play_widget = Play(value=0, min=0, max=len(dates)-1, step=1,
                               interval=500, description='▶')
            speed_slider = IntSlider(value=500, min=100, max=2000, step=100,
                                      description='Speed (ms):', layout=W,
                                      style={'description_width': 'initial'})
            hist_lbl = HTML("")

            def _apply_snapshot(date_str):
                """Apply the snapshot for `date_str` and refresh. Shared by
                the slider observer and the initial-load path — assigning
                the slider to its existing value does NOT fire observers,
                so the first day must be applied explicitly."""
                if not data_mod.apply_history_snapshot(state, date_str):
                    return
                if state.delta_enabled and state.ref_qubit_props is not None:
                    state.delta_qubit_props, state.delta_gate_props = export_mod.compute_error_delta(
                        state.ref_qubit_props, state.ref_gate_props,
                        state.qubit_props_data, state.gate_props_data)
                    state.delta_date_a = f"ref {state.ref_timestamp or '?'}"
                    state.delta_date_b = date_str
                    state.delta_mode = True
                elif not state.delta_enabled:
                    state.delta_mode = False
                _refresh()
                cals_html = _hist_known_cals_html(date_str)
                hist_lbl.value = (
                    f"<div style='font-size:15px;background:#f0f4f7;padding:4px;"
                    f"border-radius:4px;'>📅 <b>{esc(date_str)}</b>{cals_html}</div>")

            def _on_slider_change(change):
                _apply_snapshot(change['new'])

            sel_slider.observe(_on_slider_change, names='value')

            def _on_play_change(change):
                idx = change['new']
                if 0 <= idx < len(dates):
                    sel_slider.value = dates[idx]
            play_widget.observe(_on_play_change, names='value')

            speed_slider.observe(
                lambda c: setattr(play_widget, 'interval', c['new']), names='value')

            hist_slider_box.children = [
                HBox([play_widget, sel_slider]), speed_slider, hist_lbl]
            hist_slider_box.layout.display = 'block'

            # Enable the Time-lapse button now that history is loaded
            # (mounted always-visible in the Export tab but starts
            # disabled — see btn_export_gif declaration for rationale).
            btn_export_gif.disabled = False
            btn_export_gif.tooltip = ''

            # Apply the first snapshot explicitly — sel_slider.value =
            # dates[0] (existing value) doesn't fire ipywidgets observers,
            # so without this the pre-fetch calibration would stay shown.
            _apply_snapshot(dates[0])

        def _err(exc):
            hist_progress.layout.display = 'none'
            btn_fetch_hist.disabled = False
            btn_fetch_hist.description = '📡 Fetch History'
            hist_status.value = (
                f"<span style='font-size:15px;color:#e74c3c;'>"
                f"❌ Fetch failed: {esc(str(exc))}</span>")

        # Daemon thread so the kernel stays responsive (30-day history =
        # dozens of seconds). default_spawn_bg marshals on_done back onto
        # the main event loop via tornado/asyncio.
        from ..async_util import default_spawn_bg
        default_spawn_bg(_work, on_done=_done, on_error=_err)

    btn_fetch_hist.on_click(_on_fetch_history)

    _gif_overwrite_pending = [False]

    def _on_export_gif(*_args, **_kwargs):
        # Two callers — button click (passes the widget) and ui.time_panel
        # (passes state, dates). Both have already populated
        # state.cal_history, so ignore the args and use the closure's
        # `state` as the source of truth.
        if not state.cal_history_dates:
            lbl_timelapse_status.value = status_html(
                'error', "No history data. Fetch calibration history first.")
            return

        # ── Resolve user choices: format, fps, name ───────────────
        # If new widgets aren't wired, fall back to fname_img + GIF + 2 fps.
        fmt = (dd_timelapse_format.value
               if dd_timelapse_format is not None else 'gif')
        fps = (slider_timelapse_fps.value
               if slider_timelapse_fps is not None else 2)
        if fname_timelapse is not None and fname_timelapse.value.strip():
            base_name = fname_timelapse.value.strip()
        else:
            base_name = f"{fname_img.value}_history"
        ext = '.gif' if fmt == 'gif' else '.mp4'

        btn_export_gif.disabled = True
        btn_export_gif.description = '⏳ Rendering...'
        try:
            out_path = _user_save_path(f"{base_name}{ext}", ext)
        except ValueError as e:
            lbl_timelapse_status.value = status_html('error', str(e))
            btn_export_gif.disabled = False
            btn_export_gif.description = '🎬 Export Time-Lapse'
            return

        if os.path.exists(out_path) and not _gif_overwrite_pending[0]:
            _gif_overwrite_pending[0] = True
            btn_export_gif.disabled = False
            btn_export_gif.description = '🎬 Export Time-Lapse'
            lbl_timelapse_status.value = status_html(
                'warn',
                f"'{os.path.basename(out_path)}' already exists. "
                f"Click Export Time-Lapse again to overwrite.",
                bold=os.path.basename(out_path))
            return
        _gif_overwrite_pending[0] = False

        try:
            import imageio
        except ImportError:
            lbl_timelapse_status.value = status_html(
                'error', "'imageio' not installed. Run: pip install imageio")
            btn_export_gif.disabled = False
            btn_export_gif.description = '🎬 Export Time-Lapse'
            return

        # Probe imageio's ffmpeg plugin early so the user sees the install
        # hint before rendering frames is wasted.
        if fmt == 'mp4':
            try:
                import imageio_ffmpeg  # noqa: F401
            except ImportError:
                lbl_timelapse_status.value = status_html(
                    'error',
                    "MP4 export requires 'imageio-ffmpeg'. Run: pip install imageio-ffmpeg")
                btn_export_gif.disabled = False
                btn_export_gif.description = '🎬 Export Time-Lapse'
                return

        # Snapshot pre-export state outside the try so `finally` can always
        # restore it.
        _saved = {
            'qubit_props_data': state.qubit_props_data,
            'gate_props_data': state.gate_props_data,
            'calibration_data_label': state.calibration_data_label,
            'delta_mode': state.delta_mode,
            'delta_qubit_props': state.delta_qubit_props,
            'delta_gate_props': state.delta_gate_props,
            'delta_date_a': state.delta_date_a,
            'delta_date_b': state.delta_date_b,
        }

        try:
            import io
            frames = []
            opts = plot_mod.collect_render_opts(widgets_ref)

            # Determine if delta is active and get reference data
            _gif_delta = (state.delta_enabled and
                          state.ref_qubit_props is not None)
            _gif_ref_qp = state.ref_qubit_props
            _gif_ref_gp = state.ref_gate_props
            _gif_ref_label = state.ref_timestamp or ''

            n_total = len(state.cal_history_dates)
            for frame_idx, date_str in enumerate(state.cal_history_dates):
                # Live progress in the button label — each frame round-trips
                # through Plotly+kaleido, so a 30-day history is 30+ s of
                # wall clock and the user needs to see we're not stuck.
                btn_export_gif.description = (
                    f'⏳ Frame {frame_idx + 1}/{n_total}'
                )
                data_mod.apply_history_snapshot(state, date_str)
                if _gif_delta:
                    state.delta_qubit_props, state.delta_gate_props = export_mod.compute_error_delta(
                        _gif_ref_qp, _gif_ref_gp,
                        state.qubit_props_data, state.gate_props_data)
                    state.delta_date_a = f"ref {_gif_ref_label}"
                    state.delta_date_b = date_str
                    state.delta_mode = True
                else:
                    state.delta_mode = False

                ann = f"Δ {date_str} vs {_gif_ref_label}" if state.delta_mode else date_str
                _qp = state.qubit_props_data
                _gp = state.gate_props_data
                _re = _qp.get('readout_error', np.array([]))
                _cz = _gp.get('cz_error', np.array([]))
                _vre = _re[~np.isnan(_re)] if len(_re) > 0 else np.array([])
                _vcz = _cz[~np.isnan(_cz)] if len(_cz) > 0 else np.array([])
                stats_parts = []
                if len(_vre) > 0:
                    stats_parts.append(f"RO med={np.median(_vre):.3E}")
                if len(_vcz) > 0:
                    stats_parts.append(f"CZ med={np.median(_vcz):.3E}")
                if stats_parts:
                    ann += f"  ({' · '.join(stats_parts)})"
                frame_fig = plot_mod.build_static_figure(
                    state, opts, margin=20, annotation=ann)
                img_bytes = frame_fig.to_image(
                    format="png", scale=2,
                    width=opts['fig_width'], height=opts['fig_height'])
                frames.append(imageio.imread(io.BytesIO(img_bytes)))

            # ── Encode ────────────────────────────────────────────
            # mimsave(..., fps=N) is canonical for both formats since
            # imageio 2.x. For MP4 we pass through to ffmpeg with
            # yuv420p — required for playback in Slack, browsers, and
            # most paper viewers.
            btn_export_gif.description = '⏳ Encoding...'
            if fmt == 'mp4':
                # libx264 requires even width/height — pad odd PNG frames
                # by 1 pixel so the encoder accepts them.
                first_h, first_w = frames[0].shape[:2]
                if first_h % 2 or first_w % 2:
                    import numpy as _np
                    pad_h = first_h % 2
                    pad_w = first_w % 2
                    frames = [
                        _np.pad(f, ((0, pad_h), (0, pad_w), (0, 0)),
                                mode='edge')
                        for f in frames
                    ]
                imageio.mimsave(
                    out_path, frames, fps=fps,
                    codec='libx264', macro_block_size=1,
                    output_params=['-pix_fmt', 'yuv420p'],
                )
            else:
                # GIF: loop=0 = infinite loop (default); fps kept explicit
                # for parity with the MP4 branch.
                imageio.mimsave(out_path, frames, fps=fps, loop=0)

            lbl_timelapse_status.value = status_html(
                'ok',
                f"{fmt.upper()} saved: {out_path} ({len(frames)} frames @ {fps} fps)",
                bold=out_path)

        except Exception as e:
            lbl_timelapse_status.value = status_html(
                'error', f"{fmt.upper()} export failed: {e}")
        finally:
            btn_export_gif.disabled = False
            btn_export_gif.description = '🎬 Export Time-Lapse'
            # ── Restore pre-export state ──────────────────────────
            state.qubit_props_data = _saved['qubit_props_data']
            state.gate_props_data = _saved['gate_props_data']
            state.calibration_data_label = _saved['calibration_data_label']
            state.delta_mode = _saved['delta_mode']
            state.delta_qubit_props = _saved['delta_qubit_props']
            state.delta_gate_props = _saved['delta_gate_props']
            state.delta_date_a = _saved['delta_date_a']
            state.delta_date_b = _saved['delta_date_b']
            _refresh()

    btn_export_gif.on_click(_on_export_gif)
    if fname_timelapse is not None:
        fname_timelapse.observe(
            lambda c: _gif_overwrite_pending.__setitem__(0, False), 'value')
    if dd_timelapse_format is not None:
        dd_timelapse_format.observe(
            lambda c: _gif_overwrite_pending.__setitem__(0, False), 'value')

    return {
        'hist_known_cals_html': _hist_known_cals_html,
        # Exposed for build_time_panel(on_gif_export=...): ui.time_panel's
        # Multi-day GIF button calls this directly instead of clicking the
        # legacy hidden button.
        'on_export_gif': _on_export_gif,
    }
