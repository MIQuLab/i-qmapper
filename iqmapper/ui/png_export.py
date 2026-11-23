"""
ui.png_export.py — PNG export popup + render-and-save wiring.

Modal-style popup for picking margin / crop / scale before writing a
high-resolution PNG: Save PNG → popup; Save now → render via
plot.build_static_figure, write_image, banner via lbl_png_status (most
common failure is a missing/outdated kaleido — `_classify_write_error`
maps it to a targeted hint); Cancel → hide popup, no write.

Public API:
  setup_png_export(ctx) -> (png_popup, margin_slider, crop_checkbox)

The returned widgets are mounted by the Export tab. ctx supplies state,
the rendering options (via WidgetsRef), the filename Text, the safe-path
helper, and the inline status label.
"""
import logging
import os

from .status import status_html

from ipywidgets import (
    VBox, HBox, HTML, IntSlider, Checkbox, Button, Layout,
)

_log = logging.getLogger(__name__)

# Default resolution multiplier. 3× = "high quality" default; the slider
# lets the user bump for paper figures or drop for fast previews.
_DEFAULT_PNG_SCALE = 3
_MIN_PNG_SCALE = 1
_MAX_PNG_SCALE = 6

# Substrings used to classify write_image() errors so the user gets a
# targeted hint, not a one-size-fits-all "install kaleido". Lowercase /
# substring on purpose — kaleido phrases the same failure differently
# across releases.
_KALEIDO_HINTS = ('kaleido', 'plotly_get_chrome', 'orca')
_PERMISSION_HINTS = ('permission denied', 'read-only')
_DISK_FULL_HINTS = ('no space left', 'disk full', 'enospc')


def _append_live_colorbars(export_fig, live_fig, plot_mod):
    """Copy the live figure's visible colorbar traces onto the export figure.

    The static renderer (plot.build_static_figure) colors nodes/edges but has
    no colorbar code; the interactive figure's two placeholder traces
    (_META_CB_QUBIT / _META_CB_GATE) hold the exact colorbar configuration the
    user is looking at (ranges, log/linear/delta regime, tick labels, titles).
    Copying them guarantees the PNG shows what the screen shows. No-op when
    the bars are hidden (design mode / no data) or the placeholders are
    missing. Returns the number of bars appended.
    """
    import plotly.graph_objects as go
    appended = 0
    if live_fig is None:
        return 0
    for meta in (plot_mod._META_CB_QUBIT, plot_mod._META_CB_GATE):
        try:
            tr = plot_mod._trace_by_meta(live_fig, meta)
            if tr is None or not tr.marker.showscale:
                continue
            export_fig.add_trace(go.Scatter(
                x=[None], y=[None], mode='markers',
                hoverinfo='none', showlegend=False,
                marker=tr.marker.to_plotly_json(),
            ))
            appended += 1
        except Exception:                    # noqa: BLE001 - never break export
            _log.exception("colorbar copy failed for %s; PNG saved without it",
                           meta)
    if appended:
        # The bars render below the axes (y=-0.08); make sure the export
        # canvas leaves room for them, whatever margin the user picked.
        cur_b = getattr(export_fig.layout.margin, 'b', None) or 0
        export_fig.layout.margin.b = max(cur_b, 120)
    return appended


def _classify_write_error(exc: BaseException) -> str:
    """Map an exception from fig.write_image() to a short user-friendly hint.

    Returns the hint text WITHOUT the leading "❌" — callers wrap it.
    """
    msg = (str(exc) or '').lower()
    if isinstance(exc, ImportError) or any(h in msg for h in _KALEIDO_HINTS):
        return ("PNG export needs the kaleido renderer. Install it with: "
                "pip install -U kaleido")
    if isinstance(exc, PermissionError) or any(h in msg for h in _PERMISSION_HINTS):
        return ("Permission denied writing the PNG — check the destination "
                "directory's write permissions.")
    if isinstance(exc, OSError) and any(h in msg for h in _DISK_FULL_HINTS):
        return "No disk space left to write the PNG."
    if isinstance(exc, FileNotFoundError):
        return ("Destination directory does not exist. "
                "Create it first or pick a filename without subdirectories.")
    # Generic fallback — include the exception type as a searchable
    # keyword (the full traceback is in the logger).
    return f"PNG export failed ({type(exc).__name__}): {exc!s}"


def _ensure_png_extension(name: str) -> str:
    """Strip a trailing '.png' before re-adding it once.

    Prevents 'foo.png.png' when the user types the extension manually.
    Empty string is left alone so safe_path() can produce a clear error.
    """
    n = (name or '').strip()
    if not n:
        return n
    if n.lower().endswith('.png'):
        n = n[:-4]
    return f"{n}.png"


def setup_png_export(ctx):
    """Build the PNG export popup + wire its three buttons. Returns
    `(popup_vbox, margin_slider, crop_checkbox)`; caller mounts the popup
    in the Export tab. `btn_save_img` (from ctx) opens it.

    Required ctx fields:
      state                 — SessionState (passed to build_static_figure)
      widgets_ref           — for collect_render_opts (figure dimensions etc.)
      fname_img             — Text widget holding the user-chosen filename
      btn_save_img          — Button that opens the popup (we wire its on_click)
      W                     — shared "full-width" Layout for the slider
      plot_mod              — module exposing collect_render_opts /
                              build_static_figure
      safe_path             — path-sanitiser
      out                   — Output widget for log messages

    Optional ctx fields:
      fig                   — the live FigureWidget; its visible colorbars are
                              copied onto the PNG (the static renderer has none).
                              Missing → PNG saved without colorbars, as before.
    """
    state = ctx.state
    widgets_ref = ctx.widgets_ref
    fname_img = ctx.fname_img
    btn_save_img = ctx.btn_save_img
    W = ctx.W
    plot_mod = ctx.plot_mod
    user_save_path = ctx.user_save_path
    save_dir = ctx.save_dir
    lbl_png_status = ctx.lbl_png_status
    live_fig = getattr(ctx, 'fig', None)   # for _append_live_colorbars

    margin_slider = IntSlider(
        value=40, min=0, max=200, step=5,
        description='PNG Margin (px):', layout=W,
        style={'description_width': 'initial'},
    )
    scale_slider = IntSlider(
        value=_DEFAULT_PNG_SCALE,
        min=_MIN_PNG_SCALE, max=_MAX_PNG_SCALE, step=1,
        description='Scale (×DPI):', layout=W,
        style={'description_width': 'initial'},
    )
    crop_checkbox = Checkbox(
        value=False, description='Crop to selected qubits only', layout=W,
    )

    btn_confirm = Button(
        description='✅ Export now', button_style='success',
        layout=Layout(width='48%'),
    )
    btn_cancel = Button(
        description='✖ Cancel', button_style='',
        layout=Layout(width='48%'),
    )

    popup = VBox(
        [
            HTML("<div style='font-size:14px;font-weight:bold;color:#1a5276;"
                 "margin-bottom:6px;'>📸 PNG export settings</div>"),
            margin_slider,
            scale_slider,
            crop_checkbox,
            HBox(
                [btn_confirm, btn_cancel],
                layout=Layout(justify_content='space-between',
                              margin='6px 0 0 0'),
            ),
        ],
        layout=Layout(
            display='none', border='2px solid #2980B9',
            padding='12px', margin='6px 0',
        ),
    )

    _png_overwrite_pending = [False]

    def _on_open(b):
        # Show the popup; don't save yet — user gets to adjust margin /
        # scale / crop before committing.
        _png_overwrite_pending[0] = False
        popup.layout.display = 'block'

    def _on_cancel(b):
        popup.layout.display = 'none'

    def _on_confirm(b):
        # Resolve + validate the path first, then gate on an existing
        # file (same "click again to overwrite" pattern as session/style).
        try:
            img_path = user_save_path(_ensure_png_extension(fname_img.value), '.png',
                                      default_dir=save_dir())
        except ValueError as e:
            popup.layout.display = 'none'
            _png_overwrite_pending[0] = False
            lbl_png_status.value = status_html('error', str(e))
            return

        if os.path.exists(img_path) and not _png_overwrite_pending[0]:
            _png_overwrite_pending[0] = True
            # keep the popup open so the next Confirm overwrites
            lbl_png_status.value = status_html(
                'warn',
                f"'{os.path.basename(img_path)}' already exists. "
                f"Click Confirm again to overwrite.",
                bold=os.path.basename(img_path))
            return

        _png_overwrite_pending[0] = False
        popup.layout.display = 'none'
        try:
            margin_px = margin_slider.value
            scale = int(scale_slider.value)
            opts = plot_mod.collect_render_opts(widgets_ref)
            export_fig = plot_mod.build_static_figure(
                state, opts,
                margin=margin_px,
                crop_to_selection=crop_checkbox.value,
            )
            # The static renderer has no colorbar code; copy the visible ones
            # off the live figure so the PNG matches the screen (no-op in
            # design mode / when the live fig is unavailable).
            _append_live_colorbars(export_fig, live_fig, plot_mod)
            export_fig.write_image(
                img_path, scale=scale,
                width=opts['fig_width'], height=opts['fig_height'],
            )
            crop_note = " (cropped)" if crop_checkbox.value else ""
            lbl_png_status.value = status_html(
                'ok',
                f"Image saved: {img_path} (margin={margin_px}px, scale={scale}×{crop_note})",
                bold=img_path)
        except ValueError as e:
            lbl_png_status.value = status_html('error', str(e))
        except Exception as e:
            # Full traceback to the logger only — never to the user
            # banner (could echo sensitive paths / chained data). The
            # classified hint is the only thing the user sees.
            _log.exception("PNG export failed")
            lbl_png_status.value = status_html('error', _classify_write_error(e))

    fname_img.observe(
        lambda c: _png_overwrite_pending.__setitem__(0, False), 'value')
    btn_save_img.on_click(_on_open)
    btn_confirm.on_click(_on_confirm)
    btn_cancel.on_click(_on_cancel)

    return popup, margin_slider, crop_checkbox
