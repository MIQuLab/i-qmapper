"""
ui.color_picker.py — `PowerPointColorPicker` widget composition.

A reusable colour-swatch widget pulled out of ui.py so it can be imported
by other UI modules without going through the full ui.build_mapper_ui
import path. The class itself is unchanged — only the import location moves.

Usage:
    from .color_picker import PowerPointColorPicker
    pp = PowerPointColorPicker(state, '#636EFA', 'Background Fill:',
                               on_change_callback=lambda c: refresh())
    display(pp.widget)
    color = pp.value
"""
import logging
import re
from typing import Callable, Optional

from ipywidgets import (
    HBox, VBox, Label, ColorPicker, Button, HTML, Layout,
)

from ..constants import STANDARD_COLORS, save_custom_colors

_log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Layout constants (kept short — repeated literals harm diff readability
# more than they harm runtime).
# ─────────────────────────────────────────────────────────────────────
_ROW_LAYOUT       = Layout(flex_flow='row wrap', width='100%', margin='2px 0')
_TOGGLE_BTN_LAYOUT = Layout(width='100%', margin='2px 0')
_OUTER_LAYOUT     = Layout(
    border='1px solid #ddd', padding='6px', margin='4px 0',
)
_PICKER_LAYOUT    = Layout(width='60px')
_ADD_BTN_LAYOUT   = Layout(width='50px')
_CSTM_LBL_LAYOUT  = Layout(width='40px')

# NOTE: the advanced-box Layout (display toggled by _toggle_advanced) and the
# per-swatch Layout (border toggled on selection) are deliberately NOT module
# constants. ipywidgets does not clone a Layout passed to a widget — every
# widget would reference the SAME object, so mutating one would mutate them
# all (advanced box would toggle on every picker instance at once; a swatch
# border write would re-render every swatch — the "jumping" boxes). Both are
# created per-instance / per-swatch below.

# Hex colour validation. Accepts `#RGB`, `#RRGGBB`, and `#RRGGBBAA`
# — the three forms ColorPicker can produce on different ipywidgets builds.
# Bare names like 'red' or rgb(...) strings are rejected here because they
# never round-trip cleanly through save_custom_colors → load_custom_colors
# (the persisted JSON keeps the raw string but consumers expect hex).
_HEX_RE = re.compile(r'^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$')


def _is_valid_hex(value: object) -> bool:
    """Return True iff `value` is a hex colour string we can persist."""
    return isinstance(value, str) and bool(_HEX_RE.match(value))


def _preview_swatch_html(hex_color: str) -> str:
    """HTML snippet for the small colour preview block in the header row."""
    return (
        f"<div style='width:20px;height:20px;background:{hex_color};"
        f"border:1px solid #000;border-radius:3px;'></div>"
    )


class PowerPointColorPicker:
    """A standard-palette + custom-palette colour picker, modelled on the
    PowerPoint colour grid: a row of standard swatches, a row of
    user-saved custom swatches, plus an "advanced" pop-out with a
    full-spectrum ColorPicker for picking arbitrary RGB and saving them
    into the user's `state.custom_colors`.

    The widget is fully self-contained — `pp.widget` is the assembled
    VBox; `pp.value` returns the currently-selected hex string. Custom
    colours are persisted via `save_custom_colors` (defined in constants).
    """

    def __init__(
        self,
        state,
        default_color: str = '#636EFA',
        label: str = 'Color:',
        on_change_callback: Optional[Callable[[str], None]] = None,
    ):
        self._state = state
        self.current_color = default_color
        self.on_change_callback = on_change_callback
        self.label = label
        self.swatches_box = HBox(layout=_ROW_LAYOUT)
        self.custom_swatches_box = HBox(layout=_ROW_LAYOUT)
        self.advanced_picker = ColorPicker(
            value=default_color, description='', layout=_PICKER_LAYOUT)
        self.btn_add_custom = Button(
            description='Add', button_style='success', layout=_ADD_BTN_LAYOUT)
        # Per-instance Layout (see module note): _toggle_advanced mutates
        # .display, so a shared Layout would toggle every picker's advanced
        # box together.
        self.advanced_box = HBox(
            [Label('Cstm:', layout=_CSTM_LBL_LAYOUT),
             self.advanced_picker, self.btn_add_custom],
            layout=Layout(margin='2px 0', display='none'),
        )
        self.btn_toggle_advanced = Button(
            description='+ More', button_style='info',
            layout=_TOGGLE_BTN_LAYOUT, icon='plus',
        )

        # Explicit reference to the header preview swatch, so it does not
        # depend on the VBox/HBox child order.
        self._swatch_preview = HTML(_preview_swatch_html(self.current_color))

        self.widget = VBox([
            HBox([
                Label(self.label,
                      style={'font_weight': 'bold', 'font_size': '11px'}),
                self._swatch_preview,
            ], layout=Layout(justify_content='space-between')),
            HTML("<div style='font-size:14px; color:#666;'>Standard:</div>"),
            self.swatches_box,
            HTML("<div style='font-size:14px; color:#666;'>Custom:</div>"),
            self.custom_swatches_box,
            self.btn_toggle_advanced,
            self.advanced_box,
        ], layout=_OUTER_LAYOUT)

        self.btn_toggle_advanced.on_click(self._toggle_advanced)
        self.btn_add_custom.on_click(self._add_custom_color)
        self.advanced_picker.observe(self._on_advanced_pick, 'value')
        self.refresh_swatches()

    def _toggle_advanced(self, b):
        if self.advanced_box.layout.display == 'none':
            self.advanced_box.layout.display = 'flex'
            self.btn_toggle_advanced.description = '- Hide'
            self.btn_toggle_advanced.icon = 'minus'
        else:
            self.advanced_box.layout.display = 'none'
            self.btn_toggle_advanced.description = '+ More'
            self.btn_toggle_advanced.icon = 'plus'

    def _add_custom_color(self, b):
        nc = self.advanced_picker.value
        # Validate before persisting. ColorPicker on some ipywidgets builds
        # can briefly hold a non-hex value during user editing; without
        # this guard we'd round-trip garbage through save_custom_colors
        # and downstream consumers would render a broken palette.
        if not _is_valid_hex(nc):
            _log.warning(
                "Ignoring custom color %r — not a valid hex string", nc)
            return
        if nc in STANDARD_COLORS or nc in self._state.custom_colors:
            # Already present — just select it.
            self.set_color(nc)
            return

        self._state.custom_colors.append(nc)
        # Persisting can fail on read-only filesystems / full disks; we
        # keep the in-memory palette update either way so the user can
        # still use the colour this session. The error is logged for
        # debugging but stays out of the UI surface (no notify channel
        # available from inside the picker).
        try:
            save_custom_colors(self._state.custom_colors)
        except Exception:
            _log.exception(
                "save_custom_colors failed; palette change kept in memory only")
        self.refresh_swatches()
        self.set_color(nc)

    def _on_advanced_pick(self, change):
        self.set_color(change['new'])

    def _create_swatch(self, hex_color):
        # Each swatch gets its OWN Layout (see module note). Passing a shared
        # module-level Layout made every Button reference the same object, so
        # `btn.layout.border = ...` wrote to ALL swatches at once and each
        # write re-rendered every swatch sharing it — the visible "jump" on
        # neighbouring boxes. A fresh per-swatch Layout makes the selection
        # border truly per-button and removes the cascade.
        selected = (hex_color == self.current_color)
        btn = Button(
            tooltip=hex_color,
            layout=Layout(
                width='24px', height='24px', margin='1px', padding='0',
                # Constant 2px border; only the colour changes on selection,
                # so the swatch never changes size — no horizontal reflow.
                border='2px solid #2980B9' if selected else '2px solid #bdc3c7',
            ),
        )
        btn.style.button_color = hex_color
        btn._swatch_hex = hex_color   # canonical key used by _sync_swatch_borders
        btn.on_click(lambda b, color=hex_color: self.set_color(color))
        return btn

    def refresh_swatches(self):
        """Full rebuild of both swatch rows. Call only when the palette
        itself changes (init, or a custom colour added) — NOT on a mere
        selection change (use _sync_swatch_borders for that, which avoids
        tearing down and recreating every Button on each pick)."""
        self.swatches_box.children = [
            self._create_swatch(c) for c in STANDARD_COLORS]
        self.custom_swatches_box.children = [
            self._create_swatch(c) for c in self._state.custom_colors]

    def _sync_swatch_borders(self):
        """Reflect the current selection by updating ONLY the border of the
        already-built swatches — no teardown/rebuild. Each swatch owns its
        Layout, so a border write re-renders just that one button; the
        no-op guard keeps writes to the (at most two) swatches that changed."""
        for btn in (*self.swatches_box.children,
                    *self.custom_swatches_box.children):
            selected = getattr(btn, '_swatch_hex', None) == self.current_color
            new_border = '2px solid #2980B9' if selected else '2px solid #bdc3c7'
            if btn.layout.border != new_border:
                btn.layout.border = new_border

    def set_color(self, hex_color):
        # Guard against the swatch-click re-entry: clicking a swatch sets
        # `advanced_picker.value`, which fires `_on_advanced_pick`, which
        # calls back into set_color with the same colour. Without this
        # short-circuit, refresh_swatches() and on_change_callback both
        # run twice in quick succession — visible as a swatch-row flicker
        # ("scorrimento") and a doubled plot redraw on every pick.
        if hex_color == self.current_color:
            return
        self.current_color = hex_color
        self.advanced_picker.value = hex_color
        self._swatch_preview.value = _preview_swatch_html(hex_color)
        # Selection change flips at most two borders — no full rebuild, so
        # neighbouring swatches do not re-render or jump.
        self._sync_swatch_borders()
        if self.on_change_callback:
            self.on_change_callback(hex_color)

    @property
    def value(self):
        return self.current_color
