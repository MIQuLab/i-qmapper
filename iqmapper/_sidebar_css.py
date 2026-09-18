"""
_sidebar_css.py — Scoped CSS for the I-QMapper sidebar.

Extracted from ui.py for git-diff readability: was a 200+ line raw string
embedded mid-builder, which obscured the surrounding Python logic and
forced reviewers to scroll through stylesheet rules to find related
widget wiring.

The CSS targets EVERY widget under `.iqmapper-sidebar` (added to the
container via `add_class('iqmapper-sidebar')`). Selectors are
deliberately broad (`[class*="Accordion"]`, `[class*="Collapse-header"]`,
etc.) so the rules survive the ipywidgets class-name rename trail
`p-Accordion → lm-Accordion → jupyter-widget-Accordion` and the
LabelView/LabelStyle/RadioButtonsView variants.

Import:
    from ._sidebar_css import SIDEBAR_CSS
    sidebar_css_widget = HTML(SIDEBAR_CSS)
"""

# Single source of truth for the sidebar stylesheet. If you need a
# dark-theme variant, fork this constant rather than monkey-patching the
# HTML widget after construction.
SIDEBAR_CSS = """
<style>
/* Base font for all standard widget chrome (label, dropdown, text
   input, checkbox label, radio label, button, datepicker). */
.iqmapper-sidebar .widget-label,
.iqmapper-sidebar .widget-readout,
.iqmapper-sidebar .widget-dropdown > select,
.iqmapper-sidebar .widget-text > input,
.iqmapper-sidebar .widget-textarea > textarea,
.iqmapper-sidebar .widget-checkbox label,
.iqmapper-sidebar .widget-radio-box label,
.iqmapper-sidebar button.jupyter-button,
.iqmapper-sidebar .widget-datepicker input {
    font-size: 20px !important;
}
.iqmapper-sidebar .widget-inline-hbox .widget-label {
    font-size: 20px !important;
    min-width: 130px !important;
}

/* ───────────────────────────────────────────────────────────────
   CATEGORY 1 — Inline widget LABELS (the "Opacity:", "Qubit Prop:",
   "Track qubits:", etc. text on the left of every widget). The
   canonical class .widget-label sometimes does not apply on
   ipywidgets 8.x because the label moved to a wrapper variant.
   We catch every class name containing "label" or "Label" inside
   the sidebar — covers .widget-label, .jupyter-widget-Label, and
   the LabelView/LabelStyle variants.
   ─────────────────────────────────────────────────────────────── */
.iqmapper-sidebar [class*="widget-label"],
.iqmapper-sidebar [class*="LabelView"],
.iqmapper-sidebar [class*="jupyter-widget-Label"] {
    font-size: 20px !important;
}

/* ───────────────────────────────────────────────────────────────
   CATEGORY 2 — RADIO BUTTONS and CHECKBOXES. Classes used across
   ipywidgets versions: .widget-radio-box, .widget-checkbox,
   .jupyter-widget-RadioButtonsView, .jupyter-widget-CheckboxView.
   The text label of each option is wrapped in a <label> child;
   we bump both the wrapper and the descendant label/span so the
   cascade reaches the actual rendered text.
   ─────────────────────────────────────────────────────────────── */
.iqmapper-sidebar .widget-radio-box label,
.iqmapper-sidebar .widget-checkbox label,
.iqmapper-sidebar [class*="RadioButtons"] label,
.iqmapper-sidebar [class*="Checkbox"] label,
.iqmapper-sidebar [class*="radio-box"] label,
.iqmapper-sidebar [class*="checkbox"] label {
    font-size: 20px !important;
}
/* Descendant span/text inside radio/checkbox labels — some ipywidgets
   versions wrap the option text in a nested <span> whose font-size
   does not inherit through the user-agent button reset. */
.iqmapper-sidebar .widget-radio-box label *,
.iqmapper-sidebar .widget-checkbox label *,
.iqmapper-sidebar [class*="RadioButtons"] label *,
.iqmapper-sidebar [class*="Checkbox"] label * {
    font-size: 20px !important;
}

/* ───────────────────────────────────────────────────────────────
   CATEGORY 3 — SLIDER READOUT (the numeric value shown on the right
   of each IntSlider/FloatSlider, e.g. "35", "1.00", "80"). The
   canonical class is .widget-readout but on ipywidgets 8.x it is
   sometimes rendered as a child of .jupyter-widget-IntSliderView
   or .jupyter-widget-FloatSliderView with a different inner class.
   We match both flavours. */
.iqmapper-sidebar .widget-readout,
.iqmapper-sidebar [class*="readout"],
.iqmapper-sidebar [class*="Readout"] {
    font-size: 20px !important;
    font-weight: 500 !important;
}

/* H3/H4 panel titles. */
.iqmapper-sidebar h3, .iqmapper-sidebar h4 {
    font-size: 23px !important;
}

/* ───────────────────────────────────────────────────────────────
   NAVIGATION — main nav (Customize / Load / Auto-LUCJ / Export /
   Error). This is a ToggleButtons widget. We match BOTH the wrapper
   class names AND any descendant button text, so the rule survives
   across ipywidgets versions that wrap the label deeper.
   ─────────────────────────────────────────────────────────────── */
.iqmapper-sidebar [class*="ToggleButtons"] button,
.iqmapper-sidebar [class*="toggle-buttons"] button,
.iqmapper-sidebar .widget-toggle-buttons button {
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 8px 6px !important;
    min-height: 42px !important;
    min-width: 120px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    text-overflow: clip !important;
}

/* ───────────────────────────────────────────────────────────────
   ACCORDION — section panels (Hardware / Display / Colors /
   Multi-Programming). Across ipywidgets versions, the header has
   lived in classes named: p-Collapse-header, lm-Collapse-header,
   jupyter-widget-Collapse-header. Catch all of them with a single
   attribute-contains selector. Then force the font on every text
   descendant to defeat user-agent button text reset rules.
   ─────────────────────────────────────────────────────────────── */
.iqmapper-sidebar [class*="Accordion"] [class*="Collapse-header"],
.iqmapper-sidebar [class*="accordion"] [class*="collapse-header"],
.iqmapper-sidebar [class*="Collapse-header"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 10px 14px !important;
    min-height: 42px !important;
}
/* Descendants — button, span, strong, text node containers — must
   also pick up the bump because ipywidgets typically renders the
   header label as a nested <button> whose font-size is reset by
   the browser default unless we override it explicitly. */
.iqmapper-sidebar [class*="Collapse-header"] *,
.iqmapper-sidebar [class*="collapse-header"] *,
.iqmapper-sidebar [class*="Accordion"] button,
.iqmapper-sidebar [class*="Accordion"] [role="button"] {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* ───────────────────────────────────────────────────────────────
   TAB BAR — sub-tabs inside CUSTOMIZE (C1/C2/C3) and inside ERROR
   (Calibration / Analysis).

   NOTE: the previous version of this rule used
   `[class*="tab"]` which is dangerously broad — it matches classes
   like tabBackground, tabContainer, tab-content, etc., causing
   layout glitches (phantom empty tab slots). We now match ONLY the
   specific Lumino/Phosphor class names that wrap an individual tab
   header, using full segment matches.
   ─────────────────────────────────────────────────────────────── */
.iqmapper-sidebar .p-TabBar-tab,
.iqmapper-sidebar .lm-TabBar-tab,
.iqmapper-sidebar .jupyter-widget-TabBar-tab {
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 8px 14px !important;
    min-height: 38px !important;
}
.iqmapper-sidebar .p-TabBar-tabLabel,
.iqmapper-sidebar .lm-TabBar-tabLabel,
.iqmapper-sidebar .jupyter-widget-TabBar-tabLabel {
    font-size: 20px !important;
    font-weight: 600 !important;
}

/* Inline HTML "subtitle" widgets ("🎨 PNG export", "🎬 Time-lapse",
   etc.) emit divs with hardcoded font-size in the inline style.
   Override them so the typographic hierarchy stays consistent.

   We split the override into two flavours:
     - bold subtitle headers (font-weight in the inline style) → 17px
     - plain inline text (no bold) → 16px
   This preserves the visual distinction between section headers and
   descriptive subtext that the source code intends, even after the
   font bump. */
.iqmapper-sidebar .widget-html-content,
.iqmapper-sidebar .jupyter-widgets.widget-html .widget-html-content {
    font-size: 19px !important;
}
/* Plain inline (no bold) — 16px */
.iqmapper-sidebar .widget-html-content > div[style*="font-size:13px"]:not([style*="font-weight:bold"]),
.iqmapper-sidebar .widget-html-content > div[style*="font-size:14px"]:not([style*="font-weight:bold"]),
.iqmapper-sidebar .widget-html-content > div[style*="font-size:15px"]:not([style*="font-weight:bold"]) {
    font-size: 19px !important;
}
/* Bold subtitle headers (often emoji + section label) — 17px */
.iqmapper-sidebar .widget-html-content > div[style*="font-size:13px"][style*="font-weight:bold"],
.iqmapper-sidebar .widget-html-content > div[style*="font-size:14px"][style*="font-weight:bold"],
.iqmapper-sidebar .widget-html-content > div[style*="font-size:15px"][style*="font-weight:bold"] {
    font-size: 20px !important;
    font-weight: 700 !important;
}

/* Containment: prevent slider tracks, dropdowns or long-labelled
   buttons from overflowing the sidebar. */
.iqmapper-sidebar .widget-box,
.iqmapper-sidebar .widget-vbox,
.iqmapper-sidebar .widget-hbox {
    max-width: 100% !important;
    box-sizing: border-box !important;
}
.iqmapper-sidebar .widget-toggle-buttons {
    flex-wrap: wrap !important;
}
.iqmapper-sidebar {
    overflow-x: hidden !important;
}

/* BoundedIntText / BoundedFloatText — keep input dimensions
   consistent with the bumped slider/dropdown chrome. */
.iqmapper-sidebar .widget-int-text input,
.iqmapper-sidebar .widget-float-text input {
    font-size: 20px !important;
    height: 32px !important;
    padding: 2px 8px !important;
}
</style>
"""
