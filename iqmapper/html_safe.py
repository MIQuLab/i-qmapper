"""
html_safe.py — Centralized HTML escaping for user-influenced strings.

Single source of truth for escaping untrusted content before inserting it
into ipywidgets.HTML value fields. Use this EVERY TIME you interpolate a
value that could come from:
  - a loaded session JSON (backend_name, chip, timestamp, labels, …)
  - a calibration API response (processor type, error strings, …)
  - user typed input (qubit IDs in inspector, circuit names, file paths)
  - exception messages

Why this matters:
  ipywidgets.HTML renders its value as raw HTML — no sandboxing. In a
  JupyterHub shared environment, arbitrary DOM injection via a loaded
  session file gives access to adjacent notebooks, cookies, and the
  Jupyter origin. This helper closes that vector.

Usage:
    from .html_safe import esc

    label.value = f"<b>Backend:</b> {esc(state.active_backend_name)}"
    error_banner.value = f"<span>Error: {esc(str(e))}</span>"
"""
import html as _html


def esc(x: object) -> str:
    """Escape any value into an HTML-safe string.

    Handles:
      - None → empty string (instead of literal 'None')
      - non-string types → str(x) first, then escape
      - escapes both single and double quotes (html.escape(..., quote=True)),
        so values are safe inside '…' / "…" attributes, not just between tags
    """
    if x is None:
        return ''
    return _html.escape(str(x), quote=True)
