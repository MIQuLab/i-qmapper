"""Inline status banner used by the Export panel sections, matching the
session-save style so every export gives feedback the same way."""
from html import escape as _esc

_PALETTE = {'warn': ('#e67e22', '#fef9e7', '#f5cba7'),
            'ok':   ('#1e8449', '#eafaf1', '#abebc6')}
_ICON = {'ok': '✅', 'warn': '⚠️', 'error': '❌'}


def status_html(level, msg, *, bold=None):
    """Return a styled HTML banner. level: 'ok'|'warn'|'error'.
    msg is plain text (HTML-escaped); bold (optional) is a substring of msg
    rendered <b> (e.g. a filename/path)."""
    body = _esc(str(msg))
    if bold:
        body = body.replace(_esc(str(bold)), f"<b>{_esc(str(bold))}</b>", 1)
    icon = _ICON[level]
    if level == 'error':
        return f"<span style='font-size:15px;color:#e74c3c;'>{icon} {body}</span>"
    fg, bg, br = _PALETTE[level]
    return (f"<div style='font-size:15px;color:{fg};background:{bg};"
            f"padding:4px 8px;border-radius:4px;border:1px solid {br};"
            f"margin:3px 0;'>{icon} {body}</div>")
