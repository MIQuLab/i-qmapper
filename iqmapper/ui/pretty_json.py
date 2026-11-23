"""
ui.pretty_json.py — JSON → syntax-highlighted HTML renderer.

A pure helper extracted from ui.py so the live JSON preview in the export
panel is implemented in one place. Module-level cache (size 1) means
repeated calls with the same data short-circuit through equality of the
serialised JSON string — this matters because _get_export() in ui.py
fires on every slider drag / color-picker change even when the export
data is unchanged, and the previous in-place implementation re-tokenised
~10 KB of JSON on every event.

The output is intended for ipywidgets.HTML.value; it is fully
self-contained (`<div>` wrapper with inline style) and HTML-escapes every
token's content.
"""
import html as _html
import json


# ─────────────────────────────────────────────────────────────────────
# Tunables
# ─────────────────────────────────────────────────────────────────────

# Recursion depth cap for the array-collapse walker. The realistic JSON
# shape produced by export.get_export_data is ≤ 3 levels deep
# ({circuit: {key: [list of primitives or 2-element pairs]}}), so this
# is generous. The cap protects against pathologically nested external
# input ([[[[...]]]] with 1000+ levels) that would otherwise blow the
# Python recursion limit and crash the pretty-printer — and the whole
# UI refresh — when displaying an imported JSON.
MAX_DEPTH = 32

# Token colours. Chosen to be readable on the default Jupyter light
# theme and acceptable on dark themes. If you need a dark-theme-specific
# palette, swap these for CSS custom properties and theme via a Jupyter
# classlist on the outer div.
_C_KEY = '#0550ae'   # dark blue, bold (JSON object keys)
_C_STR = '#0a7f4b'   # dark green     (string values)
_C_NUM = '#b45309'   # dark orange    (numbers)
_C_KW  = '#8b5cf6'   # purple, bold   (true / false / null)
_C_PUN = '#64748b'   # gray           (brackets, commas, colons)


# ─────────────────────────────────────────────────────────────────────
# Cache — size 1 by design
# ─────────────────────────────────────────────────────────────────────
# Holds the last (json_string, html) pair produced by pretty_json_html.
# Hits short-circuit straight after json.dumps so we skip the per-char
# tokeniser walk (the dominant cost). Lives at module scope so the
# cache survives across UI refreshes.
_cache = {'json': None, 'html': None}


def _collapse_simple_arrays(text, depth=0):
    """Collapse arrays containing only primitives onto a single line.

    json.dumps(indent=2) puts every element of every array on its own
    line, which is unreadable for 32-qubit layouts. This walker scans
    the serialised text and rewrites primitive-only arrays as
    `[a, b, c]` while leaving arrays that contain nested objects /
    arrays in their multi-line form (so structure stays visible).

    Recursive on nested arrays; bounded by MAX_DEPTH (see above).
    Implemented as a manual character walker because regex can't
    handle nested bracket structure reliably.
    """
    if depth >= MAX_DEPTH:
        # Beyond the depth cap, return the body unchanged. The outer
        # JSON still pretty-prints; only deeply-nested fragments keep
        # their original (already-valid) indentation.
        return text
    result = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == '[':
            # Scan forward to the matching ']', tracking nesting.
            # If no nested '{' or '[' is found in the body, collapse.
            depth_b = 1
            j = i + 1
            has_nested = False
            in_string = False
            while j < n and depth_b > 0:
                ch = text[j]
                if ch == '"' and (j == 0 or text[j - 1] != '\\'):
                    in_string = not in_string
                elif not in_string:
                    if ch == '[':
                        depth_b += 1
                        has_nested = True
                    elif ch == ']':
                        depth_b -= 1
                    elif ch == '{':
                        has_nested = True
                j += 1
            if depth_b != 0:
                # Malformed — bail out, copy rest verbatim.
                result.append(text[i:])
                break
            body = text[i + 1:j - 1]
            if not has_nested:
                # Primitive-only array: collapse whitespace and newlines
                # down to single spaces after commas.
                collapsed = '[' + ', '.join(
                    tok.strip()
                    for tok in body.split(',')
                    if tok.strip()
                ) + ']'
                result.append(collapsed)
            else:
                # Nested: recurse so inner primitive arrays collapse
                # while outer multi-line form is preserved.
                result.append('[' + _collapse_simple_arrays(body, depth + 1) + ']')
            i = j
        else:
            result.append(text[i])
            i += 1
    return ''.join(result)


def _fallback_html(data):
    """Render non-JSON-serialisable data as a warning panel."""
    return (
        f"<pre style='font-family:monospace;font-size:14px;"
        f"background:#fff3cd;padding:8px;border-radius:4px;"
        f"border:1px solid #ffc107;color:#856404;'>"
        f"⚠️ Non-serializable data:<br>{_html.escape(str(data))}</pre>"
    )


def pretty_json_html(data):
    """Render a dict / list as pretty-printed, syntax-highlighted HTML.

    Tokenises the json.dumps output and wraps each token in a `<span>`
    with an appropriate colour: keys (dark blue, bold), string values
    (green), numbers (orange), keywords true/false/null (purple, bold),
    brackets and commas (gray). Every token's text content is HTML-escaped.

    Cached: if `data` serialises to the same JSON string as the previous
    call, the previously-built HTML is returned without re-running the
    tokeniser. Serialisation itself is still O(N) but is much cheaper
    than the per-character HTML escape walk it gates.

    A lightweight ~120-line reimplementation instead of pulling in
    Pygments — the JSON grammar is small enough that a hand-written
    state machine is fine. Returns a fully self-contained HTML string
    including the wrapping `<div>`.
    """
    try:
        raw = json.dumps(data, indent=2, ensure_ascii=False)
    except (TypeError, ValueError):
        # Fallback: show as plain str if not JSON-serialisable. Don't
        # cache this branch — non-serialisable input is rare and the
        # fallback HTML is cheap to rebuild.
        return _fallback_html(data)

    # Cache hit: same JSON as the previous call, return the rendered
    # HTML without re-tokenising. Most refreshes during slider drag /
    # color-picker change come through here unchanged.
    if raw == _cache['json'] and _cache['html'] is not None:
        return _cache['html']

    cache_key = raw                       # pre-collapse — matches the lookup above
    raw = _collapse_simple_arrays(raw)

    out = []
    i = 0
    n = len(raw)

    while i < n:
        c = raw[i]
        if c == '"':
            # Scan the string respecting backslash escapes.
            j = i + 1
            while j < n:
                if raw[j] == '\\' and j + 1 < n:
                    j += 2
                    continue
                if raw[j] == '"':
                    break
                j += 1
            full = raw[i:j + 1]
            # Key if the next non-whitespace char is ':'.
            k = j + 1
            while k < n and raw[k] in ' \t':
                k += 1
            is_key = (k < n and raw[k] == ':')
            color = _C_KEY if is_key else _C_STR
            weight = 'font-weight:600;' if is_key else ''
            out.append(
                f'<span style="color:{color};{weight}">'
                f'{_html.escape(full)}</span>'
            )
            i = j + 1
        elif c.isdigit() or (c == '-' and i + 1 < n and raw[i + 1].isdigit()):
            j = i
            # Accept one leading minus, then digits / decimal point /
            # exponent. Deliberately permissive — json.dumps output is
            # always well-formed so we don't need strict validation.
            if raw[j] == '-':
                j += 1
            while j < n and raw[j] in '0123456789.eE+-':
                j += 1
            out.append(f'<span style="color:{_C_NUM}">{raw[i:j]}</span>')
            i = j
        elif c.isalpha():
            # Bare word — must be one of the JSON keywords.
            j = i
            while j < n and raw[j].isalpha():
                j += 1
            word = raw[i:j]
            if word in ('true', 'false', 'null'):
                out.append(
                    f'<span style="color:{_C_KW};font-weight:600">'
                    f'{word}</span>'
                )
            else:
                # Shouldn't happen for valid json.dumps output, but don't
                # crash — just print as-is.
                out.append(_html.escape(word))
            i = j
        elif c in '{}[],:':
            out.append(f'<span style="color:{_C_PUN}">{c}</span>')
            i += 1
        else:
            # Whitespace, newlines — preserved verbatim so indentation
            # from json.dumps survives into the rendered HTML.
            out.append(_html.escape(c))
            i += 1

    body = ''.join(out)
    # 14px matches the rest of the sidebar body text; smaller is hard to
    # read on a 720px sidebar.
    result = (
        "<div style='font-family:ui-monospace,SFMono-Regular,Menlo,"
        "Consolas,monospace;font-size:14px;line-height:1.5;"
        "background:#f6f8fa;padding:10px 12px;border-radius:6px;"
        "border:1px solid #d0d7de;max-height:260px;overflow:auto;"
        "white-space:pre;'>"
        f"{body}"
        "</div>"
    )
    # Memoise for the next call.
    _cache['json'] = cache_key
    _cache['html'] = result
    return result
