"""
tests/test_html_safe.py — Unit tests for iqmapper.html_safe.esc().

Closes the audit gap (AUDIT_REPORT § html_safe.py): the escape helper
was previously untested even though it's the single defence against
XSS via user-influenced strings in ipywidgets.HTML widgets (loaded
session JSON, calibration label strings, exception messages, ...).
"""

from iqmapper.html_safe import esc


class TestEscBasic:
    """Core escape behaviour: every HTML-significant character is replaced."""

    def test_script_tag(self):
        # The headline case the audit report calls out explicitly.
        assert esc("<script>") == "&lt;script&gt;"

    def test_full_script_injection(self):
        assert esc("<script>alert('xss')</script>") == (
            "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;")

    def test_amp_first(self):
        # & must be escaped BEFORE < and > to avoid double-encoding.
        assert esc("Tom & Jerry") == "Tom &amp; Jerry"
        assert esc("a &amp; b") == "a &amp;amp; b"

    def test_plain_text_unchanged(self):
        assert esc("hello world") == "hello world"
        assert esc("ibm_kingston") == "ibm_kingston"
        assert esc("Heron r3") == "Heron r3"

    def test_empty_string(self):
        assert esc("") == ""


class TestEscQuotes:
    """`quote=True` is the project default — both kinds of quotes escape.
    This matters because some HTML interpolations land inside a `style='...'`
    or `tooltip="..."` attribute, where an unescaped quote breaks the
    parser and lets the attacker close the attribute early.
    """

    def test_double_quote(self):
        assert esc('say "hi"') == "say &quot;hi&quot;"

    def test_single_quote(self):
        # html.escape with quote=True maps ' to &#x27; (not &apos;)
        # so the result is what HTML5 parsers consume identically.
        assert esc("it's fine") == "it&#x27;s fine"

    def test_attribute_breakout_blocked(self):
        # If an attacker can land their string inside a `title='...'`
        # attribute, escaping ' is what prevents them from closing it
        # and adding `onclick=...`. We verify the relevant char is gone.
        out = esc("' onclick='alert(1)")
        assert "'" not in out
        assert "onclick" in out  # plain text content is preserved


class TestEscNonString:
    """Non-string inputs are coerced to str() before escaping — this is
    important because callers pass exception objects, ints, dicts, etc.
    A previous bare html.escape() call here would TypeError.
    """

    def test_none_returns_empty(self):
        # None → '' (not the literal 'None'), so optional state fields
        # that haven't been populated yet render as blank instead of
        # the misleading text "None".
        assert esc(None) == ""

    def test_int(self):
        assert esc(42) == "42"

    def test_float(self):
        # str(float) result; we don't reformat — caller decides precision.
        assert esc(3.14) == "3.14"

    def test_bool(self):
        assert esc(True) == "True"
        assert esc(False) == "False"

    def test_list_repr(self):
        # str(list) → "[1, 2, 3]" — brackets must be escaped.
        assert esc([1, 2, 3]) == "[1, 2, 3]"  # square brackets are not HTML-significant

    def test_exception_object(self):
        # Common pattern: esc(str(e)) inside an except clause. Verify
        # both forms work — both via str() coercion and the explicit one.
        exc = ValueError("<bad>")
        assert esc(exc) == "&lt;bad&gt;"
        assert esc(str(exc)) == "&lt;bad&gt;"


class TestEscReturnType:
    """The return type must always be str — never bytes, never None."""

    def test_always_returns_str(self):
        for inp in (None, 0, "", "<", 3.14, [], {}, ValueError("x")):
            assert isinstance(esc(inp), str), f"esc({inp!r}) was not a str"


class TestEscRealisticPayloads:
    """Strings shaped like the actual values that flow through the
    ipywidgets.HTML pipeline in this project."""

    def test_calibration_label(self):
        # state.calibration_data_label format (no escape-needing chars).
        label = "ibm_kingston — 2026-01-20 19:18:17"
        assert esc(label) == "ibm_kingston — 2026-01-20 19:18:17"

    def test_backend_name_with_underscore(self):
        # Underscores must not be touched — backend names depend on them.
        assert esc("ibm_torino") == "ibm_torino"

    def test_props_warning_with_html(self):
        # A malicious session file could carry a props_warning with HTML.
        adversarial = "<img src=x onerror=alert(1)>"
        out = esc(adversarial)
        assert "<" not in out
        assert ">" not in out
        assert "onerror" in out  # text content is preserved, just rendered as text

    def test_unicode_passthrough(self):
        # Non-ASCII characters are NOT HTML-significant — must pass through.
        assert esc("α/β layout") == "α/β layout"
        assert esc("📡 connected") == "📡 connected"
