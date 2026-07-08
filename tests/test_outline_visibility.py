from qpu_mapper.plot import _ensure_visible_outline


def test_line_only_outline_stays_visible():
    # outline equal to the background -> substituted with a visible colour
    c, w = _ensure_visible_outline('#ffffff', '#ffffff', 2)
    assert c != '#ffffff' and w >= 1.0
    # zero width -> bumped to a visible width, colour kept
    c, w = _ensure_visible_outline('#000000', '#ffffff', 0)
    assert w >= 1.0 and c == '#000000'
    # a clearly contrasting outline is left untouched
    c, w = _ensure_visible_outline('#000000', '#ffffff', 2)
    assert c == '#000000' and w == 2.0
    # non-hex input must not crash (falls through unchanged)
    c, w = _ensure_visible_outline('black', 'white', 2)
    assert w == 2.0
