"""Unit tests for iqmapper.constants — deterministic layout coords,
color utilities, and backend-topology resolution.

All tests are offline and use SimpleNamespace fakes for backend objects;
no qiskit / no network."""
from types import SimpleNamespace

import pytest

from iqmapper.constants import (
    heron_coords, nighthawk_coords,
    hex_to_rgba, blend_colors, text_color_for_bg,
    resolve_coords,
)


# ═══════════════════════════════════════════════════════════════════════
# Hardcoded chip coordinate tables
# ═══════════════════════════════════════════════════════════════════════

def test_heron_coords_has_156_qubits():
    coords = heron_coords()
    assert len(coords) == 156


def test_nighthawk_coords_has_120_qubits():
    coords = nighthawk_coords()
    assert len(coords) == 120


def test_coord_tables_are_deterministic():
    """Two calls must return identical geometry — the lookup is a
    frozen chip layout, not something that can drift between calls."""
    assert heron_coords() == heron_coords()
    assert nighthawk_coords() == nighthawk_coords()


# ═══════════════════════════════════════════════════════════════════════
# Color helpers
# ═══════════════════════════════════════════════════════════════════════

class TestHexToRgba:
    def test_full_hex(self):
        assert hex_to_rgba('#ff0000', 0.5) == 'rgba(255,0,0,0.5)'

    def test_short_hex_expanded(self):
        # #abc → #aabbcc
        assert hex_to_rgba('#abc', 1.0) == 'rgba(170,187,204,1.0)'

    def test_invalid_input_falls_back_to_black(self):
        # Non-hex / bogus input must not crash — the helper's contract
        # is "returns black on invalid input".
        assert hex_to_rgba('not-a-color', 0.7) == 'rgba(0,0,0,0.7)'

    def test_deterministic(self):
        assert hex_to_rgba('#123456', 0.3) == hex_to_rgba('#123456', 0.3)


class TestBlendColors:
    def test_midpoint_mix(self):
        # Even mix of pure red and pure blue at ratio=0.5 → magenta-ish.
        out = blend_colors('#ff0000', '#0000ff', mix_ratio=0.5)
        assert out.startswith('rgb(') and out.endswith(')')

    def test_deterministic(self):
        a = blend_colors('#123456', '#abcdef', mix_ratio=0.4)
        b = blend_colors('#123456', '#abcdef', mix_ratio=0.4)
        assert a == b

    def test_returns_string(self):
        assert isinstance(blend_colors('#ffffff', '#000000'), str)


class TestTextColorForBg:
    def test_white_bg_gets_black_text(self):
        assert text_color_for_bg('rgb(255,255,255)') == 'black'

    def test_black_bg_gets_white_text(self):
        assert text_color_for_bg('rgb(0,0,0)') == 'white'

    def test_falls_back_on_garbage(self):
        # No parsable RGB triple → default 'black' (never crash).
        assert text_color_for_bg('completely bogus') == 'black'


# ═══════════════════════════════════════════════════════════════════════
# resolve_coords — dispatch on backend.processor_type / num_qubits
# ═══════════════════════════════════════════════════════════════════════

def _fake_backend(family, revision, num_qubits, name='fake'):
    """Build a SimpleNamespace that mimics the attributes resolve_coords
    reads: processor_type (dict), num_qubits, name."""
    return SimpleNamespace(
        processor_type={'family': family, 'revision': revision},
        num_qubits=num_qubits,
        name=name,
    )


def test_resolve_coords_heron_r3():
    backend = _fake_backend('Heron', '3', num_qubits=156)
    coords, label = resolve_coords(backend)
    assert coords == heron_coords()
    assert 'Heron' in label
    assert '3' in label


def test_resolve_coords_nighthawk_r1():
    backend = _fake_backend('Nighthawk', '1', num_qubits=120)
    coords, label = resolve_coords(backend)
    assert coords == nighthawk_coords()
    assert 'Nighthawk' in label


def test_resolve_coords_unsupported_family_raises():
    backend = _fake_backend('Sycamore', '1', num_qubits=54, name='fake_google')
    with pytest.raises(NotImplementedError):
        resolve_coords(backend)
