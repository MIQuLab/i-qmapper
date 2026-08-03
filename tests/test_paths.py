"""
tests/test_paths.py — Unit tests for iqmapper.paths.safe_path().

Closes the audit gap (AUDIT_REPORT § paths.py): tests/test_mapper.py
historically contained a buggy reimplementation of safe_path using
str.startswith — it tested the broken copy, not the real function.
This module imports and exercises the REAL safe_path from
iqmapper.paths, covering:

  • the original guards (empty path, '..', '/', '~', escape via realpath)
  • the new guards added in the audit pass: null byte rejection,
    component-level '..' detection (so legitimate names like
    'my..file.json' are accepted), and the 255-byte component limit.

Two known-not-yet-tested gaps (Windows drive-letter behaviour, symlink
containment) are covered by skipped tests rather than removed entries —
left as TODOs the CI can fix when run on Windows / when a tmpdir-based
symlink fixture is added.
"""
import os
import sys

import pytest

from iqmapper.paths import safe_path, _has_dotdot_component


# ─────────────────────────────────────────────────────────────────────
# Fixture: pin CWD to a tmp_path so the tests don't accidentally
# create / touch files in the real working directory.
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Run every test with `os.getcwd() == tmp_path`.

    Without this, `safe_path('foo.json')` would resolve under whatever
    directory `pytest` happens to be invoked from — which would
    pollute the developer's checkout and cause flaky pass/fail under
    CI vs. local runs.
    """
    monkeypatch.chdir(tmp_path)
    yield tmp_path


# ═════════════════════════════════════════════════════════════════════
# Acceptance — paths that MUST resolve cleanly under CWD.
# ═════════════════════════════════════════════════════════════════════

class TestAccept:
    def test_simple_name(self, _isolated_cwd):
        out = safe_path('my_layout.json')
        # The resolved path must live inside the CWD we just pinned.
        assert out == os.path.realpath(str(_isolated_cwd / 'my_layout.json'))

    def test_subdirectory(self, _isolated_cwd):
        out = safe_path('output/layout.json')
        assert out == os.path.realpath(str(_isolated_cwd / 'output' / 'layout.json'))

    def test_name_with_literal_dotdot_chars(self, _isolated_cwd):
        # The legitimate case the audit fix targets: '..' as substring
        # in a normal filename (e.g. produced by some date format) must
        # NOT be rejected. The previous `'..' in p` guard blocked this.
        out = safe_path('my..file.json')
        assert out.endswith('my..file.json')

    def test_name_with_dots(self, _isolated_cwd):
        # Single dots are always allowed.
        out = safe_path('file.v2.json')
        assert out.endswith('file.v2.json')

    def test_unicode_name(self, _isolated_cwd):
        # Non-ASCII filenames pass through. The 255-byte limit is
        # enforced in UTF-8 bytes, so short unicode names are fine.
        out = safe_path('α-layout.json')
        assert out.endswith('α-layout.json')


# ═════════════════════════════════════════════════════════════════════
# Rejection — the original guards from before the audit pass.
# ═════════════════════════════════════════════════════════════════════

class TestRejectOriginalGuards:
    def test_empty(self):
        with pytest.raises(ValueError, match="Empty"):
            safe_path('')

    def test_whitespace_only(self):
        # Stripped to empty → same "Empty path" reject.
        with pytest.raises(ValueError, match="Empty"):
            safe_path('   ')

    def test_absolute_posix(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('/tmp/evil.json')

    def test_tilde(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('~/.bashrc')

    def test_dotdot_posix(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('../../etc/passwd')

    def test_dotdot_anywhere_in_path(self):
        # '..' anywhere as a separate component is rejected.
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('output/../escape.json')


# ═════════════════════════════════════════════════════════════════════
# Rejection — the NEW guards added during the audit pass.
# ═════════════════════════════════════════════════════════════════════

class TestRejectNullByte:
    """Null bytes (\\x00) terminate strings at the C level and are a
    classic smuggling vector against open()-family syscalls."""

    def test_embedded_null(self):
        with pytest.raises(ValueError, match="Null byte"):
            safe_path('file\x00.json')

    def test_leading_null(self):
        with pytest.raises(ValueError, match="Null byte"):
            safe_path('\x00file.json')

    def test_trailing_null(self):
        with pytest.raises(ValueError, match="Null byte"):
            safe_path('file.json\x00')


class TestHasDotdotComponent:
    """Direct unit tests on the internal helper so a regression that
    accepts a real '..' component (or rejects a plain 'my..file') is
    caught even if the outer safe_path path shape changes."""

    def test_real_dotdot_component_true(self):
        assert _has_dotdot_component('a/../b') is True
        assert _has_dotdot_component('..') is True
        assert _has_dotdot_component('../foo') is True

    def test_windows_style_separator_true(self):
        assert _has_dotdot_component('a\\..\\b') is True

    def test_literal_dots_in_name_false(self):
        assert _has_dotdot_component('my..file') is False
        assert _has_dotdot_component('a/my..file.json') is False


class TestRejectDotdotComponent:
    """Component-level '..' detection: substring 'XX..YY' is fine
    (see TestAccept.test_name_with_literal_dotdot_chars), but a
    SEPARATE component equal to '..' must reject."""

    def test_dotdot_with_unix_separator(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('a/../b')

    def test_dotdot_with_windows_separator(self):
        # The component scan normalises \\ → / so a Windows-style
        # traversal path is rejected on any platform.
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('a\\..\\b')

    def test_dotdot_first_component(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('../foo.json')

    def test_dotdot_only(self):
        with pytest.raises(ValueError, match="Unsafe"):
            safe_path('..')


class TestRejectTooLong:
    """255-byte limit on a single path component — matches ext4/APFS/NTFS
    NAME_MAX. Counted in UTF-8 BYTES, not characters, because that's
    what the filesystem enforces."""

    def test_ascii_too_long(self):
        # 256 single-byte chars > 255 byte limit.
        with pytest.raises(ValueError, match="too long"):
            safe_path('a' * 256 + '.json')

    def test_ascii_exactly_at_limit_accepted(self):
        # Exactly 255 bytes — must pass.
        name = 'a' * 255
        out = safe_path(name)
        assert out.endswith(name)

    def test_unicode_long_in_bytes(self):
        # Each 'α' is 2 UTF-8 bytes — 128 chars = 256 bytes → reject.
        name = 'α' * 128 + '.json'
        # Encode check: that name's longest component is the basename;
        # ensure it exceeds 255 in bytes.
        assert len(name.encode('utf-8')) > 255
        with pytest.raises(ValueError, match="too long"):
            safe_path(name)


# ═════════════════════════════════════════════════════════════════════
# Containment — paths that resolve outside CWD via realpath() get
# rejected by commonpath, not just by the up-front substring guards.
# ═════════════════════════════════════════════════════════════════════

class TestContainment:
    def test_realpath_escapes_via_symlink(self, _isolated_cwd, tmp_path):
        """A symlink inside CWD that points outside CWD must be
        rejected — realpath follows the link, commonpath catches it."""
        if sys.platform.startswith('win'):
            pytest.skip("Symlinks need admin on Windows; skipped")
        # Build a target dir as a sibling of the CWD so it's outside.
        outside = tmp_path.parent / 'outside_target'
        outside.mkdir(exist_ok=True)
        # Inside the CWD, a symlink "escape" pointing at the sibling.
        link = _isolated_cwd / 'escape'
        try:
            link.symlink_to(outside)
        except OSError:
            pytest.skip("Filesystem does not support symlinks")
        with pytest.raises(ValueError, match="escapes working directory"):
            safe_path('escape/foo.json')


# ═════════════════════════════════════════════════════════════════════
# Platform — known coverage gaps. Tracked as skipped tests instead of
# TODO comments so they show up in pytest -v and can be implemented
# incrementally.
# ═════════════════════════════════════════════════════════════════════

class TestWindowsDrives:
    """`os.path.commonpath` raises ValueError when paths are on
    different drives — we map that to a 'Path escapes' rejection.
    Exercise it on Windows where the behaviour is observable."""

    @pytest.mark.skipif(not sys.platform.startswith('win'),
                        reason="Drive-letter semantics only on Windows")
    def test_cross_drive_path_rejected(self):
        # On Windows, commonpath(['C:\\Users', 'D:\\Other']) raises.
        # safe_path turns that into a "Path escapes working directory".
        # The exact construction depends on the test CWD's drive, so
        # this test is a placeholder — fill in when CI gains a Windows job.
        with pytest.raises(ValueError):
            safe_path('D:\\\\somewhere.json')
