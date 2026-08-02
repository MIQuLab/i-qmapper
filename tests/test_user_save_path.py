import os
import pytest
from iqpu_mapper.paths import resolve_user_save_path


def test_absolute_path_and_parent_creation(tmp_path):
    target = tmp_path / "sub" / "out.png"
    out = resolve_user_save_path(str(target))
    assert out == os.path.realpath(str(target))
    assert (tmp_path / "sub").is_dir()           # parent created


def test_bare_name_resolves_against_default_dir(tmp_path):
    out = resolve_user_save_path("x", default_ext=".json", default_dir=str(tmp_path))
    assert out == os.path.realpath(str(tmp_path / "x.json"))


def test_rejects_empty_and_nul():
    with pytest.raises(ValueError):
        resolve_user_save_path("")
    with pytest.raises(ValueError):
        resolve_user_save_path("a\x00b")
