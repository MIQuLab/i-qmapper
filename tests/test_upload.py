"""Smoke tests for _upload.read_upload()."""
import pytest
from iqmapper._upload import read_upload


def test_empty_payload_rejected():
    with pytest.raises(ValueError, match="Empty"):
        read_upload(None)
    with pytest.raises(ValueError, match="Empty"):
        read_upload(())
    with pytest.raises(ValueError, match="Empty"):
        read_upload({})


def test_ipywidgets_v8_tuple_shape():
    payload = ({"name": "test.json", "content": b"hello"},)
    content, name = read_upload(payload)
    assert content == b"hello"
    assert name == "test.json"


def test_ipywidgets_v7_dict_shape():
    payload = {"test.json": {"content": b"hello",
                             "metadata": {"name": "test.json"}}}
    content, name = read_upload(payload)
    assert content == b"hello"
    assert name == "test.json"


def test_missing_content_field():
    with pytest.raises(ValueError, match="missing 'content'"):
        read_upload(({"name": "test.json"},))


def test_unknown_shape():
    with pytest.raises(ValueError, match="Unknown FileUpload"):
        read_upload("just a string")
