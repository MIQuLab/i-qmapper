"""
_upload.py — Centralised parser for ipywidgets.FileUpload payloads.

ipywidgets ships two incompatible payload shapes across major versions:

  • v7: dict-of-dicts keyed by filename →
        {'foo.json': {'content': memoryview, 'metadata': {...}}}
  • v8: tuple of dicts →
        ({'name': 'foo.json', 'content': memoryview, 'last_modified': ...},)

Five call sites across the codebase (`ui.py` × 2, `ui.session_io.py` × 2,
plus a future Session/style adopter) used to reinvent the same
`bytes(new_val[0]['content'] if isinstance(new_val, tuple) else
list(new_val.values())[0]['content'])` expression — and the
implementations had already drifted: some returned only the content,
some also extracted the filename from `item.get('name')`, some from
`item.get('metadata', {}).get('name')`. This module is the single
source of truth.

Lives in its own module (not in ui.py) so `ui.session_io.py` and other
sibling modules can import it without pulling in the heavy ui.py
import-graph or risking a circular import.
"""
from typing import Tuple


def read_upload(new_value) -> Tuple[bytes, str]:
    """Normalise a FileUpload `change['new']` payload.

    Args:
        new_value: the value of `FileUpload.value` (or `change['new']`).

    Returns:
        `(content_bytes, filename)`. `filename` is `''` if the payload
        doesn't carry one (rare; some older ipywidgets builds in certain
        configurations omitted the field).

    Raises:
        ValueError on empty payload or an unrecognised shape, so callers
        get a clear "upload was empty" message instead of a generic
        IndexError / TypeError from deep inside.
    """
    if not new_value:
        raise ValueError("Empty upload payload")
    if isinstance(new_value, tuple):
        item = new_value[0]   # non-empty: the top `if not new_value` guard ran
    elif isinstance(new_value, dict):
        items = list(new_value.values())
        if not items:
            raise ValueError("Empty upload dict")
        item = items[0]
    else:
        raise ValueError(
            f"Unknown FileUpload payload type: {type(new_value).__name__}")

    if not isinstance(item, dict):
        raise ValueError(
            f"FileUpload entry is not a dict: {type(item).__name__}")
    raw = item.get('content')
    if raw is None:
        raise ValueError("Upload payload missing 'content' field")
    filename = (item.get('name')
                or item.get('metadata', {}).get('name')
                or '')
    return bytes(raw), filename
