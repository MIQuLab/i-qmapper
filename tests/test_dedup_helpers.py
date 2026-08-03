from types import SimpleNamespace
from datetime import datetime, timezone, timedelta

from iqmapper.data import _extract_last_update, _props_shape_matches


def test_extract_last_update_both_modes():
    dt = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone(timedelta(hours=-5)))
    props = SimpleNamespace(last_update_date=dt)
    assert _extract_last_update(props) == "2025-01-02 08:04:05"               # UTC, no label
    assert _extract_last_update(props, display=True) == "2025-01-02 03:04:05 (-0500)"
    assert _extract_last_update(SimpleNamespace(last_update_date=None)) is None


def test_props_shape_matches():
    qp = {'readout_error': [0.0] * 5}
    gp = {'cz_error': [0.0] * 4}
    assert _props_shape_matches(qp, gp, 5, 4) is True
    assert _props_shape_matches(qp, gp, 6, 4) is False   # qubit count off
    assert _props_shape_matches(qp, gp, 5, 3) is False   # edge count off
    assert _props_shape_matches({}, {}, 5, 4) is False   # empty
