"""Regression tests for the audit §3 fixes: connectivity dedup, 8-char hex, sci format."""
from types import SimpleNamespace

import numpy as np

from qpu_mapper.export.analysis import get_connectivity_scores
from qpu_mapper.constants import hex_to_rgba, fmt_e


def test_connectivity_dedup_bidirectional_edges():
    # (0,1) listed in BOTH directions must count once, not twice.
    st = SimpleNamespace(
        edges_list=[(0, 1), (1, 0), (1, 2)],
        gate_props_data={'cz_error': np.array([0.01, 0.01, 0.02])},
        qubit_ids=['0', '1', '2'],
    )
    by_q = {s['qubit']: s for s in get_connectivity_scores(st)}
    assert by_q['0']['n_connections'] == 1   # neighbor {1}
    assert by_q['1']['n_connections'] == 2   # neighbors {0, 2}, not 3
    assert by_q['2']['n_connections'] == 1


def test_hex_to_rgba_8char_alpha():
    out = hex_to_rgba('#FF000080', 1.0)          # 0x80 = 128
    assert out.startswith('rgba(255,0,0,')
    alpha = float(out[len('rgba(255,0,0,'):-1])
    assert abs(alpha - 128 / 255) < 1e-9
    # 6-char and 3-char behaviour is unchanged
    assert hex_to_rgba('#00FF00', 0.5) == 'rgba(0,255,0,0.5)'
    assert hex_to_rgba('#F00', 1.0) == 'rgba(255,0,0,1.0)'


def test_fmt_e_minimal_exponent():
    assert fmt_e(1.23e-5) == '1.230E-5'
    assert fmt_e(5.0) == '5.000E+0'
    assert fmt_e(1.0e-15) == '1.000E-15'   # 2-digit exponent preserved
