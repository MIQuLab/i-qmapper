"""Legend-config defaults + session round-trip."""
from iqmapper.state import SessionState, _default_legend_config


def test_default_legend_config_has_six_roles():
    cfg = _default_legend_config()
    assert set(cfg['roles'].keys()) == {
        'alpha', 'beta', 'other', 'qubits', 'ancilla', 'unselected'
    }
    assert cfg['enabled'] is False
    # Unselected is opt-in by default.
    assert cfg['roles']['unselected']['show'] is False
    assert cfg['roles']['alpha']['show'] is True


def test_legend_config_roundtrip_through_session():
    st = SessionState()
    # Modify all three persisted dimensions: enabled, show, label.
    st.legend_config['enabled'] = False
    st.legend_config['roles']['alpha']['label'] = 'my-alpha'
    st.legend_config['roles']['unselected']['show'] = True

    payload = st.to_session_dict()
    st2 = SessionState()
    st2.from_session_dict(payload)

    assert st2.legend_config['enabled'] is False
    assert st2.legend_config['roles']['alpha']['label'] == 'my-alpha'
    assert st2.legend_config['roles']['unselected']['show'] is True
    # Unmodified roles keep their defaults.
    assert st2.legend_config['roles']['beta']['label'] == 'beta'


def test_legend_config_load_missing_field_uses_defaults():
    """An older session without `legend_config` must still load cleanly."""
    st = SessionState()
    payload = st.to_session_dict()
    payload.pop('legend_config', None)   # simulate v5-without-legend file
    st2 = SessionState()
    st2.from_session_dict(payload)
    # Defaults restored, no crash.
    assert st2.legend_config == _default_legend_config()
