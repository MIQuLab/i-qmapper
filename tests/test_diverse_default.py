"""Diverse enumeration follows the chip family when a backend connects: ON
for heavy-hex (Heron), OFF for the square-lattice Nighthawk. A manual toggle
holds until another backend is connected.
"""
from types import SimpleNamespace

import pytest

from iqmapper.constants import resolve_coords

from test_scan_ui import _fake_device_data, _sync_spawn

fake_provider = pytest.importorskip('qiskit_ibm_runtime.fake_provider')
# FakeNighthawk's own notice that its numbers are not typical Nighthawk values.
pytestmark = pytest.mark.filterwarnings(
    'ignore:Properties of fake_nighthawk are not intended:UserWarning')


def _connect(monkeypatch, backends, first):
    """A builder connected to `backends[first]`, as launch_mapper builds it."""
    import iqmapper.async_util as au
    monkeypatch.setattr(au, 'default_spawn_bg', _sync_spawn)
    monkeypatch.setattr("iqmapper.data.generate_device_data", _fake_device_data)
    monkeypatch.setattr("iqmapper.connection._check_properties",
                        lambda b, n: (object(), None))

    from iqmapper.state import SessionState
    from iqmapper.ui.builder import MapperUIBuilder

    backend = backends[first]
    coords, chip = resolve_coords(backend)
    state = SessionState()
    state.active_backend, state.active_backend_name = backend, first
    state.active_chip, state.active_coords = chip, coords
    state.active_service = SimpleNamespace(backend=lambda n: backends[n])
    state.available_backends = list(backends)
    state.props_warning = None
    state.initial_props = object()
    state.active_ansatz = 'lucj'
    state.active_vendor = 'IBM'
    state.custom_colors = []
    state.cal_history_dates = ['_seed_']
    b = MapperUIBuilder(state)
    b.build()
    return b


def _switch(b, name):
    dd = b.dd_backend_switch
    if name not in [o[1] if isinstance(o, tuple) else o for o in dd.options]:
        dd.options = list(dd.options) + [name]
    dd.value = name
    assert b.state.active_backend_name == name


@pytest.fixture
def backends():
    return {'fake_fez': fake_provider.FakeFez(),
            'fake_nighthawk': fake_provider.FakeNighthawk()}


def test_heron_connects_with_diverse_enumeration_on(monkeypatch, backends):
    b = _connect(monkeypatch, backends, 'fake_fez')
    assert b.state.active_chip.startswith('Heron')
    assert b.chk_diverse_cap.value is True


def test_nighthawk_connects_with_diverse_enumeration_off(monkeypatch,
                                                          backends):
    b = _connect(monkeypatch, backends, 'fake_nighthawk')
    assert b.state.active_chip.startswith('Nighthawk')
    assert b.chk_diverse_cap.value is False


def test_a_manual_toggle_holds_until_another_backend_connects(monkeypatch,
                                                              backends):
    b = _connect(monkeypatch, backends, 'fake_nighthawk')
    b.chk_diverse_cap.value = True                     # the user opts in
    b._load_device_graph()                             # date / hour reload
    assert b.chk_diverse_cap.value is True

    _switch(b, 'fake_fez')
    assert b.chk_diverse_cap.value is True
    b.chk_diverse_cap.value = False                    # the user opts out
    b._load_device_graph()
    assert b.chk_diverse_cap.value is False

    _switch(b, 'fake_nighthawk')
    assert b.chk_diverse_cap.value is False
    _switch(b, 'fake_fez')
    assert b.chk_diverse_cap.value is True             # Heron default again
