"""Package log records never print into the notebook; the log file takes
them, once per distinct message."""
import contextlib
import logging

import pytest

import iqmapper  # noqa: F401  (importing the package installs the NullHandler)
from iqmapper import log_setup


@contextlib.contextmanager
def _bare_root():
    """No handler above the package logger, as in a plain notebook kernel.
    pytest attaches its capture handlers to the root logger for the test
    call, so they are taken off inside the test body, not in a fixture."""
    root = logging.getLogger()
    saved = root.handlers[:]
    root.handlers = []
    try:
        yield
    finally:
        root.handlers = saved


@pytest.fixture(autouse=True)
def _no_file_log_left_behind():
    yield
    log_setup.remove_file_log()


def _read(path):
    for handler in logging.getLogger('iqmapper').handlers:
        handler.flush()
    with open(path, encoding='utf-8') as fh:
        return fh.read().splitlines()


def test_the_package_logger_carries_a_null_handler():
    handlers = logging.getLogger('iqmapper').handlers
    assert any(isinstance(h, logging.NullHandler) for h in handlers)


@pytest.mark.parametrize('name', ['iqmapper', 'iqmapper.auto_lucj',
                                  'iqmapper.data', 'iqmapper.ui.lucj_panel'])
def test_a_warning_writes_nothing_to_stderr(capsys, name):
    with _bare_root():
        logging.getLogger(name).warning('Requested 6 bridges does not fit')
    captured = capsys.readouterr()
    assert captured.err == '' and captured.out == ''


def test_ten_identical_messages_are_written_once(tmp_path, capsys):
    with _bare_root():
        path = log_setup.install_file_log(str(tmp_path / 'logs'))
        log = logging.getLogger('iqmapper.data')
        for _ in range(10):
            log.warning("fetch_qubit_prop failed: qubit=%d prop='%s'", 72, 't2')
        log.warning("fetch_qubit_prop failed: qubit=%d prop='%s'", 73, 't2')
        logging.getLogger('iqmapper.reservoir').info('k=1: cap reached')
        lines = _read(path)
    assert sum("qubit=72 prop='t2'" in line for line in lines) == 1
    assert sum("qubit=73 prop='t2'" in line for line in lines) == 1
    assert sum('cap reached' in line for line in lines) == 1   # INFO kept
    captured = capsys.readouterr()
    assert captured.err == '' and captured.out == ''


def test_the_file_log_lives_in_the_user_log_directory(monkeypatch, tmp_path):
    monkeypatch.setattr(log_setup, 'default_log_dir',
                        lambda: str(tmp_path / 'user-logs'))
    path = log_setup.install_file_log()
    assert path == str(tmp_path / 'user-logs' / 'iqmapper.log')
    assert log_setup.install_file_log() == path          # attached once
    file_handlers = [h for h in logging.getLogger('iqmapper').handlers
                     if isinstance(h, logging.FileHandler)]
    assert len(file_handlers) == 1


def test_removing_the_file_log_restores_propagation(tmp_path):
    log_setup.install_file_log(str(tmp_path / 'logs'))
    assert logging.getLogger('iqmapper').propagate is False
    log_setup.remove_file_log()
    assert logging.getLogger('iqmapper').propagate is True


def test_an_unusable_directory_is_skipped_silently(tmp_path, capsys):
    blocker = tmp_path / 'not-a-dir'
    blocker.write_text('x')
    assert log_setup.install_file_log(str(blocker / 'logs')) is None
    assert not any(isinstance(h, logging.FileHandler)
                   for h in logging.getLogger('iqmapper').handlers)
    captured = capsys.readouterr()
    assert captured.err == '' and captured.out == ''
