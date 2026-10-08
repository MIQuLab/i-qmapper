"""Starting the UI inside a test leaves no log file behind and no handler on
the package logger for the tests that follow (order matters here: the second
test checks what the first one left)."""
import logging


def test_a_ui_start_writes_its_log_under_tmp(tmp_path):
    from iqmapper import log_setup
    path = log_setup.install_file_log()          # what run()/launch_mapper do
    assert path == str(tmp_path / 'logs' / 'iqmapper.log')
    assert logging.getLogger('iqmapper').propagate is False


def test_the_next_test_starts_with_a_clean_package_logger():
    logger = logging.getLogger('iqmapper')
    assert logger.propagate is True
    assert not any(isinstance(h, logging.FileHandler) for h in logger.handlers)
