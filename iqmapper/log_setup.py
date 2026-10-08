"""
log_setup.py — where the package's log records go.

Records from every ``iqmapper.*`` logger stay out of the notebook: the package
logger carries a ``NullHandler``, so Python's last-resort stderr handler never
prints them under a cell. The root logger is left alone.

When the UI starts, `install_file_log` adds a rotating log file (INFO and up)
in the user log directory and stops the package logger from propagating, so a
handler some environment put on the root logger cannot print the records
either. A message repeated with the same text is written once per session, so
a warning raised for every qubit of a chip takes one line.
"""
import logging
import logging.handlers
import os

PACKAGE_LOGGER = 'iqmapper'
LOG_FILE_NAME = 'iqmapper.log'
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3

_file_handler = None


def install_null_handler():
    """Give the package logger a handler so records never fall through to
    the last-resort stderr handler. Idempotent."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    if not any(isinstance(h, logging.NullHandler) for h in logger.handlers):
        logger.addHandler(logging.NullHandler())


class _OncePerMessage(logging.Filter):
    """Lets each distinct message text through once."""

    def __init__(self):
        super().__init__()
        self._seen = set()

    def filter(self, record):
        try:
            text = record.getMessage()
        except Exception:
            return True
        if text in self._seen:
            return False
        self._seen.add(text)
        return True


def default_log_dir():
    """The user log directory for iqmapper (platform-specific)."""
    import platformdirs
    return platformdirs.user_log_dir('iqmapper')


def install_file_log(log_dir=None):
    """Attach the rotating file handler to the package logger, once.

    Returns the log file path, or None when the directory cannot be created
    or the file cannot be opened; then nothing is attached and nothing is
    printed.
    """
    global _file_handler
    if _file_handler is not None:
        return _file_handler.baseFilename
    try:
        directory = log_dir or default_log_dir()
        os.makedirs(directory, exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(
            os.path.join(directory, LOG_FILE_NAME), maxBytes=MAX_BYTES,
            backupCount=BACKUP_COUNT, encoding='utf-8', delay=True)
    except Exception:
        return None
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s %(name)s: %(message)s'))
    handler.addFilter(_OncePerMessage())
    logger = logging.getLogger(PACKAGE_LOGGER)
    if logger.level == logging.NOTSET or logger.level > logging.INFO:
        logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    _file_handler = handler
    return handler.baseFilename


def remove_file_log():
    """Detach and close the file handler (tests, or a caller that wants the
    file released)."""
    global _file_handler
    if _file_handler is None:
        return
    logger = logging.getLogger(PACKAGE_LOGGER)
    logger.removeHandler(_file_handler)
    logger.propagate = True
    _file_handler.close()
    _file_handler = None
