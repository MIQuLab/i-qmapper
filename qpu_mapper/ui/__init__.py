"""ui/ subpackage — interactive interface builders.

The public entry point is `build_mapper_ui`, re-exported here so that
existing callers can keep using either:

    from qpu_mapper.ui import build_mapper_ui   (subpackage, new path)
    from qpu_mapper.ui import MapperUIBuilder   (direct class access)

There is no separate `ui.py` shim — this package `__init__` IS the
compatibility layer: `from qpu_mapper.ui import build_mapper_ui` resolves
here unchanged, exactly as it did when `ui` was a single module file.
"""
from .builder import MapperUIBuilder, build_mapper_ui

__all__ = ['MapperUIBuilder', 'build_mapper_ui']
