"""
prop_options.py — property dropdowns list only what the loaded calibration has.

A property belongs in a dropdown when its array holds at least one finite
value. The same test serves a fresh fetch and a cached snapshot, since both
carry the full key set with NaN where the backend reports nothing. When no
property has values yet (nothing loaded), the full list is kept.
"""
import numpy as np


def has_values(arr):
    """True when `arr` holds at least one finite number."""
    if arr is None:
        return False
    try:
        values = np.asarray(arr, dtype=float)
    except (TypeError, ValueError):
        return False
    return values.size > 0 and bool(np.isfinite(values).any())


def available_options(options, data):
    """The (label, key) pairs of `options` whose array in `data` has values;
    all of `options` when none has."""
    data = data or {}
    kept = [(label, key) for label, key in options if has_values(data.get(key))]
    return kept or list(options)


def sync_dropdown(dropdown, options, preferred):
    """Give `dropdown` exactly `options`. Its value is kept when still listed,
    else `preferred` when listed, else the first option. Returns True when
    anything changed."""
    keys = [key for _label, key in options]
    if list(dropdown.options) == list(options):
        return False
    value = dropdown.value
    if value not in keys:
        value = preferred if preferred in keys else keys[0]
    dropdown.options = options
    dropdown.value = value
    return True
