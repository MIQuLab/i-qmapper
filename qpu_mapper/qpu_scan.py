"""
qpu_scan.py — Multi-QPU scan engine (pure, synchronous, testable).

Two concerns, no threads (all async/threading belongs to the 3b UI layer):

1. :func:`list_scan_targets` — enumerate the account's backends and annotate
   each with topology-family + size compatibility for a given ``norb``.
2. :func:`scan_qpus` — run the ancilla sweep on each compatible backend using
   injected ``get_backend`` / ``get_props`` callables, isolating per-backend
   failures and reporting progress via callback.

Injection, not coupling
------------------------
This module never touches the service, the network, or the calibration cache
directly. The caller (3b) injects:

* ``get_backend(name) -> backend`` — resolve a live backend object.
* ``get_props(backend) -> props`` — a ``BackendProperties``-like object
  exposing ``readout_error(q)`` and ``gate_error(name, (u, v))``, obtained
  through the SAME cache-aware ``data.py`` path the rest of the app uses
  (never a parallel fetch). 3b is responsible for that wiring.

Family detection
----------------
Family is read cheaply from ``backend.processor_type`` (or
``configuration().processor_type``) — no calibration fetch. We deliberately do
NOT go through ``constants.resolve_coords`` for the *scan*: it ties a family to
one exact qubit count, which would misreport a same-family-but-wrong-size
backend as "unknown" and erase the distinct "too small" signal. ``resolve_coords``
remains the authority for actually laying a chip out; here we only need the
family label and the raw qubit count, decoupled.

Cross-QPU comparability of ``best_lqs``
---------------------------------------
``best_lqs`` is the max ``product_score`` across ALL strata of a backend's
sweep. Because the LUCJ zigzag template for a given ``norb`` is identical on
every backend (same chain length, same bridge geometry), these multiplicative
fidelity scores are comparable across QPUs at the same ``(norb, k-range)``.
Caveat: ``product_score`` grows as bridges are removed, so the min-k stratum
always dominates ``best_lqs`` — a single headline number understates the
trade-off. The per-stratum table (rendered in 3b) is the honest view; treat
``best_lqs`` as a coarse "which QPU is worth a closer look" ranking only.
"""

import contextlib
import logging

_log = logging.getLogger(__name__)


def _detect_family(backend):
    """Return the topology family label (e.g. ``'Heron'``) or None.

    Mirrors the family-extraction half of ``constants.resolve_coords`` but
    without its qubit-count coupling. Reads ``processor_type`` directly, then
    falls back to ``configuration().processor_type``.
    """
    proc = getattr(backend, 'processor_type', None)
    if proc is None:
        cfg = getattr(backend, 'configuration', lambda: None)()
        if cfg is not None:
            proc = getattr(cfg, 'processor_type', None)
    if isinstance(proc, dict):
        fam = proc.get('family')
        return str(fam) if fam else None
    if proc is not None:
        # e.g. 'Heron_r3' -> 'Heron'
        return str(proc).split('_')[0] or None
    return None


def _family_key(label):
    """Normalise a family label to a comparison key, ignoring revision.

    ``'Heron r3'`` and ``'Heron'`` both key to ``'heron'`` — revisions of the
    same family share topology, so they scan together.
    """
    if not label:
        return None
    return str(label).strip().split()[0].lower()


def list_scan_targets(state_or_service, norb, *, same_family_as=None,
                      include_other_families=False):
    """List account backends annotated with compatibility for ``norb``.

    Args:
        state_or_service: a SessionState (``.active_service`` used) or a
            service object directly. Must expose ``backends()`` yielding
            backend objects with ``name`` / ``num_qubits`` / ``processor_type``.
        norb: number of spatial orbitals for the intended LUCJ layout.
        same_family_as: the current backend's family label (as
            ``resolve_coords`` reports it, e.g. ``'Heron r3'``). When set and
            ``include_other_families`` is False, backends of a different family
            are marked ineligible. Cross-family zigzag comparison is
            apples-to-oranges, so same-family is the default scope.
        include_other_families: widen the scan to every family (and to
            unknown-family backends). Each result still carries its family
            label so the UI can badge it.

    Returns one dict per backend:
        ``{name, family, n_qubits, eligible: bool, reason: str|None}``
    ``reason`` is set only when ``eligible`` is False:
        * ``'different family (X)'`` — filtered out by family scope,
        * ``'needs >= M qubits, has N'`` — too small to host 2*norb+1 qubits,
        * ``'family unknown'`` — detection failed (excluded unless
          ``include_other_families``).
    """
    service = getattr(state_or_service, 'active_service', None) or state_or_service
    if not hasattr(service, 'backends'):
        return []

    # Floor: two chains of `norb` plus at least one bridge (k>=1) — the
    # smallest layout the sweep can produce. Topology still has to fit (VF2
    # decides that at sweep time); this is only the cheap size pre-filter.
    min_needed = 2 * norb + 1

    targets = []
    for backend in service.backends():
        name = getattr(backend, 'name', None)
        family = _detect_family(backend)
        n_qubits = int(getattr(backend, 'num_qubits', 0) or 0)

        eligible = True
        reason = None

        if family is None:
            if not include_other_families:
                eligible = False
                reason = 'family unknown'
            # else: kept, family stays None so 3b can badge "unknown"
        elif (same_family_as is not None and not include_other_families
              and _family_key(family) != _family_key(same_family_as)):
            eligible = False
            reason = f'different family ({family})'

        # Size gate applies to whatever survived the family gate.
        if eligible and n_qubits < min_needed:
            eligible = False
            reason = f'needs >= {min_needed} qubits, has {n_qubits}'

        targets.append({
            'name': name,
            'family': family,
            'n_qubits': n_qubits,
            'eligible': eligible,
            'reason': reason,
        })
    return targets


def _best_from_sweep(sweep):
    """Return ``(best_product_score, best_k)`` over all strata, or (None, None).

    ``best_k`` is the ancilla count of the highest-``product_score`` candidate.
    """
    best_lqs = None
    best_k = None
    for k, entries in sweep.get('strata', {}).items():
        for entry in entries:
            ps = entry['product_score']
            if best_lqs is None or ps > best_lqs:
                best_lqs = ps
                best_k = k
    return best_lqs, best_k


def scan_qpus(targets, norb, get_backend, get_props, *,
              min_bridges=None, top_n=3, progress_cb=None):
    """Run the ancilla sweep on each target backend, isolating failures.

    Args:
        targets: entries from :func:`list_scan_targets`. Plain name strings are
            also accepted. Entries with ``eligible=False`` are skipped (status
            ``'skipped'``); everything else is swept.
        norb: spatial orbitals (passed straight to the sweep).
        get_backend: ``name -> backend object`` (injected; keeps this module
            free of service/network specifics and makes it testable).
        get_props: ``backend -> props`` (injected; MUST be wired by 3b to the
            cache-aware ``data.py`` path).
        min_bridges, top_n: forwarded to ``sweep_ancilla_layouts``.
        progress_cb: optional ``callable(name, status, detail)`` with status in
            ``'start' | 'ok' | 'skipped' | 'error'``. Exceptions raised by the
            callback are swallowed — progress reporting must never break a scan.

    Returns::

        {
          'results': [ {name, family, sweep, best_lqs, best_k, error} ... ],
          'n_scanned', 'n_failed', 'n_skipped'
        }

    ``results`` is ordered: successful sweeps first, sorted by ``best_lqs``
    descending; skipped/failed backends last (in input order), with ``error``
    set and ``sweep`` None on failures.
    """
    # Lazy import so tests can monkeypatch the module attribute, and so this
    # module stays import-light for the list_scan_targets-only callers.
    from .ancilla_sweep import sweep_ancilla_layouts

    def _emit(name, status, detail=''):
        if progress_cb is None:
            return
        with contextlib.suppress(Exception):
            progress_cb(name, status, detail)

    def _entry(name, family, sweep=None, best_lqs=None, best_k=None, error=None):
        return {'name': name, 'family': family, 'sweep': sweep,
                'best_lqs': best_lqs, 'best_k': best_k, 'error': error}

    results = []
    n_scanned = n_failed = n_skipped = 0

    for target in targets:
        if isinstance(target, dict):
            name = target.get('name')
            family = target.get('family')
            eligible = target.get('eligible', True)
        else:
            name, family, eligible = target, None, True

        if not eligible:
            n_skipped += 1
            results.append(_entry(name, family))
            _emit(name, 'skipped', target.get('reason', '') if isinstance(target, dict) else '')
            continue

        _emit(name, 'start')
        try:
            backend = get_backend(name)
            props = get_props(backend)
            sweep = sweep_ancilla_layouts(
                norb, backend, props,
                min_bridges=min_bridges, top_n=top_n)
        except Exception as exc:
            # PER-BACKEND ISOLATION: one bad backend must not kill the scan.
            _log.warning("QPU scan: backend %s failed: %s", name, exc)
            n_failed += 1
            results.append(_entry(name, family, error=str(exc)))
            _emit(name, 'error', str(exc))
            continue

        best_lqs, best_k = _best_from_sweep(sweep)
        if best_lqs is None:
            # Sweep ran but yielded no candidates — nothing to rank.
            n_skipped += 1
            results.append(_entry(name, family, sweep=sweep))
            _emit(name, 'skipped', 'no candidates')
            continue

        n_scanned += 1
        results.append(_entry(name, family, sweep=sweep,
                              best_lqs=best_lqs, best_k=best_k))
        _emit(name, 'ok', f'best LQS {best_lqs * 100:.1f}%')

    # Successful sweeps first, ranked by best_lqs desc; the rest keep input
    # order at the end.
    ranked = [r for r in results if r['best_lqs'] is not None]
    rest = [r for r in results if r['best_lqs'] is None]
    ranked.sort(key=lambda r: r['best_lqs'], reverse=True)

    return {
        'results': ranked + rest,
        'n_scanned': n_scanned,
        'n_failed': n_failed,
        'n_skipped': n_skipped,
    }
