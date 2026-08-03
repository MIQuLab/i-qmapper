# Changelog

All notable changes to this project are documented here. The format is
based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.0.5] — 2026-08

First public-ready release.

### Added
- Interactive QPU layout mapper for Jupyter (LUCJ + general circuits) on
  IBM heavy-hex backends: live calibration heatmaps, history/delta views,
  auto-LUCJ placement, layout scoring/rankings, JSON/PNG/code exports,
  time-lapse, session save/restore.
- Layout import from .json / .npy / .qpy  with hardened parsing
  (size caps, allow_pickle=False, partition validation).
- Per-user calibration cache (platformdirs) with atomic writes.
- Test suite (175 tests), Apache-2.0 licensing with IBM attributions.
- Companion paper: "I-QMapper: Error-Aware Layout Optimization and
  Device Diagnostics for NISQ Hardware" (arXiv:2606.27508).

### Security
- Path-traversal-safe file resolution (commonpath containment); HTML
  escaping across all widget sinks; malformed-input fallbacks on all
  untrusted import paths.
