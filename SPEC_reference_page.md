# SPEC — Reference page and tooltip pointers

**Branch:** `feature/ancilla-range` · English only · Milana runs every git command.
No frozen engine is touched (verify the 19 md5s at start and end; any change → STOP).

## Situation

README.md and TUTORIAL.md were rewritten (commits aa07867..02f8abd). They link to
`docs/` pages that do not exist, and `tests/test_copy_lengths.py` fails (2 tests)
because the tooltips still say `· README: <entry>` and the README no longer has
those headings.

## Files provided — copy verbatim over the repo

- `docs/reference.md` (new): every tooltip entry plus the benchmarking reference.
- `README.md`, `TUTORIAL.md`: same as committed, with the links changed to
  `docs/reference.md` (no other change).

## Tasks

1. **Tooltip pointers.** In `iqmapper/ui/`, change every `· README: <entry>` to
   `· Reference: <entry>`. All ten entries (LQS, Search cap, Search engine,
   Diverse enumeration, Multi-circuit planner, Delta references, Offline mode,
   Levels, Relative mode, Calibration check) are headings in `docs/reference.md`.
2. **Test.** `tests/test_copy_lengths.py`: collect headings from
   `docs/reference.md` instead of README.md, match the `Reference:` pointer
   format, and drop the assertion on an "Understanding the messages" section.
   Keep the copy-length checks unchanged.
3. **Fact check.** Every button, checkbox, default, status line and message
   quoted in `docs/reference.md` must match the current UI copy; fix the page
   (not the code) where it differs and list each correction.
4. **Links.** No link in README.md, TUTORIAL.md, docs/reference.md or
   CODE_OVERVIEW.md may point to a missing file or heading (check
   `README.md#installation`, every `docs/reference.md#...` anchor, CODE_OVERVIEW).
   Grep the whole repo for `user-guide.md`, `benchmark.md`, `development.md`,
   `Understanding the messages`: zero hits left.
5. **CHANGELOG** `[Unreleased]`, Changed: "Documentation reorganised: short README,
   tutorial, and a reference page under docs/".
6. `python -m pytest -q -W error` green, 0 warnings; `python -m build` +
   `twine check dist/*` pass (README has no images).

Report: corrections from step 3, `git diff --stat`, md5 table, one commit message:
`Docs: reference page under docs/; tooltip pointers → Reference; links fixed`
