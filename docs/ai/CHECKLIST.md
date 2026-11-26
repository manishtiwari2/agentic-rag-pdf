# Checklist: prove it works before approving a diff

Run every step. Paste the real output. A step that was skipped must be marked
"skipped" with the reason.

## 1. Scope
- [ ] `git branch --show-current` is not `main`.
- [ ] `git diff --stat` shows only files this task needs.
- [ ] No file from `CONSTRAINTS.md` was changed without approval.

## 2. Code
- [ ] No new package outside `DEPENDENCIES.md`.
- [ ] No dead code, no commented-out code, no unused imports.
- [ ] Every new prompt with document text includes `DOCUMENT_TEXT_IS_DATA`.
- [ ] No page numbers sent to the LLM.
- [ ] Errors use `src/errors.py` classes. No bare `except: pass`.

## 3. Tests
- [ ] Targeted tests for the changed area pass: `python -m pytest tests/test_<area>.py -o addopts="" -q`
- [ ] A new test covers the new behaviour, including one failure case.
- [ ] Full suite passes before the PR: `python -m pytest -o addopts="" -q 2>&1 | tail -20`
- [ ] Benchmark files still valid: `python -m src.cli validate-benchmark`

## 4. Behaviour (when answers can change)
- [ ] One real question run end to end: `python -m src.cli ask --help` for the flags.
- [ ] The answer cites real pages, or refuses.
- [ ] Quality claims come with a paired-bootstrap CI and say which model stack was used.

## 5. Records
- [ ] New design choice added as the next `DD-0xx` in `docs/DESIGN_DECISIONS.md`.
- [ ] `BUGS.md` updated if a bug was found or fixed.
- [ ] Commit message in simple English, no co-author lines.
- [ ] After merge: `DIFFS.md`, `MODELS.md` and `HANDOVER.md` updated.
