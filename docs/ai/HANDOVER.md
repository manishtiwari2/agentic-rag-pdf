# Handover (keep to 15 lines; overwrite each session)

- **Updated:** 2026-11-26
- **Goal:** make the repo production-ready (see `ROADMAP.md`; full brief in `source/PRODUCTION-PROMPT.md`, local only).
- **Done:** three-repo code review (`source/Three-Repo-Code-Guide.html`, local); AI workflow docs added on branch `docs/ai-workflow`.
- **State of main:** `c5b40aa`. Open issues: #43 (conflict benchmark), #46 (py3.10 test flake).
- **Next step:** start ROADMAP P1.1 (fail-closed verifier) on a new branch.
- **Key files:** `src/pipeline.py`, `src/config.py`, `src/generation/`, `src/agents/`, `src/chat/ui.py`.
- **Commands:** `python -m pytest -o addopts="" -q` (all tests); `python -m src.cli validate-benchmark`.
- **Open decisions:** none.
