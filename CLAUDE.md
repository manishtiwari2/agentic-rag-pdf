# CLAUDE.md: rules for AI sessions in this repo

Read this first, every session. These rules always apply. Your global
`~/.claude/CLAUDE.md` also applies. If the two conflict, the stricter rule wins.

## Start of every session
1. Read `docs/ai/HANDOVER.md`. It says where the last session stopped.
2. Read `docs/ai/CONSTRAINTS.md`. Never break it.
3. Pick ONE task from `docs/ai/ROADMAP.md`. One task per session or prompt.
4. Run `git status` and `git branch --show-current`. Never work on `main`.

## Always
- Simple English in code comments, commits, PRs and docs.
- Shortest clear code. Reuse what exists in `src/`. No dead or commented-out code.
- Follow the existing style: config dataclasses in `src/config.py`, errors from
  `src/errors.py`, prompts in `src/generation/prompts.py`.
- Every prompt that contains document text must include `DOCUMENT_TEXT_IS_DATA`.
- Never show page numbers to the LLM. Pages come from chunk metadata in code
  (`src/generation/citations.py`).
- New dependency? Check `docs/ai/DEPENDENCIES.md` first. If it is not listed, ask me.
- New design choice? Add the next `DD-0xx` entry to `docs/DESIGN_DECISIONS.md`
  (DD-075 is reserved for issue #43, so the next new one is DD-076):
  what, why, alternatives, how it is checked.
- Before you say "done", run every step in `docs/ai/CHECKLIST.md`.
- Bug or wrong AI output found? Add it to `docs/ai/BUGS.md`.
- Show test command output. Never claim a test passed without running it.

## Git
- Branch per task. Stage files by name. Never `git add .` or `git add -A`.
- No `Co-Authored-By` or "Generated with" lines. Ever.
- Ask me before any push, PR or merge. Show the exact command first.
- Before a push, run `git log origin/main..HEAD` and `git diff origin/main --stat`.
- After a merge, add a line to `docs/ai/DIFFS.md` and `docs/ai/MODELS.md`.
- Something broke? Use `docs/ai/ROLLBACK.md`. Never `git reset --hard` or
  force-push without asking me.

## End of every session
Update `docs/ai/HANDOVER.md` (15 lines or fewer): goal, what was done, next step,
key files and exact commands.

## Map of the AI docs
| File | Use it for |
|---|---|
| `docs/ai/CONSTRAINTS.md` | Files, deps and logic that must not be touched |
| `docs/ARCHITECTURE.md` | Full system design (the detailed one) |
| `docs/DESIGN_DECISIONS.md` | Why each choice was made (the decisions log) |
| `docs/ai/FLOW.md` | How data moves, as diagrams |
| `docs/ai/MENTAL-MODEL.md` | The whole system in plain English |
| `docs/ai/HANDOVER.md` | Where the last session stopped |
| `docs/ai/CHECKLIST.md` | Proof steps before approving a diff |
| `docs/ai/BUGS.md` | Known bugs, AI mistakes, regressions |
| `docs/ai/ROADMAP.md` | Small tasks, one per prompt |
| `docs/ai/ROLLBACK.md` | Safe git undo recipes |
| `docs/ai/DEPENDENCIES.md` | Approved packages |
| `docs/ai/DIFFS.md` | Log of merged changes and hashes |
| `docs/ai/ENV.template.md` | Every setting, secret and tool the app needs |
| `docs/ai/MODELS.md` | Which AI model built which part |
