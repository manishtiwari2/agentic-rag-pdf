# Rollback recipes

Always run `git status` and `git branch --show-current` first. Ask Manish
before anything marked **ASK**.

## Undo uncommitted changes
| Situation | Command |
|---|---|
| One file went wrong | `git restore path/to/file.py` |
| A file was staged by mistake | `git restore --staged path/to/file.py` |
| Keep the changes aside for later | `git stash push -m "why" path/to/file.py` |
| Everything on the branch is bad (**ASK**) | `git restore .` |

## Undo commits that are NOT pushed
| Situation | Command |
|---|---|
| Fix the last commit message | `git commit --amend` |
| Undo the last commit, keep its changes | `git reset --soft HEAD~1` |
| Throw away the last commit (**ASK**) | `git reset --hard HEAD~1` |

## Undo commits that ARE pushed or merged
Never rewrite shared history. Add a new commit that undoes the old one.
| Situation | Command |
|---|---|
| Undo one commit | `git revert <hash>` |
| Undo a merged PR | `git revert -m 1 <merge-hash>` |
| Get one file back from an old commit | `git restore --source <hash> path/to/file.py` |

## Find the good version
- `git log -n 20 --oneline` lists recent commits.
- `git log -n 10 --oneline -- path/to/file.py` shows the history of one file.
- `git diff <hash> -- path/to/file.py` shows what changed since that commit.
- Merged hashes are also listed in `DIFFS.md`.

## A big refactor went sideways
1. Stop. Do not make more edits.
2. `git diff --stat` shows the damage.
3. If nothing is committed: `git stash push -m "bad refactor"`, so it can be recovered later.
4. Start again from the ROADMAP task, but smaller: one file per step.
5. Add a row to `BUGS.md` (type: AI mistake).
