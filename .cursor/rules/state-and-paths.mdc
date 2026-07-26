---
description: Refresh repo state before routing edits; portable paths, scratch hygiene, Windows git hygiene
alwaysApply: true
---

# State refresh and portable paths

## When to refresh first

Before edits that touch **linkages** — routing (`AGENTS.md`), README, workflows, rules, skills, WSL shell scripts, docs cross-links, gitignore allowlists, or path conventions — refresh state:

1. `git status` and `git log -1 --oneline`
2. `git ls-files` for paths you will cite or link
3. Read target files; confirm they exist on disk (especially gitignored `docs/` and `.cursor/`)
4. For WSL helpers: prefer `scripts/wsl_common.sh` + env vars over hardcoded clone or username paths

If a linked file is missing, fix the link or create the target — do not leave broken references in **tracked** files.

## Machine-specific paths (tracked files only)

Use `git grep`, not workspace-wide search (gitignored local docs may contain stale paths):

```text
git grep -E "C:\\\\Users|/mnt/c/Users/"
```

Expect **no matches** in tracked files.

## Windows + WSL (portable)

| Workload | Pattern |
| -------- | ------- |
| Windows CPU | `venv\Scripts\python.exe` from repo root |
| WSL GPU Python | `"${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"` (Linux `$HOME`, override with `WSL_VENV`) |
| WSL from Windows | `venv\Scripts\python.exe scripts/...` when script uses `wsl_gpu` dispatch |
| Data / models | `data/v2`, `data/final_data`, `models_wsl/` (repo-relative) |

Never hardcode `C:\Users\...` or `/mnt/c/Users/<name>/...` clone paths. WSL `$HOME/stepcovnet-venv-wsl` is a **project default venv name**, not a machine-specific Windows path.

## Disposable scratch (agents)

| Purpose | Path |
| ------- | ---- |
| One-off scripts, parse helpers, throwaway analysis | `_tmp/<topic>/...` (gitignored) |
| Command capture | `logs/<topic>_<run>.log` (gitignored) |

**Do not** create `tmp_*` or `_tmp_*` files at the **repository root**. Delete `_tmp/` contents when done; never commit them.

On user steering about agent process (paths, hygiene, “remember this”), run [steering-correction-promotion](../../.cursor/skills/steering-correction-promotion/SKILL.md) **same turn** — journal alone is incomplete.

## Git on Windows (PowerShell)

**Commit messages** — build multi-line messages with **repeated `-m` flags**, one per paragraph. Bash heredoc (`-m "$(cat <<'EOF' …)"`) does not parse in PowerShell, and a scratch file + `git commit -F` is **not** the fallback — `_tmp/` is for analysis artifacts, not commit prose.

```powershell
git commit -m "subject line" -m "body paragraph" -m "second paragraph"
```

**Destructive checkout** — `git checkout -- <path>` and line-ending renormalization discard uncommitted edits with no prompt. Check `git status` and commit or stash **first**. With `core.autocrlf=true` this is easy to trigger while fixing `*.sh` endings.

## Pre-push validation

Mirror CI before push: `venv\Scripts\python.exe pre_submit.py` (see [pre-submit.yml](../../.github/workflows/pre-submit.yml)).
