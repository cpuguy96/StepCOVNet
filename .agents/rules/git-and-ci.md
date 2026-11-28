---
description: Git commits, pushes, rebases, pre-submit CI gates, multi-phase push cadence, cloud-VM git author, and Windows PowerShell git rules. Apply before committing, pushing, checking out, or changing CI/lint tooling.
trigger: model_decision
---

# Git and CI hygiene

- **CI gate:** Never push Python changes unless `pre_submit.py` passes (`--fast` for ruff + pyright; `--skip-install` for full CI mirror). Never use `--no-verify` unless explicitly requested.
- **Keep CI green:** When adding or modifying CI/lint/audit tooling, fix existing violations in the same change set.
- **Clean commits:** Review `git status` / `git diff` before committing; exclude unrelated local files (e.g. `logs/`, untracked scratch) and keep docs synced with code.
- **Multi-phase tasks:** On approved multi-phase work, commit and push to the remote after **each verified phase** rather than holding commits locally until the end.
- **Cloud-VM git author:** Fresh cloud VMs default `/etc/gitconfig` to `noreply@github.com` (`web-flow`). Ensure `~/.gitconfig` is set to the repo owner before the first commit (`bash scripts/colab_cloud_bootstrap.sh` sets it automatically).
- **Windows PowerShell git:** Pass multi-line commit messages with repeated `-m` flags (not bash heredocs or scratch files). Check `git status` and commit/stash before any checkout or line-ending normalization.
