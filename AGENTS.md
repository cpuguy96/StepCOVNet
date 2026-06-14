# StepCOVNet — agent entry point

Read this file first. It routes you to topic context under `docs/agents/` — do not load everything at once.

**Living doc:** When the user establishes a new convention, work area, or routing need, update this table and add or extend a file under `docs/agents/`. Keep this file minimal (~routing only).

---

## Quick routing

| If the task involves… | Read first |
| --------------------- | ---------- |
| Matching playbook for tide overfit, ablations, WSL, logging, diagnostics | [docs/agents/skills-index.md](docs/agents/skills-index.md) |
| Agent mistakes, process improvements, skill gaps | [docs/agents/self-journal.md](docs/agents/self-journal.md) |
| What we're researching now, latest results, next steps | [docs/agents/current-focus.md](docs/agents/current-focus.md) |
| Experiments, discussion notes, paper, pipeline architecture | [docs/agents/research.md](docs/agents/research.md) |
| Training, overfit smoke tests, WSL GPU, configs | [docs/agents/training-gpu.md](docs/agents/training-gpu.md) |
| Python style, tests, imports, venv, docstrings | [docs/agents/code-conventions.md](docs/agents/code-conventions.md) |
| Repo layout (`src/`, `scripts/`, `configs/`, models) | [docs/agents/project-layout.md](docs/agents/project-layout.md) |

Full index and extension guide: [docs/agents/README.md](docs/agents/README.md).

---

## Always-on rules (Cursor)

These apply in every session (`.cursor/rules/`, `alwaysApply: true`). Follow them; do not copy their full text here.

| Rule file | Topic |
| --------- | ----- |
| `python-venv.mdc` | Use `venv\Scripts\python.exe` on Windows for CPU/pytest |
| `wsl-gpu.mdc` | GPU training via WSL + `~/stepcovnet-venv-wsl` |
| `research-notebook.mdc` | Log EXP/NOTE, timestamps, update paper outline |
| `tests-coverage.mdc` | Tests + coverage for code changes |
| `python-imports.mdc` | Module-level imports only |
| `python-docstrings.mdc` | Docstring conventions |
| `removed-features-no-backcompat.mdc` | No legacy/backcompat commentary |

---

## Session checklist (research / training work)

0. **Skills** — check [skills-index.md](docs/agents/skills-index.md) for a matching playbook; read that skill before improvising.
1. **Route** — open the row from the table above that matches the task.
2. **Read** — load only the linked `docs/agents/*.md` and the specific research doc it points to (e.g. one EXP entry, not the whole log).
3. **Persist** — per `research-notebook.mdc`: log runs and insights; update `PAPER_OUTLINE.md` when findings land.
4. **Extend routing** — new recurring topic → new `docs/agents/<topic>.md` + row in this file.

---

## North star

Target system shape: [docs/research/PIPELINE_ARCHITECTURE.md](docs/research/PIPELINE_ARCHITECTURE.md) (audio → PRE → MODEL → POST → METRICS → training feedback). Structure code, tests, and experiments around swappable stages.
