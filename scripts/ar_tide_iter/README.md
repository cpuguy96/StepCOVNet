# AR tide overfit iteration harness

Train/eval loop for free-run **634/634 @ 20 ms** experiments on tide.

| Path | Role |
| ---- | ---- |
| `scripts/ar_tide_iter/session_brief.py` | **Agent decision brief** — session best, config diffs vs prior runs, tried recipes |
| `logs/ar_tide_iter/next_experiment.json` | **Agent-written** next plan (gitignored); see `next_experiment.example.json` |
| `scripts/ar_tide_iter/run_overnight.py` | Agent autoresearch: `--autoresearch --once` or `--autoresearch --hours N`; lattice planner: `--hours --allow-planner` |
| `scripts/ar_tide_iter/run_summary.py` | JSON summary + exit-code remap for `--autoresearch` |
| `scripts/ar_tide_iter/overnight_planner.py` | Unattended JSON lattice search (not for meaningful autoresearch) |
| `.cursor/skills/autoresearch/SKILL.md` | Generic one-prompt autoresearch loop |
| `.cursor/skills/autoresearch/profiles/ar-tide-overfit.md` | Tide iter: brief, `next_experiment.json`, `--once` |
| `scripts/ar_tide_iter/run_exp.py` | One experiment: build config, train (WSL GPU), offline `--ar_decode` eval, append logs |
| `scripts/ar_tide_iter/results_history.py` | Parse `results.jsonl` and config snapshots for briefs |
| `scripts/ar_tide_iter/experiments.json` | Registry — one recipe per id (tracked) |
| `scripts/ar_tide_iter/experiments.README.md` | Registry vs retries vs config snapshots |
| `scripts/ar_tide_iter/config_builder.py` | Merge registry + champion template → full config JSON |
| `logs/ar_tide_iter/` | Gitignored outputs: built configs, train logs, `results.jsonl`, `ITER_LOG.md` |
| `docs/research/AR_TIDE_OVERFIT_ITER_LOG.md` | Tracked human summary (also appended by `run_exp.py`) |

## Quick start

**Agent-driven session** (read results → you decide → run one plan):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
REM write logs/ar_tide_iter/next_experiment.json — only overrides to change (see next_experiment.example.json)
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --autoresearch --once --brief none
```

Look for `=== AUTORESEARCH_SUMMARY ===` in output (teacher/free-run counts, remapped exit code).

**Budgeted autoresearch:** when the user gives hours (e.g. 7 h), the agent must **replan and run until deadline or 634/634** — not stop when a fixed plan list ends. See [.cursor/skills/autoresearch/SKILL.md](../../.cursor/skills/autoresearch/SKILL.md) § Budget discipline.

**Single manual experiment** (recipe already in `experiments.json`):

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --notes "overnight hypothesis"
```

`run_exp.py` freezes a per-attempt config snapshot under `logs/ar_tide_iter/configs/`.
**Scratch training:** `init_model_path` is never used for tide iteration — every run starts from random init (stripped in `config_builder`). Old registry entries may still list it for history; built configs omit it.
To regenerate attempt-1 snapshots for every registry id without training:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --build-all
```

Primary metric: `ar_decode.ordered_onset_match` @ 20 ms tolerance (634 onsets).

**Champion / iteration policy** (`configs/ar/tide_overfit.json`): offline AR decode only; `checkpoint_metric: val_overfit_gate`. Free-run gate via `debug_ar_onset_overfit.py --ar_decode` after each run. **Code changes require tests + pytest before GPU training.**

## Adding experiments

See [experiments.README.md](experiments.README.md). Summary:

| Situation | Edit `experiments.json`? | Command |
| --------- | ------------------------ | ------- |
| New hypothesis | **Yes** — new `id` | `run_exp.py --id iter39` |
| Recipe fix, same id | **Yes** — edit entry | `run_exp.py --id iter31 --retry-reason "…"` |
| Infra failure, same recipe | **No** | `run_exp.py --id iter31 --reuse-last-config` |

Retries log attempt 2+ with separate config/train snapshots when possible.

## Watching a run

Training output goes to gitignored `logs/ar_tide_iter/train_logs/<id>.log` (Keras progress bars make it noisy). Use:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/show_status.py --id iter30
venv\Scripts\python.exe scripts/ar_tide_iter/show_status.py --id iter30 --watch
```

`show_status` reads the gitignored train log and writes a snapshot to `logs/ar_tide_iter/status/<id>.json` (also gitignored — use the command above, not the file tree). Foreground `run_exp.py` also prints epoch lines and val metrics to the terminal.

**One GPU workload at a time.** Before training, `run_exp.py` takes an exclusive lock at `logs/ar_tide_iter/gpu_training.lock` and checks WSL `nvidia-smi` (`training_lock.assert_gpu_training_available`). This prevents two shells (e.g. a manual `run_exp` loop and `run_overnight.py`) from starting training in the same race window. Override with `STEPCOVNET_FORCE_GPU=1` or `run_exp.py --force` only when you are sure no other iteration is running. Run **one** driver shell at a time — do not leave an old `run_overnight.py --hours --allow-planner` process running alongside `--autoresearch`.
