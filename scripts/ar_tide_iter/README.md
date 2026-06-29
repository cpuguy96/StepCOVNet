# AR tide overfit iteration harness

Train/eval loop for free-run **634/634 @ 20 ms** experiments on tide.

| Path | Role |
| ---- | ---- |
| `scripts/ar_tide_iter/run_exp.py` | One experiment: build config, train (WSL GPU), offline `--ar_decode` eval, append logs |
| `scripts/ar_tide_iter/experiments.json` | Registry — one recipe per id (tracked) |
| `scripts/ar_tide_iter/experiments.README.md` | Registry vs retries vs config snapshots |
| `scripts/ar_tide_iter/config_builder.py` | Merge registry + champion template → full config JSON |
| `logs/ar_tide_iter/` | Gitignored outputs: built configs, train logs, `results.jsonl`, `ITER_LOG.md` |
| `docs/research/AR_TIDE_OVERFIT_ITER_LOG.md` | Tracked human summary (also appended by `run_exp.py`) |

## Quick start

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 ^
    --notes "overnight hypothesis"
```

`run_exp.py` freezes a per-attempt config snapshot under `logs/ar_tide_iter/configs/`.
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

**One GPU workload at a time.** Before training, StepCOVNet queries WSL `nvidia-smi` for **any** active GPU compute process (`wsl_gpu.assert_wsl_gpu_free_for_training`). This runs automatically in `maybe_dispatch_for_training` and `train_onset_ar.py`. Override with `STEPCOVNET_FORCE_GPU=1` or `run_exp.py --force`.
