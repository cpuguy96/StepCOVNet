# Project layout

**When to read:** Locating code, configs, scripts, or model artifacts. Routed from [AGENTS.md](../../AGENTS.md).

---

## Environment conventions

Commands assume **repository root** as the working directory (clone path is arbitrary).

| Workload | Executable | Notes |
| -------- | ---------- | ----- |
| CPU pytest, lint, `preprocess_dataset.py` | `python` (project venv activated) | From repository root |
| GPU train / MERT (WSL) | `python` after `source scripts/wsl_gpu_env.sh` | Override with `STEPCOVNET_WSL_PYTHON` or `WSL_VENV` |
| GPU from Windows | `python scripts/...` | Auto-dispatch via `wsl_gpu.py` when the script supports it |
| Checkpoints / callbacks | `models_wsl/`, `callbacks/` | Gitignored; paths in JSON configs are repo-relative |

Shared WSL shell vars: `scripts/wsl_common.sh`. See [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md).

Pre-push CI mirror: `python pre_submit.py` (from repository root).

---

## Source

| Path | Contents |
| ---- | -------- |
| `src/stepcovnet/ddc/` | DDC C-LSTM placement (`donahue2017ddc`, `M-ddc-20ms`) |
| `src/stepcovnet/ddcl/` | DDCL ConvLSTM placement (`omalley2025ddcl`, `M-slot48`) |
| `src/stepcovnet/itgpt/` | ITGPT hierarchical transformer placement (`omalley2026itgpt`, `M-slot48`) |
| `src/stepcovnet/core/` | PyTorch core dense onset, arrow sequence, generator, losses, ONNX export |
| `src/stepcovnet/runtime.py` | Shared PyTorch device, seeding, AMP/TF32, EMA, LR, and lazy SummaryWriter helpers |
| `src/stepcovnet/slot48_eval.py` | Shared ``M-slot48`` scoring, threshold sweep, null baseline, and audio-corruption helpers |
| `src/stepcovnet/beat_this_model.py` | Self-contained pure-PyTorch Beat This! neural beat/downbeat tracker (`foscarin2024beatthis`, MIT) |
| `src/stepcovnet/beat_tracking.py` | `BeatGrid`, Beat This! / `librosa` backends, offline fallback, transient snapping, ITGPT raw-audio inference |
| `src/stepcovnet/onset_events/` | K-query event onset pipeline (research track) |
| `src/stepcovnet/dataset_prep/` | Raw simfile → `final_data` preprocessing, `training_index`, `training_loader` (P8–P9) |
| `src/stepcovnet/pairing.py` | Audio/chart pairing; `list_training_samples` for `final_data` |
| `src/stepcovnet/mel_onset.py` | Mel spectrogram helpers (shared by dense path; breaks import cycles) |
| `src/stepcovnet/wsl_gpu.py` | WSL GPU bootstrap / re-exec helpers |

**Onset events modules (by pipeline stage):**

| Stage | Modules |
| ----- | ------- |
| PRE | `audio.py`, `frontend.py`, `preprocess.py`, `datasets.py` |
| MODEL | `models.py`, `encoder.py`, `losses.py` |
| POST | `inference.py` |
| METRICS | `metrics.py`, `matching.py`, `diagnostics.py` |
| Train | `trainers.py`, `config.py` |

See [PIPELINE_ARCHITECTURE.md](../research/PIPELINE_ARCHITECTURE.md) for the full mapping.

---

## Scripts and configs

| Path | Role |
| ---- | ---- |
| `scripts/` | CLI entry points (train, extract, suite runners) |
| `scripts/preprocess_dataset.py` | Raw simfile packs → nested `data/final_data` |
| `scripts/build_training_index.py` | Train/val split manifest (`training_index.json`) |
| `configs/` | JSON experiment configs — see [configs/README.md](../../configs/README.md) |
| `configs/dense/` | Dense frame onset baselines and scoreboard |
| `configs/event/` | K-query event onset |
| `configs/overfit_tide/` | Tide single-song overfit smoke configs |

**Common entry points → skill:**

| Script | Skill |
| ------ | ----- |
| `scripts/train_onset.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_onset_event.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/run_overfit_tide_suite.py` | [tide-overfit-protocol](../../.agents/skills/tide-overfit-protocol/SKILL.md) |
| `scripts/run_overfit_tide_ablations.py` | [tide-ablations](../../.agents/skills/tide-ablations/SKILL.md) |
| `scripts/run_overfit_tide_bisection.py` | `EXP-11` — no skill yet ([skills README § Scripts without skills](../../.agents/skills/README.md#scripts-without-skills-yet)) |
| `scripts/debug_onset_overfit.py` | [onset-event-eval-matching](../../.agents/skills/onset-event-eval-matching/SKILL.md) |

---

## Data and models

| Path | Role |
| ---- | ---- |
| `data/v2/` | Legacy audio + `.txt` charts (train/val/test) |
| `data/raw_data/` | Downloaded StepMania packs (input to `dataset_prep`) |
| `data/final_data/` | Preprocessed nested output (`{bundle}/{id}/*.chart.json` + audio); **1942** chart rows locally; `training_index.json` for train/val |
| `models_wsl/` | WSL-trained checkpoints (gitignored patterns may apply) |
| `callbacks/` | TensorBoard / checkpoint roots |
| `logs/` | Local command captures (gitignored) — not repo-root `tmp_*` |
| `_tmp/` | Disposable scratch files (gitignored); delete when task ends |

---

## Docs

| Path | Role |
| ---- | ---- |
| `docs/research/` | Lab notebook (EXP, NOTE, paper) — incl. [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md) |
| `docs/research/archive/` | Superseded research plans (e.g. event-onset WP handoff) |
| `docs/agents/` | Agent state — [README](README.md), layout, self-journal |
| `.agents/skills/` | Task playbooks (procedures) |
| `docs/onset_output_targets_planning.md` | Design planning |
| `AGENTS.md` (repo root) | Session entry router |
