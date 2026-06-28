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

Shared WSL shell vars: `scripts/wsl_common.sh`. See [wsl-gpu-stepcovnet](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md).

Pre-push CI mirror: `python pre_submit.py` (from repository root).

---

## Source

| Path | Contents |
| ---- | -------- |
| `src/stepcovnet/` | Main package (dense onset, arrows, shared utils) |
| `src/stepcovnet/onset_events/` | K-query event onset pipeline (research track) |
| `src/stepcovnet/onset_ar/` | Autoregressive onset (`gate-tide-overfit` …); design [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md); **Phase 0+1 implemented** — gate failing (EXP-20260627-02) |
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
| `configs/` | JSON experiment configs |
| `configs/overfit_tide/` | Tide single-song overfit smoke configs |
| `configs/onset_ar_tide.json` | AR tide overfit (`gate-tide-overfit`) |
| `configs/onset_ar_*.json` | AR smoke configs (10-song etc.; see AR design doc) |
| `scripts/train_onset_ar.py` | AR onset train / `--verify-only` — WSL GPU for full gate ([AR_ONSET_DESIGN.md §10](../research/AR_ONSET_DESIGN.md#10-experiment-protocol)) |

**Common entry points → skill:**

| Script | Skill |
| ------ | ----- |
| `scripts/train_onset.py` | [wsl-gpu-stepcovnet](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_onset_event.py` | [wsl-gpu-stepcovnet](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_onset_ar.py` | [wsl-gpu-stepcovnet](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) — AR tide gate |
| `scripts/run_overfit_tide_suite.py` | [tide-overfit-protocol](../../.cursor/skills/tide-overfit-protocol/SKILL.md) |
| `scripts/run_overfit_tide_ablations.py` | [tide-ablations](../../.cursor/skills/tide-ablations/SKILL.md) |
| `scripts/run_overfit_tide_bisection.py` | `EXP-11` — no skill yet ([skills README § Scripts without skills](../../.cursor/skills/README.md#scripts-without-skills-yet)) |
| `scripts/debug_onset_overfit.py` | [onset-event-eval-matching](../../.cursor/skills/onset-event-eval-matching/SKILL.md) |

---

## Data and models

| Path | Role |
| ---- | ---- |
| `data/v2/` | Legacy audio + `.txt` charts (train/val/test) |
| `data/raw_data/` | Downloaded StepMania packs (input to `dataset_prep`) |
| `data/final_data/` | Preprocessed nested output (`{bundle}/{id}/*.chart.json` + audio); **1942** chart rows locally; `training_index.json` for train/val |
| `models_wsl/` | WSL-trained checkpoints (gitignored patterns may apply) |
| `callbacks/` | TensorBoard / checkpoint roots |

---

## Docs

| Path | Role |
| ---- | ---- |
| `docs/research/` | Lab notebook (EXP, NOTE, paper) — incl. [AR_ONSET_DESIGN.md](../research/AR_ONSET_DESIGN.md) |
| `docs/research/archive/` | Superseded research plans (e.g. event-onset WP handoff) |
| `docs/agents/` | Agent state — [README](README.md), layout, self-journal |
| `.cursor/skills/` | Task playbooks (procedures) |
| `docs/onset_output_targets_planning.md` | Design planning |
| `AGENTS.md` (repo root) | Session entry router |
