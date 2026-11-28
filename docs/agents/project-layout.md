# Project layout

**When to read:** Locating code, configs, scripts, or model artifacts. Routed from [AGENTS.md](../../AGENTS.md).

---

## Environment conventions

Commands assume **repository root** as the working directory. See [python-env.md](../../.agents/rules/python-env.md) for Windows vs WSL vs cloud-VM Python executables, `_tmp/`/`logs/` scratch paths, and `pre_submit.py`. Checkpoints and TensorBoard roots (`models_wsl/`, `callbacks/`) are repo-relative and gitignored.

---

## Source

| Path | Contents |
| ---- | -------- |
| `src/stepcovnet/ddc/` | DDC C-LSTM placement (`donahue2017ddc`, `M-ddc-20ms`) |
| `src/stepcovnet/ddcl/` | DDCL ConvLSTM placement (`omalley2025ddcl`, `M-slot48`) |
| `src/stepcovnet/itgpt/` | ITGPT hierarchical transformer placement (`transformer.py`, `encoders.py`, `losses.py`, `models.py`, `evaluation.py`, `trainers.py`) |
| `src/stepcovnet/core/` | PyTorch core dense onset, arrow sequence, generator, losses, ONNX export |
| `src/stepcovnet/runtime.py` | Shared PyTorch device, seeding, AMP/TF32, EMA, LR, and lazy SummaryWriter helpers |
| `src/stepcovnet/slot48_eval.py` | Shared ``M-slot48`` scoring, threshold sweep, null baseline, and audio-corruption helpers |
| `src/stepcovnet/beat_this_model.py` | Self-contained pure-PyTorch Beat This! neural beat/downbeat tracker (`foscarin2024beatthis`, MIT) |
| `src/stepcovnet/beat_tracking.py` | `BeatGrid`, Beat This! / `librosa` backends, offline fallback, transient snapping, ITGPT raw-audio inference |
| `src/stepcovnet/onset_events/` | Shared onset timing matching and F1 metrics (`charts.py`, `matching.py`, `metrics.py`) |
| `src/stepcovnet/dataset_prep/` | Raw simfile → `final_data` preprocessing, `training_index`, `training_loader` (P8–P9) |
| `src/stepcovnet/pairing.py` | Audio/chart pairing; `list_training_samples` for `final_data` |
| `src/stepcovnet/mel_onset.py` | Mel spectrogram helpers (shared by dense path; breaks import cycles) |
| `src/stepcovnet/wsl_gpu.py` | WSL GPU bootstrap / re-exec helpers |

See [PIPELINE_ARCHITECTURE.md](../research/PIPELINE_ARCHITECTURE.md) for the full mapping.

---

## Scripts and configs

| Path | Role |
| ---- | ---- |
| `scripts/` | CLI entry points (train, eval, extract, generate) |
| `scripts/preprocess_dataset.py` | Raw simfile packs → nested `data/final_data` |
| `scripts/build_training_index.py` | Train/val split manifest (`training_index.json`) |
| `configs/` | JSON experiment configs — see [configs/README.md](../../configs/README.md) |
| `configs/ddc/` | Placement baselines (`DDC`, `DDCL`, `ITGPT`) and ladder/full configs |

**Common entry points → skill:**

| Script | Skill |
| ------ | ----- |
| `scripts/train_itgpt_placement.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) · [colab-gpu-stepcovnet](../../.agents/skills/colab-gpu-stepcovnet/SKILL.md) |
| `scripts/train_ddcl_placement.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_ddc_placement.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_onset.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |
| `scripts/train_arrow.py` | [wsl-gpu-stepcovnet](../../.agents/skills/wsl-gpu-stepcovnet/SKILL.md) |

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
| `docs/research/` | Lab notebook (`EXPERIMENT_LOG.md`, `DISCUSSION_NOTES.md`, `PAPER_LEDGER.md`, `PIPELINE_ARCHITECTURE.md`) |
| `docs/research/archive/` | Superseded research plans |
| `docs/agents/project-layout.md` | Repo layout (this file) |
| `docs/agents/self-journal.md` | Agent process receipt journal (`JRN-…`) |
| `.agents/rules/` | Declarative rules (`model_decision` and file-scoped `glob` / `paths`) |
| `.agents/skills/` | Agent Skills playbooks (`SKILL.md`) + local `evals/suite.json` (gitignored; scored by `scripts/eval_agent_setup.py`) |
| `AGENTS.md` · `CLAUDE.md` (repo root) | Harness-neutral entry router (`CLAUDE.md` delegates via `@AGENTS.md`) |
