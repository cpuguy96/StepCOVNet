---
name: onset-evaluation
description: Evaluates StepCOVNet onset and placement checkpoints (ITGPT, DDCL, DDC, raw-audio Beat This! vs Oracle, dense onset), runs grounding and error-breakdown diagnostics, and ranks runs via itgpt_scoreboard.py. Use when evaluating models, inspecting worst failing songs/slices, comparing beat trackers, or checking scoreboards.
---

# Onset and placement evaluation

Canonical metric definitions live in [ONSET_METRICS.md](../../../docs/research/ONSET_METRICS.md). Follow [python-env.md](../../rules/python-env.md) for Python executables and [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md) if running GPU eval locally (serial GPU lock).

## Track → script & metric map

Never mix incompatible metrics across tracks in one comparison table:

| Track / scope | Script | Primary metrics & diagnostics |
| ------------- | ------ | ----------------------------- |
| **ITGPT placement** | `scripts/eval_itgpt_placement.py` | `M-slot48` `F1@0.5`, `F1_max`, `--diagnose_grounding`, `--error_breakdown`, `--diagnose_decoding` |
| **Raw-audio end-to-end** (Beat This! vs Oracle GT beat grids) | `scripts/eval_raw_audio_onset.py` | ±20 ms event F1 across `oracle`, `beat_this_raw`, `beat_this_reg`, `beat_this_full`, `librosa` |
| **DDCL placement** | `scripts/eval_ddcl_placement.py` | `M-slot48` F1, `--diagnose_grounding`, `--diagnose_timing` |
| **DDC placement** | `scripts/eval_ddc_placement.py` | `M-ddc-20ms` Hamming ±20 ms `F-score_c` / `F-score_m`, optional `--slot48` |
| **Dense frame onset** | `scripts/eval_dense_onset.py` | Peak-pick ±20 ms event F1, `--corruption` |
| **Scoreboard ranking** | `scripts/itgpt_scoreboard.py` | Ranks `experiments/colab/` and `models_wsl/ddc/` runs by `f1_at_05` + guardrails |

## Standard diagnostics

### ITGPT grounding + error breakdown (single pass)

`--diagnose_grounding --error_breakdown` reuses the uncorrupted forward pass for both grounding drops and `eval_val_breakdown.json` (zero redundant inference):

```bash
python scripts/eval_itgpt_placement.py \
  --config configs/ddc/<config>.json \
  --model_path <run_dir>/best.pt \
  --output <run_dir>/eval_val_grounding_best.json \
  --diagnose_grounding \
  --error_breakdown
```

Add `--diagnose_decoding` to compare zero-training decoding strategies (`fixed 0.5`, top-K density matching, clamped top-K, LOSO subdivision thresholds).

**Mandatory ITGPT champion guardrails:**

- Audio grounding (shuffle): `slot_f1_drop_shuffle >= 0.08`
- Audio grounding (zeros): `slot_f1_drop_zeros >= 0.10`
- Slice audit (`eval_val_breakdown.json`): no unmotivated regression on `intro`, `16th_off`, `24th`, or worst charts

### Raw-audio end-to-end eval (`eval_raw_audio_onset.py`)

```bash
python scripts/eval_raw_audio_onset.py \
  --training_index_path data/literature_fraxtil_orig/training_index_standard.json \
  --model_path <run_dir>/best_slot_f1.pt \
  --split val \
  --output <run_dir>/eval_raw_audio.json
```

Compare `oracle` upper bound against `beat_this_raw`, `beat_this_reg`, `beat_this_full`, and `librosa`.

### Scoreboard check

```bash
python scripts/itgpt_scoreboard.py --filter fraxtil_orig
python scripts/itgpt_scoreboard.py --filter 300t_50v
```

## Hygiene and interpretation rules

- **Check disk before re-running:** Inspect [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md), `experiments/colab/`, and `logs/` first — if an eval JSON already exists for that checkpoint, read it rather than re-running.
- **Inspect worst slices before claiming saturation:** Never call a stage "saturated" or "at diminishing returns" without auditing worst failing songs/slices in `eval_val_breakdown.json` or per-song raw-audio outputs.
- **Leave frozen baselines alone:** Do not add extra diagnostic columns to an accepted/frozen baseline checkpoint unless explicitly asked.
- **Log same turn:** Prepend/update `EXP-…` after measurable eval runs via [research-session-workflow](../research-session-workflow/SKILL.md).
