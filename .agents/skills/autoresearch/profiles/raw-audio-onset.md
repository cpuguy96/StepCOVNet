# Profile: raw-audio-onset

Fast CPU / local-GPU RSI loop for **end-to-end raw-audio beat tracking & onset decoding** (`src/stepcovnet/beat_tracking.py`, `src/stepcovnet/beat_this_model.py`, `scripts/eval_raw_audio_onset.py`) — zero Colab VM startup required, seconds per iteration.

## Success criteria & pinned gates

- **Primary metrics:** `beat_this_full` pooled ±20 ms `evF1` (ratio to `oracle` upper bound) and `BeatF1@70ms` across validation songs; check `beat_this_reg` and offline `librosa` as secondary diagnostics.
- **Mandatory guardrails (no exceptions):**
  1. **Exact steady-BPM non-regression (`+0.0000`):** Steady-BPM validation tracks (e.g. `deal_with_it`, `trap`, `spirit`, `stay_awake`) must stay bit-for-bit identical unless fixing a proven bug on them.
  2. **Multi-model transfer:** Verify improvements transfer across at least 2 active ITGPT checkpoints (e.g. fine-tuned `Arm C`, Zero-Shot `1,713t`, and `Soup_F_H_I`).
  3. **Unit & CI gate green:** `pytest tests/beat_tracking_test.py` + `python pre_submit.py --fast` pass before declaring a win.

## Preflight & Orient (~5 s)

1. Read [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) § Current phase and recent raw-audio `EXP-…` runs.
2. Run [eval_raw_audio_onset.py](../../../../scripts/eval_raw_audio_onset.py) on the active baseline checkpoint (reuse existing JSON on disk if unchanged).
3. Inspect the **worst failing songs and failure modes** (`kokeshi_neko_medley`, `west`, `goin_under`, `isis`, `right_osiris`: BPM drift, octave `0.5×/2×` flicker, `#BPMS` segmentation errors, bar-phase modulo-4 alignment, attack-snap phase offset).

## Iteration loop (Headroom → Mechanism → Test → Multi-Model Check)

1. **Headroom & mechanism proof:** Measure the per-song gap vs `oracle` on the worst failing song(s) and inspect the beat grid / onset curve directly on CPU before coding heuristics.
2. **Implement cleanly:** Keep beat-tracking improvements general (piecewise lattice fitting, phase alignment, transient snapping) — **never** overfit song-specific constants. Add regression unit tests in `tests/beat_tracking_test.py`.
3. **Verify parity + transfer:**
   ```bash
   python scripts/eval_raw_audio_onset.py \
     --training_index_path data/literature_fraxtil_orig/training_index_standard.json \
     --model_path <checkpoint_path> \
     --split val \
     --output logs/eval_raw_audio_<tag>.json
   ```
   Confirm steady-BPM songs remain `+0.0000`, worst medley/outlier songs improve, and `pytest tests/beat_tracking_test.py` passes.
4. **Log same turn:** Prepend `EXP-…` in [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) via [research-session-workflow](../../research-session-workflow/SKILL.md).
5. **Outer meta-loop every 3–5 iterations:** Synthesize structural causal beliefs in `DISCUSSION_NOTES.md`, add missing diagnostics, and promote reusable workflow lessons via [agent-maintenance](../../agent-maintenance/SKILL.md).
