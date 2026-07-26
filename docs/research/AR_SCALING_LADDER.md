# AR scaling ladder

**Status:** proposed protocol — not yet run. **Created:** 2026-07-25.

How to grow AR onset training from one song to the full dataset so that each step produces a number that can be compared to the step before it.

**Related:** [experiment log](EXPERIMENT_LOG.md) · [AR onset design](AR_ONSET_DESIGN.md) · [onset metrics](ONSET_METRICS.md) · [discussion notes](DISCUSSION_NOTES.md) · [dataset prep](DATASET_PREP_PIPELINE.md)

---

## 1. Why this doc exists

The AR model is perfect on one song (tide: free-run **634/634**) and mediocre-to-broken on many songs. The results so far do not form a trend:

| Train size | Val teacher F1 | Source |
| ---------- | -------------- | ------ |
| 1 song | 1.0 (train song itself) | [EXP-20260630-01](EXPERIMENT_LOG.md#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion) |
| 10 songs | ~0.11 | [EXP-20260723-02](EXPERIMENT_LOG.md#exp-20260723-02-ar-corrected-mask-10-song-smoke-50-ep-in-memory-cache) |
| 50 rows | 0.126 @ 50 ep, 0.221 @ 466 ep | [EXP-20260724-01](EXPERIMENT_LOG.md#exp-20260724-01-ar-corrected-mask-50t50v-500-ep-scale-up) |
| 200 rows | 0.120 | [EXP-20260724-02](EXPERIMENT_LOG.md#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode) |
| 50 rows (local rebuild) | 0.0128 | [EXP-20260724-04](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) |

More data appears not to help, and one rung failed to reproduce entirely. **Both readings are premature**, because the rungs were never measured against the same yardstick. Section 2 shows why.

A ladder is the right strategy, but only if every rung is scored on the **same held-out songs** and selected by the **same rule**. Otherwise "50 songs beats 200 songs" is unfalsifiable.

## 2. Why the current rungs are not comparable

Three defects found on 2026-07-25 by inspection, before any new run:

| # | Defect | Evidence | Consequence |
| - | ------ | -------- | ----------- |
| D1 | **The val set changes when train size changes.** `build_training_index_subset` draws both splits from one `random.Random(seed)`, sampling train first. A different `train_rows` consumes a different amount of RNG state, so the val draw shifts even with identical `seed` and `--val-rows`. | `src/stepcovnet/dataset_prep/training_index.py` L619–620; reproduced with `train_rows` ∈ {10, 50, 200, 300} against a 1755/186 pool — four different val samples | Every rung above was scored on a **different** held-out set. Cross-rung F1 comparisons are meaningless, including the "200 is no better than 50" conclusion |
| D2 | **The source manifest is untracked and has drifted.** `data/final_data/training_index.json` is gitignored (`.gitignore:139`), and the copy on this machine has **1755/186** rows where [EXP-20260623-02](EXPERIMENT_LOG.md#exp-20260623-02-p8-trainval-manifest-on-full-final_data) and § Current phase record **1745/197** | on-disk `counts` + `created_at` 2026-07-03 | Even a fixed seed cannot reproduce an old subset, because the pool it was drawn from no longer exists. Contributes to [EXP-20260724-04](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) |
| D3 | **Checkpoint selection differs between rungs.** The 50-row run reported its best F1 at ep **466** with no early stopping; the rebuild early-stopped at ep **131** on `val_loss` patience 25 and restored ep **106** | [EXP-20260724-01](EXPERIMENT_LOG.md#exp-20260724-01-ar-corrected-mask-50t50v-500-ep-scale-up) vs [EXP-20260724-04](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) | Two rungs can differ purely by where training was cut, independent of data scale. `val_loss` and val F1 peak at very different epochs here (ep 65 vs ep 466) |

D1 is the one that invalidates the ladder outright. D3 additionally means `checkpoint_metric: val_loss` is selecting a checkpoint hundreds of epochs before the F1 peak.

## 3. Protocol

Fixed for every rung. A rung that deviates on any row is not on the ladder.

| Knob | Value | Why |
| ---- | ----- | --- |
| **Val set** | One frozen set, identical rows at every rung | The only way rung-to-rung deltas mean anything (fixes D1) |
| **Train sets** | **Nested**: R1 ⊂ R2 ⊂ R3 ⊂ … | A rung adds songs rather than resampling, so any change is attributable to the added data; also lets MERT extraction be incremental |
| **Source manifest** | Pinned by SHA-256, recorded in every subset summary and every `EXP-` entry | Makes a subset regenerable later (fixes D2) |
| **Epoch budget** | Same cap at every rung; early stopping on the **reported metric**, not `val_loss` | Removes "it just trained longer" as an explanation (fixes D3) |
| **Checkpoint selection** | Best val teacher F1, not best `val_loss` | These peak ~400 epochs apart on the 50-row rung |
| **Reported numbers** | teacher F1, free-run F1, predicted/GT onset-count ratio, EOS behavior, epochs-to-best, wall time | Free-run and length are where multi-song runs actually fail; a teacher-only number hides it |
| **Seed** | `apply_training_seed(42)` per [research-logging](../../.cursor/rules/research-logging.mdc) | — |

**Promotion rule:** climb to the next rung only when the current rung beats the previous rung's val teacher F1 **on the frozen val set**. If a rung fails to improve, that is the result — stop and investigate rather than adding more data on top of a broken rung.

### Manifest naming

| Rung | Manifest | Notes |
| ---- | -------- | ----- |
| R1–R5 | `data/final_data/training_index_ladder_{N}t_{V}v.json` | `ladder_` prefix distinguishes these from the existing non-nested `scoreboard_`/`{N}t_{V}v` subsets, which stay valid for their own logged runs |

`split_policy` carries a `ladder_v1` tag so a manifest self-identifies as ladder-built.

## 4. The ladder

| Rung | Train rows | Question it answers | Bar to climb |
| ---- | ---------- | ------------------- | ------------ |
| **R0** | 1 (tide) | Is the pipeline itself still healthy on this machine? | Free-run **634/634** vs `target_times` — the existing champion gate |
| **R1** | 10 | Does a multi-song run reach the ballpark of the logged 10-song smoke? | Val teacher F1 in the **~0.1** range ([EXP-20260723-02](EXPERIMENT_LOG.md#exp-20260723-02-ar-corrected-mask-10-song-smoke-50-ep-in-memory-cache)). **Not** an exact reproduction: that run used its own 10-song val set, this one uses the frozen 50-row set |
| **R2** | 50 | First real generalization signal | Beat R1 on the frozen val set |
| **R3** | 200 | Does an order of magnitude more data help? | Beat R2 |
| **R4** | 300 | MERT already extracted for this tier | Beat R3 |
| **R5** | 1755 (full) | Scoreboard | Beat R4 |

**R0 and R1 replace the standalone feature-integrity check.** If local MERT extraction were producing bad features, R0 would fail outright and R1 would land far below the logged range — the ladder tests it for free, on the way up, instead of as a separate errand.

Each rung is ~15 min to a few hours of WSL GPU, serialized by `logs/gpu_wsl.lock`.

### R0 outcome (2026-07-25)

**Partial** — [EXP-20260725-01](EXPERIMENT_LOG.md#exp-20260725-01-ladder-r0--mert-extraction-is-bit-identical-tide-champion-artifact-was-overwritten). Re-extracted tide features are **bit-for-bit identical** to the known-good copy, and the stored checkpoint decodes at **627/634** teacher with correct EOS placement, so the pipeline is healthy. The **634/634** bar itself is unverifiable: the graduated v8 champion at `models_wsl/ar/tide_overfit/ar_onset_model.keras` was overwritten by a later run on 2026-07-02, and `callbacks/.../best.keras` holds the same replacement weights. Restoring an exact R0 reference requires retraining tide from scratch.

This adds a fourth defect to § 2: **D4 — graduated artifacts are not write-protected.** Reusing `model_output_dir` across runs destroys the checkpoint a gate was graduated against.

### R1 outcome (2026-07-25)

**Supported** — [EXP-20260725-02](EXPERIMENT_LOG.md#exp-20260725-02-ladder-r1--first-rung-on-the-frozen-val-set-10-train-rows). Val Hungarian F1 **0.178** at ep 497 of 500 (~41 min). The ladder now has a real baseline.

Two things the rung taught beyond its own number:

- **D3 quantified.** Best `val_loss` lands at ep **104**, where F1 is **0.0055** — **32×** below the F1-selected checkpoint. Every earlier scale-up selected on `val_loss`, so those checkpoints were chosen near the worst possible point for timing skill.
- **D5 — monitors were silently inert.** `resolve_checkpoint_metric` returns legacy metric names, but the AR trainer publishes canonical ones outside overfit runs, so `ModelCheckpoint` and `EarlyStopping` matched nothing and no `best.keras` was written. Fixed with `MetricAliasCallback` in `src/stepcovnet/onset_ar/trainers.py`, covered by a test. R1's own number is unaffected because F1 peaked three epochs before the budget ended.

**Open:** F1 was still rising at ep 500, so 0.178 is a lower bound, and a fixed epoch budget gives larger rungs more gradient steps (10 rows × 500 ep = 5k steps; 50 rows × 500 ep = 25k). Decide before R3 whether the budget should be counted in steps.

## 5. Work required before R1 — **done 2026-07-25**

| # | Change | Status |
| - | ------ | ------ |
| W1 | Fix D1 in `build_training_index_subset`: independent generator per split, shuffle-then-take so train sets nest | **Done** — `_nested_sample` seeded from `{seed}:{split}`; `policy_tag` now defaults to `ladder_v1`. Locked by three tests in `tests/dataset_prep/training_index_test.py` |
| W2 | Record source-manifest SHA-256 in the subset manifest and CLI summary | **Done** — optional `source_sha256` field on `TrainingIndex`; emitted only when set, so existing manifests are unchanged |
| W3 | Ladder configs with unified epoch budget and F1-based checkpoint selection | **Done** — `configs/ar/ladder_{10,50,200,300}t_50v.json`, 500 ep, ES patience 50 |
| W4 | Confirm `checkpoint_metric` supports the aux F1 metric | **Done, no code change** — `resolve_checkpoint_metric("val_aux_f1_hungarian")` → `val_event_onset_f1`, and monitor mode is `max` for non-loss metrics |

**Verified on the real manifest:** the four rungs share an identical 50-row / 42-song val set, train rows nest strictly (10 ⊂ 50 ⊂ 200 ⊂ 300), and train/val audio overlap is **0**. Source pinned at `1fac516f06fe69b6…`.

W1 changes the output of an existing script. Old subsets stay reproducible only via the old policy tag, and D2 means the pre-July-25 ones are not reproducible regardless.

## 6. Open items

- **Should manifests be tracked in git?** They are gitignored today (D2). A subset manifest is ~46 KB; the full one is ~900 KB. Tracking the ladder subsets would make rungs reproducible across machines outright.
- **Scheduled sampling** ([NOTE-20260724-02](DISCUSSION_NOTES.md#note-20260724-02-hypotheses-eliminated-for-multi-song-free-run-under-generation)) stays deferred until at least R2 gives a trustworthy baseline to measure it against.
- **Dense track reuse:** the same frozen val set should be used for the dense scoreboard track so AR and dense are directly comparable.
- **Free-run bar:** no rung bar is set for free-run F1 yet, because no multi-song run has produced a healthy one. Set it once R2 lands.
