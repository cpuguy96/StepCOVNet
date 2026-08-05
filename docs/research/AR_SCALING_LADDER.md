# AR scaling ladder

**Status:** **halted 2026-08-04 — blocked on a model defect, not on data.** The pointer head never reads the audio: zeroing every feature moves R2 val F1 **0.1886 → 0.1885**, and the tide champion holds **0.9984** with the audio reversed ([EXP-20260804-05](EXPERIMENT_LOG.md#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer) · [NOTE-20260804-03](DISCUSSION_NOTES.md#note-20260804-03-the-pointer-head-is-absolute-index-classification-so-the-model-never-had-to-hear-the-audio)). That is why every rung sat at or below an audio-blind baseline ([EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke) · [NOTE-20260804-02](DISCUSSION_NOTES.md#note-20260804-02-the-ladder-was-climbing-a-chance-floor--scale-was-never-the-binding-problem)). Do not climb R4. Fix `pointer_logits` (§ 2, D7) first; the protocol below is otherwise sound. **Created:** 2026-07-25.

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

**Resolved 2026-08-04:** neither reading was right. Once the rungs *were* put on one yardstick (§ 3–4), they still failed to beat an audio-blind baseline at matched onset count, so "more data does not help" was never falsifiable — there was no skill for data to improve. The cause is D7 below: the model does not read the audio, so the 1.0 in the first row and the 0.12–0.22 in the rest measure the same audio-blind position prior at different amounts of memorization. See § *Ladder halted*.

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
| **Checkpoint selection** | Best **`val_timing_match_teacher`** | Hungarian F1 has a **0.225–0.336** chance floor here and never exceeded it, so selecting on it selected noise (fixes D6). The ordered metric's floor is ≈ 0 and it is now published on multi-song runs |
| **Chance floor** | Every reported F1 carries the audio-blind floor at **its own** `pred/GT` ratio, and skill over the strongest null | Without it, a count-matching change is indistinguishable from a timing improvement (fixes D6) |
| **Reported numbers** | teacher and free-run: F1 **plus null floor plus skill**, ordered `timing_match`, predicted/GT onset-count ratio, **train-split F1** beside val, EOS behavior, epochs-to-best, wall time | Free-run and length are where multi-song runs appear to fail; train-beside-val is what distinguishes a generalization failure from a scale problem |
| **Difficulty mix** | **Not controlled** — rows are sampled without a difficulty filter, so the challenge share drifts across rungs (80% → 60% from R1 to R4) and multi-chart songs give one audio several targets. Audio-only val ceiling is **0.957**, so this is not the current cap; see [NOTE-20260803-01](DISCUSSION_NOTES.md#note-20260803-01-difficulty-is-unconditioned-and-unfiltered-but-it-is-not-what-caps-the-ladder) |
| **Initialization** | Random init. `init_model_path` **not used** on any ladder run | Same rule the tide harness enforces by stripping the key (`scripts/ar_tide_iter/config_builder.py`): warm-started numbers are not valid bars. The checkpoint path is also mutable — rerun a rung and a warm-started child's starting point silently changes |
| **Seed** | `apply_training_seed(42)` per [research-logging](../../.cursor/rules/research-logging.mdc) | — |

**Promotion rule:** a rung counts only when its val score shows **positive skill over the strongest audio-blind null** at its own prediction count; only then does beating the previous rung mean anything. Beating the previous rung while both sit under the floor is not a result. If a rung fails either test, stop and investigate rather than adding more data on top of a broken rung.

**D6 — the ladder had no chance floor.** Rungs were compared, and checkpoints selected, on Hungarian F1 @ 20 ms. On this val set (5.52 onsets/sec, mean IOI 181 ms) that metric scores **0.225** for uniform random guesses, **0.275** for a metronome, and **0.336** for shuffled ground-truth intervals at matched count — above every value any rung ever reached. Measured by `stepcovnet.onset_null_baseline`; reported automatically by `eval_ar_onset_offline.py`.

Confirmed end-to-end on the best rung: re-scoring R2 through the updated harness gives teacher F1 **0.1886** against a metronome null of **0.2696** (skill **−0.1109**) and free-run skill **−0.0150**, with both gates failing and `n_patch_wrong` at **99.4%** ([EXP-20260804-04](EXPERIMENT_LOG.md#exp-20260804-04-in-harness-null-floor-reproduces-the-finding-on-the-r2-checkpoint--both-gates-fail)).

**D7 — the model does not read the audio, and no gate could see it.** `pointer_logits = Dense(max_patches)(decoder)` (`models.py:358`) scores **absolute patch indices**, never encoder content. Corrupting only `mert_patches` leaves R2 teacher F1 at **0.1885** (zeros) vs **0.1886** (real), `patch_wrong` at **0.9942** in every variant, pointer NLL pinned at **16.88** nats, and the argmax patch unchanged on **99.96%** of steps. The tide champion scores an identical **0.9984** with audio reversed, shuffled, or taken from another song. A single-song overfit is fully determined by the teacher-forced prefix, so it cannot detect this — which is why the gate passed for months. Measured by `scripts/audio_ablation_ar_onset.py` ([EXP-20260804-05](EXPERIMENT_LOG.md#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer)).

D7 subsumes D6: the reason no rung cleared the chance floor is that the model *is* an audio-blind predictor. **Ladder gate (standing):** every rung must pass `scripts/audio_ablation_ar_onset.py --gate` — pointer **and** token scores must collapse under `shuffle`/`zeros` or the rung does not count at any scale.

**D7 fixed 2026-08-04.** `model.pointer_head: content` (new default) scores decoder queries against encoder patch keys instead of absolute indices. Tide passes at **0.9921** on real audio and drops to **0.18** under shuffled audio ([EXP-20260804-06](EXPERIMENT_LOG.md#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate)). **Rungs must be re-run on the new head** — every number in § 1 and § 4 was produced by the audio-blind `index` head and none of them carry over.

### Manifest naming

| Rung | Manifest | Notes |
| ---- | -------- | ----- |
| R1–R5 | `data/final_data/training_index_ladder_{N}t_{V}v.json` | `ladder_` prefix distinguishes these from the existing non-nested `scoreboard_`/`{N}t_{V}v` subsets, which stay valid for their own logged runs |

`split_policy` carries a `ladder_v1` tag so a manifest self-identifies as ladder-built.

### Run artifacts

Every ladder run — rungs and variants alike — shares `"callback_root_dir": "callbacks/ar/ladder"` and sets `"run_label"` (`r1_10t`, `r2_50t`, `r3_200t`, `r2_ss_50t`, …). Runs land in `callbacks/ar/ladder/logs/{timestamp}-AR_ONSET-{run_label}-…`, so one command shows the whole ladder in chronological order:

```powershell
& venv\Scripts\tensorboard.exe --logdir callbacks/ar/ladder/logs --port 6006
```

`model_output_dir` stays **per variant** (`models_wsl/ar/ladder_{N}t_{V}v[...]`) — it writes a fixed `ar_onset_model.keras` that a shared directory would overwrite.

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

### R2 attempt 1 (2026-07-26) — aborted

**Aborted at ep 152**, unscored — [EXP-20260726-01](EXPERIMENT_LOG.md#exp-20260726-01-ladder-r2-aborted-at-ep-152--wsl-vm-terminated-mid-run). The WSL VM shut down mid-run at 01:29; the partial curve was ~**2.2×** ahead of R1 at matched epochs but must **not** be reported as R2's number. Suspected in-guest memory exhaustion (15 GB WSL ceiling, ~5.5 GB of cached features), unconfirmed because the kernel log was wiped by the VM restart.

### R2 complete (2026-08-02)

**Supported** — [EXP-20260802-01](EXPERIMENT_LOG.md#exp-20260802-01-ladder-r2--50-train-rows-beats-r1-on-frozen-val). Val Hungarian F1 **0.2266 @ ep 470** of 500 (~**1.27×** R1's **0.178**). Rerun used `.wslconfig` `memory=24GB` (~23 GiB guest); held through full budget with mid-run guest use ~7–8 GiB. D3 again: best `val_loss` @ ep **42** has F1 only **0.120**.

### R3 complete (2026-08-02)

**Partial** — [EXP-20260802-02](EXPERIMENT_LOG.md#exp-20260802-02-ladder-r3--200-train-rows-does-not-beat-r2). Val Hungarian F1 **0.1991 @ ep 361**, ES @ ep **411**. **Below** R2 (**0.227**); still above R1 (**0.178**). Teacher F1 is not monotonic in train size on the frozen val set.

### R2 free-run (2026-08-02)

**Partial** — [EXP-20260802-03](EXPERIMENT_LOG.md#exp-20260802-03-ladder-r2-offline-val-free-run--eos_trace). Teacher F1 **0.227** holds offline; free-run F1 **0.132**; **pred/GT = 0.36**. All 50 val songs stop on `<EOS>` at exactly **252** onset tokens (`eos_trace` final ~**1.0**). Free-run bar for this rung: **0.132**.

### R2 vs R3 free-run (2026-08-02)

**Supported** (compare) — [EXP-20260802-04](EXPERIMENT_LOG.md#exp-20260802-04-ladder-r2-vs-r3-offline-val-free-run-compare). R3 free-run F1 **0.003**, **pred/GT = 0.021**, every song EOS at **15** onsets — much worse than R2, and a different failure mode (early EOS vs fixed-length **252**).

### R2 + scheduled sampling (2026-08-03)

**Not supported** — [EXP-20260803-01](EXPERIMENT_LOG.md#exp-20260803-01-scheduled-sampling-now-actually-running-does-not-improve-free-run-on-r2). First run with the feature actually live (it had been compiled out of the traced `train_step`, [EXP-20260802-05](EXPERIMENT_LOG.md#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step)). Teacher F1 **0.2235** vs R2 **0.2266**; free-run F1 **0.1313** vs the **0.132** bar; `pred/GT` **0.3555** and all 50 songs still stop at exactly **252** onsets — the length pathology is untouched. Note the run is not seed-exact against R2: `tf.cond` adds random ops that shift dropout seeds, worth ±0.007 on teacher F1 during warmup.

**Supported** — [EXP-20260803-02](EXPERIMENT_LOG.md#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run). **Meter** density conditioning on the R2 recipe: teacher F1 **0.227** (holds); free-run F1 **0.234** vs the **0.132** bar; `pred/GT` **0.82**; stop lengths vary (**6** modes, only **8/50** @ **252**). Code: `model.density_conditioning` (`none` | `meter` | `onset_density`); config [`ladder_50t_50v_density.json`](../../configs/ar/ladder_50t_50v_density.json).

### Ladder halted (2026-08-04)

**All rungs void as measurements** — [EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke). At matched `pred/GT`, audio-blind nulls score **0.225–0.336** (r=1.0) and **0.217–0.313** (r=0.90). Teacher F1 topped out at **0.227** and free-run at **0.263**, so nothing on the ladder cleared chance. Ordered `timing_match` (floor ≈ 0) reads **0.0026**, and val `pointer_loss` is **15.6–18.6** nats against **6.7** for uniform guessing — the pointer is confidently wrong on held-out audio.

The density-conditioning result is **retracted as a win**: free-run F1 tracked `pred/GT` along the null curve (0.36→0.132, 0.82→0.234, 0.90→0.263 vs nulls 0.154/0.250/0.261). There is no generation champion.

Train vs val at each best epoch — R2 **0.746/0.227**, R3 **0.529/0.199**, R2+density **0.736/0.227**, R3+density **0.539/0.206**, train token accuracy **0.98** vs val **0.37** — makes this a **generalization** failure, not a data-scale one. More rows cannot produce a trend on a recipe with zero transfer.

**Next:** re-score existing checkpoints under the updated harness; **done 2026-08-04:** ladder configs use `checkpoint_metric: val_timing_match_teacher`; standing audio-grounding gate is `scripts/audio_ablation_ar_onset.py --gate`. Then work generalization. **Hold** R4–R5, EOS weighting, and further density variants.

## 5. Work required before R1 — **done 2026-07-25**

| # | Change | Status |
| - | ------ | ------ |
| W1 | Fix D1 in `build_training_index_subset`: independent generator per split, shuffle-then-take so train sets nest | **Done** — `_nested_sample` seeded from `{seed}:{split}`; `policy_tag` now defaults to `ladder_v1`. Locked by three tests in `tests/dataset_prep/training_index_test.py` |
| W2 | Record source-manifest SHA-256 in the subset manifest and CLI summary | **Done** — optional `source_sha256` field on `TrainingIndex`; emitted only when set, so existing manifests are unchanged |
| W3 | Ladder configs with unified epoch budget and timing-match checkpoint selection | **Done** — `configs/ar/ladder_{10,50,200,300}t_50v.json`, 500 ep, ES patience 50, `checkpoint_metric: val_timing_match_teacher` |
| W4 | Confirm `checkpoint_metric` supports the aux F1 metric | **Done, no code change** — `resolve_checkpoint_metric("val_aux_f1_hungarian")` → `val_event_onset_f1`, and monitor mode is `max` for non-loss metrics |

**Verified on the real manifest:** the four rungs share an identical 50-row / 42-song val set, train rows nest strictly (10 ⊂ 50 ⊂ 200 ⊂ 300), and train/val audio overlap is **0**. Source pinned at `1fac516f06fe69b6…`.

W1 changes the output of an existing script. Old subsets stay reproducible only via the old policy tag, and D2 means the pre-July-25 ones are not reproducible regardless.

## 6. Open items

- **Should manifests be tracked in git?** They are gitignored today (D2). A subset manifest is ~46 KB; the full one is ~900 KB. Tracking the ladder subsets would make rungs reproducible across machines outright.
- **Scheduled sampling** ([NOTE-20260724-02](DISCUSSION_NOTES.md#note-20260724-02-hypotheses-eliminated-for-multi-song-free-run-under-generation)) — **unblocked on R2.** R3 free-run is worse (**0.003**, early EOS @ **15**) — do not SS there first ([EXP-20260802-04](EXPERIMENT_LOG.md#exp-20260802-04-ladder-r2-vs-r3-offline-val-free-run-compare)).
- **Dense track reuse:** the same frozen val set should be used for the dense scoreboard track so AR and dense are directly comparable. The dense **0.686** on `data/v2` also needs its own null floor before it is trusted as a target.
- **Free-run bar:** withdrawn. R2 **0.132**, R3 **0.003**, R2+density **0.263** are all at or below the audio-blind floor at their own prediction counts ([EXP-20260804-03](EXPERIMENT_LOG.md#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)). The next bar is **skill > 0**, not an F1 value.
