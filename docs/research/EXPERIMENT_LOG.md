# Experiment log

**Authoritative record** for runs and ablations. IDs: `EXP-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

Promote selected findings to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) only when drafting the paper — do not duplicate the full log there.

**Related:** [discussion notes](DISCUSSION_NOTES.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [dataset prep plan](DATASET_PREP_PIPELINE.md) · [AR onset design](AR_ONSET_DESIGN.md) · [decisions checklist](DECISIONS_CHECKLIST.md)

---

## Current phase

**Updated:** 2026-07-24

### Dataset prep (PRE ingestion)

| Phase | Status |
| ----- | ------ |
| P0–P9 | **Done** — **1942** chart rows; `training_index.json` (`stratified_song_v1`: **1010** / **110** songs, **1745** / **197** chart rows train/val) |
| MERT subset | **Done** for scale-up — `training_index_300t_50v.json` (314 unique audio); `training_index_200t_50v.json` / `50t_50v` subsets |

**Recommended next step (Track A — scoreboard):** First full multi-song **dense** training on `final_data` via `--training_index_path=data/final_data/training_index.json` (WSL GPU). Extract MERT features for `final_data` if not using mel baseline; then `eval_dense_onset.py` + threshold sweep on val.

### Onset detection (research track)

| Item | Status |
| ---- | ------ |
| Dense val best (`data/v2`) | BiLSTM 256u — micro event F1 **0.686** @ thr=0.30 (EXP-20260610-03) |
| Event tide formulation (`data/v2`) | ~27–30% F1 plateau; oracle ~31% (EXP-20260606-11) — formulation ceiling for K-query slots |
| `final_data` training hookup | **Done** — dense + event trainers accept `--training_index_path`; 10-song CPU smoke **10/10** batches (EXP-20260624-01/02) |
| Multi-song val on `final_data` | **Unblocked** — awaiting first full GPU dense train + eval |
| **AR tide perfect overfit** | **PASS** — scratch **iter175** / champion **v8**: teacher + free-run **634/634** ordered @ 20 ms vs **`target_times`** ([EXP-20260630-01](#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)) |
| **AR 10-song smoke** | **PASS** — 5-ep corrected-mask ([EXP-20260723-01](#exp-20260723-01-ar-corrected-mask-10song-smoke)); **50-ep cached** `val_loss` **35.0 → 12.1**, teacher F1 **0.11** ([EXP-20260723-02](#exp-20260723-02-ar-corrected-mask-10song-smoke-50ep)) |
| **AR 50t/50v scale-up** | **Partial** — 500 ep: best `val_loss` **~20.9 @ ep 65**, then severe overfit; val F1 peaks **~0.22** ([EXP-20260724-01](#exp-20260724-01-ar-corrected-mask-50t50v-500-ep-scale-up)) |
| **AR 200t/50v scale-up** | **Partial** — ES @ ep **65**, best `val_loss` **~12.7 @ ep 40**; offline val teacher F1 **0.120**, free-run F1 **0.036** (severe under-gen) ([EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode)) |
| **AR corrected-mask regression gate** | **Partial** — run1 + run2 both teacher + free-run **633/634**; free-run tracks teacher; short of perfect bar ([EXP-20260716-02](#exp-20260716-02-ar-corrected-mask-tide-overfit-regression)) |
| **AR free-run length diagnostics** | **Ready** — `eos_trace` + `--ar_decode_min_onset_tokens` / `--ar_decode_eos_logit_bias` land offline decode probes; healthy tide reference: EOS mean **0.0017**, single spike at step **634** ([EXP-20260724-03](#exp-20260724-03-ar-decode-length-control--eos-trace-diagnostics)) |
| **AR next gate** | Free-run / length control on multi-song before **`gate-val-vs-dense`**; optional denser train (300t cache-safe) still open |
| **Local artifact gap** | July 16–24 AR checkpoints, subset indices, and logs are **absent from this clone** — EXP-20260724-01/02 cannot be re-measured here without rebuilding indices + MERT and retraining |
| **AR tide class weights (champion recipe)** | **Deferred** — drop-in on v8 failed free-run ([EXP-20260703-01](#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe)); champion stays `none`; co-tuned recipe revisit [NOTE-20260703-01](DISCUSSION_NOTES.md#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred) |
| **AR training throughput / validation** | **Improved** — val aggregation + dynamic buckets (**18.6%** on smoke); single-song overfit batch cache default-on (~**9×** steady epoch on tide) ([EXP-20260716-01](#exp-20260716-01-ar-validation-aggregation--dynamic-length-bucketing), [EXP-20260716-02](#exp-20260716-02-ar-corrected-mask-tide-overfit-regression)) |

**Recommended when resuming onset work:**

- **Track A (scoreboard):** Full `final_data` dense MERT (or mel) train/val; compare to `data/v2` session best (0.686).
- **Track B (AR scale-up):** 200t teacher Hungarian F1 matches train val (~**0.12**), but free-run **collapses** (early EOS / ~**70** preds/song vs ~**700** GT). Source review eliminates chart truncation, audio/target duration mismatch, and EOS over-representation; **exposure bias** (`scheduled_sampling_max_p: 0.0`) remains the live cause ([NOTE-20260724-02](DISCUSSION_NOTES.md#note-20260724-02-hypotheses-eliminated-for-multi-song-free-run-under-generation)). Length control is now measurable offline, but on tide it trades FN for **1479** FP — treat it as a probe, not a fix. Next: rebuild a multi-song checkpoint locally, read its `eos_trace`, then test **scheduled sampling** at train time.
- **Event track (optional):** Continue K-query probes on `data/v2` in parallel if not blocking Track A.

---

## Experiment index

Newest first. Stage tags: `pre` | `model` | `post` | `metric` | `train`. Discussion context: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

| ID | Stage tag | Question | Status | One-line outcome |
| -- | --------- | -------- | ------ | ---------------- |
| EXP-20260724-03 | `post` + `metric` | Can free-run length be probed offline, and what does healthy `<EOS>` look like? | **Supported** (tooling) | `eos_trace` + `ArLengthControl` added; tide EOS flat ~**0.0017**, spikes only at step **634**; forcing length yields **1479** FP |
| EXP-20260724-02 | `train` + `metric` | Does 200t/50v + ES improve val, and does offline free-run hold? | **Partial** | Best `val_loss` **12.7 @ ep 40**; offline teacher F1 **0.120**, free-run F1 **0.036** |
| EXP-20260724-01 | `train` + `metric` | Does corrected-mask AR on 50t/50v keep improving through 500 ep? | **Partial** | Best `val_loss` **20.9 @ ep 65**; ep 500 overfits (train **0.06** / val **33.6**); val F1 **~0.22** |
| EXP-20260723-02 | `train` + `metric` | Does corrected-mask 10-song smoke keep improving to 50 ep with in-memory cache? | **Supported** | `val_loss` **35.0 → 12.1**; teacher F1 **0.01 → 0.11**; ~**2 s**/ep with cache |
| EXP-20260723-01 | `train` + `metric` | Does corrected-mask + dynamic-pad 10-song smoke still pass the gate? | **Supported** | **10/10** steps; `val_loss` **35.0 → 26.9**; teacher F1 > 0 |
| EXP-20260716-02 | `train` + `metric` | Does corrected-mask champion recipe still hit tide **634/634**? | **Partial** | Run1+run2 teacher+free-run **633/634**; free-run tracks teacher; `λ_inc=0`; cache ~**9×** |
| EXP-20260716-01 | `pre` + `model` + `metric` + `train` | Do correct val aggregation and dynamic length buckets improve AR smoke training? | **Supported** | Val now covers both batches; matched steady epoch **21.5 s → 17.5 s** (**18.6% faster**); found/fixed inverted attention-mask semantics for new models |
| EXP-20260703-01 | `train` + `metric` | AR tide token class weight ablation on champion v8 recipe | **Partial** | Drop-in on v8: teacher **634/634**, free-run **≤360/634**; co-tuned recipe revisit later |
| EXP-20260630-03 | `pre` + `train` | AR tide MERT normalization A/B (`normalize_mert_features`) | **Supported** | Raw wins perfect gate **1.0** @ ep 399; norm plateaus **0.9984** — keep raw |
| EXP-20260630-02 | `pre` + `train` | AR **`gate-10song-smoke`** (`training_index_path`) | **Supported** | **10/10** train + **2/2** val batches; `val_loss` **53.4 → 38.7** @ 5 ep GPU; teacher F1 > 0 |
| EXP-20260630-01 | `train` + `metric` | AR tide **scratch** perfect overfit (iter175 → v8 champion) | **Supported** | Free-run **634/634** vs `target_times`; graduated [`tide_overfit.json`](../../configs/ar/tide_overfit.json) |
| EXP-20260628-02 | `train` + `metric` | AR tide **perfect overfit** (warm-start runs) | **Partial** | Best warm-start free-run **619/634** (run2); superseded by scratch iter175 — see EXP-20260630-01 |
| EXP-20260628-01 | `train` + `model` | AR `gate-ar-decode` v2–v4 (SS ramp, warm-start) | **Supported** | v4: teacher F1 **1.0**; offline AR F1 **~0.35**; tide pass via scratch path (EXP-20260630-01) |
| EXP-20260627-04 | `train` + `model` | AR `gate-tide-overfit` pass (residual MSE + λ ramp) | **Supported** | `val_event_onset_f1` **1.0** @ ep 180+; debug 634/634 within 20 ms |
| EXP-20260627-03 | `train` + `model` | AR tide overfit training fixes (class weights, argmax decode, λ ramp) | **Supported** | F1 **~0.83** (`gate_v4`); residual head untrained — see NOTE-20260627-02 |
| EXP-20260627-02 | `train` + `model` | AR `gate-tide-overfit` WSL 300ep (tide) | **Fail** | Best `val_event_onset_f1` **~0.14**; `val_token_accuracy` **~0.48** plateau; gate not passed |
| EXP-20260627-01 | `pre` + `model` | AR Phase 0+1 scaffold + tide verify | **Supported** | `onset_ar/` + `train_onset_ar.py`; tide batch loads (634 onsets, vocab 339) |
| EXP-20260624-01 | `pre` + `train` | 10-song dense smoke via `training_index_path` | **Supported** | **10/10** train batches, 2 ep CPU; manifest-as-pointer OK |
| EXP-20260624-02 | `pre` + `train` | 10-song event smoke @ 2048 caps | **Supported** | **10/10** train batches after `n_max_onsets` / `num_queries` / `max_steps_per_chart` = 2048 |
| EXP-20260623-02 | `pre` | P8 `training_index.json` on full `final_data` | **Supported** | `stratified_song_v1`; 1010/110 songs; 1745/197 rows |
| EXP-20260622-01 | `pre` + loader | P9 smoke on local `final_data` | **Supported** | 1942 chart rows; multi-`chart_index` OK; 0 missing pairs |
| EXP-20260610-03 | `model` + `train` + `post` | BiLSTM 256u 50-train 200ep scale-up | **Supported** | **0.686** @ 0.30 — session best (+0.3 pp vs U-Net EXP-12) |
| EXP-20260610-02 | `model` + `post` | BiLSTM/TCN round-2 follow-ups | **Supported** | BiLSTM 256u **0.680** @ 0.25; TCN blocks=6 rejected (0.655) |
| EXP-20260610-01 | `model` + `post` | Onset backbone smoke (9 configs) | **Supported** | BiLSTM **0.677** @ 0.15; TCN **0.664**; transformer OOM; U-Net ~0.647 |
| EXP-20260609-12 | `train` + `post` | Gaussian 50-train 200ep no-ES | **Supported** | **0.683** @ 0.35 on mid-train ckpt — session best (+0.9 pp) |
| EXP-20260609-11 | `post` | Gaussian 100-train callback sweep | **Supported** | Post-hoc best **0.671** @ 0.30 — frame-F1 peak ckpt only 0.654 |
| EXP-20260609-10 | `train` | Gaussian 100-train 200ep no-ES | **Supported** | **0.654** @ 0.25 — callback export, not ep200 weights |
| EXP-20260609-09 | `train` | Gaussian 100-train 200ep | **Supported** | **0.654** @ 0.25 — identical to 40ep; ep11 ES blocks scaling |
| EXP-20260609-08 | `train` + `post` | Gaussian 100-train 40ep | **Supported** | Micro **0.654** @ thr=0.25 — below 50-train; ep11 ckpt undertrain |
| EXP-20260609-07 | `train` + `post` | Gaussian 50-train 40ep smoke | **Supported** | Micro **0.674** @ thr=0.35 — superseded by EXP-12 |
| EXP-20260609-06 | `train` + `post` | Gaussian 20-train 40ep | **Supported** | Micro **0.667** @ thr=0.25 — beats 100-train 0.635 |
| EXP-20260609-04 | `model` | arch_large 10-train (32f/d3) | **Supported** | Collapse: micro **0.326** @ thr=0.10 — epoch-1 restore |
| EXP-20260609-03 | `train` + `post` | Gaussian 10-train 40ep | **Supported** | Micro **0.633** @ thr=0.25 — beats binary; ≈100-train POST-tuned |
| EXP-20260609-02 | `post` | 75-train @ thr=0.20 (matched POST) | **Supported** | Micro **0.550** — 100-train gain not threshold-only |
| EXP-20260609-01 | `train` + `post` | Binary 10-train 40ep baseline | **Supported** | Micro **0.609** @ thr=0.20 (0.505 @ 0.05) |
| EXP-20260608-03 | `pre` (baseline) | Librosa spectral-flux val pilot | **Partial** | 5 songs: micro **0.32** @ 0.05 — ≪ MERT dense |
| EXP-20260608-02 | `train` / eval | ep100 vs final checkpoint | **Supported** | Identical **0.572** @ 0.05 — early-stop restores best frame weights |
| EXP-20260608-01 | `post` | 100-train val threshold sweep | **Supported** | Optimal thr=**0.20** → micro **0.635** (+6.3 pp vs 0.05) |
| EXP-20260607-02 | `pre` + dense + `train` | 100-train scale (frame-F1 ckpt) | **Supported** | **0.635** @ thr=0.20; 0.572 @ 0.05 (threshold transfer error) |
| EXP-20260607-01 | `metric` | Canonical dense val eval | **Supported** | Micro event F1 **0.577** @ thr=0.05 (EXP-21 re-eval) |
| EXP-20260606-21 | `pre` + dense + `train` | 75-train / 200-ep plateau curve | **Supported** | Peak val F1 **0.50** @ ep156; late drift to 0.40 |
| EXP-20260606-20 | `pre` + dense + `train` | 50-train / 40-val MERT scale-up | **Supported** | Peak val onset F1 **0.48** (ep 49); +15% vs 20-train |
| EXP-20260606-19 | `pre` + dense + `train` | 20-train / 40-val MERT pilot | **Supported** | Peak val onset F1 **0.42** (ep 44); best val PR-AUC 0.25 |
| EXP-20260606-01 | `pre` (dense) | Dense mel vs MERT val | **Supported** | MERT ≫ mel (~0.36 vs ~0.12) |
| EXP-20260606-02 | `train` | Tide overfit raw + learnable deltas | **Supported** | ~28% F1 plateau |
| EXP-20260606-03 | `train` (shortcuts) | GT refs + frozen deltas | **Supported** | F1 = 1.0 — pipeline check only |
| EXP-20260606-04 | `model` | K=634 vs K=1024 + shortcuts | **Supported** | Both F1=1; K ≥ max onsets suffices |
| EXP-20260606-07 | `pre` + ordered `train` | Tide suite 50 ep, three frontends | **Supported** | MERT > mel >> conv1d (0%); ordered train collapses conv1d |
| EXP-20260606-08 | `train` | Hungarian L1 train + suite 50 ep | **Supported** | All ~27–29% F1; conv1d 0→27% |
| EXP-20260606-09 | `train` | Suite 100 ep (Hungarian) | **Supported** | ~28–30% F1; no overfit — epoch plateau |
| EXP-20260606-10 | `train` / `post` / `model` | Threshold + loss + arch ablations | **Supported** | None break ~30% plateau |
| EXP-20260606-11 | `model` / `metric` | Bisection: grid oracle + half-cheat | **Supported** | Oracle 31%; not a bug; formulation limit |
| EXP-20260606-12 | `pre` + dense | Dense MERT tide overfit standalone | **Supported** | ~98% event F1 (607/634 TP) |
| EXP-20260606-13 | `pre` + dense | Dense mel vs MERT tide 100 ep suite | **Partial** | MERT > mel event F1; MERT ~35% (EXP-12 not replayed) |

---

## Experiment entries

Full write-ups below; prepend new entries here after each measurable run. Per-run configs: `configs/`, `callbacks/`.

### EXP-20260724-03: AR decode length control + EOS trace diagnostics

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 23:02:12 |
| **Track** | `post` + `metric` (AR) |
| **Question** | Can free-run under-generation be probed offline without retraining, and what does `<EOS>` behavior look like on a checkpoint that does **not** collapse? |
| **Tooling** | `inference.ArLengthControl` (`eos_logit_bias`, `min_onset_tokens`) threaded through both free-run paths (prefix loop and KV decode); `ArDecodeStats.eos_prob_trace` records per-step `<EOS>` probability **before** control is applied; `debug_ar_onset_overfit.py` gains `--ar_decode_eos_logit_bias` / `--ar_decode_min_onset_tokens` and an `ar_decode.eos_trace` report block (per song and split-aggregated) |
| **Checkpoint** | `models_wsl/ar/tide_overfit/ar_onset_model.keras` (local 2026-07-02 retrain; **not** the graduated EXP-20260630-01 artifact — see caveat) |
| **Baseline (no control)** | Teacher ordered **627/634** (0.9890) @ 20 ms vs `target_times`; free-run two-pass **622/634** (0.9811); chart aux **617/634**; Hungarian F1 **0.9795**. Decode length **636**, stopped on EOS. |
| **EOS trace (baseline)** | first step **0.0001**, mean **0.0017** over 635 steps, max **0.9778**, first crosses 0.5 at step **634** — i.e. `<EOS>` stays near zero for the whole song and spikes exactly at the true end |
| **Arm: `--ar_decode_min_onset_tokens 650`** | `<EOS>` suppressed past its spike; decode runs to the **2048** cap without stopping. Ordered **547/2048** (0.2671); Hungarian F1 **0.4243** (**569** TP, **1479** FP, **65** FN). After the suppressed spike, EOS probability falls back to ~**0.002** rather than staying high. |
| **CPU/GPU parity** | Identical metrics on Windows CPU (`STEPCOVNET_NO_WSL=1`, **330 s**) and WSL GPU (**94 s**) |
| **Tests** | `venv\Scripts\python.exe -m pytest tests/onset_ar -q` — **70 passed** (7 new: EOS softmax, argmax default, min-token suppression, logit bias, trace length, prefix/KV parity under control, forced-length invariant) |
| **Incidental fix** | WSL GPU dispatch was failing outright: `.gitattributes` had no `*.sh` rule, so with `core.autocrlf=true` all `scripts/wsl_*.sh` were checked out CRLF and bash rejected `set -o pipefail`. Added `*.sh text eol=lf` and renormalized. |
| **Logs** | `logs/ar_tide_decode_baseline_gpu.log` · `logs/ar_tide_decode_min_tokens_650.log` |
| **Conclusion** | **Supported** as tooling; **cautionary** as a fix. The `eos_trace` block gives a direct, retrain-free readout of length behavior and establishes the healthy reference shape (flat-near-zero, single spike at the true end) to compare a collapsing multi-song checkpoint against. But forcing length on tide converts the failure rather than fixing it — **1479** false positives once decode is pushed past the true end. Expect the same on multi-song: `min_onset_tokens` / EOS bias are **diagnostics**, and length control alone will not deliver **`gate-val-vs-dense`**. Remaining live cause is exposure bias (`scheduled_sampling_max_p: 0.0`); see [NOTE-20260724-02](DISCUSSION_NOTES.md#note-20260724-02-hypotheses-eliminated-for-multi-song-free-run-under-generation). |
| **Caveat** | None of the July 16–24 AR artifacts (200t/50v and 50t/50v checkpoints, subset indices, their logs) exist in this clone, so EXP-20260724-02 could not be re-measured here. The local tide checkpoint scores **627/634** teacher, below the graduated **634/634**; treat it as a working checkpoint, not the champion artifact. |

### EXP-20260724-02: AR corrected-mask 200t/50v train + offline val decode

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 09:45:05 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With early stopping on a 200-train / 50-val corrected-mask run, does best-`val_loss` transfer to offline teacher + free-run on the val split? |
| **Config** | [`configs/ar/scale_200t_50v.json`](../../configs/ar/scale_200t_50v.json) — index `data/final_data/training_index_200t_50v.json`; `cache_in_memory` + `cache_max_samples: 250`; `epochs: 500`, `early_stopping_patience: 25`; `checkpoint_metric: val_loss`; `λ_inc=0`; corrected masks |
| **Train** | WSL GPU **200/50** steps/ep; early stop **ep 65**, restored best **ep 40**. Best `val_loss` **~12.73**; at best: `val_aux_f1_hungarian` **~0.120**, `val_token_accuracy` **~0.50**. Steady ~**50–55 s**/ep with cache. |
| **Offline eval** | `debug_ar_onset_overfit.py --split val --ar_decode` on `models_wsl/ar/scale_200t_50v_corrected_masks/ar_onset_model.keras` (**50** songs, ~**524 s**). Teacher: ordered **105/36860** @ 20 ms (`rate` **0.0028**); Hungarian F1 **0.1199** (matches train val). Free-run: ordered **1/36860**; Hungarian F1 **0.0360**; **3400** preds vs **36860** GT (`ar_decode_length_sum` **3500**; all **50** songs stopped on EOS). |
| **Artifacts** | Model: `models_wsl/ar/scale_200t_50v_corrected_masks/ar_onset_model.keras`. Train log: `logs/ar_scale_200t_50v_corrected_masks.log`. Decode: `logs/ar_scale_200t_50v_val_decode_clean.json` (+ mixed `logs/ar_scale_200t_50v_val_decode.json`). |
| **Conclusion** | **Partial.** Scaling train rows + ES improves `val_loss` vs 50t (**~20.9 → ~12.7**) and holds teacher Hungarian F1 ~**0.12**, but **free-run collapses** via early EOS / severe under-generation. Ordered `timing_match` remains near-zero (pointer-dominated). Blocker for **`gate-val-vs-dense`** is multi-song free-run length/quality, not more train songs alone. |

### EXP-20260724-01: AR corrected-mask 50t/50v 500-ep scale-up

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 02:52:28 |
| **Track** | `train` + `metric` (AR) |
| **Question** | On a 50-train / 50-val corrected-mask smoke, does extending to **500** epochs keep improving val quality, or does the run overfit? |
| **Config** | [`configs/ar/scale_50t_50v.json`](../../configs/ar/scale_50t_50v.json) (logged as `smoke_50t_50v.json`; renamed) — index `data/final_data/training_index_50t_50v.json` (50/50 rows; 48/45 songs); `legacy_inverted_attention_masks: false`; `cache_in_memory: true`, `cache_max_samples: 128`; `checkpoint_metric: val_loss`; `λ_inc=0`; **500** ep |
| **Prior 50-ep** | Same recipe/paths earlier the same evening: `val_loss` **44.3 → 21.0**, `val_aux_f1_hungarian` **0.008 → 0.126**, `val_token_accuracy` **0.10 → 0.43** (run `20260723-222238`) |
| **Train (500 ep)** | WSL GPU **50/50** steps/ep; wall ~**2.3 h**. Ep1 → ep65 → ep500: train loss **51.1 → 7.2 → 0.06**; `val_loss` **44.3 → 20.9 → 33.6**; `val_aux_f1_hungarian` **0.008 → 0.141 → 0.221**; `val_token_accuracy` **0.10 → 0.43 → 0.36** (train token acc ep500 **0.98**). Best `val_loss` **20.911 @ ep 65**; best val F1 **0.223 @ ep 466**. Steady ~**16–17 s**/ep with cache. |
| **Artifacts** | Model: `models_wsl/ar/smoke_50t_50v_corrected_masks/ar_onset_model.keras` (final weights; pre-rename path). Callbacks/TB: `callbacks/ar/smoke_50t_50v_corrected_masks/` run **`20260723-225906-…`**. Log: `logs/ar_smoke_50t_50v_corrected_masks.log`. Best-`val_loss` Keras ckpt under `callbacks/.../models/20260723-225906-…/`. |
| **Conclusion** | **Partial.** Multi-song corrected-mask training on 50/50 is healthy through ~ep **65**, then **severe overfit** (train collapses while `val_loss` rises ~**21 → 34**). Late val F1 gains are small and do not justify 500 ep. Prefer best-`val_loss` checkpoint / early stop; next scale **train rows** (300t/50v MERT ready), not epoch count. |

### EXP-20260723-02: AR corrected-mask 10-song smoke (50 ep, in-memory cache)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-23 21:42:59 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With keep-valid masks, dynamic padding, and in-memory sample cache, does the 10-song smoke keep improving through 50 epochs? |
| **Config** | [`configs/ar/smoke.json`](../../configs/ar/smoke.json) — `epochs: 50`, `cache_in_memory: true`, `cache_max_samples: 64`, `legacy_inverted_attention_masks: false`, `dynamic_padding: true` |
| **Train** | WSL GPU **10/10** steps/ep; `val_loss` **35.0410 → 12.0962** (best **11.5262** @ ep 49); `val_event_onset_f1` **0.0128 → 0.1141**; `val_token_accuracy` **0.0017 → 0.1585**; train loss **52.2 → 9.6**. Steady epochs ~**2 s** (cache warmed at dataset build; vs ~16–18 s uncached 5-ep run). |
| **Artifacts** | `models_wsl/ar/smoke_10song_corrected_masks_ep50/` · `callbacks/ar/smoke_10song_corrected_masks_ep50/` · `logs/ar_smoke_10song_corrected_masks_ep50.log` |
| **Conclusion** | **Supported.** Corrected-mask multi-song training continues past the 5-ep gate with clear loss/F1 gains; in-memory cache is the right default for this smoke size. Next: **`final-data-mert`**. |

### EXP-20260723-01: AR corrected-mask 10-song smoke

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-23 21:35:54 |
| **Track** | `train` + `metric` (AR) |
| **Gate** | **`gate-10song-smoke`** on corrected masks + dynamic padding — batches run; loss decreases; teacher F1 > 0 |
| **Config** | [`configs/ar/smoke.json`](../../configs/ar/smoke.json) — `legacy_inverted_attention_masks: false`, `dynamic_padding: true`, `λ_residual=5`, 5 ep |
| **Train** | WSL GPU: **10/10** steps/ep; epoch walls **43 / 16 / 16 / 17 / 18 s**. `val_loss` **35.0410 → 26.8546**; `val_event_onset_f1` **0.0128 → 0.0143** (> 0); `val_token_accuracy` ep5 **0.0041** (not all-EOS) |
| **Compare** | Legacy-mask smoke ([EXP-20260630-02](#exp-20260630-02-ar-gate-10song-smoke)): `val_loss` **53.4 → 38.7**. Curves not 1:1 comparable (masks + padding), but gate criteria met. |
| **Artifacts** | `models_wsl/ar/smoke_10song_corrected_masks/` · `callbacks/ar/smoke_10song_corrected_masks/` · `logs/ar_smoke_10song_corrected_masks.log` |
| **Conclusion** | **PASS.** Corrected-mask multi-song smoke is green after the Partial tide regression. Next: **`final-data-mert`** then AR scale-up / **`gate-val-vs-dense`**. |

### EXP-20260716-02: AR corrected-mask tide overfit regression

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-23 21:22:19 (log); run1 train/decode 2026-07-16 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With Keras keep-valid attention masks (`legacy_inverted_attention_masks: false`), does the champion tide recipe still reach teacher + free-run **634/634** @ 20 ms vs `target_times`? |
| **Config** | [`configs/ar/tide_overfit_corrected_masks.json`](../../configs/ar/tide_overfit_corrected_masks.json) — v8 stack (`d_model=384`, `λ_residual=30`, `lr=1e-4`, 400 ep, seed 42); **`λ_incremental_consistency=0`** (champion used **0.01**; OOM on RTX 3070 Ti ~5.5 GB at ep~2 with `0.01`) |
| **Run1 train** | 400 ep completed; final `val_token_accuracy=1.0`, `val_overfit_gate≈0.9984`. Log: `logs/ar_tide_overfit_corrected_masks.log`. Model: `models_wsl/ar/tide_overfit_corrected_masks/ar_onset_model.keras` |
| **Run1 decode** | Offline `--ar_decode`: teacher **633/634**, free-run **633/634** ordered @ 20 ms vs `target_times`; chart aux **627/634**; AR F1 **≈0.989**. Log: `logs/ar_tide_overfit_corrected_masks_decode.log` |
| **Throughput (2026-07-23)** | Default-on in-memory overfit batch cache (`dataset.cache_overfit_batch=true`): matched 5-ep tide A/B steady epoch **~3.5 s → ~0.39 s** (~**9×**). Host RAM ~**123 MB**/song fixed-pad. |
| **Run2** | Same recipe, fresh scratch (`models_wsl/ar/tide_overfit_corrected_masks_run2/`); 400 ep in ~5.5 min with batch cache. Offline decode: teacher **633/634**, free-run **633/634**; chart aux **630/634**. Log: `logs/ar_tide_overfit_corrected_masks_run2.log` · `logs/ar_tide_overfit_corrected_masks_run2_decode.log` |
| **Conclusion** | **Partial** — two independent scratch runs both stop at **633/634** with free-run matching teacher exactly. Corrected masks are trainable and free-run-consistent; the perfect overfit bar is not recovered on this GPU recipe (`λ_inc=0`). Do not graduate as a new champion. Proceed to corrected-mask multi-song smoke / scale-up gates, or chase the missing onset only if perfect-bar parity is required before that. |

### EXP-20260716-01: AR validation aggregation + dynamic length bucketing

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-16 01:15:00 |
| **Track** | `pre` + `model` + `metric` + `train` (AR) |
| **Question** | On the 10-song smoke, what changes after (1) fixing validation accumulation and (2) replacing fixed 300 s / 2048-step tensors with dynamic padding and normalized length buckets? |
| **Baseline** | [`configs/ar/smoke.json`](../../configs/ar/smoke.json), RTX 3070 Ti, seed 42, 10 train + 2 val batches. Historical fixed-padding run: epoch times **57 / 23 / 22 s**; `test_step` reset metrics per batch, so reported val values represented only the final val chart. |
| **Step 1 — validation fix** | Removed the per-batch metric reset and added a two-batch regression test. One-epoch retry: **52 s**; full two-chart aggregate `val_loss=51.0036`, `val_aux_f1_hungarian=0.0143`, `val_token_accuracy=0.0144`. Baseline epoch 1 reported last-chart-only `val_loss=53.4066`, F1 `0.0125`, token accuracy `0.0`. Runtime difference (**57 → 52 s**) is compile/run noise; the gain is correct checkpoint data. An earlier retry hit the existing fixed-padding 5.5 GB GPU ceiling. |
| **Step 2 — dynamic buckets** | Added compact per-sample arrays, dynamic Keras sequence dimensions, normalized encoder/decoder length buckets `[512, 768, 1024, 1536]`, and pointer-logit cropping. Matched corrected-mask fixed-padding control: **51 / 22 / 21 s**. Dynamic: **47 / 17 / 18 s** — steady mean **21.5 → 17.5 s** (**18.6% faster**), first epoch **51 → 47 s** (7.8% faster), with no OOM. |
| **Attention-mask discovery** | During step 2, source inspection confirmed Keras keeps entries where `attention_mask=True`; the AR layers historically returned the inverse and therefore attended to padding. New smoke/scale-up models use correct masks; legacy configs/checkpoints retain historical semantics through `legacy_inverted_attention_masks=true` / serialized layer defaults ([NOTE-20260716-01](DISCUSSION_NOTES.md#note-20260716-01-ar-attention-mask-semantics-were-inverted)). |
| **Final 3-epoch signal** | Correct-mask dynamic run: train loss **52.1889 → 50.3317**; `val_loss` **35.0410 / 37.2351 / 38.9519**; val Hungarian F1 **0.0128 / 0.0128 / 0.0114**; val token accuracy **0.0017 / 0.0041 / 0.0041**. Matched fixed-padding ep3: val F1 **0.0086**, token accuracy **0.0554**, `val_loss=59.6918`. Loss magnitudes are not directly comparable because dynamic pointer CE has fewer padded classes; three epochs show no reliable convergence winner. |
| **Tests** | `venv\Scripts\python.exe -m pytest tests/onset_ar -q` — **48 passed**. |
| **Logs** | `logs/ar_perf_baseline.log` · `logs/ar_perf_validation_fix_retry.log` · `logs/ar_perf_fixed_padding_valid_masks.log` · `logs/ar_perf_dynamic_buckets.log` (diagnostic legacy-mask run) · `logs/ar_perf_dynamic_buckets_valid_masks.log` |
| **Conclusion** | **Supported.** Step 1 fixes model selection correctness without a meaningful speed cost. Step 2 gives a matched **18.6%** steady epoch speedup on the small smoke and removes fixed-padding OOM pressure. Convergence quality remains open; run a longer matched scale-up after the corrected-mask tide regression gate. |

### EXP-20260703-01: AR tide token class weight ablation (champion recipe)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-03 01:42:00 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does `token_class_weight: inverse_freq` or `inverse_sqrt_freq` improve tide overfit on the **v8 champion stack** (`lambda_residual=30`, `d_model=384`, …)? |
| **Baseline** | Champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) uses `token_class_weight: none` and offline free-run **634/634** ([EXP-20260630-01](#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)) |
| **Configs** | [`v9_inverse_freq.json`](../../configs/ar/versions/tide_overfit/v9_inverse_freq.json) · [`v9_inverse_sqrt_freq.json`](../../configs/ar/versions/tide_overfit/v9_inverse_sqrt_freq.json) — checkpoint / early-stop monitor `val_timing_match_teacher` |
| **Arm A (`inverse_freq`)** | WSL GPU ~382 ep (~16 min); `val_token_accuracy` **~7.5%**; `val_timing_match_teacher` **1.0**; early stop did not fire (0.9984 ↔ 1.0 oscillation) |
| **Arm A offline** | Teacher **634/634** PASS; free-run `--ar_decode` **360/634** (56.8%) FAIL |
| **Arm B (`inverse_sqrt_freq`)** | WSL GPU ~317 ep (~14 min); `val_token_accuracy` **~42%**; `val_timing_match_teacher` **1.0** |
| **Arm B offline** | Teacher **634/634** PASS; free-run **343/634** (54.1%) FAIL |
| **Logs** | `logs/ar_tide_inverse_freq.log` · `logs/ar_tide_inverse_sqrt_freq.log` |
| **Artifacts** | `models_wsl/ar/tide_overfit_inverse_freq/` · `models_wsl/ar/tide_overfit_inverse_sqrt_freq/` |
| **Conclusion** | **Partial** — drop-in class weights on champion v8 **do not** improve free-run (arms A/B fail `--ar_decode`). Not a clean test of class weighting: high `lambda_residual` masks weak tokens while teacher timing hits **634/634**. Historical gate-tide pass used `inverse_freq` **with** lower `lambda_residual`, `lambda_time_ramp_epochs`, and smaller `d_model` ([NOTE-20260627-02](DISCUSSION_NOTES.md#note-20260627-02-gate-tide-overfit-resolution), [`v1.json`](../../configs/ar/versions/tide_overfit/v1.json)). |
| **Open follow-up** | Revisit with **co-tuned** recipe: lower `lambda_residual`, time ramp, checkpoint on `val_gate_teacher`, judge `--ar_decode`; try `inverse_sqrt_freq`/capped weights or focal CE ([NOTE-20260703-01](DISCUSSION_NOTES.md#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred)). **For now:** champion stays `token_class_weight: none`; scale-up not blocked. |

### EXP-20260630-03: AR tide MERT normalization A/B

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-01 01:00:00 |
| **Track** | `pre` + `train` (AR ablation) |
| **Question** | Does dense-style per-song MERT z-score (`normalize_onset_spectrogram`) speed tide perfect overfit vs raw hidden states? |
| **Config** | [`tide_overfit_raw_ab.json`](../../configs/ar/tide_overfit_raw_ab.json) · [`tide_overfit_norm_ab.json`](../../configs/ar/tide_overfit_norm_ab.json) — champion tide recipe (`d_model=384`, `lr=1e-4`, `lambda_residual=30`, 400 ep, `val_overfit_gate` checkpoint); `lambda_incremental_consistency=0` (OOM at 0.01 on 3070 Ti — see NOTE) |
| **Arms** | **Raw:** `normalize_mert_features: false` · **Norm:** `normalize_mert_features: true` (applied after `resample_features_to_hop_grid`, before patch) |
| **Logs** | `_tmp/tide_norm_ab/raw.log`, `norm.log` (local; UTF-16 from PowerShell `Tee-Object`) |
| **Raw outcome** | First `val_overfit_gate` ≥ 0.9999 @ **ep 399** (gate **1.0**); ≥ 0.99 @ ep 289 |
| **Norm outcome** | Best gate **0.9984** @ ep 313; never ≥ 0.999; slightly faster to 0.99 (ep 282 vs 289) |
| **Conclusion** | **Reject** per-song MERT norm for AR. Default stays **raw** ([NOTE-20260701-01](DISCUSSION_NOTES.md#note-20260701-01-ar-tide-overfit-reject-per-song-mert-z-score)). Dense norm remains dense-only. |

### EXP-20260630-02: AR `gate-10song-smoke`

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-30 23:59:00 |
| **Track** | `pre` + `train` (AR) |
| **Gate** | **`gate-10song-smoke`** — batches build on `training_index_10songs.json`; loss decreases; teacher decode F1 > 0; no all-EOS collapse |
| **Config** | [`configs/ar/smoke.json`](../../configs/ar/smoke.json) — champion stack (`d_model=384`), `overfit_one_song: false`, `lambda_residual=5`, 5 ep |
| **MERT** | `extract_mert_features.py --training_index_path=.../training_index_10songs.json --beside_audio --device=cuda` (11 extracted, 1 cached) |
| **Verify** | `--verify-only`: **10/10** train batches, **2/2** val batches |
| **Train** | WSL GPU 5 ep: ep1 `val_loss` **53.41** → ep5 **38.69**; `val_event_onset_f1` **0.0125 → 0.0062** (teacher-fed, > 0); `val_token_accuracy` ep5 **0.025** (not all-EOS) |
| **Artifacts** | `models_wsl/ar/smoke_10song/`, `callbacks/ar/smoke_10song/` |
| **Conclusion** | **`gate-10song-smoke` PASS**. Multi-song `training_index_path` wired in `onset_ar/datasets.py` + `train_ar_onset`. Next: full `final_data` MERT + scale-up / **`gate-val-vs-dense`**. |

### EXP-20260630-01: AR tide scratch perfect overfit (iter175 → v8 champion)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-30 20:23:12 |
| **Track** | `train` + `metric` (AR) |
| **Gate** | Tide single-chart **`gate-ar-decode`** / perfect overfit — offline free-run **`ordered_onset_match` 634/634** @ 20 ms vs **`target_times`** ([NOTE-20260630-01](DISCUSSION_NOTES.md#note-20260630-01-ar-free-run-primary-vs-target_times)) |
| **Config** | Scratch **iter175** → [`configs/ar/versions/tide_overfit/v8.json`](../../configs/ar/versions/tide_overfit/v8.json); champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) |
| **Recipe** | `d_model=384`, `lr=1e-4`, `lambda_residual=30`, `lambda_incremental_consistency=0.01`, `incremental_consistency_max_steps=32`, `eos_token_weight_scale=0.2`, `scheduled_sampling_max_p=0`, **400 ep**, random init (no warm-start) |
| **Harness** | `scripts/ar_tide_iter/` — iter174–217 autoresearch; **iter175** first scratch teacher **634/634**; decode sweeps plateaued at **633/634** vs raw chart until eval reference fixed |
| **Model (source)** | `models_wsl/ar/tide_overfit_iter/iter175/` (iter tree removed after graduation) |
| **Model (champion)** | `models_wsl/ar/tide_overfit/ar_onset_model.keras` — promoted via `graduate_ar_tide_overfit.py` |
| **Offline eval** | iter175 attempt 3: teacher **634/634**, free-run **634/634** (primary); chart aux **633/634** (hop-quant gap at onset 318 — not a decode bug) |
| **Conclusion** | **`gate-ar-decode` PASS** on tide. Champion manifest **1.0** free-run. Next: **`gate-10song-smoke`** ([AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) §10.1). Session log: [AR_TIDE_OVERFIT_ITER_LOG.md](AR_TIDE_OVERFIT_ITER_LOG.md). |

### EXP-20260628-02: AR tide perfect overfit (`val_overfit_gate`)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-28 06:45:00 |
| **Track** | `train` + `metric` (AR) |
| **Gate** | Single-song overfit — **`val_overfit_gate` = min(`val_token_accuracy`, `val_ordered_onset_match`)**; offline `--ar_decode` **`ordered_onset_match` 634/634** |
| **Code** | `OverfitGateCallback`, `perfect_overfit_early_stop`; `test_step` metric reset (fix val contamination) |
| **Config (run1)** | `configs/onset_ar_tide_overfit_perfect.json` — SS **0**, `token_class_weight: none`, warm-start `gate_v5`, checkpoint **`val_overfit_gate`** |
| **Config (run2)** | `configs/onset_ar_tide_overfit_perfect_run2.json` — warm-start run1; `lambda_residual: 10`, **`ar_decode_val_every_n_epochs: 0`** (offline AR only), checkpoint **`val_overfit_gate`** |
| **Log (run1)** | `logs/ar_tide_overfit_perfect_run1.log` — 300 ep |
| **Model (run1)** | `models_wsl/ar_tide_overfit_perfect/ar_onset_model.keras` |
| **Run1 outcome** | `val_token_accuracy` **1.0** (ep ~20+); `val_event_onset_f1` **~0.9905**; offline AR: **634/634** tokens, `first_mismatch_step: null`, AR F1 **~0.954** (two-pass) — 6 Hungarian match collisions (per-step timing OK) |
| **Log (run2)** | `logs/ar_tide_overfit_perfect_run2.log` — **14 ep** early stop (`perfect_overfit_early_stop`, gate **1.0** ep 12–14) |
| **Model (run2)** | `models_wsl/ar_tide_overfit_perfect_v2/ar_onset_model.keras` |
| **Run2 training** | Ep 2 dip (`val_overfit_gate` **~0.64**); recovered ep 12+: `val_token_accuracy` **1.0**, `val_event_onset_f1` **1.0**, `val_overfit_gate` **1.0** |
| **Run2 offline (teacher)** | `debug_ar_onset_overfit.py` — event F1 **~1.0** (634/634 TP); **632/634** within 20 ms; mean abs err **5.09 ms**, max **23.67 ms** |
| **Run2 offline (`--ar_decode`)** | Two-pass AR F1 **0.978** (620 TP / 14 FP / 14 FN); decode length **636**, EOS OK; incremental pointer+residual F1 **0.957** |
| **Config (run3)** | `configs/onset_ar_tide_overfit_perfect_run3.json` — warm-start v2; `lambda_residual: 10`, LR **5e-5**, **200 ep**, no perfect early stop |
| **Log (run3)** | `logs/ar_tide_overfit_perfect_run3.log` — 200 ep |
| **Model (run3)** | `models_wsl/ar_tide_overfit_perfect_v3/ar_onset_model.keras` |
| **Run3 offline (teacher)** | `logs/ar_perfect_v3_baseline_decode.json.txt` — event F1 **~1.0**; **633/634** within 20 ms; mean abs err **4.0 ms** (no `--ar_decode`) |
| **Log (run4 / v4 train)** | `logs/ar_perfect_v4_train.log` — **200 ep**; adds **`incremental_consistency_loss`** in train metrics |
| **Model (run4 / v4)** | `models_wsl/ar_tide_overfit_perfect_v4/ar_onset_model.keras` |
| **Run4 offline (teacher)** | `logs/ar_perfect_v4_decode.log` — event F1 **~1.0** (634/634 TP); **633/634** within 20 ms; mean abs err **3.6 ms**, max **22.75 ms** |
| **Run4 offline (`--ar_decode`)** | Two-pass AR F1 **0.975** (618 TP / 16 FP / 16 FN); decode length **636**; incremental pointer+residual F1 **0.942**; GT-parallel on free-run tokens **1.0** |
| **Config (run5)** | `configs/ar/overfit_perfect/run5.json` — warm-start run2; `lambda_incremental_consistency: 0.1`, `incremental_consistency_max_steps: 16`, `lambda_residual: 10`, LR **5e-5**, **200 ep** |
| **Log (run5)** | `logs/ar_tide_overfit_perfect_run5.log` — **200 ep** |
| **Model (run5)** | `models_wsl/ar_tide_overfit_perfect_v5/ar_onset_model.keras` |
| **Run5 training** | Ep 3+: `val_token_accuracy` **1.0**; final `val_event_onset_f1` **0.9968**, `val_overfit_gate` **0.9968** |
| **Run5 offline (teacher)** | event F1 **~1.0** (634/634 TP); **633/634** within 20 ms; mean abs err **3.86 ms**, max **24.84 ms** |
| **Run5 offline (`--ar_decode`)** | Two-pass AR F1 **0.976** (619 TP / 15 FP / 15 FN); decode length **636**, EOS OK; incremental pointer+residual F1 **0.940** |
| **Conclusion** | Teacher path **1.0** on run2/v3/v4/v5. **Best warm-start free-run: run2 (0.978)**. Warm-start path did not reach **634/634** free-run bar. **Scratch iter175** closed the gate ([EXP-20260630-01](#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)). |

### EXP-20260628-01: AR `gate-ar-decode` v2 (WSL 150ep, warm-start gate_v5)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-28 05:15:00 |
| **Track** | `train` + `model` (AR) |
| **Gate** | `gate-ar-decode` — **PASS** via scratch iter175 ([EXP-20260630-01](EXPERIMENT_LOG.md#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)); warm-start v2 runs below document the earlier SS path |
| **Config** | `configs/onset_ar_tide_decode_v2.json` — `init_model_path` → `gate_v5`; full tide loss (`lambda_time` ramp, `lambda_residual: 5`); `eos_token_weight_scale: 0.2`; SS warmup **15** + ramp **100** → `p=1`; `ar_decode_val_every_n_epochs: 0`; checkpoint **`val_event_onset_f1`** |
| **Code** | Scheduled sampling (`trainers.py`); eager `ArDecodeValidationCallback` (AR decode off compiled `test_step`); KV-cache free-run (`kv_decode.py`, default in `inference.py`) |
| **Log (run 1)** | `logs/ar_tide_overfit_gate_decode_v2_run1.log` — interrupted ep **68**/150 |
| **Log (run 2)** | `logs/ar_tide_overfit_gate_decode_v2.log` — restarted after KV-cache + eager-callback land |
| **Model** | `models_wsl/ar_tide_overfit_gate_decode_v2/ar_onset_model.keras` |
| **Run 1 outcome** | Ep 1: early-EOS (`val_ar_decode_length` **13**); ep 7+: full chart (**633–635** steps). Best `val_ar_decode_event_f1` **~0.50** (ep 51–57). Teacher-fed `val_event_onset_f1` fell **1.0 → ~0.51** under token-only + SS — expected tradeoff while AR path learns |
| **Conclusion** | Gate not passed; run 2 continues with faster AR validation (KV cache). See [NOTE-20260628-01](DISCUSSION_NOTES.md#note-20260628-01-gate-ar-decode-v2-infra). |

### EXP-20260627-04: AR `gate-tide-overfit` pass (WSL 300ep)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-27 19:53:03 |
| **Track** | `train` + `model` (AR) |
| **Gate** | `gate-tide-overfit` — **PASS** |
| **Config** | `configs/onset_ar_tide.json`; `lambda_time=1.0`, `lambda_time_ramp_epochs=100`, `lambda_residual=5.0`, `token_class_weight=inverse_freq`, `dropout_rate=0`, argmax decode |
| **Log** | `logs/ar_tide_overfit_gate_v5.log` |
| **Model** | `models_wsl/ar_tide_overfit_gate_v5/ar_onset_model.keras` |
| **Outcome** | Best/final `val_event_onset_f1` **1.0** (from ep ~180); `debug_ar_onset_overfit`: **634/634** within 20 ms, 0 patch errors |
| **Conclusion** | Residual MSE (`lambda_residual=5`) required after pointer-only + λ ramp reached F1 ~0.83. Proceed to **`gate-ar-decode`**. |

### EXP-20260627-03: AR tide overfit training fixes (λ ramp ablation)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-27 19:00:00 |
| **Track** | `train` + `model` (AR) |
| **Gate** | `gate-tide-overfit` — partial (**F1 ~0.83**, not pass) |
| **Changes** | Class-weighted token CE, argmax F1/time decode, `dropout=0`, linear `lambda_time` ramp 0→1 over 100 ep |
| **Log** | `logs/ar_tide_overfit_gate_v4.log` |
| **Outcome** | Best `val_event_onset_f1` **~0.834**; pointer CE **~0.0015**; debug: **0** patch errors, **103** residual timing errors |
| **Conclusion** | Pointer learns; residual head needs direct supervision — led to EXP-20260627-04. See [NOTE-20260627-02](DISCUSSION_NOTES.md#note-20260627-02-gate-tide-overfit-resolution). |

### EXP-20260627-02: AR `gate-tide-overfit` WSL 300ep (tide)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-27 18:30:00 |
| **Track** | `train` + `model` (AR) |
| **Gate** | `gate-tide-overfit` — **FAIL** |
| **Config** | `configs/onset_ar_tide.json`; WSL GPU; 300 epochs; `lambda_time=1.0`; teacher forcing only |
| **Code** | `a56f3aa` + local uncommitted attention-mask fix (`models.py`, `losses.py`) |
| **Log** | `logs/ar_tide_overfit_gate_v2.log` |
| **Outcome** | Best `val_event_onset_f1` **~0.137** (~epoch 29); final **0.0**; `val_token_accuracy` **0.4803** on 282/300 epochs; `val_token_loss` ↓ to ~1.7; `val_pointer_loss` ~6.46 |
| **Conclusion** | Single-song overfit gate not met. **305/635** target steps use token 83 (Δ17 frames = 170 ms) — matches accuracy plateau numerically; likely majority-class / training-dynamics issue (see NOTE-20260627-01). Pointer head near uniform. Further root-cause work open. |

### EXP-20260627-01: AR Phase 0+1 implementation + tide verify

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-27 12:00:00 |
| **Track** | `pre` + `model` (AR) |
| **Config** | `configs/onset_ar_tide.json`; `--verify-only` |
| **Code** | `86117f9` (scaffold), `a56f3aa` (enc/dec + trainer), `e1ad6b9` (seed/F1 fixes) |
| **Outcome** | Package `src/stepcovnet/onset_ar/`; `scripts/train_onset_ar.py`; tide assets load; 634 onsets, 1607 patches, decoder len 635, vocab 339 |
| **Conclusion** | Locked v1 stack wired for tide overfit; proceed to `gate-tide-overfit` training |

### EXP-20260624-01: 10-song dense training smoke (`training_index_path`)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-24 12:00:00 |
| **Track** | `pre` + dense `train` |
| **Config** | `configs/onset_baseline.json`; `--training_index_path=data/final_data/training_index_10songs.json`; `--epochs=2`; CPU |
| **Outcome** | **10/10** train batches; model saved under `models/final_data_10song_dense_smoke` |
| **Conclusion** | Dense `create_dataset` + manifest-as-pointer path works end-to-end on nested `.chart.json` |

### EXP-20260624-02: 10-song event training smoke (2048 caps)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-24 12:30:00 |
| **Track** | `pre` + event `train` |
| **Config** | `configs/onset_event_audio_baseline.json` (`n_max_onsets`, `max_steps_per_chart`, `num_queries` = **2048**); same 10-song manifest; `--epochs=2` |
| **Outcome** | **10/10** train batches (Raputa chart has 1164 steps — needs full 2048 cap, not 1024) |
| **Conclusion** | Event path ready for `final_data`; align loader cap with model `num_queries` |

### EXP-20260623-02: P8 train/val manifest on full `final_data`

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-24 04:26:00 |
| **Track** | `pre` / dataset prep |
| **Config** | `scripts/build_training_index.py --output-dir data/final_data --overwrite` |
| **Outcome** | `training_index.json` — `stratified_song_v1`, seed 42, val_fraction 0.1; **1010** train / **110** val songs; **1745** / **197** chart rows |
| **Conclusion** | Song-level stratified split per bundle; trainers should point at this manifest, not duplicate dirs |

### EXP-20260622-01: P9 final_data loader smoke

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-22 01:11:00 |
| **Track** | `pre` / dataset prep |
| **Config** | Local `data/final_data`; `training_loader.discover_training_rows`, `pairing.list_training_samples`, `create_onset_event_dataset_from_pairs` (1 sample) |
| **Outcome** | **1942** chart rows; ITL 246 / Mizuki 1310 / Vocaloid 386; **822** with `chart_index > 0`; 0 missing audio or `.chart.json`; all ≤2048 steps; TF batch builds with GT onsets when `max_audio_seconds` covers chart offset |
| **Conclusion** | P9 loaders ready; superseded for train/val routing by P8 manifest (EXP-20260623-02) |

---

Older dense/event runs (EXP-20260606-* through EXP-20260610-*) are indexed above; add full per-run entries here when re-running or when promoting a result to the paper. Cross-links: `NOTE-…` in [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).
