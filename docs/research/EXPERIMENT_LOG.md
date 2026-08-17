# Experiment log

**Authoritative record** for runs and ablations. IDs: `EXP-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

Promote selected findings to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) only when drafting the paper — do not duplicate the full log there. Citations: [PAPER_LEDGER.md](PAPER_LEDGER.md) · [paper.bib](paper.bib).

**Related:** [discussion notes](DISCUSSION_NOTES.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [dataset prep plan](DATASET_PREP_PIPELINE.md) · [AR onset design](AR_ONSET_DESIGN.md) · [decisions checklist](DECISIONS_CHECKLIST.md)

---

## Current phase

**Updated:** 2026-08-16
**Primary track:** Literature recreation — DDCL on Dataset A (`omalley2025ddcl`), then ITGPT on B
**Status:** Full-split DDCL placement ran 8 ep. Best-`val_loss` val `M-slot48` F1@0.5 **0.601** vs beat-shuffle null **0.397** (skill **+0.204**); max-F1 **0.634** @ 0.40. Not a T-repro vs ITGPT Table 2 **0.70 / 0.76** (expanded Fraxtil). Two WSL OOMs (exit 9) were window preload; on-demand windows fixed it ([EXP-20260816-02](#exp-20260816-02-ddcl-48-slot-placement-full-split-on-dataset-a)).

**Next action:** DDCL audio-in **selection** on Dataset A (48-slot placement now has a real number). Do not train `final_data` for comparison.
**Blockers:** None — GPU free; placement ckpt `models_wsl/ddc/ddcl_placement_fraxtil/best.keras`.
**Defer:** `final_data` dense/AR as a literature number; more DDC C-LSTM eval; Dataset B until DDCL-on-A selection exists; AR-on-times locality / ladder scale-up; longer DDCL placement (more steps/epochs) unless asked.

### Dataset prep (PRE ingestion)

| Phase | Status |
| ----- | ------ |
| **Dataset A (Fraxtil)** | **Done** — `training_index_standard.json`: **90** songs / **450** standard charts, **81 / 9** train/val. Placement 128-ep val **0.652 / 0.734** ([EXP-20260815-03](#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap)). Cite `donahue2017ddc`. |
| P0–P9 (`final_data`) | **Done** — **1942** chart rows; `training_index.json` (`stratified_song_v1`: **1010** / **110** songs, **1745** / **197** chart rows train/val). **Drift:** the untracked copy on this clone reports **1009** / **110** songs and **1755** / **186** rows ([NOTE-20260725-02](DISCUSSION_NOTES.md#note-20260725-02-subset-sampling-gives-every-train-size-a-different-val-set)) |
| MERT subset | **Done** for scale-up — `training_index_300t_50v.json` (314 unique audio); `training_index_200t_50v.json` / `50t_50v` subsets |

**Recommended next step:** DDCL audio-in selection on Dataset A. `final_data` is transfer only.

### Onset detection (research track)

| Item | Status |
| ---- | ------ |
| Dense val best (`data/v2`) | BiLSTM 256u — micro event F1 **0.686** @ thr=0.30 (EXP-20260610-03) |
| Event tide formulation (`data/v2`) | ~27–30% F1 plateau; oracle ~31% (EXP-20260606-11) — formulation ceiling for K-query slots |
| `final_data` training hookup | **Done** — dense + event trainers accept `--training_index_path`; 10-song CPU smoke **10/10** batches (EXP-20260624-01/02) |
| Multi-song val on `final_data` | **Partial** — 50/100-row MERT BiLSTM smoke: event F1 **0.666** (skill **+0.371**); `timing_match` **0.0047** at floor ([EXP-20260815-07](#exp-20260815-07-final_data-dense-50t100v-mert-bilstm-smoke)) |
| **AR tide perfect overfit** | **PASS** — scratch **iter175** / champion **v8**: teacher + free-run **634/634** ordered @ 20 ms vs **`target_times`** ([EXP-20260630-01](#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)) |
| **AR 10-song smoke** | **PASS** — 5-ep corrected-mask ([EXP-20260723-01](#exp-20260723-01-ar-corrected-mask-10song-smoke)); **50-ep cached** `val_loss` **35.0 → 12.1**, teacher F1 **0.11** ([EXP-20260723-02](#exp-20260723-02-ar-corrected-mask-10song-smoke-50ep)) |
| **AR 50t/50v scale-up** | **Partial** — 500 ep: best `val_loss` **~20.9 @ ep 65**, then severe overfit; val F1 peaks **~0.22** ([EXP-20260724-01](#exp-20260724-01-ar-corrected-mask-50t50v-500-ep-scale-up)) |
| **AR 200t/50v scale-up** | **Partial** — ES @ ep **65**, best `val_loss` **~12.7 @ ep 40**; offline val teacher F1 **0.120**, free-run F1 **0.036** (severe under-gen) ([EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode)) |
| **AR corrected-mask regression gate** | **Partial** — run1 + run2 both teacher + free-run **633/634**; free-run tracks teacher; short of perfect bar ([EXP-20260716-02](#exp-20260716-02-ar-corrected-mask-tide-overfit-regression)) |
| **AR free-run length diagnostics** | **Ready** — `eos_trace` + `--ar_decode_min_onset_tokens` / `--ar_decode_eos_logit_bias` land offline decode probes; healthy tide reference: EOS mean **0.0017**, single spike at step **634** ([EXP-20260724-03](#exp-20260724-03-ar-decode-length-control--eos-trace-diagnostics)) |
| **AR next gate** | **Chance floor, not EOS** — no rung clears an audio-blind baseline ([EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)). Gate is now **positive skill over the strongest null** on `timing_match_teacher` and Hungarian F1 |
| **AR density conditioning** | **Retracted as a win** — the free-run gains track `pred/GT` along the null curve (0.36→0.132, 0.82→0.234, 0.90→0.263 vs nulls 0.154/0.250/0.261). Conditioning fixes prediction *count*, not timing ([EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)) |
| **AR chance floor** | **Measured** — dense charts (5.5 onsets/s) give Hungarian F1 @ 20 ms a floor of **0.225–0.336** at matched count; `timing_match` floor ≈ **0**. `stepcovnet.onset_null_baseline` computes it in-harness |
| **AR scheduled sampling** | **Closed, negative** — the feature was compiled out of the traced `train_step` ([EXP-20260802-05](#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step)); with `p` now a `tf.Variable` under `tf.cond`, a full rerun gives free-run **0.1313** vs the **0.132** bar and an unchanged fixed-**252** stop ([EXP-20260803-01](#exp-20260803-01-scheduled-sampling-now-actually-running-does-not-improve-free-run-on-r2)) |
| **Local artifact gap** | July 16–24 AR checkpoints, subset indices, and logs are **absent from this clone**; a 50t/50v rebuild reached comparable `val_loss` but **10× worse** val F1 and the **opposite** free-run pathology ([EXP-20260724-04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)) — local rebuilds do **not** currently stand in for the logged runs |
| **AR termination stability** | **Open** — same recipe gives early EOS at 200t ([EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode)) and never-EOS at 50t ([EXP-20260724-04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)); termination is unstable, not one-directionally biased |
| **AR tide class weights (champion recipe)** | **Deferred** — drop-in on v8 failed free-run ([EXP-20260703-01](#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe)); champion stays `none`; co-tuned recipe revisit [NOTE-20260703-01](DISCUSSION_NOTES.md#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred) |
| **AR training throughput / validation** | **Improved** — val aggregation + dynamic buckets (**18.6%** on smoke); single-song overfit batch cache default-on (~**9×** steady epoch on tide) ([EXP-20260716-01](#exp-20260716-01-ar-validation-aggregation--dynamic-length-bucketing), [EXP-20260716-02](#exp-20260716-02-ar-corrected-mask-tide-overfit-regression)) |

**Recommended when resuming onset work:**

- **First, for any track:** report the audio-blind floor beside every number ([EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)). A Hungarian F1 on dense charts is unreadable without it.
- **Second, for any track:** run the audio-corruption ablation before believing any score ([EXP-20260804-05](#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer)). A single-song overfit **cannot** detect an audio-blind model — the tide gate passes with the audio reversed.
- **Track B (AR):** the pointer head is the bug, not the data. Replace `Dense(max_patches)` with a content-based pointer against encoder memory before adding rows or tuning anything else.
- **Track A (literature):** Recreate DDCL on Dataset A, then ITGPT on B, before any incremental claim. `final_data` is transfer only — not the comparison set.
- **Event track (optional):** Continue K-query probes on `data/v2` in parallel if not blocking Track A.

---

## Experiment index

Newest first. Stage tags: `pre` | `model` | `post` | `metric` | `train`. Discussion context: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

| ID | Stage tag | Question | Status | One-line outcome |
| -- | --------- | -------- | ------ | ---------------- |
| EXP-20260816-02 | `train` + `metric` | Does full-split DDCL 48-slot ConvLSTM placement on Dataset A beat the beat-shuffle null? | **Partial** | Best F1@0.5 **0.601** vs null **0.397** (skill **+0.204**); max-F1 **0.634**; not vs ITGPT **0.70 / 0.76** |
| EXP-20260816-01 | `pre` + `model` + `metric` + `train` | Can we run DDCL 48-slot ConvLSTM placement on Dataset A with `M-slot48` + null? | **Partial** | Smoke 1-ep F1@0.5 **0.032** vs null **0.491**; pipeline works, no skill |
| EXP-20260815-07 | `train` + `metric` | Does a 50/100-row `final_data` dense MERT BiLSTM beat the ioi-shuffle floor? | **Partial** | Event F1 **0.666** vs null **0.294**; `timing_match` **0.0047** vs **0.0081** |
| EXP-20260815-06 | `metric` | Does keeping `n_gt` highest-salience peaks recover `timing_match`? | **Not supported** | Matched-count **0.0154** ≈ null **0.0151**; raw **0.0024** |
| EXP-20260815-05 | `metric` | Do accepted DDC peaks have `timing_match` skill beside F-score_m **0.734**? | **Not supported** | timing_match **0.0024** vs null **0.0109**; F-score_m unchanged **0.734** |
| EXP-20260815-04 | `train` + `metric` | Do best-`val_loss` weights close the ~0.03 F1 gap to paper? | **Not supported** | Best **0.650** / **0.735** ≈ last **0.652** / **0.734**; gap remains |
| EXP-20260815-03 | `train` + `metric` | Does 128-ep C-LSTM placement close the ~0.09 F1 gap to paper? | **Partial** | Val **0.652** / **0.734** vs paper **0.681** / **0.756** (~**96%** / **97%**); vs 8-ep **+0.058** / **+0.067** |
| EXP-20260815-02 | `train` + `metric` | Does an 8-ep DDC C-LSTM on original Fraxtil match paper F-score_c / F-score_m? | **Partial** | Val **0.594** / **0.667** vs paper **0.681** / **0.756**; skill **+0.397** vs ioi-shuffle |
| EXP-20260815-01 | `pre` | Can we ingest original Fraxtil (DDC Dataset A) with measured song/chart counts? | **Supported** | **90** songs / **463** rows (450 standard + 13 edit); 81/9 train/val seed 42 |
| EXP-20260807-20 | `train` + `metric` | Does soft-α anneal (0.5→0) teach content-gap localization that survives α=0? | **Not supported** | α=0 timing **0.0016**; F1 skill **−0.44** — same collapse |
| EXP-20260807-19 | `metric` | Do content-gap soft-α weights keep skill with α=0 at decode? | **Supported** (collapse) | α=0 timing **0.0019** vs matched **0.0968**; F1 skill **−0.44** |
| EXP-20260807-18 | `train` + `metric` | Does soft Δ prior (α=0.5) on content-gap fix wrong-far overshoot on R2? | **Partial** | Offline timing **0.0968** (~**13×** vs **0.0075**); F1 skill **−0.056**; α=0 collapses |
| EXP-20260807-17 | `metric` | On content-gap R2, is leftover error stickiness, residual, or wrong-far Δ soup? | **Supported** | Val **92%** wrong_far; pred Δ p50 **92** vs GT **2**; H/Huni **0.93** |
| EXP-20260807-16 | `train` + `metric` | Does R2 content gap beat ptrloss timing **~0.0035** without α / hard R? | **Supported** | Offline timing **0.0075** (~**2.1×**); F1 skill **−0.43** |
| EXP-20260807-15 | `model` + `train` + `metric` | Does content gap (`q·k(memory[prev+Δ])`) pass tide timing ≈1 **and** audio grounding? | **Supported** | Teacher **0.987**; ablation **PASS** (shuffle/zeros collapse) |
| EXP-20260807-14 | `train` + `metric` | Does Dense gap+residual pass tide timing ≈1 **and** audio grounding? | **Not supported** | Teacher **632/634**; ablation same_pred **1.0** under zeros — audio-blind |
| EXP-20260807-13 | `train` + `metric` | Does soft distance-from-prev prior (no hard cutoff) beat full CE without starving long gaps? | **Partial** | α=0.5 timing **0.105** / F1 **0.279** vs ptrloss **0.0035**; α=0 eval still collapses |
| EXP-20260807-12 | `metric` | Do R=4 weights keep skill if the hard window is removed at eval? | **Supported** (collapse) | Unmasked timing **0.0016** vs masked **0.156**; patch-acc **0.25%** — crutch-dependent |
| EXP-20260807-11 | `metric` | At R=4, is leftover error mid-window soup, residual, or stickiness? | **Supported** | Val **31%** at_prev + **30%** near_miss≤2; H/Huni **0.89**; resid secondary; **R=2 starves 20%** gaps |
| EXP-20260807-10 | `train` + `metric` | Does shrinking prev-local R 8→4 raise val timing / patch-acc / F1? | **Supported** | Offline timing **0.156** vs R8 **0.070**; patch-acc **26.9%**; F1 **0.251** (skill **−0.084**) |
| EXP-20260807-09 | `metric` | On R=8, is the window still diffuse, and is residual the binding leftover? | **Supported** (soup yes; residual secondary) | Val H/Huni **0.96**; **83%** patch_wrong vs **10%** patch_ok_timing_wrong; resid p50 **28 ms** when patch_ok |
| EXP-20260807-08 | `train` + `metric` | Does shrinking prev-local R 32→8 raise val timing / patch-acc? | **Supported** | Offline timing **0.0697** vs R32 **0.0228**; patch-acc **16.6%** vs **5.5%**; timing skill **+0.062** |
| EXP-20260807-07 | `metric` | Where inside `[prev, prev+32]` do v3 errors land? | **Supported** | Val: GT offset p50 **2**, pred offset p50 **15**; H/Huni **0.95**; **63%** wrong-far-in-window — diffuse mid-window vs left-skewed GT |
| EXP-20260807-06 | `train` + `metric` | Does decode-consistent prev-relative local CE (r=32) beat ptrloss ep2? | **Supported** | Offline timing **0.0228** vs **0.0035**; patch-acc **~5.5%**; timing skill **+0.0145** (F1 skill still **−0.40**) |
| EXP-20260807-05 | `train` + `metric` | Does clean local CE (r=32, no STE) beat ptrloss ep2 on val timing / patch-acc? | **Not supported** | Offline **0.0025** &lt; ep2 **0.0035**; **88%** preds outside ±32 window — train/infer mismatch |
| EXP-20260807-04 | `metric` | Is far_ahead diffuse mono-suffix mass or confident wrong peaks? | **Supported** | Ep2 val: H/Huni **0.92**, top-1 **0.016**, n_allowed ~**800** — diffuse; ep31 val top-1 **0.20** peaked wrong → local CE next |
| EXP-20260807-03 | `metric` | At the selected ep2 weights, is the failure train≫val or both-weak? | **Supported** | Both weak (train timing **0.0039** ≈ val **0.0035**); late 11% train patch-acc is memorization — scale-up declined; next is fixed-R2 far_ahead entropy |
| EXP-20260807-02 | `train` + `metric` | Does ckpt on `val_pointer_loss` beat the patch-acc-selected QK-LN ep31 on val timing / NLL? | **Partial** | Picks ep **2** (NLL **5.97**); offline timing **0.0035** **worse** than ep31 **0.0070** — selection fixed, transfer not |
| EXP-20260807-01 | `metric` + `train` | Does QK-LN train-split ablation pass, and does switching ckpt to `val_pointer_loss` close the selection defect? | **Supported** (ablation + selection) | Train gate **PASS** (timing **0.043→0.001** shuffle; NLL **3.64**); configs now `val_pointer_loss`; retrain still needed to judge transfer |
| EXP-20260806-07 | `model` + `train` + `metric` | Does pointer QK LayerNorm raise R2 patch-acc / beat encode-then-PE short probe? | **Partial** | Tide gate **PASS** (timing **0.99**). R2: val patch-acc **0.0185**, offline timing **0.0070** ≈ ctx-pefree **0.0069**; ablation pointer **PASS** after floor-fix |
| EXP-20260806-06 | `model` | Does content-only decoder cross (no PE residual) help without breaking tide? | **Not supported** | Tide peak timing **0.67** vs mix **~0.94**; default stays **False** |
| EXP-20260806-05 | `train` + `metric` | Does full 500-ep R2 with encode-then-PE beat prior full hard R2 **0.0085**? | **Supported** | Best val **0.00945 @ ep 144**; offline **334/35439 = 0.0094**; F1 **0.046** (was **0.021**) |
| EXP-20260806-04 | `model` + `train` + `metric` | Does encode-then-PE (contextualized pe-free keys) beat hard-time R2 short probe? | **Supported** | Val/offline timing **0.0069** vs hard **0.005**; tide gate **PASS**; val ptr NLL **6.14** &lt; uniform |
| EXP-20260806-03 | `train` + `metric` | Does STE + full CE **without** correct-patch mask beat hard-time **0.005**? | **Not supported** | ES @ ep **26**, restore ep **16**; best val **0.0041**; `time_loss` ~**21** (active) but still below hard |
| EXP-20260806-02 | `train` + `metric` | Does STE + correct-patch `λ_time` + **full** CE beat hard-time / local-CE probes? | **Not supported** | ES @ ep **18**, restore ep **8**; best val timing **0.004** &lt; hard **0.005**; `time_loss` still ~**0.02** |
| EXP-20260806-01 | `model` + `train` + `metric` | Do STE + local CE + correct-patch `λ_time` + offline monotonic fix raise R2 timing / close train–offline gap? | **Partial** | Offline mono fix: **28→305**/35439 (**0.00079→0.0086**). Localizing probe best val **0.0037** &lt; hard **0.005** — not a beat |
| EXP-20260805-07 | `train` + `metric` | Does tide `lambda_residual: 30` + hard time beat lam5 hard-time R2 on val timing? | **Not supported** | 50 ep probe: ES @ ep **26**, restore ep **16**; best val timing **0.0065** (lam5 hard probe **0.005**, full lam5 R2 **0.0085**) — modest vs short probe, below full R2 |
| EXP-20260805-06 | `train` + `metric` | Does full R2 with hard pointer time beat soft-time R2 on val timing? | **Partial** | ES @ ep **146**, restore ep **96**; best val timing **0.0085** (soft R2 **0.0014**); train timing **0.054** @ best — still below null skill |
| EXP-20260805-05 | `train` + `metric` | Does hard-argmax pointer time (`use_soft_pointer_time: false`) train multi-song timing where soft time fails? | **Partial** | 50 ep probe: best val timing **0.005 @ ep 7** vs soft/no-mono **~0.0005** (~**10×**); still at null floor — first recipe axis with measurable movement |
| EXP-20260805-04 | `train` + `metric` | Is monotonic pointer blocking multi-song timing learning? | **Not supported** | 50 ep probe `monotonic_pointer: false`; ES @ ep **17**, restore ep **7**; best val timing **0.00054** — same floor as monotonic R2 |
| EXP-20260805-03 | `train` + `metric` | Does the fixed decoder-audio stack give R2 content-pointer val timing above the noise floor? | **Not supported** | ES @ ep **54**, restore ep **4**; best `val_timing_match_teacher` **0.0014**; train timing also ~**10⁻³** at best/final — tokens overfit, timing does not |
| EXP-20260805-02 | `model` + `train` + `metric` | Do the three decoder-audio fixes (mask default, query/zeros gate, PE-free cross + monotonic + soft time) make tide content-pointer pass a non-keys-only gate? | **Supported** | Tide timing **0.94**; zeros `query_cosine` **0.42** / tok **0.12** (was ~**1.0** / unchanged); gate **PASS** (pointer+token+query). Shuffle query can stay ~1 — zeros is the decoder probe |
| EXP-20260805-01 | `model` + `metric` | Is content-pointer “audio grounding” coming from the decoder, or only from `pointer_key(memory)`? | **Root cause** | Decoder/`pointer_query` cosine ≈ **1.0** under shuffle (R2 + tide); pointer collapse is **keys-only**. Tide content-pointer used **inverted** attention masks (`legacy_inverted_attention_masks` default True). Prior train pointer-gate PASS was a false positive for decoder grounding |
| EXP-20260804-09 | `train` + `metric` | Was early stopping @ ep 16 hiding val improvement? Does training to 120 ep without ES beat the ep-16 val timing peak? | **Not supported** | Best val timing still @ ep **16** (**0.0022**); ep **120** val **0.0015** / train **0.511**; exported checkpoint **bit-identical** offline to ES run — val never learned past noise |
| EXP-20260804-08 | `metric` + `post` | Is R2 content-pointer val failure a wiring bug or generalization? Does audio ablation differ on train vs val? | **Supported** | Train timing mean **0.024** vs val **0.002**; train ablation matched **0.036** → shuffle **0.001** (pointer gate **PASS**); val ablation all ~**0.001** (gate **FAIL** at floor); ES restore @ ep **16** is correct — no epoch checkpoints beyond `best.keras` |
| EXP-20260804-07 | `train` + `metric` | Does R2 on `pointer_head: content` show positive skill over null on the frozen val set? | **Not supported** | Teacher **78/35439** (`timing_match` **0.0022**); F1 **0.0598**; skill **−0.36** vs null; ablation gate **FAIL** on val (shuffle ≈ matched); free-run preflight would skip |
| EXP-20260804-06 | `model` + `train` | Does a content-based pointer restore audio grounding while still passing the tide gate? | **Supported** | Gate **0.9921** on real audio; collapses to F1 **0.18–0.19** / `timing_match` **0.0016** / `patch_wrong` **99.8%** under reverse+shuffle. Old head held **0.9984** on the same corruption |
| EXP-20260804-05 | `model` + `metric` | Does the AR pointer use the audio at all? | **Root cause** | **No.** Zeroing all features leaves R2 F1 **0.1886 → 0.1885** and the predicted patch unchanged **99.96%** of steps; tide champion holds **0.9984** with reversed/shuffled/cross-song audio. Pointer head is `Dense(max_patches)` over decoder state — absolute-index classification, not content-based pointing |
| EXP-20260804-04 | `metric` + `post` | Does the in-harness null floor reproduce the offline finding on a real checkpoint end-to-end? | **Supported** | R2 re-score: teacher F1 **0.1886** vs strongest null **0.2696** → skill **−0.1109**; free-run skill **−0.0150**; both gates **FAIL** |
| EXP-20260804-03 | `metric` | What does an audio-blind predictor score on the frozen val set at matched onset count? | **Supported** | Null F1 **0.225–0.336** @ r=1.0 vs best teacher **0.227** / champion free-run **0.263**: **no rung clears chance**; `timing_match` floor ≈ **0** |
| EXP-20260804-02 | `metric` + `post` | Is R3 free-run collapse early-EOS (recoverable by length force)? | **Supported** (superseded) | Bare R3 5-song: free **0.0007**@15 → **0.200** with `min_onset_tokens=200`; but **0.200 < 0.239** null at that count ([EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke)) |
| EXP-20260804-01 | `train` + `metric` | Does R3 (200t) + onset_density fix early-EOS@15 and beat R2 free-run? | **Not supported** | Teacher **0.206** (≥ R3 **0.199**); free-run **0.034** vs R3 **0.003** / R2+density **0.263**; pred/GT **0.16**; **50/50** EOS @ ~**115** preds/song avg |
| EXP-20260803-03 | `model` + `train` + `metric` | Does `onset_density` beat meter density on R2 free-run? | **Supported** | Free-run **0.263** vs meter **0.234**; teacher **0.227**; **pred/GT 0.90**; all **50/50** EOS |
| EXP-20260803-02 | `model` + `train` + `metric` | Does meter density conditioning on R2 break the fixed-252 stop and beat free-run 0.132? | **Supported** | Free-run **0.234** vs **0.132**; teacher **0.227**; **pred/GT 0.82**; **6** stop-length modes (**8/50** @ **252**) |
| EXP-20260803-01 | `train` + `metric` | With SS actually running, does it beat the R2 free-run bar of 0.132? | **Not supported** | Free-run **0.1313** vs **0.1319**; teacher **0.2235** vs **0.2266**; all 50 songs still stop at exactly **252** onsets |
| EXP-20260802-05 | `train` | Does scheduled sampling on the R2 recipe improve free-run? | **Void (defect)** | Run is **bit-identical** to R2 (0.2266 @ ep 470, ep500 0.2236) — SS branch is compiled out of the traced `train_step`; feature has never been active |
| EXP-20260802-04 | `metric` | R2 vs R3 free-run on the same frozen val? | **Supported** | R2 free-run **0.132** ≫ R3 **0.003**; R3 early-EOS @ **15** toks; SS on R2 |
| EXP-20260802-03 | `metric` | Does R2 free-run hold on the frozen val set? | **Partial** | Teacher F1 **0.227**; free-run F1 **0.132**; all 50 songs EOS at **252** onsets (pred/GT **0.36**) |
| EXP-20260802-02 | `train` + `metric` | Ladder R3: does 200 train rows beat R2's 0.227 on the frozen val set? | **Partial** | Best val F1 **0.1991 @ ep 361**; ES @ ep **411**; **below** R2 (**0.227**) — scale not monotonic |
| EXP-20260802-01 | `train` + `metric` | Ladder R2 rerun: does 50 train rows beat R1's 0.178 on the frozen val set? | **Supported** | Best val F1 **0.2266 @ ep 470**; beats R1 (**0.178**); finished 500 ep under ~23 GiB WSL ceiling |
| EXP-20260726-01 | `train` | Ladder R2: does 50 train rows beat R1's 0.178? | **Aborted** | WSL VM terminated at ep **152**; partial curve ~2.2× ahead of R1 at matched epochs; suspected guest OOM (unconfirmed) |
| EXP-20260725-02 | `train` + `metric` | Ladder R1: what does 10 train rows score on the frozen val set? | **Supported** | val F1 **0.178** @ ep 497 (still climbing); `val_loss` best @ ep 104 is **32×** worse on F1; found inert checkpoint/ES monitor |
| EXP-20260725-01 | `pre` + `metric` | Ladder R0: are local MERT features and the tide champion still good? | **Partial** | Features **bit-identical**; champion path decodes **627/634** teacher — graduated v8 weights were **overwritten** on 2026-07-02 |
| EXP-20260724-04 | `train` + `metric` | Does a local 50t/50v rebuild reproduce the early-EOS free-run collapse? | **Fail to reproduce** | Opposite failure: **0/10** songs emit EOS, all hit the **2048** cap; EOS prob max **0.0087**; teacher ordered **14/6544** |
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

### EXP-20260816-02: DDCL 48-slot placement full-split on Dataset A

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-16 23:41:37 |
| **Track** | `train` + `metric` (DDCL placement, `omalley2025ddcl`) |
| **Question** | On the Dataset A standard split, does an 8-ep ConvLSTM 48-slot placement run beat a beat-shuffle null on `M-slot48`? |
| **Setup** | [`ddcl_placement_fraxtil.json`](../../configs/ddc/ddcl_placement_fraxtil.json): 81/9 songs, 405/45 charts, memlen **15**, lstm **200**, batch **8**, 8 ep × 100/20 steps, Adam 1e-4, binary focal loss, seed **42**. Two prior launches died WSL OOM (exit **9**, ~14 GiB RSS) while pre-stacking causal windows; retry used on-demand `window_at_beat`. Train `logs/ddcl_placement_fraxtil.log`. TB `callbacks/ddc/ddcl_placement/logs` (port **6007**). ~**10.2** min WSL GPU after load |
| **Train** | Best `val_loss` **0.0107** @ ep **6**; last ep-8 `val_loss` **0.0115**. Saved `models_wsl/ddc/ddcl_placement_fraxtil/best.keras` and `ddcl_placement_fraxtil.keras` |
| **Val (45 charts, 17523 beats)** | **Best weights:** F1@0.5 **0.601** (TP **10388** / FP **6325** / FN **7480**); max-F1 **0.634** @ thr **0.40**. **Last weights:** F1@0.5 **0.221** (TP **2303** / FP **683** / FN **15565**); max-F1 **0.656** @ thr **0.35**. Beat-shuffle null F1@0.5 **0.397**. JSON `eval_val_slot48.json` / `eval_val_slot48_best.json` |
| **vs null** | Best skill F1@0.5 **+0.204**. Last skill **−0.176** (under-predicts at 0.5; threshold sweep recovers max-F1) |
| **vs published** | **Not comparable** — ITGPT Table 2 DDCL **0.70 / 0.76** is expanded Fraxtil (`D-frax-exp`), not this 81/9 split. 100 steps/ep is a thin pass over 405 charts |
| **Conclusion** | **Partial.** Dataset A now has a real `M-slot48` number with audio-grounded skill on best-`val_loss` weights. Do not cite **0.601** against **0.70**. Next is DDCL selection, not more DDC eval and not `final_data` |

### EXP-20260816-01: DDCL 48-slot placement smoke on Dataset A

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-16 18:33:28 |
| **Track** | `pre` + `model` + `metric` + `train` (DDCL placement, `omalley2025ddcl`) |
| **Question** | Can we port DDCL beat-grid PRE (32 frames/beat, 48 slots) and the branched ConvLSTM onto original Fraxtil and emit `M-slot48` with a null floor? |
| **Setup** | [`ddcl_placement_fraxtil_smoke.json`](../../configs/ddc/ddcl_placement_fraxtil_smoke.json): 2 train / 1 val songs, memlen **3**, 1 epoch, 4 steps, Adam 1e-4, binary focal loss. Manifest `training_index_standard.json`. Mel reused from DDC `*.ddc_mel.npy` (`donahue2017ddc`). Upstream: [miguelomalley/DDCL](https://github.com/miguelomalley/DDCL) commit `5b1375c`. Train `logs/ddcl_placement_smoke.log`. TB `http://localhost:6007/`. ~**33** s WSL GPU after load |
| **Val (5 charts, 1420 beats)** | F1@0.5 **0.032** (TP **553** / FP **32555** / FN **1043**); max-F1 **0.046** @ thr **0.05**. Beat-shuffle null F1@0.5 **0.491**; skill **−0.459**. JSON `models_wsl/ddc/ddcl_placement_smoke/eval_val_slot48.json` |
| **vs published** | **Not comparable** — 1-ep smoke on 1 val song. ITGPT Table 2 DDCL **0.70 / 0.76** is expanded Fraxtil, not this split |
| **Conclusion** | **Partial.** PRE / ConvLSTM / `M-slot48` run on Dataset A with citations. Untrained sigmoids flood positives, so F1 sits below the beat-shuffle floor. Next is a full-split train, not selection |

### EXP-20260815-07: `final_data` dense 50t/100v MERT BiLSTM smoke

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 18:43:16 |
| **Track** | `train` + `metric` (dense `final_data` scoreboard) |
| **Question** | Does the first GPU dense MERT BiLSTM on a 50/100-row `final_data` subset clear an ioi-shuffle floor on val event F1 and `timing_match`? |
| **Setup** | [`final_data_mert_bilstm_scoreboard_50t_100v.json`](../../configs/dense/final_data_mert_bilstm_scoreboard_50t_100v.json): 256u BiLSTM, MERT, seed **42**, 200 ep, ES patience **25** on `val_onset_f1_score`. Manifest `training_index_scoreboard_50t_100v.json` (**49 / 70** songs, **50 / 100** rows). MERT: **17** new + **102** cached. Train `logs/dense_scoreboard_50t_100v_train.log`. TB `http://localhost:6006/`. ~**3.6** h WSL GPU including post-hoc sweep |
| **Train** | ES @ ep **61** (restored monitor-best ep **36**, `val_onset_f1` **0.7735**). Post-hoc peak-pick sweep exported ep-**7** ckpt (`VAL_ONSET_F1_SCORE-0.49538`) @ thr **0.25** — best micro event F1 **0.666**. Saved `models_wsl/final_data_dense_bilstm_scoreboard_50t_100v/stepcovnet_ONSET-final_data_dense_bilstm_scoreboard_50t_100v.keras` |
| **Val (100 rows, thr 0.25)** | micro event F1 **0.666**; mean **0.648**; TP **52008** / FP **30617** / FN **21612**. `timing_match` **390 / 82722 = 0.0047**. JSON `eval_val_event_f1.json`. Log `logs/dense_scoreboard_50t_100v_eval.log` |
| **vs ioi-shuffle** | Event F1 null **0.294**, skill **+0.371**. `timing_match` null **0.0081**, skill **−0.0034** |
| **vs `data/v2` 0.686** | **Not comparable** — different corpus and split (EXP-20260610-03 was `data/v2` @ thr 0.30) |
| **Conclusion** | **Partial.** Pipeline works and event F1 is audio-grounded. Ordered `timing_match` is still at the floor (over-count: **82722** pred vs **73620** ref), same pattern as DDC peaks. Do not treat **0.666** as beating **0.686**. Full `final_data` train only if asked |

### EXP-20260815-06: Matched-count does not recover DDC timing_match

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 14:31:26 |
| **Track** | `metric` (literature DDC placement) |
| **Question** | On the frozen 128-ep C-LSTM, does keeping the `n_gt` highest-salience Hamming peaks recover `timing_match` vs raw **0.0024**? |
| **Setup** | No retrain. Same ckpt as [EXP-20260815-05](#exp-20260815-05-accepted-ddc-peaks-have-no-ordered-timing_match-skill). Diagnostic `timing_match_matched_count` drops extras to `n_gt` by salience; `M-ddc-20ms` F-scores unchanged. Ioi-shuffle floor at `n_gt`. JSON `models_wsl/ddc/placement_fraxtil_128ep/eval_val_ddc_timing.json`. Log `logs/ddc_placement_matched_count.log`. ~**2.4** min WSL GPU |
| **Val `M-ddc-20ms`** | F-score_c **0.652**; F-score_m **0.734**; null **0.292**; skill **+0.442** — matches EXP-03/05 |
| **Raw `timing_match`** | **52 / 21816 = 0.0024**; null **0.0109**; skill **−0.0085** |
| **Matched-count `timing_match`** | **275 / 17868 = 0.0154**; `n_pred` kept **17533** (some charts under-predict); ioi-shuffle at `n_gt` **0.0151**; skill **+0.0003** |
| **Conclusion** | **Not supported.** Dropping extras lifts **0.0024 → 0.0154** but that is still the ordered floor. Leftover is not “good times, extra peaks.” Do not retune Hamming / thresholds. Stay on onset/placement; no step selection |

### EXP-20260815-05: Accepted DDC peaks have no ordered timing_match skill

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 08:33:54 |
| **Track** | `metric` (literature DDC placement) |
| **Question** | On the frozen 128-ep C-LSTM, does `timing_match` @ 20 ms show skill beside `M-ddc-20ms` F-score_m **0.734**? |
| **Setup** | No retrain. [`eval_ddc_placement.py`](../../scripts/eval_ddc_placement.py) on EXP-03 last ckpt `models_wsl/ddc/placement_fraxtil_128ep/ddc_placement_fraxtil_128ep.keras` ([`placement_fraxtil_128ep.json`](../../configs/ddc/placement_fraxtil_128ep.json)). Same Hamming peaks as F-score; ordered match vs `chart.gt_times`. Ioi-shuffle floor at the same peak counts. JSON `models_wsl/ddc/placement_fraxtil_128ep/eval_val_ddc_timing.json`. Log `logs/ddc_placement_timing_match.log`. ~**2.4** min WSL GPU |
| **Val `M-ddc-20ms`** | F-score_c **0.652**; F-score_m **0.734**; null **0.292**; skill **+0.442**. TP **14573** / FP **7243** / FN **3295**. Matches [EXP-20260815-03](#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap) |
| **Val `timing_match` @ 20 ms** | **52 / 21816 = 0.0024**; ioi-shuffle null **0.0109**; skill **−0.0085**. `n_pred` **21816** vs `n_ref` **17868** |
| **Why the columns diverge** | Greedy ±20 ms F1 can pair any nearby peak to a step. Ordered match requires `pred[i] ≈ ref[i]`; **7243** extra peaks shift ranks, so F-score_m **0.734** is not ordered skill |
| **Conclusion** | **Not supported.** Keep both columns; do not mix. Do not retune Hamming / thresholds to chase `timing_match` — that would change the literature `M-ddc-20ms` table. Stay on onset/placement; no step selection |

### EXP-20260815-04: Best-val DDC weights do not close the paper gap

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 04:08:18 |
| **Track** | `train` + `metric` (literature DDC placement) |
| **Question** | Does scoring `val_loss`-best weights (ep **112**) beat last-epoch F-score_c / F-score_m and close the ~0.03 gap to paper **0.681** / **0.756**? |
| **Setup** | Same recipe as [EXP-20260815-03](#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap) ([`placement_fraxtil_128ep.json`](../../configs/ddc/placement_fraxtil_128ep.json), seed **42**, 128 ep). New dir `models_wsl/ddc/placement_fraxtil_128ep_ckpt/` so EXP-03 last keras is kept. `ModelCheckpoint(monitor=val_loss)`. Train `logs/ddc_placement_fraxtil_128ep_ckpt.log`. ~**47** min WSL GPU |
| **Train** | Best ckpt ep **112** val_loss **0.0524**; last ep **128** val_loss **0.0588** (same as EXP-03) |
| **Val last** | F-score_c **0.652**; F-score_m **0.734**; null **0.294**; skill **+0.440**. `eval_val_ddc.json` — matches EXP-03 |
| **Val best (ep 112)** | F-score_c **0.650**; F-score_m **0.735**; null **0.303**; skill **+0.432**. `eval_val_ddc_best.json` |
| **vs last / paper** | Best vs last: C **−0.002**, M **+0.001**. vs paper still **−0.031** / **−0.021** |
| **Conclusion** | **Not supported.** Last-vs-best is not the leftover hole. Do not retrain for checkpointing. Residual ~0.03 is PRE/eval (batch 32 vs 256, Librosa, chunked LSTM, split). Stay on onset/placement; no step selection |

### EXP-20260815-03: DDC 128-ep placement closes most of the paper gap

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 02:40:12 |
| **Track** | `train` + `metric` (literature DDC placement) |
| **Question** | Does training the same C-LSTM to the paper epoch budget (128) recover F-score_c **0.681** / F-score_m **0.756**? |
| **Setup** | [`configs/ddc/placement_fraxtil_128ep.json`](../../configs/ddc/placement_fraxtil_128ep.json): same recipe as [EXP-20260815-02](#exp-20260815-02-ddc-c-lstm-placement-8-ep-on-dataset-a--below-paper-above-null) (batch **32**, nunroll **100**, 2×200 LSTM, SGD 0.1 / clipnorm 5, seed **42**), **128** ep × 200 steps. Ckpt `models_wsl/ddc/placement_fraxtil_128ep/ddc_placement_fraxtil_128ep.keras` (last epoch). Train `logs/ddc_placement_fraxtil_128ep.log`. ~**42** min WSL GPU |
| **Train** | ep1 **0.163** / val **0.148** (matches 8-ep); ep8 **0.090** / **0.091**; ep48 **0.072** / **0.058**; ep112 best val **0.066** / **0.052**; ep128 saved **0.066** / **0.059** |
| **Val `M-ddc-20ms` (9 songs / 45 charts)** | F-score_c **0.652**; F-score_m **0.734**; ioi-shuffle null **0.292**; skill **+0.442**. TP **14573** / FP **7243** / FN **3295**. Thr beginner/medium **0.15**, easy/hard/challenge **0.20**. `models_wsl/ddc/placement_fraxtil_128ep/eval_val_ddc.json` |
| **vs 8-ep** | F-score_c **+0.058** (**0.594→0.652**); F-score_m **+0.067** (**0.667→0.734**). FP dropped **11469→7243** |
| **vs paper** | Gap **−0.029** / **−0.022** (~**96%** / **97%** of **0.681** / **0.756**). Still val-tuned on val; last weights, not best-val ep **112** |
| **Conclusion** | **Partial.** Epoch budget was the main 8-ep shortfall. Residual ~0.03 is in the documented-deviation band (batch 32 vs 256, Librosa, chunked eval LSTM, split, last vs best ckpt). **Routing (same day):** stay on onset/placement — do not start step selection until asked |

### EXP-20260815-02: DDC C-LSTM placement 8-ep on Dataset A — below paper, above null

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 01:30:53 |
| **Track** | `train` + `metric` (literature DDC placement) |
| **Question** | Does a DDC-faithful C-LSTM on original Fraxtil (`donahue2017ddc`) recover paper F-score_c **0.681** / F-score_m **0.756** (`M-ddc-20ms`)? |
| **Setup** | [`configs/ddc/placement_fraxtil.json`](../../configs/ddc/placement_fraxtil.json); `data/literature_fraxtil_orig/training_index_standard.json` (**81 / 9** songs, **405 / 45** charts, `stratified_song_v1+standard_v1` seed **42**). SGD lr **0.1** clipnorm **5**, 2×200 LSTM, dropout **0.5**. Ckpt `models_wsl/ddc/placement_fraxtil/ddc_placement_fraxtil.keras`. Train `logs/ddc_placement_fraxtil.log`. Metric `M-ddc-20ms`. Cite `schluter2014onset`, `hamel2012multiscale`. |
| **Smoke** | [`placement_fraxtil_smoke.json`](../../configs/ddc/placement_fraxtil_smoke.json): 2 train / 1 val song, 1 ep × 4 steps, 1×64 LSTM. Pipeline **PASS** after reshape CNN (`DdcPerFrameCNN`; TimeDistributed froze T=32). Eval `models_wsl/ddc/placement_smoke/eval_val_ddc.json`: F-score_c **0.121** / F-score_m **0.128** vs null **0.184** (skill **−0.056**, 5 charts) — not a score, only an e2e check. `logs/ddc_placement_smoke.log` |
| **Full train (8 ep × 200 steps, batch 32, nunroll 100)** | ep1 loss **0.163** / val **0.148**; ep5 **0.101** / **0.090**; ep7 **0.093** / **0.082**; ep8 (saved) **0.090** / **0.091**. ~**14** min WSL GPU |
| **Val `M-ddc-20ms` (9 songs / 45 charts)** | F-score_c **0.594**; F-score_m **0.667**; ioi-shuffle null F-score_m **0.270**; skill **+0.397**. TP **14684** / FP **11469** / FN **3184**. Per-diff thr beginner/easy/medium/challenge **0.35**, hard **0.30**. `models_wsl/ddc/placement_fraxtil/eval_val_ddc.json` |
| **vs paper** | C-LSTM Dataset A test: **0.681** / **0.756**. Gap **−0.087** / **−0.089** (~**87%** / **88%** of paper). Thresholds tuned on this val set (optimistic vs DDC’s unpublished test) |
| **Deviations** | Librosa STFT not Essentia; per-song z-score not dataset-wide moments; eval LSTM state reset every **256**-frame chunk (not carried BPTT); batch **32** not **256**; **8** ep not **128**; split is seed-42 90/10 not DDC 80/10/10 |
| **Conclusion** | **Partial.** Placement is audio-grounded (skill **+0.397**) and in the paper’s F1 neighborhood, not a match. Val_loss still improved through ep **7**, so the gap is consistent with under-training vs a PRE/POST bug. Next: longer placement train before DDC selection |

### EXP-20260815-01: Original Fraxtil Dataset A ingested (90 songs)

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 00:28:51 |
| **Track** | `pre` (literature Dataset A) |
| **Question** | Can the three DDC Fraxtil SM5 packs (`donahue2017ddc`) be prepared in-repo with a documented song-grouped split? |
| **Setup** | Raw `data/raw_literature/`; prep `data/literature_fraxtil_orig/`. Arrow Arrangements + Beast Beats from official SM5 zips. Tsunamix III zip **404** on fra.xtil.net; reconstructed from unpacked `.sm`+`.ogg` per song (50/50). `logs/literature_fraxtil_dryrun.log`, `logs/literature_fraxtil_prep.log`, `logs/literature_fraxtil_index.log` |
| **Prep** | 90 packs exported, 0 errors, 0 skipped. Charts: **463** = **450** standard (90×5) + **13** `edit` |
| **Per pack** | Arrow Arrangements 20 songs / 104 charts (4 edit); Beast Beats 20 / 104 (4 edit); Tsunamix III 50 / 255 (5 edit) |
| **Split** | `stratified_song_v1` seed **42**, val_fraction **0.1**: **81 / 9** songs, **417 / 46** rows; 0 song overlap. Not DDC’s unpublished 80/10/10 IDs |
| **vs DDC table** | Paper: 90 songs / 450 charts. Extra 13 are S-Edit rows our exporter keeps |
| **Conclusion** | **Supported.** Dataset A is on disk. Next is DDC placement recreation with `M-ddc-20ms`, not `final_data` training |

### EXP-20260807-20: Content-gap soft-α anneal still collapses at α=0

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 22:58:58 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does holding soft Δ α=0.5 then linearly annealing to 0 teach content-gap weights that keep skill at α=0 decode? |
| **Setup** | [`configs/ar/ladder_r2_gap_content_soft_anneal_probe.json`](../../configs/ar/ladder_r2_gap_content_soft_anneal_probe.json): hold **10** / anneal **20** → α=0 by ep **30**; ES `start_from_epoch=30`, restore ep **31**. Train `logs/r2_gap_content_soft_anneal_train.log`. α=0 eval [`ladder_r2_gap_content_soft_anneal_a0_eval.json`](../../configs/ar/ladder_r2_gap_content_soft_anneal_a0_eval.json) → `logs/r2_gap_content_soft_anneal_a0_teacher_val.log` |
| **In-train (best ep 31, α=0)** | `val_pointer_loss` **5.225**; val patch-acc **0.015**; val timing **0.0064** |
| **Offline α=0 teacher val** | Timing **58/35439 = 0.0016**; F1 **0.0042**; patch_wrong **35148**. Skill: timing **−0.0069**, F1 **−0.441** |
| **vs fixed soft-α** | Matched α=0.5 **0.0968** ([EXP-18](#exp-20260807-18-content-gap--soft-δ-prior-raises-r2-timing--still-prior-class)); fixed α=0 collapse **0.0019** ([EXP-19](#exp-20260807-19-content-gap-soft-α-collapses-at-α0-eval)) — anneal does not beat the collapse floor |
| **vs content-gap α=0** | Timing **worse** than content-gap alone **0.0075** ([EXP-16](#exp-20260807-16-r2-content-gap-beats-ptrloss-timing-still-near-floor)) |
| **Conclusion** | **Not supported.** Annealing the decode prior does not internalize localization. Close the soft-α product path; next must be a different mechanism |

### EXP-20260807-19: Content-gap soft-α collapses at α=0 eval

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 22:24:53 |
| **Track** | `metric` (AR) |
| **Question** | Do content-gap weights trained with soft Δ α=0.5 ([EXP-20260807-18](#exp-20260807-18-content-gap--soft-δ-prior-raises-r2-timing--still-prior-class)) keep skill when the prior is removed at decode? |
| **Setup** | Same ckpt `models_wsl/ar/ladder_r2_gap_content_soft_a0p5/`; eval config [`ladder_r2_gap_content_soft_a0_eval.json`](../../configs/ar/ladder_r2_gap_content_soft_a0_eval.json) with `pointer_soft_distance_alpha: 0`. Offline teacher val 50 songs. `logs/r2_gap_content_soft_a0_eval_teacher_val.log` |
| **α=0 offline** | Timing **66/35439 = 0.0019**; F1 **0.0036**; patch_wrong **35119**. Skill: timing **−0.0066**, F1 **−0.442** |
| **vs matched α=0.5** | Timing **~51×** drop (**0.0968 → 0.0019**); F1 **0.270 → 0.0036** |
| **vs absolute soft-α α=0** | Same class as [EXP-20260807-13](#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent) α=0 collapse (**0.00011**) |
| **Conclusion** | **Supported** (collapse). Soft Δ on gap is a decode crutch, not learned localization. Do not ship soft-α / Phase 5 default; next must survive α→0 |

### EXP-20260807-18: Content-gap + soft Δ prior raises R2 timing — still prior class

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 18:07:18 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does soft Δ-distance prior (`logits[Δ] -= α·decode_delta(Δ)`, α=0.5, no hard R) on content-gap fix the wrong-far overshoot from [EXP-20260807-17](#exp-20260807-17-content-gap-r2-errors-are-wrong-far-δ-soup--not-stickiness)? |
| **Setup** | [`configs/ar/ladder_r2_gap_content_soft_a0p5_probe.json`](../../configs/ar/ladder_r2_gap_content_soft_a0p5_probe.json); 50 ep, ES patience 10 on `val_pointer_loss`; restore ep **15**. Train `logs/r2_gap_content_soft_a0p5_train.log`; offline `logs/r2_gap_content_soft_a0p5_teacher_val.log` |
| **In-train (best ep 15)** | `val_pointer_loss` **2.440**; val patch-acc **0.100**; val timing **0.0969** |
| **Offline teacher val** | Timing **3432/35439 = 0.0968**; F1 **0.270**; patch_wrong **31122**; patch_ok_timing_wrong **3051**. Skill: timing **+0.089**, F1 **−0.056** vs ioi_shuffle |
| **vs content-gap α=0** | Timing **~13×** (**0.0968** vs **0.0075**); F1 skill much less negative (**−0.056** vs **−0.43**) |
| **vs absolute soft-α** | Comparable to [EXP-20260807-13](#exp-20260807-13-soft-distance-prior-beats-full-ce-still-prior-dependent) (timing **0.105** / F1 **0.279**) — same magnitude class |
| **Conclusion** | **Partial.** Soft Δ prior moves the needle hard on the stated overshoot failure. α=0 eval → collapse ([EXP-20260807-19](#exp-20260807-19-content-gap-soft-α-collapses-at-α0-eval)) — prior-dependent, not learned localization |

### EXP-20260807-17: Content-gap R2 errors are wrong-far Δ soup — not stickiness

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 17:46:26 |
| **Track** | `metric` (AR) |
| **Question** | On the content-gap R2 ckpt ([EXP-20260807-16](#exp-20260807-16-r2-content-gap-beats-ptrloss-timing-still-near-floor)), is leftover error at_prev stickiness, residual, or diffuse wrong-Δ? |
| **Setup** | `_tmp/r2_gap_content/diagnose_error_mix.py` on `models_wsl/ar/ladder_r2_gap_content/` (8 train / 12 val). `logs/r2_gap_content_error_mix.log` · `_tmp/r2_gap_content/error_mix.json` |
| **Val mix (8925 steps)** | patch_ok_time_ok **0.6%**; patch_ok_timing_wrong **1.2%**; **patch_wrong 98.2%** |
| **Val patch buckets** | correct **1.8%**; at_prev **2.0%**; near_miss≤4 **4.0%**; **wrong_far 91.9%** |
| **Val Δ geometry** | tgt_off p50 **2** / pred_off p50 **92**; Δerr p50 **+90**; sticky Δ=0 when tgt>0 only **2.0%** |
| **Val concentration** | H/Huni **0.932**; top-1 **0.025**; tgt_gap_rank p50 **65** — near-uniform over allowed gap ids |
| **Residual (patch_ok)** | resid_err_ms p50 **39** / p90 **60** — secondary (only **~1.8%** steps patch_ok) |
| **Train contrast** | wrong_far **85%**; correct **4.7%**; same overshoot (pred_off p50 **64** vs tgt **3**) — not a train≫val story |
| **Conclusion** | **Supported.** Binding failure is **wrong-far Δ soup with systematic overshoot**, not stickiness or residual. Differs from hard-R leftovers (at_prev). Next one-knob with mechanism: soft Δ-distance prior on gap logits (no hard cutoff); do not flip design default yet |

### EXP-20260807-16: R2 content gap beats ptrloss timing; still near floor

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 16:08:08 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does content gap on R2 50t/50v beat full-CE ptrloss timing (**~0.0035**) **without** soft α / hard R? |
| **Setup** | [`configs/ar/ladder_r2_gap_content_probe.json`](../../configs/ar/ladder_r2_gap_content_probe.json); 50 ep, ES patience 10 on `val_pointer_loss`; restore ep **11**. Train `logs/r2_gap_content_train.log`; offline `logs/r2_gap_content_teacher_val.log` |
| **In-train (best ep 11)** | `val_pointer_loss` **5.131**; val patch-acc **0.018**; val timing **0.0075** |
| **Offline teacher val** | Timing **266/35439 = 0.0075**; F1 **0.0119**; patch_wrong **34827**; patch_ok_timing_wrong **383**. Skill: timing **−0.0009**, F1 **−0.43** vs ioi_shuffle null |
| **vs ptrloss ep2** | Timing **~2.1×** (**0.0075** vs **0.0035**); comparable to encode-then-PE / QK-LN offline (**~0.007**) |
| **Conclusion** | **Supported** on the Phase 4 bar (beats ptrloss without decode prior). Localization still weak vs null skill — content gap clears the crutch-dependent failure class but does not yet transfer. Next: longer train, error-mode dig, or design default |

### EXP-20260807-15: Content gap tide gate PASS — audio-grounded

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 15:52:00 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | Does content-based gap (`gap_head: content`, score Δ via `q · k(memory[prev+Δ])`) pass tide: teacher timing ≈1 **and** audio ablation moves the gap head? |
| **Setup** | [`configs/ar/tide_gap_residual.json`](../../configs/ar/tide_gap_residual.json) with `gap_head: content`; 400 ep; ckpt `models_wsl/ar/tide_gap_residual/`. Train `logs/tide_gap_content_train.log`; decode `logs/tide_gap_content_decode.log`; ablation `logs/tide_gap_content_ablation.log` / `logs/audio_ablation_tide_gap_residual.json` |
| **In-train** | Peak `val_overfit_gate` **~0.986**; patch-acc **1.0**; late epochs stable (unlike Dense gap destabilization) |
| **Offline teacher** | Ordered @ 20 ms **626/634 (0.9874)** vs `target_times`; patch errors **0**. Free-run skipped (not perfect) |
| **Audio ablation** | matched timing **0.9874**; reverse **0.0126** / shuffle **0.0000** / zeros **0.0000**; `same_pred` **0** under shuffle+zeros. Gate **PASS** (pointer+token+query) |
| **vs Dense gap** | [EXP-20260807-14](#exp-20260807-14-tide-gap-residual-overfit--timing-near1-audio-blind) held **0.9968** under zeros — content gap fixes the audio-blind failure |
| **Conclusion** | **Supported.** Relative Δ CE + content scoring passes the non-keys-only tide gate. Timing short of perfect **634/634** but ≈1 and grounded. Next: R2 probe without decode prior |

### EXP-20260807-14: Tide gap_residual overfit — timing near-1, audio-blind

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 15:35:57 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does `alignment: gap_residual` (`Dense(gap_vocab)` + residual, no α / hard R) pass the tide gate: teacher timing ≈1 **and** audio ablation must move the gap head? |
| **Setup** | [`configs/ar/tide_gap_residual.json`](../../configs/ar/tide_gap_residual.json); 400 ep; ckpt `models_wsl/ar/tide_gap_residual/`. Train `logs/tide_gap_residual_train.log`; decode `logs/tide_gap_residual_decode.log`; ablation `logs/tide_gap_residual_ablation.log` / `logs/audio_ablation_tide_gap_residual.json`. Offline diagnose fixed for gap-only models (`pointer_logits` → `gap_logits`). |
| **In-train** | Peak `val_overfit_gate` **0.9968** (~ep 382); late epochs destabilized |
| **Offline teacher** | Ordered @ 20 ms **632/634 (0.9968)** vs `target_times`; patch errors **0**; patch-ok timing-wrong **2**. Free-run skipped (gate not perfect) |
| **Audio ablation** | matched / reverse / shuffle / zeros all timing **0.9968**, `same_pred_as_matched` **1.0**, token acc **1.0**. Gate **FAIL** (pointer+token+query). `query_cosine` defaults to **1.0** when no `pointer_query` layer — decisive signal is same_pred / timing under zeros |
| **Conclusion** | **Not supported.** Relative Δ CE memorizes the gap sequence from the teacher prefix without reading audio — same failure class as absolute `Dense(max_patches)` ([EXP-20260804-05](#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer)). Content pointer fixed that for absolute indices ([EXP-20260804-06](#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate)); gap needs the same: score Δ against `memory[prev+Δ]`, not a closed Dense classifier. Do not run R2 on this head |

### EXP-20260807-13: Soft distance prior beats full CE; still prior-dependent

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 13:44:23 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Can a **soft** ahead penalty (`logits -= α·max(0,p−prev)`, no hard cutoff) raise val timing vs full-CE ptrloss while keeping long gaps reachable? |
| **Setup** | [`configs/ar/ladder_r2_soft_distance_probe.json`](../../configs/ar/ladder_r2_soft_distance_probe.json) — QK-LN + hard time, `pointer_local_ce_radius: 0`, `pointer_soft_distance_alpha: 0.5`. ES @ ep **20**, restore ep **10**. `logs/r2_soft_distance_a0p5_train.log` |
| **In-train (best ep 10)** | `val_pointer_loss` **2.444**; val patch-acc **0.090**; val timing **0.105** |
| **Offline α=0.5 (matched)** | Timing **3717/35439 = 0.105**; F1 **0.279**; patch-acc **11.0%**. Skill: timing **+0.097**, F1 **−0.043**. `logs/r2_soft_distance_a0p5_teacher_val.log` |
| **Offline α=0 (prior off)** | Timing **4/35439 = 0.00011**; F1 **0.00014**; patch-acc **0.05%**. Collapse. `logs/r2_soft_distance_a0_eval_teacher_val.log` |
| **vs baselines** | vs ptrloss ep2 timing **0.0035** (~**30×**); vs hard R=4 masked **0.156** (lower timing, **higher** F1 **0.279** vs **0.251**); vs hard R=4 unmasked collapse **0.0016** |
| **Conclusion** | **Partial.** Soft prior is a strictly better diagnostic than hard-R (no unreachable gaps; strong timing/F1 with α on). Weights still do **not** localize when the prior is removed at eval — same failure class as hard-R, milder constraint. Next: gap head or anneal α→0 so skill survives without a decode prior |

### EXP-20260807-12: R=4 weights collapse without hard window

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:45:07 |
| **Track** | `metric` (AR) |
| **Question** | After hard-R was labeled a crutch ([NOTE-20260807-06](DISCUSSION_NOTES.md#note-20260807-06-hard-r-is-diagnostic-not-the-holistic-system)), do R=4-trained weights retain skill when eval uses **mono-only** (no `prev+R` cap)? |
| **Setup** | Same ckpt `models_wsl/ar/ladder_r2_prev_local_ce_r4/`; eval config [`ladder_r2_prev_local_ce_r4_unmasked_eval.json`](../../configs/ar/ladder_r2_prev_local_ce_r4_unmasked_eval.json) with `pointer_local_ce_radius: 0`. Offline teacher val 50 songs. `logs/r2_prev_local_ce_r4_unmasked_teacher_val.log` |
| **Unmasked offline** | Timing **57/35439 = 0.0016**; F1 **0.0023**; patch-acc **0.25%** (`patch_wrong` **35349**). Skill: timing **−0.0069**, F1 **−0.443** |
| **vs masked R=4** | Timing **0.0016** vs **0.156** (~**100×** drop); patch-acc **0.25%** vs **26.9%**; F1 **0.002** vs **0.251** |
| **vs ptrloss ep2 (full CE)** | Unmasked R=4 weights (**0.0016**) are **worse** than the earlier full-CE ptrloss ep2 (**0.0035**) |
| **Conclusion** | **Supported.** Hard-R training did **not** teach open-set localization — skill is mask-dependent. Hard-R closed as product path. Next must localize under mono/full support without hard unreachable gaps |

### EXP-20260807-11: R=4 error-mix — stickiness, not mid-window soup

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:33:28 |
| **Track** | `metric` (AR) |
| **Question** | On the R=4 ckpt, is the leftover failure still a diffuse mini-soup, residual, or a new mode (e.g. at_prev stickiness)? Also: does R=2 starve too many GT gaps? |
| **Setup** | `_tmp/r2_qk_ln_gap/diagnose_r8_error_mix.py` on R=4 ckpt (8 train / 12 val). Gap starve: `_tmp/r2_qk_ln_gap/gap_starve_r2_r4_r8.json`. Logs: `logs/r2_prev_local_ce_r4_error_mix.log` |
| **Val mix** | patch_ok_time_ok **15.6%**; patch_ok_timing_wrong **12.2%**; **patch_wrong 72.2%** (was 83% @ R=8) |
| **Val buckets** | **at_prev 30.7%**; near_miss≤2 **30.0%**; correct **27.8%**; near_miss 3–4 **7.6%**; wrong_far **0.3%** |
| **Val concentration** | H/Huni **0.892**; top-1 **0.347**; tgt rank p50 **2**; tgt_off=pred_off p50 **2** (geometry matched — no mid-window bias) |
| **Residual (patch_ok)** | resid_err_ms p50 **17.1** / p90 **41.6**; among patch_ok, **44%** still miss 20 ms tol |
| **Train contrast** | correct **41%**; at_prev **44%**; H/Huni **0.65**; top-1 **0.52** — train peaks, still sticky |
| **Gap starve (val later onsets)** | R=8 **1.8%**; R=4 **5.8%**; **R=2 20.3%**. Gap hist: =1 **44%**, =2 **34%**, =0 **1.6%**, ≥3 **20%** |
| **Conclusion** | **Supported.** Mid-window soup is largely gone; binding in-window miss is **at_prev stickiness** (+ near-miss≤2). Residual improved (p50 under tol) but secondary. **Defer R=2** — 20% of GT gaps are unreachable under `[prev, prev+2]`. Next: force-advance (`min_ahead=1`) on R=4 |

### EXP-20260807-10: Prev-local R=4 beats R=8

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:30:51 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After [EXP-20260807-09](#exp-20260807-09-r8-still-diffuse--residual-secondary) showed R=8 still a ~9-way soup with GT offset p50=2, does **R=4** beat R=8? |
| **Setup** | [`configs/ar/ladder_r2_prev_local_ce_r4_probe.json`](../../configs/ar/ladder_r2_prev_local_ce_r4_probe.json) — same recipe, `pointer_local_ce_radius: 4`. ES @ ep **29**, restore ep **19**. `logs/r2_prev_local_ce_r4_train.log` |
| **In-train (best ep 19)** | `val_pointer_loss` **1.743**; val patch-acc **0.244**; val timing **0.156** |
| **Offline val (50 songs)** | **5539/35439 = 0.156**; F1 **0.251**; patch-acc **26.9%** (`patch_wrong` **25911**); `patch_ok_timing_wrong` **3997**. Skill: timing **+0.149**, F1 **−0.084**. `logs/r2_prev_local_ce_r4_teacher_val.log` |
| **vs R=8** | Timing **0.156** vs **0.070** (~**2.2×**); patch-acc **26.9%** vs **16.6%**; F1 **0.251** vs **0.108**; F1 skill **−0.084** vs **−0.29** |
| **vs ptrloss ep2** | Timing **~45×** (**0.156** vs **0.0035**) |
| **Conclusion** | **Supported.** R-shrink continues to pay. F1 approaching the audio-blind floor; next measure soup/residual/gap-starve at R=4 before R=2 |

### EXP-20260807-09: R=8 still diffuse; residual secondary

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:12:02 |
| **Track** | `metric` (AR) |
| **Question** | On the R=8 ckpt, is the local window still a mini-soup, and how much of the leftover error is residual (`patch_ok_timing_wrong`) vs wrong patch? |
| **Setup** | `_tmp/r2_qk_ln_gap/diagnose_r8_error_mix.py` — teacher-forced prev + mono + `max_ahead=8`; 8 train / 12 val. `logs/r2_prev_local_ce_r8_error_mix.log` · `_tmp/r2_qk_ln_gap/r8_error_mix.json` |
| **Val mix (8925 steps)** | patch_ok_time_ok **6.8%**; **patch_ok_timing_wrong 10.0%**; **patch_wrong 83.2%**. Among patch_ok, **60%** miss 20 ms tol |
| **Val buckets** | correct **16.8%**; at_prev **17.3%**; near_miss≤2 **20.6%**; near_miss 3–4 **19.5%**; wrong_far **23.7%** (was **63%** at R=32) |
| **Val concentration** | H/Huni **0.961**; top-1 **0.178**; target rank p50 **4** (of ~9); tgt_off p50 **2** / pred_off p50 **4** |
| **Residual (patch_ok only)** | resid_err_ms p50 **27.5** / p90 **50.4** — equals abs time err when patch matches; systematically above 20 ms tol |
| **vs R=32 in-window** | Soup shrunk 33→9 bins but still near-uniform; mid-window bias reduced (pred_off 15→4); at_prev stickiness up (6%→17%) |
| **Conclusion** | **Supported.** Binding leftover is still **wrong patch** (83%), not residual. Residual is a real but secondary tax (~10% of steps; ~60% of correct patches). Evidence-backed next: **R=4** prev-local probe; defer residual-focused trains until patch localization improves further |

### EXP-20260807-08: Prev-local R=8 beats R=32

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:08:36 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After [EXP-20260807-07](#exp-20260807-07-in-window-errors-are-left-skewed-gt-vs-diffuse-mid-window-preds) showed GT gaps near `prev` (p50=2) under a diffuse 33-way window, does **R=8** beat R=32 on val timing / patch-acc? |
| **Setup** | [`configs/ar/ladder_r2_prev_local_ce_r8_probe.json`](../../configs/ar/ladder_r2_prev_local_ce_r8_probe.json) — same as v3 but `pointer_local_ce_radius: 8`, separate `model_output_dir`. ES @ ep **16**, restore ep **6**. `logs/r2_prev_local_ce_r8_train.log` |
| **In-train (best ep 6)** | `val_pointer_loss` **2.178**; val patch-acc **0.164**; val timing **0.0697** (peak timing **0.0975 @ ep 5**, not selected) |
| **Offline val (50 songs)** | **2471/35439 = 0.0697**; F1 **0.108**; patch-acc **16.6%** (`patch_wrong` **29551**); `patch_ok_timing_wrong` **3478**. Skill: timing **+0.0618**, F1 **−0.291**. `logs/r2_prev_local_ce_r8_teacher_val.log` |
| **vs R=32 v3** | Timing **0.0697** vs **0.0228** (~**3.1×**); patch-acc **16.6%** vs **5.5%**; timing skill **+0.062** vs **+0.015** |
| **vs ptrloss ep2** | Timing **~20×** (**0.0697** vs **0.0035**) |
| **Conclusion** | **Supported.** Matching R to gap mass is a real lever. Remaining: still below null F1; **3478** correct-patch / wrong-time steps; check whether R=8 window is still diffuse before shrinking further or attacking residual |

### EXP-20260807-07: In-window errors are left-skewed GT vs diffuse mid-window preds

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 11:53:22 |
| **Track** | `metric` (AR) |
| **Question** | After prev-local r=32 beat ([EXP-20260807-06](#exp-20260807-06-prev-relative-local-ce-beats-ptrloss-ep2)), where inside the decode window do remaining errors sit — at_prev stickiness, near-miss, or diffuse mid-window? |
| **Setup** | `_tmp/r2_qk_ln_gap/diagnose_inwindow_errors.py` on v3 ckpt; teacher-forced prev + mono + `max_ahead=32`. 8 train / 12 val. `logs/r2_prev_local_ce_v3_inwindow.log` · `_tmp/r2_qk_ln_gap/inwindow_errors.json` |
| **Val buckets (8925 steps)** | correct **5.7%**; at_prev **6.2%**; near_miss≤2 **6.4%**; near_miss 3–8 **17.8%**; **wrong_far_in_window 63.3%**; at_prev+1 **0.6%**. outside/behind **0** (mask holds) |
| **Val geometry** | target_offset_from_prev p50 **2** / p90 **3**; pred_offset p50 **15** / p90 **30**; signed(pred−tgt) mean **+14.3**; abs_delta p50 **13** |
| **Val concentration** | window H/Huni **0.95**; top-1 **0.064**; target rank p50 **11** (of ~33 allowed) |
| **Train (contrast)** | correct **9.5%**; same mid-window bias (pred offset p50 **15**, target p50 **3**); slightly less diffuse (H/Huni **0.86**) |
| **Conclusion** | **Supported.** Local CE stopped the 800-way soup but left a **~33-way soup**. GT gaps are left-skewed near `prev`; argmax lands near the window center as if nearly uniform. Evidence-backed next: **shrink R** toward gap mass (R≈8; gap p99≈12 from earlier gap scan) — not scale-up |

### EXP-20260807-06: Prev-relative local CE beats ptrloss ep2

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 11:47:29 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After target-centered local CE failed ([EXP-20260807-05](#exp-20260807-05-clean-local-ce-r32--no-beat--88-preds-outside-window)), does **decode-consistent** prev-relative CE `[prev, prev+R]` (R=32) raise val timing / patch-acc vs ptrloss ep2? |
| **Setup** | [`configs/ar/ladder_r2_prev_local_ce_probe.json`](../../configs/ar/ladder_r2_prev_local_ce_probe.json) — QK-LN + hard time + `pointer_local_ce_anchor: prev`, r=32, no STE, ckpt `val_pointer_loss`. Run label `r2_prev_local_ce_v3`. Plumbing: skip upper bound when `prev==0`; drop teacher gaps `>R` from pointer CE (rare section gaps otherwise poison mean CE ~1e6). `logs/r2_prev_local_ce_v3_train.log` |
| **Gap poison (pre-fix)** | Val later-onset gap `>32`: **45/35389 (0.13%)** → approx mean CE **~1.3e6**; train **~0.45e6**. Max gap **134** patches (~10.7 s). `_tmp/r2_qk_ln_gap/prev_local_gaps.json` |
| **In-train** | ES @ ep **22**, restore ep **12**. Best `val_pointer_loss` **3.275**; val patch-acc **0.061**; val timing **0.0228** |
| **Offline val (50 songs)** | **809/35439 = 0.0228**; F1 **0.0315**; `patch_wrong` **33479** (patch-acc **~5.53%**); `patch_ok_timing_wrong` **1256**. Skill vs strongest null: timing **+0.0145**, F1 **−0.401**. `logs/r2_prev_local_ce_v3_teacher_val.log` |
| **vs baselines** | ptrloss ep2 **0.0035**; QK-LN ep31 **0.0070**; target local CE **0.0025** — **~6.5×** timing vs ep2, **~3×** vs ep31 |
| **Conclusion** | **Supported.** Decode-aligned locality is the first clear timing beat on fixed R2 since the plumbing fixes. Binding failure moves from “diffuse 800-way soup” to **in-window miss** (~5% patch-acc; F1 still below null). Next: in-window error modes before another loss train — no R3+ |

### EXP-20260807-05: Clean local CE r=32 — no beat; 88% preds outside window

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:50:35 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After [NOTE-20260807-03](DISCUSSION_NOTES.md#note-20260807-03-far_ahead-at-ep2-is-diffuse-mono-suffix-mass-not-confident-wrong-peaks) (diffuse ~800-way CE), does **target-centered** local pointer CE alone (radius **32**, hard time, **no** STE, ckpt `val_pointer_loss`) beat ptrloss ep2? |
| **Setup** | [`configs/ar/ladder_r2_local_ce_probe.json`](../../configs/ar/ladder_r2_local_ce_probe.json) — QK-LN + encode-then-PE + local CE r=32. ES @ ep **20**, restore ep **10**. `logs/r2_local_ce_probe_train.log` |
| **In-train (best ep 10)** | `val_pointer_loss` **3.36** (local support — **not** comparable to full CE ~6); val patch-acc **0.0092**; val timing **0.0025** |
| **Offline val (50 songs)** | **89/35439 = 0.0025**; F1 **0.0017**; `patch_wrong` **35138**; skill F1 **−0.44**. `logs/r2_local_ce_probe_teacher_val.log` |
| **vs ptrloss ep2** | Timing **0.0025** vs **0.0035**; patch-acc **0.009** vs **0.014** — worse |
| **Outside-window check (12 val)** | Correct **0.9%**; wrong-inside-±32 **11%**; **outside ±32: 88%**. `_tmp/r2_qk_ln_gap/local_ce_outside_radius.json` |
| **Conclusion** | **Not supported.** Local CE shrinks the *loss* support but inference still argmaxes the full mono suffix; unconstrained outside-window logits win. Next must be **decode-consistent** locality (e.g. prev-relative `[prev, prev+R]`), not another target-centered radius |

### EXP-20260807-04: Far_ahead entropy — ep2 diffuse, ep31 peaked-wrong

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:34:45 |
| **Track** | `metric` (AR) |
| **Question** | At selected ep2 (and contrast ep31), is far_ahead near-uniform over the mono-allowed suffix or confident wrong peaks? |
| **Setup** | `_tmp/r2_qk_ln_gap/diagnose_far_ahead_entropy.py` — mono-masked softmax entropy, top-1/5/10, target rank; 8 train / 12 val. Models: ptrloss ep2 + probe ep31. `logs/r2_qk_ln_far_ahead_entropy.log` · `_tmp/r2_qk_ln_gap/far_ahead_entropy.json` |
| **Ep2 val (far_ahead 97%)** | n_allowed ~**809**; H/H_uniform **0.921**; top-1 **0.016**; top-5 **0.062**; target rank p50 **153** |
| **Ep31 val (far_ahead 96%)** | Same n_allowed; H/H_uniform **0.584**; top-1 **0.202**; top-5 **0.419**; target rank p50 **94** |
| **Conclusion** | **Supported.** Selected operating point is **diffuse** (~800-way soup), not peaked-wrong. Late peaking is wrong-mode memorization. Fixed-R2 next: isolate **local pointer CE** (no STE) — [NOTE-20260807-03](DISCUSSION_NOTES.md#note-20260807-03-far_ahead-at-ep2-is-diffuse-mono-suffix-mass-not-confident-wrong-peaks) |

### EXP-20260807-03: Gap diagnosis — selected ep2 both splits weak

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:16:09 |
| **Track** | `metric` (AR) |
| **Question** | Is the binding failure a train≫val gap at the exported `val_pointer_loss` weights, or are both splits weak (with late train localization as memorization)? |
| **Setup** | Ptrloss ckpt `models_wsl/ar/ladder_r2_qk_ln_ptrloss/`; contrast ep31 `ladder_r2_qk_ln_probe/`. Offline train `--limit 8`; error-modes 8/12; epoch curves from both train logs. Artifacts: `_tmp/r2_qk_ln_gap/`, `_tmp/r2_qk_ln_ptrloss_diag/error_modes.json`, `logs/r2_qk_ln_ptrloss_teacher_train.log`, `logs/r2_qk_ln_ptrloss_error_modes.log` |
| **Curves** | Ep2: train patch-acc **0.0138** ≈ val **0.0137**. Late: train → **~0.08+**, val flat **~0.016**, val NLL → uniform+ |
| **Offline (ptrloss ep2)** | Train timing **0.0039**; val **0.0035** ([EXP-20260807-02](#exp-20260807-02-val_pointer_loss-selection-picks-ep2--timing-worse-than-patch-acc-ckpt)) |
| **Error-modes (ptrloss ep2)** | Train **2.7%** / NLL **5.57** / median \|Δ\| **107**; val **1.2%** / NLL **6.12** / median **157**; both ~**97%** far_ahead |
| **Contrast (ep31)** | Train **11.5%** / NLL **3.64** / median **12** vs val **1.7%** / NLL **7.65** |
| **Conclusion** | **Supported.** Selected operating point is **both-weak**, not train-ahead-of-val. Dropout-first lacks a gap to close. Scale-up (R3) was a candidate but **user declined** — stay fixed-R2; next measure far_ahead entropy ([NOTE-20260807-02](DISCUSSION_NOTES.md#note-20260807-02-at-selected-weights-both-splits-are-weak--late-train-localization-is-memorization)) |

### EXP-20260807-02: `val_pointer_loss` selection picks ep2 — timing worse than patch-acc ckpt

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 01:27:17 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After [EXP-20260807-01](#exp-20260807-01-qk-ln-train-ablation-pass--ckpt-on-val_pointer_loss), does retraining QK-LN R2 with `checkpoint_metric: val_pointer_loss` export a better val checkpoint than patch-acc ep **31**? |
| **Setup** | [`configs/ar/ladder_r2_qk_ln_probe.json`](../../configs/ar/ladder_r2_qk_ln_probe.json) — `run_label: r2_qk_ln_ptrloss`, `model_output_dir: models_wsl/ar/ladder_r2_qk_ln_ptrloss`, 50 ep / ES **10**. `logs/r2_qk_ln_ptrloss_train.log` |
| **Stop** | ES @ ep **12**, restore ep **2** (best `val_pointer_loss` **5.9728**; patch-acc **0.0137**; timing **0.0035**) |
| **Offline val (50 songs)** | **124/35439 = 0.0035**; F1 **0.0035**; `patch_wrong` **34997**; skill F1 **−0.44**. `logs/r2_qk_ln_ptrloss_teacher_val.log` |
| **Compare (patch-acc ckpt)** | Prior QK-LN export (ep **31**): offline timing **0.0070**, F1 **0.0094**, `patch_wrong` **34828**, val NLL ~uniform **~7.39** |
| **Conclusion** | **Partial.** Monitor swap **correctly** prefers NLL≪uniform (ep2). Absolute val skill does **not** improve — best-NLL weights are weaker on timing than the mis-selected late epoch. Selection was a real bug but not the binding failure; train→val gap remains |

### EXP-20260807-01: QK-LN train ablation PASS + ckpt on `val_pointer_loss`

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 00:38:19 |
| **Track** | `metric` + `train` (AR) |
| **Question** | After [NOTE-20260807-01](DISCUSSION_NOTES.md#note-20260807-01-train-localizes-val-does-not--gap-is-the-binding-failure), does train-split audio ablation on the QK-LN R2 ckpt confirm pointer grounding, and does switching checkpoint/ES to `val_pointer_loss` address the ep31≈uniform selection bug? |
| **Setup** | Same ckpt `models_wsl/ar/ladder_r2_qk_ln_probe/ar_onset_model.keras`; `audio_ablation_ar_onset.py --split train --limit 8 --gate`. Configs flipped: [`ladder_r2_qk_ln_probe.json`](../../configs/ar/ladder_r2_qk_ln_probe.json), [`ladder_50t_50v_content_pointer.json`](../../configs/ar/ladder_50t_50v_content_pointer.json), [`ladder_r2_ctx_pefree_full.json`](../../configs/ar/ladder_r2_ctx_pefree_full.json), [`ladder_r2_content_cross_probe.json`](../../configs/ar/ladder_r2_content_cross_probe.json) → `checkpoint_metric: val_pointer_loss` |
| **Train ablation (8 songs)** | Gate **PASS** (pointer+token+query). Matched timing **0.0428** / F1 **0.0417** / ptr NLL **3.64** (uniform **7.31**); shuffle timing **0.0012** / same_pred **0.0003**; zeros query cos **0.90**. `logs/r2_qk_ln_ablation_train.json` |
| **Val ablation (prior, 12 songs)** | Matched timing **0.0057** / ptr NLL **7.65**; pointer gate PASS at floor; tokens barely move. `logs/r2_qk_ln_ablation_gate.json` |
| **Conclusion** | **Supported** for “train is grounded / selection was wrong monitor.” Does **not** yet prove val transfer under the new monitor — need a retrain that actually selects on min `val_pointer_loss` |

### EXP-20260806-07: Pointer QK LayerNorm — tide PASS; R2 patch-acc visible, timing flat

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 20:22:33 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | After encode-then-PE, does LayerNorm on pointer Q/K streams improve R2 localization vs short ctx-pefree **0.0069**? |
| **Code** | `pointer_qk_layernorm` (default True) → `pointer_query_ln` / `pointer_key_ln` before Dense; inference rebuild aware. Also: ablation floor skip, `pointer_patch_accuracy`, mono-aware ablation NLL, PE-key footguns closed ([NOTE-20260806-03](DISCUSSION_NOTES.md#note-20260806-03-remaining-defect-inventory-after-encode-then-pe)) |
| **Tide** | Gate **PASS** — matched timing **0.968**, zeros query cos **−0.12**, shuffle same_pred **0**. `logs/tide_qk_ln_ablation_gate.json` |
| **Setup (R2)** | [`configs/ar/ladder_r2_qk_ln_probe.json`](../../configs/ar/ladder_r2_qk_ln_probe.json) — hard time, ckpt `val_pointer_patch_accuracy`, 50 ep / ES **10**. `logs/r2_qk_ln_probe_train.log` |
| **Best** | val patch-acc **0.0185**, val timing **0.0070–0.0073**; offline **247/35439 = 0.0070**; `patch_wrong` **34828**; F1 **0.0094**. `logs/r2_qk_ln_teacher_val.log` |
| **Ablation (12 val, floor-fixed gate)** | Pointer **PASS** (shuffle same_pred **0**); query **PASS** (zeros cos **0.71**); token zeros still ≈ matched. Overall gate **PASS**. `logs/r2_qk_ln_ablation_gate.json` |
| **Compare** | Ctx-pefree short offline **0.0069** / patch_wrong **34877** — QK LN ≈ ties timing, ~**50** more correct patches |
| **Conclusion** | **Partial.** QK LN is safe (tide) and makes patch-acc a usable signal; not enough alone for null skill. Ablation floor fix is the diagnostic unlock |

### EXP-20260806-06: Content-only decoder cross regresses tide

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 19:48:00 |
| **Track** | `model` (AR) |
| **Question** | Does removing PE residual from decoder cross (`decoder_cross_content_only`) help localization without breaking the tide gate? |
| **Setup** | Default True briefly; tide 400-ep train (`logs/tide_content_cross_train.log`). First attempt used unsafe `Lambda` (reload failed); replaced with `ContentOnlyCrossMemory` |
| **Result** | Peak `val_timing_match_teacher` **0.6735** vs encode-then-PE mix **~0.94–0.99** |
| **Conclusion** | **Not supported** as default. Flag kept **False**; serializable path remains for R2 A/B only |

### EXP-20260806-05: Full R2 encode-then-PE beats prior hard-time full run

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 14:12:44 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does 500-ep R2 with encode-then-PE beat the prior full hard-time content-pointer R2 (**0.0085** in-train / **0.0086** offline mono)? |
| **Setup** | [`configs/ar/ladder_r2_ctx_pefree_full.json`](../../configs/ar/ladder_r2_ctx_pefree_full.json) — hard time, ES patience **50**, 500 ep. WSL GPU ~**67** min. `logs/r2_ctx_pefree_full_train.log` |
| **Stop** | ES around ep **194** (best @ **144**); TensorBoard `epoch_timing_match_teacher` |
| **Best (ep 144)** | val timing **0.00945** |
| **Offline val (50 songs, mono teacher)** | **334/35439 = 0.0094**; F1 **0.0464**; `patch_wrong` **34774**; skill F1 **−0.38**. `logs/r2_ctx_pefree_full_teacher_val.log` |
| **Compare** | Prior full hard: in-train **0.0085**, offline mono **0.0086**, F1 **0.021** — encode-then-PE **+11%** timing, **~2.2×** F1 |
| **Conclusion** | **Supported.** Architecture fix scales to full R2. Absolute performance still at floor vs null — next is localization beyond pe-free contextualization |

### EXP-20260806-04: Encode-then-PE — contextualized pe-free keys beat hard short probe

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 13:02:01 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | Prior `pointer_keys_pe_free` keyed on raw ``Dense(MERT)`` (encoder skipped). Does encoding **before** absolute PE give contextualized pe-free keys that beat hard-time short probe **0.005**? |
| **Code** | `_encode_patches`: encoder on `patch_embed`, then `enc_pos`; pe-free keys = encoder output. Inference `pointer_key_input` from `enc_*_ln2`. Unit test: content ≠ raw embed |
| **Tide check** | Gate **PASS** — matched timing **0.94**, zeros query/token PASS. `logs/tide_ctx_pefree_ablation_gate.json` |
| **Setup (R2 probe)** | [`configs/ar/ladder_r2_ctx_pefree_probe.json`](../../configs/ar/ladder_r2_ctx_pefree_probe.json) — hard time, full CE, 50 ep / ES **10**. WSL ~**13** min. `logs/r2_ctx_pefree_probe_train.log` |
| **Stop** | ES @ ep **25**, restore ep **15** |
| **Best (ep 15)** | val timing **0.0069**, train timing **0.0124**; val pointer CE **6.14** (&lt; uniform ~**7.4**); train pointer **5.04** |
| **Offline val (50 songs, mono teacher)** | **245/35439 = 0.0069** (matches in-train); `patch_wrong` **34877**; F1 **0.0077**; skill **−0.44**. `logs/r2_ctx_pefree_teacher_val.log` |
| **Compare** | Hard short probe **0.005**; prior full hard R2 **0.0085** in-train — this 50-ep probe already **1.38×** hard short bar |
| **Conclusion** | **Supported.** Skipping the encoder for pe-free keys was the architecture bug; encode-then-PE is the fix. Still at absolute floor vs null skill — full 500-ep next |
| **Related** | [NOTE-20260806-02](DISCUSSION_NOTES.md) · [EXP-20260805-05](#exp-20260805-05-hard-pointer-time-probe-raises-timing-10-but-still-at-floor) |

### EXP-20260806-03: STE without correct-patch — grads live, still below hard CE

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 12:34:53 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With correct-patch masking removed, does STE `λ_time` + full CE beat hard-time short probe **0.005**? |
| **Setup** | [`configs/ar/ladder_r2_ste_nocorrect_probe.json`](../../configs/ar/ladder_r2_ste_nocorrect_probe.json) — STE on, `time_loss_correct_patch_only: false`, `pointer_local_ce_radius: 0`, 50 ep / ES **10**. WSL GPU ~**13** min. `logs/r2_ste_nocorrect_probe_train.log` |
| **Stop** | ES @ ep **26**, restore ep **16** |
| **Best (ep 16)** | val timing **0.0041**, train timing **0.0035**; val pointer CE **19.5**; val `time_loss` **21.4** |
| **Compare** | Hard probe **0.005**; STE+correct-patch **0.004**; local-CE STE **0.0037**. `time_loss` now ~**20–28** (was ~**0.02** with correct-patch) — STE path is live |
| **Conclusion** | **Not supported.** Unmasking `λ_time` activates STE grads but soft expected-patch seconds remain a weak multi-song localizer (see NOTE-20260806-01). Hard-time CE-only stays the short-probe bar. Close STE recipe chase; next is architecture (shuffle-sensitive queries) |

### EXP-20260806-02: STE + full CE + correct-patch `λ_time` — no beat over hard time

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 11:51:00 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After local-CE STE underperformed ([EXP-20260806-01](#exp-20260806-01-ste--local-ce--offline-monotonic-fix)), does STE + correct-patch `λ_time` with **full** pointer CE beat hard-time short probe **0.005**? |
| **Setup** | [`configs/ar/ladder_r2_ste_full_ce_probe.json`](../../configs/ar/ladder_r2_ste_full_ce_probe.json) — `use_ste_pointer_time: true`, `time_loss_correct_patch_only: true`, `pointer_local_ce_radius: 0`, 50 ep / ES **10**. WSL GPU ~**11** min. `logs/r2_ste_full_ce_probe_train.log` |
| **Stop** | ES @ ep **18**, restore ep **8** |
| **Best (ep 8)** | val timing **0.0040**, train timing **0.0019**; val pointer CE **16.2**; val `time_loss` **0.017** |
| **Compare** | Hard probe **0.005**; local-CE STE **0.0037** — full CE STE slightly above local CE, still below hard |
| **Conclusion** | **Not supported.** Correct-patch gating keeps `λ_time` near-zero while patches are wrong, so STE never becomes the localization teacher. Hard-time CE-only remains the best short R2 bar. Next: STE **without** correct-patch mask, or architecture (shuffle-sensitive queries) |

### EXP-20260806-01: STE + local CE + offline monotonic fix

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 11:01:36 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | After [NOTE-20260806-01](DISCUSSION_NOTES.md#note-20260806-01-hard-λ_time-never-trains-the-pointer-soft-epatch-is-non-localizing-offline-drops-monotonic): (A) localizing pointer loss + STE/`λ_time` on correct patches, (C) offline teacher monotonic parity — does R2 timing move, and does offline match in-train? |
| **Code** | `use_ste_pointer_time`, `time_loss_correct_patch_only`, `pointer_local_ce_radius` in `losses.py` / `config.py` / `trainers.py`; offline `eval_ar_onset_offline.py` passes `monotonic` + `target_patch_indices`. Unit tests: STE `grad_pointer≠0`, local mask, correct-patch time gate |
| **Setup (probe)** | [`configs/ar/ladder_r2_localizing_pointer_probe.json`](../../configs/ar/ladder_r2_localizing_pointer_probe.json) — STE + correct-patch `λ_time` + local CE radius **32**, hard decode metrics, 50 ep / ES **10**. WSL GPU ~**12** min. `logs/r2_localizing_pointer_probe_train.log` |
| **Stop** | ES @ ep **20**, restore ep **10** |
| **Best (ep 10)** | val timing **0.0037**, train timing **0.0023**; val pointer CE **9.20** (local window) |
| **Compare** | Hard-time short probe **0.005 @ ep 7**; full hard R2 **0.0085 @ ep 96** — localizing package **below** hard short probe |
| **Offline mono fix (same hard R2 ckpt)** | Before: **28/35439 = 0.00079**, `patch_wrong` **35382**. After: **305/35439 = 0.0086**, `patch_wrong` **34885** — matches in-train/ablation (~**0.0085** / **0.0071**). `logs/r2_hard_time_teacher_val_mono_fix.log` |
| **Conclusion** | **Partial.** Metric bug **C fixed** (train/offline cliff was mostly teacher-forced monotonic). Package **A+B with local CE r=32 did not beat** hard-time baseline — `time_loss` stayed ~**0.02** (almost no correct patches → STE rarely fires) and local CE NLL stayed ≫ window-uniform. Next: STE + correct-patch **without** local window (full CE) |
| **Related** | [NOTE-20260806-01](DISCUSSION_NOTES.md#note-20260806-01-hard-λ_time-never-trains-the-pointer-soft-epatch-is-non-localizing-offline-drops-monotonic) · [EXP-20260805-05](#exp-20260805-05-hard-pointer-time-probe-raises-timing-10-but-still-at-floor) · [EXP-20260805-06](#exp-20260805-06-full-r2-hard-pointer-time--timing-rises-still-below-skill) |

### EXP-20260805-07: Hard time + tide `lambda_residual: 30` — no beat over lam5 R2

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-06 00:53:53 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Tide champion uses `lambda_residual: 30` vs ladder **5**. Does tide parity + hard pointer time raise multi-song timing beyond the lam5 hard-time R2? |
| **Setup** | [`configs/ar/ladder_r2_hard_time_lam30_probe.json`](../../configs/ar/ladder_r2_hard_time_lam30_probe.json) — fixed stack, `use_soft_pointer_time: false`, `lambda_residual: 30`, 50 ep / ES **10**. WSL GPU ~**13** min. `logs/r2_hard_time_lam30_probe_train.log` |
| **Stop** | ES @ ep **26**, restore ep **16** |
| **Best (ep 16)** | val timing **0.0065**, train timing **0.0106** |
| **Compare** | Lam5 hard probe **0.005 @ ep 7**; full lam5 hard R2 **0.0085 @ ep 96** — lam30 ~**1.3×** short probe, **below** full R2 |
| **Conclusion** | **Not supported.** Higher residual weight does not beat the best hard-time ladder run. Keep **`lambda_residual: 5`** for R2; next recipe axis is not λ_residual alone |

### EXP-20260805-06: Full R2 hard pointer time — timing rises, still below skill

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 22:40:53 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does a full 500 ep R2 run with `use_soft_pointer_time: false` leave the ~10⁻³ floor? |
| **Setup** | [`configs/ar/ladder_50t_50v_content_pointer.json`](../../configs/ar/ladder_50t_50v_content_pointer.json) — hard time, fixed decoder-audio stack. WSL GPU ~**48** min. `logs/ladder_r2_content_pointer_train.log` |
| **Stop** | ES @ ep **146**, restore ep **96** |
| **Best (ep 96)** | val timing **0.0085**, train timing **0.054** |
| **Final (ep 146)** | val timing **0.0074**, train timing **0.222** |
| **Compare** | Soft-time R2 best val **0.0014 @ ep 4**; 50 ep hard probe **0.005 @ ep 7** — full hard run ~**6×** soft / ~**1.7×** probe on in-train val |
| **Offline val (50 songs, teacher)** | Timing **28/35439 = 0.00079**; F1 **0.0212**; skill vs null **−0.42**; `patch_wrong` **99.8%**. `logs/r2_hard_time_teacher_val.log` |
| **Val ablation (12 songs, `--gate`)** | Matched timing **0.0071**; gate **FAIL** on pointer (matched≈floor); zeros query **0.18**, tok **0.24** → decoder grounding OK. `logs/r2_hard_time_ablation_gate.json` |
| **Conclusion** | **Partial on train curve only.** Hard time moves in-train val timing to **0.0085** but offline val still at floor with negative null skill — not ready for R3+. Next: hard time + tide `lambda_residual: 30` |

### EXP-20260805-05: Hard pointer time probe raises timing ~10× but still at floor

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:51:02 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With soft pointer time, `time_loss` dominates nats but timing stays at ~10⁻³. Does **hard-argmax** time (`use_soft_pointer_time: false`) backprop into the pointer and raise timing? |
| **Setup** | [`configs/ar/ladder_r2_hard_time_probe.json`](../../configs/ar/ladder_r2_hard_time_probe.json) — fixed stack, `use_soft_pointer_time: false`, monotonic **true**, 50 ep / ES **10**. WSL GPU ~**10** min. `logs/r2_hard_time_probe_train.log` |
| **Stop** | ES @ ep **17**, restore ep **7** |
| **Best (ep 7)** | val timing **0.0050**, train timing **0.0046**; val pointer loss **7.21** nats, val time loss **22.6** (hard CE on argmax patch) |
| **Compare** | No-monotonic soft probe best val **0.00054**; fixed soft R2 best **0.0014** — hard time ~**10×** / ~**3.5×** respectively, but still ≪ 1% and below any null skill bar |
| **Conclusion** | **Partial.** Soft expected time was likely starving pointer CE on multi-song batches. Next: **full R2** with hard time (500 ep) or hard time + tide `lambda_residual: 30` short probe |

### EXP-20260805-04: R2 no-monotonic probe still at timing floor

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:39:29 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Does `monotonic_pointer: true` (design §7) prevent multi-song pointer timing from training? |
| **Setup** | [`configs/ar/ladder_r2_no_monotonic_probe.json`](../../configs/ar/ladder_r2_no_monotonic_probe.json) — same 50t/50v ladder as fixed R2 but `monotonic_pointer: false`, 50 ep / ES patience **10**. WSL GPU ~**10** min. `logs/r2_no_monotonic_probe_train.log` |
| **Teacher offline (5 train songs, fixed R2 ckpt)** | Micro timing **1/1796 = 0.00056**; F1 **0.0006**; skill vs null **−0.60**; `n_patch_wrong` **1793/1796**. `logs/r2_fix_teacher_train5.log` |
| **Probe stop** | ES @ ep **17**, restore ep **7** |
| **Best (ep 7)** | val timing **0.00054**, train timing **0.00025** |
| **Conclusion** | **Not supported.** Monotonic mask is not the binding constraint. `time_loss` ~**33–37** nats vs pointer ~**24** — recipe/loss balance or pointer CE path still suspect. Next: hard pointer time or tide-parity λ |

### EXP-20260805-03: Fixed-stack R2 content-pointer still at timing floor

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 21:16:11 |
| **Track** | `train` + `metric` (AR) |
| **Question** | After [EXP-20260805-02](#exp-20260805-02-decoder-audio-fix--tide-content-pointer-passes-queryzeros-gate) fixed decoder audio on tide, does the same stack on R2 produce `val_timing_match_teacher` above the ~0 floor? |
| **Setup** | [`configs/ar/ladder_50t_50v_content_pointer.json`](../../configs/ar/ladder_50t_50v_content_pointer.json) — fixed masks, pe-free keys, pointer/decoder cross on `patch_embed` mix, monotonic pointer, soft pointer time, `checkpoint_metric: val_timing_match_teacher`, ES patience **50**, 500 ep budget. WSL GPU ~**21** min. `logs/ladder_r2_content_pointer_train.log` |
| **Stop** | ES restored **best epoch 4**; training stopped @ ep **54** |
| **Best (ep 4)** | `val_timing_match_teacher` **0.0014**, train timing **0.0009**; val tok **0.125** / train tok **0.178**; val F1 **0.039** / train F1 **0.076** |
| **Final (ep 54, pre-restore)** | val timing **0.00065**, train timing **0.00098**; val tok **0.173** / train tok **0.424**; val F1 **0.080** / train F1 **0.173** |
| **Contrast vs pre-fix R2** | Old keys-only R2 climbed train timing to **~0.42** while val stayed at floor ([EXP-20260804-08](#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring)). Fixed stack **does not** buy train timing either — both splits stay at noise |
| **Train ablation (5 songs, ep-4 ckpt)** | Matched timing **0.0078** (already floor); shuffle **0.0017** / zeros **0.0**. Zeros `query_cosine` **0.40**, tok **0.12** (decoder hears silence). Gate **FAIL** on **pointer** (matched≈corrupt within eps — no skill to collapse), token/query **PASS**. `logs/r2_content_pointer_fix_ablation_train.json` |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v_content_pointer/ar_onset_model.keras` · `logs/ladder_r2_content_pointer_train.log` · TensorBoard `callbacks/ar/ladder/logs` run `20260805-203515…` |
| **Conclusion** | **Not supported.** Decoder-audio wiring was necessary and tide-validated, but **not sufficient** for multi-song timing. Next lever is multi-song learning dynamics / recipe (why pointer timing never trains), not another identical R2 retrain |

### EXP-20260805-02: Decoder-audio fix — tide content-pointer passes query/zeros gate

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 20:18:58 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | Do the three fixes from [EXP-20260805-01](#exp-20260805-01-content-pointer-audio-signal-is-keys-only--decoder-is-audio-blind) / [NOTE-20260805-01](DISCUSSION_NOTES.md#note-20260805-01-pointer-gate-pass-was-keys-only--the-decoder-never-read-the-audio) restore real decoder audio grounding on tide? |
| **Changes** | (1) `legacy_inverted_attention_masks` default **False**; tide + ladder content-pointer configs set explicitly. (2) Audio gate requires zeros **token or query** collapse (keys-only must not pass). (3) PE-free pointer keys; dedicated `pointer_cross_attn` on `patch_embed`; decoder cross-attn = `patch_embed + Dense(memory)`; monotonic pointer mask; `use_soft_pointer_time: true`. |
| **Setup** | Retrain [`configs/ar/tide_overfit_content_pointer.json`](../../configs/ar/tide_overfit_content_pointer.json) 400 ep WSL GPU (~5 min). Ablation `--gate`. Logs: `logs/tide_content_pointer_fix_train.log`, `logs/tide_content_pointer_fix_ablation_gate.json`, `logs/tide_content_pointer_fix_probe.json` |
| **Before (EXP-01 tide)** | `query_cosine` under shuffle/zeros ≈ **1.0**; tokens unchanged; pointer collapsed via keys only; tide masks inverted (`valid_region_true_frac=0`) |
| **After — train** | Final `val_timing_match_teacher` / `val_overfit_gate` **0.9148**; token acc **1.0** |
| **After — ablation** | Matched timing **0.9416**, tok **1.0**. Shuffle: timing **0**, same_ptr **0**, query cos **1.0**, tok **1.0**. Zeros: timing **0.0016**, tok **0.118**, **query cos 0.418**. Gate **PASS** (pointer+token+query) |
| **Probe detail** | Post-encoder `memory` still nearly shuffle-invariant (cos **0.99**); `patch_embed` / `cross_memory` move (cos **0.18** / **0.14**). Pointer cross on PE-free stream: zeros moves query; shuffle query stays ~1 (pooling invariance) — gate keys off **zeros** for decoder proof |
| **Conclusion** | **Supported.** Keys-only false positive is closed on tide: silence moves `pointer_query` and tokens. **Next:** retrain R2 with the same stack and re-check val skill + gate |

### EXP-20260805-01: Content-pointer audio signal is keys-only — decoder is audio-blind

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-05 19:49:17 |
| **Track** | `model` + `metric` (AR) |
| **Question** | User rejected the regularization next-step: is the R2 content-pointer val floor a **bug**? Specifically: does “pointer gate PASS on train” mean the **decoder** reads audio, or only that `pointer_key(memory)` changes under corruption? |
| **Setup** | Offline probe on saved checkpoints (no retrain). Extract `pointer_query`, `pointer_key`, decoder state, token logits under matched vs shuffled `mert_patches`. Swap query/key combos. Also reconstruct `CrossAttentionMask` polarity from config. Scripts: `_tmp/pointer_query_key_probe/probe_audio_blind_query.py`, `probe_tide_and_attn.py`. Logs: `logs/r2_pointer_query_key_probe.json`, `logs/r2_tide_query_attn_probe.json` |
| **R2 content-pointer (ep-16 ckpt, 5 train songs)** | Mean `cos_query` **0.99997**, `cos_decoder` **0.99997**, `cos_token_logits` **1.00000** under shuffle; `cos_key` **0.365**. Patch acc: `q_matched×k_matched` **0.095** = `q_shuffle×k_matched` **0.095**; `q_matched×k_shuffle` **0.003**. `same_ptr_pred` **0.0** — pointer argmax flips **only** because keys change |
| **Tide content-pointer** | Same pattern: `cos_query` / `cos_decoder` / `cos_token_logits` ≈ **1.0**; `cos_key` **0.089**; `same_ptr_pred` **0.0**. Config omits `legacy_inverted_attention_masks` → default **True** → `keep_valid=False` → cross-attn `valid_region_true_frac` **0.0** (all real query/key pairs masked out; pad pairs kept) |
| **R2 mask polarity** | Explicit `legacy_inverted_attention_masks: false` → `valid_region_true_frac` **1.0** (correct). Decoder still audio-blind — correct masks are not sufficient; training never routes audio through cross-attn |
| **Re-read of prior ablations** | Train ablation token gate already **failed**: `same_token_as_matched` under shuffle **0.996** ([`logs/r2_content_pointer_ablation_train.json`](../../logs/r2_content_pointer_ablation_train.json)). Pointer-only PASS was misread as decoder grounding |
| **Conclusion** | **Root cause.** Content pointer attaches audio to **keys only**. Queries are a function of the teacher-forced token prefix (chart memory), so train can look “grounded” when keys match memorized queries and val cannot transfer. [EXP-20260804-08](#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring)’s “generalization, not wiring” is **superseded**: the wiring/architecture allows skipping decoder audio. Tide EXP-06 gate also did not prove decoder grounding. **Next:** fix mask default + force decoder audio sensitivity; do not regularize |

### EXP-20260804-07: R2 content-pointer rerun — zero val skill, audio gate fails on val

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 22:00:00 |
| **Track** | `train` + `metric` (AR) |
| **Question** | [EXP-20260804-06](#exp-20260804-06-content-based-pointer-restores-audio-grounding-and-still-passes-the-tide-gate) fixed the pointer on tide. Does the same head on **R2** (50 train / 50 val) produce **positive skill over null** — the first ladder number that could be real? |
| **Setup** | [`configs/ar/ladder_50t_50v_content_pointer.json`](../../configs/ar/ladder_50t_50v_content_pointer.json) — R2 ladder manifest, `pointer_head: content`, `checkpoint_metric: val_timing_match_teacher`, random init. WSL GPU ~**26** min; ES @ ep **66**, restored weights from ep **16** (only epoch where val `timing_match_teacher` peaked). `logs/ladder_r2_content_pointer_train.log` |
| **Train @ stop (ep 66)** | Train `timing_match_teacher` **0.4207**, val **0.0016**; train token acc **0.648**, val **0.415**; val Hungarian F1 **0.194**; val `pointer_loss` **14.99** nats |
| **Teacher-fed val (50 songs)** | `eval_ar_onset_offline.py --split val` — ordered **78/35439 = 0.0022**; Hungarian F1 **0.0598** (2121 TP / 33318 FP / 33318 FN). Null F1 @ matched count: ioi_shuffle **0.3088**. **Skill: F1 −0.3602, `timing_match` −0.0063**. `n_patch_wrong` **35310/35439 = 99.6%**; p50 abs err **14.1 s**. ~**130** s wall. `logs/r2_content_pointer_teacher_val.log` |
| **Audio ablation (5 val songs)** | `audio_ablation_ar_onset.py --split val --limit 5 --gate` — matched `timing_match` **0.0010**, shuffle **0.0003**, zeros **0.0000**; token acc **0.322** under matched/shuffle/zeros. **`audio_grounding_gate: FAIL`** (shuffle/zeros ≈ matched — val checkpoint is audio-blind in practice). `logs/r2_content_pointer_ablation_gate.json` |
| **Free-run** | Not run to completion — `eval_ar_onset_offline.py --ar_decode` teacher preflight fails (0% teacher timing on probe songs); would hit **2048** cap if forced ([never-EOS pathology](EXPERIMENT_LOG.md#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)). `logs/teacher_preflight_test_r2.log` |
| **vs index-head R2** | Old R2 ([EXP-20260802-01](#exp-20260802-01-ladder-r2-rerun-does-50-train-rows-beat-r1s-178-on-the-frozen-val-set)): val Hungarian F1 **0.2266**, teacher `timing_match` **~0.0029**. Content-pointer R2 is **worse** on F1 and no better on ordered timing — fixing audio grounding on tide did **not** transfer to 50-song val |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v_content_pointer/ar_onset_model.keras` · `logs/ladder_r2_content_pointer_train.log` · `logs/r2_content_pointer_teacher_val.log` · `logs/r2_content_pointer_ablation_gate.json` |
| **Conclusion** | **Not supported.** The pointer architecture fix is necessary but not sufficient: val still has **negative null skill**, the standing audio gate **fails** on held-out songs, and train/val gap remains extreme (train timing **0.42** vs val **0.002**). **Do not** scale to R3/R4 on this recipe. Val-transfer diagnosis: [EXP-20260804-08](#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring) |

### EXP-20260804-08: R2 content-pointer val-transfer diagnosis — generalization, not wiring

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 22:15:00 |
| **Track** | `metric` + `post` (AR) |
| **Question** | [EXP-20260804-07](#exp-20260804-07-r2-content-pointer-rerun--zero-val-skill-audio-gate-fails-on-val) failed the val audio gate. Is that a **multi-song wiring bug** (pointer still audio-blind at scale), a **bad checkpoint** (ES restored ep **16** while train metrics at ep **66** were **0.42**), or **pure generalization failure**? |
| **Setup** | Saved checkpoint `models_wsl/ar/ladder_50t_50v_content_pointer/ar_onset_model.keras` (ES restore @ ep **16**; `callbacks/ar/ladder/models/.../best.keras` only other copy). `eval_ar_onset_offline.py --split train\|val --limit 50` (teacher only); `audio_ablation_ar_onset.py --split train --limit 10` vs existing val ablation (5 songs); train-log epoch curve parsed. `logs/r2_content_pointer_val_transfer_diagnosis.json` |
| **Train log curve** | Best val `timing_match_teacher` @ ep **16**: val **0.0022**, train **0.0137**. @ ep **66** (stop): val **0.0016**, train **0.4207**. Val never beat ep-16 peak; late epochs are extreme overfit |
| **Per-song teacher (50 each)** | **Train** micro mean timing **0.0245** (median **0.019**, max **0.111** on `1267_07_watch_me_medium`; **1/50** songs ≥ **0.10**). **Val** mean **0.0020** (max **0.007**; **0/50** ≥ **0.10**). Patch wrong **95.5%** train vs **99.7%** val |
| **Audio ablation — train (10 songs)** | Matched timing **0.0364**, shuffle **0.0008**, zeros **0.0002**; ptr NLL **4.72** → **10.09** under shuffle; **same_ptr** shuffle **0.001**. **Pointer gate PASS** — corruption collapses timing |
| **Audio ablation — val (5 songs)** | Matched **0.0010**, shuffle **0.0003**, zeros **0.0000**; all variants at timing floor. **Gate FAIL** — not because pointer ignores audio on val, but because matched performance is already ~zero |
| **Checkpoint compare** | No per-epoch `.keras` beyond `best.keras` (ep **16**). Final `ar_onset_model.keras` matches ES restore; comparing ep **66** weights is **not possible** without retrain |
| **Artifacts** | `logs/r2_content_pointer_teacher_train.json` · `logs/r2_content_pointer_teacher_val.json` · `logs/r2_content_pointer_ablation_train.json` · `logs/r2_content_pointer_val_transfer_diagnosis.json` |
| **Conclusion** | **Superseded by [EXP-20260805-01](#exp-20260805-01-content-pointer-audio-signal-is-keys-only--decoder-is-audio-blind).** Originally read as generalization: train pointer ablation PASS, val at floor. The PASS was **keys-only** — decoder/`pointer_query` cosine ≈ 1 under shuffle — so “train reads audio” was overstated. No-ES rerun still useful as a negative: [EXP-20260804-09](#exp-20260804-09-no-es-120ep--val-timing-never-improves-after-ep-16) |

### EXP-20260804-09: No-ES 120ep — val timing never improves after ep 16

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 23:00:00 |
| **Track** | `train` + `metric` (AR) |
| **Question** | [EXP-20260804-08](#exp-20260804-08-r2-content-pointer-val-transfer-diagnosis--generalization-not-wiring) showed ES restored ep **16** while train timing at stop was **0.42**. Was early stopping hiding later val improvement? |
| **Setup** | [`configs/ar/ladder_50t_50v_content_pointer_no_es_120ep.json`](../../configs/ar/ladder_50t_50v_content_pointer_no_es_120ep.json) — same R2 recipe as [EXP-20260804-07](#exp-20260804-07-r2-content-pointer-rerun--zero-val-skill-audio-gate-fails-on-val) but `early_stopping_patience: 0`, `epochs: 120`, fresh output dir. WSL GPU ~**43** min. `logs/ladder_r2_content_pointer_no_es_train.log` |
| **Val timing curve** | Best val `timing_match_teacher` @ ep **16**: **0.0022** (train **0.0137**) — **same epoch and value** as ES run. Never higher through ep **120**; @ ep **120**: val **0.0015**, train **0.5114** |
| **Offline eval (best checkpoint)** | Teacher val (50 songs): timing mean **0.00196** — **identical** to [EXP-20260804-07](#exp-20260804-07-r2-content-pointer-rerun--zero-val-skill-audio-gate-fails-on-val). Train mean **0.0245**. `logs/r2_no_es_teacher_val.json` |
| **Audio ablation** | Train (10): matched **0.0364** → shuffle **0.0008** (pointer **PASS**). Val (10): matched **0.0017**, shuffle **0.0007** (gate **FAIL** at floor). `logs/r2_no_es_ablation_train.json` · `logs/r2_no_es_ablation_val.json` |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v_content_pointer_no_es_120ep/ar_onset_model.keras` · `callbacks/ar/ladder/models/20260804-221933-AR_ONSET-r2_50t_content_no_es-.../best.keras` |
| **Conclusion** | **Not supported.** ES was **not** the bottleneck: val timing peaked @ ep **16** and never improved through **104** additional epochs while train overfit to **51%**. Change the **training recipe** (regularization, LR, checkpoint metric), not epoch budget or SS |

### EXP-20260804-06: Content-based pointer restores audio grounding and still passes the tide gate

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 20:58:12 |
| **Track** | `model` + `train` (AR) |
| **Question** | Does replacing the `Dense(max_patches)` pointer with a content-based pointer make the tide overfit **fail** under corrupted audio while still passing on real audio? A fix that passes both is not a fix. |
| **Change** | `models.py`: `pointer_logits[k] = q(dec) · k(memory[k]) / √d` via new `ContentPointerLogits` + `pointer_query` / `pointer_key` projections, still masked by `MaskPointerLogits`. New `model.pointer_head` config: **`content`** (new default) or `index` (legacy, for rebuilding old runs). `build_ar_onset_inference_models` detects the head from the loaded model, so existing checkpoints still rebuild |
| **Setup** | [`configs/ar/tide_overfit_content_pointer.json`](../../configs/ar/tide_overfit_content_pointer.json) — champion tide recipe, `pointer_head: content`, fresh `model_output_dir` so the champion is not overwritten. 400 ep WSL GPU, ~6.4 min. `logs/tide_content_pointer_train.log` |
| **Real audio** | `val_overfit_gate` **0.9921**, `val_timing_match_teacher` **0.9921**, `token_accuracy` **1.0000**, `val_pointer_loss` **0.0018**. Offline: F1 **0.9858**, `timing_match` **0.9921**, `patch_wrong` **0.0000** |
| **Corrupted audio (the gate that matters)** | reverse F1 **0.1924** / `timing_match` **0.0016** / `patch_wrong` **0.9984** / NLL **27.78**; shuffle **0.1814** / **0.0016** / **0.9984** / **27.68**; zeros **0.1025** / **0.0032** / **0.9968** / **9.88**. Argmax patch agrees with the real-audio run on only **0.16–0.32%** of steps |
| **vs the old head** | Same corruption on the legacy champion left F1 at **0.9984** and `timing_match` at **1.0000** with `patch_wrong` **0.0000** ([EXP-20260804-05](#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer)). The head now **requires** the audio: corrupt it and the model falls to roughly the audio-blind chance floor, which is the expected floor, not a bug |
| **Cost** | Head size is now independent of `max_patches`: **1,443,750 → 295,680** params at production dims (`max_patches` 3750, `d_model` 384), a net **−1,148,070**. Logit count follows actual encoder length, so the `[..., :n_patches]` slice is no longer load-bearing |
| **Caveats** | Gate is **0.9921** vs the legacy champion's **0.9984** on real audio, at 400 ep with an untuned recipe carried over from the old head — the ~0.006 gap is not yet explained and may just be tuning. `cross_song` is degenerate on a single-song run (donor is the song itself) and is skipped rather than reported |
| **Artifacts** | `models_wsl/ar/tide_overfit_content_pointer/ar_onset_model.keras` · `logs/tide_content_pointer_train.log` · `logs/ar_audio_ablation_tide_content.log` · `_tmp/ladder_debug/audio_ablation_tide_content.json` |
| **Conclusion** | **Supported.** The pointer now reads the audio, and the tide gate is no longer passable blind. 95 AR tests pass, including new coverage that the content head length-generalizes, masks padding, rebuilds identically for inference, and that the legacy head still builds. R2 content-pointer rerun: [EXP-20260804-07](#exp-20260804-07-r2-content-pointer-rerun--zero-val-skill-audio-gate-fails-on-val) |

### EXP-20260804-05: The AR pointer never reads the audio — the head is absolute-index classification, not a pointer

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 20:32:40 |
| **Track** | `model` + `metric` (AR) |
| **Question** | [EXP-20260804-04](#exp-20260804-04-in-harness-null-floor-reproduces-the-finding-on-the-r2-checkpoint--both-gates-fail) leaves two stories that predict the same numbers: the pointer overfits, or the pointer ignores the encoder. Does corrupting **only** `mert_patches` — decoder prefix and all targets untouched — change teacher-forced output? |
| **Setup** | `_tmp/ladder_debug/audio_ablation.py`, WSL GPU, no retrain. Variants replace the valid patch region: `cross_song` (next val song's real features, tiled/cropped to length), `reverse`, `shuffle` (permuted patch axis), `zeros`. Padding rows stay zero and `patch_mask` is never touched, so pointer geometry is identical across variants. `logs/ar_audio_ablation.log`, `logs/ar_audio_ablation_tide.log` |
| **R2 on 12 val songs** | F1 **matched 0.1886** / cross_song **0.1883** / reverse **0.1881** / shuffle **0.1882** / **zeros 0.1885**. `timing_match` **0.0029** in all five. `patch_wrong` **0.9942** in all five. Pointer NLL **16.88** nats in all five (uniform **7.37**). Predicted patch identical to the matched run on **99.94–100.00%** of onset steps |
| **Tide champion (the PASS gate)** | F1 **0.9984** for matched, cross_song, reverse **and** shuffle — bit-identical. `timing_match` **1.0000** for all four; `patch_wrong` **0.0000** for all five variants including zeros. All-zero features still give F1 **0.9558**, `timing_match` **0.9890** |
| **Root cause** | `pointer_logits = Dense(max_patches)(decoder)` (`models.py:358`). Logit *k* is a learned output unit for **absolute patch index k**, not a score against `memory[k]`. Audio can only reach patch choice indirectly through cross-attention updating the decoder state, and measurably it does not. **1,443,750** params (`max_patches` **3750** × `d_model` **384**) spent on a head that cannot length-generalize and is not content-addressed |
| **Why the gate never caught it** | Single-song overfit is fully determined by the teacher-forced prefix, so an audio-blind model scores **1.0**. The tide gate passes with the audio **reversed**. Every AR "PASS" to date is compatible with zero audio grounding |
| **Consistency** | Explains [EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke) exactly: a model that does not read audio should land at the audio-blind floor, and every rung did. Also explains val `pointer_loss` **15.6–18.6** nats vs uniform **6.7** — a position prior fit to train songs is actively wrong on held-out ones |
| **Artifacts** | `_tmp/ladder_debug/audio_ablation.py` · `audio_ablation.json` · `audio_ablation_tide.json` · `logs/ar_audio_ablation.log` · `logs/ar_audio_ablation_tide.log` |
| **Conclusion** | **Root cause.** Not overfitting, not scale, not EOS, not the metric alone. Fix: replace the `Dense(max_patches)` head with a content-based pointer scoring decoder state against encoder memory (`logits[k] = q(dec) · k(memory[k]) / √d`), which is length-generalizing and forces audio into the patch decision. Add the audio-corruption ablation as a **standing gate** — any run whose score survives `shuffle` is not using audio |

### EXP-20260804-04: In-harness null floor reproduces the finding on the R2 checkpoint — both gates fail

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 19:58:18 |
| **Track** | `metric` + `post` (AR) |
| **Question** | [EXP-20260804-03](#exp-20260804-03-every-ladder-rung-is-at-or-below-an-audio-blind-baseline--the-metric-not-the-data-is-what-broke) measured the floor offline with a scratch script. Does the floor wired into `eval_ar_onset_offline.py` reproduce it on a real checkpoint, on a real GPU decode, without hand-held bookkeeping? |
| **Setup** | `eval_ar_onset_offline.py --config configs/ar/ladder_200t_50v.json --split val --limit 12 --ar_decode` on WSL GPU. R2 is the highest-teacher-F1 rung ever trained. No retrain, no config change — only the new reporting path. 12 val songs, **8925** GT onsets. `logs/rescore_r2_null.log` |
| **Teacher-fed** | Hungarian F1 **0.1886** (1683 TP / 7242 FP / 7242 FN). Null F1 at the same count: uniform **0.2257**, metronome **0.2696**, IOI-shuffle **0.2289**. **Skill over strongest null: F1 −0.1109, `timing_match` −0.0088** |
| **Free-run** | Hungarian F1 **0.0009** (4 TP / 176 FP / 8921 FN), `n_pred` **180** vs `n_gt` **8925**. Nulls **0.0112 / 0.0156 / 0.0156**. **Skill: F1 −0.0150, `timing_match` −0.0027** |
| **Order-aware** | `timing_match_teacher` **26/8925 = 0.0029**; `timing_match_ar_decode` **0/8925 = 0.0000**. Both **FAIL** against `target_times` and against the raw chart (**25/8925**) |
| **Where the error is** | `n_patch_wrong` **8873/8925 = 99.4%** — the pointer picks the wrong patch almost every step. Conditional on the right patch, residuals are fine (p50 **20.0 ms**), but absolute error is p50 **3.65 s** / p90 **13.3 s** / max **120 s**. Only **26** onsets land inside tolerance |
| **Reproducibility** | Free-run emits **15** onsets/song on the per-song path, matching the logged bare-R3 stop-length pathology; teacher F1 **0.1886** on 12 songs is consistent with **0.2266** on the full 50-row val. The offline scratch measurement and the in-harness one agree |
| **Artifacts** | `logs/rescore_r2_null.log` · `models_wsl/ar/ladder_200t_50v/ar_onset_model.keras` · `src/stepcovnet/onset_null_baseline.py` |
| **Conclusion** | **Supported.** The floor is now reported automatically beside every AR number and the gate fails on negative skill without a human noticing. R2 — the best rung — is **0.111 F1 below** a metronome that never hears the audio, and its pointer is wrong **99.4%** of the time under teacher forcing. Confirms the halt: fix encoder→pointer generalization, not scale |

### EXP-20260804-03: Every ladder rung is at or below an audio-blind baseline — the metric, not the data, is what broke

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 20:05:45 |
| **Track** | `metric` (AR) |
| **Question** | Hungarian F1 @ 20 ms is the metric every rung is compared and checkpointed on. What does a predictor that never hears the audio score on the frozen val set at the same prediction count? |
| **Setup** | CPU only, no retrain. Frozen ladder val (50 rows / 39 songs, **35439** GT onsets — matches the logged decodes exactly). GT via `charts.load_onset_times` + `clip_times_to_duration`, durations from `soundfile.info` capped at `max_audio_seconds=300`. Scored with the harness matcher (`onset_events.metrics.count_event_onset_errors_numpy`, tol **0.02 s**) and `timing_match`. Nulls snap to the 20 ms hop grid, as model predictions do. `_tmp/ladder_debug/null_baseline.py` |
| **Why a floor exists** | Val charts average **5.52** onsets/sec (mean IOI **181 ms**). A ±20 ms match window covers ~22% of the timeline, so emitting the right *number* of onsets scores well above zero without any audio |
| **Null F1 @ matched count** | r=**1.00**: uniform-over-duration **0.2250**, uniform-in-support **0.2370**, metronome grid **0.2745**, GT-IOI-shuffle **0.3359**, cross-song **0.3214**. r=**0.90**: 0.2171 / 0.2265 / 0.2605 / 0.3129 / 0.3113. r=**0.36**: 0.1284 / 0.1412 / 0.1544 / 0.1815 / 0.1784. r=**0.16**: 0.0717 / 0.0745 / 0.0789 / 0.0836 / 0.0927 |
| **Null stability** | 20 seeds: uniform-in-support r=1.0 **0.2373 ± 0.0019**; metronome **0.2745 ± 0.0000**; IOI-shuffle **0.3370 ± 0.0087**. Gaps below are far outside seed noise |
| **Every rung vs its own floor** | Teacher (r=1.0): R1 **0.178**, R2 **0.2266**, R3 **0.1991**, R3+density **0.2059** — all **below** even the weakest null (0.2250). Free-run: R2 bare **0.132** @ r=0.36 vs 0.128–0.182; R2+meter **0.234** @ r=0.82 vs 0.208–0.295; R2+onset_density (champion) **0.263** @ r=0.90 vs 0.217–0.313 — ties the metronome (0.2605), loses to IOI-shuffle; R3+density **0.034** @ r=0.16 vs 0.072–0.093 |
| **The density "wins" were count-matching** | Free-run F1 tracks `pred/GT`, not timing: 0.36→**0.132**, 0.82→**0.234**, 0.90→**0.263**, against nulls 0.154, 0.250, 0.261 at the same ratios. [EXP-20260804-02](#exp-20260804-02-r3-early-eos-is-the-scaling-failure--length-force-recovers-free-run)'s `min_onset_tokens=200` recovery (**0.200** @ r≈1.05) is also **below** the r=1.05 floor (0.239–0.336) |
| **Order-aware metric is discriminative** | Same nulls under `timing_match`: **0.000–0.029** (floor ≈ 0). Measured val `timing_match_teacher` is **0.0026** (12/2877 on the 5-song probe) — also at the floor. Independent of the chance argument: val `pointer_loss` **15.6–18.6** nats vs **ln(~810 patches) ≈ 6.7** for uniform guessing, and `n_patch_wrong` **0.99–1.00** on all 50 songs with teacher-forced median timing error **1.1–73 s** |
| **Not underfitting** | Train vs val at each run's best epoch: R2 **0.746 / 0.227**, R3 **0.529 / 0.199**, R2+density **0.736 / 0.227**, R3+density **0.539 / 0.206**; train token accuracy **0.98** vs val **0.37**; train `pointer_loss` **0.01–0.04**. The model fits training songs and transfers nothing |
| **Artifacts** | `_tmp/ladder_debug/null_baseline.py` · `null_baseline.json` · `per_song_audit.py` · `train_val_gap.py` · `logs/ladder_null_baseline.log` · `logs/ladder_train_val_gap.log` · `logs/ladder_null_wiring_probe.log` |
| **Conclusion** | **Supported.** No AR rung has ever cleared an audio-blind baseline on the frozen val set, so rung-to-rung deltas (0.178 → 0.227 → 0.199) and the density ablation ranking are not measurements of skill. Scale was never the binding problem. Fixes landed: `stepcovnet.onset_null_baseline` (floor + `skill_over_null`) wired into `eval_ar_onset_offline.py`, and `timing_match_teacher` now published on multi-song runs so checkpoints can be selected on a near-zero-floor metric. **Do not** run R4, further EOS work, or more density variants until a rung shows positive skill |

### EXP-20260804-02: R3 early-EOS is the scaling failure — length force recovers free-run

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 12:25:23 |
| **Track** | `metric` + `post` (AR) |
| **Question** | Is the R3 free-run collapse caused by early `<EOS>` (recoverable by decode length control), rather than destroyed timing? |
| **Setup** | Recompute from logged full-val decodes + existing bare-R3 probes in `_tmp/ladder_debug/r3_probe*.json` (5 songs; preds match bare R3 stop@**15**, not density). No new train. |
| **Full-val compare** | R2+onset_density free **0.263**, pred/GT **0.90**, corr(pred,GT) **0.78**. R3+onset_density free **0.034**, pred/GT **0.16**, corr **−0.42**, **36/50** songs stop at **15** or **19**. Bare R3: free **0.003**, all **50** @ **15**. |
| **Length-force probe** | none / `eos_logit_bias=+3`: free **0.0007**, stop@**15**. `min_onset_tokens=200`: free **0.1999**, all stop@**603**, pred/GT **1.05**. |
| **Reading** | Forcing past early EOS recovers usable free-run on the probe set — timing content after step 15 exists. Density that works at 50t does not teach termination at 200t. Positive EOS logit bias is inert; hard min-length is the lever. |
| **Artifacts** | `_tmp/ladder_debug/r3_probe*.json` · `_tmp/ladder_debug/r3_onset_density.json` · [NOTE-20260804-01](DISCUSSION_NOTES.md#note-20260804-01-scale-up-fails-on-eos-termination-timing-is-recoverable-once-length-is-forced) |
| **Conclusion** | **Supported.** Next: full 50-val R3+onset_density with density-derived `min_onset_tokens`; then fix train-side EOS weighting ([NOTE-20260724-01](DISCUSSION_NOTES.md#note-20260724-01-eos_token_weight_scale-is-a-no-op-under-token_class_weight-none)). Do not R4. |

### EXP-20260804-01: R3 + onset_density lifts teacher but free-run still collapses

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-04 06:47:07 |
| **Track** | `train` + `metric` (AR) |
| **Question** | At 200 train rows, does **onset_density** fix R3's early-EOS free-run collapse (@ **15** toks, F1 **0.003**) while beating R3 teacher **0.199**? |
| **Setup** | [`configs/ar/ladder_200t_50v_density_onset.json`](../../configs/ar/ladder_200t_50v_density_onset.json) — R3 ladder manifest + `density_conditioning: onset_density`. Train ES @ ep **462**, best @ **412** (~7.2 h WSL GPU); offline val decode ~11 min |
| **Teacher** | Offline Hungarian F1 **0.2059** (matches in-train peak; **≥** bare R3 **0.199**, **<** R2 **0.227**) |
| **Free-run** | Hungarian F1 **0.0340** vs bare R3 **0.0030** and R2+onset_density **0.2630**; **pred/GT = 0.162** (**5728** / **35439**). TP/FP/FN **698 / 4930 / 34741** |
| **Length / eos_trace** | **50/50** stopped on `<EOS>`; decode length sum **5728** (~**115** preds/song avg vs bare R3 **15**). Many songs still early-stop; density partial fix only |
| **vs bare R3** | Teacher holds at 200t; free-run improved **10×** (0.003 → 0.034) but still far below usable. More rows without a stable termination recipe does not transfer R2 free-run gains |
| **Artifacts** | `models_wsl/ar/ladder_200t_50v_density_onset/ar_onset_model.keras` · `logs/ladder_r3_density_onset_train.log` · `logs/ladder_r3_density_onset_val_decode.log` |
| **Conclusion** | **Not supported** for scale-up. **Champion for free-run remains R2 + onset_density (0.263).** Pause R4 until R3 termination is understood or accept R2 as the generation rung |

### EXP-20260803-03: Onset_density conditioning on R2 beats meter on free-run

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-03 18:09:19 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | Does conditioning on measured onset rate (`n_onsets/duration/15`) beat **meter** density on R2 free-run? |
| **Setup** | [`configs/ar/ladder_50t_50v_density_onset.json`](../../configs/ar/ladder_50t_50v_density_onset.json) — R2 recipe with `density_conditioning: onset_density`. Train ES @ best epoch **484** (~2.4 h WSL GPU); offline val decode ~54 min |
| **Teacher** | Offline Hungarian F1 **0.2271** (matches R2 / meter density) |
| **Free-run** | Hungarian F1 **0.2630** vs meter **0.2338** and R2 bar **0.132**; **pred/GT = 0.901** (**31908** / **35439**). TP/FP/FN **8855 / 23053 / 26584** |
| **Length / eos_trace** | **50/50** stopped on `<EOS>`; decode length sum **32008** (~640 preds/song). `eos_trace` final_mean ~**1.0** |
| **vs meter density** | Meter ([EXP-20260803-02](#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run)): free-run **0.234**, pred/GT **0.82**, varied stop lengths. Onset_density wins on F1 and pred/GT; meter had more length diversity |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v_density_onset/ar_onset_model.keras` · `logs/ladder_r2_density_onset_train.log` · `logs/ladder_r2_density_onset_val_decode.log` |
| **Conclusion** | **Supported.** Use **onset_density** for R3 scale-up and blind customer decode (no simfile meter). Next: **R3 (200t)** with same conditioning |

### EXP-20260803-02: Meter density conditioning on R2 breaks the 252 stop and lifts free-run

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-03 14:26:44 |
| **Track** | `model` + `train` + `metric` (AR) |
| **Question** | Does conditioning the decoder on normalized chart `#METER` supply the missing length prior and beat R2's free-run bar of **0.132**? |
| **Setup** | New `density_conditioning: meter` path (global decoder embed from `meter/32`). Config [`configs/ar/ladder_50t_50v_density.json`](../../configs/ar/ladder_50t_50v_density.json) — otherwise identical to R2 ([`ladder_50t_50v.json`](../../configs/ar/ladder_50t_50v.json)): 50/50 rows, 500 ep, LR **1e-4**, ES patience **50**, `scheduled_sampling_max_p: 0`. ~2 h 37 m train (WSL GPU), then offline val decode (~55 min) |
| **Teacher** | Offline Hungarian F1 **0.2269** (matches R2 **0.227**). In-train best `val_aux_f1_hungarian` peaked ~**0.201** — checkpoint selection still restores a competitive teacher |
| **Free-run** | Hungarian F1 **0.2338** vs R2 bar **0.132**; **pred/GT = 0.820** (**29073** / **35439** onsets). TP/FP/FN **7540 / 21533 / 27899** |
| **Length / eos_trace** | **50/50** stopped on `<EOS>`. Stop-length modes: **589** (18 songs), **675** (9), **252** (8), **894** (8), **272** (5), **934** (2) — **not** the uniform **252** pathology. `eos_trace` final_mean ~**1.0**, `n_songs_ge_half` **50** |
| **vs R2 / SS** | R2 fixed **252** for all 50 songs @ free-run F1 **0.132** ([EXP-20260802-03](#exp-20260802-03-ladder-r2-offline-val-free-run--eos_trace)); SS unchanged ([EXP-20260803-01](#exp-20260803-01-scheduled-sampling-now-actually-running-does-not-improve-free-run-on-r2)) |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v_density/ar_onset_model.keras` · `callbacks/ar/ladder/logs/…-AR_ONSET-r2_density_50t-…` · `logs/ladder_r2_density_train.log` · `logs/ladder_r2_density_val_decode.log` |
| **Conclusion** | **Supported.** Meter conditioning is the first lever that both breaks the fixed-length stop and materially lifts free-run F1. Next: same conditioning on **R3** (early EOS @ **15**) or ablate **onset_density** vs **meter** |

### EXP-20260803-01: Scheduled sampling, now actually running, does not improve free-run on R2

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-03 02:23:15 |
| **Track** | `train` + `metric` (AR) |
| **Question** | With the trace-time defect fixed ([EXP-20260802-05](#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step)), does scheduled sampling on the R2 recipe beat the free-run bar of **0.132**? |
| **Setup** | [`configs/ar/ladder_50t_50v_ss.json`](../../configs/ar/ladder_50t_50v_ss.json) — random init, `scheduled_sampling_max_p: 0.3` with warmup **150** and ramp **250** (so `p` reaches 0.3 at ep 400), 500 ep, LR **1e-4**, ES patience **100**; every other knob identical to R2. 2 h 43 m train, then offline val decode (26 min) with the same command shape as [EXP-20260802-03](#exp-20260802-03-ladder-r2-offline-val-free-run--eos_trace) |
| **Teacher result** | Best `val_aux_f1_hungarian` **0.2235 @ ep 499** (R2: **0.2266 @ ep 470**); offline teacher F1 **0.2235** vs R2 **0.2267** |
| **Free-run result** | **0.1313** vs R2's **0.1319** — no improvement. `pred/GT` **0.3555** in both, and all 50 val songs again stop on `<EOS>` at exactly **252** onset tokens, for an identical **12600** predicted onsets. R3 for contrast: **0.0030**, stop at **15** |
| **SS did engage** | The inert run is bit-identical to R2 on all 500 epochs (500/500 equal to 4 dp); this run diverges from R2 at **ep 2** and shares only 8 of 500 epoch values. Contrast confirmed against the same baseline the defect was found with |
| **Caveat — not seed-exact vs R2** | Divergence begins in **warmup**, where `p = 0`, so part of the gap is not scheduled sampling. `tf.cond` traces the sampling branch, and its `tf.random.uniform` plus the probe pass's dropout ops shift the graph-level op-seed sequence, so the teacher branch draws different dropout masks than the pre-fix graph did. Size of that noise: warmup (ep 1–150) mean Δ **+0.0010**, max |Δ| **0.0066**. Size of the post-ramp effect (ep 401–500, `p = 0.3`): mean Δ **−0.0015**, max |Δ| **0.0204**. The SS effect on teacher F1 is at or below the seed-shift noise floor |
| **Reading** | Scheduled sampling at `p = 0.3` moves neither teacher F1 nor free-run F1 outside noise, and leaves the length pathology **exactly** where it was. Feeding the model its own tokens during training does not teach it when to stop: termination is invariant to the fix, which argues the fixed **252** is not error accumulation but a missing length/density signal — the same gap flagged in [NOTE-20260803-01](DISCUSSION_NOTES.md#note-20260803-01-difficulty-is-unconditioned-and-unfiltered-but-it-is-not-what-caps-the-ladder) |
| **Artifacts** | `callbacks/ar/ladder/logs/20260802-230953-AR_ONSET-r2_ss_50t-…` · `logs/ladder_r2_ss_train.log` · `logs/ladder_r2_ss_val_decode.log` |
| **Conclusion** | **Not supported.** SS is not the lever for exposure bias here. Free-run bar stays R2's **0.132**. Next candidate is conditioning the decoder on target onset density rather than more sampling-schedule tuning |

### EXP-20260802-05: Scheduled sampling on R2 is a no-op — the branch is compiled out of `train_step`

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-02 17:12:38 |
| **Track** | `train` (AR) |
| **Question** | Does scheduled sampling on the R2 recipe improve free-run F1 over the **0.132** baseline? |
| **Setup** | [`configs/ar/ladder_50t_50v_ss.json`](../../configs/ar/ladder_50t_50v_ss.json) — **random init** (no warm start, per [AR_SCALING_LADDER](AR_SCALING_LADDER.md) § 3), `scheduled_sampling_max_p: 0.3`, warmup **150**, ramp **250**, 500 ep, LR **1e-4**, ES patience **100**. Every other knob identical to R2 so SS is the only variable. ~2 h 15 m wall |
| **Result** | Best `val_aux_f1_hungarian` **0.2266 @ ep 470**; ep 500 **0.2236** — **identical to R2 ([EXP-20260802-01](#exp-20260802-01-ladder-r2--50-train-rows-beats-r1-on-frozen-val)) to 4 dp on all 500 epochs**. Same seed, same data, and an inert feature reproduce the baseline exactly |
| **Root cause** | `ArOnsetTrainingModel.train_step` gates the SS path on a **Python** conditional over a **Python float attribute** (`if self.scheduled_sampling_p > 0.0`). Keras wraps `train_step` in a `tf.function`, so that branch is resolved at **trace time**. `__init__` seeds the probability via `scheduled_sampling_for_epoch(-1, …)`, which returns **0.0** for every config (`-1 < warmup_epochs` for any warmup ≥ 0), so the graph is always traced with the branch removed. `ScheduledSamplingRampCallback` then mutates a plain Python attribute, which does **not** trigger retracing. Secondary defect: `scheduled_sampling_p` is passed into `build_scheduled_decoder_inputs` as a Python float, so even a live branch would freeze `p` at its trace-time value |
| **Repro** | Minimal `tf.function` with a Python-attribute branch: returns the `p=0` result after setting `p=0.3`, and only picks up the new branch when a **new input shape** forces a retrace |
| **Nondeterminism risk** | Because retracing *is* triggered by new input shapes, a config using `dynamic_padding` + length buckets can activate SS partway through a run if an unseen bucket shape first appears after `p > 0`. Whether SS runs at all is therefore data-order dependent, not configured |
| **Test gap** | `tests/onset_ar/trainers_test.py` has five scheduled-sampling tests; all five assert the **arithmetic** of `scheduled_sampling_for_epoch`. None assert that a nonzero `p` changes a training step. The suite is green on dead code |
| **Scope** | Affects every SS run in this repo, since the trace-time probability is 0.0 regardless of config. Results previously attributed in part to an SS ramp — notably [EXP-20260628-01](#) (`gate-ar-decode` v2–v4) — should be re-read as **not** demonstrating SS |
| **Artifacts** | `callbacks/ar/ladder/logs/20260802-145100-VOID-inert-ss-…` (renamed so the flat curve is not mistaken for the rerun) · `logs/ladder_r2_ss_train.log` |
| **Fix** | Landed same session. `scheduled_sampling_p` is now a non-trainable `tf.Variable`; input selection moved into `_decoder_inputs_for_step` using `tf.cond`, so the ramp callback's `assign` is read at run time with no retrace. Probe forward pass now skips the discarded loss computation. Regression test `test_scheduled_sampling_applies_after_train_step_is_traced` traces the selector, then asserts a post-trace `p = 1.0` changes decoder inputs — **verified to fail** against the original float-attribute code (traced graph still returned teacher inputs at `p = 1.0`) |
| **Fix verified on GPU** | 5-epoch 10-song A/B (`max_p` **1.0** vs **0.0**, warmup **2**, ramp **1**): epochs 1–2 **bit-identical** (loss 54.2058 / 53.0184 both), epochs 3–5 **diverge** (51.2107 vs 51.1122; 46.6786 vs 47.0173; 37.6176 vs 37.0898). Sampling engages at the ramp boundary and nowhere earlier — the signature the pre-fix run could not produce |
| **Conclusion** | **Void — not a result about scheduled sampling.** The exposure-bias hypothesis remains untested; rerun `configs/ar/ladder_50t_50v_ss.json` on the fixed code. Free-run bar remains R2's **0.132** |

### EXP-20260802-04: Ladder R2 vs R3 offline val free-run compare

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-02 13:07:26 |
| **Track** | `metric` (AR) |
| **Question** | On the same frozen 50-row val set, how do R2 (50t) and R3 (200t) free-run + `eos_trace` compare? |
| **Setup** | Same decode recipe: `eval_ar_onset_offline.py --split val --ar_decode`. R2: [EXP-20260802-03](#exp-20260802-03-ladder-r2-offline-val-free-run--eos_trace). R3: `configs/ar/ladder_200t_50v.json` · `models_wsl/ar/ladder_200t_50v/ar_onset_model.keras` · ~**3.4** min wall |
| **Compare** | Same frozen val (50 songs). Teacher F1 **0.227** (R2) vs **0.199** (R3). Free-run F1 **0.132** vs **0.003**. Pred/GT **0.36** (12600/35439) vs **0.021** (750/35439). EOS: **50/50** stop @ **252** onsets (R2) vs @ **15** onsets / `len=17` (R3). `eos_trace` final ~**1.0** both; ge_half @ **252** vs **15** |
| **Process exit** | Shell exit **1** from PowerShell stderr→NativeCommandError under `2>&1 \| Tee`; JSON completed |
| **Artifacts** | `logs/ladder_r3_val_decode.log` · `logs/ladder_r3_val_decode_clean.json` · R2: `logs/ladder_r2_val_decode_clean.json` |
| **Conclusion** | **Supported** as a compare. More train rows made free-run **worse** (classic early-EOS under-generation), not better. R2 remains the free-run baseline for scheduled sampling; do not SS on R3 first |

### EXP-20260802-03: Ladder R2 offline val free-run + eos_trace

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-02 12:59:54 |
| **Track** | `metric` (AR) |
| **Question** | On best ladder teacher (**R2**), does offline free-run hold on the frozen 50-row val set, and what does `eos_trace` look like? |
| **Setup** | [`configs/ar/ladder_50t_50v.json`](../../configs/ar/ladder_50t_50v.json) · `models_wsl/ar/ladder_50t_50v/ar_onset_model.keras` · `eval_ar_onset_offline.py --split val --ar_decode` · WSL GPU · ~**22.7** min wall |
| **Teacher** | Hungarian F1 **0.2267** (matches train best **0.2266**). Ordered @ 20 ms vs `target_times`: **53/35439** (**0.0015**) — harsh gate still near-zero |
| **Free-run** | Hungarian F1 **0.1319**; ordered **7/35962** (**0.00019**); **12600** pred / **35439** GT onsets (**pred/GT = 0.36**); TP/FP/FN **3167 / 9433 / 32272** |
| **eos_trace** | **50/50** songs `stopped_on_eos`. Aggregate: first_mean **~4.9e-6**, final_mean **~1.0**, max_mean **~1.0**, `n_songs_ge_half` **50**. Every song: decode length **254**, **252** onset tokens, `first_step_ge_half` **252** — fixed-length stop, not song-adaptive (GT onsets range **69–1300**, mean **~709**; only **5/50** songs have GT ≤ 252) |
| **vs prior multi-song free-run** | Better than July 200t free-run F1 **0.036** (~70 preds/song) ([EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode)); opposite of never-EOS rebuild ([EXP-20260724-04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)). Still a collapse vs teacher and vs GT length |
| **Process exit** | Shell exit code **1** from PowerShell `1> … 2>&1 \| Tee` (UTF-16 capture + broken `$LASTEXITCODE`); JSON report completed cleanly |
| **Artifacts** | `logs/ladder_r2_val_decode.json` (UTF-16 + stderr prefix) · `logs/ladder_r2_val_decode_clean.json` |
| **Conclusion** | **Partial.** R2 free-run bar is set: **0.132** F1 with systematic **252**-onset EOS truncation. Exposure bias remains the live train-time lever — proceed to **scheduled sampling** on the R2 recipe rather than climbing R4 for teacher F1 |

### EXP-20260802-02: Ladder R3 — 200 train rows does not beat R2

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-02 09:28:18 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Rung **R3** of [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md): does 200 train rows beat R2's **0.227** on the frozen val set? |
| **Setup** | [`configs/ar/ladder_200t_50v.json`](../../configs/ar/ladder_200t_50v.json) — `training_index_ladder_v1_200t_50v.json` (200/50 rows; nested train vs R2; same frozen val). MERT: 222 extracted (~21m). WSL GPU, `cache_max_samples: 250`, ES patience **50**, checkpoint `val_aux_f1_hungarian`. ~23 GiB WSL ceiling |
| **Result** | Best `val_aux_f1_hungarian` **0.1991 @ ep 361** (restored); early stop @ ep **411**. Best `val_loss` **14.52 @ ep 44** (F1 **0.1298** — D3 again) |
| **vs R2 / R1** | **0.199 < 0.227** (R2); still **> 0.178** (R1). Teacher F1 is **not** monotonic in train size on this ladder |
| **Caveat** | Fixed **epoch** budget: R3 sees more steps/epoch (200 vs 50) but ES cut at 411; not a matched step-budget comparison. Exposure bias / free-run still unmeasured on these checkpoints |
| **Artifacts** | `models_wsl/ar/ladder_200t_50v/ar_onset_model.keras` · `callbacks/ar/ladder/logs/20260802-032843-AR_ONSET-r3_200t-…` · `logs/ladder_r3_train.log` · `logs/ladder_r3_mert_extract.log` |
| **Conclusion** | **Partial.** R3 completes but fails the “beat R2” bar. Best ladder teacher remains R2. Prefer free-run / `eos_trace` on R2 over blind R4 |

### EXP-20260802-01: Ladder R2 — 50 train rows beats R1 on frozen val

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-02 02:12:04 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Rung **R2** of [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md): does 50 train rows beat R1's **0.178** on the frozen val set? |
| **Setup** | [`configs/ar/ladder_50t_50v.json`](../../configs/ar/ladder_50t_50v.json) — `training_index_ladder_v1_50t_50v.json` (50/50 rows; nested train vs R1; same frozen val). MERT: 88 extracted (~8m). WSL GPU (RTX 3070 Ti), `cache_max_samples: 128`, ~**15–17 s**/ep. Memory mitigation: `%UserProfile%\.wslconfig` `memory=24GB` → guest **~23 GiB** (was 15 GiB) |
| **Result** | Best `val_aux_f1_hungarian` **0.2266 @ ep 470** (restored); ep 500: **0.2236**. Best `val_loss` **20.41 @ ep 42** (F1 only **0.1205** there — D3 again). Train overfits after ~ep 60 (val pointer/token climb); val F1 keeps slow gains to ~ep 470 |
| **vs R1** | **0.227 > 0.178** (~**1.27×**). First trustworthy “more train rows helps” step on a fixed val set |
| **vs abort** | Prior attempt died at ep **152** under 15 GiB ([EXP-20260726-01](#exp-20260726-01-ladder-r2-aborted-at-ep-152--wsl-vm-terminated-mid-run)); this rerun held through **500** ep with ~7–8 GiB guest used mid-run |
| **Process exit** | Shell reported exit code **1** from WSL/TF stderr after a clean Keras restore of best ep **470** — not a training failure |
| **Artifacts** | `models_wsl/ar/ladder_50t_50v/ar_onset_model.keras` · `callbacks/ar/ladder/logs/20260801-235427-AR_ONSET-r2_50t-…` · `logs/ladder_r2_train.log` · `logs/ladder_r2_mert_extract.log` · `logs/ladder_r2_wsl_memory_probe.log` |
| **Conclusion** | **Supported.** R2 lands: **0.227** val Hungarian F1. Memory mitigation unblocks the ladder. Next rung **R3** (200t). Free-run / scheduled sampling still deferred |

### EXP-20260726-01: Ladder R2 aborted at ep 152 — WSL VM terminated mid-run

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-26 02:21:59 |
| **Track** | `train` (AR) |
| **Question** | Rung **R2** of [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md): does 50 train rows beat R1's **0.178** on the frozen val set? |
| **Setup** | [`configs/ar/ladder_50t_50v.json`](../../configs/ar/ladder_50t_50v.json) — `training_index_ladder_v1_50t_50v.json` (50 train rows / 49 songs, nested superset of R1's 10; same frozen **50**-row / 42-song val). MERT: 36 extracted, **55 reused** from R1 (2m23s). WSL GPU, 50 steps/ep, ~**13 s**/ep |
| **Progress before abort** | **152** epochs (~32 min). Best `val_aux_f1_hungarian` **0.0204 @ ep 146**; best `val_loss` **19.52 @ ep 87`. Ahead of R1 at matched epochs: ep 150 gives **0.0192** here vs **0.0087** in R1 (~2.2×), with `val_loss` **22.9** vs **28.3** |
| **Abort** | Training stopped at **01:28**; the WSL VM shut down at **01:29:22** (Hyper-V VmSwitch port disconnect, event 69). No new VM until a status probe cold-booted one at 02:19. No Windows sleep/resume events in the window. The trainer left no traceback — only cancelled-rendezvous lines, consistent with abrupt termination |
| **Suspected cause** | Guest memory. No `.wslconfig`, so WSL gets **15 GB** of a 31 GB host. R2 caches **100** samples (`cache_max_samples: 128`) at ~50 MB of MERT features each ≈ **5.5 GB**, up from R1's 60 (~3 GB). An in-guest OOM kill would leave no processes, and WSL then terminates the VM — matching the observed sequence. **Unconfirmed:** `dmesg` was wiped by the VM restart, and the cache fills during epoch 1, so a plain cache blowup should have aborted far earlier than ep 152. A slow leak or host-side pressure fits the timing better |
| **Salvage** | `callbacks/ar/ladder_50t_50v/models/20260726-005833-…/best.keras` (ep ~146) survived — the first checkpoint the AR trainer has written since the monitor fix in [EXP-20260725-02](#exp-20260725-02-ladder-r1--first-rung-on-the-frozen-val-set-10-train-rows). Before that fix this crash would have left nothing |
| **Artifacts** | `logs/ladder_r2_train.log` · `logs/ladder_r2_mert_extract.log` |
| **Conclusion** | **Aborted, not a result.** R2 is unscored; the partial curve is encouraging but must not be compared to R1's completed 0.178. Rerun needs a memory mitigation (raise the WSL ceiling via `.wslconfig`, or lower `cache_max_samples`) and a guest-memory probe to convert the suspicion into evidence |

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-25 23:45:40 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Rung **R1** of [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md): what does 10 train rows score on the **frozen** 50-row val set, establishing the baseline every later rung must beat? |
| **Setup** | [`configs/ar/ladder_10t_50v.json`](../../configs/ar/ladder_10t_50v.json) — `training_index_ladder_v1_10t_50v.json` (10 train rows / 10 songs; **50** val rows / **42** songs; source pinned `1fac516f06fe69b6…`). MERT extracted for the 32 uncached audio (2m14s). WSL GPU, 10 steps/ep, ~**5 s**/ep, 500 ep in ~**41 min** |
| **Result** | Best `val_aux_f1_hungarian` **0.1784 @ ep 497** (ep 500: **0.1784**). Curve: **0.0003** @ ep 10 → 0.0021 @ 50 → 0.0087 @ 150 → 0.0292 @ 250 → 0.128 @ 350 → **0.178** @ 500. Best `val_loss` **26.59 @ ep 104** |
| **`val_loss` vs F1 (D3 confirmed)** | The `val_loss` optimum sits at ep **104**, where val F1 is **0.0055** — **32×** worse than the F1-selected checkpoint. Selecting on `val_loss`, as [EXP-20260724-01/02/04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction) did, discards nearly all of the model's timing skill. `val_loss` rises from ep ~150 onward while F1 climbs monotonically to the end |
| **Defect found** | Early stopping and best-checkpoint saving were **silently inert** all run: `resolve_checkpoint_metric("val_aux_f1_hungarian")` returns the legacy key `val_event_onset_f1`, but the AR trainer publishes only canonical names outside single-song overfit runs. Keras warned `metric val_event_onset_f1 which is not available` each epoch, and `callbacks/ar/ladder_10t_50v/models/` was never created. Fixed by `MetricAliasCallback` (dual-publishes both spellings ahead of the monitors); test added |
| **Impact on this run** | None material — F1 peaked at ep 497 of a 500-ep budget, so the saved final weights are within **0.0000** of the best epoch. The number stands; only the artifact-selection path was broken |
| **Caveat** | val F1 was **still climbing** at the budget end, so **0.178 is a lower bound**, not a converged value. A fixed *epoch* budget also gives larger rungs proportionally more gradient steps (10 rows × 500 ep = 5k steps vs 50 rows × 500 ep = 25k) |
| **Artifacts** | `models_wsl/ar/ladder_10t_50v/ar_onset_model.keras` · TB logs not retained (legacy per-rung tree, since consolidated into `callbacks/ar/ladder/`) · `logs/ladder_r1_train.log` · `logs/ladder_r1_mert_extract.log` |
| **Conclusion** | **Supported.** The ladder has its first trustworthy rung: **0.178** val Hungarian F1 on a frozen, reproducible val set with a pinned source manifest. R2 (50 rows, same val set, nested train set) must beat it |

### EXP-20260725-01: Ladder R0 — MERT extraction is bit-identical; tide champion artifact was overwritten

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-25 23:06:36 |
| **Track** | `pre` + `metric` (AR) |
| **Question** | Rung **R0** of [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md): is this machine's pipeline healthy — do locally extracted MERT features match the known-good ones, and does the graduated tide champion still hit **634/634**? |
| **Feature check** | Re-extracted `data/v2/test/tide.ogg` to a scratch dir (`--device cuda`, 4.4 s) and compared to the on-disk `tide.mert.npy`: shape **(12852, 1024)** float32 both, **`np.array_equal` True**, max abs diff **0.0**. Local MERT extraction is **bit-for-bit reproducible** |
| **Champion gate** | [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) + `models_wsl/ar/tide_overfit/ar_onset_model.keras`, `--ar_decode`: teacher ordered **627/634** (0.9890), free-run ordered **622/634** (0.9811), Hungarian F1 **0.9811** teacher / **0.9795** free-run. Decode length **636**, EOS fires correctly (`first_step_ge_half` **634**, max prob **0.978**, mean **0.0017**) |
| **Artifact provenance** | [`configs/ar/tide_overfit.manifest.json`](../../configs/ar/tide_overfit.manifest.json) records this exact path as the graduated **v8** champion at **634/634** teacher **and** free-run, dated **2026-06-30**. The file on disk is dated **2026-07-02**, and `callbacks/ar/tide_overfit/models/20260702-000621-…/best.keras` returns **identical** numbers — a later run reusing the same `model_output_dir` overwrote the graduated weights |
| **Conclusion** | **Partial.** The two integrity questions separate cleanly. Feature extraction and the decode path are **healthy** — features are bit-identical and a real checkpoint decodes at **98.9%** teacher with correct termination, so [NOTE-20260725-01](DISCUSSION_NOTES.md#note-20260725-01-the-50t50v-rebuild-gap-is-val-side-not-recipe-or-early-stopping)'s feature hypothesis is **eliminated**. But the perfect-overfit bar cannot be verified from stored weights, because the champion artifact no longer exists in this clone. Third artifact loss in a week, after the July 16–24 AR checkpoints and the 200t/50v model |
| **Follow-up** | Reusing `model_output_dir` across runs (per the rerun-hygiene convention) silently destroys graduated checkpoints. Graduated artifacts need a write-protected path or a copy under `configs/ar/versions/`. Retraining tide from scratch is the only way to restore an R0 reference |
| **Artifacts** | `logs/ladder_r0_mert_tide.log` · `logs/ladder_r0_tide_gate.log` · `logs/ladder_r0_tide_best_ckpt.log` |

### EXP-20260724-04: AR 50t/50v local rebuild — free-run fails in the opposite direction

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-25 00:24:12 |
| **Track** | `train` + `metric` (AR) |
| **Question** | Rebuilding 50t/50v locally (the July artifacts are absent from this clone), does the checkpoint reproduce the early-EOS free-run collapse of [EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode), so the new `eos_trace` can be read against it? |
| **Setup** | `build_training_index_subset.py --train-rows 50 --val-rows 50` (seed 42) → **48** train / **43** val songs, 50/50 rows (logged run had 48/**45**; source manifest here is 1755/186 rows vs 1745/197, so the sample is **not** identical). MERT extracted for all **91** unique audio (`--device=cuda`, **5m36s**). Hardware: **RTX 2080 6 GB** (logged runs used an RTX 3070 Ti). |
| **Train** | [`configs/ar/scale_50t_50v.json`](../../configs/ar/scale_50t_50v.json) unchanged. WSL GPU 50/50 steps/ep, ~**13–15 s**/ep. Early stop **ep 131**, restored best **ep 106**. Best `val_loss` **21.772**; best `val_aux_f1_hungarian` **0.0128**. |
| **Divergence from logged run** | `val_loss` is comparable (**21.8** here vs **20.9** at ep 65 there), but val Hungarian F1 is **~10× worse** (**0.0128** vs **0.126**). Same config, same recipe — the difference is the subset sample and/or hardware. |
| **Offline val decode** | `--split val --limit 10 --ar_decode` (**1284 s**). Teacher ordered **14/6544** (0.0021) @ 20 ms — essentially no timing skill. Free-run: **0/10** songs stopped on EOS; `ar_decode_length_sum` **20490** = the **2048** cap on every song. Ordered **1/20480**; Hungarian F1 **0.0035** (**47** TP, **20433** FP, **6497** FN). |
| **EOS trace** | first **0.0007**, final **0.0053**, max **0.0087** across 10 songs; **0** songs ever reach 0.5. Compare tide ([EXP-20260724-03](#exp-20260724-03-ar-decode-length-control--eos-trace-diagnostics)): mean **0.0017** but a clean spike to **0.978** at the true end. |
| **Artifacts** | `models_wsl/ar/scale_50t_50v_corrected_masks/ar_onset_model.keras` · `callbacks/ar/scale_50t_50v_corrected_masks/` · `logs/ar_scale_50t_50v_rebuild.log` · `logs/ar_scale_50t_50v_val_decode_10.log` · `logs/mert_extract_50t_50v.log` |
| **Conclusion** | **Fail to reproduce.** This rebuild does **not** show early-EOS under-generation; it shows the **opposite** — `<EOS>` is never selected and every song runs to the decode cap, producing **20433** false positives. Two runs of the same recipe therefore land at opposite termination pathologies (early EOS at 200t vs no EOS at 50t), which says the model is not learning termination **stably** at this data scale rather than being biased in one direction. It also means this checkpoint cannot stand in for the EXP-20260724-02 artifact when testing length fixes. Teacher timing is far too weak here (**0.0021** ordered) to attribute the free-run behavior to exposure bias alone. |
| **Next** | Either recover the original 200t/50v artifacts from the machine that produced them, or first close the teacher-quality gap on a local rebuild (investigate why val F1 is 10× lower on the same config) before drawing conclusions about free-run length. |

### EXP-20260724-03: AR decode length control + EOS trace diagnostics

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-07-24 23:02:12 |
| **Track** | `post` + `metric` (AR) |
| **Question** | Can free-run under-generation be probed offline without retraining, and what does `<EOS>` behavior look like on a checkpoint that does **not** collapse? |
| **Tooling** | `inference.ArLengthControl` (`eos_logit_bias`, `min_onset_tokens`) threaded through both free-run paths (prefix loop and KV decode); `ArDecodeStats.eos_prob_trace` records per-step `<EOS>` probability **before** control is applied; `eval_ar_onset_offline.py` gains `--ar_decode_eos_logit_bias` / `--ar_decode_min_onset_tokens` and an `ar_decode.eos_trace` report block (per song and split-aggregated) |
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
| **Offline eval** | `eval_ar_onset_offline.py --split val --ar_decode` on `models_wsl/ar/scale_200t_50v_corrected_masks/ar_onset_model.keras` (**50** songs, ~**524 s**). Teacher: ordered **105/36860** @ 20 ms (`rate` **0.0028**); Hungarian F1 **0.1199** (matches train val). Free-run: ordered **1/36860**; Hungarian F1 **0.0360**; **3400** preds vs **36860** GT (`ar_decode_length_sum` **3500**; all **50** songs stopped on EOS). |
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
| **Run2 offline (teacher)** | `eval_ar_onset_offline.py` — event F1 **~1.0** (634/634 TP); **632/634** within 20 ms; mean abs err **5.09 ms**, max **23.67 ms** |
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
| **Outcome** | Best/final `val_event_onset_f1` **1.0** (from ep ~180); `eval_ar_onset_offline`: **634/634** within 20 ms, 0 patch errors |
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
