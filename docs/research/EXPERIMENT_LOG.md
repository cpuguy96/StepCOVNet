# Experiment log

**Authoritative record** for runs and ablations. IDs: `EXP-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

Promote selected findings to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) only when drafting the paper — do not duplicate the full log there.

**Related:** [discussion notes](DISCUSSION_NOTES.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [dataset prep plan](DATASET_PREP_PIPELINE.md) · [AR onset design](AR_ONSET_DESIGN.md) · [decisions checklist](DECISIONS_CHECKLIST.md)

---

## Current phase

**Updated:** 2026-07-03

### Dataset prep (PRE ingestion)

| Phase | Status |
| ----- | ------ |
| P0–P9 | **Done** — **1942** chart rows; `training_index.json` (`stratified_song_v1`: **1010** / **110** songs, **1745** / **197** chart rows train/val) |

**Recommended next step (Track A — scoreboard):** First full multi-song **dense** training on `final_data` via `--training_index_path=data/final_data/training_index.json` (WSL GPU). Extract MERT features for `final_data` if not using mel baseline; then `eval_dense_onset.py` + threshold sweep on val.

### Onset detection (research track)

| Item | Status |
| ---- | ------ |
| Dense val best (`data/v2`) | BiLSTM 256u — micro event F1 **0.686** @ thr=0.30 (EXP-20260610-03) |
| Event tide formulation (`data/v2`) | ~27–30% F1 plateau; oracle ~31% (EXP-20260606-11) — formulation ceiling for K-query slots |
| `final_data` training hookup | **Done** — dense + event trainers accept `--training_index_path`; 10-song CPU smoke **10/10** batches (EXP-20260624-01/02) |
| Multi-song val on `final_data` | **Unblocked** — awaiting first full GPU dense train + eval |
| **AR tide perfect overfit** | **PASS** — scratch **iter175** / champion **v8**: teacher + free-run **634/634** ordered @ 20 ms vs **`target_times`** ([EXP-20260630-01](#exp-20260630-01-ar-tide-scratch-perfect-overfit-iter175--v8-champion)) |
| **AR 10-song smoke** | **PASS** — **10/10** train batches, **2/2** val; `val_loss` **53.4 → 38.7** over 5 ep; teacher `event_onset_f1` > 0 ([EXP-20260630-02](#exp-20260630-02-ar-gate-10song-smoke)) |
| **AR next gate** | **`final-data-mert`** (full manifest MERT cache) → **`gate-val-vs-dense`** ([AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) §10.1) |
| **AR tide class weights (champion recipe)** | **Deferred** — drop-in on v8 failed free-run ([EXP-20260703-01](#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe)); champion stays `none`; co-tuned recipe revisit [NOTE-20260703-01](DISCUSSION_NOTES.md#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred) |

**Recommended when resuming onset work:**

- **Track A (scoreboard):** Full `final_data` dense MERT (or mel) train/val; compare to `data/v2` session best (0.686).
- **Track B (AR scale-up):** Champion [`configs/ar/tide_overfit.json`](../../configs/ar/tide_overfit.json) (graduated **v8**, iter175 recipe). Checkpoint: `models_wsl/ar/tide_overfit/`. **10-song smoke passed** ([`configs/ar/smoke.json`](../../configs/ar/smoke.json), EXP-20260630-02). Next: extract MERT for full `final_data` if needed, then AR multi-song train / **`gate-val-vs-dense`**.
- **Event track (optional):** Continue K-query probes on `data/v2` in parallel if not blocking Track A.

---

## Experiment index

Newest first. Stage tags: `pre` | `model` | `post` | `metric` | `train`. Discussion context: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

| ID | Stage tag | Question | Status | One-line outcome |
| -- | --------- | -------- | ------ | ---------------- |
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
