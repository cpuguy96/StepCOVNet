# Experiment log

**Authoritative record** for runs and ablations. IDs: `EXP-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

Promote selected findings to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) only when drafting the paper — do not duplicate the full log there.

**Related:** [discussion notes](DISCUSSION_NOTES.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [dataset prep plan](DATASET_PREP_PIPELINE.md) · [AR onset design](AR_ONSET_DESIGN.md) · [decisions checklist](DECISIONS_CHECKLIST.md)

---

## Current phase

**Updated:** 2026-08-03
**Primary track:** AR scaling ladder (Track B) — [AR_SCALING_LADDER.md](AR_SCALING_LADDER.md)
**Next action:** **Meter density conditioning works** on R2 ([EXP-20260803-02](#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run)): free-run **0.234** vs the **0.132** bar; stop lengths vary (**6** modes, only **8/50** at **252**). Next: rerun **R3** (200t) with the same `density_conditioning: meter` to test whether early-EOS@**15** also fixes, or ablate **onset_density** vs raw **meter**.
**Blockers:** None — GPU free.
**Alternate track:** Ladder **R4** (300t) teacher-only with density, or Track A dense scoreboard vs **0.686**.
**Defer:** R4–R5 as primary climb without density, `gate-val-vs-dense`, further sampling-schedule tuning.

### Dataset prep (PRE ingestion)

| Phase | Status |
| ----- | ------ |
| P0–P9 | **Done** — **1942** chart rows; `training_index.json` (`stratified_song_v1`: **1010** / **110** songs, **1745** / **197** chart rows train/val). **Drift:** the untracked copy on this clone reports **1009** / **110** songs and **1755** / **186** rows ([NOTE-20260725-02](DISCUSSION_NOTES.md#note-20260725-02-subset-sampling-gives-every-train-size-a-different-val-set)) |
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
| **AR next gate** | Ladder teacher **R2 = 0.227** > **R3 = 0.199**; free-run **R2+density = 0.234** (stop lengths vary; **8/50** still @ **252**) vs baseline R2 **0.132** (all **252**) ([EXP-20260803-02](#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run)). **Next:** density on **R3** for early-EOS@**15** |
| **AR density conditioning** | **Supported on R2** — decoder global embed from normalized `#METER` ([EXP-20260803-02](#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run)); `onset_density` mode implemented but not yet run |
| **AR scheduled sampling** | **Closed, negative** — the feature was compiled out of the traced `train_step` ([EXP-20260802-05](#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step)); with `p` now a `tf.Variable` under `tf.cond`, a full rerun gives free-run **0.1313** vs the **0.132** bar and an unchanged fixed-**252** stop ([EXP-20260803-01](#exp-20260803-01-scheduled-sampling-now-actually-running-does-not-improve-free-run-on-r2)) |
| **Local artifact gap** | July 16–24 AR checkpoints, subset indices, and logs are **absent from this clone**; a 50t/50v rebuild reached comparable `val_loss` but **10× worse** val F1 and the **opposite** free-run pathology ([EXP-20260724-04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)) — local rebuilds do **not** currently stand in for the logged runs |
| **AR termination stability** | **Open** — same recipe gives early EOS at 200t ([EXP-20260724-02](#exp-20260724-02-ar-corrected-mask-200t50v-train--offline-val-decode)) and never-EOS at 50t ([EXP-20260724-04](#exp-20260724-04-ar-50t50v-local-rebuild--free-run-fails-in-the-opposite-direction)); termination is unstable, not one-directionally biased |
| **AR tide class weights (champion recipe)** | **Deferred** — drop-in on v8 failed free-run ([EXP-20260703-01](#exp-20260703-01-ar-tide-token-class-weight-ablation-champion-recipe)); champion stays `none`; co-tuned recipe revisit [NOTE-20260703-01](DISCUSSION_NOTES.md#note-20260703-01-class-weights-need-co-tuned-loss-recipe-deferred) |
| **AR training throughput / validation** | **Improved** — val aggregation + dynamic buckets (**18.6%** on smoke); single-song overfit batch cache default-on (~**9×** steady epoch on tide) ([EXP-20260716-01](#exp-20260716-01-ar-validation-aggregation--dynamic-length-bucketing), [EXP-20260716-02](#exp-20260716-02-ar-corrected-mask-tide-overfit-regression)) |

**Recommended when resuming onset work:**

- **Track A (scoreboard):** Full `final_data` dense MERT (or mel) train/val; compare to `data/v2` session best (0.686).
- **Track B (AR scale-up):** Meter density conditioning on R2 breaks the fixed-**252** stop and lifts free-run to **0.234** ([EXP-20260803-02](#exp-20260803-02-meter-density-conditioning-on-r2-breaks-the-252-stop-and-lifts-free-run)). Next: same lever on **R3** (early EOS @ **15**) or **onset_density** ablation vs **meter**.
- **Event track (optional):** Continue K-query probes on `data/v2` in parallel if not blocking Track A.

---

## Experiment index

Newest first. Stage tags: `pre` | `model` | `post` | `metric` | `train`. Discussion context: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

| ID | Stage tag | Question | Status | One-line outcome |
| -- | --------- | -------- | ------ | ---------------- |
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
| **Setup** | Same decode recipe: `debug_ar_onset_overfit.py --split val --ar_decode`. R2: [EXP-20260802-03](#exp-20260802-03-ladder-r2-offline-val-free-run--eos_trace). R3: `configs/ar/ladder_200t_50v.json` · `models_wsl/ar/ladder_200t_50v/ar_onset_model.keras` · ~**3.4** min wall |
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
| **Setup** | [`configs/ar/ladder_50t_50v.json`](../../configs/ar/ladder_50t_50v.json) · `models_wsl/ar/ladder_50t_50v/ar_onset_model.keras` · `debug_ar_onset_overfit.py --split val --ar_decode` · WSL GPU · ~**22.7** min wall |
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
