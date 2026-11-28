# StepCOVNet — Paper Outline (Draft)

**Status:** Active paper draft outline backed by verified claims `C1`–`C16` in [PAPER_LEDGER.md](PAPER_LEDGER.md). Lab logs stay in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

**Citations and claim tracking:** [PAPER_LEDGER.md](PAPER_LEDGER.md) · [paper.bib](paper.bib). Every citation key below has a matching BibTeX entry in `paper.bib` and a ledger row in `PAPER_LEDGER.md`.

**Architecture reference:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) — PRE → MODEL → POST → METRICS → training feedback.

---

## Working title

**Representation Over Architecture — and How to Tell If Your Chart Generator Listens to the Music: Beat-Aligned SSL Step Placement, Audio-Grounding Diagnostics, and Raw-Audio Robustness for DDR/ITG**

---

## Abstract (draft)

Automatic step-chart placement for four-panel rhythm games (DDR/ITG) has progressed from 10 ms spectrogram classifiers (`donahue2017ddc`) to beat-aligned 48-slot models using ConvLSTMs (`omalley2025ddcl`) and hierarchical Transformers (`omalley2026itgpt`). Yet three fundamental questions remain unmeasured: (1) how much of a beat-grid model's high slot F1 actually depends on audio versus beat-grid and difficulty/BPM priors, (2) whether architectural scaling or pretrained music representations and training signals drive real gains, and (3) how well beat-grid placement survives real-world raw audio without oracle `.sm` beat grids. We introduce an architecture-agnostic **audio-grounding diagnostic** (`M-grounding`) pairing distribution-preserving beat-audio permutation (`shuffle`) and silence (`zeros`) with decoded ±20 ms bipartite event skill (`M-event-20ms`) and train→val audio transfer ratios. First, we show that beat-grid checkpoints selected on early `val_loss` can retain 95.8% of predicted steps under zeroed audio (`0.727 → 0.722` F1) while looking strong on raw `M-slot48`. Second, on a controlled **2×2 factorization** on Original Fraxtil (`D-frax-orig`, 81/9 train/val songs, 45 val charts), replacing scratch multi-scale log-mel features with beat-aligned pretrained **MERT** embeddings (`li2024mert`) lifts DDCL by **+7.07 pp** (`0.6804 → 0.7511` F1@0.5) and ITGPT by **+4.15 pp** (`0.7104 → 0.7519`, max-F1 **0.7633**), proving representation dominates architecture and tracing log-mel ITGPT's +3.00 pp edge over DDCL to a **6.6×** audio-grounding gap (`−0.1192` vs `−0.0181`). Third, adding multi-layer MERT `(2, 6, 10, 16)`, chart density conditioning, downbeat-preserving SpecAugment (`park2019specaugment`), auxiliary beat-activity + soft-Dice + positive-weighted BCE supervision, and R-Drop (`liang2021rdrop`) lifts from-scratch Dataset A ITGPT-MERT to **0.7797** peak (**0.7759 ± 0.0034** 3-run mean, **0.8009** ±20 ms `evF1`, **+6.93 pp** over log-mel ITGPT) with zero extra inference parameters. Fourth, scaling across `48 → 197 → 294 → 1,713` non-Fraxtil multi-pack charts (`D-final`, 1,109 songs) lifts frozen 48-chart val F1 to **0.8649** (`10,507` errors, `16th_off 0.7635`, `24th 0.5922`) and achieves **0.7787** F1@0.5 (**0.8034** `evF1`) **zero-shot** on Dataset A (beating in-domain from-scratch MERT mean); fine-tuning onto Dataset A (`81` songs) reaches **0.7857** F1@0.5 @ `0.50` exact (**0.8083** `evF1`, **7,909–7,946** errors, **+7.53 pp** over log-mel ITGPT; **0.8092** `evF1` ensembled), with `8/9` validation songs scoring **0.8193** while one fractional-BPM gimmick song (`tsunamix_iii/isis`) drives 21.4% of all errors. Finally, evaluating end-to-end on raw audio with zero oracle `.sm` beats shows raw `BeatThis` (`foscarin2024beatthis`) lags by ~13 ms (`0.5804` ±20 ms `evF1`), whereas pairing built-in pure-PyTorch `BeatThis` with exact grid-edge decoding, progressive octave-aware global/piecewise lattice regularization with integer-BPM locking, and global/per-section zero-lag attack snapping lifts `BeatF1@70ms` to `0.8977` and ±20 ms `evF1` by **+21.26 pp** — retaining **98.2%** of oracle ±20 ms `evF1` (`0.7930` vs `0.8079`) on fine-tuned Arm C and **99.2%** (`0.7954` vs `0.8021`) completely zero-shot from raw audio alone.

---

## 1. Introduction

- **Problem (`C1`):** StepMania / DDR / ITG chart generation decomposes into **step placement** (when notes occur) and **step selection** (which arrows/holds occur; `donahue2017ddc`). Placement errors cascade directly into unplayable or off-beat charts.
- **Hidden failure modes in onset placement (`C8`, `C9`, `C15`, `C16`):**
  1. High `M-slot48` F1 on a beat-aligned grid does **not** guarantee audio grounding: a model conditioning on beat position, BPM, and chart meter can score >0.72 F1 while ignoring `beat_audio` almost completely (`EXP-20261119-01`).
  2. Naive ordered `timing_match` collapses near zero on real placement models even at strong ±20 ms timing precision because any single `FP` or `FN` shifts all subsequent indices (`EXP-20261118-01`), while fractional-beat BPM gimmicks (`tsunamix_iii/isis`) can collapse `M-slot48` (`0.3274`) even when continuous-time ±20 ms `evF1` doubles (`0.6564`; `EXP-20261127-06`).
  3. Standard beat-grid benchmarks assume oracle `.sm` beat grids unavailable on raw user audio; unaligned automatic beat trackers suffer severe ±20 ms timing collapse (`0.0369` raw `librosa`, `0.5804` raw `BeatThis`).
- **Contributions (`C8`–`C16`):**
  1. **Reproducible Dataset A baseline suite (`T-repro`):** Paper-faithful reimplementations of DDC (`donahue2017ddc`), DDCL (`omalley2025ddcl`), and ITGPT (`omalley2026itgpt`) on a documented song-grouped split (`D-frax-orig`, seed 42), separating published-split citations from same-split comparisons (`C5`).
  2. **Audio-grounding & transfer-ratio protocol (`M-grounding`, `C9`):** Distribution-preserving beat permutation (`shuffle`) vs silence (`zeros`) paired with decoded ±20 ms bipartite skill (`M-event-20ms`) and train→val audio transfer ratios.
  3. **Dataset A 2×2 factorization & zero-param champion recipe (`T-frax-2x2`, `C10`, `C13`):** Isolating audio representation (scratch log-mel vs beat-aligned `MERT-v1-330M`, `li2024mert`) from placement architecture (DDCL vs ITGPT), then demonstrating cumulative gains from multi-layer MERT `(2, 6, 10, 16)`, density conditioning, downbeat-preserving SpecAugment (`park2019specaugment`), `beat_aux + dice + pw` loss synergy, and R-Drop (`liang2021rdrop`) while falsifying 14 extra-parameter architectural hypotheses (`EXP-20261127-03`).
  4. **Multi-pack scaling & cross-corpus transfer (`T-ladder-scale`, `T-transfer-scaling`, `C11`, `C14`):** 4-point scaling (`48 → 197 → 294 → 1,713` charts, `48v 0.86493`), zero-shot transfer beating in-domain Fraxtil MERT mean (`0.77870` / `0.80337` `evF1`), and fine-tuned Dataset A transfer (`0.78574` / `0.80834` `evF1`, `7,909` errors; `0.80918` ensembled).
  5. **Per-song error audit & real-world raw-audio beat tracking (`T-error-audit`, `T-raw-audio-beats`, `C15`, `C16`):** Identifying the `tsunamix_iii/isis` fractional-BPM artifact (`8/9` songs at `0.8193`) and demonstrating an end-to-end pure-PyTorch `BeatThis` (`foscarin2024beatthis`) + octave-aware global/piecewise lattice-regularized + per-section zero-lag attack-snapped pipeline reaching `0.8977` `BeatF1@70ms` and retaining `98.2%` (`0.7930` Arm C) / `99.2%` (`0.7954` zero-shot `1,713t`) of oracle **±20 ms** `evF1` directly from raw audio.

---

## 2. Related work

All keys verified in [PAPER_LEDGER.md](PAPER_LEDGER.md) and [paper.bib](paper.bib):

- **DDR/ITG chart generation (`C1`, `C2`):** Placement-then-selection with 10 ms C-LSTM (`donahue2017ddc`), beat-aligned 48-slot ConvLSTM (`omalley2025ddcl`), hierarchical full-song Transformer (`omalley2026itgpt`), and beat-aligned sequence generation (`yi2023goct`). Adjacent rhythm-game work: `halina2021taikonation`, `lin2018generationmania`, `tsujino2018ddg`, `okeeffe2003dancingmonkeys`.
- **Musical onset detection, SSL representations, and beat tracking:** Classical spectral onset detection (`bello2005onset`), BLSTM (`eyben2010blstm`), CNN on multi-scale 80-band log-mel spectrograms (`schluter2014onset`, `hamel2012multiscale`), non-causal rhythm-game BPM estimation (`vandewetering2016bpm`), neural beat/downbeat tracking without DBN postprocessing (`foscarin2024beatthis`), and self-supervised music foundation models (`li2024mert`).
- **Regularization and training signals:** SpecAugment (`park2019specaugment`) and R-Drop dropout consistency (`liang2021rdrop`).
- **Set prediction and autoregressive timing (diagnostic contrast, `C3`):** Distinguish literature AR (**arrows given known times**) from continuous-time AR (`onset_ar`) and DETR-style K-query slots (`carion2020detr`). Brief note in Methods/Discussion on why beat-grid 48-slot placement outperforms K-query ceilings (~31% oracle) and avoids absolute-index pointer collapse (`EXP-20260804-05`).

---

## 3. Methods

### 3.1 Datasets and splits (`C4`, `C5`)

- **`D-frax-orig` (Dataset A — Original Fraxtil, `donahue2017ddc`):** 90 songs / 450 standard charts across 3 packs (*Tsunamix III*, *Fraxtil's Arrow Arrangements*, *Fraxtil's Beast Beats*); song-grouped `stratified_song_v1` (seed 42) → **81 / 9** train/val songs (**405 / 45** charts, **17,523** val beats, **17,868** val GT steps).
- **`D-frax-exp` (Dataset B — Expanded Fraxtil, `omalley2026itgpt`):** 222 exported songs / 722 standard charts (201/21 split; Vol. 4 unavailable) — reported separately, never mixed into Dataset A tables.
- **`D-final` (StepCOVNet multi-pack corpus):** *ITL Online 2026*, *Mizuki's Simfiles*, and *Vocaloid Project Pad Pack 4th*. Full valid corpus (`10` negative-BPM Vocaloid songs excluded): **1,109** songs, **1,713 / 176** train/val charts (**61,347** full-val beats, **134,327** full-val GT steps; `60.08 GB` float32 multi-layer MERT cache loaded via bounded LRU lazy caching). Frozen-val 4-point scaling ladder evaluated on the exact same **48 val charts** (**40** songs, **17,057** beats, **37,617** GT steps, `0` song overlap with all 4 train rungs): `48t`, `197t`, `294t`, and `1,713t`.

### 3.2 Beat-aligned 48-slot placement, MERT frontends, and champion training recipe

- **Beat-grid formulation (`omalley2025ddcl`, `omalley2026itgpt`):** Each integer beat is quantized into 48 sub-beat slots (covering 16ths, 24ths/triplets, 32nds, and micro-divisions). Oracle decoding of ground-truth slot centers via `#OFFSET`/`#BPMS` achieves **0.999972** ±20 ms `timing_match` (`EXP-20261010-01`).
- **Scratch log-mel baselines:**
  - *DDCL (`omalley2025ddcl`):* 32 frames/beat × 80-band 3-channel multi-scale log-mel (`schluter2014onset`, `hamel2012multiscale`), 16-beat causal/anti-causal `ConvLSTM2D` windows (`memlen=15`) fused with per-beat `[meter, bpm]`, dense `[512, 256]`, focal BCE.
  - *ITGPT (`omalley2026itgpt`):* Full-song beat tensor padded to multiples of 64; `ItgptBeatCNNEncoder` maps `(32, 80, 3) → (24, d_model)` into 5 hierarchical intra-chunk attention levels (`1/4/16/32/64` beats), gated frame/beat compression, 8-layer global Transformer (`d_model=256`, `n_heads=8`) conditioned on scalar `diff_tok` + `bpm_tok`, residual depthwise `ConvHead`, and grid-weighted BCE.
- **Beat-aligned MERT frontends (`li2024mert`):** Precomputed 100 Hz float32 hidden states of `m-a-p/MERT-v1-330M` (dim `1024`), resampled per beat and shared zero-copy across difficulties of the same song:
  - *Controlled 2×2 swap:* single layer-6 MERT (`DDCL-MERT` 16 frames/beat, `ITGPT-MERT` 24 frames/beat matching `CNN_FRAMES_OUT = 24`).
  - *Multi-layer MERT `(2, 6, 10, 16)`:* per-layer normalized mean of layers `(2, 6, 10, 16)` (guided by SUPERB-style learnable layer weights in `EXP-20261125-07`), keeping identical `(T, 1024)` disk/RAM/parameter footprint while unlocking 5.5× higher triplet (`24th`) recall (`EXP-20261125-03/10`).
- **Zero-extra-inference-parameter champion recipe (`C13`):**
  1. *Optimization:* Warmup + Cosine LR decay + EMA (`0.995`) + per-block `torch.compile(dynamic=True)` in PyTorch (`bf16` on L4, 3.5× faster than Keras with full parity, `EXP-20261126-05/06`).
  2. *Cross-pack density conditioning (`difficulty_source="density"`):* Replaces pack-specific integer `#METER` with chart average step density (`steps/beat`), lifting both in-domain F1 and cross-pack zero-shot transfer (`EXP-20261126-14/15`).
  3. *Downbeat-preserving SpecAugment (`park2019specaugment`, `keep0`):* 4 MERT time/feature masks (`p=0.15`) that never mask beat frame `0` (preserving downbeat alignment; `EXP-20261126-15/16`).
  4. *3-way loss + auxiliary synergy:* `ItgptBeatAuxHead` (per-beat activity + count auxiliary supervision, dropped at inference) + masked soft-Dice (`0.05`) + positive class weighting (`bce_pos_weight`: `1.25–1.35` from scratch, `1.00` during transfer fine-tuning for exact `τ=0.50` calibration) + optional bidirectional Bernoulli R-Drop (`liang2021rdrop`, `rdrop_kl_weight=2.0`; `EXP-20261127-01/02/05`).
- **End-to-end raw-audio beat tracking (`foscarin2024beatthis`, `C16`):** Self-contained pure-PyTorch implementation of Beat This! (`stepcovnet.beat_this_model`, loading CPJKU `final0.ckpt` `strict=True`, plus 3-point parabolic sub-frame peak refinement and optional song-level transient phase snapping) feeding beat/downbeat grids directly into MERT beat alignment, ITGPT slot decoding, and downbeat-anchored `.sm` export (`#OFFSET`, `#BPMS`).

### 3.3 Evaluation and audio-grounding protocol (`M-slot48`, `M-event-20ms`, `M-grounding`)

- **`M-slot48`:** Pooled micro F1 at τ=0.5 and threshold-swept `max-F1`, reported beside the beat-shuffled GT null floor (`M-null`: `0.3971` Dataset A; `0.6007` `D-final` ladder val).
- **`M-event-20ms`:** Decode occupied slots (τ=0.5) to onset times in seconds and score bipartite Hungarian ±20 ms (plus ±50 ms for raw-audio beat tracking) F1 and headroom-normalized skill `(F1 − F1_null) / (1 − F1_null)` over the strongest audio-blind null (`ioi_shuffle`, `regular_grid`, `uniform_duration`).
- **`M-grounding` (`--diagnose_grounding`):** Score identical checkpoints under three `beat_audio` conditions while keeping metadata and positional embeddings intact: `none`, `shuffle` (beat permutation within each chart), and `zeros`. Report `slot_f1_drop_shuffle`, `event_skill_drop_shuffle`, `1 − IoU`, and train→val audio transfer (`val_drop_shuffle / train_drop_shuffle`).

---

## 4. Results

### 4.1 Dataset A literature baseline recreation (`T-repro`, `C6`, `C7`)

All reimplementations scored on `D-frax-orig` (`81 / 9` songs, `45` val charts, `17,523` beats, seed 42). Do **not** compare against published Table 2 Expanded Fraxtil (`0.78 / 0.80`) on a different split.

| Model | Grid / Metric | Our `D-frax-orig` Val | Null Floor | Reference Context | Source |
| ----- | ------------- | --------------------- | ---------- | ----------------- | ------ |
| DDC C-LSTM (`donahue2017ddc`) | `M-ddc-20ms` (F-score_c / F-score_m) | **0.652 / 0.734** | `+0.397` skill (8-ep) | Published Fraxtil: `0.681 / 0.756` (unpublished split) | `EXP-20260815-03` |
| DDC peak-snap (frozen) | `M-slot48` @ 0.5 | `0.364` | `0.397` (`−0.033` skill) | Confirms 10 ms peaks do not transfer zero-shot to 48-slot | `EXP-20260819-04` |
| DDCL ConvLSTM (`omalley2025ddcl`) | `M-slot48` @ 0.5 / max-F1 | **0.6804 / 0.6849** | `0.3971` (`+0.2833`) | Same split as ITGPT below | `EXP-20260817-01` |
| ITGPT log-mel (`omalley2026itgpt`) | `M-slot48` @ 0.5 / max-F1 | **0.7104 / 0.7166** | `0.3971` (`+0.3133`) | Beats DDCL by **+3.00 pp** on matched split | `EXP-20260819-03` |

*(Dataset B `D-frax-exp` 201/21 ITGPT log-mel `last`: **0.735 / 0.740** max-F1 vs null `0.390`, `EXP-20260819-02` — reported as separate corpus.)*

### 4.2 Dataset A 2×2 factorization & from-scratch recipe progression (`T-frax-2x2`, `C10`, `C13`)

Controlled frontend swap (`log-mel → MERT`) and cumulative from-scratch recipe upgrades on `D-frax-orig` (`81/9` songs, `45` val charts, `17,523` beats, `17,868` GT steps, `null F1 = 0.3971`).

| Architecture | Frontend & Recipe | Ckpt | `M-slot48` F1@0.5 | `M-slot48` max-F1 | ±20 ms `evF1` | `shuffle` drop (F1) | Errors | Source |
| ------------ | ----------------- | ---- | ----------------- | ----------------- | ------------- | ------------------- | ------ | ------ |
| DDCL (BiLSTM) | Scratch log-mel | `last` (ep 128) | `0.6804` | `0.6849` @ 0.45 | `0.7008` | `−0.0181` | — | `EXP-20260817-01`, `EXP-20261123-03` |
| ITGPT (Transformer) | Scratch log-mel | `last` (ep 128) | `0.7104` | `0.7166` @ 0.35 | `0.7291` | `−0.1192` | — | `EXP-20260819-03`, `EXP-20261123-02` |
| DDCL (BiLSTM) | MERT L6 | `last` (ep 128) | `0.7511` (+7.07 pp) | `0.7554` @ 0.45 | `0.7689` | `−0.1312` (7.3×) | — | `EXP-20261123-03` |
| ITGPT (Transformer) | MERT L6 (`const128`) | `best` (ep 38) | `0.7519` (+4.15 pp) | `0.7633` @ 0.35 | `0.7679` | `−0.1296` | — | `EXP-20261123-02` |
| ITGPT (Transformer) | MERT `L2+6+10+16` + Cos64 + EMA | `best` (ep 41) | `0.7680` | `0.7698` @ 0.45 | `0.7848` | `−0.1396` | `8,182` | `EXP-20261125-10` |
| ITGPT (Transformer) | + `density` + `ls=0.01` | `best_slot_f1` (ep 20) | `0.7729` | `0.7740` @ 0.45 | `0.7923` | `−0.1447` | `8,166` | `EXP-20261126-15` |
| **ITGPT (Transformer)** | **+ `beat_aux=0.10` + `dice=0.05` + `pw=1.35`** | **peak (3-run mean)** | **`0.7797`** (`0.7759 ± 0.0034`) | **`0.7797`** @ 0.50 | **`0.8009`** (`0.7963 ± 0.0040`) | **`−0.1420`** | **`8,024`** | `EXP-20261127-01` |

*(Adding `rdrop=2.0` beat matched `EXP-20261127-01` on 2/2 seeds and pushed `last.pt` to `0.7659` / `0.7684` at `rdrop=4.0`, `EXP-20261127-02`.)*

### 4.3 Multi-pack transfer and 4-point data scaling (`T-ladder-scale`, `C11`, `C12`)

All runs evaluated on the **exact same frozen 48 val charts** of `D-final` (`17,057` beats, `37,617` GT steps, `null F1 = 0.6007`, `0` song overlap with full `1,713t` train).

| Model & Recipe | Train Charts | Ckpt | Val `M-slot48` F1@0.5 | Val max-F1 | Val ±20 ms `evF1` | Val `shuffle` drop | Errors | Source |
| -------------- | ------------ | ---- | --------------------- | ---------- | ----------------- | ------------------ | ------ | ------ |
| Log-mel DDCL (`best_val_loss`) | `48` | `best` | `0.7272` | `0.7272` | `0.7272` | `−0.0048` (~blind) | — | `EXP-20261119-01` |
| Log-mel DDCL | `48` | `last` (ep 128) | `0.7307` | `0.7308` | `0.7308` | `−0.0230` (11.6% xfer) | — | `EXP-20261119-02/03` |
| MERT-LSTM (`Adam`, `drop=0.2`) | `48` | `last` (ep 128) | `0.7647` | `0.7712` @ 0.35 | — | `−0.0782` (33.6% xfer) | — | `EXP-20261119-04/05` |
| ITGPT-MERT L6 (`const128`) | `48` | `best_slot_f1` | `0.7719` | `0.7728` | — | `−0.0848` (32.0% xfer) | — | `EXP-20261123-01` |
| ITGPT-MERT L6 (`const128`) | `197` | `best_slot_f1` | `0.8181` | `0.8192` | — | `−0.1103` (42.3% xfer) | — | `EXP-20261124-01` |
| ITGPT-MERT L6 (`const128`) | `294` | `best` | `0.8276` | `0.8324` | — | `−0.1189` (44.7% xfer) | — | `EXP-20261124-02` |
| ITGPT-MERT (`L2+6+10+16`, Cos64, `#METER`) | `294` | `best_slot_f1` | `0.8388` | `0.8388` @ 0.50 | — | `−0.1151` | `12,276` | `EXP-20261124-03` |
| ITGPT-MERT + `density` + `keep0` + `beat_aux` + `dice` + `pw=1.25` | `294` | `best_slot_f1` | `0.8497` | `0.8500` @ 0.45 | `0.8499` | `−0.1226` | `11,593` | `EXP-20261127-01` |
| **ITGPT-MERT full `final_data` (same recipe)** | **`1,713`** (`1,109` songs) | **`best_slot_f1` (ep 8)** | **`0.8649`** (+1.52 pp) | **`0.8649`** @ 0.50 | **`0.8650`** | **`−0.1371`** | **`10,507`** (−9.4%) | `EXP-20261127-04` |

*(Full 176-chart val on `1,713t` (`61,347` beats, `134,327` GT steps): **`0.8422` F1@0.5 @ `0.50` exact**. Frozen `48v` subdivision jumps from `300t → 1,713t`: `8th 0.8869 → 0.8988`, `16th_off 0.7278 → 0.7635` (+3.57 pp), `24th 0.5358 → 0.5922` (+5.64 pp), `intro 0.8015 → 0.8327` (+3.12 pp).)*

### 4.4 Metric diagnostic: why rank-indexed `timing_match` breaks (`T-timing-diag`, `C8`)

Counterfactual masks on the `D-final` 48-val DDCL checkpoint (`EXP-20261118-01`; oracle `EXP-20261010-01`):

| Mask Condition | `M-slot48` F1 | Ordered `timing_match` (±20 ms) | Bipartite `micro_event_f1` / `skill_event_f1` | Interpretation |
| -------------- | ------------- | ------------------------------- | --------------------------------------------- | -------------- |
| Oracle GT slot centers | `1.0000` | **0.999972** (`1.0000` on local 48v) | `1.0000` / `+1.0000` | 48-slot grid loses virtually zero ±20 ms precision |
| Raw model predictions (τ=0.5) | `0.7272` | `0.0072` | **0.7272 / +0.6155** | Strong ±20 ms timing hidden by cumulative rank drift |
| `fp_removed` (strip FPs only) | — | `0.0536` | — | Removing FPs alone does not fix drift (FNs still shift ranks) |
| `fn_filled` (fill FNs only) | — | `0.0408` | — | Filling FNs alone does not fix drift (FPs still shift ranks) |
| `chart_top_k` (match chart count) | — | `0.0201` | — | Chart-level count match still drifts mid-song |
| `beat_top_k` (match per-beat count) | — | **0.8338** | — | Zeroing per-beat `FP − FN` drift immediately recovers `timing_match` |

### 4.5 Cross-corpus pretraining, fine-tuning, and ensembling on Dataset A (`T-transfer-scaling`, `C14`)

All rows clearly labeled by training regime (never mixed with from-scratch Dataset A claims). Evaluated on `D-frax-orig` (`45` val charts, `17,868` GT steps).

| Regime & Model | Dataset A Train Songs | `M-slot48` F1@0.5 | `M-slot48` max-F1 | ±20 ms `evF1` | Errors | `intro` F1 | Source |
| -------------- | --------------------- | ----------------- | ----------------- | ------------- | ------ | ---------- | ------ |
| From-scratch log-mel ITGPT | `81` | `0.7104` | `0.7166` | `0.7291` | — | — | `EXP-20260819-03` |
| From-scratch ITGPT-MERT (3-run mean / peak) | `81` | `0.7759` / `0.7797` | `0.7797` @ 0.50 | `0.7963` / `0.8009` | `8,024` | `0.6700` | `EXP-20261127-01` |
| **Zero-shot `1,713t` → Dataset A** | **`0`** | **`0.7787`** | **`0.7822`** @ 0.60 | **`0.8034`** | — | `0.6883` | `EXP-20261127-04` |
| **Fine-tuned Arm C (`keep0`, `pw=1.00`, Cos16)** | **`81`** | **`0.7857`** (+7.53 pp) | **`0.7857`** @ 0.50 | **`0.8083`** | `7,946` | `0.735+` | `EXP-20261127-05` |
| **Fine-tuned Arm D (`rdrop=2.0`, `pw=1.00`, Cos16)** | **`81`** | **`0.7853`** | **`0.7853`** @ 0.50 | **`0.8082`** | **`7,909`** | **`0.7539`** (+8.39 pp) | `EXP-20261127-05` |
| **Diverse Logit Ensemble (`Arm C + D + ZS`)** | `81` + `0` | `0.7854` | **`0.7863`** @ 0.55 | **`0.8092`** | — | — | `EXP-20261127-06` |

### 4.6 Per-song error audit on Dataset A (`T-error-audit`, `C15`)

Auditing `Arm C` (`0.78574` overall, `7,946` errors across 9 val songs / 45 charts, `EXP-20261127-06`):

| Subset | Songs / Charts | `M-slot48` F1@0.5 | ±20 ms `evF1` | Errors (% of total) | Root Cause |
| ------ | -------------- | ----------------- | ------------- | ------------------- | ---------- |
| Clean 8/9 val songs | `8 / 40` | **`0.8193`** (`0.764–0.868`) | **`0.8195`** | `6,245` (`78.6%`) | Standard constant/clean BPM grids |
| Outlier: `tsunamix_iii/isis` | `1 / 5` | `0.3274` | **`0.6564`** (2.0×) | `1,701` (`21.4%`) | `31` fractional-beat BPM gimmicks → `61.4%` `micro/32nd` slots |

### 4.7 End-to-end raw-audio beat tracking robustness (`T-raw-audio-beats`, `C16`)

Evaluating `Arm C` (Dataset A transfer champion) and **Zero-Shot `1,713t`** (`0` Fraxtil songs) on Dataset A val (`9` songs / `45` charts / `17,868` GT steps) with **zero `.sm` beat grid** via [`scripts/eval_raw_audio_onset.py`](../../scripts/eval_raw_audio_onset.py) (`EXP-20261127-07`–`09`; prior raw slot-center baseline in `EXP-20261127-06`):

| Beat Grid Pipeline (`Arm C` / `Zero-Shot 1,713t`) | Beat `F1@70ms` | ±20 ms `evF1` (`%` oracle) | ±35 ms `evF1` | ±50 ms `evF1` (`%` oracle) | Notes |
| ------------------------------------------------- | -------------- | -------------------------- | ------------- | -------------------------- | ----- |
| Oracle `.sm` beat grid | `1.0000` | `0.8079` / `0.8021` (`100%`) | `0.8098` / `0.8067` | `0.8151` / `0.8161` (`100%`) | Upper bound (exact grid-edge + musical slots) |
| Raw `BeatThis` (slot-center, prior `EXP-20261127-06`) | — | `0.5804` (`71.8%`) | — | `0.7747` (`95.2%`) | `12.9 ms` MIR-annotation phase lag |
| `beat_this_raw` (exact grid-edge + musical slots) | `0.7965` | `0.7131` / `0.7087` (`88.3%`) | `0.7726` / `0.7721` | `0.7788` / `0.7801` (`95.5%`) | Exact grid-edge decoding removes half-slot jitter |
| `beat_this_reg` (+ octave-aware global/piecewise lattice & int-BPM lock) | **`0.8977`** | `0.7917` / `0.7849` (`+7.86 pp`) | `0.7967` / `0.7975` | `0.8007` / `0.8039` (`98.2%`) | Unifies `0.5×/1×/2×` octave flickers & smooths medley BPM sections |
| **`beat_this_full` (+ global/per-section zero-lag attack snap)** | **`0.8977`** | **`0.7930` (`98.2%`) / `0.7954` (`99.2%`)** | **`0.7957` / `0.8026`** | **`0.7991` (`98.0%`) / `0.8067` (`98.8%`)** | **`+7.99 / +8.67 pp` over raw (`+21.26 pp` over prior); beats oracle by `+18.0–19.7 pp` on `tsunamix_iii/isis` (`0.8335 / 0.8320`)** |
| `librosa` (+ octave-aware lattice reg & zero-lag attack snap) | `0.7413` | `0.7658` / `0.7767` | `0.7717` / `0.7838` | `0.7765` / `0.7894` | Offline fallback (`0.0369 → 0.7658/0.7767` with reg+snap; `−15.6 pp` lower beat F1 than `BeatThis`) |

---

## 5. Discussion

1. **Why log-mel ITGPT beat log-mel DDCL (`C10`):** Our 2×2 grounding matrix reveals that log-mel DDCL on Dataset A was largely a metadata/beat-grid model (`shuffle` drop `−0.0181`), whereas log-mel ITGPT grounded 6.6× more strongly in audio (`−0.1192`). Providing beat-aligned MERT repairs DDCL's audio pathway (`−0.1312`, 7.3×) and collapses the architecture gap on homogeneous Fraxtil, while full-song Transformer attention still provides an edge on higher-diversity multi-pack charts (`D-final`) and scales smoothly to `1,713` charts.
2. **Why `shuffle` is the architecture-agnostic grounding metric (`C9`):** Under `zeros`, MERT-LSTM/DDCL-MERT collapses completely because zeroed inputs push Conv/LSTM biases off-manifold, whereas ITGPT-MERT's RMSNorm + residuals gracefully fall back to a prior template. Distribution-preserving `shuffle` keeps activations in-distribution and cleanly measures audio steering across architectures.
3. **Training signals > extra architecture parameters (`C12`, `C13`):** Fourteen extra-parameter architectural probes (`EXP-20261127-03`: RoPE at level6, SwiGLU, sub-beat skip/refinement heads, multiscale conv heads, delta features) were cleanly falsified, whereas zero-inference-cost representation and training-signal upgrades (multi-layer MERT `(2,6,10,16)`, density conditioning, downbeat-preserving SpecAugment `keep0`, `beat_aux + dice + pw`, and R-Drop) delivered `+2.78 pp` on Dataset A and `+2.21 pp` on `300t/50v`.
4. **Scale unlocks cross-corpus zero-shot and intro generalization (`C11`, `C14`):** Scaling `ITGPT-MERT` to `1,713` non-Fraxtil charts lifts frozen `48v` to **0.8649** (`16th_off +3.57 pp`, `24th +5.64 pp`), beats in-domain Fraxtil MERT mean **zero-shot** (`0.7787` / `0.8034` `evF1`), and after fine-tuning with `pw=1.00` boosts Fraxtil `intro` F1 by **+8.39 pp** (`0.670 → 0.754`).
5. **Continuous-time `M-event-20ms` exposes fractional-BPM artifacts and raw-audio readiness (`C15`, `C16`):** Exact-slot `M-slot48` penalizes `tsunamix_iii/isis`'s 31 fractional-BPM gimmicks by `−33 pp` vs continuous-time ±20 ms `evF1` (`0.3274` vs `0.6564`). On raw audio without `.sm` beats, raw Beat This! (`foscarin2024beatthis`) suffers a ~13 ms MIR-annotation phase lag (`0.5804` ±20 ms `evF1`) and mid-song octave flickers (`0.5×/1×/2×`), which our progressive octave-aware global/piecewise lattice regularization (`+7.86 pp`, `BeatF1@70ms 0.7965 → 0.8977`) and global/per-section zero-lag attack snapping nearly eliminate — retaining **98.2% (`0.7930`)** of oracle ±20 ms `evF1` on Arm C and **99.2% (`0.7954`)** completely zero-shot on `1,713t`, while *outperforming* oracle GT beats by `+18.0–19.7 pp` on `tsunamix_iii/isis`.
6. **Why dense/event/AR-on-times tracks were diagnostic rather than primary (`C3`):** K-query event prediction hit a ~31% oracle ceiling (`EXP-20260606-11`), dense TCNs plateaued at `+0.414` event skill (`EXP-20260824-01`), and AR-on-times `Dense(max_patches)` pointers passed single-song overfit while being 99.96% audio-blind (`EXP-20260804-05`), directly motivating `M-grounding`.

---

## 6. Conclusion and Future Work

Beat-aligned multi-layer pretrained music representations (`li2024mert`), zero-inference-cost training signals (`density`, `keep0` SpecAugment, `beat_aux + dice + pw`, R-Drop), and architecture-agnostic audio-grounding diagnostics (`M-grounding`) deliver decisive gains across both literature and community step-placement benchmarks: lifting DDCL (**+7.07 pp**) and ITGPT (**+6.93 pp** from scratch / **+7.53 pp** transferred to **0.7857** F1@0.5 / **0.8083** `evF1`) on Original Fraxtil, scaling monotonically to **0.8649** on `1,713`-chart multi-pack community data, and retaining **98.2–99.2%** of oracle **±20 ms** event F1 (`0.7930` transferred / `0.7954` zero-shot; `BeatF1@70ms 0.8977`) end-to-end on raw audio via a built-in pure-PyTorch Beat This! tracker (`foscarin2024beatthis`) with progressive octave-aware lattice regularization and per-section zero-lag attack snapping. Future work: (1) upgrading Stage 2 four-panel step selection (autoregressive foot alternation, crossovers, holds/rolls) conditioned on the beat/downbeat grid and MERT features, and (2) multi-author evaluation on ITG (`D-itg`).

---

## Appendix

- Paper ledger (citations, claims, datasets, metrics): [PAPER_LEDGER.md](PAPER_LEDGER.md)
- BibTeX: [paper.bib](paper.bib)
- Pipeline: [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
- Experiment log (authoritative): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- Design reasoning: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md)
- Onset metrics: [ONSET_METRICS.md](ONSET_METRICS.md)
- Config references (seed 42): `configs/ddc/ddcl_placement_fraxtil.json`, `configs/ddc/ddcl_mert_placement_fraxtil_128ep.json`, `configs/ddc/itgpt_placement_fraxtil_orig_128ep.json`, `configs/ddc/itgpt_mert_placement_fraxtil_orig_128ep.json`, `configs/ddc/itgpt_mert_placement_ladder_{50t,200t,300t}_50v.json`, `configs/ddc/itgpt_mert_placement_final_data_full_keep0_aux_dice_pw125.json`
