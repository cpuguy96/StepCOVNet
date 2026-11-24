# StepCOVNet — Paper Outline (Draft)

**Status:** Active paper draft outline backed by verified claims `C1`–`C12` in [PAPER_LEDGER.md](PAPER_LEDGER.md). Lab logs stay in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

**Citations and claim tracking:** [PAPER_LEDGER.md](PAPER_LEDGER.md) · [paper.bib](paper.bib). Every citation key below has a matching BibTeX entry in `paper.bib` and a ledger row in `PAPER_LEDGER.md`.

**Architecture reference:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) — PRE → MODEL → POST → METRICS → training feedback.

---

## Working title

**Representation Over Architecture — and How to Tell If Your Chart Generator Listens to the Music: Beat-Aligned SSL Step Placement and Audio-Grounding Diagnostics for DDR/ITG**

---

## Abstract (draft)

Automatic step-chart placement for four-panel rhythm games (DDR/ITG) has progressed from 10 ms spectrogram classifiers (`donahue2017ddc`) to beat-aligned 48-slot models using ConvLSTMs (`omalley2025ddcl`) and hierarchical Transformers (`omalley2026itgpt`). Yet two basic questions remain unmeasured: (1) how much of a beat-grid model's high slot F1 actually depends on audio versus beat-grid and difficulty/BPM priors, and (2) whether architectural scaling or pretrained music representations drive real gains. We introduce an architecture-agnostic **audio-grounding diagnostic** (`M-grounding`) pairing distribution-preserving beat-audio permutation (`shuffle`) and silence (`zeros`) with decoded ±20 ms bipartite event skill (`M-event-20ms`) and train→val audio transfer ratios. First, we show that beat-grid checkpoints selected on early `val_loss` can retain 95.8% of predicted steps under zeroed audio (`0.727 → 0.722` F1) while looking strong on raw `M-slot48`. Second, on a controlled **2×2 factorization** on Original Fraxtil (`D-frax-orig`, 81/9 train/val songs, 45 val charts), replacing scratch multi-scale log-mel features with beat-aligned pretrained **MERT** embeddings (`li2024mert`) lifts DDCL by **+7.07 pp** (`0.6804 → 0.7511` F1@0.5) and ITGPT by **+4.15 pp** (`0.7104 → 0.7519`, max-F1 **0.7633**, ±20 ms `skill_event_f1` **+0.6638**). Under scratch log-mel, ITGPT's +3.00 pp edge over DDCL traces to a **6.6×** audio-grounding gap (`−0.1192` vs `−0.0181` `shuffle` drop); once both models share MERT, DDCL's grounding jumps **7.3×** (`−0.1312`, 100% silence suppression) and the architecture gap narrows to **+0.08 pp** F1@0.5 (**+0.79 pp** max-F1). Finally, on a harder multi-pack corpus (`D-final`, frozen 48 val charts), beat-aligned MERT triples train→val audio transfer (`11.6% → 33.6%`), full-song hierarchical attention (`ITGPT-MERT`) beats BiLSTM (`0.7719` vs `0.7647`), and 4× training scale (`48 → 197` charts) pushes val F1@0.5 to **0.8181** (**0.8192** max-F1, **+0.7330** ±20 ms skill) with **100%** of the +4.62 pp gain coming from generalized audio grounding (`shuffle` floor flat `0.710 → 0.709`, audio transfer `32.0% → 42.3%`).

---

## 1. Introduction

- **Problem (`C1`):** StepMania / DDR / ITG chart generation decomposes into **step placement** (when notes occur) and **step selection** (which arrows/holds occur; `donahue2017ddc`). Placement errors cascade directly into unplayable or off-beat charts.
- **Hidden failure modes in onset placement (`C8`, `C9`):**
  1. High `M-slot48` F1 on a beat-aligned grid does **not** guarantee audio grounding: a model conditioning on beat position, BPM, and chart meter can score >0.72 F1 while ignoring `beat_audio` almost completely (`EXP-20261119-01`).
  2. Conversely, naive ordered `timing_match` collapses near zero on real placement models even at strong ±20 ms timing precision because any single `FP` or `FN` shifts all subsequent indices (`EXP-20261118-01`).
- **Contributions (`C8`–`C12`):**
  1. **Reproducible Dataset A baseline suite (`T-repro`):** Paper-faithful reimplementations of DDC (`donahue2017ddc`), DDCL (`omalley2025ddcl`), and ITGPT (`omalley2026itgpt`) on a documented song-grouped split (`D-frax-orig`, seed 42), separating published-split citations from same-split comparisons (`C5`).
  2. **Audio-grounding & transfer-ratio protocol (`M-grounding`, `C9`):** Distribution-preserving beat permutation (`shuffle`) vs silence (`zeros`) paired with decoded ±20 ms bipartite skill (`M-event-20ms`) and train→val audio transfer ratios.
  3. **Dataset A 2×2 factorization (`T-frax-2x2`, `C10`):** Isolating audio representation (scratch log-mel vs beat-aligned `MERT-v1-330M`, `li2024mert`) from placement architecture (DDCL BiLSTM vs ITGPT hierarchical Transformer), showing representation dominates architecture and explaining log-mel ITGPT's lead as a 6.6× audio-grounding gap.
  4. **Multi-pack generalization & scaling (`T-ladder-scale`, `C11`, `C12`):** Demonstrating on `D-final` that MERT triples audio transfer, heavy audio-path regularization asymmetrically hurts grounding, and 4× data scaling converts memorized audio features cleanly into **0.8181** val F1 (**42.3%** audio transfer).

---

## 2. Related work

All keys verified in [PAPER_LEDGER.md](PAPER_LEDGER.md) and [paper.bib](paper.bib):

- **DDR/ITG chart generation (`C1`, `C2`):** Placement-then-selection with 10 ms C-LSTM (`donahue2017ddc`), beat-aligned 48-slot ConvLSTM (`omalley2025ddcl`), hierarchical full-song Transformer (`omalley2026itgpt`), and beat-aligned sequence generation (`yi2023goct`). Adjacent rhythm-game work: `halina2021taikonation`, `lin2018generationmania`, `tsujino2018ddg`, `okeeffe2003dancingmonkeys`.
- **Musical onset detection & audio frontends:** Classical spectral onset detection (`bello2005onset`), BLSTM (`eyben2010blstm`), CNN on multi-scale 80-band log-mel spectrograms (`schluter2014onset`, `hamel2012multiscale`), beat tracking (`vandewetering2016bpm`), and self-supervised music foundation models (`li2024mert`).
- **Set prediction and autoregressive timing (diagnostic contrast, `C3`):** Distinguish literature AR (**arrows given known times**) from continuous-time AR (`onset_ar`) and DETR-style K-query slots (`carion2020detr`). Brief note in Methods/Discussion on why beat-grid 48-slot placement outperforms K-query ceilings (~31% oracle) and avoids absolute-index pointer collapse (`EXP-20260804-05`).

---

## 3. Methods

### 3.1 Datasets and splits (`C4`, `C5`)

- **`D-frax-orig` (Dataset A — Original Fraxtil, `donahue2017ddc`):** 90 songs / 450 standard charts across 3 packs (*Tsunamix III*, *Fraxtil's Arrow Arrangements*, *Fraxtil's Beast Beats*); song-grouped `stratified_song_v1` (seed 42) → **81 / 9** train/val songs (**405 / 45** charts, **17,523** val beats, **17,868** val GT steps).
- **`D-frax-exp` (Dataset B — Expanded Fraxtil, `omalley2026itgpt`):** 222 exported songs / 722 standard charts (201/21 split; Vol. 4 unavailable) — reported separately, never mixed into Dataset A tables.
- **`D-final` (StepCOVNet multi-pack corpus):** *ITL Online 2026*, *Mizuki's Simfiles*, and *Vocaloid Project Pad Pack 4th*. Frozen-val scaling ladder on the exact same **48 val charts** (**40** songs, **17,057** beats, **37,617** GT steps): `48t/48v` (`training_index_ladder_v1_50t_50v.json`) and `197t/48v` (`training_index_ladder_v1_200t_50v.json`, **187** train songs), excluding 4 negative-BPM Vocaloid charts (`hold_release_rakshasa_and_carcasses`, `the_rampage_of_hatsune_miku`, `news_39`, `viva_happy`).

### 3.2 Beat-aligned 48-slot placement and MERT frontends

- **Beat-grid formulation (`omalley2025ddcl`, `omalley2026itgpt`):** Each integer beat is quantized into 48 sub-beat slots (covering 16ths, 24ths/triplets, 32nds, and micro-divisions). Oracle decoding of ground-truth slot centers via `#OFFSET`/`#BPMS` achieves **0.999972** ±20 ms `timing_match` (`EXP-20261010-01`).
- **Scratch log-mel baselines:**
  - *DDCL (`omalley2025ddcl`):* 32 frames/beat × 80-band 3-channel multi-scale log-mel (`schluter2014onset`, `hamel2012multiscale`), 16-beat causal/anti-causal `ConvLSTM2D` windows (`memlen=15`) fused with per-beat `[meter, bpm]`, dense `[512, 256]`, focal BCE.
  - *ITGPT (`omalley2026itgpt`):* Full-song beat tensor padded to multiples of 64; `ItgptBeatCNNEncoder` maps `(32, 80, 3) → (24, d_model)` into 5 hierarchical intra-chunk attention levels (`1/4/16/32/64` beats), gated frame/beat compression, 8-layer global Transformer (`d_model=256`, `n_heads=8`) conditioned on scalar `diff_tok` + `bpm_tok`, residual depthwise `ConvHead`, and grid-weighted BCE.
- **Beat-aligned MERT frontends (`li2024mert`):** Precomputed 100 Hz hidden states from layer 6 of `m-a-p/MERT-v1-330M` (dim `1024`), resampled per beat and shared zero-copy across difficulties of the same song (`src/stepcovnet/ddcl/datasets.py`):
  - *DDCL-MERT (`build_mert_placement_model`):* 16 sub-beat MERT frames/beat → shared 4D `LayerNorm → Dense(128, gelu) → Dropout → Conv2D(64, (1,3)) → AvgPool2D(1,2)` into the identical bidirectional `LSTM(200)` + `[meter, bpm]` + `[512, 256]` head.
  - *ITGPT-MERT (`ItgptBeatMertEncoder`):* 24 sub-beat MERT frames/beat (exact 16th + triplet alignment matching `CNN_FRAMES_OUT = 24`) → `ItgptRmsNorm → Dense(256, gelu) → Dropout(0.1) → Conv1D(256, 3, same, gelu) → Dense(256)` plugged directly into ITGPT's unchanged hierarchical Transformer.
  - *Checkpoint selection:* Periodic full-val `M-slot48` checkpointing (`eval_slot_f1_every=16`, saving `best_slot_f1.keras`) alongside `best.keras` (`val_loss`) and `last.keras`.

### 3.3 Evaluation and audio-grounding protocol (`M-slot48`, `M-event-20ms`, `M-grounding`)

- **`M-slot48`:** Pooled micro F1 at τ=0.5 and threshold-swept `max-F1`, reported beside the beat-shuffled GT null floor (`M-null`: `0.3971` Dataset A; `0.6007` `D-final` ladder val).
- **`M-event-20ms`:** Decode occupied slots (τ=0.5) to onset times in seconds and score bipartite Hungarian ±20 ms F1 and headroom-normalized skill `(F1 − F1_null) / (1 − F1_null)` over the strongest audio-blind null (`ioi_shuffle`, `regular_grid`, `uniform_duration`).
- **`M-grounding` (`--diagnose_grounding`):** Score identical checkpoints under three `beat_audio` conditions while keeping metadata (`[meter, bpm]` or `diff_tok`/`bpm_tok`) and positional embeddings intact:
  1. `none` (real audio),
  2. `shuffle` (distribution-preserving random permutation of beats within each chart),
  3. `zeros` (zeroed `beat_audio`).
  Report `slot_f1_drop_shuffle`, `event_skill_drop_shuffle`, occupied Jaccard distance `1 − IoU`, and **train→val audio transfer** (`val_drop_shuffle / train_drop_shuffle`).

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

### 4.2 Dataset A 2×2 factorization: representation vs architecture (`T-frax-2x2`, `C10`)

Strict single-knob frontend swap (`log-mel → MERT`) on the exact same `D-frax-orig` split (`45` val charts, `17,523` beats, `17,868` GT steps, `null F1 = 0.3971`).

| Architecture | Frontend | Ckpt | `M-slot48` F1@0.5 | `M-slot48` max-F1 | ±20 ms `micro_event_f1` / `skill_event_f1` | `shuffle` drop (F1 / event-skill) | `zeros` drop (F1 / `1 − IoU`) | Source |
| ------------ | -------- | ---- | ----------------- | ----------------- | ------------------------------------------ | --------------------------------- | ----------------------------- | ------ |
| DDCL (BiLSTM) | Scratch log-mel | `last` (ep 128) | `0.6804` | `0.6849` @ 0.45 | `0.7008` / `+0.5670` | `−0.0181` / `−0.0331` | `−0.0805` / `0.421` | `EXP-20260817-01`, `EXP-20261123-03` |
| ITGPT (Transformer) | Scratch log-mel | `last` (ep 128) | `0.7104` | `0.7166` @ 0.35 | `0.7291` / `+0.6085` | `−0.1192` / `−0.1827` | `−0.0713` / `0.401` | `EXP-20260819-03`, `EXP-20261123-02` |
| **DDCL (BiLSTM)** | **Beat-aligned MERT** | **`last` / `best_slot_f1` (ep 128)** | **`0.7511`** (+7.07 pp) | **`0.7554`** @ 0.45 | **`0.7689` / `+0.6623`** (+9.53 pp) | **`−0.1312` / `−0.2063`** (7.3×) | **`−0.7511` / `1.000`** (`0` steps) | `EXP-20261123-03` |
| **ITGPT (Transformer)** | **Beat-aligned MERT** | **`best` (ep 38)** (`last` `0.7421`) | **`0.7519`** (+4.15 pp) | **`0.7633`** @ 0.35 | **`0.7679` / `+0.6638`** (+5.53 pp) | **`−0.1296` / `−0.1965`** | **`−0.2692` / `0.566`** | `EXP-20261123-02` |

- **Takeaway:** On 81-song Fraxtil, representation (**+7.07 / +4.15 pp**) dominates architecture (**+0.08 pp** F1@0.5 / **+0.79 pp** max-F1 under MERT), and ITGPT-MERT on 81 songs (**0.7519 / 0.7633**) outperforms 201-song Dataset B log-mel ITGPT (**0.735 / 0.740**).

### 4.3 Multi-pack transfer and data scaling (`T-ladder-scale`, `C11`, `C12`)

All runs evaluated on the **exact same frozen 48 val charts** of `D-final` (`17,057` beats, `37,617` GT steps, `null F1 = 0.6007`).

| Model & Recipe | Train Charts | Ckpt | Val `M-slot48` F1@0.5 | Val max-F1 | Val ±20 ms `skill_event_f1` | Val `shuffle` drop (F1 / event-skill) | Train `shuffle` drop | Audio Transfer | Source |
| -------------- | ------------ | ---- | --------------------- | ---------- | --------------------------- | ------------------------------------- | -------------------- | -------------- | ------ |
| Log-mel DDCL (`best_val_loss`) | `48` | `best` | `0.7272` | `0.7272` | `+0.6155` | `−0.0048` / `−0.0067` | — | ~audio-blind | `EXP-20261119-01` |
| Log-mel DDCL | `48` | `last` (ep 128) | `0.7307` | `0.7308` | `+0.6197` | `−0.0230` / `−0.0301` | `−0.1989` (`0.9392`) | **11.6%** | `EXP-20261119-02/03` |
| MERT-LSTM (`Adam`, `drop=0.2`) | `48` | `last` (ep 128) | `0.7647` | `0.7712` @ 0.35 | `+0.6656` | `−0.0782` / `−0.1085` | `−0.2324` (`0.9622`) | **33.6%** | `EXP-20261119-04/05` |
| MERT-LSTM + heavy reg (`wd=0.01`, `drop=0.4`, `noise=0.05`) | `48` | `best_slot_f1` (ep 96) | `0.7612` | `0.7612` | `+0.6611` | `−0.0533` / `−0.0766` | — | reg hurt grounding | `EXP-20261119-06` |
| ITGPT-MERT (`48t/48v`) | `48` | `best_slot_f1` (ep 32) / `last` (ep 128) | `0.7719` / `0.7702` | `0.7728` / `0.7703` | `+0.6695` / `+0.6652` | `−0.0619` / `−0.0848` | `−0.2649` (`0.9928`) | **32.0%** | `EXP-20261123-01` |
| **ITGPT-MERT (`200t/50v`)** | **`197`** (`187` songs) | **`best_slot_f1` (ep 64)** / **`last` (ep 128)** | **`0.8181`** / **`0.8151`** (+4.62 pp) | **`0.8192`** @ 0.55 | **`+0.7330`** (+6.35 pp) | **`−0.1092` / `−0.1103`** (4.8× mel) | `−0.2610` (`0.9875`) | **42.3%** (+10.3 pp) | `EXP-20261124-01` |

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

---

## 5. Discussion

1. **Why log-mel ITGPT beat log-mel DDCL (`C10`):** Our 2×2 grounding matrix reveals that log-mel DDCL on Dataset A was largely a metadata/beat-grid model (`shuffle` drop `−0.0181`), whereas log-mel ITGPT grounded 6.6× more strongly in audio (`−0.1192`). Providing beat-aligned MERT repairs DDCL's audio pathway (`−0.1312`, 7.3×) and collapses the architecture gap on homogeneous Fraxtil, while full-song Transformer attention still provides an edge on higher-diversity multi-pack charts (`D-final` `48t/48v`) and higher max-F1 on Dataset A (`0.7633` vs `0.7554`).
2. **Why `shuffle` is the architecture-agnostic grounding metric (`C9`):** Under `zeros`, MERT-LSTM/DDCL-MERT collapses completely (`0` steps on Dataset A, `0.1296` F1 on `D-final`) because zeroed inputs push its Conv/LSTM biases off-manifold, whereas ITGPT-MERT's RMSNorm + residuals + `diff_tok`/`bpm_tok`/`level6_pos` gracefully degrade to a prior template (`0.7290` on `D-final`, `0.4826` on Dataset A). Distribution-preserving `shuffle` keeps activations in-distribution and cleanly measures audio steering across architectures.
3. **Regularization asymmetry (`C12`):** Heavy dropout + input noise on the 1024-dim MERT stream (`EXP-20261119-06`) selectively penalizes the audio branch relative to clean low-dimensional metadata (`[meter, bpm]`), simultaneously lowering val F1 and `shuffle` drop. Light `AdamW(wd=0.01)` + `dropout=0.1` with zero audio noise stabilizes ITGPT-MERT cleanly on both Dataset A and `200t/50v`.
4. **Data scaling converts directly into audio transfer (`C11`):** Scaling `ITGPT-MERT` from 48 to 197 training charts left the `shuffle` prior floor flat (`0.710 → 0.709`) while raising val F1 by **+4.62 pp** (`0.7719 → 0.8181`) and train→val audio transfer from **32.0% → 42.3%**. With train F1 still at `0.9875` (`57.7%` un-transferred audio lift), scaling to `300t/50v` and full `final_data` (~1,750 charts) has substantial headroom.
5. **Why dense/event/AR-on-times tracks were diagnostic rather than primary (`C3`):** K-query event prediction hit a ~31% oracle ceiling (`EXP-20260606-11`), dense TCNs plateaued at `+0.414` event skill (`EXP-20260824-01`), and AR-on-times `Dense(max_patches)` pointers passed single-song overfit while being 99.96% audio-blind (`EXP-20260804-05`), motivating the `M-grounding` protocol now used on beat-grid models.

---

## 6. Conclusion and Future Work

Beat-aligned pretrained music representations (`li2024mert`) paired with architecture-agnostic audio-grounding diagnostics (`M-grounding`) resolve two central bottlenecks in neural DDR/ITG step placement: they lift both DDCL (**+7.07 pp**) and ITGPT (**+4.15 pp**) on Original Fraxtil, prove representation drives larger gains than architecture alone at 81-song scale, and scale cleanly to **0.8181** `M-slot48` F1 (**+0.7330** ±20 ms event skill, **42.3%** audio transfer) on multi-pack community charts. Future work: (1) completing the `300t/50v` and full `final_data` (~1,750 charts) scaling curve, (2) pairing the **0.818** ITGPT-MERT placement head with Stage 2 four-panel step selection (arrows, jumps, holds) for end-to-end playable `.sm` generation, and (3) multi-author evaluation on ITG (`D-itg`).

---

## Appendix

- Paper ledger (citations, claims, datasets, metrics): [PAPER_LEDGER.md](PAPER_LEDGER.md)
- BibTeX: [paper.bib](paper.bib)
- Pipeline: [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
- Experiment log (authoritative): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- Design reasoning: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md)
- Onset metrics: [ONSET_METRICS.md](ONSET_METRICS.md)
- Config references (seed 42): `configs/ddc/ddcl_placement_fraxtil.json`, `configs/ddc/ddcl_mert_placement_fraxtil_128ep.json`, `configs/ddc/itgpt_placement_fraxtil_orig_128ep.json`, `configs/ddc/itgpt_mert_placement_fraxtil_orig_128ep.json`, `configs/ddc/ddcl_mert_placement_ladder_50t_50v.json`, `configs/ddc/itgpt_mert_placement_ladder_50t_50v.json`, `configs/ddc/itgpt_mert_placement_ladder_200t_50v.json`
