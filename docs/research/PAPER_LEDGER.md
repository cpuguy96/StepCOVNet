# Paper ledger — citations, claims, and what to keep

**Purpose:** Everything that should survive into the research paper lives here or is linked from here. Lab noise stays in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) and [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md). The draft skeleton is [PAPER_OUTLINE.md](PAPER_OUTLINE.md). Machine-readable refs: [paper.bib](paper.bib).

**Prior art is cited, not copied.** Recreating a published pipeline on its dataset is how we establish a comparable baseline. The paper must name the source, the split, and the metric whenever we report those numbers.

**Agent rule:** When a paper, codebase, dataset pack, or metric definition is used as prior art, baseline, method, or data source, add a row here and a BibTeX entry in `paper.bib` **in the same turn**. Cite by key (e.g. `donahue2017ddc`).

---

## 1. What to keep for the paper

| Track | Keep | Do not dump into the outline |
| ----- | ---- | ---------------------------- |
| **Citations** | Key, venue, URL, *how we used it* (related work / baseline / method / dataset / metric) | Full paper notes; chat paraphrases without a key |
| **Claims** | One sentence + status + supporting `EXP-` / `NOTE-` | Every failed probe |
| **Contributions vs prior art** | What is new vs what we reimplement and cite | “Inspired by” without a citation key |
| **Datasets** | Pack names, song/chart counts, split rule, seed | Raw download logs |
| **Metrics** | Name, tolerance, peak-pick vs beat-grid, whether a null floor is required | Ad-hoc spreadsheet columns |
| **Baselines** | Reimplemented (ours, same split) vs cited-from-paper (their split) | Mixing those two in one table |
| **Negative results** | Only if they change a claim (e.g. audio-blind pointer) | Full AR locality wrap-up |
| **Code / configs** | Commit, config path, training seed for any number in the paper | Every smoke checkpoint |
| **Figures / tables planned** | Caption + which EXP it comes from | Debug plots |

Promotion path: ledger row → [PAPER_OUTLINE.md](PAPER_OUTLINE.md) section → camera-ready text. Never skip the ledger.

---

## 2. Citation catalog

Status: **in_paper** = committed for related work or methods; **likely** = cite if that section is written; **candidate** = read, not yet needed.

Role: **related_work** | **baseline** | **method** | **dataset** | **metric** | **background**.

| Key | Work | Role | Status | How we use it |
| --- | ---- | ---- | ------ | ------------- |
| `donahue2017ddc` | Donahue, Lipton, McAuley. Dance Dance Convolution. ICML 2017. [arxiv:1703.06891](https://arxiv.org/abs/1703.06891). Code: [chrisdonahue/ddc](https://github.com/chrisdonahue/ddc) | related_work, baseline, dataset, metric | **in_paper** | Defines learning-to-choreograph: placement then selection. Original Fraxtil (90 songs / 3 packs) and ITG (133 songs). 10 ms C-LSTM, ±20 ms peak-pick F1, LSTM selection without audio. |
| `omalley2025ddcl` | O’Malley. Dance Dance ConvLSTM. arXiv:2507.01644, 2025. Code: [miguelomalley/DDCL](https://github.com/miguelomalley/DDCL) | related_work, baseline, method | **in_paper** | BPM-first, 32 frames/beat, 48 slots/beat, ConvLSTM, audio in selection. Same original Fraxtil as DDC. |
| `omalley2026itgpt` | O’Malley. ITGPT. arXiv:2607.14148, 2026. Code: [miguelomalley/ITGPT](https://github.com/miguelomalley/ITGPT) | related_work, baseline, dataset | **in_paper** | Hierarchical transformer placement, diagnostic BPM/difficulty net, 500-step selection, expanded Fraxtil (253 songs / 8 packs). |
| `yi2023goct` | Yi, Lee, Lee. Beat-aligned spectrogram-to-sequence rhythm-game charts. arXiv:2311.13687, 2023 | related_work | **likely** | Transformer chart generation; ITGPT compares as GOCT. Timing-only for DDR/ITG. |
| `schluter2014onset` | Schlüter & Böck. CNN onset detection. ICASSP 2014 | related_work, method | **in_paper** | DDC’s CNN placement baseline and 80-band multi-scale log-mel PRE (`stepcovnet.ddc.features`). |
| `eyben2010blstm` | Eyben et al. BLSTM onset detection. ISMIR 2010 | related_work | **likely** | Classic RNN onset detection. |
| `bello2005onset` | Bello et al. Onset detection tutorial. IEEE 2005 | background | **likely** | Problem statement for musical onsets. |
| `hamel2012multiscale` | Hamel, Bengio, Eck. Multiple timescale features. ISMIR 2012 | method | **in_paper** | 23/46/93 ms STFT channels in DDC PRE (`FFT_SIZES` 1024/2048/4096). |
| `vandewetering2016bpm` | van de Wetering. Non-causal beat tracking / ArrowVortex | method | **likely** | Cite if generation uses ArrowVortex BPM (DDCL/ITGPT). |
| `li2024mert` | Li et al. MERT. ICLR 2024. arXiv:2306.00107 | method | **in_paper** | Beat-aligned 100 Hz `MERT-v1-330M` frontend (single layer-6 in Dataset A 2×2; per-layer-normalized multi-layer mean `(2, 6, 10, 16)` in champion recipes) replacing scratch log-mel (`EXP-20261119-04`, `EXP-20261123-01/02/03`, `EXP-20261125-03/10`, `EXP-20261127-01`–`06`). |
| `foscarin2024beatthis` | Foscarin, Schlüter, Widmer. Beat This! ISMIR 2024. [arxiv:2407.21658](https://arxiv.org/abs/2407.21658). Code: [CPJKU/beat_this](https://github.com/CPJKU/beat_this) | method, baseline | **in_paper** | Neural beat/downbeat tracker without DBN postprocessing; implemented in pure PyTorch (`stepcovnet.beat_this_model`) and paired with octave-folded global/piecewise lattice regularization, integer-BPM lock, per-section zero-lag attack snapping, and ITGPT downbeat bar-phase alignment (`stepcovnet.beat_tracking`) to lift `BeatF1@70ms` to `0.9254` and surpass oracle ±20 ms `evF1` end-to-end on raw audio (**`0.8204` Arm K** `101.0%` oracle at `100.5%` density, `0.8196` zero-shot `1,713t` `102.2%`, `0.8193` Arm C `101.4%`; `EXP-20261127-06`–`EXP-20261128-01`). |
| `park2019specaugment` | Park et al. SpecAugment. Interspeech 2019. arXiv:1904.08779 | method | **in_paper** | Downbeat-preserving MERT-only SpecAugment (`keep0`, 4 time/freq masks, `p=0.15`, never masking beat frame `0`; `EXP-20261126-15/16`). |
| `liang2021rdrop` | Liang et al. R-Drop. NeurIPS 2021. arXiv:2106.14448 | method | **in_paper** | Bidirectional Bernoulli KL consistency (`rdrop_kl_weight=2.0`) over dropout sub-networks; improves Dataset A from-scratch & transfer convergence (`EXP-20261127-02/05`). |
| `halina2021taikonation` | Halina & Guzdial. TaikoNation. FDG 2021 | related_work | **candidate** | Other rhythm-game chart generation. |
| `lin2018generationmania` | Lin, Xiao, Riedl. GenerationMania. arXiv:1806.11170 | related_work | **candidate** | BeatMania choreography. |
| `tsujino2018ddg` | Tsujino & Yamanishi. Dance Dance Gradation. 2018 | related_work | **candidate** | DDC trained per coarse difficulty. |
| `okeeffe2003dancingmonkeys` | O’Keeffe. Dancing Monkeys. 2003 | related_work | **candidate** | Rule-based DDR charts; DDC’s pre-learning baseline. |
| `qi2020prophetnet` | Qi et al. ProphetNet. EMNLP 2020 | method | **candidate** | Cite only if we use ITGPT’s 4-step future-prediction curriculum. |
| `carion2020detr` | Carion et al. DETR. ECCV 2020 | related_work | **candidate** | Cite if the K-query event-onset track is in the paper. |
| `zeghidour2021soundstream` | Zeghidour et al. SoundStream. arXiv:2107.03312 | method | **candidate** | Cite only if we use residual vector quantization in selection. |

When adding a row: copy the BibTeX into [paper.bib](paper.bib) first, then this table.

---

## 3. Claim ledger

Status: **hypothesis** | **supported** | **not_supported** | **positioning** (no new experiment; literature fact).

| ID | Claim | Status | Evidence | Paper section |
| -- | ----- | ------ | -------- | ------------- |
| C1 | Chart generation decomposes into step placement and step selection | positioning | `donahue2017ddc` | Intro / related work |
| C2 | Published increments on this task are mostly the time grid (10 ms vs 48 slots/beat) and whether selection sees audio | positioning | `donahue2017ddc`, `omalley2025ddcl`, `omalley2026itgpt`; [NOTE-20260814-01](DISCUSSION_NOTES.md#note-20260814-01-literature-recreation-before-incremental-claims) | Related work |
| C3 | Literature AR generates arrows given known times; StepCOVNet `onset_ar` generates times — not the same task | positioning | NOTE-20260814-01; [EXP-20260804-05](EXPERIMENT_LOG.md#exp-20260804-05-the-ar-pointer-never-reads-the-audio--the-head-is-absolute-index-classification-not-a-pointer) | Related work / discussion |
| C4 | `final_data` (ITL / Mizuki / Vocaloid) is not the Fraxtil literature corpus | positioning | [TRAINING_DATA_SETUP.md](TRAINING_DATA_SETUP.md) vs DDC/ITGPT pack lists | Methods (data) |
| C5 | A new method is comparable to DDC/DDCL/ITGPT only on their dataset, split, grid, and metric | hypothesis | NOTE-20260814-01 | Methods (eval) |
| C6 | Recreated DDC matches published Fraxtil placement/selection numbers | hypothesis (placement **Partial**) | Placement T-repro accepted: val **0.652 / 0.734** vs paper **0.681 / 0.756** ([EXP-20260815-03](EXPERIMENT_LOG.md#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap)). Selection not recreated. | Results |
| C7 | Recreated DDCL / ITGPT match published ordering on the matching corpus | supported (Dataset A 2-way; not vs expanded Table 2) | Dataset A 128-ep ITGPT last F1@0.5 **0.7104** vs DDCL **0.6804** on `M-slot48` ([EXP-20260819-03](EXPERIMENT_LOG.md#exp-20260819-03-itgpt-48-slot-placement-128-ep-on-dataset-a), [EXP-20260817-01](EXPERIMENT_LOG.md#exp-20260817-01-ddcl-48-slot-placement-128-ep-on-dataset-a)). Frozen DDC Hamming-peak snap F1@0.5 **0.364** vs null **0.397** (no slot skill) ([EXP-20260819-04](EXPERIMENT_LOG.md#exp-20260819-04-ddc-hamming-peak-snap-onto-m-slot48-on-dataset-a)). Dataset B last **0.735** is a different corpus ([EXP-20260819-02](EXPERIMENT_LOG.md#exp-20260819-02-itgpt-48-slot-placement-128-ep-on-dataset-b)). Not vs Table 2 **0.78 / 0.80**. DDC literature column stays `M-ddc-20ms` **0.652 / 0.734**. | Results (`T-repro`) |
| C8 | Beat-aligned `M-slot48` preserves ±20 ms timing (oracle **0.999972**), while rank-indexed `timing_match` collapses on real models from cumulative `FP − FN` rank drift rather than timing error | supported | Oracle: [EXP-20261010-01](EXPERIMENT_LOG.md#exp-20261010-01-48-slot-oracle-clears-timing_match-on-ladder-val); counterfactual mask ablations (`raw` **0.0072**, `fp_removed` **0.0536**, `fn_filled` **0.0408**, `beat_top_k` **0.8338** vs bipartite ±20 ms `skill_event_f1` **+0.616**): [EXP-20261118-01](EXPERIMENT_LOG.md#exp-20261118-01-counterfactual-slot-masks-prove-rank-drift-breaks-timing_match-while-decoded-slot-times-beat-dense-tcn), [NOTE-20261118-01](DISCUSSION_NOTES.md#note-20261118-01-counterfactual-slot-masks-confirm-cumulative-fp--fn-rank-drift-breaks-timing_match-and-decoded-48-slot-times-beat-dense-tcn) | Methods (`M-timing` / `M-event-20ms`) & Results (`T-timing-diag`) |
| C9 | High `M-slot48` F1 can mask an audio-blind model; distribution-preserving beat-audio shuffling (`none − shuffle`) is the architecture-agnostic audio-grounding test (unlike `zeros`, which confounds OOD collapse with prior fallback) | supported | Early `val_loss` DDCL `best.keras` keeps **95.8%** of steps under `zeros` (**0.727 → 0.722**): [EXP-20261119-01](EXPERIMENT_LOG.md#exp-20261119-01-ladder-ddcl-bestkeras-is-almost-entirely-audio-blind--958-of-steps-survive-zeroed-audio); ITGPT-MERT drops further under `shuffle` (**0.685**) than `zeros` (**0.729**): [EXP-20261123-01](EXPERIMENT_LOG.md#exp-20261123-01-beat-aligned-itgpt-mert-beats-mert-lstm-on-both-val-f1-0772-and-shuffle-grounding-0085), [NOTE-20261123-01](DISCUSSION_NOTES.md#note-20261123-01-beat-aligned-itgpt-mert-sets-a-new-ladder-champion-0772-f1-0085-shuffle-drop-and-saturates-the-48-song-split-0993-train-f1) | Methods (`M-grounding`) & Discussion |
| C10 | **Dataset A 2×2 factorization:** replacing scratch log-mel with beat-aligned MERT (`li2024mert`) lifts DDCL by **+7.07 pp** (**0.6804 → 0.7511**) and ITGPT by **+4.15 pp** (**0.7104 → 0.7519**, max-F1 **0.7633**); representation (+4.2 to +7.1 pp) dominates architecture (+0.08 pp F1@0.5 / +0.79 pp max-F1 under MERT), and log-mel ITGPT's +3.0 pp lead over DDCL was driven by a **6.6×** audio-grounding gap (**−0.1192** vs **−0.0181**) | supported | [EXP-20261123-02](EXPERIMENT_LOG.md#exp-20261123-02-itgpt-mert-single-knob-swap-lifts-dataset-a-fraxtil-42-pp-over-paper-faithful-itgpt-0752-vs-0710), [EXP-20261123-03](EXPERIMENT_LOG.md#exp-20261123-03-ddcl-mert-completes-dataset-a-2x2-matrix--71-pp-and-73x-audio-grounding-over-log-mel-ddcl), [NOTE-20261123-03](DISCUSSION_NOTES.md#note-20261123-03-dataset-a-2x2-factorization-proves-representation-42-to-71-pp-dominates-architecture-01-to-08-pp-under-mert-and-explains-itgpts-original-log-mel-lead-as-a-66x-audio-grounding-gap) | Abstract, Results (`T-frax-2x2`), Discussion |
| C11 | **Multi-pack transfer & scaling (`final_data`, frozen 48 val charts):** MERT triples train→val audio transfer (**11.6% → 33.6%**), full-song hierarchical attention (`ITGPT-MERT`) beats BiLSTM (**0.7719 / −0.0848** vs **0.7647 / −0.0782**), and scaling across `48 → 197 → 294 → 1,713` train charts lifts frozen-48v F1@0.5 from **0.7719 → 0.8181 → 0.8276** (constant LR) and **0.84974 → 0.86493** (champion recipe, **10,507** errors, `shuffle −0.1371`, `16th_off 0.7635`, `24th 0.5922`) | supported | [EXP-20261119-04](EXPERIMENT_LOG.md#exp-20261119-04-beat-aligned-mert-lifts-ladder-48-slot-f1-to-0765-and-triples-val-audio-grounding-at-9x-speed), [EXP-20261123-01](EXPERIMENT_LOG.md#exp-20261123-01-beat-aligned-itgpt-mert-beats-mert-lstm-on-both-val-f1-0772-and-shuffle-grounding-0085), [EXP-20261124-01/02](EXPERIMENT_LOG.md#exp-20261124-02-300t50v-completes-3-point-itgpt-mert-scaling-curve-at-0828-f1--0832-max-f1-and-447-audio-transfer), `EXP-20261127-01`, `EXP-20261127-04` | Abstract, Results (`T-ladder-scale`), Discussion |
| C12 | Heavy audio-path regularization (`AdamW wd=0.01`, `dropout=0.4`, `noise=0.05`) on MERT-LSTM hurts both val F1 (**0.7647 → 0.7612**) and audio grounding (`shuffle` drop **−0.0782 → −0.0533**) by pushing weight back onto `[meter, bpm]`, whereas light `AdamW(wd=0.01)` + `dropout=0.1` + downbeat-preserving SpecAugment (`keep0`) + R-Drop improves both val F1 and grounding | supported | [EXP-20261119-06](EXPERIMENT_LOG.md#exp-20261119-06-heavy-regularization-hurts-both-mert-val-f1-and-audio-grounding-by-pushing-back-onto-meter-bpm), `EXP-20261126-15/16`, `EXP-20261127-02` | Discussion (`T-grounding`) |
| C13 | **Multi-layer MERT + training recipe (`density` + `keep0` SpecAugment + `beat_aux + dice + pw` + R-Drop):** Pre-averaging normalized MERT layers `(2, 6, 10, 16)`, replacing pack-specific `#METER` with chart step density (`steps/beat`), adding downbeat-preserving SpecAugment (`park2019specaugment`), 3-way auxiliary/loss synergy (`ItgptBeatAuxHead` + soft-Dice + positive class weight), and R-Drop (`liang2021rdrop`) lifts from-scratch Dataset A ITGPT-MERT from **0.7519 → 0.77971** peak (**0.77585 ± 0.00341** 3-run mean, **0.80085** ±20 ms `evF1`, **+6.93 pp** over log-mel ITGPT; from-scratch `fine_subdiv 0.03 + pw=1.00` reaches all-time lowest from-scratch **7,984** errors, `99.97%` exact density, and `shuffle −0.1484`) and `300t/50v` from **0.8276 → 0.84974** (**11,593** errors) with zero extra inference parameters, outperforming wider/deeper `d_model=384, n_layers=8` (`0.84836`) while `silence_negative_prob=0.04` achieves **100%** silence FP suppression (`35,462 → 0` FPs) | supported | `EXP-20261125-03/09/10`, `EXP-20261126-15/16`, `EXP-20261127-01`, `EXP-20261127-02`, `EXP-20261127-03` (14 negative Tier-C ablations cleanly falsified), `EXP-20261128-02`, `EXP-20261128-03` | Abstract, Methods, Results (`T-frax-2x2`, `T-ladder-scale`) |
| C14 | **Cross-corpus zero-shot & subdivision-selective `fine_subdiv` soft-Dice transfer (`1,713t → Dataset A`):** Pretraining ITGPT-MERT on `1,713` non-Fraxtil charts (`1,109` songs) achieves **0.77870** F1@0.5 / **0.80337** ±20 ms `evF1` **zero-shot** on Dataset A (beating in-domain from-scratch MERT mean `0.77585`); fine-tuning onto Dataset A (`81` songs) reveals a clean loss progression from global Dice (`Arm C` `0.78574`, `7,946` err) → no-Dice (`Soup_F_H_I` `0.78726`, `7,708` err, `24th −9.6 pp`) → **subdivision-selective `fine_subdiv` soft-Dice** (`16th_off` + `24th` only, zero-gated on sparse charts), where **single checkpoint `Arm K` shatters `0.790` at `0.79043` F1@0.50 exact / `0.81303` ±20 ms `evF1` / `7,549` errors (`+8.00 pp` over log-mel ITGPT, `shuffle −0.1466`, `<=8 0.7402`, `24th 0.3889`)** with 2-seed verification (`Arm K + L` mean `0.78882 / 0.81134`) | supported | `EXP-20261127-04`, `EXP-20261127-05`, `EXP-20261127-06`, `EXP-20261127-11`, `EXP-20261128-01` | Abstract, Results (`T-transfer-scaling`), Discussion |
| C15 | **Per-song Dataset A error audit (`tsunamix_iii/isis` fractional-BPM artifact):** On `8/9` Dataset A val songs, the fine-tuned model scores **0.81928** `M-slot48` F1 / **0.81946** ±20 ms `evF1` (`0.764–0.868`); ONE outlier song (`tsunamix_iii/isis`, `31` fractional-beat BPM gimmicks → `61.4%` `micro/32nd` slots) drives `21.4%` of total errors (`1,701/7,946`) and collapses exact-slot F1 (`0.3274`) while grid-free ±20 ms `evF1` doubles (`0.6564`) | supported | `EXP-20261127-06` | Results (`T-error-audit`), Discussion |
| C16 | **Real-world raw-audio beat tracking, octave-folded medley regularization, attack snapping, & downbeat bar-phase alignment (0 `.sm` beats):** Without oracle simfile beat grids, raw `BeatThis` (`foscarin2024beatthis`) + slot-center decoding suffers a MIR-phase offset at ±20 ms (`0.5804` `evF1`; raw `librosa` `0.0369`), whereas pairing built-in pure-PyTorch `BeatThis` with exact grid-edge decoding (`0.7151–0.7159`), octave-folded global/piecewise lattice regularization (`BeatF1@70ms 0.7965 → 0.9257`), Beat-0 `[-5 ms, 0)` preservation + per-section zero-lag attack snapping, and ITGPT downbeat bar-phase alignment (`index % 4 == 0`) **surpasses oracle `.sm` beat grids across all models**: **`0.8204` (`101.0%` of oracle `0.8125`) at `100.5%` exact density on single-checkpoint `Arm K`**, **`0.8196` (`102.2%` of oracle `0.8021`) completely zero-shot on `1,713t`** (all `9/9` songs within `0.57 pp` of or beating oracle), and `0.8193` (`101.4%`) on Arm C | supported | `EXP-20261127-06`–`EXP-20261128-01`, `scripts/eval_raw_audio_onset.py`, `stepcovnet.beat_this_model`, `stepcovnet.beat_tracking` | Abstract, Methods, Results (`T-raw-audio-beats`), Discussion |

Do not promote a claim to the abstract until an `EXP-` supports it. Claims C8–C16 are all supported and promoted into [PAPER_OUTLINE.md](PAPER_OUTLINE.md).

---

## 4. Datasets (methods)

| ID | Name | Packs | Size (as published) | Split | Paper use |
| -- | ---- | ----- | ------------------- | ----- | --------- |
| D-frax-orig | Original Fraxtil | Tsunamix III; Fraxtil’s Arrow Arrangements; Fraxtil’s Beast Beats | **Measured 2026-08-15:** 90 songs, **463** chart rows (**450** standard + **13** edit). Paper table: 90 / 450 (`donahue2017ddc`) | `stratified_song_v1` seed 42, val_fraction 0.1 → **81 / 9** songs. Standard-only: `data/literature_fraxtil_orig/training_index_standard.json` (**405 / 45** rows, **17,523** val beats, **17,868** val GT steps). Not DDC’s unpublished 80/10/10 IDs | Literature 2×2 benchmark, from-scratch champion ablations, zero-shot/fine-tuned transfer (`1,713t → Dataset A`), per-song error audit, and raw-audio beat tracking benchmark (`EXP-20260815-01` through `EXP-20261127-06`) |
| D-itg | In The Groove | ITG 1; ITG 2 | 133 songs, 652 charts | same rule | Multi-author DDC check (not yet ingested) |
| D-frax-exp | Expanded Fraxtil | D-frax-orig plus Cute Charts; Sweet Arrows and Hella Steps vols 1–3 (vol 4 **404** on fra.xtil.net) | Paper: 253 songs / 952 charts (`omalley2026itgpt`). **Measured 2026-08-18:** 232 raw folders, **222** exported songs / **747** chart rows (**722** standard + **25** edit). Ten SAHS songs skipped | `stratified_song_v1` seed 42, val_fraction 0.1 → **201 / 21** songs. Standard-only: `data/literature_fraxtil_exp/training_index_standard.json` (**653 / 69** rows). Not ITGPT’s unpublished split | Recreate ITGPT placement (held; vol 4 missing). Ingest [EXP-20260818-01](EXPERIMENT_LOG.md#exp-20260818-01-expanded-fraxtil-dataset-b-ingested); 128-ep [EXP-20260819-02](EXPERIMENT_LOG.md#exp-20260819-02-itgpt-48-slot-placement-128-ep-on-dataset-b) |
| D-final | StepCOVNet `final_data` | ITL Online 2026; Mizuki’s Simfiles; Vocaloid Project Pad Pack 4th | Full valid index (`10` negative-BPM Vocaloid songs excluded): **1,109** songs, **1,713 / 176** train/val chart rows (**61,347** full-val beats, **134,327** full-val GT steps; `60.08 GB` float32 MERT `L2+6+10+16` cache). Frozen-val scaling ladder rungs: `50t_50v` (**48 / 48**), `200t_50v` (**197 / 48**), `300t_50v` (**294 / 48**), and full `1,713t` (**0** song overlap with frozen **48v**: **17,057** beats, **37,617** GT steps) | `stratified_song_v1` seed 42; frozen 48-chart val across all 4 scaling rungs + full 176-chart val | Multi-pack transfer, grounding diagnostics, 4-point data-scaling ladder (`48 → 197 → 294 → 1,713`), and pretraining corpus for Dataset A transfer (`EXP-20261118-01` through `EXP-20261127-05`). Never mixed with Dataset A in one table |

Record the **actual song IDs in each split** (file under `data/` or a tracked manifest) before quoting a number. If DDC’s unpublished split cannot be recovered, say so and use a documented seed-42 song-grouped split.

---

## 5. Metrics (methods)

| ID | Name | Definition | Used by | Notes |
| -- | ---- | ---------- | ------- | ----- |
| M-ddc-20ms | Peak-pick F1 @ ±20 ms | Hamming-smoothed 10 ms probs, per-difficulty threshold, ±20 ms match | `donahue2017ddc` | In-repo: `stepcovnet.ddc.peak_pick` / `scripts/eval_ddc_placement.py`. Do not mix with beat-grid F1 |
| M-slot48 | 48-slot/beat F1 | Binary vector of length 48 per beat; report @ 0.5 and max-F1 | `omalley2025ddcl`, `omalley2026itgpt` | In-repo: `stepcovnet.ddcl.evaluation` / `scripts/eval_ddcl_placement.py` / `scripts/eval_itgpt_placement.py`. DDC POST snap: `scripts/eval_ddc_placement.py --slot48` ([EXP-20260819-04](EXPERIMENT_LOG.md#exp-20260819-04-ddc-hamming-peak-snap-onto-m-slot48-on-dataset-a)). No 20 ms window |
| M-sel-acc | Selection top-1 / top-2 / hold acc | Teacher-forced next-step over 256 classes | all three | Hold acc is the leftover error in ITGPT vs DDCL |
| M-timing | `timing_match` @ 20 ms | Ordered match / max(n_pred, n_ref) | StepCOVNet diagnostic | Rank-indexed; proved fragile to cumulative `FP − FN` rank drift ([EXP-20261118-01](EXPERIMENT_LOG.md#exp-20261118-01-counterfactual-slot-masks-prove-rank-drift-breaks-timing_match-while-decoded-slot-times-beat-dense-tcn)). DDC literature column stays `M-ddc-20ms` ([EXP-20260815-05](EXPERIMENT_LOG.md#exp-20260815-05-accepted-ddc-peaks-have-no-ordered-timing_match-skill)) |
| M-event-20ms | Decoded ±20 ms / ±50 ms bipartite event F1 & `skill_event_f1` | Decode occupied `M-slot48` centers to seconds via `#OFFSET`/`#BPMS` (or automatic beat tracker timestamps), score Hungarian ±20 ms (and ±50 ms for raw-audio beat tracking) F1 and headroom-normalized skill `(F1 − F1_null) / (1 − F1_null)` over strongest audio-blind null (`ioi_shuffle` / `regular_grid` / `uniform_duration`) | StepCOVNet placement eval (`stepcovnet.ddcl.evaluation`, `stepcovnet.onset_events.metrics`, `stepcovnet.onset_null_baseline`) | Bridges beat-grid `M-slot48` and continuous-time onset evaluation without rank-drift collapse or fractional-BPM slot artifacts (`EXP-20261127-06`) |
| M-grounding | Audio-corruption grounding & transfer ratio | Evaluate `none`, beat-permuted `shuffle`, and zeroed `zeros` on `beat_audio` with metadata/position intact; report `slot_f1_drop_shuffle`, `event_skill_drop_shuffle`, `1 − IoU`, and train→val audio transfer `val_drop_shuffle / train_drop_shuffle` | StepCOVNet (`--diagnose_grounding` in `eval_ddcl_placement.py` and `eval_itgpt_placement.py`) | Required beside `M-slot48` to distinguish genuine audio grounding from metadata/grid templates (`EXP-20261119-01`–`EXP-20261127-05`) |
| M-null | Audio-blind / chance floor | Beat-shuffled GT `M-slot48` floor (`0.3971` Dataset A; `0.6007` ladder val) and density-matched ±20 ms `ioi_shuffle` floor | EXP-20260804-03/05, all `M-slot48` & `M-event-20ms` runs | Required beside any number we claim as skill |

Canonical in-repo definitions: [ONSET_METRICS.md](ONSET_METRICS.md). Paper tables must name **M-*** IDs so DDC and ITGPT columns stay comparable.

---

## 6. Baselines vs contributions

| Item | Cite | Reimplement on our split? | Counts as our contribution? |
| ---- | ---- | ------------------------- | --------------------------- |
| DDC C-LSTM placement + LSTM selection | `donahue2017ddc` | Yes, before claiming gains | No — baseline |
| DDCL beat-grid ConvLSTM | `omalley2025ddcl` | Yes, if we report vs DDCL | No — baseline |
| ITGPT hierarchical transformer | `omalley2026itgpt` | Yes, if we report vs ITGPT | No — baseline |
| Fraxtil / ITG packs | `donahue2017ddc`, `omalley2026itgpt` | N/A (data) | No — cite the papers and pack authors |
| 80-band multi-scale mel | `schluter2014onset`, `hamel2012multiscale`, `donahue2017ddc` | If we use it | No — method citation |
| MERT frontend on Dataset A 2×2 (`DDCL-MERT`, `ITGPT-MERT`) | `li2024mert` | Ablation (`configs/ddc/{ddcl,itgpt}_mert_placement_fraxtil*_128ep.json`, seed 42) | **Yes — measured** (+7.07 pp DDCL, +4.15 pp ITGPT, 2×2 factorization; `EXP-20261123-02/03`) |
| Multi-layer MERT `(2, 6, 10, 16)` + density conditioning + `keep0` SpecAugment + `beat_aux` + subdivision-selective `fine_subdiv` soft-Dice + R-Drop recipe | `li2024mert`, `park2019specaugment`, `liang2021rdrop` | Measured on Dataset A + `D-final` (`EXP-20261125-03`–`EXP-20261128-01`) | **Yes — measured** (+6.93 pp from-scratch / **+8.00 pp** transferred to `0.79043` / `0.81303 evF1` / `7,549` err; 0 extra inference params) |
| Audio-corruption grounding & transfer-ratio protocol (`M-grounding`) | ours (`stepcovnet.ddcl.evaluation`, `stepcovnet.itgpt.evaluation`) | Applied across all Dataset A & `final_data` experiments | **Yes — measured** (exposes early-ckpt audio blindness & log-mel DDCL's 6.6× grounding gap) |
| Full `final_data` 4-point scaling (`48 → 197 → 294 → 1,713`) + zero-shot & `fine_subdiv` Dice fine-tuned transfer to Dataset A | this ledger `D-final` | Experiment (frozen 48v + full 176v + Dataset A 45v) | **Yes — measured** (`48v 0.86493`, zero-shot Dataset A `0.77870 / 0.80337 evF1`, single-ckpt `Arm K` **`0.79043 / 0.81303 evF1 / 7,549 err`**, 2-seed mean `0.78882`; `EXP-20261127-04/05/11`, `EXP-20261128-01`) |
| Raw-audio automatic beat tracking, octave-folded medley regularization, attack snapping, & downbeat bar-phase alignment | `foscarin2024beatthis` | Measured on Dataset A (`EXP-20261127-06`–`EXP-20261128-01`); built into `stepcovnet.beat_this_model` + `stepcovnet.beat_tracking` | **Yes — measured** (`BeatF1@70ms 0.9254`, raw-audio ±20 ms `evF1` **101.0–102.2% of oracle**: **`0.8204` Arm K** at `100.5%` exact density, `0.8196` ZS `1,713t`, `0.8193` Arm C, with zero `.sm` beat grid) |
| Hold-note modeling that beats DDCL hold acc | `omalley2025ddcl` as the number to beat | Deferred (Stage 2 selection) | **Yes**, if measured |

---

## 7. Planned tables / figures

| ID | Content | Source | Status |
| -- | ------- | ------ | ------ |
| T-related | Placement F1 and selection acc for DDC / DDCL / ITGPT as **published** | ITGPT Table 2 and 5; DDC Tables 2–3 | draft numbers in NOTE-20260814-01 |
| T-repro | Our Dataset A baseline reimplementations (`M-ddc-20ms` for DDC; `M-slot48` for DDCL / ITGPT / DDC peak-snap) | `EXP-20260815-03`, `EXP-20260817-01`, `EXP-20260819-03`, `EXP-20260819-04` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.1 |
| T-frax-2x2 | **Dataset A 2×2 factorization + recipe progression** (Log-mel vs MERT × DDCL vs ITGPT → multi-layer MERT + density + loss/reg + from-scratch `fine_subdiv` recipe) | `EXP-20260817-01`, `EXP-20260819-03`, `EXP-20261123-02/03`, `EXP-20261126-15`, `EXP-20261127-01/02`, `EXP-20261128-02` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.2 |
| T-ladder-scale | **`final_data` 4-point data scaling & recipe progression** (frozen 48 val charts): Log-mel DDCL → MERT-LSTM → ITGPT-MERT `48t → 197t → 294t → 1,713t` + `300t/50v` screen (`finedice003`, `silneg004`, `d_model=384`) | `EXP-20261119-02`–`06`, `EXP-20261123-01`, `EXP-20261124-01/02`, `EXP-20261127-01/04`, `EXP-20261128-03` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.3 |
| T-transfer-scaling | **Cross-corpus transfer, global vs no-Dice vs `fine_subdiv` soft-Dice ablation on Dataset A** (`1,713t` zero-shot → Arms C–L & soups) | `EXP-20261127-04`, `EXP-20261127-05`, `EXP-20261127-06`, `EXP-20261127-11`, `EXP-20261128-01` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.5 |
| T-error-audit | **Per-song Dataset A error audit** (`8/9` songs `0.819` vs `tsunamix_iii/isis` 31 fractional-BPM gimmicks) | `EXP-20261127-06` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.6 |
| T-raw-audio-beats | **End-to-end raw-audio beat tracking robustness** (`oracle` vs `beat_this_raw` → `beat_this_reg` → `beat_this_full` vs `librosa`, 0 `.sm` beats across Arm K, ZS `1,713t`, and Arm C) | `EXP-20261127-06`–`EXP-20261128-01` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.7 |
| T-timing-diag | Counterfactual slot masks proving cumulative `FP − FN` rank drift breaks `timing_match` while ±20 ms bipartite skill is strong | `EXP-20261010-01`, `EXP-20261118-01` | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.4 |
| F-pipeline | Raw audio → Beat This! beat/downbeat grid + beat-aligned MERT `(2,6,10,16)` → ITGPT placement → decoded ±20 ms + `M-grounding` + `.sm` export | [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | methods figure (to draw) |

Published numbers in T-related are citations, not our results. All tables above are backed by committed configs (`configs/ddc/*.json`, seed 42) and `EXP-` records.

---

## Changelog

| Date | Change |
| ---- | ------ |
| 2026-11-28 | Updated `C13`, `T-frax-2x2`, and `T-ladder-scale` with `EXP-20261128-02` (from-scratch Dataset A `fine_subdiv 0.03 + pw=1.00`: all-time from-scratch lowest `7,984` errors, `99.97%` exact density, `shuffle −0.1484`, `0.77655` F1@0.5 / `0.79664` `evF1`) and `EXP-20261128-03` (`Arm K` error breakdown: `intro 0.7382` vs `body 0.7957`, `10.3%/11.2%` `<=1`-slot jitter, `m=1 beginner` tail; plus 4-arm `300t/50v` screen showing global `dice=0.05, pw=1.25, d=256` keeps `0.84974`, `silneg=0.04` achieves **100%** silence FP suppression `35,462 → 0` FPs, `finedice003 pw=1.15` hits `0.84890 / 24th 0.5401`, and `d_model=384, L=8` trails at `0.84836` while lifting `intro` to `0.8200`). |
| 2026-11-28 | Updated `C14`, `C16`, `T-transfer-scaling`, and `T-raw-audio-beats` with `EXP-20261128-01` (subdivision-selective `fine_subdiv` soft-Dice on `16th_off` + `24th` only, zero-gated on sparse charts: single checkpoint `Arm K` shatters `0.790` on Dataset A at **`0.79043` F1@0.50 / `0.81303` `evF1` / `7,549` err / `shuffle −0.1466`** with zero slice tradeoff — `<=8 0.7402`, `8th 0.7964`, `16th_off 0.6706`, `24th 0.3889` — and hits a new raw-audio record **`0.8204` @ 20 ms = `101.0%` oracle** at `100.5%` exact density; 2-seed verified via `Arm L` `s=123`). |
| 2026-11-27 | Updated `foscarin2024beatthis`, `C14`, `C16`, `T-transfer-scaling`, and `T-raw-audio-beats` with `EXP-20261127-10` (octave-folded medley segmentation, Beat-0 clamp, and ITGPT downbeat bar-phase alignment lifting `BeatF1@70ms` to `0.9254` and raw-audio ±20 ms `evF1` to **101.4%** Arm C `0.8193` / **102.2%** ZS `1,713t` `0.8196` / **101.2%** `Soup_F_H_I` `0.8174` of oracle) and `EXP-20261127-11` (Dice-free `keep0 + rdrop=2.0` fine-tuning + zero-overhead `Soup_F_H_I` setting Dataset A records: `0.78726` F1@0.5 / `0.80846` `evF1` / `7,708` err; `Arm H` `7,688` err). |
| 2026-11-27 | Added `foscarin2024beatthis`, `park2019specaugment`, `liang2021rdrop`; added claims `C13`–`C16` and tables `T-transfer-scaling`, `T-error-audit`, `T-raw-audio-beats` covering full `final_data` `1,713t` scaling (`48v 0.86493`), zero-shot Dataset A (`0.77870` / `0.80337 evF1`), fine-tuned Dataset A (`0.78574` / `0.80834 evF1`; ensemble `0.80918`), `isis` fractional-BPM audit, and Beat This! raw-audio robustness (`95.2%` oracle ±50 ms `evF1`; `EXP-20261127-01`–`06`). |
| 2026-11-24 | Added `300t/50v` rung (`EXP-20261124-02`): `best.keras` **0.8276** F1@0.5 / **0.8324** max-F1 / **+0.7462** ±20 ms `skill_event_f1` / **44.7%** audio transfer, completing the 3-point scaling curve (`48t → 197t → 294t`). |
| 2026-11-24 | Promoted `li2024mert` to `in_paper`, added claims `C8`–`C12`, metrics `M-event-20ms` & `M-grounding`, and tables `T-frax-2x2`, `T-ladder-scale`, `T-timing-diag` from `EXP-20261010-01` through `EXP-20261124-01` (`0.8181` on `200t/50v`, Dataset A 2×2 **0.680 / 0.710 / 0.751 / 0.752**). |
| 2026-08-19 | Dataset A DDC Hamming-peak snap `M-slot48` F1@0.5 **0.364** vs null **0.397** (skill **−0.033**). EXP-20260819-04. |
| 2026-08-19 | Dataset A ITGPT 128-ep last F1@0.5 **0.710** vs DDCL **0.680** (`M-slot48`). EXP-20260819-03. |
| 2026-08-19 | Dataset B ITGPT 128-ep last F1@0.5 **0.735** / max-F1 **0.740** vs null **0.390**. Not vs Table 2 **0.78 / 0.80**. EXP-20260819-02. |
| 2026-08-15 | DDC best-val vs last: **0.650** / **0.735** ≈ last **0.652** / **0.734**. EXP-20260815-04. |
| 2026-08-15 | DDC 128-ep placement val F-score_c **0.652** / F-score_m **0.734** vs paper **0.681** / **0.756**. EXP-20260815-03. |
| 2026-08-15 | DDC 8-ep placement val F-score_c **0.594** / F-score_m **0.667** vs paper **0.681** / **0.756**. EXP-20260815-02. |
| 2026-08-15 | DDC placement PRE/POST/C-LSTM in `stepcovnet.ddc`; standard-only 450-chart index. Cite `schluter2014onset`, `hamel2012multiscale`. |
| 2026-08-15 | Dataset A ingested: 90 songs / 463 rows; Tsunamix III reconstructed after SM5 zip 404. EXP-20260815-01. |
