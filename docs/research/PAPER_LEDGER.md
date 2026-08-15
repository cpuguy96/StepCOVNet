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
| `li2024mert` | Li et al. MERT. ICLR 2024. arXiv:2306.00107 | method | **likely** | Cite if MERT is a PRE ablation or a reported frontend. |
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
| C6 | Recreated DDC matches published Fraxtil placement/selection numbers | hypothesis | *no EXP yet* | Results |
| C7 | Recreated DDCL / ITGPT match published numbers on the matching corpus | hypothesis | *no EXP yet* | Results |

Do not put C6–C7 in the abstract until an `EXP-` supports them.

---

## 4. Datasets (methods)

| ID | Name | Packs | Size (as published) | Split | Paper use |
| -- | ---- | ----- | ------------------- | ----- | --------- |
| D-frax-orig | Original Fraxtil | Tsunamix III; Fraxtil’s Arrow Arrangements; Fraxtil’s Beast Beats | **Measured 2026-08-15:** 90 songs, **463** chart rows (**450** standard + **13** edit). Paper table: 90 / 450 (`donahue2017ddc`) | `stratified_song_v1` seed 42, val_fraction 0.1 → **81 / 9** songs. Standard-only: `data/literature_fraxtil_orig/training_index_standard.json` (**405 / 45** rows). Not DDC’s unpublished 80/10/10 IDs | Recreate DDC and DDCL. Ingest [EXP-20260815-01](EXPERIMENT_LOG.md#exp-20260815-01-original-fraxtil-dataset-a-ingested-90-songs); placement 8-ep [EXP-20260815-02](EXPERIMENT_LOG.md#exp-20260815-02-ddc-c-lstm-placement-8-ep-on-dataset-a--below-paper-above-null) |
| D-itg | In The Groove | ITG 1; ITG 2 | 133 songs, 652 charts | same rule | Multi-author DDC check |
| D-frax-exp | Expanded Fraxtil | D-frax-orig plus Cute Charts; Sweet Arrows and Hella Steps vols 1–4 | 253 songs, 952 charts (`omalley2026itgpt`) | same rule | Recreate ITGPT; freeze test songs for later ablations |
| D-final | StepCOVNet `final_data` | ITL Online 2026; Mizuki’s Simfiles; Vocaloid Project Pad Pack 4th | ~1010/110 songs train/val; ~1942 charts | `stratified_song_v1` seed 42 | Transfer / generalization only, until C5 is satisfied |

Record the **actual song IDs in each split** (file under `data/` or a tracked manifest) before quoting a number. If DDC’s unpublished split cannot be recovered, say so and use a documented seed-42 song-grouped split.

---

## 5. Metrics (methods)

| ID | Name | Definition | Used by | Notes |
| -- | ---- | ---------- | ------- | ----- |
| M-ddc-20ms | Peak-pick F1 @ ±20 ms | Hamming-smoothed 10 ms probs, per-difficulty threshold, ±20 ms match | `donahue2017ddc` | In-repo: `stepcovnet.ddc.peak_pick` / `scripts/eval_ddc_placement.py`. Do not mix with beat-grid F1 |
| M-slot48 | 48-slot/beat F1 | Binary vector of length 48 per beat; report @ 0.5 and max-F1 | `omalley2025ddcl`, `omalley2026itgpt` | No 20 ms window |
| M-sel-acc | Selection top-1 / top-2 / hold acc | Teacher-forced next-step over 256 classes | all three | Hold acc is the leftover error in ITGPT vs DDCL |
| M-timing | `timing_match` @ 20 ms | Ordered match / max(n_pred, n_ref) | current StepCOVNet AR/dense; DDC eval parallel column | Internal until a seconds-list track is in the paper. DDC literature column stays `M-ddc-20ms` ([EXP-20260815-05](EXPERIMENT_LOG.md#exp-20260815-05-accepted-ddc-peaks-have-no-ordered-timing_match-skill)) |
| M-null | Audio-blind / chance floor | Same metric with shuffled or silent audio, or density-matched null | EXP-20260804-03/05 | Required beside any number we claim as skill |

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
| MERT frontend on the frozen Fraxtil test set | `li2024mert` | Ablation | **Yes**, if it beats the cited baseline on the same split/metric |
| Hold-note modeling that beats DDCL hold acc | `omalley2025ddcl` as the number to beat | Ablation | **Yes**, if measured |
| Transfer to `final_data` | this ledger D-final | Experiment | **Yes**, as generalization, not as a DDC replacement table |

---

## 7. Planned tables / figures

| ID | Content | Source | Status |
| -- | ------- | ------ | ------ |
| T-related | Placement F1 and selection acc for DDC / DDCL / ITGPT as **published** | ITGPT Table 2 and 5; DDC Tables 2–3 | draft numbers in NOTE-20260814-01 |
| T-repro | Same metrics from **our** reimplementations | [EXP-20260815-03](EXPERIMENT_LOG.md#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap) · [EXP-20260815-04](EXPERIMENT_LOG.md#exp-20260815-04-best-val-ddc-weights-do-not-close-the-paper-gap) | 128-ep last **0.652** / **0.734**; best-val **0.650** / **0.735** vs paper **0.681** / **0.756**. Last≈best. **Accepted** as close enough |
| F-pipeline | PRE → placement → selection → POST → metrics | [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | methods figure |

Published numbers in T-related are citations, not our results. T-repro is the first results table that can support C6–C7.

---

## Changelog

| Date | Change |
| ---- | ------ |
| 2026-08-15 | DDC best-val vs last: **0.650** / **0.735** ≈ last **0.652** / **0.734**. EXP-20260815-04. |
| 2026-08-15 | DDC 128-ep placement val F-score_c **0.652** / F-score_m **0.734** vs paper **0.681** / **0.756**. EXP-20260815-03. |
| 2026-08-15 | DDC 8-ep placement val F-score_c **0.594** / F-score_m **0.667** vs paper **0.681** / **0.756**. EXP-20260815-02. |
| 2026-08-15 | DDC placement PRE/POST/C-LSTM in `stepcovnet.ddc`; standard-only 450-chart index. Cite `schluter2014onset`, `hamel2012multiscale`. |
| 2026-08-15 | Dataset A ingested: 90 songs / 463 rows; Tsunamix III reconstructed after SM5 zip 404. EXP-20260815-01. |
