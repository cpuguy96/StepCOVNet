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
| `li2024mert` | Li et al. MERT. ICLR 2024. arXiv:2306.00107 | method | **in_paper** | Beat-aligned 100 Hz layer-6 `MERT-v1-330M` frontend (`build_mert_placement_model`, `ItgptBeatMertEncoder`) replacing scratch log-mel in the Dataset A 2×2 factorization and `final_data` scaling ladder (`EXP-20261119-04`, `EXP-20261123-01/02/03`, `EXP-20261124-01`). |
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
| C11 | **Multi-pack transfer & 3-point scaling (`final_data`, frozen 48 val charts):** MERT triples train→val audio transfer (**11.6% → 33.6%**), full-song hierarchical attention (`ITGPT-MERT`) beats BiLSTM (**0.7719 / −0.0848** vs **0.7647 / −0.0782**), and scaling `48t → 197t → 294t` monotonically lifts val F1@0.5 **0.7719 → 0.8181 → 0.8276** (max-F1 **0.7728 → 0.8192 → 0.8324**, ±20 ms `skill_event_f1` **+0.6695 → +0.7330 → +0.7462**, `last` `shuffle` drop **−0.0848 → −0.1103 → −0.1189**, audio transfer **32.0% → 42.3% → 44.7%**) with the `shuffle` prior floor staying flat (~`0.70–0.71`) | supported | [EXP-20261119-04](EXPERIMENT_LOG.md#exp-20261119-04-beat-aligned-mert-lifts-ladder-48-slot-f1-to-0765-and-triples-val-audio-grounding-at-9x-speed), [EXP-20261119-05](EXPERIMENT_LOG.md#exp-20261119-05-mert-train-grounding-shows-29x-better-audio-transfer-than-log-mel-with-66-overfit-headroom-left), [EXP-20261123-01](EXPERIMENT_LOG.md#exp-20261123-01-beat-aligned-itgpt-mert-beats-mert-lstm-on-both-val-f1-0772-and-shuffle-grounding-0085), [EXP-20261124-01](EXPERIMENT_LOG.md#exp-20261124-01-scaling-itgpt-mert-to-200t50v-shatters-080-with-0818-val-f1-and-423-audio-transfer), [EXP-20261124-02](EXPERIMENT_LOG.md#exp-20261124-02-300t50v-completes-3-point-itgpt-mert-scaling-curve-at-0828-f1--0832-max-f1-and-447-audio-transfer) | Abstract, Results (`T-ladder-scale`), Discussion |
| C12 | Heavy audio-path regularization (`AdamW wd=0.01`, `dropout=0.4`, `noise=0.05`) on MERT-LSTM hurts both val F1 (**0.7647 → 0.7612**) and audio grounding (`shuffle` drop **−0.0782 → −0.0533**) by pushing weight back onto `[meter, bpm]`, whereas light `AdamW(wd=0.01)` + `dropout=0.1` without audio noise stabilizes ITGPT-MERT cleanly | supported (negative for heavy reg) | [EXP-20261119-06](EXPERIMENT_LOG.md#exp-20261119-06-heavy-regularization-hurts-both-mert-val-f1-and-audio-grounding-by-pushing-back-onto-meter-bpm), [EXP-20261123-02](EXPERIMENT_LOG.md#exp-20261123-02-itgpt-mert-single-knob-swap-lifts-dataset-a-fraxtil-42-pp-over-paper-faithful-itgpt-0752-vs-0710), [EXP-20261124-01](EXPERIMENT_LOG.md#exp-20261124-01-scaling-itgpt-mert-to-200t50v-shatters-080-with-0818-val-f1-and-423-audio-transfer) | Discussion (`T-grounding`) |

Do not promote a claim to the abstract until an `EXP-` supports it. Claims C8–C12 are now supported and ready for [PAPER_OUTLINE.md](PAPER_OUTLINE.md).

---

## 4. Datasets (methods)

| ID | Name | Packs | Size (as published) | Split | Paper use |
| -- | ---- | ----- | ------------------- | ----- | --------- |
| D-frax-orig | Original Fraxtil | Tsunamix III; Fraxtil’s Arrow Arrangements; Fraxtil’s Beast Beats | **Measured 2026-08-15:** 90 songs, **463** chart rows (**450** standard + **13** edit). Paper table: 90 / 450 (`donahue2017ddc`) | `stratified_song_v1` seed 42, val_fraction 0.1 → **81 / 9** songs. Standard-only: `data/literature_fraxtil_orig/training_index_standard.json` (**405 / 45** rows, **17,523** val beats, **17,868** val GT steps). Not DDC’s unpublished 80/10/10 IDs | Literature 2×2 benchmark (DDC, DDCL, ITGPT, DDCL-MERT, ITGPT-MERT). Ingest [EXP-20260815-01](EXPERIMENT_LOG.md#exp-20260815-01-original-fraxtil-dataset-a-ingested-90-songs); DDC [EXP-20260815-03](EXPERIMENT_LOG.md#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap); DDCL [EXP-20260817-01](EXPERIMENT_LOG.md#exp-20260817-01-ddcl-48-slot-placement-128-ep-on-dataset-a); ITGPT [EXP-20260819-03](EXPERIMENT_LOG.md#exp-20260819-03-itgpt-48-slot-placement-128-ep-on-dataset-a); ITGPT-MERT [EXP-20261123-02](EXPERIMENT_LOG.md#exp-20261123-02-itgpt-mert-single-knob-swap-lifts-dataset-a-fraxtil-42-pp-over-paper-faithful-itgpt-0752-vs-0710); DDCL-MERT [EXP-20261123-03](EXPERIMENT_LOG.md#exp-20261123-03-ddcl-mert-completes-dataset-a-2x2-matrix--71-pp-and-73x-audio-grounding-over-log-mel-ddcl) |
| D-itg | In The Groove | ITG 1; ITG 2 | 133 songs, 652 charts | same rule | Multi-author DDC check (not yet ingested) |
| D-frax-exp | Expanded Fraxtil | D-frax-orig plus Cute Charts; Sweet Arrows and Hella Steps vols 1–3 (vol 4 **404** on fra.xtil.net) | Paper: 253 songs / 952 charts (`omalley2026itgpt`). **Measured 2026-08-18:** 232 raw folders, **222** exported songs / **747** chart rows (**722** standard + **25** edit). Ten SAHS songs skipped | `stratified_song_v1` seed 42, val_fraction 0.1 → **201 / 21** songs. Standard-only: `data/literature_fraxtil_exp/training_index_standard.json` (**653 / 69** rows). Not ITGPT’s unpublished split | Recreate ITGPT placement (held; vol 4 missing). Ingest [EXP-20260818-01](EXPERIMENT_LOG.md#exp-20260818-01-expanded-fraxtil-dataset-b-ingested); 8-ep [EXP-20260819-01](EXPERIMENT_LOG.md#exp-20260819-01-itgpt-paper-sized-placement-oom-then-mixed-fp16-8-ep-on-dataset-b); 128-ep [EXP-20260819-02](EXPERIMENT_LOG.md#exp-20260819-02-itgpt-48-slot-placement-128-ep-on-dataset-b) |
| D-final | StepCOVNet `final_data` | ITL Online 2026; Mizuki’s Simfiles; Vocaloid Project Pad Pack 4th | Full index: ~1010/110 songs (~1942 rows). Frozen-val ladder manifests: `50t_50v` (**48 / 48** valid charts), `200t_50v` (**197 / 48** valid charts, **187 / 40** songs), and `300t_50v` (**294 / 48** valid charts, **277 / 40** songs), excluding 6 negative-BPM Vocaloid songs (`hold_release_rakshasa_and_carcasses`, `pantomime`, `the_rampage_of_hatsune_miku`, `vrosteroid`, `news_39`, `viva_happy`; **17,057** val beats, **37,617** val GT steps) | `stratified_song_v1` seed 42, frozen 48-chart val across ladder rungs | Multi-pack transfer, grounding diagnostics, and 3-point data-scaling ladder ([EXP-20261118-01](EXPERIMENT_LOG.md#exp-20261118-01-counterfactual-slot-masks-prove-rank-drift-breaks-timing_match-while-decoded-slot-times-beat-dense-tcn) through [EXP-20261124-02](EXPERIMENT_LOG.md#exp-20261124-02-300t50v-completes-3-point-itgpt-mert-scaling-curve-at-0828-f1--0832-max-f1-and-447-audio-transfer)). Never mixed with Dataset A in one table |

Record the **actual song IDs in each split** (file under `data/` or a tracked manifest) before quoting a number. If DDC’s unpublished split cannot be recovered, say so and use a documented seed-42 song-grouped split.

---

## 5. Metrics (methods)

| ID | Name | Definition | Used by | Notes |
| -- | ---- | ---------- | ------- | ----- |
| M-ddc-20ms | Peak-pick F1 @ ±20 ms | Hamming-smoothed 10 ms probs, per-difficulty threshold, ±20 ms match | `donahue2017ddc` | In-repo: `stepcovnet.ddc.peak_pick` / `scripts/eval_ddc_placement.py`. Do not mix with beat-grid F1 |
| M-slot48 | 48-slot/beat F1 | Binary vector of length 48 per beat; report @ 0.5 and max-F1 | `omalley2025ddcl`, `omalley2026itgpt` | In-repo: `stepcovnet.ddcl.evaluation` / `scripts/eval_ddcl_placement.py` / `scripts/eval_itgpt_placement.py`. DDC POST snap: `scripts/eval_ddc_placement.py --slot48` ([EXP-20260819-04](EXPERIMENT_LOG.md#exp-20260819-04-ddc-hamming-peak-snap-onto-m-slot48-on-dataset-a)). No 20 ms window |
| M-sel-acc | Selection top-1 / top-2 / hold acc | Teacher-forced next-step over 256 classes | all three | Hold acc is the leftover error in ITGPT vs DDCL |
| M-timing | `timing_match` @ 20 ms | Ordered match / max(n_pred, n_ref) | StepCOVNet diagnostic | Rank-indexed; proved fragile to cumulative `FP − FN` rank drift ([EXP-20261118-01](EXPERIMENT_LOG.md#exp-20261118-01-counterfactual-slot-masks-prove-rank-drift-breaks-timing_match-while-decoded-slot-times-beat-dense-tcn)). DDC literature column stays `M-ddc-20ms` ([EXP-20260815-05](EXPERIMENT_LOG.md#exp-20260815-05-accepted-ddc-peaks-have-no-ordered-timing_match-skill)) |
| M-event-20ms | Decoded ±20 ms bipartite event F1 & `skill_event_f1` | Decode occupied `M-slot48` centers to seconds via `#OFFSET`/`#BPMS`, score Hungarian ±20 ms F1 and headroom-normalized skill `(F1 − F1_null) / (1 − F1_null)` over strongest audio-blind null (`ioi_shuffle` / `regular_grid` / `uniform_duration`) | StepCOVNet placement eval (`stepcovnet.ddcl.evaluation`, `stepcovnet.onset_null_baseline`) | Bridges beat-grid `M-slot48` and continuous-time ±20 ms onset evaluation without rank-drift collapse |
| M-grounding | Audio-corruption grounding & transfer ratio | Evaluate `none`, beat-permuted `shuffle`, and zeroed `zeros` on `beat_audio` with metadata/position intact; report `slot_f1_drop_shuffle`, `event_skill_drop_shuffle`, `1 − IoU`, and train→val audio transfer `val_drop_shuffle / train_drop_shuffle` | StepCOVNet (`--diagnose_grounding` in `eval_ddcl_placement.py` and `eval_itgpt_placement.py`) | Required beside `M-slot48` to distinguish genuine audio grounding from metadata/grid templates (`EXP-20261119-01`–`EXP-20261124-01`) |
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
| Audio-corruption grounding & transfer-ratio protocol (`M-grounding`) | ours (`stepcovnet.ddcl.evaluation`, `stepcovnet.itgpt.evaluation`) | Applied across all 4 Dataset A cells + `final_data` ladder | **Yes — measured** (exposes early-ckpt audio blindness & log-mel DDCL's 6.6× grounding gap) |
| Hold-note modeling that beats DDCL hold acc | `omalley2025ddcl` as the number to beat | Deferred (Stage 2 selection) | **Yes**, if measured |
| Multi-pack transfer & 3-point scaling ladder on `final_data` (`48t → 197t → 294t`) | this ledger `D-final` (`configs/ddc/itgpt_mert_placement_ladder_{50t,200t,300t}_50v.json`) | Experiment (frozen 48 val charts) | **Yes — measured** (generalization & scaling **0.731 → 0.828 / 0.832 max-F1**, transfer **11.6% → 44.7%**; `EXP-20261124-01/02`) |

---

## 7. Planned tables / figures

| ID | Content | Source | Status |
| -- | ------- | ------ | ------ |
| T-related | Placement F1 and selection acc for DDC / DDCL / ITGPT as **published** | ITGPT Table 2 and 5; DDC Tables 2–3 | draft numbers in NOTE-20260814-01 |
| T-repro | Our Dataset A baseline reimplementations (`M-ddc-20ms` for DDC; `M-slot48` for DDCL / ITGPT / DDC peak-snap) | [EXP-20260815-03](EXPERIMENT_LOG.md#exp-20260815-03-ddc-128-ep-placement-closes-most-of-the-paper-gap), [EXP-20260817-01](EXPERIMENT_LOG.md#exp-20260817-01-ddcl-48-slot-placement-128-ep-on-dataset-a), [EXP-20260819-03](EXPERIMENT_LOG.md#exp-20260819-03-itgpt-48-slot-placement-128-ep-on-dataset-a), [EXP-20260819-04](EXPERIMENT_LOG.md#exp-20260819-04-ddc-hamming-peak-snap-onto-m-slot48-on-dataset-a) | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.1 |
| T-frax-2x2 | **Dataset A 2×2 factorization** (Log-mel vs MERT × DDCL vs ITGPT): `M-slot48`, ±20 ms `skill_event_f1`, `shuffle`/`zeros` grounding | [EXP-20260817-01](EXPERIMENT_LOG.md#exp-20260817-01-ddcl-48-slot-placement-128-ep-on-dataset-a), [EXP-20260819-03](EXPERIMENT_LOG.md#exp-20260819-03-itgpt-48-slot-placement-128-ep-on-dataset-a), [EXP-20261123-02](EXPERIMENT_LOG.md#exp-20261123-02-itgpt-mert-single-knob-swap-lifts-dataset-a-fraxtil-42-pp-over-paper-faithful-itgpt-0752-vs-0710), [EXP-20261123-03](EXPERIMENT_LOG.md#exp-20261123-03-ddcl-mert-completes-dataset-a-2x2-matrix--71-pp-and-73x-audio-grounding-over-log-mel-ddcl) | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.2 |
| T-ladder-scale | **`final_data` multi-pack progression & 3-point data scaling** (frozen 48 val charts): Log-mel DDCL → MERT-LSTM → ITGPT-MERT `48t → 197t → 294t` (`M-slot48`, ±20 ms skill, `shuffle` drop, train→val audio transfer) | [EXP-20261119-02/03/04/05/06](EXPERIMENT_LOG.md#exp-20261119-04-beat-aligned-mert-lifts-ladder-48-slot-f1-to-0765-and-triples-val-audio-grounding-at-9x-speed), [EXP-20261123-01](EXPERIMENT_LOG.md#exp-20261123-01-beat-aligned-itgpt-mert-beats-mert-lstm-on-both-val-f1-0772-and-shuffle-grounding-0085), [EXP-20261124-01](EXPERIMENT_LOG.md#exp-20261124-01-scaling-itgpt-mert-to-200t50v-shatters-080-with-0818-val-f1-and-423-audio-transfer), [EXP-20261124-02](EXPERIMENT_LOG.md#exp-20261124-02-300t50v-completes-3-point-itgpt-mert-scaling-curve-at-0828-f1--0832-max-f1-and-447-audio-transfer) | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.3 |
| T-timing-diag | Counterfactual slot masks proving cumulative `FP − FN` rank drift breaks `timing_match` while ±20 ms bipartite skill is strong | [EXP-20261010-01](EXPERIMENT_LOG.md#exp-20261010-01-48-slot-oracle-clears-timing_match-on-ladder-val), [EXP-20261118-01](EXPERIMENT_LOG.md#exp-20261118-01-counterfactual-slot-masks-prove-rank-drift-breaks-timing_match-while-decoded-slot-times-beat-dense-tcn) | **Ready** — promoted to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) § 4.4 |
| F-pipeline | PRE (log-mel vs beat-aligned MERT) → placement (DDCL vs ITGPT) → decoded ±20 ms + `M-grounding` | [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) | methods figure (to draw) |

Published numbers in T-related are citations, not our results. T-repro, T-frax-2x2, T-ladder-scale, and T-timing-diag are backed by committed configs (`configs/ddc/*.json`, seed 42) and `EXP-` records.

---

## Changelog

| Date | Change |
| ---- | ------ |
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
