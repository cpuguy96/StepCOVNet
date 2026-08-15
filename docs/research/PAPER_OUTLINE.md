# StepCOVNet onset detection — paper outline (draft)

**Status:** Paper draft skeleton — **not** an experiment log. Log runs in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); promote interesting findings here only when drafting methods/results for publication.

**Citations and paper tracking:** [PAPER_LEDGER.md](PAPER_LEDGER.md) · [paper.bib](paper.bib). Cite by key (e.g. `donahue2017ddc`). Prior art is cited; recreation on a published dataset is a baseline, not a substitute for a citation.

**Architecture reference:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) — PRE → MODEL → POST → METRICS → training feedback.

## Working title (TBD)

Event-based, dense, and autoregressive onset detection for rhythm-game chart generation from audio.

## Abstract (stub)

_TBD — write after selecting which results from [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) are paper-worthy._

- **Problem:** map audio → sparse list of onset times in seconds for downstream arrow/chart models.
- **Approach:** staged pipeline (PRE / MODEL / POST / METRICS); compare dense frame baseline, K-query event model, and planned AR seq2seq track.
- **Key claims:** _not drafted yet_

## 1. Introduction

- StepMania / StepCOVNet context; need for accurate onset times.
- Limitations of dense hop-grid targets vs K-query slots vs ordered token streams.
- Frame work as swappable pipeline stages ([PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)).

## 2. Related work

Keys from [PAPER_LEDGER.md](PAPER_LEDGER.md) / [paper.bib](paper.bib). Do not cite a paper here without a ledger row.

- **Chart generation (DDR/ITG):** placement then selection — `donahue2017ddc`; beat-grid ConvLSTM — `omalley2025ddcl`; hierarchical transformer ITGPT — `omalley2026itgpt`. Other games / variants: `yi2023goct`, `halina2021taikonation`, `lin2018generationmania`, `tsujino2018ddg`, `okeeffe2003dancingmonkeys`.
- **Musical onset detection:** `bello2005onset`, `eyben2010blstm`, `schluter2014onset` (DDC’s CNN placement baseline).
- **Features / BPM:** multi-scale spectrograms `hamel2012multiscale`; ArrowVortex BPM `vandewetering2016bpm`; SSL frontend `li2024mert` if used.
- **Set prediction / AR-on-times:** DETR-style slots `carion2020detr` only if the K-query track is in the paper. Distinguish literature AR (**arrows given times**) from StepCOVNet `onset_ar` (**times**).

## 3. Methods

See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for full stage definitions and ablation protocol.

### 3.1 Pipeline overview

- PRE: audio I/O + optional features (raw, cached mel, cached MERT).
- Dataset: `data/v2` (legacy `.txt`) vs `data/final_data` (nested `.chart.json` + `training_index.json` train/val).
- MODEL: core detector → raw outputs.
- POST: threshold, sort, min-gap → onset list (inference).
- METRICS: Hungarian match @ tolerance → F1; optional mingap metric path.
- Training feedback: loss + val metrics.

### 3.2 Dense onset baseline

- Cached mel and MERT; frame classifier (U-Net / BiLSTM / TCN); peak extraction + min distance.

### 3.3 Event onset model (`onset_events`)

- PRE: raw Conv1D, cached mel, or cached MERT → query head.
- MODEL: encoder → K query slots `(time, confidence)`; learnable time deltas on uniform grid.
- Training: Hungarian L1 assignment + time + confidence losses.

### 3.4 Metrics

- Event F1 @ tolerance (default 20 ms); TP/FP/FN after Hungarian matching.
- `event_onset_f1_mingap`: min-gap filter before match (50 ms default).
- AR track (planned): primary eval without min-gap; checkpoint on decoded event F1.

### 3.5 Autoregressive onset (`onset_ar`) — planned

_See [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md) — no results yet._

- PRE: cached MERT on 10 ms hop; patch P=8; frozen encoder weights for initial gates.
- MODEL: encoder–decoder; `delta_bucketed` token LM + pointer/residual alignment.
- Training: teacher forcing → scheduled sampling; gates on tide before multi-song val.
- POST: detokenize pointer times; no min-gap on primary metric.

## 4. Results

_TBD — pull numbers and figures from [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) when a result is selected for the paper. Do not mirror the full experiment index here._

## 5. Discussion

_TBD._

## 6. Conclusion

_TBD._

## Appendix

- Paper ledger (citations, claims, datasets, metrics): [PAPER_LEDGER.md](PAPER_LEDGER.md)
- BibTeX: [paper.bib](paper.bib)
- Pipeline: [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
- Experiment log (authoritative): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- Design reasoning: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md)
- AR formulation: [AR_ONSET_DESIGN.md](AR_ONSET_DESIGN.md)
- Decisions: [DECISIONS_CHECKLIST.md](DECISIONS_CHECKLIST.md)
- Config reference: `configs/overfit_tide/`, `configs/event/audio_baseline.json`
