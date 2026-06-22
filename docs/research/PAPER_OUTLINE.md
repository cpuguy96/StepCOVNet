# StepCOVNet onset detection — paper outline (draft)

**Status:** Paper draft skeleton — **not** an experiment log. Log runs in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md); promote interesting findings here only when drafting methods/results for publication.

**Architecture reference:** [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) — PRE → MODEL → POST → METRICS → training feedback.

## Working title (TBD)

Event-based vs dense onset detection for rhythm-game chart generation from audio.

## Abstract (stub)

_TBD — write after selecting which results from [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) are paper-worthy._

- **Problem:** map audio → sparse list of onset times in seconds for downstream arrow/chart models.
- **Approach:** staged pipeline (PRE / MODEL / POST / METRICS); compare dense frame baseline vs K-query event model.
- **Key claims:** _not drafted yet_

## 1. Introduction

- StepMania / StepCOVNet context; need for accurate onset times.
- Limitations of dense hop-grid targets vs direct event lists.
- Frame work as swappable pipeline stages ([PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)).

## 2. Related work

- Frame-wise onset detection (mel + CNN/RNN).
- Set prediction / DETR-style event detection.
- SSL audio features (MERT) for music.

## 3. Methods

See [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md) for full stage definitions and ablation protocol.

### 3.1 Pipeline overview

- PRE: audio I/O + optional features (raw, cached mel, cached MERT).
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

## 4. Results

_TBD — pull numbers and figures from [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) when a result is selected for the paper. Do not mirror the full experiment index here._

## 5. Discussion

_TBD._

## 6. Conclusion

_TBD._

## Appendix

- Pipeline: [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)
- Experiment log (authoritative): [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md)
- Design reasoning: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md)
- Config reference: `configs/overfit_tide/`, `configs/onset_event_audio_baseline.json`
