# Experiment log

**Authoritative record** for runs and ablations. IDs: `EXP-YYYYMMDD-NN`. Each entry includes **Timestamp** (`YYYY-MM-DD HH:MM:SS`, local system time at write).

Promote selected findings to [PAPER_OUTLINE.md](PAPER_OUTLINE.md) only when drafting the paper — do not duplicate the full log there.

**Related:** [discussion notes](DISCUSSION_NOTES.md) · [pipeline architecture](PIPELINE_ARCHITECTURE.md) · [dataset prep plan](DATASET_PREP_PIPELINE.md)

---

## Current phase

**Updated:** 2026-06-22

### Dataset prep (PRE ingestion)

| Phase | Status |
| ----- | ------ |
| P0–P7, P6, P9, **P8** | **Done** — full three-bundle export; **1942** chart rows; loaders + `training_index.json` |

**Recommended next step:** Run `build_training_index.py` on `data/final_data` if needed, then first multi-song onset training with `data_dir=val_data_dir=data/final_data` (WSL GPU).

### Onset detection (research track)

| Item | Status |
| ---- | ------ |
| Dense val best | BiLSTM 256u — micro event F1 **0.686** @ thr=0.30 (EXP-20260610-03) |
| Event tide formulation | ~27–30% F1 plateau; oracle ~31% (EXP-20260606-11) |
| Multi-song val on `final_data` | **Blocked on P8** |

**Recommended when resuming onset work:** After P8, re-run dense/event val on `data/final_data` train/val split; or continue tide overfit / formulation probes on `data/v2` in parallel.

---

## Experiment index

Newest first. Stage tags: `pre` | `model` | `post` | `metric` | `train`. Discussion context: [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).

| ID | Stage tag | Question | Status | One-line outcome |
| -- | --------- | -------- | ------ | ---------------- |
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

### EXP-20260622-01: P9 final_data loader smoke

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-06-22 01:11:00 |
| **Track** | `pre` / dataset prep |
| **Config** | Local `data/final_data`; `training_loader.discover_training_rows`, `pairing.list_training_samples`, `create_onset_event_dataset_from_pairs` (1 sample) |
| **Outcome** | **1942** chart rows; ITL 246 / Mizuki 1310 / Vocaloid 386; **822** with `chart_index > 0`; 0 missing audio or `.chart.json`; all ≤2048 steps; TF batch builds with GT onsets when `max_audio_seconds` covers chart offset |
| **Conclusion** | P9 loaders ready for training; P8 split manifest is the remaining gate for proper val |

---

Older dense/event runs (EXP-20260606-* through EXP-20260610-*) are indexed above; add full per-run entries here when re-running or when promoting a result to the paper. Cross-links: `NOTE-…` in [DISCUSSION_NOTES.md](DISCUSSION_NOTES.md).
