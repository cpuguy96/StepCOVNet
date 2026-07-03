# Training data setup — download to `final_data`

**Purpose:** Recreate the multi-song **`data/final_data`** corpus used for dense / AR training (`training_index.json`, **1942** chart rows after prep).

**Related:** [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md) (prep design) · [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) § Current phase

All three raw bundles are hosted on **[StepMania Online](https://stepmaniaonline.net/)**. Each pack page lists song metadata and has **Download Pack** / **Mirror** buttons; the direct URLs below match those buttons.

---

## 1. Download raw packs (StepMania Online)

Extract each zip so the **top-level folder names** under `data/raw_data/` match the **Install folder** column (rename after extract if the zip root differs).

| Bundle | Songs (SMO) | Pack page (metadata + download buttons) | Direct download | Mirror |
| ------ | ----------- | --------------------------------------- | --------------- | ------ |
| **ITL Online 2026** | 310 | https://stepmaniaonline.net/pack/9671 | https://stepmaniaonline.net/download/pack/9671/ | https://stepmaniaonline.net/download/mirror/9671/ |
| **Mizuki's Simfiles** | 820 | https://stepmaniaonline.net/pack/7685 | https://stepmaniaonline.net/download/pack/7685/ | https://stepmaniaonline.net/download/mirror/7685/ |
| **Vocaloid Project Pad Pack 4th** | 76 | https://stepmaniaonline.net/pack/5162 | https://stepmaniaonline.net/download/pack/5162/ | https://stepmaniaonline.net/download/mirror/5162/ |

**Browse / search other packs:** https://stepmaniaonline.net/packs — use the **Pack Name** filter.

**Expected layout after extract:**

```text
data/raw_data/
  ITL Online 2026/          ← simfile + audio per song folder
  Mizuki's Simfiles/
  Vocaloid Project Pad Pack 4th/
```

Disk: allow **~5 GB** for the three zips plus extracted audio (ITL ~1.3 GB, Mizuki ~2.7 GB, Vocaloid ~1.6 GB on SMO as of 2026-07).

---

## 2. Install prep dependencies (Windows CPU)

From repo root:

```powershell
venv\Scripts\python.exe -m pip install -e ".[dataset-prep]"
```

---

## 3. Preprocess → `data/final_data`

Dry-run discovery first (no writes):

```powershell
venv\Scripts\python.exe scripts/preprocess_dataset.py `
  --input-dir data/raw_data `
  --output-dir data/final_data `
  --dry-run
```

Full prep (nested `.chart.json` + audio; exports all `dance-single` charts per pack):

```powershell
venv\Scripts\python.exe scripts/preprocess_dataset.py `
  --input-dir data/raw_data `
  --output-dir data/final_data `
  --workers 8
```

**Expected outcome:** ~**1942** chart rows across the three bundles (250 ITL singles + Mizuki/Vocaloid multi-chart exports; see [DATASET_PREP_PIPELINE.md](DATASET_PREP_PIPELINE.md) §3 for per-bundle notes). Reports: `data/final_data/preprocess_report.json`, `name_map.json`.

---

## 4. Build train/val manifest

```powershell
venv\Scripts\python.exe scripts/build_training_index.py `
  --output-dir data/final_data `
  --overwrite
```

**Expected outcome:** `data/final_data/training_index.json` — split policy `stratified_song_v1`, seed **42**, **1010 / 110** songs train/val, **1745 / 197** chart rows train/val.

Optional smaller subset for scoreboard smoke (50 train / 100 val chart rows):

```powershell
venv\Scripts\python.exe scripts/build_training_index_subset.py `
  --train-rows 50 --val-rows 100 --overwrite
```

---

## 5. Verify loaders (optional)

```powershell
venv\Scripts\python.exe -c "from stepcovnet import pairing; rows=pairing.list_training_samples('data/final_data/training_index.json', split='train'); print('train rows', len(rows))"
```

Dense training config entry point: `configs/dense/final_data_mert_bilstm.json` with `--training_index_path=data/final_data/training_index.json`.

---

## 6. MERT feature cache (before GPU training)

MERT is **not** part of prep; extract after `final_data` exists. From repo root (auto-dispatches to WSL when `--device=cuda`):

```powershell
venv\Scripts\python.exe scripts/extract_mert_features.py `
  --training_index_path=data/final_data/training_index.json `
  --beside_audio --device=cuda --skip_existing
```

Writes `{song}.mert.npy` beside each audio file under `data/final_data/`. Requires `pip install '.[ssl]'` in the WSL GPU venv.

**Scoreboard subset only:**

```powershell
venv\Scripts\python.exe scripts/extract_mert_features.py `
  --training_index_path=data/final_data/training_index_scoreboard_50t_100v.json `
  --beside_audio --device=cuda --skip_existing
```

Then train: `configs/dense/final_data_mert_bilstm_scoreboard_50t_100v.json` or full `final_data_mert_bilstm.json` — see [wsl-gpu-stepcovnet skill](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md).

---

## 7. Legacy `data/v2` (optional)

Single-song tide overfit and older dense configs use **`data/v2`**, not `final_data`. Repo-hosted zip:

https://drive.google.com/file/d/1YszVRR82hH3nRpp5zAeLrApjiWSxtxvD/view?usp=drive_link

Extract to `data/v2/` (`train/`, `val/`, `test/` with `tide.ogg` / `tide.txt` under `test/`).

---

## Quick checklist

| Step | Artifact |
| ---- | -------- |
| Download 3 SMO packs | `data/raw_data/{ITL Online 2026,Mizuki's Simfiles,Vocaloid Project Pad Pack 4th}/` |
| `preprocess_dataset.py` | `data/final_data/{bundle}/{id}/` + reports |
| `build_training_index.py` | `data/final_data/training_index.json` |
| `extract_mert_features.py` | `*.mert.npy` beside audio |
| Train | `--training_index_path=data/final_data/training_index.json` |
