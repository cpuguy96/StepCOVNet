# Dataset prep golden fixtures (P6)

Self-contained raw pack bundles for `tests/dataset_prep/golden_fixtures_test.py`.
Each subdirectory is a **single-bundle** `--input-dir` for `preprocess_dataset.py`.

| Bundle | Role |
| ------ | ---- |
| `itl_challenge_ssc/` | ITL-like `.ssc`, one `dance-single` Challenge chart, `#OFFSET` stored in metadata |
| `vocaloid_multi_sm/` | Vocaloid-like `.sm`, Beginner + Challenge singles, `default_chart_index` → Challenge |
| `edge_nul_inferred/` | Reserved slug (`NUL` → `nul_dir`) and inferred audio (no `#MUSIC`) |

Audio files are tiny placeholders (not real OGG). Do not use for listening tests.
