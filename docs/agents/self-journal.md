# Agent self-journal

Iterative improvements to agent behavior, process, and conventions on this project. **Not** a substitute for [EXPERIMENT_LOG.md](../research/EXPERIMENT_LOG.md) or [DISCUSSION_NOTES.md](../research/DISCUSSION_NOTES.md) — those hold research findings; this holds **how we work better**.

**Order:** newest entries first (prepend below).

## When to prepend

After substantive sessions when you discover:

- A mistake that wasted time or produced wrong conclusions
- A fix or workaround that should become default
- A user-established convention worth preserving
- A repeated workflow missing from `.cursor/skills/`

Maintenance rule: prepend in the same session when possible; link `EXP-…` / `NOTE-…` when relevant. See [agent-self-improvement skill](../../.cursor/skills/agent-self-improvement/SKILL.md).

## Entry format

Insert **at the top** of [Entries](#entries) (below this section):

```markdown
### JRN-YYYYMMDD-NN: Short title

| Field            | Value                                            |
| ---------------- | ------------------------------------------------ |
| **Timestamp**    | YYYY-MM-DD HH:MM:SS (system clock at write time) |
| **Category**     | mistake \| fix \| convention \| skill-gap        |
| **Summary**      | What happened                                    |
| **Action taken** | Code, doc, skill, or rule change                 |
| **Related**      | EXP-…, NOTE-…, skill name                        |
```

---

## Entries

### JRN-20260606-09: Dense seed after model init caused train lottery

| Field            | Value                                                                                                                                                              |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Timestamp**    | 2026-06-06 16:54:25                                                                                                                                                |
| **Category**     | mistake                                                                                                                                                            |
| **Summary**      | `tf.random.set_seed` ran in `_fit_and_save_model` after `build_unet_wavenet_model`, so seed 42 in config did not control initialization — EXP-12 vs EXP-13/14 gap. |
| **Action taken** | Added `reproducibility.apply_training_seed()` before model build; repro gate script; EXP-15 logged.                                                                |
| **Related**      | EXP-15, NOTE-20260606-16, `check_dense_mert_reproducibility.py`                                                                                                    |

### JRN-20260606-08: Timestamps from system clock only

| Field            | Value                                                                                                                                                 |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 16:37:35                                                                                                                                   |
| **Category**     | convention                                                                                                                                            |
| **Summary**      | User required full `YYYY-MM-DD HH:MM:SS` on all log timestamps, captured from the machine clock at write time — no `(approx.)` or estimated suffixes. |
| **Action taken** | Updated `research-notebook.mdc`, skills, templates, and agent docs; stripped `(approx.)` from existing EXP/NOTE/JRN rows.                             |
| **Related**      | NOTE-20260606-15, `research-session-workflow` skill                                                                                                   |

### JRN-20260606-07: Newest-first research logs

| Field            | Value                                                                                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 18:00:00                                                                                                        |
| **Category**     | convention                                                                                                                 |
| **Summary**      | EXP/NOTE/JRN entries and index tables were append-only (oldest at top), forcing scroll to see latest runs.                 |
| **Action taken** | Reordered logs newest-first; updated `research-notebook.mdc`, templates, and skills to **prepend** entries and index rows. |
| **Related**      | `research-session-workflow` skill, NOTE-20260606-09                                                                        |

### JRN-20260606-06: Skills index without SKILL.md files

| Field            | Value                                                                                                                                                             |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 16:45:00                                                                                                                                               |
| **Category**     | skill-gap                                                                                                                                                         |
| **Summary**      | Initial agent-self-improvement pass created `skills-index.md` and routing updates but did not write `.cursor/skills/*/SKILL.md` bodies — index links were broken. |
| **Action taken** | Created all six project skills under `.cursor/skills/`; verify [skills README](../../.cursor/skills/README.md) after any index change.                             |
| **Related**      | `agent-self-improvement` skill, [skills README](../../.cursor/skills/README.md)                                                                                   |

### JRN-20260606-05: Ablation threshold sweep None min_gap TypeError

| Field            | Value                                                                                                                                                            |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 15:00:00                                                                                                                                              |
| **Category**     | fix                                                                                                                                                              |
| **Summary**      | `sweep_confidence_thresholds` passed `min_onset_distance_ms=None` into code expecting a float → `TypeError` during ablation threshold phase.                     |
| **Action taken** | Fixed in `diagnostics.py` (coerce None → 0.0); partial `ablation_summary.json` saved after arch_large OOM. Documented EXP-10 outcomes in `tide-ablations` skill. |
| **Related**      | EXP-10, `tide-ablations` skill                                                                                                                                   |

### JRN-20260606-04: F1=0 without running diagnostics first

| Field            | Value                                                                                                                                                                                         |
| ---------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 14:30:00                                                                                                                                                                           |
| **Category**     | convention                                                                                                                                                                                    |
| **Summary**      | When conv1d showed 0% F1, the first instinct was to change epochs or architecture. Diagnostics (`debug_onset_overfit.py`, confidence stats) revealed assignment collapse, not a broken model. |
| **Action taken** | `onset-event-eval-matching` skill: **run diagnostics before retrain** when F1=0 or suspiciously low.                                                                                          |
| **Related**      | NOTE-20260606-13, `scripts/debug_onset_overfit.py`, `diagnostics.py`                                                                                                                          |

### JRN-20260606-03: WSL GPU env not sourced — silent CPU fallback

| Field            | Value                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 13:00:00                                                                                                                                        |
| **Category**     | mistake                                                                                                                                                    |
| **Summary**      | TensorFlow training launched in WSL without `source scripts/wsl_gpu_env.sh`. TF saw zero GPUs and trained on CPU with no obvious error — long runs wasted. |
| **Action taken** | Documented mandatory `wsl_gpu_env.sh` in `wsl-gpu.mdc` and `wsl-gpu-stepcovnet` skill; all WSL command templates include `source` before Python.           |
| **Related**      | `wsl-gpu-stepcovnet` skill, `wsl-gpu.mdc`                                                                                                                  |

### JRN-20260606-02: Ordered training vs Hungarian eval collapsed conv1d F1

| Field            | Value                                                                                                                                                                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 14:12:41                                                                                                                                                                                                                                        |
| **Category**     | mistake                                                                                                                                                                                                                                                    |
| **Summary**      | Training used ordered slot→GT pairing while eval used Hungarian matching. On tide (634 GT, 1024 uniform slots), zero ordered pairs fell within 20 ms tolerance → loss pushed all confidences toward 0 → **0% F1** on conv1d despite reasonable pred times. |
| **Action taken** | Switched training to `assign_onset_pairs_l1` (Hungarian L1) in `losses.py`; logged EXP-08; created `onset-event-eval-matching` skill with diagnostics-first playbook.                                                                                      |
| **Related**      | EXP-07, EXP-08, NOTE-20260606-13                                                                                                                                                                                                                           |

### JRN-20260606-01: Wrong calendar dates in research logs

| Field            | Value                                                                                                                                                   |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-06 10:30:00                                                                                                                                     |
| **Category**     | mistake                                                                                                                                                 |
| **Summary**      | Early EXP/NOTE entries used placeholder or incorrect dates (e.g. 2025) instead of the session calendar day. Broke ID consistency and paper cross-links. |
| **Action taken** | Renumbered IDs to `EXP-20260606-*` / `NOTE-20260606-*`; required full `YYYY-MM-DD HH:MM:SS` timestamps in `research-notebook.mdc`.                      |
| **Related**      | `research-session-workflow` skill, `research-notebook.mdc`                                                                                              |
