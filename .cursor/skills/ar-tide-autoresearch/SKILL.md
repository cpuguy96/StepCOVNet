---
name: ar-tide-autoresearch
description: >-
  Agent-driven AR tide scratch overfit loop — meaningful hypotheses, not knob spam.
  Use for overnight 634/634 free-run research, autoresearch sessions, session_brief
  planning, next_experiment.json, or when overnight_planner plateaued on teacher match.
disable-model-invocation: true
---

# AR tide autoresearch

Autonomous **agent** loop for single-chart tide AR overfit. The harness trains and evals; **you** choose hypotheses from evidence. Do **not** use `overnight_planner.py` for meaningful research — it only mutates JSON values seen in history.

North star: [PIPELINE_ARCHITECTURE.md](../../../docs/research/PIPELINE_ARCHITECTURE.md).  
Full handoff: [ar-tide-overnight-prompt.md](../../../docs/agents/ar-tide-overnight-prompt.md).  
GPU: [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md).

## Goal and gates

| Phase                              | Primary metric                         | Bar                              |
| ---------------------------------- | -------------------------------------- | -------------------------------- |
| **Memorize** (scratch)             | Teacher `ordered_onset_match` @ 20 ms  | **634/634** before decode tuning |
| **Decode** (after teacher perfect) | Free-run `ordered_onset_match` @ 20 ms | **634/634**                      |

**Fixed eval policy** (do not change between runs): teacher gate before `--ar_decode`, checkpoint `val_overfit_gate`, offline eval only. See [ONSET_METRICS.md](../../../docs/research/ONSET_METRICS.md).

**Scratch-only:** `init_model_path` is stripped every run. Warm-start 614/634 results are **not** valid pass bars.

## When to use this skill

- Overnight or multi-hour **agent** session on tide AR iter
- `overnight_planner` / `--hours` plateaued (same teacher score ± few matches for ≥5 runs)
- User asks for **meaningful** experiments: code, architecture, loss, or strategy — not λ/lr tweakers
- Adapting Karpathy / cursor-autoresearch pattern to this repo (agent edits + measure + keep/revert)

## Change tiers (pick the **lowest tier that tests the hypothesis**)

| Tier               | What you may change                                             | Before GPU                                        |
| ------------------ | --------------------------------------------------------------- | ------------------------------------------------- |
| **A — Strategy**   | `run` / `model` / `dataset` overrides in `next_experiment.json` | None (config-only)                                |
| **B — Model/data** | `d_model`, layers, `patch_frames`, `max_steps_per_chart`, etc.  | Brief notes why arch change is justified          |
| **C — Code**       | `src/`, training scripts (not iter JSON logs)                   | `pytest tests/onset_ar/` + `pre_submit.py --fast` |

**Tier A is not “tweak one knob.”** A valid Tier A plan states _what failed_, _what you expect_, and _which keys differ from the parent recipe_ — usually several coordinated overrides (e.g. return to champion memorization: `lr=5e-5`, `scheduled_sampling_max_p=0`, `perfect_overfit_early_stop=false`).

**Tier C** is appropriate when val/train metrics disagree, decode drift is suspected, or scratch cannot memorize in 200 ep with any recipe class.

## Anti-spam rules (mandatory)

Before writing `next_experiment.json`, read:

```text
venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
```

1. **Plateau:** If scratch teacher best unchanged for **≥5 runs** (same ordered match ±10), **forbid** single-key `learning_rate` / `lambda_incremental_consistency` / `lambda_residual` / `eos_token_weight_scale` mutations. Require **tier upgrade** (new strategy class, model block, or Tier C code).
2. **Memorization first:** If scratch teacher **&lt; 0.95**, do **not** add scheduled sampling, in-loop AR decode, or decode checkpoint metrics. Focus on memorization recipe.
3. **No repeats:** Check `tried_recipes` in the brief. Do not rerun the same fingerprint unless diagnosing infra failure.
4. **Pinned keys:** Do not change `epochs` (200), `checkpoint_metric` (`val_overfit_gate`), or `tolerance_sec` (0.02) unless the champion template or design doc changes.
5. **One GPU job:** Never start a second `run_exp` / `run_overnight` shell. See `training_lock.py`.

## Session phase detection

Use brief + [AR_TIDE_OVERFIT_ITER_LOG.md](../../../docs/research/AR_TIDE_OVERFIT_ITER_LOG.md):

| Signal                                          | Phase           | Agent focus                                                                            |
| ----------------------------------------------- | --------------- | -------------------------------------------------------------------------------------- |
| Scratch teacher &lt; 634/634                    | **Memorize**    | lr schedule, CE length norm, no SS, enough epochs, loss weights — or Tier B/C if stuck |
| Teacher 634/634, free-run &lt; 634              | **Decode**      | mild SS, `lambda_incremental_consistency`, residual path — not warm-start              |
| `val_ordered_onset_match` flat &lt;0.5 @ ~30 ep | **Kill recipe** | pivot early; do not burn 200 ep                                                        |

Known low-yield (need new angle, not repeats): heavy SS `p≥0.4`, `use_soft_pointer_time`, iter17 warm-start polish alone.

## Autoresearch loop (agent)

Repeat until time budget or **634/634** free-run.

```text
venv\Scripts\python.exe scripts/ar_tide_iter/session_brief.py
```

1. **Orient** — session best, scratch-era teacher best, `last_run_vs_session_best`, recent `config_changes_vs_previous`, tried recipes.
2. **Diagnose** — one sentence: memorization vs decode vs bug. Cite numbers from brief/log.
3. **Hypothesize** — tier A/B/C; what outcome would confirm/refute.
4. **Plan** — write `logs/ar_tide_iter/next_experiment.json`:

```json
{
  "id": "iterNNN",
  "notes": "Short hypothesis (one line)",
  "reasoning": "Evidence from brief + log; why this tier; expected metric movement",
  "run": {}
}
```

Partial overrides only (`run` / `model` / `dataset`). See [next_experiment.example.json](../../../scripts/ar_tide_iter/next_experiment.example.json).

5. **Verify code** (Tier C only) — `venv\Scripts\python.exe -m pytest tests/onset_ar/ -q --tb=short` then `venv\Scripts\python.exe pre_submit.py --fast`.
6. **Run one experiment:**

```text
venv\Scripts\python.exe scripts/ar_tide_iter/run_overnight.py --once
```

7. **Log** — append [AR_TIDE_OVERFIT_ITER_LOG.md](../../../docs/research/AR_TIDE_OVERFIT_ITER_LOG.md). Code fixes or graduation → [EXPERIMENT_LOG.md](../../../docs/research/EXPERIMENT_LOG.md) per [research-session-workflow](../research-session-workflow/SKILL.md).
8. **Judge** — keep direction (next variant), pivot (new tier), or revert Tier C via git if hypothesis failed and code is suspect.

**Watch training:**

```text
venv\Scripts\python.exe scripts/ar_tide_iter/show_status.py --id iterNNN --watch
```

**Do not use** `run_overnight.py --hours` for this skill — that invokes `overnight_planner` (knob lattice only).

## Hypothesis classes (use when plateaued)

Pick **one class per run**; combine keys inside the class.

| Class                     | When                               | Example direction                                          |
| ------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| **Champion memorization** | Scratch teacher stalled mid-range  | `lr=5e-5`, no SS, `λ_inc=0.1`, full 200 ep                 |
| **High-lr memorization**  | Slow climb from low teacher        | `lr=1e-4`, no SS, drop decode extras                       |
| **Capacity**              | Underfitting / flat val gate early | Tier B: `d_model`, `n_enc_layers`, `patch_frames`          |
| **Consistency / decode**  | Teacher perfect, free-run short    | mild SS, `λ_inc`, `incremental_consistency_max_steps`      |
| **Loss rebalance**        | Teacher ok, event_f1 lags ordered  | `lambda_residual`, `lambda_time`, `eos_token_weight_scale` |
| **Implementation**        | Metric/training path suspect       | Tier C: fix val loss, decode, callbacks + tests            |

## Outcome matrix

| Result                     | Next action                                                                |
| -------------------------- | -------------------------------------------------------------------------- |
| Teacher ↑ new scratch best | Explore **within class** (one more coordinated change) or adjacent class   |
| Teacher flat ≥5 runs       | **Upgrade tier** or switch class — never another single-key tweak          |
| Teacher ≥634/634           | Switch to decode class; offline `--ar_decode` is automatic in `run_exp.py` |
| Free-run **634/634**       | `graduate_ar_tide_overfit.py`, stop                                        |
| Train crash / no ckpt      | Infra retry same id with `--reuse-last-config`; recipe change → new id     |

## Optional: cursor-autoresearch MCP

For keep/revert git discipline on **Tier C** edits, you may adopt [cursor-autoresearch](https://github.com/ergenekonyigit/cursor-autoresearch):

- Benchmark script wraps `run_exp.py` and parses teacher or free-run from `results.jsonl`
- Metric: scratch teacher ordered match (memorize phase) or free-run ordered match (decode phase)
- One atomic commit per hypothesis before `run_experiment`

Not required for this repo; the loop above is sufficient with `next_experiment.json` + `--once`.

## Stop conditions

- **Success:** offline 634/634 free-run → graduate → stop
- **Time:** session summary at top of `AR_TIDE_OVERFIT_ITER_LOG.md`
- **Blocked:** document blocker in log + NOTE; do not spin silently

## Related

| Item                     | Path                                                                      |
| ------------------------ | ------------------------------------------------------------------------- |
| Iter harness README      | [scripts/ar_tide_iter/README.md](../../../scripts/ar_tide_iter/README.md) |
| AR design / debug        | [AR_ONSET_DESIGN.md](../../../docs/research/AR_ONSET_DESIGN.md)           |
| Dense ablations (non-AR) | [tide-ablations](../tide-ablations/SKILL.md)                              |
