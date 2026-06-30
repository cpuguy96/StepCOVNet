---
name: autoresearch
description: >-
  One-prompt agent autoresearch loop for StepCOVNet: goal + time budget → orient,
  hypothesize, run, log, repeat until success or deadline. Use when the user says
  "run autoresearch", gives a research goal with a budget, or names a profile
  (e.g. ar-tide-overfit). Not for unattended Python planners (--hours / lattice).
disable-model-invocation: true
---

# Autoresearch (agent loop)

**You** run the research loop in this Cursor session. Harness scripts train, eval, and record numbers; **you** choose hypotheses from evidence. Do **not** hand reasoning to unattended knob mutators (`overnight_planner`, fixed queues, lattice search) when this skill is active.

North star: [PIPELINE_ARCHITECTURE.md](../../../docs/research/PIPELINE_ARCHITECTURE.md).  
Logging: [research-session-workflow](../research-session-workflow/SKILL.md).  
GPU: [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md).

## One-prompt invocation

When the user sends **one message** with a **goal** and **time budget**, run the full loop **without asking between iterations** unless blocked.

**Parse from the user message:**

| Field       | Required                                                    | Default                                                                                                 |
| ----------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Profile** | No                                                          | Infer from goal (see [Choose a profile](#choose-a-profile)); if ambiguous, ask **once** before the loop |
| **Goal**    | No                                                          | Profile default success criterion                                                                       |
| **Budget**  | No                                                          | **3 hours**                                                                                             |
| **Go**      | Implied if user says autoresearch / do not ask between runs | Loop autonomously                                                                                       |

**Example (tide):**

```text
Run autoresearch.
Profile: ar-tide-overfit
Goal: scratch teacher 634/634 then free-run 634/634 on tide @ 20 ms.
Budget: 7 hours.
Go — do not ask me between runs.
```

**Example (generic):**

```text
Run autoresearch.
Goal: reduce pytest time for tests/onset_ar/ by 20% without failing tests.
Budget: 2 hours.
Metric: pytest wall time + pass/fail.
Go.
```

### Agent obligations (every profile)

1. **Load this skill** + the active [profile doc](profiles/).
2. **Preflight** — profile-specific cleanup (stray jobs, locks, env); one concurrent heavy job unless profile says otherwise.
3. **Record deadline** — `deadline = now + budget` at session start; treat the budget as a **wall-clock obligation**, not a suggestion.
4. **Loop until** goal met **or** `now >= deadline` (see [Budget discipline](#budget-discipline)):
   - **Orient** — profile brief / logs / last metrics
   - **Diagnose** — one sentence; cite numbers
   - **Hypothesize** — tier A/B/C; what confirms/refutes
   - **Plan** — profile experiment artifact (JSON, config path, or code edit plan)
   - **Verify** — tier C: tests + `pre_submit.py --fast` before expensive run
   - **Run** — profile run command; wait for completion
   - **Log** — profile research log
   - **Judge** — if goal not met and time remains → **immediately** plan the next iteration; do not end the session
5. **Do not ask** the user mid-loop except: hard block, cheat/policy violation, tests red, unrecoverable GPU.
6. **End with** session summary: best metric, last experiment id/run, stop reason (**only** success / budget exhausted / blocked).

**Aliases:** `run ar-tide autoresearch` → profile `ar-tide-overfit`.

## Choose a profile

| Profile             | Use when                                                                   | Detail                                                                     |
| ------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------- |
| **ar-tide-overfit** | AR tide single-chart 634/634 gate, `ar_tide_iter`, overnight tide prompt   | [profiles/ar-tide-overfit.md](profiles/ar-tide-overfit.md)                 |
| **experiment-log**  | General research; log `EXP-…` in EXPERIMENT_LOG; no `ar_tide_iter` harness | [profiles/experiment-log.md](profiles/experiment-log.md)                   |
| **custom**          | User defines metric + run command in the goal                              | You define orient/run/log for that session; still follow tiers + anti-spam |

If the user names a playbook skill (e.g. tide-ablations), treat it as **custom** or follow that skill **inside** one autoresearch iteration, then return to the judge step.

## Change tiers (all profiles)

Pick the **lowest tier that tests the hypothesis**.

| Tier                   | What you may change                                                    | Before expensive run                       |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------------------ |
| **A — Parameters**     | Config / CLI / JSON overrides (coordinated recipe, not one stray knob) | Profile-specific (often none)              |
| **B — Structure**      | Model shape, data schema, pipeline stage parameters                    | Note why in plan                           |
| **C — Implementation** | `src/`, `scripts/` behavior (not gitignored iter logs)                 | Targeted `pytest` + `pre_submit.py --fast` |

Tier A must include **reasoning**: what failed, what you expect, what changed vs parent/baseline.

## Anti-spam (all profiles)

1. **Plateau:** Primary metric unchanged for **≥5 runs** (profile defines tolerance) → **forbid** single-parameter tweakers; **upgrade tier** or switch hypothesis **class**.
2. **No repeats:** Do not rerun the same recipe/config fingerprint unless infra retry.
3. **Pinned policy:** Do not change eval gates / success metrics mid-session unless the profile allows it.
4. **One job:** No duplicate training drivers (profile preflight).
5. **No planner handoff:** Do not use unattended `--hours` / lattice planners while this skill is active — they replace agent reasoning.

## Budget discipline

When the user gives a **budget** (e.g. “7 hours”, “overnight”, “until morning”):

| Rule | Requirement |
| ---- | ----------- |
| **Use the full budget** | Keep iterating until **goal met** or **`now >= deadline`**. A budget is wasted if you stop early with time left. |
| **Valid stop reasons only** | **Success** (goal met), **budget exhausted**, or **hard block** (GPU dead, tests red, policy). |
| **Invalid stop reasons** | “Queue finished”, “ran N experiments”, “no more pre-written plans”, “good enough progress”, “session summary written”. |
| **No finite queue exit** | Never use a fixed list of experiments as the session boundary. Seed plans are OK as a **buffer**; when the buffer empties, **replan from evidence** and continue until deadline. |
| **Throughput estimate** | Before starting: `expected_runs ≈ budget_minutes / minutes_per_run` (profile-specific). A 7 h tide session ≈ **40–50** runs at ~8–10 min each — plan for that volume, not a handful. |
| **After each run** | If `now < deadline` and goal not met → **next plan + next run** in the same session (or feed the harness before `plan-wait` expires). |
| **Background drivers** | If you background a harness, **you** remain responsible for supplying the next plan until deadline or goal. Monitor completion; do not fire-and-forget. |
| **Resume** | If context or chat ends early but wall-clock budget remains, user may say _continue autoresearch_ — pick up with **remaining** budget, not a fresh short queue. |

**Wrong (stops at ~1 h on a 7 h budget):** pre-write 8 recipes → custom script iterates list → exit when list empty.

**Right:** `deadline = now + 7h` → loop: `session_brief` → `next_experiment.json` → `--autoresearch --once` → repeat until `now >= deadline` or 634/634 free-run.

## Core loop (pseudocode)

```text
deadline = now + budget
while now < deadline and not success(goal):
    orient(profile)
    if plateau(primary_metric): require tier_upgrade or new class
    plan = hypothesize_tier_A_B_or_C()
    write profile experiment artifact
    if tier C: pytest + pre_submit --fast
    run profile command (foreground); wait for completion
    log profile
    if success(goal): break
    # mandatory: do not exit here just because a seed queue is empty — replan
assert stop_reason in (success, budget_exhausted, blocked)
summarize session
```

## Optional: cursor-autoresearch MCP

For Tier C with git keep/revert discipline, see [cursor-autoresearch](https://github.com/ergenekonyigit/cursor-autoresearch). Not required; this skill works with profile harnesses alone.

## Stop conditions

Stop **only** when one of these is true:

- **Success:** profile success criterion met
- **Budget exhausted:** `now >= deadline` → summary in profile log with time used
- **Blocked:** document in profile log + `NOTE-…` if appropriate; tell user what is needed

Do **not** stop because a pre-seeded experiment list finished while `now < deadline`.

## Gotchas (all profiles)

| Gotcha                                  | What goes wrong                                                                             | What to do                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Planner instead of agent**            | `run_overnight --hours` / lattice scripts tweak JSON knobs without reasoning; plateaus fast | Use `--autoresearch --once` or `--autoresearch --hours N`; never bare `--hours`       |
| **Duplicate drivers**                   | Two `run_overnight` / `run_exp` shells → overlapping GPU jobs, corrupt lock state           | Preflight every session; one training job; see profile preflight                                       |
| **Tier C without tests**                | 200-epoch GPU run on broken decode/metrics path                                             | `pytest` + `pre_submit.py --fast` **before** WSL train                                                 |
| **Single-knob spam after plateau**      | Primary metric flat for many runs; agent keeps nudging one λ or lr                          | Enforce [anti-spam](#anti-spam); upgrade tier or switch hypothesis class                               |
| **Changing success metric mid-session** | Incomparable runs (e.g. swap checkpoint metric or eval gate)                                | Keep profile pinned eval policy; change only with user approval                                        |
| **Finite queue / early exit**           | Pre-written N experiments finish in ~1 h; agent declares “session done” on 7 h budget        | [Budget discipline](#budget-discipline): replan when buffer empty; only stop at deadline or goal       |
| **Long Cursor session**                 | Context fills after many iterations; agent may stop early                                   | User says _continue autoresearch_ with **remaining** budget; summarize state first                   |
| **Uncommitted harness vs disk**         | `experiments.json` / logs ahead of git; brief may still work off `results.jsonl`            | Do not assume registry is committed; read `logs/` + brief                                              |
| **Background train**                    | No epoch lines in chat; looks idle while GPU is busy                                        | Profile watch command or `show_status`; foreground `--once` when possible                              |
| **Committing `logs/`**                  | Huge artifacts, machine paths in PR                                                         | Never commit `logs/`; research logs in `docs/research/` only                                           |
| **Bare `python`**                       | Wrong env on Windows/WSL                                                                    | `venv\Scripts\python.exe` from repo root; WSL per [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md) |

Profile-specific gotchas: see each file under [profiles/](profiles/).

## Add a profile

See [profiles/README.md](profiles/README.md).
