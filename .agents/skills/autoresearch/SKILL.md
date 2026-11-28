---
name: autoresearch
description: >-
  One-prompt agent autoresearch loop for StepCOVNet: goal + time budget → orient,
  hypothesize, run, log, repeat until success or deadline. Use when the user says
  "run autoresearch", gives a research goal with a budget, or names a profile
  (e.g. itgpt-onset).
---

# Autoresearch (agent loop)

**You** run the research loop in this agent session. Harness scripts train, eval, and record numbers; **you** choose hypotheses from evidence rather than handing reasoning to unattended knob mutators.

North star: [PIPELINE_ARCHITECTURE.md](../../../docs/research/PIPELINE_ARCHITECTURE.md).  
Logging: [research-session-workflow](../research-session-workflow/SKILL.md) · Eval: [onset-evaluation](../onset-evaluation/SKILL.md) · Colab: [colab-gpu-stepcovnet](../colab-gpu-stepcovnet/SKILL.md) · WSL GPU: [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md).

## One-prompt invocation

When the user sends **one message** with a **goal** and **time budget**, run the full loop **without asking between iterations** unless blocked.

**Parse from the user message:**

| Field       | Required                                                    | Default                                                                                                 |
| ----------- | ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| **Profile** | No                                                          | Infer from goal (see [Choose a profile](#choose-a-profile)); if ambiguous, ask **once** before the loop |
| **Goal**    | No                                                          | Profile default success criterion                                                                       |
| **Budget**  | No                                                          | **3 hours**                                                                                             |
| **Go**      | Implied if user says autoresearch / do not ask between runs | Loop autonomously                                                                                       |

**Example (ITGPT onset):**

```text
Run autoresearch.
Profile: itgpt-onset
Goal: Beat Stage 1 Dataset A and Stage 2 300t/50v F1@0.5 champions with clean grounding.
Budget: 4 hours.
Go — do not ask me between runs.
```

**Example (generic):**

```text
Run autoresearch.
Goal: reduce pytest time for tests/itgpt/ by 20% without failing tests.
Budget: 2 hours.
Metric: pytest wall time + pass/fail.
Go.
```

### Agent obligations (every profile)

1. **Load this skill** + the active [profile doc](profiles/).
2. **Preflight** — profile-specific cleanup (stray jobs, locks, env); one concurrent heavy job unless profile says otherwise.
3. **Record deadline** — `deadline = now + budget` at session start; treat the budget as a **wall-clock obligation**, not a suggestion.
4. **Loop until** goal met **or** `now >= deadline` (see [Budget discipline](#budget-discipline)):
   - **Orient** — live scoreboard / breakdown JSON / `EXPERIMENT_LOG.md`
   - **Diagnose** — one sentence; cite measured numbers
   - **Hypothesize & prove headroom** — tier A/B/C; quantified bucket ceiling + falsifiable prediction
   - **Plan** — profile experiment artifact (JSON, config path, or code edit plan)
   - **Pre-GPU verify** — Tier B/C: 30s CPU mechanism check (when feasible) + `pytest` + `pre_submit.py --fast`
   - **Run** — portfolio lane command; abort early if clearly dominated mid-run
   - **Log** — prepend `EXP-…` same turn
   - **Outer meta-loop (every 3–5 runs)** — wave causal-belief synthesis, diagnostic tooling check, and process promotion via [agent-maintenance](../agent-maintenance/SKILL.md)
   - **Judge** — if goal not met and time remains → **immediately** plan the next iteration; do not end the session
5. **Do not ask** the user mid-loop except: hard block, cheat/policy violation, tests red, unrecoverable GPU.
## Choose a profile

| Profile            | Use when                                                                                 | Detail                                                    |
| ------------------ | ---------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| **itgpt-onset**    | PyTorch ITGPT-MERT onset placement hill-climbing (multi-GPU Colab + WSL, diagnostics)    | [profiles/itgpt-onset.md](profiles/itgpt-onset.md)        |
| **raw-audio-onset** | Fast CPU/local-GPU end-to-end raw-audio beat tracking & onset decoding (`eval_raw_audio_onset.py`) | [profiles/raw-audio-onset.md](profiles/raw-audio-onset.md) |
| **experiment-log** | General research; log `EXP-…` in EXPERIMENT_LOG                                          | [profiles/experiment-log.md](profiles/experiment-log.md)  |
| **custom**         | User defines metric + run command in the goal                                            | You define orient/run/log for that session; still follow tiers + anti-spam |

If the user names a specific skill or script workflow, treat it as **custom** or execute that workflow inside one autoresearch iteration, then return to the judge step.

## Change tiers (all profiles)

Pick the **lowest tier that tests the hypothesis**.

| Tier                   | What you may change                                                    | Before expensive run                       |
| ---------------------- | ---------------------------------------------------------------------- | ------------------------------------------ |
| **A — Parameters**     | Config / CLI / JSON overrides (coordinated recipe, not one stray knob) | Profile-specific (often none)              |
| **B — Structure**      | Model shape, data schema, pipeline stage parameters                    | Note why in plan                           |
| **C — Implementation** | `src/`, `scripts/` behavior (not gitignored iter logs)                 | Targeted `pytest` + `pre_submit.py --fast` |

Tier A must include **reasoning**: what failed, what you expect, what changed vs parent/baseline.

### Pre-GPU Headroom & Mechanism Proof (Tier B/C gate)

Before spending GPU minutes on a Tier-B or Tier-C change, state and verify two gates:

1. **Measured headroom:** Quantify how many errors the target bucket actually holds from `eval_val_breakdown.json` or per-song outputs (e.g., *"this slice causes X of 7,708 errors, max possible ceiling +Y pp"*).
2. **Cheap CPU falsification check (30 s, when feasible):** Test the hypothesis on saved logits, `eval_val_breakdown.json`, or the worst-3 songs on CPU *before* launching GPU training — if the premise fails offline, drop it without burning GPU compute.

## Portfolio scheduling & fail-fast screening

When running multiple lanes concurrently, assign explicit **portfolio roles** instead of correlated tweaks:

| Lane | Portfolio role | What belongs here |
| ---- | -------------- | ----------------- |
| **`L4` (Colab primary)** | **Exploit / Verify** | Highest-EV champion refinement, 2nd-seed confirmation, Stage-2 transfer gate |
| **`T4` (Colab secondary)** | **Orthogonal Explore** | Distinct Tier-B/C hypothesis class targeting the #1 remaining error slice |
| **CPU / Zero-GPU (concurrent)** | **Diagnostics & Prep** | `eval_raw_audio_onset.py`, `--diagnose_decoding`, model soups, worst-song error audits, Wave N+1 Tier-C coding/tests |

- **Fail-fast early abort:** Inspect mid-run validation checkpoints; abort clearly dominated runs early instead of burning remaining epochs.

## Outer meta-loop (Wave reflection every 3–5 runs)

True RSI improves both the model *and* the search process. At the end of every 3–5 run wave (before launching Wave N+1):

1. **Causal belief synthesis:** Record *why* an entire hypothesis class succeeded or failed (mechanism-level insight in `DISCUSSION_NOTES.md` / `EXPERIMENT_LOG.md`), not just flat flag lists.
2. **Diagnostic tooling check:** If you could not see *why* a slice or song degraded (blind diagnosis), spend ~5 minutes adding/extending a reusable diagnostic (`--error_breakdown`, `eval_raw_audio_onset.py`, slice dump) before launching more GPU runs.
3. **Process & skill self-improvement:** Promote repeated workflow friction or search-strategy lessons into the active profile or skill via [agent-maintenance](../agent-maintenance/SKILL.md) (backed by local evals).

## Anti-spam (all profiles)

1. **Plateau:** Primary metric unchanged for **≥5 runs** (profile defines tolerance) → **forbid** single-parameter tweakers; **upgrade tier** or switch hypothesis **class**.
2. **No repeats:** Do not rerun the same recipe/config fingerprint unless infra retry.
3. **Pinned policy:** Do not change eval gates / success metrics mid-session unless the profile allows it.
4. **One job per GPU target:** No duplicate training drivers on the same GPU (profile preflight). Parallel chains across distinct GPUs (e.g. Colab `L4` + `T4` + local WSL) and parallel Tier-C coding are allowed when the profile isolates git worktrees (see [profiles/itgpt-onset.md](profiles/itgpt-onset.md)).
5. **No planner handoff:** Do not delegate reasoning to unattended lattice/sweep planners while this skill is active.

## Budget discipline

When the user gives a **budget** (e.g. “7 hours”, “overnight”, “until morning”):

| Rule | Requirement |
| ---- | ----------- |
| **Use the full budget** | Keep iterating until **goal met** or **`now >= deadline`**. A budget is wasted if you stop early with time left. |
| **Valid stop reasons only** | **Success** (goal met), **budget exhausted**, or **hard block** (GPU dead, tests red, policy). |
| **Invalid stop reasons** | “Queue finished”, “ran N experiments”, “no more pre-written plans”, “good enough progress”, “session summary written”. |
| **No finite queue exit** | Never use a fixed list of experiments as the session boundary. Seed plans are OK as a **buffer**; when the buffer empties, **replan from evidence** and continue until deadline. |
| **Throughput estimate** | Before starting: `expected_runs ≈ budget_minutes / minutes_per_run` (profile-specific; e.g. ~16–20 min per Stage-1 Colab run). |
| **After each run** | If `now < deadline` and goal not met → **next plan + next run** in the same session. |
| **Background drivers** | If you background a run, **you** remain responsible for monitoring completion (`--attach` / logs) and supplying the next plan until deadline or goal. |
| **Resume** | If context or chat ends early but wall-clock budget remains, user may say _continue autoresearch_ — pick up with **remaining** budget, not a fresh short queue. |

**Wrong (stops early on a multi-hour budget):** pre-write 4 configs → run queue → exit when queue empties.

**Right:** `deadline = now + budget` → loop: orient (`itgpt_scoreboard.py` + `eval_val_breakdown.json`) → hypothesize Tier A/B/C → train/eval (`colab_run_remote.sh` or local WSL) → log `EXP-…` → repeat until `now >= deadline` or target metric + guardrails pass.

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

## Stop conditions

Stop **only** when one of these is true:

- **Success:** profile success criterion met
- **Budget exhausted:** `now >= deadline` → summary in profile log with time used
- **Blocked:** document in profile log + `NOTE-…` if appropriate; tell user what is needed

Do **not** stop because a pre-seeded experiment list finished while `now < deadline`.

### Live-truth resume & Handoff Packet

- **No separate session state file:** On compaction or resume (*continue autoresearch*), orient directly from live disk truth (`docs/research/EXPERIMENT_LOG.md`, `scripts/itgpt_scoreboard.py`, `bash scripts/colab_worktree.sh list`, `colab sessions`).
- **Termination Handoff Packet:** At stop, report: (1) current champion commit/config + metrics, (2) top-3 causal findings from this session, (3) remaining open hypotheses ranked by measured headroom, and (4) confirmed `0` active Colab VMs (or active local jobs).

## Gotchas (all profiles)

| Gotcha                                  | What goes wrong                                                                             | What to do                                                                                             |
| --------------------------------------- | ------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------ |
| **Unattended sweeper instead of agent** | Automated parameter sweeps tweak JSON knobs without reasoning and plateau fast              | Drive hypotheses directly per iteration in this session                                                |
| **Duplicate drivers**                   | Two drivers on the same GPU → overlapping GPU jobs, OOM, corrupt lock state                 | Preflight every session; max one job per GPU target; isolate parallel lanes in worktrees               |
| **Tier C without tests**                | Long GPU run on broken decode/metrics path                                                  | `pytest` + `pre_submit.py --fast` **before** GPU launch                                                |
| **Single-knob spam after plateau**      | Primary metric flat for many runs; agent keeps nudging one λ or lr                          | Enforce [anti-spam](#anti-spam); upgrade tier or switch hypothesis class                               |
| **Changing success metric mid-session** | Incomparable runs (e.g. swap checkpoint metric or eval gate)                                | Keep profile pinned eval policy; change only with user approval                                        |
| **Finite queue / early exit**           | Pre-written N experiments finish early; agent declares “session done” with budget left      | [Budget discipline](#budget-discipline): replan when buffer empty; only stop at deadline or goal       |
| **Long agent session**                  | Context fills after many iterations; agent may stop early                                   | User says _continue autoresearch_ with **remaining** budget; summarize state first                     |
| **Uncommitted outputs vs disk**         | `experiments/colab/` / `logs/` ahead of git                                                 | Check disk outputs and scoreboard before concluding a run is missing                                   |
| **Background train**                    | Long remote run returns poll timeout                                                      | Re-attach (`--attach`) or inspect live logs / TensorBoard                                              |
| **Committing `logs/`**                  | Huge artifacts, machine paths in PR                                                         | Never commit `logs/`; research logs in `docs/research/` only                                           |
| **Bare `python`**                       | Wrong env on Windows/WSL                                                                    | Project venv `python` from repo root; WSL per [wsl-gpu-stepcovnet](../wsl-gpu-stepcovnet/SKILL.md)     |

Profile-specific gotchas: see each file under [profiles/](profiles/).

## Add a profile

See [profiles/README.md](profiles/README.md).
