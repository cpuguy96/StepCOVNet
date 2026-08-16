# Agent self-journal

Iterative improvements to agent behavior, process, and conventions on this project. **Not** a substitute for [EXPERIMENT_LOG.md](../research/EXPERIMENT_LOG.md) or [DISCUSSION_NOTES.md](../research/DISCUSSION_NOTES.md) — those hold research findings; this holds **how we work better**.

**Order:** newest entries first (prepend below).

**Same-turn promotion:** The journal is a receipt, not the fix. Every entry must cite an **Artifact** path (rule, skill, or code) created or updated **in the same turn** before the JRN is written. See [agent-self-improvement skill](../../.cursor/skills/agent-self-improvement/SKILL.md).

## When to prepend

**Immediately** when the user makes a **steering correction** — how you decide, prioritize, run commands, log output, route docs, commit, etc. Do not wait for a long or “substantive” session to end.

**Also** when you discover:

- A mistake that wasted time or produced wrong conclusions
- A fix or workaround that should become default
- A user-established convention worth preserving
- A repeated workflow missing from `.cursor/skills/`

**Periodically** during work: after discrete tasks with process lessons, when the same friction recurs, or before switching task areas — journal in the same session; do not defer everything to session end.

Maintenance rule: **artifact first, journal second** in the same turn; link `EXP-…` / `NOTE-…` when relevant.

## Entry format

Insert **at the top** of [Entries](#entries) (below this section):

```markdown
### JRN-YYYYMMDD-NN: Short title

| Field            | Value                                            |
| ---------------- | ------------------------------------------------ |
| **Timestamp**    | YYYY-MM-DD HH:MM:SS (system clock at write time) |
| **Category**     | mistake \| fix \| convention \| skill-gap        |
| **Summary**      | What happened                                    |
| **Artifact**     | Path(s) created or updated (required)            |
| **Action taken** | One line: what the artifact enforces             |
| **Related**      | EXP-…, NOTE-…, skill name                        |
```

---

## Entries

### JRN-20260815-03: Stop DDC placement eval unless asked

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 14:53:03 |
| **Category** | convention |
| **Summary** | User: done with DDC eval for now; move on. Do not stack more placement diagnostics (FP histograms, matched-count, threshold sweeps) on the frozen ckpt. |
| **Artifact** | skill: `.cursor/skills/whats-next/SKILL.md` (§ No more DDC placement eval unless asked); Current phase primary track → dense `final_data` scoreboard |
| **Action taken** | what's-next forbids further DDC eval in Now/do-it after T-repro accepted or user says eval is done; Current phase next is 50/100-row dense MERT smoke. No new alwaysApply rule |
| **Related** | EXP-20260815-03 · EXP-20260815-06 |

### JRN-20260815-02: Onset/placement only — no DDC step selection unless asked

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 02:43:38 |
| **Category** | convention |
| **Summary** | User correction: do not route to DDC step selection; current work is onsets (placement) only. |
| **Artifact** | skill: `.cursor/skills/whats-next/SKILL.md` (§ No DDC step selection unless asked); Current phase Next action / Defer |
| **Action taken** | what's-next forbids selection in Now/do-it unless explicitly requested; Current phase next is best-val placement checkpoint, not arrow-pattern LSTM. No new alwaysApply rule |
| **Related** | EXP-20260815-03 |

### JRN-20260815-01: DDC TensorBoard uses one stage logs tree with per-run folders

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-15 02:30:19 |
| **Category** | convention |
| **Summary** | User could not compare 8-ep vs 128-ep DDC runs in TensorBoard because each experiment wrote to a separate `callback_root_dir` (`placement/` vs `placement_128ep/`) and Keras logged to the stage `logs/` root (`logs/train`) instead of a timestamped run folder. |
| **Artifact** | code: `src/stepcovnet/ddc/trainers.py`; scoped rule: `.cursor/rules/scripts-execution.mdc`; config: `configs/ddc/placement_fraxtil_128ep.json` (`callback_root_dir` shared) |
| **Action taken** | DDC TensorBoard now writes `callbacks/ddc/placement/logs/{timestamp}-{model_name}/` like dense/AR; 128-ep config shares the placement stage tree; live TB restarted with `--logdir_spec=8ep:…,128ep:…`. No new alwaysApply rule |
| **Related** | EXP-20260815-02 · scripts-execution TensorBoard stage table |

### JRN-20260814-01: Cite prior art; paper ledger is the citation home

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-14 22:15:45 |
| **Category** | convention |
| **Summary** | User correction: recreation of DDC/DDCL/ITGPT is understanding prior art to cite, not stealing. Paper-bound references and claims must be catalogued, not left in chat. |
| **Artifact** | scoped rule: `.cursor/rules/research-logging.mdc`; skill: `.cursor/skills/research-session-workflow/SKILL.md`; `docs/research/PAPER_LEDGER.md`; `docs/research/paper.bib`; AGENTS.md paper-citations row |
| **Action taken** | Same-turn citation keys in PAPER_LEDGER + paper.bib; prior art described as cited baselines; no new alwaysApply rule |
| **Related** | NOTE-20260814-02 · NOTE-20260814-01 |

### JRN-20260807-03: Hard locality masks are diagnostic, not the product path

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 12:39:00 |
| **Category** | convention |
| **Summary** | User rejected stacking hard prev-local R / force-advance hacks for val gains. Goal is a holistic open-set predictor (incl. long pauses), not gap-histogram crutches. |
| **Artifact** | skill: `.cursor/skills/whats-next/SKILL.md` (§ No hard locality hacks as the product path); NOTE-20260807-06; Current phase pivot |
| **Action taken** | whats-next forbids further hard-R / force-advance as Now/do-it; hard-R EXPs kept as evidence only; next = eval R=4 without hard window then non-hard-mask localization |
| **Related** | NOTE-20260807-06 · EXP-20260807-10 · EXP-20260807-11 |

### JRN-20260807-02: Do not recommend ladder scale-up unless asked

| Field | Value |
| ----- | ----- |
| **Timestamp** | 2026-08-07 10:19:30 |
| **Category** | convention |
| **Summary** | User declined R3/scale-up after gap diagnosis pointed at more data. Agents must not put ladder scale-up in what's-next **Now** / **do it** unless explicitly requested. |
| **Artifact** | `.cursor/skills/whats-next/SKILL.md` (§ No ladder scale-up unless asked); Current phase + NOTE-20260807-02 retracted R3 as next |
| **Action taken** | Skill forbids R3+ in Now/do-it; scale only under Alternate when evidence supports and user may switch; research docs defer scale-up |
| **Related** | NOTE-20260807-02 · EXP-20260807-03 · JRN-20260807-01 |

### JRN-20260807-01: Evidence before suggesting the next train

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-08-07 00:18:48 |
| **Category**     | convention |
| **Summary**      | Suggested attn-mass aux as the next R2 probe because Current phase named it, without measurements showing it would help. User: do not recommend experiments without evidence first. |
| **Artifact**     | Skill: `.cursor/skills/whats-next/SKILL.md` § Evidence before recommending a run; `.cursor/skills/research-session-workflow/SKILL.md` § During work; Current phase demoted speculative attn-mass in `docs/research/EXPERIMENT_LOG.md` |
| **Action taken** | whats-next / research workflow require cheap evidence (or a mechanism already supported by logged numbers) before **Now** / do-it trains; Current phase next action is diagnose QK-LN ckpt first |
| **Related**      | NOTE-20260806-03 · EXP-20260806-07 |

### JRN-20260802-04: A green test suite hid a dead feature for months

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-02 23:12:05                                                                                        |
| **Category**     | fix                                                                                                        |
| **Summary**      | A 2 h 15 m scheduled-sampling run reproduced its baseline to 4 dp on all 500 epochs. `train_step` gated sampling on a Python attribute inside Keras' traced `tf.function`, so the branch was compiled out at trace time (always during warmup, `p = 0`). Five existing tests asserted only the ramp **arithmetic**, so the suite was green on dead code. |
| **Artifact**     | Code: `onset_ar/trainers.py` (`scheduled_sampling_p` → `tf.Variable`, new `_decoder_inputs_for_step` under `tf.cond`); test: `tests/onset_ar/models_test.py`; scoped rule `.cursor/rules/python-tests.mdc` § Test patterns — “Schedules and knobs” + “Values read inside `tf.function`” |
| **Action taken** | Verified the new test **fails** against a faithful reproduction of the original bug before trusting it, then confirmed on GPU with a 5-epoch A/B: warmup epochs bit-identical, post-ramp epochs diverge. |
| **Lesson**       | Two traps worth remembering. A knob's schedule passing its unit tests says nothing about the knob being *read*; assert the behavior change. And `tf.function(model.bound_method)` does not reliably build a graph — my first regression test passed against broken code because of it. Suspiciously identical numbers between a variant and its baseline are a defect signal, not a null result. |
| **Related**      | [EXP-20260802-05](../research/EXPERIMENT_LOG.md#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step) · JRN-20260802-03 |

### JRN-20260802-03: Warm start on a ladder run contradicted a standing scratch-only rule

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-02 14:52:40                                                                                        |
| **Category**     | mistake                                                                                                    |
| **Summary**      | Built the scheduled-sampling config with `init_model_path` → R2's checkpoint, plus a lowered LR and a shortened epoch budget. The tide harness already strips `init_model_path` as “cheating” (`session_brief.py`, `ar-tide-overfit.md`), but that rule lived only in tide tooling, so I did not apply it to the ladder. User: “i don't think it make sense to be training warm start at all … seems like a bug waiting to happen.” |
| **Artifact**     | `docs/research/AR_SCALING_LADDER.md` § 3 Protocol — new **Initialization** row (random init; `init_model_path` not used); `configs/ar/ladder_50t_50v_ss.json` restored to the R2 recipe (500 ep, LR 1e-4) |
| **Action taken** | Killed the warm-started run, deleted its artifacts, restarted from random init with SS as the only variable vs the R2 rung. Wrote the scratch-only constraint into the ladder protocol table where the ladder's other invariants live. |
| **Lesson**       | Warm start silently breaks reproducibility: `model_output_dir` paths are mutable, so a rerun of the parent rung changes a child run's starting weights after the fact. It also stacked three deviations at once (init, LR, epochs), so nothing would have been attributable to scheduled sampling. When a norm exists in one harness, check whether the new track needs it before deviating. |
| **Related**      | JRN-20260802-02 · [AR_SCALING_LADDER.md](../research/AR_SCALING_LADDER.md) § 3                              |

### JRN-20260802-02: Fix the directory layout, not the launcher

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-02 14:40:12                                                                                        |
| **Category**     | convention                                                                                                 |
| **Summary**      | Ladder rungs each had their own `callback_root_dir`, so TensorBoard sorted `ladder_200t…` before `ladder_50t…`. I answered with sorting/labeling logic inside `scripts/tensorboard_compare.py` instead of fixing the paths. User: “do it properly and organize things by file paths … i don't think there needs to really be a seperate tensorboard launch script.” |
| **Artifact**     | Code: `onset_ar/config.py` + `trainers.py` (`run.run_label`); configs `configs/ar/ladder_*.json`; scoped rule `.cursor/rules/scripts-execution.mdc` § TensorBoard with training + § AR config naming; deleted `scripts/tensorboard_compare.py` and its test |
| **Action taken** | One `callback_root_dir` per **stage** (`callbacks/ar/ladder`), `run_label` per variant inside the timestamped run folder — so `tensorboard --logdir <stage>/logs` is name-sorted chronologically with no launcher. Migrated existing rung dirs into the stage tree. |
| **Lesson**       | When a helper script exists to compensate for a layout, fix the layout. Tooling that reorders or relabels paths at read time is a smell, not a feature. |
| **Related**      | JRN-20260802-01                                                                                            |

### JRN-20260802-01: PowerShell `1>` + Tee hides console

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-02 12:52:20                                                                                        |
| **Category**     | mistake                                                                                                    |
| **Summary**      | R2 val decode used `1> logs/...json 2>&1 \| Tee-Object` — stdout bound to file before pipeline, so Tee/Cursor terminal got nothing despite scripts-execution visible-console rule. |
| **Artifact**     | `.cursor/rules/scripts-execution.mdc` (scoped rule — PowerShell anti-pattern + JSON/stderr tee pattern)  |
| **Action taken** | Documented that `1>`/`>` before Tee defeats visible console; for split stdout/stderr scripts, tee merged streams only. |
| **Related**      | ladder R2 free-run decode                                                                                  |

### JRN-20260801-02: whats-next "do it" contract

| Field            | Value                                                                                                      |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-01 21:35:00                                                                                        |
| **Category**     | convention                                                                                                 |
| **Summary**      | User wants "do it" after whats-next to unambiguously trigger execution; added **If you say "do it"** block and § "Do it" rules to the skill. |
| **Artifact**     | `.cursor/skills/whats-next/SKILL.md`                                                                       |
| **Action taken** | Every whats-next answer lists numbered execute steps + **Done when**; follow-up "do it" runs that list without re-asking. |
| **Related**      | [whats-next](../../.cursor/skills/whats-next/SKILL.md), JRN-20260801-01                                    |

### JRN-20260801-01: whats-next orientation stack

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-08-01 21:30:00                                                                                                                   |
| **Category**     | skill-gap                                                                                                                             |
| **Summary**      | User asks "what to do now?" often; answers varied because strategic doc was stale and local clone state was not checked systematically. |
| **Artifact**     | `.cursor/skills/whats-next/SKILL.md`, `scripts/project_status.py`, `EXPERIMENT_LOG.md` § Current phase compact block, `AGENTS.md` row |
| **Action taken** | Skill runs `project_status.py` + reads Current phase compact fields; fixed answer template; research-session-workflow updates block on routing EXPs. |
| **Related**      | [research-session-workflow](../../.cursor/skills/research-session-workflow/SKILL.md), [steering-correction-promotion](../../.cursor/skills/steering-correction-promotion/SKILL.md) |

### JRN-20260725-04: Check for uncommitted work before renormalizing line endings

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-25 22:34:00 |
| **Category**     | mistake |
| **Summary**      | While force re-checking out `*.sh` to fix CRLF, I also clobbered uncommitted edits to `scripts/eval_ar_onset_offline.py` and had to reapply the whole change from memory. `git checkout --` gives no prompt and no recovery. |
| **Artifact**     | alwaysApply rule `.cursor/rules/state-and-paths.mdc` § Git on Windows (PowerShell) → **Destructive checkout** |
| **Action taken** | Check `git status` and commit or stash before any `git checkout --` or renormalization pass |
| **Related**      | JRN-20260725-02 |

### JRN-20260725-03: STEPCOVNET_NO_WSL=1 silently forces Windows CPU

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-25 22:33:00 |
| **Category**     | mistake |
| **Summary**      | I set `STEPCOVNET_NO_WSL=1` on an AR decode expecting only to skip the dispatch wrapper. It ran on Windows CPU at 330s vs 94s on GPU, and nothing in the output flagged the downgrade — the user had to point out the GPU was idle. The rule documented the variable as a neutral opt-out. |
| **Artifact**     | alwaysApply rule `.cursor/rules/python-environment.mdc` § Windows development model — cost caveat added inline where the variable is defined |
| **Action taken** | Never set `STEPCOVNET_NO_WSL=1` for training, decode, or MERT; it does not fall back to GPU |
| **Related**      | EXP-20260724-03, wsl-gpu-stepcovnet |

### JRN-20260725-02: WSL shell scripts must be checked out LF

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-25 22:32:00 |
| **Category**     | fix |
| **Summary**      | WSL GPU dispatch failed with `: invalid option name` and `set: pipefail`. Root cause was `core.autocrlf=true` checking out `scripts/wsl_*.sh` with CRLF, which bash cannot parse. Cost significant time because the error text points at the script body, not the line endings. |
| **Artifact**     | Code: `.gitattributes` — `*.sh text eol=lf` (commit `3361b04`) |
| **Action taken** | Structural fix: `*.sh` now always lands LF on checkout regardless of `core.autocrlf`, so this cannot recur for new clones |
| **Related**      | JRN-20260725-04 |

### JRN-20260725-01: Commit messages via repeated -m, never a scratch file

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-25 01:14:00 |
| **Category**     | convention |
| **Summary**      | Bash heredoc commit syntax fails to parse in PowerShell, so I fell back to writing message prose into `_tmp/commit/msg.txt` and running `git commit -F`. Nothing temp was ever staged (explicit `git add` paths; `_tmp/` gitignored), but the user flagged the pattern as wrong regardless. |
| **Artifact**     | alwaysApply rule `.cursor/rules/state-and-paths.mdc` § Git on Windows (PowerShell) — merged into existing scratch-policy rule, no new file |
| **Action taken** | Multi-line commit messages use repeated `-m` flags, one per paragraph; `_tmp/` is for analysis artifacts, not commit prose |
| **Related**      | steering-correction-promotion |

### JRN-20260724-01: Scale-up naming + TB run suffixes

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-24 03:17:32 |
| **Category**     | convention |
| **Summary**      | Multi-song AR trains were mislabeled `smoke_*`; TensorBoard run names lacked train/val/epoch so splits were hard to tell apart. |
| **Artifact**     | Code: `onset_ar/trainers.py` `_get_experiment_name`; configs `configs/ar/scale_*t_*v.json`; scoped rule `.cursor/rules/scripts-execution.mdc` § AR config / TensorBoard naming |
| **Action taken** | Reserve `smoke` for 10-song gate; scale-up uses `scale_Nt_Nv`; TB suffix includes `{N}t{M}v-ep{E}-es{P}` — not alwaysApply |
| **Related**      | scripts-execution |

### JRN-20260723-02: Do not fork artifact dirs on hyperparam reruns

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-23 22:55:36 |
| **Category**     | mistake |
| **Summary**      | On “increase epochs and rerun,” created `_ep500` model/callback/log dirs instead of reusing the existing 50t/50v paths. |
| **Artifact**     | `.cursor/rules/scripts-execution.mdc` § Rerun path hygiene (scoped) |
| **Action taken** | Scoped rule: hyperparam-only reruns keep same `model_output_dir` / `callback_root_dir` / log basename unless user asks to preserve or fork — not alwaysApply |
| **Related**      | scripts-execution |

### JRN-20260723-01: Start TensorBoard with training

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-23 22:43:34 |
| **Category**     | mistake |
| **Summary**      | Started AR 50t/50v train without TensorBoard; only launched TB after the user asked mid-run. Live metrics should be available for the whole job. |
| **Artifact**     | `.cursor/rules/scripts-execution.mdc` (scoped); pointer in `.cursor/skills/wsl-gpu-stepcovnet/SKILL.md`; catalog `docs/agents/agent-brain.md` |
| **Action taken** | Scoped rule: start `tensorboard.exe` before/with training, point `--logdir` at `callback_root_dir/logs`, give URL when train starts — not alwaysApply |
| **Related**      | scripts-execution, wsl-gpu-stepcovnet |

### JRN-20260716-01: Training output must remain live

| Field            | Value |
| ---------------- | ----- |
| **Timestamp**    | 2026-07-16 01:22:37 |
| **Category**     | mistake |
| **Summary**      | Agent launched benchmark training with stdout/stderr redirected only to log files, hiding live progress despite the established visible-console rule. |
| **Artifact**     | `.cursor/rules/scripts-execution.mdc` (scoped rule for `scripts/**`); `.cursor/skills/README.md`; `docs/agents/agent-brain.md` |
| **Action taken** | Explicitly bans log-only redirection for long jobs and provides the required PowerShell console-plus-log `Tee-Object` template, including backgrounded jobs; quick refresh also cataloged the pre-existing deprecated skill alias. |
| **Related**      | [steering-correction-promotion](../../.cursor/skills/steering-correction-promotion/SKILL.md), [wsl-gpu-stepcovnet](../../.cursor/skills/wsl-gpu-stepcovnet/SKILL.md) |

### JRN-20260703-02: Prefer create_autospec and patch autospec in tests

| Field            | Value                                                                 |
| ---------------- | --------------------------------------------------------------------- |
| **Timestamp**    | 2026-07-03                                                            |
| **Category**     | convention                                                            |
| **Summary**      | User wants `create_autospec` (not bare `MagicMock`) and `patch.object(..., autospec=True)` as the default mock pattern; early-stop test showed bool class attrs on autospec still behave like truthy mocks. |
| **Artifact**     | `.cursor/rules/python-tests.mdc` (scoped `tests/**`); `tests/onset_ar/trainers_test.py`, `tests/test_wsl_gpu_lock.py`, `tests/ar_tide_iter_training_lock_test.py`, `tests/extract_mert_features_test.py` |
| **Action taken** | Rule: real minimal instance > create_autospec(live instance) > MagicMock; require `autospec=True` on callable patches. Exemplar fixes in GPU-lock and AR trainer tests. |
| **Related**      | [steering-correction-promotion](../../.cursor/skills/steering-correction-promotion/SKILL.md) |

### JRN-20260630-03: Repo-root `_tmp_*` scratch + same-turn self-improvement

| Field            | Value                                                                 |
| ---------------- | --------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-30                                                            |
| **Category**     | mistake                                                               |
| **Summary**      | Agent wrote `_tmp_analyze_train.py` at repo root; user asked why not in `_tmp/`; agent explained but did **not** run steering-correction-promotion until user asked why self-improvement was skipped. File was left on disk after a claimed delete. |
| **Artifact**     | `.cursor/rules/state-and-paths.mdc` (alwaysApply — `_tmp/<topic>/`, ban root `_tmp_*`); `.cursor/rules/scripts-execution.mdc` (aligned ban) |
| **Action taken** | Every turn: disposable scratch under `_tmp/<topic>/` only; on process steering, artifact + JRN same turn (not chat-only). |
| **Related**      | [agent-self-improvement](../../.cursor/skills/agent-self-improvement/SKILL.md), [steering-correction-promotion](../../.cursor/skills/steering-correction-promotion/SKILL.md) |

### JRN-20260630-02: Autoresearch must consume full time budget

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-30                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: when a wall-clock budget is given (e.g. 7 h), run until goal or deadline — not stop when a finite pre-written queue empties (~1 h). |
| **Artifact**     | `.cursor/skills/autoresearch/SKILL.md` (§ Budget discipline), `profiles/ar-tide-overfit.md`, `docs/agents/ar-tide-overnight-prompt.md`, `scripts/ar_tide_iter/README.md` |
| **Action taken** | Mandate `deadline = now + budget`; replan after each run; forbid finite-queue early exit; ~40–50 tide runs per 7 h throughput estimate. |
| **Related**      | steering-correction-promotion, autoresearch, ar-tide-overfit                                                                          |

### JRN-20260630-01: Clean up commit scope before committing

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-30                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: before creating commits, clean up the change set — not “run pytest first”. Exclude local iter logs/registry unless asked.         |
| **Artifact**     | `AGENTS.md` (Before commit / push)                                                                                                    |
| **Action taken** | Agent reviews status/diff, reverts unrelated files, fixes lint, syncs docs — then commits only the intended harness/doc diff.         |
| **Related**      | steering-correction-promotion                                                                                                         |

### JRN-20260628-07: Audit script read-only; no alwaysApply budget

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: no script writes to agent brain docs; model updates catalogs; no hard alwaysApply cap but stay mindful of always-on rules.       |
| **Artifact**     | `scripts/audit_agent_brain.py`, `tests/audit_agent_brain_test.py`, `agent-brain-refresh/SKILL.md`, `steering-correction-promotion/SKILL.md` |
| **Action taken** | Script prints disk inventory + drift only; agent maintains agent-brain.md; alwaysApply budget removed from audit.                      |
| **Related**      | JRN-20260628-06, agent-brain-refresh                                                                                                  |

### JRN-20260628-06: Agent brain refresh skill and canonical catalog

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User wants holistic agent-brain maintenance; AGENTS.md wrongly listed scoped rules as always-on; periodic/user-triggered refresh.      |
| **Artifact**     | `.cursor/skills/agent-brain-refresh/SKILL.md`, `scripts/audit_agent_brain.py`, `docs/agents/agent-brain.md`; slimmed `AGENTS.md`      |
| **Action taken** | Catalog regenerated from disk; alwaysApply table removed from AGENTS.md; refresh after promotion + session end + user phrase.         |
| **Related**      | steering-correction-promotion, JRN-20260628-05                                                                                        |

### JRN-20260628-05: Context-efficient agent brain — promotion skill

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: alwaysApply rules for every steering correction wastes context; optimize agent brain on each correction via smallest durable layer. |
| **Artifact**     | `.cursor/skills/steering-correction-promotion/SKILL.md`, `.cursor/rules/scripts-execution.mdc` (scoped); demoted `long-running-console`, `temp-artifacts` |
| **Action taken** | Decision tree favors skills/scoped rules; AGENTS.md slimmed; promotion skill runs optimization checklist each correction.               |
| **Related**      | agent-self-improvement, JRN-20260628-04                                                                                               |

### JRN-20260628-08: WSL GPU — one job at a time

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: only one GPU training job at a time; do not run GPU training and GPU inference in separate processes concurrently.             |
| **Artifact**     | `.cursor/skills/wsl-gpu-stepcovnet/SKILL.md` (§ GPU scheduling); `.cursor/rules/scripts-execution.mdc`                                |
| **Action taken** | Agent checks active terminals before launching WSL GPU; train and infer (decode/eval) run sequentially, not in parallel.            |
| **Related**      | wsl-gpu-stepcovnet, python-environment                                                                                                |

### JRN-20260628-07: Log full run series; check before re-run suggestions

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User: “Remember that” — agent logged run2 but left v3/v4 offline decode as vague notes, asked whether to log, and suggested re-running v4 decode that already existed in `logs/ar_perfect_v4_decode.log`. |
| **Artifact**     | `.cursor/skills/research-session-workflow/SKILL.md` (complete the series, grep before re-run); `.cursor/rules/research-logging.mdc`; `docs/research/EXPERIMENT_LOG.md` (run3/v4 numbers) |
| **Action taken** | Same-turn EXP updates for every run in a thread; read existing `logs/*decode*` before proposing jobs; never ask to log.               |
| **Related**      | JRN-20260628-05, research-session-workflow                                                                                            |

### JRN-20260628-06: Tide overfit free-run bar = 1.0

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User corrected free-run gate from 0.95 to 1.0 — single-chart overfit must reproduce all onsets exactly.                               |
| **Artifact**     | `docs/research/AR_ONSET_DESIGN.md` (§10.1, §10.6, `overfit-free-run-f1`); `DECISIONS_CHECKLIST.md`; `DISCUSSION_NOTES.md` NOTE-20260628-02; `EXPERIMENT_LOG.md` EXP-20260628-02 conclusion |
| **Action taken** | Locked **1.0** for tide free-run F1; reclassified run1/run2 as gate fail under corrected bar.                                         |
| **Related**      | NOTE-20260628-02, gate-ar-decode                                                                                                      |

### JRN-20260628-05: Auto-log EXP after offline eval — no ask

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User corrected agent for offering to log offline decode numbers in EXPERIMENT_LOG instead of doing it automatically in the same turn. |
| **Artifact**     | `.cursor/skills/research-session-workflow/SKILL.md` (§ Log experiment — same turn, no ask); `docs/research/EXPERIMENT_LOG.md` (EXP-20260628-02 run2 offline numbers) |
| **Action taken** | Measurable runs and offline evals → prepend/update EXP immediately; never prompt user to approve logging.                             |
| **Related**      | research-logging.mdc, JRN-20260628-04                                                                                                 |

### JRN-20260628-04: Systematic temp file handling

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Agents left `tmp_*` at repo root from command redirects; user asked for a systematic policy for any temp output.                      |
| **Artifact**     | `.cursor/rules/temp-artifacts.mdc` (later merged into `scripts-execution.mdc`), `.gitignore`, `AGENTS.md`, `docs/agents/project-layout.md` |
| **Action taken** | Captures go to `logs/` or `_tmp/` (gitignored); delete `_tmp` when done; never commit or use repo-root `tmp_*`.                       |
| **Related**      | long-running-console, JRN-20260628-01, temp-artifacts                                                                               |

### JRN-20260628-03: Same-turn promotion — artifact before journal

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Journal entries are useless without durable promotion; user wants artifact (rule/skill/code) created first, JRN second, same turn.   |
| **Artifact**     | `.cursor/skills/agent-self-improvement/SKILL.md`, `docs/agents/self-journal.md`, `.cursor/rules/agents-entry.mdc`                     |
| **Action taken** | Enforces artifact-first workflow; JRN requires **Artifact** path; journal-only entries forbidden except explicit user deferral.      |
| **Related**      | JRN-20260628-02, agent-self-improvement                                                                                               |

### JRN-20260628-02: Journal on steering corrections, not only long sessions

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28                                                                                                                            |
| **Category**     | convention                                                                                                                            |
| **Summary**      | User wants self-journal updated periodically and **immediately** on steering corrections (how the agent decides/operates), not batched only after long sessions. |
| **Artifact**     | `.cursor/skills/agent-self-improvement/SKILL.md`, `docs/agents/self-journal.md`, `.cursor/rules/agents-entry.mdc`                     |
| **Action taken** | Journal on steering corrections in same turn; periodic journaling during work.                                                        |
| **Related**      | agent-self-improvement, JRN-20260628-01                                                                                               |

### JRN-20260628-01: Training output hidden in log files

| Field            | Value                                                                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **Timestamp**    | 2026-06-28 12:00:00                                                                                                                   |
| **Category**     | convention                                                                                                                            |
| **Summary**      | Agent ran WSL GPU training/decode with `*> logs/...` or `Tee-Object`, so the user could not watch epoch progress or errors live.      |
| **Artifact**     | `.cursor/rules/long-running-console.mdc`, `.cursor/skills/agent-self-improvement/SKILL.md`, `AGENTS.md`                               |
| **Action taken** | Long-running jobs must stream to visible terminal; optional log file via tee; remember-this trigger documented.                         |
| **Related**      | agent-self-improvement, wsl-gpu-stepcovnet                                                                                            |

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
