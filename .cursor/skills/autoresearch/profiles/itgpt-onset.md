# Profile: itgpt-onset

Autonomous hill-climbing loop for **PyTorch ITGPT-MERT onset placement** (`M-slot48` + decoded ±20 ms event F1 + audio grounding + error-breakdown diagnostics).

Use when optimizing onset detection via config tweaks (Tier A), architecture/data-view changes (Tier B), or new model/loss/feature code (Tier C) across Colab GPUs (`L4`, `T4`) and local WSL (`RTX 2080`).

## Success criteria & pinned gates

**Primary metric:** Pooled `M-slot48` **`f1_at_05`** on `best.pt` (also report `f1_max` and decoded ±20 ms `micro_event_f1`).

**Mandatory guardrails (every promoted run must pass all three):**

| Gate | Requirement | Why |
| ---- | ----------- | --- |
| **Audio grounding (shuffle)** | `slot_f1_drop_shuffle >= 0.08` (healthy: `~0.14+`) | Blocks audio-blind rhythmic-prior shortcuts |
| **Audio grounding (zeros)** | `slot_f1_drop_zeros >= 0.10` (healthy: `~0.15–0.73`) | Confirms audio conditioning is active |
| **Slice audit (`eval_val_breakdown.json`)** | No unmotivated catastrophic slice collapse (e.g. `intro` drop `> 3 pp` or `16th_off` drop `> 1.5 pp`) | Catches regressions hidden behind aggregate F1 (like `max_bar_shift=15`) |

Do **not** change evaluation splits, thresholds, or grounding criteria mid-session.

## Two-stage promotion funnel (Screen → Transfer)

Never crown a new champion solely from Dataset A's 9 validation songs.

1. **Stage 1 — Fast Screen (Dataset A, `81/9`, `literature_fraxtil_orig`):**
   - Full 64-epoch run on Colab `L4` (`~13.5 s/ep`, ~16 min) or `T4` (`~17.3 s/ep`, ~20 min).
   - For high-risk Tier-C architecture changes, an optional **32-epoch early screen** (`--epochs 32`, ~8 min on `L4`) is allowed because `best.pt` typically peaks around epochs 17–26; if clearly behind champion by `> 1.5 pp`, discard immediately, otherwise run full 64 epochs.
   - Promotion threshold from Stage 1: **`f1_at_05 > champion + 0.0020`** (`+0.20 pp`), OR a targeted error-slice win with non-negative overall `f1_at_05` and clean guardrails.
2. **Stage 2 — Transfer Promotion Gate (`final_data` `300t/50v`):**
   - Run the Stage-1 winner against the `300t/50v` champion using `final_data_300t_50v__meta,final_data_300t_50v__mert_l2_6_10_16` (~20–25 min on Colab `L4`).
   - Only after Stage 2 beats (or matches with clear diagnostic improvement) the `300t/50v` champion does the recipe/code replace the champion on `master`.

## Preflight

1. **Scoreboard & phase check:**
   ```bash
   python scripts/itgpt_scoreboard.py --filter fraxtil_orig
   python scripts/itgpt_scoreboard.py --filter 300t_50v
   ```
2. Read [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) § Current phase and check recent `EXP-…` entries so you never repeat a rejected knob.
3. Open the current champion's `eval_val_breakdown.json` (if missing, generate it once with `--diagnose_grounding --error_breakdown`) to identify the #1 error bucket (`16th_off`, `24th`, near-miss timing offsets, `intro`, calibration, worst charts).
4. Check active GPUs (`colab --auth=oauth2 sessions` and local `nvidia-smi`). Never run two training jobs on the same GPU.

## Multi-GPU parallel lanes & parallel coding

GPU priority: **`L4 → T4 → local (RTX 2080) → A100`**.

You may run up to **3 orthogonal hypothesis chains in parallel** (1 per GPU target: Colab `L4`, Colab `T4`, local WSL `RTX 2080`), AND **code hypothesis N+1 in parallel while GPUs train** — zero idle GPU time.

### Isolation rules (mandatory for parallel lanes & parallel coding)

1. **One worktree per coding/experiment lane:**
   ```bash
   bash scripts/colab_worktree.sh create <idea>
   ```
   Creates `../StepCOVNet-exp-<idea>` on branch `exp/<idea>` without touching the main working tree (`scripts/colab_run_remote.sh` tars `src/ scripts/ configs/` from its own worktree).
2. **Parallel coding via subagents:**
   - Spawn one coding subagent per orthogonal Tier-C hypothesis in its own isolated worktree (`Workspace="share"` or `../StepCOVNet-exp-<idea>`).
   - Each coding subagent owns: implementing the feature, adding/updating unit tests, running `pytest tests/itgpt/` + `python pre_submit.py --fast` inside its worktree, and reporting exact diff + test results.
   - **Never** let two parallel coding subagents edit the same working tree or `docs/research/EXPERIMENT_LOG.md` concurrently.
3. **Unique Colab session & state per lane:**
   - Pass unique `--session exp-<idea> --state-config /tmp/colab_exp_<idea>.json --run-name <run_name>` per lane.
   - Within a sequential lane on the same GPU type, pass `--keep` on run 1 and `--reuse-session` on runs 2..N to keep the streamed MERT archive warm on Colab SSD (~2 min saved per run).
   - Drop `--keep` on the final run of a lane and verify `colab --auth=oauth2 sessions` reports **0 active VMs** before ending the session.
4. **Serialized integration onto `master` (coordinator only):**
   - The coordinator collects results into `scripts/itgpt_scoreboard.py`, picks winners, and merges/cherry-picks **one** promoted branch onto `master` at a time, re-running `pre_submit.py --fast` + unit tests + updating `EXPERIMENT_LOG.md`.

### Swarm mode (`/swarm` — optional, for 3+ hour multi-wave sessions)

When the user explicitly launches `/swarm` for onset hill-climbing, enforce **fixed GPU-lane ownership** so workers never fight over Colab sessions or `EXPERIMENT_LOG.md`:

| Role | Workspace | Owns |
| ---- | --------- | ---- |
| **Coordinator** | main repo (`master`) | Hypothesis routing, `itgpt_scoreboard.py`, serialized cherry-picks/merges to `master`, `EXPERIMENT_LOG.md`, final `colab sessions` = 0 audit |
| **`lane-l4`** | `workspace:"share"` | Stage-1 & Stage-2 experiments exclusively on Colab `L4` (`--session exp-l4 --state-config /tmp/colab_l4.json --keep`) |
| **`lane-t4`** | `workspace:"share"` | Orthogonal Stage-1 experiments exclusively on Colab `T4` (`--session exp-t4 --state-config /tmp/colab_t4.json --keep`) |
| **`lane-local`** (optional) | `workspace:"share"` | Local WSL `RTX 2080` lane **or** pipelined Tier-C coding + unit tests for wave N+1 while `L4`/`T4` train |

- **Board reporting:** Every lane posts `F1@0.5`, `F1_max`, `±20ms`, `Δshuf`, `Δzero`, key breakdown slices (`16th_off`, `24th`, `intro`, `errors`), and branch/commit hash to the board so other lanes cross-pollinate immediately.
- **No direct `master` edits by workers:** Workers never push to `master` or edit `EXPERIMENT_LOG.md`.
- **Colab VM cleanup before worker shutdown:** Killing a swarm worker deletes its local workspace but **not** its remote Colab VM — each lane must drop `--keep` on its final run or explicitly `colab --auth=oauth2 --config /tmp/colab_<lane>.json stop -s exp-<lane>` before finishing.

## Orient

Run at the start of every iteration (takes ~1 second):

1. `python scripts/itgpt_scoreboard.py --filter fraxtil_orig` (note `Δpp`, `Δshuf`, `Δzero`, `16th`, `24th`, `Intro`, `Errs`, `Flags`, and plateau warning).
2. Inspect champion `eval_val_breakdown.json` — pick a hypothesis that directly attacks a measured bottleneck (e.g., `16th_off` F1, `24th` F1, `2-3 slot` near-miss FPs, `other_beat` FPs, or calibration bias).

## Plan artifact

- **Tier A / B (config-only):** New JSON config under `configs/ddc/itgpt_mert_*.json` derived from the current champion config; state one-sentence diagnosis + falsifiable prediction before launch.
- **Tier C (code change):** Branch `exp/<idea>` inside `../StepCOVNet-exp-<idea>` + targeted unit tests + `python pre_submit.py --fast` green **before** launching on GPU.

## Run

### Colab GPU (L4 / T4 — preferred)

Run from inside the hypothesis worktree (or main repo for Tier A config-only runs):

```bash
bash scripts/colab_run_remote.sh \
  --gpu L4 \
  --session exp-<idea> \
  --state-config /tmp/colab_exp_<idea>.json \
  --run-name <run_name> \
  --script scripts/train_itgpt_placement.py \
  --config configs/ddc/<config>.json \
  --datasets literature_fraxtil_orig__meta,literature_fraxtil_orig__mert_l2_6_10_16 \
  --eval-cmd "python scripts/eval_itgpt_placement.py --config {CONFIG} --model_path {RUN_DIR}/best.pt --output {RUN_DIR}/eval_val_grounding_best.json --diagnose_grounding --error_breakdown" \
  --max-poll-seconds 420 \
  --keep
```

- Single `--eval-cmd` with `--diagnose_grounding --error_breakdown` reuses the uncorrupted forward pass for both grounding and `eval_val_breakdown.json` (zero redundant inference) and auto-downloads both JSONs to `experiments/colab/<run_name>/`.
- Re-attach with `--attach` when exit code `75` fires.
- Stage-2 transfer datasets: `--datasets final_data_300t_50v__meta,final_data_300t_50v__mert_l2_6_10_16`.
- Live TensorBoard: cloud VMs auto-mirror to `MyDrive/StepCOVNet/live_tb/<run_name>/`; on local WSL, serve `experiments/colab/` on port `6008+`.

### Local WSL GPU (`RTX 2080`, third parallel lane)

```bash
source scripts/wsl_gpu_env.sh
python scripts/train_itgpt_placement.py --config configs/ddc/<config>.json
python scripts/eval_itgpt_placement.py \
  --config configs/ddc/<config>.json \
  --model_path models_wsl/ddc/<run_name>/best.pt \
  --diagnose_grounding \
  --error_breakdown
```

## Keep / Discard git discipline (Tier C)

Keep `master` permanently clean — **never accumulate dead feature flags in `src/`**:

- **Keep (Stage 1 + Stage 2 promoted):**
  1. Cherry-pick or merge `exp/<idea>` into `master`.
  2. Run `pytest tests/itgpt/` and `python pre_submit.py --fast` on `master`.
  3. Log `EXP-…` (status: **promoted**) in `docs/research/EXPERIMENT_LOG.md`.
  4. Remove the worktree: `bash scripts/colab_worktree.sh remove <idea> --delete-branch`.
- **Discard (rejected, regressed, or grounding-failed):**
  1. Log `EXP-…` (status: **rejected**) in `docs/research/EXPERIMENT_LOG.md` with exact diff summary, `F1@0.5`, grounding drops, and `eval_val_breakdown.json` cause so future loops never retry it.
  2. Delete the worktree and branch: `bash scripts/colab_worktree.sh remove <idea> --delete-branch`.

## Log

Follow [research-session-workflow](../../research-session-workflow/SKILL.md): prepend `EXP-…` to [EXPERIMENT_LOG.md](../../../../docs/research/EXPERIMENT_LOG.md) with `F1@0.5`, `F1_max`, `±20 ms micro_event_f1`, `Δshuf`, `Δzero`, key breakdown slices (`16th_off`, `24th`, `intro`, `errors`), and **promoted / rejected** verdict.

## Anti-spam & Do Not Repeat

- **Plateau rule:** `scripts/itgpt_scoreboard.py` raises a plateau warning after **5 consecutive timestamped runs** without `> +0.20 pp` `f1_at_05`. When plateaued, **forbid** single-knob hyperparameter nudges; upgrade to Tier B/C targeting the largest error bucket in `eval_val_breakdown.json`.
- **Already rejected on Dataset A (do NOT retry unchanged — see `EXPERIMENT_LOG.md`):**
  - `focal_gamma=1.0`, `early_film`, `grid_weight_24th=2.0`, `cyclic_measure_pos`, `continuous_beat_conv`, `gradient_accumulation_steps=4`, `mert_spectral_flux`, `max_bar_shift=15`.

## Gotchas

| Gotcha | What to do |
| ------ | ---------- |
| **Editing main repo during parallel Colab runs** | Always isolate Tier-C code edits in `scripts/colab_worktree.sh create <idea>` so `colab_run_remote.sh` tars the right code. |
| **Overfitting to Dataset A's 9 val songs** | Require Stage-2 `300t/50v` transfer pass before promoting any winner to champion. |
| **Leaked Colab VMs after `--keep`** | Drop `--keep` on the last run and verify `colab --auth=oauth2 sessions` shows 0 active VMs. |
| **Missing `eval_val_breakdown.json`** | Always include `--diagnose_grounding --error_breakdown` in `--eval-cmd`. |
| **Dead flags piling up on `master`** | Discard rejected Tier-C branches via `colab_worktree.sh remove <idea> --delete-branch` after logging `EXP-…`. |
