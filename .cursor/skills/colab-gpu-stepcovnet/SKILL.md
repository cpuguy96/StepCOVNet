---
name: colab-gpu-stepcovnet
description: Launches and manages StepCOVNet GPU training runs on Google Colab via google-colab-cli and private Google Drive (rclone), without pushing WIP code to GitHub. Use proactively whenever preparing parallel experiments, syncing datasets/code for Colab, or inspecting remote Colab results.
disable-model-invocation: true
---

# Colab GPU (StepCOVNet)

Run StepCOVNet training and eval directly on Google Colab GPUs (`T4`, `L4`, `A100`, `H100`) from WSL via `google-colab-cli` (`~/.local/bin/colab`) + private Google Drive (`~/.local/bin/rclone`, remote `gdrive:StepCOVNet`).

## Policies

- **GPU selection order (strict default):** **`L4 → T4 → local (RTX 2080) → A100`**. Prefer Colab `L4` first (then `T4`) so the user's local RTX 2080 stays free for other work. Only fall back to local GPU when both Colab `L4` and `T4` slots are occupied or unavailable, or when the user explicitly requests local; reserve `A100` for heavy runs that exceed `L4` speed/VRAM.
- **Launch autonomously (do not hand commands back):** Whenever `colab` (`~/.local/bin/colab`) and `rclone` (`~/.local/bin/rclone`) are present in the environment, sync datasets/code and launch Colab jobs directly yourself via `scripts/colab_run_remote.sh` without waiting to be asked or asking the user to run the command.
- **Never push WIP code to GitHub for Colab.** Code is uploaded directly from the local working tree via `colab upload` (or mirrored privately to `gdrive:StepCOVNet/code/code.tar.gz`).
- **Keep credentials local:** `rclone`'s `client_secret` and `refresh_token` stay in `~/.config/rclone/rclone.conf` (`chmod 600`); `colab_run_remote.sh` only passes a short-lived `access_token` in memory during dataset streaming and immediately drops it.
- **Always stop idle VMs:** `colab_run_remote.sh` stops the Colab session automatically on exit unless `--keep` is passed. Verify no leaked sessions with `~/.local/bin/colab sessions` and check quota with `~/.local/bin/colab usage`.
- **Low-verbosity logs & live TensorBoard:** `colab_run_remote.sh` filters per-step progress-bar spam (keeping only epoch summaries, checkpoint/eval lines, and errors) and syncs TensorBoard `events.out.tfevents.*` every ~60s into `experiments/colab/<run_name>/tb/`. Always start a local TensorBoard daemon on a free port (`6008+`) comparing the live Colab `tb/` folder against the baseline run.

## Prerequisites (already installed in WSL)

| Tool | Path | Verify |
| ---- | ---- | ------ |
| `colab` (`google-colab-cli`) | `~/.local/bin/colab` (venv `~/.local/share/colab-cli-venv`, includes `keyrings.alt`) | `~/.local/bin/colab whoami` · `~/.local/bin/colab usage` |
| `rclone` | `~/.local/bin/rclone` (remote `gdrive:StepCOVNet`) | `bash scripts/colab_sync.sh check` |

## Dataset archives on Google Drive

Store base datasets and MERT feature caches as separate `.tar` archives under `gdrive:StepCOVNet/data_archives/` so Colab only streams and unpacks what a run needs (at ~75–100 MiB/s inside Google's network):

```bash
# Check what is already uploaded
bash scripts/colab_sync.sh status

# Upload base dataset once (excluding *.mert*.npy)
bash scripts/colab_sync.sh push-data literature_fraxtil_orig --exclude-mert

# Upload a specific MERT feature cache once
bash scripts/colab_sync.sh push-data-glob \
  literature_fraxtil_orig \
  literature_fraxtil_orig__mert_l2_6_10_16 \
  '*.mert_l2_6_10_16.npy'
```

## Launching a Colab GPU run (automated, no browser needed)

```bash
bash scripts/colab_run_remote.sh \
  --session exp1 \
  --gpu L4 \
  --run-name itgpt_mert_fraxtil_orig_cosine64_l2_6_10_16_ema \
  --script scripts/train_itgpt_placement.py \
  --config configs/ddc/itgpt_mert_placement_fraxtil_orig_cosine64_l2_6_10_16_ema.json \
  --datasets literature_fraxtil_orig,literature_fraxtil_orig__mert_l2_6_10_16 \
  --eval-cmd "python scripts/eval_itgpt_placement.py --config={CONFIG} --model_path={RUN_DIR}/best.pt --diagnose_grounding --output={RUN_DIR}/eval_val_grounding_best.json"
```

Useful flags:
- `--gpu <L4|T4|A100>` — accelerator tier (default `L4`; follow `L4 → T4 → local → A100` priority).
- `--extra-args "..."` — forwarded verbatim to the training script (e.g. `"--epochs=2 --steps_per_epoch=10"`).
- `--archive-models <best|all|none>` — archives outputs directly from Colab to `MyDrive/StepCOVNet/runs/<run_name>/` before VM shutdown (default: `best`, saving `best*.pt` + all logs/JSONs; pull any good model down to `experiments/colab/<run_name>/` later via `bash scripts/colab_sync.sh pull-runs <run_name>`).
- `--download-all` — also download model checkpoints directly to `experiments/colab/<run_name>/` immediately (default downloads only lightweight `*.log`, `*.txt`, `*.json`, `*.csv`, `*.md` locally while keeping checkpoints in Drive).
- `--keep` / `--reuse-session` — keep a VM alive across back-to-back runs on the same unpacked dataset (remember to run `~/.local/bin/colab stop -s <session>` afterward).

## Parallel multi-GPU code chains (Priority: L4 → T4 → Local → A100)

- **Verified concurrency limits on this account:** Colab caps at **1 concurrent session per GPU type** (a 2nd `T4` or 2nd `L4` is refused with `precondition failed`), **but different GPU types stack concurrently**:
  - **Priority 1 (Colab):** `--gpu L4` (24 GiB, `2.33 CU/hr`, native `bf16`)
  - **Priority 2 (Colab):** `--gpu T4` (15 GiB, `1.24 CU/hr`, `fp16`)
  - **Priority 3 (local):** WSL RTX 2080 (8 GiB, free — leave free for user unless `L4` + `T4` are both busy or local is requested)
  - **Priority 4 (Colab):** `--gpu A100` (for heavy runs exceeding `L4`; `H100` is not entitled on this account)
  - Total parallel chains: up to **3 concurrent chains** (`L4 + T4` = `3.57 CU/hr`, plus local RTX 2080 only when needed).
- For **parallel code-changing chains**, isolate each branch in its own `git worktree` (never let two sessions edit the same working directory):
  ```bash
  bash scripts/colab_worktree.sh create <name>        # creates ../StepCOVNet-exp-<name> on exp/<name>
  bash scripts/colab_worktree.sh list                 # shows worktrees + active Colab sessions
  bash scripts/colab_worktree.sh remove <name> --delete-branch
  ```
- Isolate each concurrent Colab session with a unique `--session`, `--run-name`, and `--state-config /tmp/colab_exp_<name>.json`. Pass `--keep` on iteration 1 and `--reuse-session` on follow-up iterations to reuse the unpacked dataset on that VM (always stop when the chain finishes).
- `colab_run_remote.sh` uploads each worktree's own code directly via `colab upload`, so parallel branches never overwrite each other's code.

## Optional Drive code sync & notebook fallback

- `bash scripts/colab_sync.sh push-code` mirrors `pyproject.toml`, `src/`, `scripts/`, `configs/` to `gdrive:StepCOVNet/code/code.tar.gz` and updates `gdrive:StepCOVNet/notebooks/colab_runner.ipynb`.
- `bash scripts/colab_sync.sh pull-runs --logs-only` pulls notebook-based Drive runs into `experiments/colab/`.
