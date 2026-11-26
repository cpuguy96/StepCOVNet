#!/usr/bin/env bash
# CLI-only Google Drive sync for StepCOVNet + Google Colab (zero GitHub pushes).
#
# Drive layout (default remote: gdrive, root: StepCOVNet):
#   gdrive:StepCOVNet/
#     code/             <- synced local src/, scripts/, configs/, pyproject.toml
#     notebooks/        <- colab_runner.ipynb (open in Google Colab)
#     data_archives/    <- one-time .tar archives of data/<dataset>
#     data_patches/     <- incremental overlays (e.g. newly extracted *.mert_*.npy)
#     runs/             <- Colab training outputs (logs, metrics, checkpoints)
#
# Local pulled runs land in: experiments/colab/ (gitignored).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

REMOTE="${COLAB_RCLONE_REMOTE:-gdrive}"
DRIVE_ROOT="${COLAB_DRIVE_ROOT:-StepCOVNet}"
LOCAL_RUNS_DIR="${COLAB_LOCAL_RUNS_DIR:-$REPO_ROOT/experiments/colab}"
REMOTE_BASE="${REMOTE}:${DRIVE_ROOT}"

find_rclone() {
  if command -v rclone >/dev/null 2>&1; then
    command -v rclone
  elif [[ -x "$HOME/.local/bin/rclone" ]]; then
    echo "$HOME/.local/bin/rclone"
  else
    echo "ERROR: rclone not found in PATH or ~/.local/bin/rclone" >&2
    exit 1
  fi
}

RCLONE="$(find_rclone)"

ensure_remote() {
  if ! "$RCLONE" listremotes | grep -qx "${REMOTE}:"; then
    echo "ERROR: rclone remote '${REMOTE}:' is not configured yet." >&2
    echo "Run once in your terminal:  $RCLONE config" >&2
    echo "  -> New remote -> name: ${REMOTE} -> Storage: drive -> follow browser OAuth." >&2
    exit 1
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/colab_sync.sh <command> [args]

Commands:
  check
      Verify rclone and Google Drive remote connectivity.

  push-code
      Sync local pyproject.toml, src/, scripts/, configs/ (~4 MB) to Drive
      and update notebooks/colab_runner.ipynb. Does NOT touch GitHub.

  push-data <dataset> [--exclude-mert | --meta-only]
      Stream data/<dataset> as a .tar archive directly into Drive
      (e.g. literature_fraxtil_orig, literature_fraxtil_exp, final_data, v2).
      Pass --exclude-mert to skip *.mert*.npy caches (upload those separately with push-data-glob).
      Pass --meta-only to upload a tiny <dataset>__meta.tar excluding raw audio and *.npy caches
      (keeps only training_index*.json, *.chart.json, *.sm/*.ssc for MERT runs).

  push-data-glob <dataset> <archive_name> <glob>
      Stream only files matching <glob> under data/<dataset> into
      data_archives/<archive_name>.tar (great for uploading one MERT layer cache at a time).

  push-data-patch <dataset> <glob>
      Incrementally upload only matching files under data/<dataset>
      (e.g. bash scripts/colab_sync.sh push-data-patch literature_fraxtil_orig '*.mert_l2_6_10_16.npy').

  list-runs
      List experiment runs currently stored on Drive.

  pull-runs [run_name] [--logs-only]
      Pull Colab run outputs into experiments/colab/.
      Pass --logs-only to fetch only logs/metrics/JSONs (skips model weights).

  status
      Show code manifest, uploaded datasets, patches, and runs on Drive.

Environment overrides:
  COLAB_RCLONE_REMOTE  (default: gdrive)
  COLAB_DRIVE_ROOT     (default: StepCOVNet)
  COLAB_LOCAL_RUNS_DIR (default: experiments/colab)
EOF
}

cmd_check() {
  echo "rclone: $("$RCLONE" version | head -n 1)"
  ensure_remote
  "$RCLONE" about "${REMOTE}:"
  echo "OK: remote '${REMOTE}:' is ready (root: ${REMOTE_BASE})."
}

cmd_push_code() {
  ensure_remote
  mkdir -p "$REPO_ROOT/_tmp"
  local manifest="$REPO_ROOT/_tmp/colab_sync_manifest.txt"
  {
    echo "synced_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "branch=$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
    echo "head=$(git -C "$REPO_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
    echo "--- git status --short ---"
    git -C "$REPO_ROOT" status --short 2>/dev/null || true
  } > "$manifest"

  echo "Packaging local code (pyproject.toml, src/, scripts/, configs/) -> ${REMOTE_BASE}/code/code.tar.gz ..."
  tar -czf - -C "$REPO_ROOT" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='*.pyo' \
    --exclude='.DS_Store' \
    --exclude='*.egg-info' \
    pyproject.toml src scripts configs \
    | "$RCLONE" rcat "${REMOTE_BASE}/code/code.tar.gz" --stats=5s --stats-one-line

  "$RCLONE" copyto "$manifest" "${REMOTE_BASE}/code/SYNC_MANIFEST.txt"

  if [[ -f "$REPO_ROOT/scripts/colab_runner.ipynb" ]]; then
    "$RCLONE" copyto "$REPO_ROOT/scripts/colab_runner.ipynb" "${REMOTE_BASE}/notebooks/colab_runner.ipynb"
  fi
  echo "Code sync complete."
  cat "$manifest"
}

cmd_push_data() {
  ensure_remote
  local dataset="${1:-}"
  shift || true
  if [[ -z "$dataset" ]]; then
    echo "ERROR: specify a dataset folder under data/, e.g.: literature_fraxtil_orig" >&2
    exit 1
  fi
  local exclude_mert="0"
  local meta_only="0"
  for arg in "$@"; do
    if [[ "$arg" == "--exclude-mert" ]]; then
      exclude_mert="1"
    elif [[ "$arg" == "--meta-only" ]]; then
      meta_only="1"
    else
      echo "ERROR: unexpected argument '$arg'" >&2
      exit 1
    fi
  done
  local src_dir="$REPO_ROOT/data/$dataset"
  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: directory not found: $src_dir" >&2
    exit 1
  fi
  local tar_args=()
  local archive_name="$dataset"
  if [[ "$meta_only" == "1" ]]; then
    archive_name="${dataset}__meta"
    tar_args+=(
      --exclude='*.ogg'
      --exclude='*.mp3'
      --exclude='*.wav'
      --exclude='*.flac'
      --exclude='*.ddc_mel.npy'
      --exclude='*.mert*.npy'
    )
  elif [[ "$exclude_mert" == "1" ]]; then
    tar_args+=(--exclude='*.mert*.npy')
  fi
  local target="${REMOTE_BASE}/data_archives/${archive_name}.tar"
  echo "Streaming tar of data/${dataset} (exclude_mert=${exclude_mert}, meta_only=${meta_only}) -> ${target} ..."
  tar -cf - -C "$REPO_ROOT/data" "${tar_args[@]}" "$dataset" \
    | "$RCLONE" rcat "$target" --stats=10s --stats-one-line
  echo "Dataset upload complete: ${target}"
}

cmd_push_data_glob() {
  ensure_remote
  local dataset="${1:-}"
  local archive_name="${2:-}"
  local pattern="${3:-}"
  if [[ -z "$dataset" || -z "$archive_name" || -z "$pattern" ]]; then
    echo "ERROR: usage: push-data-glob <dataset> <archive_name> <glob>" >&2
    exit 1
  fi
  local src_dir="$REPO_ROOT/data/$dataset"
  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: directory not found: $src_dir" >&2
    exit 1
  fi
  local list_file
  list_file="$(mktemp)"
  (cd "$REPO_ROOT/data" && find "$dataset" -type f -name "$pattern" | sort) > "$list_file"
  local count
  count="$(wc -l < "$list_file")"
  if [[ "$count" -eq 0 ]]; then
    rm -f "$list_file"
    echo "ERROR: no files matching '${pattern}' under data/${dataset}" >&2
    exit 1
  fi
  local target="${REMOTE_BASE}/data_archives/${archive_name}.tar"
  echo "Streaming ${count} files matching '${pattern}' from data/${dataset} -> ${target} ..."
  tar -cf - -C "$REPO_ROOT/data" -T "$list_file" \
    | "$RCLONE" rcat "$target" --stats=10s --stats-one-line
  rm -f "$list_file"
  echo "Archive upload complete: ${target}"
}

cmd_push_data_patch() {
  ensure_remote
  local dataset="${1:-}"
  local pattern="${2:-}"
  if [[ -z "$dataset" || -z "$pattern" ]]; then
    echo "ERROR: usage: push-data-patch <dataset> <glob>" >&2
    exit 1
  fi
  local src_dir="$REPO_ROOT/data/$dataset"
  if [[ ! -d "$src_dir" ]]; then
    echo "ERROR: directory not found: $src_dir" >&2
    exit 1
  fi
  local target="${REMOTE_BASE}/data_patches/${dataset}"
  echo "Copying matching '${pattern}' from data/${dataset} -> ${target} ..."
  "$RCLONE" copy "$src_dir" "$target" \
    --include "$pattern" \
    --transfers=8 \
    --stats=10s \
    --stats-one-line
  echo "Patch upload complete: ${target}"
}

cmd_list_runs() {
  ensure_remote
  "$RCLONE" lsd "${REMOTE_BASE}/runs" 2>/dev/null || echo "(no runs directory yet)"
}

cmd_pull_runs() {
  ensure_remote
  local run_name=""
  local logs_only="0"
  for arg in "$@"; do
    if [[ "$arg" == "--logs-only" ]]; then
      logs_only="1"
    elif [[ -z "$run_name" ]]; then
      run_name="$arg"
    else
      echo "ERROR: unexpected argument '$arg'" >&2
      exit 1
    fi
  done

  local src="${REMOTE_BASE}/runs"
  local dst="$LOCAL_RUNS_DIR"
  if [[ -n "$run_name" ]]; then
    src="${src}/${run_name}"
    dst="${dst}/${run_name}"
  fi
  mkdir -p "$dst"

  local extra_flags=()
  if [[ "$logs_only" == "1" ]]; then
    extra_flags+=(
      --include '*.log'
      --include '*.txt'
      --include '*.json'
      --include '*.csv'
      --include '*.md'
      --exclude '*'
    )
    echo "Pulling logs/metrics only: ${src} -> ${dst} ..."
  else
    echo "Pulling full run outputs: ${src} -> ${dst} ..."
  fi

  "$RCLONE" copy "$src" "$dst" \
    "${extra_flags[@]}" \
    --transfers=8 \
    --fast-list \
    --stats=5s \
    --stats-one-line
  echo "Pull complete -> ${dst}"
}

cmd_status() {
  ensure_remote
  echo "=== Code Manifest ==="
  "$RCLONE" cat "${REMOTE_BASE}/code/SYNC_MANIFEST.txt" 2>/dev/null || echo "(not pushed yet)"
  echo ""
  echo "=== Data Archives ==="
  "$RCLONE" ls "${REMOTE_BASE}/data_archives" 2>/dev/null || echo "(none)"
  echo ""
  echo "=== Data Patches ==="
  "$RCLONE" lsd "${REMOTE_BASE}/data_patches" 2>/dev/null || echo "(none)"
  echo ""
  echo "=== Runs ==="
  "$RCLONE" lsd "${REMOTE_BASE}/runs" 2>/dev/null || echo "(none)"
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    check) cmd_check "$@" ;;
    push-code) cmd_push_code "$@" ;;
    push-data) cmd_push_data "$@" ;;
    push-data-glob) cmd_push_data_glob "$@" ;;
    push-data-patch) cmd_push_data_patch "$@" ;;
    list-runs) cmd_list_runs "$@" ;;
    pull-runs) cmd_pull_runs "$@" ;;
    status) cmd_status "$@" ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
