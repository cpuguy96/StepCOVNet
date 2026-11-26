#!/usr/bin/env bash
# Manage isolated git worktrees for parallel code-changing experiment chains.
#
# Usage:
#   bash scripts/colab_worktree.sh create <name> [base_ref]
#   bash scripts/colab_worktree.sh list
#   bash scripts/colab_worktree.sh remove <name> [--delete-branch]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PARENT_DIR="$(cd "$REPO_ROOT/.." && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"

find_colab() {
  if command -v colab >/dev/null 2>&1; then
    command -v colab
  elif [[ -x "$HOME/.local/bin/colab" ]]; then
    echo "$HOME/.local/bin/colab"
  else
    echo ""
  fi
}

usage() {
  cat <<'EOF'
Usage: bash scripts/colab_worktree.sh <command> [args]

Commands:
  create <name> [base_ref]
      Create an isolated worktree at ../<repo>-exp-<name> on branch exp/<name>
      (defaults to HEAD). Ensures colab_*.sh helpers are present even if not yet
      committed on base_ref.

  list
      Show active git worktrees and active Colab GPU sessions.

  remove <name> [--delete-branch]
      Stop any Colab session named exp-<name>, remove ../<repo>-exp-<name>,
      and optionally delete branch exp/<name>.
EOF
}

cmd_create() {
  local name="${1:-}"
  local base_ref="${2:-HEAD}"
  if [[ -z "$name" ]]; then
    echo "ERROR: provide an experiment chain name (e.g. flux, loss, arch)." >&2
    exit 1
  fi
  if [[ ! "$name" =~ ^[a-zA-Z0-9._-]+$ ]]; then
    echo "ERROR: name must match ^[a-zA-Z0-9._-]+$ (got: $name)" >&2
    exit 1
  fi

  local wt_dir="${PARENT_DIR}/${REPO_NAME}-exp-${name}"
  local branch="exp/${name}"

  if [[ -e "$wt_dir" ]]; then
    echo "ERROR: path already exists: $wt_dir" >&2
    exit 1
  fi

  if git -C "$REPO_ROOT" show-ref --verify --quiet "refs/heads/${branch}"; then
    echo "Checking out existing branch '${branch}' into ${wt_dir} ..."
    git -C "$REPO_ROOT" worktree add "$wt_dir" "$branch"
  else
    echo "Creating branch '${branch}' from '${base_ref}' into ${wt_dir} ..."
    git -C "$REPO_ROOT" worktree add -b "$branch" "$wt_dir" "$base_ref"
  fi

  # Ensure Colab helpers exist in the new worktree even before they are committed to base_ref
  for helper in colab_sync.sh colab_run_remote.sh colab_worktree.sh; do
    if [[ ! -f "$wt_dir/scripts/$helper" && -f "$REPO_ROOT/scripts/$helper" ]]; then
      cp "$REPO_ROOT/scripts/$helper" "$wt_dir/scripts/$helper"
      chmod +x "$wt_dir/scripts/$helper"
    fi
  done

  local safe_id="${name//[^a-zA-Z0-9]/_}"
  cat <<EOF

Worktree ready:
  Path:         ${wt_dir}
  Branch:       ${branch}
  Colab flags:  --session exp-${name} --state-config /tmp/colab_exp_${safe_id}.json --run-name exp_${safe_id}_v1

Tip: pass --keep on v1 and --reuse-session on v2+ to keep the 6 GiB dataset warm on that Colab GPU.
EOF
}

cmd_list() {
  echo "=== Git Worktrees ==="
  git -C "$REPO_ROOT" worktree list
  local colab_bin
  colab_bin="$(find_colab)"
  if [[ -n "$colab_bin" ]]; then
    echo ""
    echo "=== Active Colab Sessions ==="
    "$colab_bin" sessions || true
  fi
}

cmd_remove() {
  local name="${1:-}"
  shift || true
  if [[ -z "$name" ]]; then
    echo "ERROR: provide the experiment chain name to remove." >&2
    exit 1
  fi
  local delete_branch="0"
  for arg in "$@"; do
    if [[ "$arg" == "--delete-branch" ]]; then
      delete_branch="1"
    else
      echo "ERROR: unknown flag '$arg'" >&2
      exit 1
    fi
  done

  local wt_dir="${PARENT_DIR}/${REPO_NAME}-exp-${name}"
  local branch="exp/${name}"
  local safe_id="${name//[^a-zA-Z0-9]/_}"
  local state_cfg="/tmp/colab_exp_${safe_id}.json"

  local colab_bin
  colab_bin="$(find_colab)"
  if [[ -n "$colab_bin" ]]; then
    if [[ -f "$state_cfg" ]]; then
      "$colab_bin" --config "$state_cfg" stop -s "exp-${name}" >/dev/null 2>&1 || true
      rm -f "$state_cfg"
    else
      "$colab_bin" stop -s "exp-${name}" >/dev/null 2>&1 || true
    fi
  fi

  if [[ -d "$wt_dir" ]]; then
    git -C "$REPO_ROOT" worktree remove "$wt_dir"
    echo "Removed worktree: ${wt_dir}"
  else
    echo "Worktree directory not found (skipping): ${wt_dir}"
  fi

  if [[ "$delete_branch" == "1" ]]; then
    git -C "$REPO_ROOT" branch -D "$branch"
    echo "Deleted branch: ${branch}"
  fi
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    create) cmd_create "$@" ;;
    list) cmd_list "$@" ;;
    remove) cmd_remove "$@" ;;
    help|-h|--help) usage ;;
    *)
      echo "Unknown command: $cmd" >&2
      usage
      exit 1
      ;;
  esac
}

main "$@"
