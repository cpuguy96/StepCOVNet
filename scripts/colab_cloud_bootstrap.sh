#!/usr/bin/env bash
# One-command bootstrap for google-colab-cli (Python 3.13) + rclone on a fresh Linux / cloud VM.
#
# Usage:
#   bash scripts/colab_cloud_bootstrap.sh [--rclone-conf PATH] [--rclone-token JSON] [--save-to-drive]
#
# Behavior:
#   1. Installs google-colab-cli + keyrings.alt into ~/.local/share/colab-cli-venv (~/.local/bin/colab) on Python 3.13.
#   2. Installs rclone static binary into ~/.local/bin/rclone.
#   3. Ensures ~/.ssh/id_ed25519 exists for `colab ssh`.
#   4. Configures ~/.config/rclone/rclone.conf (chmod 600) automatically via the first available source:
#        a) --rclone-conf PATH
#        b) /home/antigravity/upload/rclone.conf (moved & scrubbed from upload/)
#        c) --rclone-token '<json>' (from `rclone authorize "drive"`)
#        d) Private Google Drive auto-pull: MyDrive/StepCOVNet/.secrets/rclone.conf (via `gdrive` CLI)
#      Pass --save-to-drive with (a)/(b)/(c) to save/update MyDrive/StepCOVNet/.secrets/rclone.conf
#      so all future cloud VMs configure rclone automatically with zero clicks.
#   5. Verifies rclone and prints Colab auth status (or starts remote copy-paste OAuth if needed).

set -euo pipefail

RCLONE_CONF_SRC=""
RCLONE_TOKEN=""
SAVE_TO_DRIVE="0"
REMOTE="${COLAB_RCLONE_REMOTE:-gdrive}"
DRIVE_ROOT="${COLAB_DRIVE_ROOT:-StepCOVNet}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rclone-conf) RCLONE_CONF_SRC="$2"; shift 2 ;;
    --rclone-token) RCLONE_TOKEN="$2"; shift 2 ;;
    --save-to-drive) SAVE_TO_DRIVE="1"; shift ;;
    -h|--help)
      sed -n '2,18p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "$HOME/.local/bin" "$HOME/.local/share" "$HOME/.config/rclone"

# Helper: find MyDrive/<DRIVE_ROOT>/.secrets folder ID via gdrive CLI
find_drive_secrets_folder() {
  if ! command -v gdrive >/dev/null 2>&1; then
    return 1
  fi
  local root_id secrets_id
  root_id="$(gdrive readonly search --name-exact "$DRIVE_ROOT" --json 2>/dev/null \
    | jq -r '.[] | select(.mimeType=="application/vnd.google-apps.folder") | .id' | head -n 1 || true)"
  if [[ -z "$root_id" ]]; then
    return 1
  fi
  secrets_id="$(gdrive readonly ls "$root_id" --json 2>/dev/null \
    | jq -r '.[] | select(.name==".secrets" and .mimeType=="application/vnd.google-apps.folder") | .id' | head -n 1 || true)"
  if [[ -z "$secrets_id" ]]; then
    gdrive mutate mkdir ".secrets" --parent "$root_id" >/dev/null 2>&1 || return 1
    secrets_id="$(gdrive readonly ls "$root_id" --json 2>/dev/null \
      | jq -r '.[] | select(.name==".secrets" and .mimeType=="application/vnd.google-apps.folder") | .id' | head -n 1 || true)"
  fi
  [[ -n "$secrets_id" ]] || return 1
  echo "$secrets_id"
}

# 1. google-colab-cli on Python 3.13
if [[ ! -x "$HOME/.local/bin/colab" ]]; then
  echo "[bootstrap] Installing google-colab-cli + keyrings.alt (Python 3.13)..."
  uv venv "$HOME/.local/share/colab-cli-venv" --python 3.13
  uv pip install --python "$HOME/.local/share/colab-cli-venv/bin/python" google-colab-cli keyrings.alt
  ln -sf "$HOME/.local/share/colab-cli-venv/bin/colab" "$HOME/.local/bin/colab"
else
  echo "[bootstrap] colab already installed: $("$HOME/.local/bin/colab" version)"
fi

# 2. rclone static binary
if ! command -v rclone >/dev/null 2>&1 && [[ ! -x "$HOME/.local/bin/rclone" ]]; then
  echo "[bootstrap] Installing rclone into ~/.local/bin/rclone..."
  TMPD="$(mktemp -d)"
  curl -fsSL https://downloads.rclone.org/rclone-current-linux-amd64.zip -o "$TMPD/rclone.zip"
  unzip -q -o "$TMPD/rclone.zip" -d "$TMPD/ext"
  install -m 755 "$TMPD"/ext/rclone-*/rclone "$HOME/.local/bin/rclone"
  rm -rf "$TMPD"
else
  RCLONE_BIN="$(command -v rclone || echo "$HOME/.local/bin/rclone")"
  echo "[bootstrap] rclone already installed: $("$RCLONE_BIN" version | head -n 1)"
fi
RCLONE_BIN="$(command -v rclone || echo "$HOME/.local/bin/rclone")"

# 3. SSH key for `colab ssh`
if [[ ! -f "$HOME/.ssh/id_ed25519" ]]; then
  mkdir -p "$HOME/.ssh"
  chmod 700 "$HOME/.ssh"
  ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519" -q
  echo "[bootstrap] Generated ~/.ssh/id_ed25519"
fi

# 4. Configure rclone (explicit source -> uploaded file -> token -> Drive auto-pull)
TARGET_CONF="$HOME/.config/rclone/rclone.conf"
UPLOAD_CONF="/home/antigravity/upload/rclone.conf"
UPDATED_LOCALLY="0"

if [[ -n "$RCLONE_CONF_SRC" ]]; then
  install -m 600 "$RCLONE_CONF_SRC" "$TARGET_CONF"
  UPDATED_LOCALLY="1"
  echo "[bootstrap] Installed rclone.conf from $RCLONE_CONF_SRC (chmod 600)."
elif [[ -f "$UPLOAD_CONF" ]]; then
  install -m 600 "$UPLOAD_CONF" "$TARGET_CONF"
  rm -f "$UPLOAD_CONF"
  UPDATED_LOCALLY="1"
  echo "[bootstrap] Installed rclone.conf from uploaded file and scrubbed $UPLOAD_CONF."
elif [[ -n "$RCLONE_TOKEN" ]]; then
  "$RCLONE_BIN" config create "$REMOTE" drive scope=drive token="$RCLONE_TOKEN" --non-interactive >/dev/null
  chmod 600 "$TARGET_CONF"
  UPDATED_LOCALLY="1"
  echo "[bootstrap] Configured rclone remote '$REMOTE:' from token JSON (chmod 600)."
elif [[ ! -f "$TARGET_CONF" ]] && command -v gdrive >/dev/null 2>&1; then
  SECRETS_ID="$(find_drive_secrets_folder || true)"
  if [[ -n "$SECRETS_ID" ]]; then
    CONF_ID="$(gdrive readonly ls "$SECRETS_ID" --json 2>/dev/null \
      | jq -r '.[] | select(.name=="rclone.conf") | .id' | head -n 1 || true)"
    if [[ -n "$CONF_ID" ]]; then
      gdrive readonly download "$CONF_ID" --out "$TARGET_CONF" >/dev/null
      chmod 600 "$TARGET_CONF"
      echo "[bootstrap] Auto-pulled MyDrive/${DRIVE_ROOT}/.secrets/rclone.conf -> $TARGET_CONF (chmod 600)."
    fi
  fi
fi

# Optional: save local rclone.conf back to private MyDrive/<DRIVE_ROOT>/.secrets/rclone.conf
if [[ "$SAVE_TO_DRIVE" == "1" && -f "$TARGET_CONF" ]]; then
  SECRETS_ID="$(find_drive_secrets_folder || true)"
  if [[ -n "$SECRETS_ID" ]]; then
    EXISTING_ID="$(gdrive readonly ls "$SECRETS_ID" --json 2>/dev/null \
      | jq -r '.[] | select(.name=="rclone.conf") | .id' | head -n 1 || true)"
    if [[ -n "$EXISTING_ID" ]]; then
      gdrive mutate update "$EXISTING_ID" "$TARGET_CONF" >/dev/null
      echo "[bootstrap] Updated MyDrive/${DRIVE_ROOT}/.secrets/rclone.conf on Drive."
    else
      gdrive mutate upload "$TARGET_CONF" --parent "$SECRETS_ID" >/dev/null
      echo "[bootstrap] Saved MyDrive/${DRIVE_ROOT}/.secrets/rclone.conf on Drive for automatic future cloud VMs."
    fi
  else
    echo "[bootstrap] WARNING: could not locate MyDrive/${DRIVE_ROOT}/.secrets via gdrive CLI." >&2
  fi
fi

if [[ -f "$TARGET_CONF" ]] && "$RCLONE_BIN" listremotes | grep -qx "${REMOTE}:"; then
  echo "[bootstrap] rclone remote '${REMOTE}:' is READY."
else
  echo "[bootstrap] rclone is installed, but remote '${REMOTE}:' is not configured yet."
  echo "  One-time setup for automatic future cloud VMs:"
  echo "    - Upload ~/.config/rclone/rclone.conf once to MyDrive/${DRIVE_ROOT}/.secrets/rclone.conf"
  echo "      (or drop it in chat and run: bash scripts/colab_cloud_bootstrap.sh --save-to-drive)"
fi

# 5. Check Colab CLI auth
export PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring
if timeout 10 "$HOME/.local/bin/colab" --auth=oauth2 sessions >/dev/null 2>&1; then
  echo "[bootstrap] colab CLI is authenticated and READY."
else
  echo "[bootstrap] colab CLI needs one-time OAuth login:"
  echo "  PYTHONUNBUFFERED=1 PYTHON_KEYRING_BACKEND=keyrings.alt.file.PlaintextKeyring ~/.local/bin/colab --auth=oauth2 whoami"
fi
