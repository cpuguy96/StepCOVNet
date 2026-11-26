#!/usr/bin/env bash
# End-to-end automated Google Colab GPU runner using google-colab-cli + private Drive datasets.
#
# Usage:
#   bash scripts/colab_run_remote.sh \
#     --session exp1 \
#     --gpu L4 \
#     --run-name my_exp \
#     --script scripts/train_itgpt_placement.py \
#     --config configs/ddc/itgpt_mert_placement_fraxtil_orig_cosine64_l2_6_10_16_ema.json \
#     --datasets literature_fraxtil_orig,literature_fraxtil_orig__mert_l2_6_10_16 \
#     [--extra-args "--epochs=2 --steps_per_epoch=10"] \
#     [--eval-cmd "python scripts/eval_itgpt_placement.py --config={CONFIG} --model_dir={RUN_DIR}"] \
#     [--archive-models best|all|none] \
#     [--reuse-session] [--keep] [--download-all]
#
# Notes:
#   - Code (~1 MB) is uploaded directly from the local working tree via `colab upload` (zero GitHub).
#   - Datasets are streamed directly from private Google Drive inside Google's network using a
#     short-lived access token held only in memory during transfer (client_secret & refresh_token
#     NEVER leave this machine).
#   - Training runs detached inside the Colab VM and is polled with automatic WebSocket retries,
#     so transient network/WebSocket drops never abort training.
#   - Best checkpoints + logs are archived automatically to MyDrive/StepCOVNet/runs/<run-name>/
#     before the VM stops (--archive-models=best by default; pull locally anytime with
#     `bash scripts/colab_sync.sh pull-runs <run-name>`).
#   - Lightweight logs/eval JSONs download locally to experiments/colab/<run-name>/ (gitignored).
#   - Unless --keep is passed, the remote Colab VM is stopped automatically when finished.

set -euo pipefail

main() {

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

find_bin() {
  local name="$1"
  if command -v "$name" >/dev/null 2>&1; then
    command -v "$name"
  elif [[ -x "$HOME/.local/bin/$name" ]]; then
    echo "$HOME/.local/bin/$name"
  else
    echo "ERROR: $name not found in PATH or ~/.local/bin/$name" >&2
    exit 1
  fi
}

COLAB="$(find_bin colab)"
RCLONE="$(find_bin rclone)"
REMOTE="${COLAB_RCLONE_REMOTE:-gdrive}"
DRIVE_ROOT="${COLAB_DRIVE_ROOT:-StepCOVNet}"
LOCAL_RUNS_DIR="${COLAB_LOCAL_RUNS_DIR:-$REPO_ROOT/experiments/colab}"

SESSION=""
GPU="T4"
RUN_NAME=""
TRAIN_SCRIPT="scripts/train_itgpt_placement.py"
CONFIG_PATH=""
DATASETS=""
EXTRA_ARGS=""
EVAL_CMD=""
REUSE_SESSION="0"
KEEP_SESSION="0"
DOWNLOAD_ALL="0"
ARCHIVE_MODELS="best"
HIGH_MEM="0"
COLAB_STATE_CONFIG=""
POLL_INTERVAL="20"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --session|-s) SESSION="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --high-mem) HIGH_MEM="1"; shift ;;
    --run-name) RUN_NAME="$2"; shift 2 ;;
    --script) TRAIN_SCRIPT="$2"; shift 2 ;;
    --config) CONFIG_PATH="$2"; shift 2 ;;
    --datasets) DATASETS="$2"; shift 2 ;;
    --extra-args) EXTRA_ARGS="$2"; shift 2 ;;
    --eval-cmd) EVAL_CMD="$2"; shift 2 ;;
    --archive-models) ARCHIVE_MODELS="$2"; shift 2 ;;
    --reuse-session) REUSE_SESSION="1"; shift ;;
    --keep) KEEP_SESSION="1"; shift ;;
    --download-all) DOWNLOAD_ALL="1"; shift ;;
    --state-config) COLAB_STATE_CONFIG="$2"; shift 2 ;;
    --poll-interval) POLL_INTERVAL="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,28p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown flag: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "$SESSION" || -z "$RUN_NAME" || -z "$CONFIG_PATH" ]]; then
  echo "ERROR: --session, --run-name, and --config are required." >&2
  exit 1
fi

case "$ARCHIVE_MODELS" in
  best|all|none) ;;
  *)
    echo "ERROR: --archive-models must be one of: best, all, none (got: $ARCHIVE_MODELS)" >&2
    exit 1
    ;;
esac

if [[ ! -f "$REPO_ROOT/$CONFIG_PATH" ]]; then
  echo "ERROR: config not found: $REPO_ROOT/$CONFIG_PATH" >&2
  exit 1
fi

COLAB_GLOBAL=()
if [[ -n "$COLAB_STATE_CONFIG" ]]; then
  COLAB_GLOBAL+=(--config "$COLAB_STATE_CONFIG")
fi

# Retry wrapper around `colab exec` with hard timeout to survive transient WebSocket drops
colab_exec_retry() {
  local max_attempts="$1"
  local hard_timeout="$2"
  shift 2
  local attempt=1
  while true; do
    if timeout "$hard_timeout" "$COLAB" "${COLAB_GLOBAL[@]}" exec -s "$SESSION" "$@"; then
      return 0
    fi
    local rc=$?
    if [[ "$attempt" -ge "$max_attempts" ]]; then
      echo "[colab-remote] ERROR: colab exec failed after ${max_attempts} attempts (rc=${rc})." >&2
      return "$rc"
    fi
    echo "[colab-remote] Transient colab exec error (rc=${rc}); retrying (${attempt}/${max_attempts}) in 5s..." >&2
    sleep 5
    attempt=$((attempt + 1))
  done
}

TMPD="$(mktemp -d)"
STOPPED="0"
cleanup() {
  rm -rf "$TMPD"
  if [[ "$KEEP_SESSION" == "0" && "$STOPPED" == "0" ]]; then
    echo "[colab-remote] Stopping session '$SESSION'..."
    timeout 30 "$COLAB" "${COLAB_GLOBAL[@]}" stop -s "$SESSION" >/dev/null 2>&1 || true
    STOPPED="1"
  fi
}
trap cleanup EXIT

# 1. Provision (or reuse) Colab GPU session
if [[ "$REUSE_SESSION" == "0" ]]; then
  NEW_ARGS=(new -s "$SESSION" --gpu "$GPU")
  if [[ "$HIGH_MEM" == "1" ]]; then
    NEW_ARGS+=(--high-mem)
  fi
  echo "[colab-remote] Provisioning session '$SESSION' (GPU=$GPU)..."
  timeout 120 "$COLAB" "${COLAB_GLOBAL[@]}" "${NEW_ARGS[@]}"
else
  echo "[colab-remote] Reusing existing session '$SESSION'..."
fi

# 2. Bundle & upload local code directly (~1 MB, zero GitHub)
CODE_TAR="$TMPD/code.tar.gz"
echo "[colab-remote] Packaging & uploading local code..."
tar -czf "$CODE_TAR" -C "$REPO_ROOT" \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='.DS_Store' \
  --exclude='*.egg-info' \
  pyproject.toml src scripts configs

timeout 60 "$COLAB" "${COLAB_GLOBAL[@]}" upload -s "$SESSION" "$CODE_TAR" /content/code.tar.gz

# 3. Extract code & install editable package on remote VM
cat > "$TMPD/setup_code.py" <<'PYEOF'
import os, pathlib, shutil, subprocess

work = pathlib.Path("/content/StepCOVNet")
work.mkdir(parents=True, exist_ok=True)
for sub in ("src", "scripts", "configs"):
    p = work / sub
    if p.exists():
        shutil.rmtree(p)
subprocess.run(["tar", "-xzf", "/content/code.tar.gz", "-C", str(work)], check=True)
subprocess.run(["pip", "install", "-q", "-e", str(work), "--no-deps"], check=True)
gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
).strip()
print(f"[remote] Code ready in {work} | GPU: {gpu}", flush=True)
PYEOF
colab_exec_retry 3 240 --timeout 180 -f "$TMPD/setup_code.py"

# 4. Stream required dataset archive(s) directly from private Google Drive -> Colab SSD
if [[ -n "$DATASETS" ]]; then
  DRIVE_JSON="$("$RCLONE" lsjson "${REMOTE}:${DRIVE_ROOT}/data_archives")"
  ACCESS_TOKEN="$(python3 - "$REMOTE" <<'PYEOF'
import configparser, json, pathlib, sys
remote = sys.argv[1]
conf = pathlib.Path.home() / ".config" / "rclone" / "rclone.conf"
c = configparser.ConfigParser()
c.read(conf)
tok = json.loads(c[remote]["token"])
print(tok["access_token"], end="")
PYEOF
)"

  ARCHIVE_SPECS="$(python3 - "$DATASETS" "$DRIVE_JSON" <<'PYEOF'
import json, sys
wanted = [x.strip() for x in sys.argv[1].split(",") if x.strip()]
items = {it["Name"]: it["ID"] for it in json.loads(sys.argv[2])}
out = []
for name in wanted:
    fname = f"{name}.tar"
    if fname not in items:
        raise SystemExit(f"ERROR: archive '{fname}' not found in Drive data_archives/. Available: {sorted(items)}")
    out.append({"name": name, "id": items[fname]})
print(json.dumps(out))
PYEOF
)"

  cat > "$TMPD/fetch_data.py" <<'PYEOF'
import json, os, pathlib, subprocess, time, urllib.request

token = os.environ.pop("GDRIVE_ACCESS_TOKEN")
specs = json.loads(os.environ["STEPCOVNET_ARCHIVE_SPECS"])
data_dir = pathlib.Path("/content/StepCOVNet/data")
data_dir.mkdir(parents=True, exist_ok=True)

for item in specs:
    name = item["name"]
    fid = item["id"]
    marker = data_dir / f".unpacked_{name}"
    if marker.is_file():
        print(f"[remote] Archive '{name}' already unpacked on local SSD.", flush=True)
        continue
    t0 = time.time()
    url = f"https://www.googleapis.com/drive/v3/files/{fid}?alt=media"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}"})
    print(f"[remote] Streaming {name}.tar from Google Drive -> {data_dir} ...", flush=True)
    proc = subprocess.Popen(["tar", "-xf", "-", "-C", str(data_dir)], stdin=subprocess.PIPE)
    assert proc.stdin is not None
    total = 0
    with urllib.request.urlopen(req) as resp:
        while True:
            chunk = resp.read(4 * 1024 * 1024)
            if not chunk:
                break
            proc.stdin.write(chunk)
            total += len(chunk)
    proc.stdin.close()
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"tar extraction failed for {name} (exit {rc})")
    marker.write_text("ok\n")
    dt = time.time() - t0
    mb = total / (1024 * 1024)
    print(f"[remote] Unpacked {name} ({mb:.1f} MiB in {dt:.1f}s = {mb / max(dt, 0.1):.1f} MiB/s)", flush=True)
PYEOF

  echo "[colab-remote] Streaming datasets from Drive inside Google's network..."
  colab_exec_retry 3 660 --timeout 600 \
    --env "GDRIVE_ACCESS_TOKEN=${ACCESS_TOKEN}" \
    --env "STEPCOVNET_ARCHIVE_SPECS=${ARCHIVE_SPECS}" \
    -f "$TMPD/fetch_data.py"
fi

# 5. Launch training DETACHED on Colab VM so WebSocket hiccups never kill training
cat > "$TMPD/start_train.py" <<'PYEOF'
import datetime, json, os, pathlib, shlex, shutil, subprocess, sys

work = pathlib.Path("/content/StepCOVNet")
run_name = os.environ["STEPCOVNET_RUN_NAME"]
train_script = os.environ["STEPCOVNET_TRAIN_SCRIPT"]
config_path = os.environ["STEPCOVNET_CONFIG_PATH"]
extra_args = os.environ.get("STEPCOVNET_EXTRA_ARGS", "")
eval_cmd = os.environ.get("STEPCOVNET_EVAL_CMD", "")
pack_all = os.environ.get("STEPCOVNET_DOWNLOAD_ALL", "0") == "1"

run_dir = pathlib.Path("/content/runs") / run_name
run_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(work / config_path, run_dir / "config.json")

cmd = [
    sys.executable,
    train_script,
    f"--config={config_path}",
    f"--model_output_dir={run_dir}",
]
if extra_args.strip():
    cmd.extend(shlex.split(extra_args))

gpu = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
    text=True,
).strip()

meta = {
    "run_name": run_name,
    "command": " ".join(shlex.quote(c) for c in cmd),
    "gpu": gpu,
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "status": "running",
}
(run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

# Write a detached wrapper script that runs training, optional eval, packs results, and writes train.done
wrapper = run_dir / "_runner.py"
wrapper.write_text(f"""
import datetime, json, os, pathlib, shlex, subprocess, sys

os.chdir({str(work)!r})
run_dir = pathlib.Path({str(run_dir)!r})
cmd = {cmd!r}
eval_cmd = {eval_cmd!r}
config_path = {config_path!r}
run_name = {run_name!r}
pack_all = {pack_all!r}

env = os.environ.copy()
env["STEPCOVNET_FORCE_GPU"] = "1"
env["STEPCOVNET_NO_WSL"] = "1"
env["PYTHONUNBUFFERED"] = "1"

log_path = run_dir / "train.log"
rc = 1
try:
    with log_path.open("w", encoding="utf-8") as lf:
        lf.write("[remote] Running: " + " ".join(shlex.quote(c) for c in cmd) + "\\n")
        lf.flush()
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)
        rc = p.wait()
        lf.write(f"\\n[remote] Training exit code: {{rc}}\\n")
        lf.flush()

    if rc == 0 and eval_cmd.strip():
        expanded = eval_cmd.format(RUN_DIR=str(run_dir), CONFIG=config_path)
        with (run_dir / "eval.log").open("w", encoding="utf-8") as ef:
            ef.write("[remote] Running eval: " + expanded + "\\n")
            ef.flush()
            ep = subprocess.Popen(shlex.split(expanded), stdout=ef, stderr=subprocess.STDOUT, env=env)
            erc = ep.wait()
            ef.write(f"\\n[remote] Eval exit code: {{erc}}\\n")
finally:
    meta = json.loads((run_dir / "run_meta.json").read_text())
    meta["finished_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    meta["exit_code"] = rc
    meta["status"] = "done" if rc == 0 else "failed"
    (run_dir / "run_meta.json").write_text(json.dumps(meta, indent=2))

    out_tar = f"/content/{{run_name}}_results.tar.gz"
    if pack_all:
        subprocess.run(["tar", "-czf", out_tar, "-C", "/content/runs", run_name], check=False)
    else:
        files = [
            str(p.relative_to("/content/runs"))
            for p in run_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in {{".log", ".txt", ".json", ".csv", ".md"}}
        ]
        if files:
            subprocess.run(["tar", "-czf", out_tar, "-C", "/content/runs", *files], check=False)
    (run_dir / "train.done").write_text(str(rc))
""")

done_file = run_dir / "train.done"
if done_file.exists():
    done_file.unlink()

proc = subprocess.Popen(
    [sys.executable, str(wrapper)],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
(run_dir / "train.pid").write_text(str(proc.pid))
print(f"[remote] Detached training launched (pid={proc.pid}) | GPU: {gpu}", flush=True)
PYEOF

echo "[colab-remote] Launching detached remote training on '$SESSION'..."
colab_exec_retry 3 90 --timeout 60 \
  --env "STEPCOVNET_RUN_NAME=${RUN_NAME}" \
  --env "STEPCOVNET_TRAIN_SCRIPT=${TRAIN_SCRIPT}" \
  --env "STEPCOVNET_CONFIG_PATH=${CONFIG_PATH}" \
  --env "STEPCOVNET_EXTRA_ARGS=${EXTRA_ARGS}" \
  --env "STEPCOVNET_EVAL_CMD=${EVAL_CMD}" \
  --env "STEPCOVNET_DOWNLOAD_ALL=${DOWNLOAD_ALL}" \
  -f "$TMPD/start_train.py"

# Poll remote detached process, streaming concise epoch/eval lines (filtering \r progress-bar spam)
# and periodically syncing TensorBoard tfevents locally into experiments/colab/<RUN_NAME>/tb/.
cat > "$TMPD/poll_train.py" <<'PYEOF'
import glob, json, os, pathlib, re, tarfile

run_dir = pathlib.Path("/content/runs") / os.environ["STEPCOVNET_RUN_NAME"]
offset = int(os.environ.get("STEPCOVNET_LOG_OFFSET", "0"))
sync_tb = os.environ.get("STEPCOVNET_SYNC_TB", "0") == "1"
log_path = run_dir / "train.log"
done_path = run_dir / "train.done"
pid_path = run_dir / "train.pid"

PROGRESS_RE = re.compile(r"^\s*\d+/\d+\s+[━─=.\-|>]+")

new_offset = offset
out_lines = []
if log_path.is_file():
    size = log_path.stat().st_size
    if size > offset:
        with log_path.open("rb") as f:
            f.seek(offset)
            data = f.read(min(size - offset, 512 * 1024))
        new_offset = offset + len(data)
        text = data.decode("utf-8", errors="replace")
        for raw_line in text.splitlines():
            # Collapse carriage-return overwrites to final segment; drop mid-epoch step bars
            seg = raw_line.split("\r")[-1].strip()
            if not seg:
                continue
            if PROGRESS_RE.match(seg) and "val_loss" not in seg and "ms/step" in seg and "- val_" not in seg:
                continue
            out_lines.append(seg)

tb_count = 0
if sync_tb:
    tb_files = sorted(glob.glob("/content/StepCOVNet/callbacks/**/*tfevents*", recursive=True))
    if tb_files:
        with tarfile.open("/content/_tb_sync.tar.gz", "w:gz") as tf:
            for f in tb_files:
                parts = f.split("/logs/")[-1].split("/", 1)
                arc = parts[1] if len(parts) > 1 else os.path.basename(f)
                tf.add(f, arcname=arc)
        tb_count = len(tb_files)

state = "running"
rc = None
if done_path.is_file():
    state = "done"
    rc = int(done_path.read_text().strip() or "1")
elif pid_path.is_file():
    pid = int(pid_path.read_text().strip() or "0")
    if pid > 0 and not pathlib.Path(f"/proc/{pid}").exists():
        state = "crashed"
        rc = 1

print("__POLL_HEADER__" + json.dumps({"state": state, "rc": rc, "offset": new_offset, "tb": tb_count}))
if out_lines:
    print("\n".join(out_lines), flush=True)
PYEOF

TB_LOCAL_DIR="${LOCAL_RUNS_DIR}/${RUN_NAME}/tb"
mkdir -p "$TB_LOCAL_DIR"
LOG_OFFSET="0"
TRAIN_RC="1"
CONSEC_ERR="0"
POLL_TICK="0"
while true; do
  sleep "$POLL_INTERVAL"
  POLL_TICK=$((POLL_TICK + 1))
  DO_TB="0"
  if (( POLL_TICK == 1 || POLL_TICK % 3 == 0 )); then
    DO_TB="1"
  fi
  POLL_OUT=""
  if POLL_OUT="$(timeout 45 "$COLAB" "${COLAB_GLOBAL[@]}" exec -s "$SESSION" --timeout 30 \
    --env "STEPCOVNET_RUN_NAME=${RUN_NAME}" \
    --env "STEPCOVNET_LOG_OFFSET=${LOG_OFFSET}" \
    --env "STEPCOVNET_SYNC_TB=${DO_TB}" \
    -f "$TMPD/poll_train.py" 2>/dev/null)"; then
    CONSEC_ERR="0"
    HEADER_LINE="$(printf '%s\n' "$POLL_OUT" | grep '^__POLL_HEADER__' | tail -n 1 || true)"
    if [[ -n "$HEADER_LINE" ]]; then
      BODY="$(printf '%s\n' "$POLL_OUT" | grep -v '^__POLL_HEADER__' || true)"
      if [[ -n "$BODY" ]]; then
        printf '%s\n' "$BODY"
      fi
      PARSED="$(python3 - "$HEADER_LINE" <<'PYEOF'
import json, sys
obj = json.loads(sys.argv[1].removeprefix("__POLL_HEADER__"))
print(obj["state"], obj["offset"], "-" if obj["rc"] is None else obj["rc"], obj.get("tb", 0))
PYEOF
)"
      read -r P_STATE LOG_OFFSET P_RC P_TB <<< "$PARSED"
      if [[ "${P_TB:-0}" -gt 0 ]]; then
        if timeout 30 "$COLAB" "${COLAB_GLOBAL[@]}" download -s "$SESSION" /content/_tb_sync.tar.gz "$TMPD/_tb_sync.tar.gz" >/dev/null 2>&1; then
          python3 - "$TMPD/_tb_sync.tar.gz" "$TB_LOCAL_DIR" <<'PYEOF' 2>/dev/null || true
import pathlib, tarfile, sys
tar_path, dst_root = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
with tarfile.open(tar_path, "r:gz") as tf:
    for m in tf.getmembers():
        if not m.isfile():
            continue
        dst = dst_root / m.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        fobj = tf.extractfile(m)
        if fobj is None:
            continue
        data = fobj.read()
        # Append in-place (preserving inode) so live TensorBoard readers see updates!
        cur = dst.stat().st_size if dst.is_file() else 0
        if len(data) > cur:
            with dst.open("ab") as out:
                out.write(data[cur:])
        elif len(data) < cur:
            with dst.open("r+b") as out:
                out.write(data)
                out.truncate(len(data))
PYEOF
          rm -f "$TMPD/_tb_sync.tar.gz"
        fi
      fi
      if [[ "$P_STATE" == "done" ]]; then
        TRAIN_RC="${P_RC:-0}"
        echo "[colab-remote] Remote training completed with exit code ${TRAIN_RC}."
        break
      elif [[ "$P_STATE" == "crashed" ]]; then
        TRAIN_RC="1"
        echo "[colab-remote] ERROR: Remote runner process exited unexpectedly without train.done." >&2
        break
      fi
    fi
  else
    CONSEC_ERR=$((CONSEC_ERR + 1))
    echo "[colab-remote] Poll hiccup (${CONSEC_ERR}/15); remote training continues — retrying..." >&2
    if [[ "$CONSEC_ERR" -ge 15 ]]; then
      echo "[colab-remote] ERROR: Lost contact with Colab session '${SESSION}' after 15 retries." >&2
      exit 1
    fi
  fi
done

# 6. Archive best checkpoints + logs directly from Colab -> Google Drive (inside Google's network)
if [[ "$ARCHIVE_MODELS" != "none" ]]; then
  echo "[colab-remote] Archiving outputs (mode=${ARCHIVE_MODELS}) -> ${REMOTE}:${DRIVE_ROOT}/runs/${RUN_NAME}/ ..."
  "$RCLONE" mkdir "${REMOTE}:${DRIVE_ROOT}/runs/${RUN_NAME}"
  RUNS_JSON="$("$RCLONE" lsjson --dirs-only "${REMOTE}:${DRIVE_ROOT}/runs")"
  FOLDER_ID="$(python3 - "$RUN_NAME" "$RUNS_JSON" <<'PYEOF'
import json, sys
target = sys.argv[1]
for it in json.loads(sys.argv[2]):
    if it.get("Name") == target:
        print(it["ID"], end="")
        raise SystemExit(0)
raise SystemExit(f"ERROR: Drive folder runs/{target} not found after mkdir")
PYEOF
)"
  UPLOAD_TOKEN="$(python3 - "$REMOTE" <<'PYEOF'
import configparser, json, pathlib, sys
remote = sys.argv[1]
conf = pathlib.Path.home() / ".config" / "rclone" / "rclone.conf"
c = configparser.ConfigParser()
c.read(conf)
tok = json.loads(c[remote]["token"])
print(tok["access_token"], end="")
PYEOF
)"

  cat > "$TMPD/upload_drive.py" <<'PYEOF'
import json, os, pathlib, time, urllib.error, urllib.parse, urllib.request

token = os.environ.pop("GDRIVE_ACCESS_TOKEN")
folder_id = os.environ["STEPCOVNET_DRIVE_FOLDER_ID"]
run_name = os.environ["STEPCOVNET_RUN_NAME"]
mode = os.environ["STEPCOVNET_ARCHIVE_MODE"]

run_dir = pathlib.Path("/content/runs") / run_name
LOG_EXTS = {".log", ".txt", ".json", ".csv", ".md"}
MODEL_EXTS = {".pt", ".pth", ".ckpt", ".safetensors", ".npz"}

def should_upload(p: pathlib.Path) -> bool:
    if not p.is_file():
        return False
    if mode == "all":
        return True
    ext = p.suffix.lower()
    if ext in LOG_EXTS:
        return True
    if ext in MODEL_EXTS and p.name.lower().startswith("best"):
        return True
    return False

files = sorted(p for p in run_dir.iterdir() if should_upload(p))
if not files:
    print(f"[remote] No matching files to archive in {run_dir}", flush=True)
    raise SystemExit(0)

q = urllib.parse.quote(f"'{folder_id}' in parents and trashed=false")
list_req = urllib.request.Request(
    f"https://www.googleapis.com/drive/v3/files?q={q}&fields=files(id,name)&pageSize=1000",
    headers={"Authorization": f"Bearer {token}"},
)
with urllib.request.urlopen(list_req) as resp:
    existing = {f["name"]: f["id"] for f in json.loads(resp.read().decode()).get("files", [])}

CHUNK = 32 * 1024 * 1024

for path in files:
    size = path.stat().st_size
    t0 = time.time()
    fid = existing.get(path.name)
    if fid:
        init_url = f"https://www.googleapis.com/upload/drive/v3/files/{fid}?uploadType=resumable"
        init_method = "PATCH"
        meta_body = json.dumps({"name": path.name}).encode()
    else:
        init_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
        init_method = "POST"
        meta_body = json.dumps({"name": path.name, "parents": [folder_id]}).encode()

    init_req = urllib.request.Request(
        init_url,
        data=meta_body,
        method=init_method,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(size),
        },
    )
    with urllib.request.urlopen(init_req) as resp:
        session_uri = resp.headers["Location"]

    if size == 0:
        put_req = urllib.request.Request(session_uri, data=b"", method="PUT", headers={"Content-Length": "0"})
        with urllib.request.urlopen(put_req):
            pass
    else:
        with path.open("rb") as f:
            offset = 0
            while offset < size:
                chunk = f.read(CHUNK)
                end = offset + len(chunk) - 1
                put_req = urllib.request.Request(
                    session_uri,
                    data=chunk,
                    method="PUT",
                    headers={
                        "Content-Length": str(len(chunk)),
                        "Content-Range": f"bytes {offset}-{end}/{size}",
                    },
                )
                try:
                    with urllib.request.urlopen(put_req):
                        pass
                except urllib.error.HTTPError as exc:
                    if exc.code != 308:
                        raise
                offset += len(chunk)

    dt = time.time() - t0
    mb = size / (1024 * 1024)
    print(f"[remote] Archived to Drive: {path.name} ({mb:.1f} MiB in {dt:.1f}s)", flush=True)
PYEOF

  colab_exec_retry 3 660 --timeout 600 \
    --env "GDRIVE_ACCESS_TOKEN=${UPLOAD_TOKEN}" \
    --env "STEPCOVNET_DRIVE_FOLDER_ID=${FOLDER_ID}" \
    --env "STEPCOVNET_RUN_NAME=${RUN_NAME}" \
    --env "STEPCOVNET_ARCHIVE_MODE=${ARCHIVE_MODELS}" \
    -f "$TMPD/upload_drive.py"
fi

# 7. Download lightweight results locally into experiments/colab/<RUN_NAME>/
mkdir -p "$LOCAL_RUNS_DIR"
LOCAL_TAR="$TMPD/${RUN_NAME}_results.tar.gz"
echo "[colab-remote] Downloading results -> ${LOCAL_RUNS_DIR}/${RUN_NAME}/ ..."
timeout 120 "$COLAB" "${COLAB_GLOBAL[@]}" download -s "$SESSION" "/content/${RUN_NAME}_results.tar.gz" "$LOCAL_TAR"
tar -xzf "$LOCAL_TAR" -C "$LOCAL_RUNS_DIR"
echo "[colab-remote] Saved locally: ${LOCAL_RUNS_DIR}/${RUN_NAME}/"
ls -la "${LOCAL_RUNS_DIR}/${RUN_NAME}/"

# 8. Stop VM unless --keep
if [[ "$KEEP_SESSION" == "0" ]]; then
  echo "[colab-remote] Stopping session '$SESSION'..."
  timeout 30 "$COLAB" "${COLAB_GLOBAL[@]}" stop -s "$SESSION" || true
  STOPPED="1"
fi

exit "$TRAIN_RC"
}

main "$@"
