"""Parse and stream AR tide iteration training logs."""

from __future__ import annotations

import contextlib
import json
import re
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITER_DIR = REPO / "logs" / "ar_tide_iter"
STATUS_DIR = ITER_DIR / "status"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PROGRESS_RE = re.compile(r"[\u2580-\u259f\u2500-\u257f]")
_EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+)$")
_VAL_KEYS = ("val_overfit_gate", "val_ordered_onset_match", "val_loss")


def sanitize_line(line: str) -> str:
    """Strip Keras progress-bar noise for readable console output."""
    text = _ANSI_RE.sub("", line)
    text = _PROGRESS_RE.sub("", text)
    return text.rstrip()


def extract_val_metrics(clean: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for key in _VAL_KEYS:
        marker = f"{key}: "
        if marker not in clean:
            continue
        tail = clean.split(marker, 1)[1]
        val = tail.split(" - ", 1)[0].strip()
        with contextlib.suppress(ValueError):
            out[key] = float(val)
    return out


def is_worth_printing(line: str) -> bool:
    clean = sanitize_line(line)
    if not clean:
        return False
    if _EPOCH_RE.match(clean):
        return True
    if "val_overfit_gate:" in clean:
        return True
    return clean.startswith("Traceback") or "Error" in clean[:80]


def parse_train_log(log_path: Path) -> dict:
    """Summarize a train log file (works while training is in progress)."""
    if not log_path.is_file():
        return {"error": f"log not found: {log_path}"}

    epoch = 0
    epochs_total: int | None = None
    last_val: dict[str, float] = {}
    last_error: str | None = None

    raw = log_path.read_bytes().decode("utf-8", errors="replace")
    for line in raw.splitlines():
        clean = sanitize_line(line)
        m_epoch = _EPOCH_RE.match(clean)
        if m_epoch:
            epoch = int(m_epoch.group(1))
            epochs_total = int(m_epoch.group(2))
            continue
        parsed_val = extract_val_metrics(clean)
        if parsed_val:
            last_val = parsed_val
        if clean.startswith("Traceback"):
            last_error = clean

    status: dict = {
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "log_path": str(log_path.relative_to(REPO)).replace("\\", "/"),
        "epoch": epoch,
        "epochs_total": epochs_total,
        "running": epoch > 0 and epochs_total is not None and epoch < epochs_total,
    }
    if last_val:
        status["last_val"] = last_val
    if last_error:
        status["last_error"] = last_error
    return status


def status_path(exp_id: str) -> Path:
    return STATUS_DIR / f"{exp_id}.json"


def write_status(exp_id: str, status: dict) -> Path:
    STATUS_DIR.mkdir(parents=True, exist_ok=True)
    out = status_path(exp_id)
    payload = {"id": exp_id, **status}
    out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return out


def format_status(status: dict) -> str:
    if status.get("error"):
        return status["error"]
    exp_id = status.get("id", "?")
    epoch = status.get("epoch", 0)
    total = status.get("epochs_total", "?")
    lines = [
        f"[{exp_id}] epoch {epoch}/{total}",
        f"log: {status.get('log_path', '?')}",
        f"status: {status_path(exp_id).relative_to(REPO)}",
        f"updated: {status.get('updated_at', '?')}",
    ]
    last_val = status.get("last_val")
    if last_val:
        lines.append(
            "val_overfit_gate={val_overfit_gate:.4f} "
            "val_ordered_onset_match={val_ordered_onset_match:.4f} "
            "val_loss={val_loss:.4f}".format(**last_val)
        )
    if status.get("last_error"):
        lines.append(f"ERROR: {status['last_error']}")
    return "\n".join(lines)


def refresh_status_from_log(exp_id: str) -> dict:
    log_path = ITER_DIR / "train_logs" / f"{exp_id}.log"
    status = parse_train_log(log_path)
    status["id"] = exp_id
    write_status(exp_id, status)
    return status
