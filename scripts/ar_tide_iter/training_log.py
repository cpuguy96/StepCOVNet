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
RESULTS_JSONL = ITER_DIR / "results.jsonl"

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_PROGRESS_RE = re.compile(r"[\u2580-\u259f\u2500-\u257f]")
_EPOCH_RE = re.compile(r"^Epoch (\d+)/(\d+)$")
_ATTEMPT_LOG_RE = re.compile(r"\.attempt(\d+)\.log$")
_VAL_KEYS = (
    "val_gate_teacher",
    "val_overfit_gate",
    "val_timing_match_teacher",
    "val_ordered_onset_match",
    "val_aux_f1_hungarian",
    "val_event_onset_f1",
    "val_token_accuracy",
    "val_loss",
)

TEACHER_PERFECT_EPS = 1e-6


def _val_metric(metrics: dict[str, float], canonical: str, legacy: str) -> float | None:
    canonical_key = f"val_{canonical}" if not canonical.startswith("val_") else canonical
    legacy_key = f"val_{legacy}" if not legacy.startswith("val_") else legacy
    if canonical_key in metrics:
        return float(metrics[canonical_key])
    if legacy_key in metrics:
        return float(metrics[legacy_key])
    return None


def teacher_metrics_perfect(metrics: dict[str, float]) -> bool:
    """True when in-loop teacher-fed primary metrics are all 1.0 (tide overfit bar)."""
    token = _val_metric(metrics, "token_accuracy", "token_accuracy")
    timing = _val_metric(metrics, "timing_match_teacher", "ordered_onset_match")
    gate = _val_metric(metrics, "gate_teacher", "overfit_gate")
    if token is None or timing is None or gate is None:
        return False
    return all(value >= 1.0 - TEACHER_PERFECT_EPS for value in (token, timing, gate))


def teacher_report_perfect(report: dict) -> bool:
    """True when offline teacher-fed eval hits the perfect-overfit bar.

    Primary gate: ordered onset match vs teacher ``target_times`` (see
    ONSET_METRICS.md). Hungarian ``event_f1`` vs raw chart times is logged but
    not required — it can diverge when residual error nears tolerance.
    """
    ordered = report.get("ordered_onset_match", {})
    if not isinstance(ordered, dict):
        return False
    n_matched = int(ordered.get("n_matched", 0))
    n_denom = int(ordered.get("n_denom", 0))
    if n_matched != n_denom or n_denom <= 0:
        return False
    rate = float(ordered.get("rate", 0.0))
    return rate >= 1.0 - TEACHER_PERFECT_EPS


def count_logged_attempts(exp_id: str) -> int:
    """Return how many prior results are recorded for ``exp_id``."""
    if not RESULTS_JSONL.is_file():
        return 0
    count = 0
    for line in RESULTS_JSONL.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        with contextlib.suppress(json.JSONDecodeError):
            if json.loads(line).get("id") == exp_id:
                count += 1
    return count


def train_log_path(exp_id: str, attempt: int) -> Path:
    """Path for a train log; attempt 1 keeps the legacy ``{id}.log`` name."""
    if attempt <= 1:
        return ITER_DIR / "train_logs" / f"{exp_id}.log"
    return ITER_DIR / "train_logs" / f"{exp_id}.attempt{attempt}.log"


def latest_train_log_path(exp_id: str) -> Path:
    """Return the newest train log for an experiment id."""
    logs_dir = ITER_DIR / "train_logs"
    candidates = list(logs_dir.glob(f"{exp_id}.attempt*.log"))

    def _attempt_num(path: Path) -> int:
        match = _ATTEMPT_LOG_RE.search(path.name)
        return int(match.group(1)) if match else 0

    if candidates:
        return max(candidates, key=_attempt_num)
    return logs_dir / f"{exp_id}.log"


def run_kind(*, attempt: int, retry_reason: str) -> str:
    """Classify a run as fresh or retry for the human log."""
    if attempt > 1 or retry_reason:
        label = "retry"
        if retry_reason:
            return f"{label} — {retry_reason}"
        return label
    return "fresh"


def format_log_heading(exp_id: str, attempt: int, timestamp: str) -> str:
    """Markdown heading for one logged run."""
    if attempt <= 1:
        return f"### {exp_id} ({timestamp})"
    return f"### {exp_id} · attempt {attempt} ({timestamp})"


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
    if "val_gate_teacher:" in clean or "val_overfit_gate:" in clean:
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
        gate = _val_metric(last_val, "gate_teacher", "overfit_gate") or 0.0
        timing = _val_metric(last_val, "timing_match_teacher", "ordered_onset_match") or 0.0
        aux_f1 = _val_metric(last_val, "aux_f1_hungarian", "event_onset_f1") or 0.0
        lines.append(
            "val_gate_teacher={gate:.4f} "
            "val_timing_match_teacher={timing:.4f} "
            "val_aux_f1_hungarian={aux_f1:.4f} "
            "val_token_accuracy={token:.4f} "
            "val_loss={loss:.4f}".format(
                gate=gate,
                timing=timing,
                aux_f1=aux_f1,
                token=last_val.get("val_token_accuracy", 0.0),
                loss=last_val.get("val_loss", 0.0),
            )
        )
    if status.get("last_error"):
        lines.append(f"ERROR: {status['last_error']}")
    return "\n".join(lines)


def refresh_status_from_log(exp_id: str) -> dict:
    log_path = latest_train_log_path(exp_id)
    status = parse_train_log(log_path)
    status["id"] = exp_id
    write_status(exp_id, status)
    return status
