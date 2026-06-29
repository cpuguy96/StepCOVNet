"""Run one AR tide-overfit iteration experiment.

Usage (repo root, Windows venv):
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_exp.py --id iter31 \\
        --notes "overnight hypothesis"

Configs are built on demand from ``experiments.json`` + champion template.
Each attempt freezes a snapshot under ``logs/ar_tide_iter/configs/``.
Use ``--reuse-last-config`` to rerun the same snapshot after infra failures.
Use ``--config`` to override with a hand-edited JSON path.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
_ITER_PKG = pathlib.Path(__file__).resolve().parent
if str(_ITER_PKG) not in sys.path:
    sys.path.insert(0, str(_ITER_PKG))
import config_builder  # noqa: E402
from training_lock import (  # noqa: E402
    acquire_training_lock,
    assert_gpu_training_available,
    release_training_lock,
)
from training_log import (  # noqa: E402
    count_logged_attempts,
    format_log_heading,
    run_kind,
    teacher_report_perfect,
    train_log_path,
)

ITER_DIR = REPO / "logs" / "ar_tide_iter"
LOG_MD = ITER_DIR / "ITER_LOG.md"
RESULTS_JSONL = ITER_DIR / "results.jsonl"
DEBUG = REPO / "scripts" / "debug_ar_onset_overfit.py"
TRAIN = REPO / "scripts" / "train_onset_ar.py"
PY = REPO / "venv" / "Scripts" / "python.exe"
DOC_LOG = REPO / "docs" / "research" / "AR_TIDE_OVERFIT_ITER_LOG.md"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--id", help="Experiment id, e.g. iter31 (required unless --build-all)"
    )
    p.add_argument(
        "--config",
        help="Path to experiment JSON; default: build from experiments.json via --id",
    )
    p.add_argument(
        "--build-all",
        action="store_true",
        help="Write all registry configs to logs/ar_tide_iter/configs/ and exit",
    )
    p.add_argument("--skip-train", action="store_true", help="Eval only")
    p.add_argument("--notes", default="", help="Hypothesis / change summary")
    p.add_argument(
        "--retry-reason",
        default="",
        help="Why this run repeats the same --id (infra kill, recipe fix, etc.)",
    )
    p.add_argument(
        "--reuse-last-config",
        action="store_true",
        help="Retry with the previous attempt's config snapshot (experiments.json ignored)",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Start training even if WSL GPU has active compute (sets STEPCOVNET_FORCE_GPU=1)",
    )
    args = p.parse_args()
    if args.build_all:
        return args
    if not args.id:
        p.error("--id is required unless --build-all is set")
    return args


def _finalize_kind(
    *,
    attempt: int,
    retry_reason: str,
    recipe_changed: bool,
    reuse_last_config: bool,
) -> str:
    if reuse_last_config:
        reason = retry_reason or "reuse prior config snapshot"
        return run_kind(attempt=attempt, retry_reason=reason)
    kind = run_kind(attempt=attempt, retry_reason=retry_reason)
    if recipe_changed and attempt > 1:
        marker = "recipe changed in experiments.json"
        if kind == "retry":
            return f"retry — {marker}" if not retry_reason else f"{kind}; {marker}"
        return f"{kind}; {marker}"
    return kind


def _resolve_config(
    args: argparse.Namespace,
    attempt: int,
) -> tuple[pathlib.Path, bool]:
    """Return config snapshot path and whether the registry recipe changed."""
    if args.config:
        return pathlib.Path(args.config).resolve(), False

    if args.reuse_last_config:
        prior = config_builder.latest_config_snapshot(
            args.id,
            before_attempt=attempt,
        )
        if prior is None:
            print(
                "no prior config snapshot; building from experiments.json",
                file=sys.stderr,
            )
        else:
            return prior.resolve(), False

    snapshot = config_builder.write_config(args.id, attempt=attempt)
    if attempt <= 1:
        return snapshot, False

    prior = config_builder.latest_config_snapshot(args.id, before_attempt=attempt)
    if prior is None:
        return snapshot, False

    old_cfg = json.loads(prior.read_text(encoding="utf-8"))
    new_cfg = json.loads(snapshot.read_text(encoding="utf-8"))
    return snapshot, not config_builder.run_blocks_equal(old_cfg, new_cfg)


def _parse_eval_json(blob: str) -> dict:
    start = blob.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in eval output:\n{blob[:1500]}")
    return json.loads(blob[start:])


def _eval(
    config: pathlib.Path,
    model_path: pathlib.Path,
    *,
    ar_decode: bool = False,
) -> dict:
    cmd = [
        str(PY),
        str(DEBUG),
        "--config",
        str(config.relative_to(REPO)),
        "--model_path",
        str(model_path.relative_to(REPO)),
        "--json-only",
    ]
    if ar_decode:
        cmd.append("--ar_decode")
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - t0
    blob = proc.stdout or proc.stderr
    report = _parse_eval_json(blob)
    report["_eval_wall_sec"] = round(elapsed, 2)
    if proc.returncode != 0:
        report["_eval_exit_code"] = proc.returncode
    return report


def _train(
    config: pathlib.Path,
    log_path: pathlib.Path,
    *,
    exp_id: str,
    force: bool = False,
) -> int:
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    from training_log import (  # noqa: PLC0415
        extract_val_metrics,
        is_worth_printing,
        sanitize_line,
        write_status,
    )

    cfg = json.loads(config.read_text(encoding="utf-8"))
    out_dir = cfg["run"]["model_output_dir"]
    cmd = [
        str(PY),
        str(TRAIN),
        "--config",
        str(config.relative_to(REPO)),
        "--model_output_dir",
        out_dir,
    ]
    cb = cfg["run"].get("callback_root_dir")
    if cb:
        cmd.extend(["--callback_root_dir", cb])
    ep = cfg["run"].get("epochs")
    if ep is not None:
        cmd.extend(["--epochs", str(ep)])
    env = os.environ.copy()
    if force:
        env["STEPCOVNET_FORCE_GPU"] = "1"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    epochs_total = int(ep) if ep is not None else None
    epoch = 0
    last_val: dict[str, float] = {}

    print(f"log file: {log_path.relative_to(REPO)}")
    print(f"status:   logs/ar_tide_iter/status/{exp_id}.json")
    print(
        f"watch:    venv\\Scripts\\python.exe scripts/ar_tide_iter/show_status.py --id {exp_id} --watch"
    )
    sys.stdout.flush()

    acquire_training_lock(exp_id)
    try:
        with log_path.open("w", encoding="utf-8") as logf:
            proc = subprocess.Popen(
                cmd,
                cwd=REPO,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=env,
                bufsize=1,
            )
            assert proc.stdout is not None
            for raw_line in proc.stdout:
                logf.write(raw_line)
                logf.flush()
                clean = sanitize_line(raw_line)
                if clean.startswith("Epoch "):
                    parts = clean.split()
                    if len(parts) >= 2 and "/" in parts[1]:
                        cur, total = parts[1].split("/", 1)
                        epoch = int(cur)
                        epochs_total = int(total)
                if "val_overfit_gate:" in clean:
                    last_val.update(extract_val_metrics(clean))
                write_status(
                    exp_id,
                    {
                        "updated_at": datetime.now().isoformat(timespec="seconds"),
                        "log_path": str(log_path.relative_to(REPO)).replace("\\", "/"),
                        "epoch": epoch,
                        "epochs_total": epochs_total,
                        "running": True,
                        **({"last_val": last_val} if last_val else {}),
                    },
                )
                if is_worth_printing(raw_line):
                    print(clean, flush=True)
            proc.wait()
    finally:
        release_training_lock(exp_id)

    write_status(
        exp_id,
        {
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "log_path": str(log_path.relative_to(REPO)).replace("\\", "/"),
            "epoch": epoch,
            "epochs_total": epochs_total,
            "running": False,
            "train_exit": proc.returncode,
            **({"last_val": last_val} if last_val else {}),
        },
    )
    return proc.returncode


def _append_logs(entry: dict) -> None:
    ITER_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True)
    with RESULTS_JSONL.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    ts = entry["timestamp"]
    exp_id = entry["id"]
    attempt = int(entry.get("attempt", 1))
    kind = entry.get("kind", run_kind(attempt=attempt, retry_reason=""))
    block = (
        f"\n{format_log_heading(exp_id, attempt, ts)}\n\n"
        f"**Hypothesis:** {entry.get('notes', '')}\n\n"
        f"| | |\n|--|--|\n"
        f"| Kind | {kind} |\n"
        f"| Attempt | {attempt} |\n"
        f"| Registry | `scripts/ar_tide_iter/experiments.json` (`{exp_id}`) |\n"
        f"| Config snapshot | `{entry['config']}` |\n"
        f"| Model | `{entry['model_path']}` |\n"
        f"| Train exit | {entry.get('train_exit', 'skipped')} |\n"
        f"| Train log | `{entry.get('train_log', '')}` |\n"
    )
    if entry.get("error"):
        block += f"| Error | {entry['error']} |\n"
    else:
        block += (
            f"| Teacher ordered | {entry['teacher']} |\n"
            f"| Free-run ordered | **{entry['free_run']}** |\n"
            f"| Decode steps | {entry.get('decode_steps', '?')} |\n"
            f"| Eval wall (s) | {entry.get('eval_wall_sec', '?')} |\n"
        )
    if entry.get("passed"):
        block += "\n**PASS — 634/634 free-run.**\n"

    for path in (LOG_MD, DOC_LOG):
        if path == DOC_LOG and not path.parent.is_dir():
            continue
        if not path.is_file():
            header = (
                "# AR tide overfit iteration log (agent session)\n\n"
                "Goal: free-run ordered **634/634 @ 20 ms**. "
                "Machine logs: `logs/ar_tide_iter/` (gitignored).\n\n"
                f"Started: {ts}\n"
            )
            path.write_text(header, encoding="utf-8")
        with path.open("a", encoding="utf-8") as f:
            f.write(block)


def main() -> int:
    args = _parse_args()
    if args.build_all:
        for path in config_builder.write_all_configs():
            print(path.relative_to(REPO))
        return 0

    attempt = count_logged_attempts(args.id) + 1
    config, recipe_changed = _resolve_config(args, attempt)

    notes = args.notes
    if not notes:
        try:
            notes = config_builder.get_experiment(args.id).get("notes", "")
        except KeyError:
            notes = ""

    kind = _finalize_kind(
        attempt=attempt,
        retry_reason=args.retry_reason,
        recipe_changed=recipe_changed,
        reuse_last_config=args.reuse_last_config,
    )

    if not config.is_file():
        print(f"config not found: {config}", file=sys.stderr)
        return 1

    cfg = json.loads(config.read_text(encoding="utf-8"))
    model_path = pathlib.Path(cfg["run"]["model_output_dir"]) / "ar_onset_model.keras"
    if not model_path.is_absolute():
        model_path = (REPO / model_path).resolve()
    train_log = train_log_path(args.id, attempt)
    train_exit: int | str = "skipped"

    def _log_entry(**extra: object) -> dict:
        base = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "id": args.id,
            "attempt": attempt,
            "kind": kind,
            "recipe_changed": recipe_changed,
            "reuse_last_config": args.reuse_last_config,
            "config": str(config.relative_to(REPO)),
            "model_path": str(model_path.relative_to(REPO)),
            "notes": notes,
            "train_exit": train_exit,
            "train_log": str(train_log.relative_to(REPO)),
        }
        base.update(extra)
        return base

    if not args.skip_train:
        print(f"[{args.id}] training -> {model_path.parent}")
        try:
            assert_gpu_training_available(exp_id=args.id, force=args.force)
        except RuntimeError as exc:
            print(f"GPU busy: {exc}", file=sys.stderr)
            return 1
        train_exit = _train(config, train_log, exp_id=args.id, force=args.force)
        if train_exit != 0:
            print(f"train failed exit={train_exit}", file=sys.stderr)
        if not model_path.is_file():
            _append_logs(_log_entry(error="checkpoint missing after train"))
            return 1

    print(f"[{args.id}] eval teacher-fed")
    teacher_report = _eval(config, model_path, ar_decode=False)
    teacher = teacher_report.get("ordered_onset_match", {})
    t_str = f"{teacher.get('n_matched')}/{teacher.get('n_denom')} ({teacher.get('rate'):.4f})"
    event_f1 = float(teacher_report.get("event_f1", 0.0))

    if not teacher_report_perfect(teacher_report):
        msg = (
            "teacher metrics not perfect "
            f"(ordered={t_str}, event_f1={event_f1:.4f}); "
            "skipped free-run eval"
        )
        print(msg, file=sys.stderr)
        _append_logs(
            _log_entry(
                error=msg,
                teacher=t_str,
                teacher_event_f1=round(event_f1, 6),
                teacher_gate_failed=True,
            ),
        )
        return 1

    print(f"[{args.id}] eval free-run")
    report = _eval(config, model_path, ar_decode=True)
    free = report["ar_decode"]["ordered_onset_match"]
    f_str = f"{free.get('n_matched')}/{free.get('n_denom')} ({free.get('rate'):.4f})"
    passed = (
        int(free.get("n_matched", 0)) == 634
        and int(free.get("n_denom", 0)) == 634
        and float(free.get("rate", 0)) >= 1.0
    )

    _append_logs(
        _log_entry(
            teacher=t_str,
            free_run=f_str,
            free_run_matched=int(free.get("n_matched", 0)),
            free_run_denom=int(free.get("n_denom", 0)),
            decode_steps=report["ar_decode"].get("ar_decode_length"),
            eval_wall_sec=report.get("_eval_wall_sec"),
            passed=passed,
        ),
    )
    print(f"teacher {t_str} | free-run {f_str} | passed={passed}")
    return 0 if passed else (2 if train_exit != 0 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
