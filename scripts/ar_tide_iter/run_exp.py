"""Run one AR tide-overfit iteration experiment.

Usage (repo root, Windows venv):
    venv\\Scripts\\python.exe scripts/ar_tide_iter/run_exp.py --id iter01 \\
        --config logs/ar_tide_iter/configs/iter01.json
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

from stepcovnet import wsl_gpu

REPO = pathlib.Path(__file__).resolve().parents[2]
ITER_DIR = REPO / "logs" / "ar_tide_iter"
LOG_MD = ITER_DIR / "ITER_LOG.md"
RESULTS_JSONL = ITER_DIR / "results.jsonl"
DEBUG = REPO / "scripts" / "debug_ar_onset_overfit.py"
TRAIN = REPO / "scripts" / "train_onset_ar.py"
PY = REPO / "venv" / "Scripts" / "python.exe"
DOC_LOG = REPO / "docs" / "research" / "AR_TIDE_OVERFIT_ITER_LOG.md"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--id", required=True, help="Experiment id, e.g. iter01")
    p.add_argument("--config", required=True, help="Path to experiment JSON config")
    p.add_argument("--skip-train", action="store_true", help="Eval only")
    p.add_argument("--notes", default="", help="Hypothesis / change summary")
    p.add_argument(
        "--force",
        action="store_true",
        help="Start training even if WSL GPU has active compute (sets STEPCOVNET_FORCE_GPU=1)",
    )
    return p.parse_args()


def _parse_eval_json(blob: str) -> dict:
    start = blob.find("{")
    if start < 0:
        raise RuntimeError(f"no JSON in eval output:\n{blob[:1500]}")
    return json.loads(blob[start:])


def _eval(config: pathlib.Path, model_path: pathlib.Path) -> dict:
    cmd = [
        str(PY),
        str(DEBUG),
        "--config",
        str(config.relative_to(REPO)),
        "--model_path",
        str(model_path.relative_to(REPO)),
        "--ar_decode",
        "--json-only",
    ]
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
    block = (
        f"\n### {exp_id} ({ts})\n\n"
        f"**Hypothesis:** {entry.get('notes', '')}\n\n"
        f"| | |\n|--|--|\n"
        f"| Config | `{entry['config']}` |\n"
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
    config = pathlib.Path(args.config).resolve()
    if not config.is_file():
        print(f"config not found: {config}", file=sys.stderr)
        return 1

    cfg = json.loads(config.read_text(encoding="utf-8"))
    model_path = pathlib.Path(cfg["run"]["model_output_dir"]) / "ar_onset_model.keras"
    if not model_path.is_absolute():
        model_path = (REPO / model_path).resolve()
    train_log = ITER_DIR / "train_logs" / f"{args.id}.log"
    train_exit: int | str = "skipped"

    if not args.skip_train:
        print(f"[{args.id}] training -> {model_path.parent}")
        try:
            wsl_gpu.assert_wsl_gpu_free_for_training(force=args.force)
        except RuntimeError as exc:
            print(f"GPU busy: {exc}", file=sys.stderr)
            return 1
        train_exit = _train(config, train_log, exp_id=args.id, force=args.force)
        if train_exit != 0:
            print(f"train failed exit={train_exit}", file=sys.stderr)
        if not model_path.is_file():
            entry = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "id": args.id,
                "config": str(config.relative_to(REPO)),
                "model_path": str(model_path.relative_to(REPO)),
                "notes": args.notes,
                "train_exit": train_exit,
                "train_log": str(train_log.relative_to(REPO)),
                "error": "checkpoint missing after train",
            }
            _append_logs(entry)
            return 1

    print(f"[{args.id}] eval free-run")
    report = _eval(config, model_path)
    teacher = report.get("ordered_onset_match", {})
    free = report["ar_decode"]["ordered_onset_match"]
    t_str = f"{teacher.get('n_matched')}/{teacher.get('n_denom')} ({teacher.get('rate'):.4f})"
    f_str = f"{free.get('n_matched')}/{free.get('n_denom')} ({free.get('rate'):.4f})"
    passed = (
        int(free.get("n_matched", 0)) == 634
        and int(free.get("n_denom", 0)) == 634
        and float(free.get("rate", 0)) >= 1.0
    )

    entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "id": args.id,
        "config": str(config.relative_to(REPO)),
        "model_path": str(model_path.relative_to(REPO)),
        "notes": args.notes,
        "train_exit": train_exit,
        "train_log": str(train_log.relative_to(REPO)),
        "teacher": t_str,
        "free_run": f_str,
        "decode_steps": report["ar_decode"].get("ar_decode_length"),
        "eval_wall_sec": report.get("_eval_wall_sec"),
        "passed": passed,
    }
    _append_logs(entry)
    print(f"teacher {t_str} | free-run {f_str} | passed={passed}")
    return 0 if passed else (2 if train_exit != 0 else 0)


if __name__ == "__main__":
    raise SystemExit(main())
