"""Promote an AR tide-overfit candidate to the champion config when it wins on free-run.

Usage:
    python scripts/graduate_ar_tide_overfit.py \\
        --config configs/ar/versions/tide_overfit/v7.json \\
        --model-path models_wsl/ar/perfect_overfit/run5/ar_onset_model.keras \\
        --version-ref configs/ar/versions/tide_overfit/v7.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
CHAMPION_CONFIG = REPO_ROOT / "configs/ar/tide_overfit.json"
CHAMPION_MANIFEST = REPO_ROOT / "configs/ar/tide_overfit.manifest.json"
CHAMPION_MODEL_DIR = REPO_ROOT / "models_wsl/ar/tide_overfit"
CHAMPION_MODEL = CHAMPION_MODEL_DIR / "ar_onset_model.keras"
DEBUG_SCRIPT = REPO_ROOT / "scripts/eval_ar_onset_offline.py"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, help="Candidate experiment config JSON."
    )
    parser.add_argument(
        "--model-path", required=True, help="Candidate checkpoint to eval/copy."
    )
    parser.add_argument(
        "--version-ref",
        required=True,
        help="Version file path recorded in the manifest (for audit).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compare metrics only; do not update champion files.",
    )
    return parser.parse_args()


def _run_free_run_eval(config: pathlib.Path, model_path: pathlib.Path) -> dict:
    cmd = [
        sys.executable,
        str(DEBUG_SCRIPT),
        "--config",
        str(config.relative_to(REPO_ROOT)),
        "--model_path",
        str(model_path.relative_to(REPO_ROOT)),
        "--ar_decode",
        "--json-only",
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    blob = proc.stdout or proc.stderr
    start = blob.find("{")
    if start < 0:
        raise RuntimeError(f"eval produced no JSON:\n{blob[:2000]}")
    report = json.loads(blob[start:])
    if proc.returncode != 0:
        raise RuntimeError(f"eval failed (exit {proc.returncode}):\n{blob[:2000]}")
    return report


def _ordered_block(report: dict, *, free_run: bool) -> dict:
    if free_run:
        return report["ar_decode"]["ordered_onset_match"]
    block = report.get("ordered_onset_match") or report["timing_match"]
    return block


def _rate_key(block: dict) -> tuple[float, int, int]:
    return (
        float(block["rate"]),
        int(block["n_matched"]),
        -abs(int(block["n_denom"]) - 634),
    )


def _load_manifest() -> dict:
    if not CHAMPION_MANIFEST.is_file():
        return {"free_run": {"rate": -1.0, "n_matched": 0, "n_denom": 634}}
    return json.loads(CHAMPION_MANIFEST.read_text(encoding="utf-8"))


def _graduate_config(candidate: pathlib.Path) -> dict:
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    run = payload.setdefault("run", {})
    run["model_output_dir"] = "models_wsl/ar/tide_overfit"
    run["callback_root_dir"] = "callbacks/ar/tide_overfit"
    return payload


def main() -> int:
    args = _parse_args()
    config = pathlib.Path(args.config).resolve()
    model_path = pathlib.Path(args.model_path).resolve()
    version_ref = pathlib.Path(args.version_ref).as_posix()

    if not config.is_file():
        print(f"config not found: {config}", file=sys.stderr)
        return 1
    if not model_path.is_file():
        print(f"model not found: {model_path}", file=sys.stderr)
        return 1

    report = _run_free_run_eval(config, model_path)
    candidate_free = _ordered_block(report, free_run=True)
    candidate_teacher = _ordered_block(report, free_run=False)
    manifest = _load_manifest()
    champion_free = manifest.get(
        "free_run", {"rate": -1.0, "n_matched": 0, "n_denom": 634}
    )

    print(
        "candidate free-run:",
        f"{candidate_free['n_matched']}/{candidate_free['n_denom']}",
        f"rate={candidate_free['rate']:.6f}",
    )
    print(
        "champion free-run:",
        f"{champion_free.get('n_matched', '?')}/{champion_free.get('n_denom', '?')}",
        f"rate={champion_free.get('rate', -1):.6f}",
    )

    if _rate_key(candidate_free) <= _rate_key(champion_free):
        print(
            "No promotion: candidate does not beat champion on free-run ordered match."
        )
        return 0

    if args.dry_run:
        print("Dry run: would promote candidate to champion.")
        return 0

    champion_payload = _graduate_config(config)
    CHAMPION_CONFIG.write_text(
        json.dumps(champion_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    CHAMPION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, CHAMPION_MODEL)

    new_manifest = {
        "graduated_from": version_ref,
        "updated": __import__("datetime").date.today().isoformat(),
        "checkpoint": "models_wsl/ar/tide_overfit/ar_onset_model.keras",
        "primary_metric": "ar_decode_ordered_onset_match",
        "free_run": {
            "n_matched": int(candidate_free["n_matched"]),
            "n_denom": int(candidate_free["n_denom"]),
            "rate": float(candidate_free["rate"]),
        },
        "teacher": {
            "n_matched": int(candidate_teacher["n_matched"]),
            "n_denom": int(candidate_teacher["n_denom"]),
            "rate": float(candidate_teacher["rate"]),
        },
    }
    CHAMPION_MANIFEST.write_text(
        json.dumps(new_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Promoted to {CHAMPION_CONFIG.relative_to(REPO_ROOT)}")
    print(f"Copied checkpoint to {CHAMPION_MODEL.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
