r"""Precompute MERT features for onset model training.

Extracts MERT hidden states from audio files paired with StepMania charts and
writes ``.mert.npy`` tensors aligned to the onset frame grid (HOP_COEFF).

Requires optional dependencies: pip install '.[ssl]'

Usage:
    python scripts/extract_mert_features.py --data_dir=data/train --output_dir=data/mert/train
    python scripts/extract_mert_features.py --data_dir=data/val --output_dir=data/mert/val --device=cuda
    python scripts/extract_mert_features.py \
        --training_index_path=data/final_data/training_index.json \
        --beside_audio --device=cuda --skip_existing
"""

from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

from stepcovnet import pairing, ssl_features, wsl_gpu

SCRIPT_REL = "scripts/extract_mert_features.py"

PARSER = argparse.ArgumentParser(
    description="Extract MERT features for StepCOVNet onset training."
)
PARSER.add_argument(
    "--data_dir",
    type=str,
    default="",
    help="Directory containing audio and chart files (legacy .txt layout).",
)
PARSER.add_argument(
    "--training_index_path",
    type=str,
    default="",
    help="Path to training_index.json or prepared final_data root (multi-chart JSON).",
)
PARSER.add_argument(
    "--split",
    type=str,
    choices=("all", "train", "val"),
    default="all",
    help="Manifest split filter when using --training_index_path (default: all).",
)
PARSER.add_argument(
    "--output_dir",
    type=str,
    default="",
    help="Directory where .mert.npy files are written (mirrors relative paths).",
)
PARSER.add_argument(
    "--beside_audio",
    action="store_true",
    help="Write each .mert.npy beside its audio file (ignores --output_dir).",
)
PARSER.add_argument(
    "--model_name",
    type=str,
    default=ssl_features.DEFAULT_MERT_MODEL,
    help="Hugging Face MERT model id.",
)
PARSER.add_argument(
    "--layer",
    type=int,
    default=ssl_features.DEFAULT_MERT_LAYER,
    help="Hidden-state layer index to extract.",
)
PARSER.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Torch device for inference (cpu or cuda).",
)
PARSER.add_argument(
    "--chunk_seconds",
    type=float,
    default=ssl_features.MERT_CHUNK_SECONDS,
    help="Chunk length in seconds for long audio files.",
)
PARSER.add_argument(
    "--skip_existing",
    action="store_true",
    help="Skip audio files whose output .mert.npy already exists.",
)


def _full_argv(argv: list[str] | None) -> list[str]:
    """Build a full argv list with the script path for WSL dispatch.

    Args:
        argv: Optional CLI args without the script name (defaults to sys.argv[1:]).

    Returns:
        Full argv including absolute script path at index 0.
    """
    cli_argv = argv if argv is not None else sys.argv[1:]
    script_path = str(pathlib.Path(__file__).resolve())
    return [script_path, *cli_argv]


def _resolve_audio_paths(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return unique audio paths and data root from CLI args."""
    index_ref = str(args.training_index_path).strip()
    data_dir = str(args.data_dir).strip()
    if index_ref:
        split = None if args.split == "all" else args.split
        return pairing.list_unique_audio_paths(index_ref, split=split)
    if not data_dir:
        raise SystemExit(
            "Provide --training_index_path or --data_dir.",
        )
    return pairing.list_unique_audio_paths(data_dir)


def _output_path(
    audio_path: str,
    *,
    beside_audio: bool,
    output_dir: str,
    data_root: str,
) -> str:
    if beside_audio:
        return ssl_features.mert_npy_path(audio_path, "", data_root)
    if not output_dir:
        raise SystemExit("--output_dir is required unless --beside_audio is set.")
    return ssl_features.mert_npy_path(
        audio_path,
        output_dir,
        data_root,
    )


def _configure_quiet_hf_logs() -> None:
    """Reduce Hugging Face / hub progress noise during batch extraction."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers.utils import logging as hf_logging  # noqa: PLC0415

        hf_logging.set_verbosity_error()
    except ImportError:
        pass


def _display_path(path: str, root: str) -> str:
    """Return a short, human-readable path for log lines."""
    try:
        return os.path.relpath(path, root)
    except ValueError:
        return pathlib.Path(path).name


def _format_bytes(size: int) -> str:
    """Format a byte count for log output."""
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KiB"
    return f"{size / (1024 * 1024):.1f} MiB"


def _format_duration(seconds: float) -> str:
    """Format elapsed seconds for log output."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    if minutes < 60:
        return f"{minutes}m {remainder}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m {remainder}s"


def _log(message: str) -> None:
    print(message, flush=True)


def _log_plan(
    *,
    total: int,
    pending: int,
    skipped: int,
    args: argparse.Namespace,
    data_root: str,
) -> None:
    if args.training_index_path:
        source = f"manifest {_display_path(args.training_index_path, str(pathlib.Path.cwd()))}"
        if args.split != "all":
            source += f" ({args.split} split)"
    else:
        source = f"data dir {_display_path(args.data_dir, str(pathlib.Path.cwd()))}"
    if args.beside_audio:
        output = (
            f"beside audio under {_display_path(data_root, str(pathlib.Path.cwd()))}"
        )
    else:
        output = _display_path(args.output_dir, str(pathlib.Path.cwd()))

    _log("MERT extraction")
    _log(f"  source:  {source}")
    _log(f"  audios:  {total} total ({pending} to extract, {skipped} cached)")
    _log(f"  model:   {args.model_name} (layer {args.layer})")
    _log(f"  device:  {args.device}")
    _log(f"  output:  {output}")
    if pending:
        _log("")


def _log_extract_result(
    *,
    index: int,
    total: int,
    label: str,
    elapsed_sec: float,
    output_path: str,
) -> None:
    out_path = pathlib.Path(output_path)
    if out_path.is_file():
        size_text = _format_bytes(out_path.stat().st_size)
    else:
        size_text = "saved"
    _log(f"[{index}/{total}] ok  {label}  {_format_duration(elapsed_sec)}  {size_text}")


def main(argv: list[str] | None = None) -> None:
    wsl_gpu.bootstrap_gpu_script(SCRIPT_REL, _full_argv(argv), dispatch="mert")
    _configure_quiet_hf_logs()
    args = PARSER.parse_args(argv)
    try:
        audio_paths, data_root = _resolve_audio_paths(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    if not args.beside_audio:
        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    pending_jobs: list[tuple[str, str, str]] = []
    skipped = 0
    for audio_path in audio_paths:
        output_path = _output_path(
            audio_path,
            beside_audio=args.beside_audio,
            output_dir=args.output_dir,
            data_root=data_root,
        )
        label = _display_path(audio_path, data_root)
        if args.skip_existing and pathlib.Path(output_path).is_file():
            skipped += 1
            continue
        pending_jobs.append((audio_path, output_path, label))

    _log_plan(
        total=len(audio_paths),
        pending=len(pending_jobs),
        skipped=skipped,
        args=args,
        data_root=data_root,
    )

    if not pending_jobs:
        _log("Nothing to do — all MERT caches already present.")
        return

    if wsl_gpu.device_requests_gpu(args.device):
        wsl_gpu.assert_wsl_gpu_free_for_training()
        wsl_gpu.guard_gpu_device_job(args.device, __file__)
        _extract_pending_jobs(pending_jobs, args, skipped=skipped)
        return
    _extract_pending_jobs(pending_jobs, args, skipped=skipped)


def _extract_pending_jobs(
    pending_jobs: list[tuple[str, str, str]],
    args: argparse.Namespace,
    *,
    skipped: int,
) -> None:
    batch_started = time.perf_counter()
    _log(f"Loading MERT model on {args.device}...")
    model_load_started = time.perf_counter()
    model, processor = ssl_features._load_mert_model(args.model_name, args.device)
    _log(f"Model ready ({_format_duration(time.perf_counter() - model_load_started)}).")
    _log("")

    extracted = 0
    failed: list[tuple[str, str]] = []
    pending_total = len(pending_jobs)
    for index, (audio_path, output_path, label) in enumerate(pending_jobs, start=1):
        started = time.perf_counter()
        try:
            saved_path = ssl_features.extract_and_save_mert_features(
                audio_path,
                output_path,
                model_name=args.model_name,
                layer=args.layer,
                device=args.device,
                chunk_seconds=args.chunk_seconds,
                model=model,
                processor=processor,
            )
        except Exception as exc:
            _log(f"[{index}/{pending_total}] FAIL  {label}  {exc}")
            failed.append((label, str(exc)))
            continue
        extracted += 1
        _log_extract_result(
            index=index,
            total=pending_total,
            label=label,
            elapsed_sec=time.perf_counter() - started,
            output_path=saved_path,
        )

    elapsed = time.perf_counter() - batch_started
    _log("")
    _log("Summary")
    _log(f"  extracted: {extracted}")
    if skipped:
        _log(f"  skipped:   {skipped} (already cached)")
    if failed:
        _log(f"  failed:    {len(failed)}")
        for label, reason in failed[:5]:
            _log(f"    - {label}: {reason}")
        if len(failed) > 5:
            _log(f"    ... and {len(failed) - 5} more")
    _log(f"  elapsed:   {_format_duration(elapsed)}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
