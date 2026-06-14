r"""Precompute MERT features for onset model training.

Extracts MERT hidden states from audio files paired with StepMania charts and
writes ``.mert.npy`` tensors aligned to the onset frame grid (HOP_COEFF).

Requires optional dependencies: pip install '.[ssl]'

Usage:
    python scripts/extract_mert_features.py --data_dir=data/train --output_dir=data/mert/train
    python scripts/extract_mert_features.py --data_dir=data/val --output_dir=data/mert/val --device=cuda
"""

import argparse
import os
import pathlib
import sys

from stepcovnet import pairing, ssl_features, wsl_gpu

SCRIPT_REL = "scripts/extract_mert_features.py"

PARSER = argparse.ArgumentParser(
    description="Extract MERT features for StepCOVNet onset training."
)
PARSER.add_argument(
    "--data_dir",
    type=str,
    required=True,
    help="Directory containing audio and chart files (same layout as training).",
)
PARSER.add_argument(
    "--output_dir",
    type=str,
    required=True,
    help="Directory where .mert.npy files are written (mirrors relative paths).",
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


def main(argv: list[str] | None = None) -> None:
    wsl_gpu.maybe_dispatch_for_mert_extract(SCRIPT_REL, _full_argv(argv))
    args = PARSER.parse_args(argv)
    pairs = pairing.list_audio_chart_pairs(args.data_dir)
    if not pairs:
        raise SystemExit(f"No audio-chart pairs found under {args.data_dir!r}")

    os.makedirs(args.output_dir, exist_ok=True)
    extracted = 0
    skipped = 0
    failed: list[str] = []
    for audio_path, _chart_path in pairs:
        output_path = ssl_features.mert_npy_path(
            audio_path,
            args.output_dir,
            args.data_dir,
        )
        if args.skip_existing and os.path.isfile(output_path):
            skipped += 1
            continue
        print(f"Extracting {audio_path} -> {output_path}")
        try:
            ssl_features.extract_and_save_mert_features(
                audio_path,
                output_path,
                model_name=args.model_name,
                layer=args.layer,
                device=args.device,
                chunk_seconds=args.chunk_seconds,
            )
        except Exception as exc:
            print(f"FAILED {audio_path}: {exc}")
            failed.append(audio_path)
            continue
        extracted += 1

    print(
        f"Done. Extracted {extracted} file(s)"
        + (f", skipped {skipped} existing." if skipped else ".")
        + (f" Failed {len(failed)}." if failed else "")
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
