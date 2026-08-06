"""Audio-grounding ablation gate for AR onset checkpoints.

Teacher-forced eval under corrupted encoder input. Pointer timing and token
accuracy must collapse under shuffle/zeros or the run does not count.

    venv\\Scripts\\python.exe scripts/audio_ablation_ar_onset.py \\
        --config configs/ar/ladder_50t_50v_content_pointer.json --split val --limit 12
    venv\\Scripts\\python.exe scripts/audio_ablation_ar_onset.py \\
        --config configs/ar/tide_overfit_content_pointer.json --split overfit --gate
"""

from __future__ import annotations

import argparse
import json
import pathlib

from stepcovnet import wsl_gpu

SCRIPT_REL = "scripts/audio_ablation_ar_onset.py"
wsl_gpu.bootstrap_gpu_script(SCRIPT_REL)

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import audio_ablation, config, datasets

VARIANTS = audio_ablation.VARIANTS


def _load_batch(
    experiment_config: config.ArExperimentConfig,
    sample: tuple[str, str, int] | None,
) -> dict[str, np.ndarray]:
    if sample is None:
        return datasets.sample_to_training_batch(
            datasets.load_overfit_sample(experiment_config),
            experiment_config,
        )
    audio_path, chart_path, chart_index = sample
    loaded = datasets.load_ar_sample(
        audio_path,
        chart_path,
        dataset_config=experiment_config.dataset,
        model_config=experiment_config.model,
        vocab=experiment_config.build_vocab(),
        chart_index=chart_index,
    )
    return datasets.sample_to_training_batch(loaded, experiment_config)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="")
    parser.add_argument("--split", default="val")
    parser.add_argument("--limit", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Exit 1 when shuffle/zeros reproduce matched scores.",
    )
    args = parser.parse_args()

    wsl_gpu.guard_tensorflow_gpu_job(__file__)
    experiment_config = config.ArExperimentConfig.from_json(args.config)
    model_path = args.model_path or str(
        pathlib.Path(experiment_config.run.model_output_dir) / "ar_onset_model.keras",
    )
    model = tf.keras.models.load_model(model_path, compile=False)

    if args.split == "overfit":
        samples = [None]
    else:
        samples = datasets._filter_valid_ar_samples(  # noqa: SLF001
            datasets.list_ar_training_samples(experiment_config, split=args.split),
            experiment_config.dataset,
        )
        if args.limit > 0:
            samples = samples[: args.limit]
    print(f"model: {model_path}")
    print(f"songs: {len(samples)} ({args.split})")

    rng = np.random.default_rng(args.seed)
    totals = audio_ablation.empty_variant_totals()

    for i, sample in enumerate(samples):
        batch = _load_batch(experiment_config, sample)
        donor_sample = samples[(i + 1) % len(samples)]
        donor_batch = _load_batch(experiment_config, donor_sample)
        donor_patches = donor_batch["mert_patches"]
        donor_valid = int(donor_batch["patch_mask"][0].sum())
        n_valid = int(batch["patch_mask"][0].sum())
        base_patches = batch["mert_patches"]

        matched_pred: np.ndarray | None = None
        matched_token_preds: np.ndarray | None = None
        matched_token_targets: np.ndarray | None = None
        matched_query: np.ndarray | None = None
        for variant in VARIANTS:
            batch["mert_patches"] = audio_ablation.corrupt_patches(
                base_patches,
                n_valid,
                variant,
                donor_patches,
                donor_valid,
                rng,
            )
            row = audio_ablation.score_batch(
                model,
                batch,
                experiment_config=experiment_config,
            )
            if variant == "matched":
                matched_pred = row["_pred_patches"]
                matched_token_preds = row["_token_preds"]
                matched_token_targets = row["_token_targets"]
                matched_query = row.get("_pointer_query")
            audio_ablation.accumulate_variant_row(
                totals,
                variant,
                row,
                matched_pred_patches=matched_pred,
                matched_token_preds=matched_token_preds,
                matched_token_targets=matched_token_targets,
                matched_pointer_query=matched_query
                if isinstance(matched_query, np.ndarray)
                else None,
            )
        batch["mert_patches"] = base_patches
        label = "overfit" if sample is None else pathlib.Path(sample[0]).stem
        print(f"  [{i + 1}/{len(samples)}] {label}")

    single_song = len(samples) < 2
    rows = audio_ablation.summarize_variants(
        totals,
        skip_cross_song=single_song,
    )
    gate = audio_ablation.audio_grounding_gate(rows)

    print()
    header = (
        f"{'variant':<11} {'F1':>7} {'timing':>8} {'tok_acc':>8} "
        f"{'patch_wrong':>12} {'ptr_nll':>8} {'same_ptr':>9} {'same_tok':>9}"
    )
    print(header)
    print("-" * len(header))
    for variant in VARIANTS:
        if variant == "cross_song" and single_song:
            print(f"{'cross_song':<11} {'(skipped: donor would be the same song)':>48}")
            continue
        r = rows[variant]
        print(
            f"{variant:<11} {r['f1_hungarian']:>7.4f} {r['timing_match']:>8.4f} "
            f"{r['token_accuracy']:>8.4f} {r['patch_wrong_rate']:>12.4f} "
            f"{r['pointer_nll']:>8.2f} {r['same_pred_as_matched']:>9.4f} "
            f"{r['same_token_as_matched']:>9.4f}",
        )

    print()
    status = "PASS" if gate.passed else "FAIL"
    print(
        f"audio_grounding_gate: {status} "
        f"(pointer={'PASS' if gate.pointer_passed else 'FAIL'}, "
        f"token={'PASS' if gate.token_passed else 'FAIL'}, "
        f"query={'PASS' if gate.query_passed else 'FAIL'})",
    )
    for failure in gate.failures:
        print(f"  - {failure}")

    payload = {
        "model": model_path,
        "config": args.config,
        "split": args.split,
        "variants": rows,
        "gate": {
            "passed": gate.passed,
            "pointer_passed": gate.pointer_passed,
            "token_passed": gate.token_passed,
            "query_passed": gate.query_passed,
            "failures": list(gate.failures),
        },
    }
    out_path = pathlib.Path(args.out) if args.out else None
    if out_path is None:
        label = pathlib.Path(args.config).stem
        out_path = pathlib.Path("logs") / f"audio_ablation_{label}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")

    if args.gate and not gate.passed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
