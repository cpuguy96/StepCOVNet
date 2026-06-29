r"""Script for training the arrow detection model.

Usage:
    # Required: config file. Optional: overrides via --set key=value (repeatable).
    python scripts/train_arrow.py --config=configs/arrow/baseline.json
    python scripts/train_arrow.py --config=configs/arrow/baseline.json --set run.epoch=30 --set dataset.batch_size=4
    python scripts/train_arrow.py --config=configs/arrow/baseline.json --set model.lstm.units=256 --set model.model_type=lstm

    Dotted paths: dataset.*, model.*, run.* (e.g. run.epoch, dataset.data_dir, model.transformer.num_layers).
"""

import argparse

import tensorflow as tf

from stepcovnet import config, trainers

PARSER = argparse.ArgumentParser(description="Train arrow detection model.")
PARSER.add_argument(
    "--config",
    type=str,
    required=True,
    help="Path to JSON config file.",
)
PARSER.add_argument(
    "--set",
    type=str,
    action="append",
    default=[],
    metavar="KEY=VALUE",
    help="Override config (repeatable). Example: --set run.epoch=30 --set model.lstm.units=256",
)
ARGS = PARSER.parse_args()


def _coerce_value(s: str):
    """Coerce string to int, float, bool, or leave as str."""
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    return s


def _set_nested(d: dict, key_path: str, value) -> None:
    """Set a possibly nested key (e.g. 'transformer.num_layers') in d, creating dicts as needed.
    Key path components must be non-empty; otherwise no-op (avoids empty string keys that
    break config reconstruction).
    Missing or None intermediate segments are created as empty dicts so nested keys can be set.
    Raises ValueError if an intermediate segment exists and is a non-dict, non-None value (e.g.
    run.epoch.foo=bar when epoch is an integer).
    """
    parts = key_path.split(".")
    if not parts or any(not p for p in parts):
        return
    current = d
    for i, part in enumerate(parts[:-1]):
        existing = current.get(part)
        if part not in current or existing is None:
            current[part] = {}
        elif not isinstance(existing, dict):
            segment = ".".join(parts[: i + 1])
            raise ValueError(
                f"Cannot set nested key '{key_path}': segment '{segment}' is not a "
                "nested object (leaf value); use a path without extra segments."
            )
        current = current[part]
    current[parts[-1]] = value


def apply_overrides_from_cli(
    base: config.ArrowExperimentConfig,
    overrides: list[str],
) -> config.ArrowExperimentConfig:
    """Apply CLI overrides (key=value strings) to base config with string coercion."""
    if not overrides:
        return base
    d = base.as_dict()
    for item in overrides:
        if "=" not in item:
            continue
        key, _, value_str = item.partition("=")
        key = key.strip()
        value_str = value_str.strip()
        if "." not in key:
            continue
        prefix, rest = key.split(".", 1)
        if prefix not in ("dataset", "model", "run"):
            continue
        if not rest or any(not part for part in rest.split(".")):
            continue
        value = _coerce_value(value_str)
        _set_nested(d[prefix], rest, value)
    return config.ArrowExperimentConfig.from_dict(d)


if tf.config.list_physical_devices("GPU"):
    import keras

    print("Training with GPU.")

    keras.mixed_precision.set_global_policy(
        keras.mixed_precision.Policy("mixed_float16")
    )

    # Enable XLA (Accelerated Linear Algebra) for TensorFlow, which can improve
    # performance by compiling TensorFlow graphs into highly optimized
    # machine code.
    tf.config.optimizer.set_jit("autoclustering")


def main():
    if not ARGS.config:
        PARSER.error("--config is required")
    experiment_config = config.ArrowExperimentConfig.from_json(ARGS.config)
    experiment_config = apply_overrides_from_cli(experiment_config, ARGS.set or [])

    if (
        not experiment_config.dataset.data_dir
        or not experiment_config.dataset.val_data_dir
    ):
        PARSER.error("dataset.data_dir and dataset.val_data_dir are required")
    if not experiment_config.run.model_output_dir:
        PARSER.error("run.model_output_dir is required")

    trainers.run_arrow_train_from_config(experiment_config)


if __name__ == "__main__":
    main()
