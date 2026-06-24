"""Configuration for raw simfile pack preprocessing."""

from __future__ import annotations

import dataclasses
import enum
import json
import pathlib

from stepcovnet.dataset_prep import constants


class _DictSerializableMixin:
    """Mixin providing default as_dict and from_dict for dataclass configs."""

    def as_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization.

        Returns:
            Serializable mapping of dataclass fields.
        """
        return dataclasses.asdict(self)  # type: ignore[arg-type]

    @classmethod
    def from_dict(cls, data: dict):
        """Create config from dictionary.

        Args:
            data: Serialized field values for the dataclass.

        Returns:
            Instance of the config class with fields taken from data.
        """
        return cls(**data)


class ExportMode(enum.StrEnum):
    """Which dance-single charts to export from each simfile pack.

    Attributes:
        EXPORT_ALL_SINGLES: Export every ``dance-single`` chart in each pack.
    """

    EXPORT_ALL_SINGLES = constants.EXPORT_MODE_ALL_SINGLES


@dataclasses.dataclass
class PrepConfig(_DictSerializableMixin):
    """CLI and JSON settings for ``scripts/preprocess_dataset.py``.

    Attributes:
        input_dir: Raw pack root or single bundle directory.
        output_dir: Processed output root (nested bundle/song layout).
        export_mode: Chart export policy; v1 supports ``export_all_singles`` only.
        max_steps_per_chart: Per-chart step cap; charts above are skipped unless
            ``allow_over_cap`` is True.
        export_legacy_txt: When True, write multi-block v2 ``.txt`` beside JSON.
        workers: Parallel worker count for pack processing.
        dry_run: When True, run discovery and normalization only (no pack writes).
        overwrite: When True, replace existing ``{bundle}/{id}/`` output dirs.
        allow_over_cap: When True, export charts above ``max_steps_per_chart``.
        limit_packs: When set, process only the first N packs (sorted by path).
    """

    input_dir: str = constants.DEFAULT_INPUT_DIR
    output_dir: str = constants.DEFAULT_OUTPUT_DIR
    export_mode: ExportMode = ExportMode.EXPORT_ALL_SINGLES
    max_steps_per_chart: int = constants.MAX_STEPS_PER_CHART
    export_legacy_txt: bool = False
    workers: int = 1
    dry_run: bool = False
    overwrite: bool = False
    allow_over_cap: bool = False
    limit_packs: int | None = None


def default_prep_config() -> PrepConfig:
    """Return documented default prep settings.

    Returns:
        PrepConfig with documented default field values.
    """
    return PrepConfig()


def validate_prep_config(config: PrepConfig) -> None:
    """Validate prep config before a batch run.

    Args:
        config: Prep settings to check.

    Raises:
        ValueError: If any field is out of range or unsupported in v1.
    """
    if config.workers < 1:
        raise ValueError(f"workers must be >= 1, got {config.workers}")
    if config.max_steps_per_chart < 1:
        raise ValueError(
            f"max_steps_per_chart must be >= 1, got {config.max_steps_per_chart}"
        )
    if config.limit_packs is not None and config.limit_packs < 1:
        raise ValueError(f"limit_packs must be >= 1, got {config.limit_packs}")
    if config.export_mode != ExportMode.EXPORT_ALL_SINGLES:
        raise ValueError(
            f"unsupported export_mode in v1: {config.export_mode!r}; "
            f"use {ExportMode.EXPORT_ALL_SINGLES!r}"
        )


def load_prep_config_json(path: str) -> PrepConfig:
    """Load prep config from a JSON file.

    Args:
        path: Path to a JSON object with PrepConfig field names.

    Returns:
        Parsed PrepConfig after validation.
    """
    path_obj = pathlib.Path(path)
    with path_obj.open(encoding="utf-8") as handle:
        data = json.load(handle)
    if "export_mode" in data and isinstance(data["export_mode"], str):
        data = dict(data)
        data["export_mode"] = ExportMode(data["export_mode"])
    cfg = PrepConfig.from_dict(data)
    validate_prep_config(cfg)
    return cfg


def save_prep_config_json(config: PrepConfig, path: str) -> None:
    """Write prep config to JSON (parent directories are created if needed).

    Args:
        config: Settings to serialize.
        path: Destination file path.
    """
    payload = config.as_dict()
    payload["export_mode"] = str(config.export_mode)
    path_obj = pathlib.Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
