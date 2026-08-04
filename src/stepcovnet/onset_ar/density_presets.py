"""Customer-facing difficulty tiers mapped to density conditioning scalars."""

from __future__ import annotations

import dataclasses
import json
import pathlib
import statistics

from stepcovnet.onset_ar import config

DEFAULT_PRESETS_PATH = "configs/ar/density_presets.json"
CUSTOMER_TIER_ORDER = (
    "beginner",
    "easy",
    "medium",
    "hard",
    "challenge",
)
DEFAULT_TIER_ORDER = CUSTOMER_TIER_ORDER
CALIBRATION_METHOD_FIXED = "fixed_onset_hz_targets"
CALIBRATION_METHOD_EQUAL_COUNT = "onset_hz_equal_count_quintiles"

# Design targets (onsets/s) for customer tiers — prior-art-style explicit difficulty input.
DEFAULT_FIXED_ONSET_HZ_TARGETS: dict[str, float] = {
    "beginner": 2.0,
    "easy": 4.0,
    "medium": 6.0,
    "hard": 8.0,
    "challenge": 10.0,
}
# Midpoints between adjacent targets; used only for optional corpus coverage reports.
DEFAULT_COVERAGE_THRESHOLDS: tuple[float, ...] = (3.0, 5.0, 7.0, 9.0)


@dataclasses.dataclass(frozen=True)
class DensityTierPreset:
    """Target or calibrated onset rate for one customer difficulty tier.

    Attributes:
        onsets_per_sec_median: Target onsets/s (fixed presets) or bucket median.
        density_scalar: ``onsets_per_sec / onset_hz_norm``, clipped to ``[0, 1]``.
        n_rows: Corpus charts in the coverage band (0 when not scanned).
    """

    onsets_per_sec_median: float
    density_scalar: float
    n_rows: int


@dataclasses.dataclass(frozen=True)
class DensityPresets:
    """Lookup table from difficulty tier to density conditioning."""

    schema_version: int
    onset_hz_norm: float
    source_training_index_path: str
    n_rows_total: int
    created_at: str
    tiers: dict[str, DensityTierPreset]
    tier_order: tuple[str, ...] = DEFAULT_TIER_ORDER
    calibration_method: str = CALIBRATION_METHOD_FIXED

    def onsets_per_sec_for_tier(self, tier: str) -> float:
        """Return the target or calibrated onsets/sec for ``tier``."""
        key = str(tier).strip().lower()
        if key in self.tiers:
            return self.tiers[key].onsets_per_sec_median
        if "medium" in self.tiers:
            return self.tiers["medium"].onsets_per_sec_median
        if not self.tiers:
            raise KeyError(f"no density presets loaded for tier {tier!r}")
        return next(iter(self.tiers.values())).onsets_per_sec_median

    def density_scalar_for_tier(self, tier: str) -> float:
        """Return ``density_scalar`` for a customer difficulty tier."""
        key = str(tier).strip().lower()
        if key in self.tiers:
            return self.tiers[key].density_scalar
        if "medium" in self.tiers:
            return self.tiers["medium"].density_scalar
        if not self.tiers:
            raise KeyError(f"no density presets loaded for tier {tier!r}")
        return next(iter(self.tiers.values())).density_scalar

    def as_dict(self) -> dict:
        """Serialize to a JSON-compatible dict."""
        return {
            "schema_version": self.schema_version,
            "onset_hz_norm": self.onset_hz_norm,
            "source": {
                "calibration_method": self.calibration_method,
                "training_index_path": self.source_training_index_path,
                "n_rows": self.n_rows_total,
                "created_at": self.created_at,
                "coverage_thresholds_hz": list(DEFAULT_COVERAGE_THRESHOLDS),
            },
            "tier_order": list(self.tier_order),
            "tiers": {
                name: {
                    "onsets_per_sec_median": preset.onsets_per_sec_median,
                    "density_scalar": preset.density_scalar,
                    "n_rows": preset.n_rows,
                }
                for name, preset in self.tiers.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> DensityPresets:
        """Load presets from a JSON object."""
        tiers_raw = data.get("tiers") or {}
        tiers = {
            str(name).strip().lower(): DensityTierPreset(
                onsets_per_sec_median=float(item["onsets_per_sec_median"]),
                density_scalar=float(item["density_scalar"]),
                n_rows=int(item.get("n_rows", 0)),
            )
            for name, item in tiers_raw.items()
        }
        source = data.get("source") or {}
        tier_order = tuple(data.get("tier_order") or DEFAULT_TIER_ORDER)
        return cls(
            schema_version=int(data.get("schema_version", 1)),
            onset_hz_norm=float(data.get("onset_hz_norm", 15.0)),
            source_training_index_path=str(
                source.get("training_index_path", ""),
            ),
            n_rows_total=int(source.get("n_rows", 0)),
            created_at=str(source.get("created_at", "")),
            tiers=tiers,
            tier_order=tier_order,
            calibration_method=str(
                source.get("calibration_method", CALIBRATION_METHOD_FIXED),
            ),
        )


def build_fixed_tier_presets(
    *,
    tier_order: tuple[str, ...] = CUSTOMER_TIER_ORDER,
    onset_hz_targets: dict[str, float] | None = None,
    onset_hz_norm: float = 15.0,
    coverage_counts: dict[str, int] | None = None,
) -> dict[str, DensityTierPreset]:
    """Build customer tiers from fixed design onsets/sec targets."""
    targets = onset_hz_targets or DEFAULT_FIXED_ONSET_HZ_TARGETS
    counts = coverage_counts or {}
    tier_stats: dict[str, DensityTierPreset] = {}
    for tier in tier_order:
        hz = float(targets[tier])
        tier_stats[tier] = DensityTierPreset(
            onsets_per_sec_median=hz,
            density_scalar=config.density_scalar_from_onsets_per_sec(
                hz,
                onset_hz_norm=onset_hz_norm,
            ),
            n_rows=int(counts.get(tier, 0)),
        )
    return tier_stats


def tier_for_onsets_per_sec(
    hz: float,
    *,
    tier_order: tuple[str, ...] = CUSTOMER_TIER_ORDER,
    thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> str:
    """Map a measured onset rate to the nearest customer tier band for coverage reports."""
    value = float(hz)
    cuts = thresholds
    tiers = tier_order
    if value < cuts[0]:
        return tiers[0]
    if value < cuts[1]:
        return tiers[1]
    if value < cuts[2]:
        return tiers[2]
    if value < cuts[3]:
        return tiers[3]
    return tiers[4]


def coverage_counts_for_onsets_per_sec(
    onsets_per_sec: list[float],
    *,
    tier_order: tuple[str, ...] = CUSTOMER_TIER_ORDER,
    thresholds: tuple[float, ...] = DEFAULT_COVERAGE_THRESHOLDS,
) -> dict[str, int]:
    """Count how many corpus charts fall in each fixed-target band."""
    counts = dict.fromkeys(tier_order, 0)
    for hz in onsets_per_sec:
        tier = tier_for_onsets_per_sec(
            hz,
            tier_order=tier_order,
            thresholds=thresholds,
        )
        counts[tier] += 1
    return counts


def build_fixed_density_presets(
    *,
    onset_hz_norm: float = 15.0,
    onset_hz_targets: dict[str, float] | None = None,
    coverage_counts: dict[str, int] | None = None,
    source_training_index_path: str = "",
    n_rows_total: int = 0,
    created_at: str = "",
) -> DensityPresets:
    """Assemble a fixed-target ``DensityPresets`` table."""
    tier_stats = build_fixed_tier_presets(
        onset_hz_targets=onset_hz_targets,
        onset_hz_norm=onset_hz_norm,
        coverage_counts=coverage_counts,
    )
    return DensityPresets(
        schema_version=1,
        onset_hz_norm=float(onset_hz_norm),
        source_training_index_path=source_training_index_path,
        n_rows_total=n_rows_total,
        created_at=created_at,
        tiers=tier_stats,
        tier_order=CUSTOMER_TIER_ORDER,
        calibration_method=CALIBRATION_METHOD_FIXED,
    )


def bucket_onsets_per_sec_equal_count(
    onsets_per_sec: list[float],
    *,
    tier_order: tuple[str, ...] = CUSTOMER_TIER_ORDER,
) -> dict[str, list[float]]:
    """Assign measured onset rates to customer tiers by equal-count rank buckets.

    Charts are sorted by onsets/sec; the lowest fifth maps to ``beginner``,
    the next to ``easy``, and so on. Simfile ``#DIFFICULTY`` labels are ignored.
    """
    tiers = tuple(tier_order)
    buckets: dict[str, list[float]] = {name: [] for name in tiers}
    if not onsets_per_sec:
        return buckets
    sorted_hz = sorted(float(value) for value in onsets_per_sec)
    n = len(sorted_hz)
    bucket_count = len(tiers)
    for index, hz in enumerate(sorted_hz):
        tier_index = min(index * bucket_count // n, bucket_count - 1)
        buckets[tiers[tier_index]].append(hz)
    return buckets


def build_tier_presets_from_buckets(
    buckets: dict[str, list[float]],
    *,
    tier_order: tuple[str, ...] = CUSTOMER_TIER_ORDER,
    onset_hz_norm: float = 15.0,
) -> dict[str, DensityTierPreset]:
    """Compute tier medians from equal-count onset-rate buckets."""
    tier_stats: dict[str, DensityTierPreset] = {}
    for tier in tier_order:
        values = buckets.get(tier, [])
        if not values:
            raise ValueError(f"empty onset-rate bucket for tier {tier!r}")
        median_hz = float(statistics.median(values))
        tier_stats[tier] = DensityTierPreset(
            onsets_per_sec_median=median_hz,
            density_scalar=config.density_scalar_from_onsets_per_sec(
                median_hz,
                onset_hz_norm=onset_hz_norm,
            ),
            n_rows=len(values),
        )
    return tier_stats


def load_density_presets(path: str | pathlib.Path | None = None) -> DensityPresets:
    """Load ``density_presets.json`` from disk.

    Args:
        path: Preset file path; defaults to ``configs/ar/density_presets.json``.
    """
    preset_path = pathlib.Path(path or DEFAULT_PRESETS_PATH)
    with preset_path.open(encoding="utf-8") as handle:
        data = json.load(handle)
    return DensityPresets.from_dict(data)


def save_density_presets(presets: DensityPresets, path: str | pathlib.Path) -> None:
    """Write presets to JSON."""
    preset_path = pathlib.Path(path)
    preset_path.parent.mkdir(parents=True, exist_ok=True)
    with preset_path.open("w", encoding="utf-8") as handle:
        json.dump(presets.as_dict(), handle, indent=2)
        handle.write("\n")


def customer_density_scalar(
    tier: str,
    *,
    model_config: config.ArModelConfig,
    presets: DensityPresets | None = None,
    duration_sec: float | None = None,
) -> float:
    """Density feature for customer-selected difficulty at decode time.

    Uses fixed or calibrated tier targets from ``density_presets.json``. For
    ``onset_density`` mode the scalar depends only on target onsets/sec
    (``duration_sec`` is accepted for API symmetry but does not change the
    scalar). For ``meter`` mode, falls back to ``meter`` presets only when
    explicitly calibrated; otherwise use ``onset_density`` tier rates.

    Args:
        tier: Customer-facing difficulty, e.g. ``medium`` or ``challenge``.
        model_config: Supplies ``density_conditioning`` and normalization.
        presets: Optional loaded presets; defaults to ``load_density_presets()``.
        duration_sec: Optional song duration (unused for ``onset_density`` tiers).

    Returns:
        ``density_scalar`` to pass to the AR decoder.
    """
    _ = duration_sec
    table = presets or load_density_presets()
    mode = config.normalize_density_conditioning_mode(
        model_config.density_conditioning,
    )
    if mode in ("", "none"):
        return 0.0
    if mode == "onset_density":
        return table.density_scalar_for_tier(tier)
    if mode == "meter":
        meter = _meter_for_tier(tier, table)
        return config.compute_density_scalar(
            n_onsets=0,
            duration_sec=1.0,
            mode="meter",
            meter=meter,
            meter_max=model_config.density_meter_max,
            onset_hz_norm=model_config.density_onset_hz_norm,
        )
    raise ValueError(f"unsupported density_conditioning mode: {mode!r}")


def customer_target_onsets(
    tier: str,
    duration_sec: float,
    *,
    presets: DensityPresets | None = None,
) -> float:
    """Expected onset count for a tier at a given song length.

    Args:
        tier: Customer-facing difficulty label.
        duration_sec: Audio duration in seconds.
        presets: Optional loaded presets.
    """
    table = presets or load_density_presets()
    hz = table.onsets_per_sec_for_tier(tier)
    return hz * float(duration_sec)


def _meter_for_tier(tier: str, presets: DensityPresets) -> int:
    """Map tier to a representative meter when ``density_conditioning`` is ``meter``."""
    fallback = {
        "beginner": 3,
        "easy": 5,
        "medium": 7,
        "hard": 10,
        "challenge": 12,
    }
    key = str(tier).strip().lower()
    return int(fallback.get(key, fallback["medium"]))
