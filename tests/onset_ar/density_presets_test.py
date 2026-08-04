"""Tests for customer difficulty density presets."""

from __future__ import annotations

import json
import tempfile
import unittest

from stepcovnet.onset_ar import config, density_presets


def _sample_presets() -> density_presets.DensityPresets:
    return density_presets.DensityPresets(
        schema_version=1,
        onset_hz_norm=15.0,
        source_training_index_path="data/final_data/training_index.json",
        n_rows_total=100,
        created_at="2026-08-03T00:00:00Z",
        tiers={
            "easy": density_presets.DensityTierPreset(
                onsets_per_sec_median=3.0,
                density_scalar=0.2,
                n_rows=20,
            ),
            "medium": density_presets.DensityTierPreset(
                onsets_per_sec_median=6.0,
                density_scalar=0.4,
                n_rows=40,
            ),
            "hard": density_presets.DensityTierPreset(
                onsets_per_sec_median=9.0,
                density_scalar=0.6,
                n_rows=40,
            ),
        },
    )


class DensityPresetsTest(unittest.TestCase):
    def test_round_trip_json(self) -> None:
        presets = _sample_presets()
        with tempfile.TemporaryDirectory() as tmp:
            path = f"{tmp}/density_presets.json"
            density_presets.save_density_presets(presets, path)
            loaded = density_presets.load_density_presets(path)
        self.assertEqual(loaded.onset_hz_norm, 15.0)
        self.assertAlmostEqual(
            loaded.density_scalar_for_tier("hard"),
            0.6,
        )
        self.assertEqual(loaded.tiers["medium"].n_rows, 40)

    def test_unknown_tier_falls_back_to_medium(self) -> None:
        presets = _sample_presets()
        self.assertAlmostEqual(
            presets.onsets_per_sec_for_tier("Expert"),
            6.0,
        )

    def test_customer_density_scalar_onset_density(self) -> None:
        model = config.ArModelConfig(density_conditioning="onset_density")
        scalar = density_presets.customer_density_scalar(
            "hard",
            model_config=model,
            presets=_sample_presets(),
        )
        self.assertAlmostEqual(scalar, 0.6)

    def test_customer_density_scalar_meter_uses_tier_meter_map(self) -> None:
        model = config.ArModelConfig(
            density_conditioning="meter",
            density_meter_max=12,
        )
        scalar = density_presets.customer_density_scalar(
            "easy",
            model_config=model,
            presets=_sample_presets(),
        )
        expected = config.compute_density_scalar(
            n_onsets=0,
            duration_sec=1.0,
            mode="meter",
            meter=5,
            meter_max=12,
        )
        self.assertAlmostEqual(scalar, expected)

    def test_customer_target_onsets(self) -> None:
        count = density_presets.customer_target_onsets(
            "medium",
            duration_sec=120.0,
            presets=_sample_presets(),
        )
        self.assertAlmostEqual(count, 720.0)

    def test_from_dict_preserves_source_path(self) -> None:
        data = {
            "schema_version": 1,
            "onset_hz_norm": 15.0,
            "source": {
                "calibration_method": density_presets.CALIBRATION_METHOD_FIXED,
                "training_index_path": "data/final_data/training_index.json",
                "n_rows": 10,
                "created_at": "2026-08-03T00:00:00Z",
            },
            "tiers": {
                "medium": {
                    "onsets_per_sec_median": 6.0,
                    "density_scalar": 0.4,
                    "n_rows": 10,
                },
            },
        }
        presets = density_presets.DensityPresets.from_dict(data)
        self.assertEqual(
            presets.source_training_index_path,
            "data/final_data/training_index.json",
        )
        self.assertEqual(
            presets.calibration_method,
            density_presets.CALIBRATION_METHOD_FIXED,
        )
        round_trip = json.loads(json.dumps(presets.as_dict()))
        self.assertEqual(
            round_trip["source"]["training_index_path"],
            data["source"]["training_index_path"],
        )

    def test_fixed_tier_presets_match_design_targets(self) -> None:
        presets = density_presets.build_fixed_density_presets()
        for tier in density_presets.CUSTOMER_TIER_ORDER:
            target = density_presets.DEFAULT_FIXED_ONSET_HZ_TARGETS[tier]
            self.assertAlmostEqual(
                presets.onsets_per_sec_for_tier(tier),
                target,
            )
            self.assertAlmostEqual(
                presets.density_scalar_for_tier(tier),
                target / 15.0,
            )

    def test_coverage_bands_use_midpoint_thresholds(self) -> None:
        self.assertEqual(
            density_presets.tier_for_onsets_per_sec(2.5),
            "beginner",
        )
        self.assertEqual(
            density_presets.tier_for_onsets_per_sec(4.0),
            "easy",
        )
        self.assertEqual(
            density_presets.tier_for_onsets_per_sec(10.0),
            "challenge",
        )
        counts = density_presets.coverage_counts_for_onsets_per_sec(
            [1.0, 4.0, 6.0, 8.0, 12.0],
        )
        self.assertEqual(counts["beginner"], 1)
        self.assertEqual(counts["easy"], 1)
        self.assertEqual(counts["medium"], 1)
        self.assertEqual(counts["hard"], 1)
        self.assertEqual(counts["challenge"], 1)

    def test_equal_count_buckets_are_monotonic(self) -> None:
        hz = [float(x) for x in range(1, 11)]
        buckets = density_presets.bucket_onsets_per_sec_equal_count(hz)
        self.assertEqual(
            [len(buckets[tier]) for tier in density_presets.CUSTOMER_TIER_ORDER],
            [2, 2, 2, 2, 2],
        )
        presets = density_presets.build_tier_presets_from_buckets(buckets)
        medians = [
            presets[tier].onsets_per_sec_median
            for tier in density_presets.CUSTOMER_TIER_ORDER
        ]
        self.assertEqual(medians, sorted(medians))


if __name__ == "__main__":
    unittest.main()
