import pathlib
import unittest

import numpy as np

from stepcovnet import constants
from stepcovnet.onset_ar import config, datasets, targets


class TargetsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.vocab = targets.DeltaBucketVocab()
        self.hop_sec = constants.HOP_COEFF
        self.patch_frames = 8

    def test_vocab_size_matches_token_ranges(self) -> None:
        self.assertEqual(self.vocab.first_abs_start, 3)
        self.assertEqual(
            self.vocab.vocab_size,
            3
            + self.vocab.n_first_abs_bins
            + self.vocab.delta_max_dense
            + self.vocab.n_log_buckets,
        )

    def test_empty_chart_is_bos_eos_only(self) -> None:
        seq = targets.encode_onset_times(
            np.zeros(0, dtype=np.float64),
            duration_sec=1.0,
            hop_sec=self.hop_sec,
            patch_frames=self.patch_frames,
            vocab=self.vocab,
        )
        self.assertEqual(seq.n_steps, 0)
        np.testing.assert_array_equal(seq.decoder_input_ids, [targets.BOS_ID])
        np.testing.assert_array_equal(seq.decoder_target_ids, [targets.EOS_ID])

    def test_encode_builds_monotonic_pointers(self) -> None:
        times = np.asarray([0.05, 0.10, 0.25], dtype=np.float64)
        seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=self.hop_sec,
            patch_frames=self.patch_frames,
            vocab=self.vocab,
        )
        self.assertTrue(np.all(np.diff(seq.patch_indices) >= 0))
        pointer_times = targets.decode_pointer_residual_to_times(
            seq.patch_indices,
            seq.residual_sec,
            patch_frames=self.patch_frames,
            hop_sec=self.hop_sec,
        )
        expected = seq.frame_indices.astype(np.float32) * self.hop_sec
        np.testing.assert_allclose(pointer_times, expected, rtol=0, atol=1e-6)

    def test_dense_delta_round_trip(self) -> None:
        for delta in (1, 5, 50, 200):
            token = self.vocab.encode_delta_frames(delta)
            self.assertEqual(self.vocab.decode_delta_frames(token), delta)

    def test_times_to_frame_indices_deduplicates_same_bin(self) -> None:
        times = np.asarray([0.050, 0.054, 0.120], dtype=np.float64)
        frames = targets.times_to_frame_indices(times, self.hop_sec)
        np.testing.assert_array_equal(frames, [5, 12])


class DatasetsTest(unittest.TestCase):
    def test_patch_mert_features_pads_last_patch(self) -> None:
        features = np.arange(30, dtype=np.float32).reshape(10, 3)
        patches, n_patches, n_frames = datasets.patch_mert_features(
            features, patch_frames=4
        )
        self.assertEqual(n_frames, 10)
        self.assertEqual(n_patches, 3)
        self.assertEqual(patches.shape, (3, 12))

    def test_config_round_trip(self) -> None:
        cfg = config.ArExperimentConfig(
            dataset=config.ArDatasetConfig(overfit_audio_path="a.ogg"),
            model=config.ArModelConfig(),
            run=config.ArRunConfig(),
        )
        restored = config.ArExperimentConfig.from_dict(cfg.as_dict())
        self.assertEqual(restored.dataset.overfit_audio_path, "a.ogg")
        self.assertEqual(restored.model.patch_frames, 8)


@unittest.skipUnless(
    pathlib.Path("data/v2/test/tide.ogg").is_file()
    and pathlib.Path("data/v2/test/tide.txt").is_file()
    and pathlib.Path("data/v2/test/tide.mert.npy").is_file(),
    "tide overfit assets not present",
)
class TideIntegrationTest(unittest.TestCase):
    def test_verify_config_loads_one_batch(self) -> None:
        experiment_config = config.ArExperimentConfig.from_json("configs/ar/tide.json")
        summary, sample = datasets.verify_config_loads_one_batch(experiment_config)
        self.assertGreater(summary["n_onsets"], 0)
        self.assertGreater(summary["n_patches"], 0)
        self.assertEqual(sample.mert_patches.shape[0], summary["n_patches"])
        self.assertEqual(sample.token_seq.n_steps, int(summary["n_onsets"]))


if __name__ == "__main__":
    unittest.main()
