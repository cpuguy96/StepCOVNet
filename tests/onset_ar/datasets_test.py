"""Tests for AR onset dataset config and in-memory caching."""

from __future__ import annotations

import logging
import unittest
import unittest.mock

import numpy as np
import tensorflow as tf

from stepcovnet.onset_ar import config, datasets, targets


def _tiny_experiment_config(
    *,
    cache_in_memory: bool = True,
    cache_max_samples: int = 64,
    dynamic_padding: bool = False,
) -> config.ArExperimentConfig:
    return config.ArExperimentConfig(
        dataset=config.ArDatasetConfig(
            max_audio_seconds=1.0,
            hop_sec=0.01,
            max_steps_per_chart=16,
            cache_in_memory=cache_in_memory,
            cache_max_samples=cache_max_samples,
            dynamic_padding=dynamic_padding,
            batch_size=1,
        ),
        model=config.ArModelConfig(
            patch_frames=8,
            d_model=32,
            n_enc_layers=1,
            n_dec_layers=1,
            max_decode_steps=16,
            dropout_rate=0.0,
        ),
        run=config.ArRunConfig(overfit_one_song=False, seed=0),
    )


def _synthetic_load_callback_arrays(
    experiment_config: config.ArExperimentConfig,
) -> tuple[np.ndarray, ...]:
    max_patches = experiment_config.max_encoder_patches()
    patch_dim = experiment_config.patch_input_dim()
    max_dec = experiment_config.max_decoder_len()
    max_gt = int(experiment_config.model.max_decode_steps)
    return (
        np.zeros((max_patches, patch_dim), dtype=np.float32),
        np.ones((max_patches,), dtype=np.float32),
        np.zeros((max_dec,), dtype=np.int32),
        np.zeros((max_dec,), dtype=np.int32),
        np.ones((max_dec,), dtype=np.float32),
        np.zeros((max_dec,), dtype=np.int32),
        np.zeros((max_dec,), dtype=np.float32),
        np.zeros((max_dec,), dtype=np.float32),
        np.ones((max_dec,), dtype=np.float32),
        np.zeros((max_gt,), dtype=np.float32),
        np.ones((max_gt,), dtype=np.float32),
        np.asarray(1.0, dtype=np.float32),
    )


class ArDatasetConfigTest(unittest.TestCase):
    def test_from_dict_maps_legacy_cache_overfit_batch(self) -> None:
        cfg = config.ArDatasetConfig.from_dict({"cache_overfit_batch": False})
        self.assertFalse(cfg.cache_in_memory)

    def test_from_dict_prefers_explicit_cache_in_memory(self) -> None:
        cfg = config.ArDatasetConfig.from_dict(
            {
                "cache_in_memory": True,
                "cache_overfit_batch": False,
            },
        )
        self.assertTrue(cfg.cache_in_memory)

    def test_from_dict_defaults_cache_in_memory_true(self) -> None:
        cfg = config.ArDatasetConfig.from_dict({})
        self.assertTrue(cfg.cache_in_memory)
        self.assertEqual(cfg.cache_max_samples, 64)

    def test_should_cache_ar_samples_follows_flag(self) -> None:
        enabled = _tiny_experiment_config(cache_in_memory=True)
        disabled = _tiny_experiment_config(cache_in_memory=False)
        self.assertTrue(datasets._should_cache_ar_samples(enabled))
        self.assertFalse(datasets._should_cache_ar_samples(disabled))


class ArDatasetCacheTest(unittest.TestCase):
    def test_create_ar_tf_dataset_caches_mapped_samples(self) -> None:
        experiment_config = _tiny_experiment_config(cache_in_memory=True)
        pairs = [("a.ogg", "a.txt"), ("b.ogg", "b.txt")]
        call_count = 0

        def fake_load(
            audio_path_t: tf.Tensor,
            chart_path_t: tf.Tensor,
            chart_index_t: tf.Tensor,
            mapped_config: config.ArExperimentConfig,
        ) -> tuple[np.ndarray, ...]:
            nonlocal call_count
            del audio_path_t, chart_path_t, chart_index_t, mapped_config
            call_count += 1
            return _synthetic_load_callback_arrays(experiment_config)

        with (
            unittest.mock.patch.object(
                datasets,
                "_filter_valid_ar_samples",
                side_effect=lambda samples, _cfg: samples,
            ),
            unittest.mock.patch.object(
                datasets,
                "_load_ar_sample_py_callback",
                side_effect=fake_load,
            ),
            self.assertLogs(level=logging.INFO) as logs,
        ):
            ds = datasets.create_ar_tf_dataset_from_pairs(
                experiment_config,
                pairs,
                shuffle=False,
            )
            warm_calls = call_count
            list(ds.as_numpy_iterator())
            after_first = call_count
            list(ds.as_numpy_iterator())
            after_second = call_count

        self.assertEqual(warm_calls, 2)
        self.assertEqual(after_first, 2)
        self.assertEqual(after_second, 2)
        self.assertTrue(
            any("Caching 2 AR samples in memory" in message for message in logs.output),
        )

    def test_create_ar_tf_dataset_skips_cache_above_max_samples(self) -> None:
        experiment_config = _tiny_experiment_config(
            cache_in_memory=True,
            cache_max_samples=1,
        )
        pairs = [("a.ogg", "a.txt"), ("b.ogg", "b.txt")]
        call_count = 0

        def fake_load(
            audio_path_t: tf.Tensor,
            chart_path_t: tf.Tensor,
            chart_index_t: tf.Tensor,
            mapped_config: config.ArExperimentConfig,
        ) -> tuple[np.ndarray, ...]:
            nonlocal call_count
            del audio_path_t, chart_path_t, chart_index_t, mapped_config
            call_count += 1
            return _synthetic_load_callback_arrays(experiment_config)

        with (
            unittest.mock.patch.object(
                datasets,
                "_filter_valid_ar_samples",
                side_effect=lambda samples, _cfg: samples,
            ),
            unittest.mock.patch.object(
                datasets,
                "_load_ar_sample_py_callback",
                side_effect=fake_load,
            ),
            self.assertLogs(level=logging.WARNING) as logs,
        ):
            ds = datasets.create_ar_tf_dataset_from_pairs(
                experiment_config,
                pairs,
                shuffle=False,
            )
            self.assertEqual(call_count, 0)
            list(ds.as_numpy_iterator())
            after_first = call_count
            list(ds.as_numpy_iterator())
            after_second = call_count

        self.assertEqual(after_first, 2)
        self.assertEqual(after_second, 4)
        self.assertTrue(
            any("Skipping AR in-memory cache" in message for message in logs.output),
        )

    def test_create_ar_training_datasets_uses_overfit_helper_when_cached(
        self,
    ) -> None:
        experiment_config = _tiny_experiment_config(cache_in_memory=True)
        experiment_config.dataset.overfit_audio_path = "a.ogg"
        experiment_config.dataset.overfit_chart_path = "a.txt"
        experiment_config.run.overfit_one_song = True
        sentinel = tf.data.Dataset.from_tensors({"duration": tf.constant([1.0])})

        with unittest.mock.patch.object(
            datasets,
            "create_overfit_tf_dataset",
            return_value=sentinel,
        ) as mock_overfit:
            train_ds, val_ds, n_train, n_val = datasets.create_ar_training_datasets(
                experiment_config,
            )

        mock_overfit.assert_called_once_with(experiment_config)
        self.assertIs(train_ds, sentinel)
        self.assertIs(val_ds, sentinel)
        self.assertEqual((n_train, n_val), (1, 1))

    def test_create_ar_training_datasets_skips_overfit_helper_when_uncached(
        self,
    ) -> None:
        experiment_config = _tiny_experiment_config(cache_in_memory=False)
        experiment_config.dataset.overfit_audio_path = "a.ogg"
        experiment_config.dataset.overfit_chart_path = "a.txt"
        experiment_config.run.overfit_one_song = True
        sentinel = tf.data.Dataset.from_tensors({"duration": tf.constant([1.0])})

        with (
            unittest.mock.patch.object(
                datasets,
                "create_overfit_tf_dataset",
            ) as mock_overfit,
            unittest.mock.patch.object(
                datasets,
                "create_ar_tf_dataset_from_pairs",
                return_value=sentinel,
            ) as mock_pairs,
        ):
            train_ds, val_ds, n_train, n_val = datasets.create_ar_training_datasets(
                experiment_config,
            )

        mock_overfit.assert_not_called()
        mock_pairs.assert_called_once()
        self.assertIs(train_ds, sentinel)
        self.assertIs(val_ds, sentinel)
        self.assertEqual((n_train, n_val), (1, 1))

    def test_create_overfit_tf_dataset_respects_dynamic_padding(self) -> None:
        experiment_config = _tiny_experiment_config(
            cache_in_memory=True,
            dynamic_padding=True,
        )
        times = np.asarray([0.05, 0.12], dtype=np.float64)
        token_seq = targets.encode_onset_times(
            times,
            duration_sec=1.0,
            hop_sec=experiment_config.dataset.hop_sec,
            patch_frames=experiment_config.model.patch_frames,
            vocab=experiment_config.build_vocab(),
            max_steps=16,
        )
        sample = datasets.ArSample(
            mert_patches=np.random.randn(
                3,
                experiment_config.patch_input_dim(),
            ).astype(np.float32),
            n_patches=3,
            n_frames=10,
            duration_sec=1.0,
            token_seq=token_seq,
            gt_times_sec=times.astype(np.float32),
            audio_path="a.ogg",
            chart_path="a.txt",
        )

        with unittest.mock.patch.object(
            datasets,
            "load_overfit_sample",
            return_value=sample,
        ):
            batch = next(iter(datasets.create_overfit_tf_dataset(experiment_config)))

        self.assertEqual(
            tuple(batch["mert_patches"].shape),
            (1, 3, sample.mert_patches.shape[1]),
        )
        self.assertEqual(
            int(batch["decoder_mask"].shape[1]),
            token_seq.n_steps + 1,
        )


if __name__ == "__main__":
    unittest.main()
