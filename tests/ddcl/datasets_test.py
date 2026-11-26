"""Tests for DDCL dataset batching."""

from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

import numpy as np

from stepcovnet.dataset_prep import constants as prep_constants
from stepcovnet.dataset_prep import models as prep_models
from stepcovnet.dataset_prep import training_index
from stepcovnet.ddcl import config, constants, datasets, features


def _chart() -> datasets.DdclChart:
    n_beats = 4
    slots = np.zeros((n_beats, constants.N_SLOTS), dtype=np.float32)
    slots[1, 3] = 1.0
    return datasets.DdclChart(
        song_key="b/s",
        difficulty="hard",
        meter=8,
        beat_audio=np.zeros(
            (n_beats, 8, constants.N_MELS, constants.N_CHANNELS),
            dtype=np.float32,
        ),
        stream=np.zeros((n_beats, constants.STREAM_DIM), dtype=np.float32),
        slots=slots,
        memlen=1,
    )


class DdclDatasetsTest(unittest.TestCase):
    def test_sample_train_batch_shapes(self):
        rng = np.random.default_rng(0)
        inputs, labels = datasets.sample_train_batch(
            [_chart()],
            batch_size=3,
            rng=rng,
        )
        self.assertEqual(inputs["audio_fwd"].shape[0], 3)
        self.assertEqual(labels.shape, (3, constants.N_SLOTS))
        predict_inputs = datasets.chart_model_inputs(_chart())
        self.assertEqual(predict_inputs["audio_fwd"].shape[0], 4)
        batches = list(datasets.iter_window_batches(_chart(), batch_size=3))
        self.assertEqual([int(batch.shape[0]) for batch in batches], [3, 1])

    def test_windows_for_indices_rejects_empty(self):
        with self.assertRaises(ValueError):
            datasets.windows_for_indices(_chart(), np.array([], dtype=np.int64))

    def test_list_split_entries_excludes_song_ids(self):
        def entry(
            song_id: str, split: training_index.SplitName
        ) -> training_index.TrainingIndexEntry:
            return training_index.TrainingIndexEntry(
                split=split,
                normalized_bundle="b",
                normalized_id=song_id,
                chart_index=0,
                output_relpath=f"b/{song_id}",
                difficulty="hard",
                meter=8,
                num_steps=4,
                audio_relpath=f"b/{song_id}/{song_id}.ogg",
                chart_relpath=f"b/{song_id}/{song_id}.chart.json",
            )

        index = training_index.TrainingIndex(
            schema_version=prep_constants.SCHEMA_VERSION,
            output_dir="data/final_data",
            split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
            split_seed=42,
            val_fraction=0.5,
            created_at="2026-10-10T00:00:00Z",
            counts=training_index.TrainingIndexCounts(
                songs={"train": 1, "val": 1},
                rows={"train": 1, "val": 1},
            ),
            entries=[entry("keep", "train"), entry("news_39", "val")],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = pathlib.Path(tmpdir) / "index.json"
            training_index.save_training_index(index, path)
            dataset_config = config.DdclDatasetConfig(
                training_index_path=str(path),
                exclude_normalized_ids=["news_39"],
            )
            kept = datasets.list_split_entries(dataset_config, "train")
            self.assertEqual([row.normalized_id for row in kept], ["keep"])
            with self.assertRaises(ValueError):
                datasets.list_split_entries(dataset_config, "val")

    def test_sample_rejects_empty(self):
        rng = np.random.default_rng(0)
        with self.assertRaises(ValueError):
            datasets.sample_train_batch([], batch_size=1, rng=rng)

    def test_mert_beat_resampling_and_zero_padding(self):
        mert = np.arange(200 * 4, dtype=np.float32).reshape(200, 4) - 500.0
        beat_times = np.array([0.0, 0.5, 1.0, 1.5], dtype=np.float64)
        beat_mert = features.beats_to_mert_tensor(mert, beat_times, n_frames=8)
        self.assertEqual(beat_mert.shape, (3, 8, 4))
        chart = datasets.DdclChart(
            song_key="b/s",
            difficulty="hard",
            meter=8,
            beat_audio=beat_mert,
            stream=np.zeros((3, constants.STREAM_DIM), dtype=np.float32),
            slots=np.zeros((3, constants.N_SLOTS), dtype=np.float32),
            memlen=2,
        )
        windows = datasets.windows_for_indices(chart, np.array([0], dtype=np.int64))
        self.assertEqual(windows["audio_fwd"].shape, (1, 3, 8, 4))
        # Past padding beats for MERT use 0.0 instead of negative outlier min().
        np.testing.assert_array_equal(windows["audio_fwd"][0, 0], np.zeros((8, 4)))
        np.testing.assert_array_equal(windows["audio_fwd"][0, 2], beat_mert[0])

    def test_load_split_charts_shares_mert_tensor_across_difficulties(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = pathlib.Path(tmpdir)
            song_dir = root / "bundle" / "song"
            song_dir.mkdir(parents=True)
            audio_path = song_dir / "song.ogg"
            audio_path.write_bytes(b"")
            mert = np.arange(400 * 4, dtype=np.float32).reshape(400, 4)
            np.save(song_dir / "song.mert.npy", mert)
            pack = prep_models.ParsedSongPack(
                schema_version=prep_constants.SCHEMA_VERSION,
                normalized_bundle="bundle",
                normalized_id="song",
                source_pack_relpath="bundle/song",
                source_simfile="song.sm",
                metadata=prep_models.SimfileMetadata(
                    title="Song",
                    artist="Artist",
                    subtitle="",
                    music_filename="song.ogg",
                    offset_sec=0.0,
                    initial_bpm=120.0,
                    bpm_segments=[prep_models.BpmSegment(start_beat=0.0, bpm=120.0)],
                    selectable=True,
                ),
                charts=[
                    prep_models.ParsedChart(
                        summary=prep_models.ChartSummary(
                            stepstype="dance-single",
                            difficulty="challenge",
                            difficulty_kind="standard",
                            meter=10,
                            chart_name="",
                            credit="",
                            num_steps=4,
                        ),
                        times_sec=[0.0, 0.5, 1.0, 1.5],
                        arrow_rows=["1000", "0100", "0010", "0001"],
                        column_codes=[64, 16, 4, 1],
                    ),
                    prep_models.ParsedChart(
                        summary=prep_models.ChartSummary(
                            stepstype="dance-single",
                            difficulty="easy",
                            difficulty_kind="standard",
                            meter=4,
                            chart_name="",
                            credit="",
                            num_steps=2,
                        ),
                        times_sec=[0.0, 0.5],
                        arrow_rows=["1000", "0100"],
                        column_codes=[64, 16],
                    ),
                ],
                default_chart_index=0,
                available_charts=[],
                audio_filename="song.ogg",
                audio_source="music_tag",
                audio_resolved_relpath="song.ogg",
                warnings=[],
            )
            (song_dir / "song.chart.json").write_text(
                json.dumps(pack.as_dict()), encoding="utf-8"
            )
            entries = [
                training_index.TrainingIndexEntry(
                    split="train",
                    normalized_bundle="bundle",
                    normalized_id="song",
                    chart_index=idx,
                    output_relpath="bundle/song",
                    difficulty=diff,
                    meter=meter,
                    num_steps=n_steps,
                    audio_relpath="bundle/song/song.ogg",
                    chart_relpath="bundle/song/song.chart.json",
                )
                for idx, diff, meter, n_steps in (
                    (0, "challenge", 10, 4),
                    (1, "easy", 4, 2),
                )
            ]
            index = training_index.TrainingIndex(
                schema_version=prep_constants.SCHEMA_VERSION,
                output_dir=str(root),
                split_policy=training_index.SPLIT_POLICY_STRATIFIED_SONG_V1,
                split_seed=42,
                val_fraction=0.0,
                created_at="2026-11-23T00:00:00Z",
                counts=training_index.TrainingIndexCounts(
                    songs={"train": 1, "val": 0},
                    rows={"train": 2, "val": 0},
                ),
                entries=entries,
            )
            idx_path = root / "training_index.json"
            training_index.save_training_index(index, idx_path)
            cfg = config.DdclDatasetConfig(
                training_index_path=str(idx_path),
                data_root=str(root),
                audio_feature_type="mert",
                n_frames_per_beat=8,
                memlen=0,
            )
            loaded = datasets.load_split_charts(cfg, "train")
            self.assertEqual(len(loaded), 2)
            self.assertTrue(np.shares_memory(loaded[0].beat_audio, loaded[1].beat_audio))
            np.testing.assert_array_equal(
                loaded[1].beat_audio,
                loaded[0].beat_audio[: loaded[1].n_beats],
            )
            # With mert_spectral_flux=True, 6 flux/energy channels are appended.
            rng = np.random.default_rng(5)
            logmel = rng.standard_normal((398, 80, 3)).astype(np.float32)
            logmel[50, :20, 0] += 20.0  # strong low-band onset spike
            np.save(song_dir / "song.ddc_mel.npy", logmel)
            cfg_flux = config.DdclDatasetConfig(
                training_index_path=str(idx_path),
                data_root=str(root),
                audio_feature_type="mert",
                n_frames_per_beat=8,
                memlen=0,
                mert_spectral_flux=True,
            )
            loaded_flux = datasets.load_split_charts(cfg_flux, "train")
            self.assertEqual(loaded_flux[0].beat_audio.shape[-1], 4 + 6)
            flux = features.compute_spectral_flux_channels(logmel, target_frames=400)
            self.assertEqual(flux.shape, (400, 6))
            self.assertEqual(int(np.argmax(flux[:, 0])), 50)
            np.testing.assert_allclose(np.mean(flux, axis=0), 0.0, atol=1e-5)
            np.testing.assert_allclose(np.std(flux, axis=0), 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
