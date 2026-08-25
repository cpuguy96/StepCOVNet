import pathlib
from types import SimpleNamespace
from unittest import mock

import keras
import numpy as np
import pytest

from stepcovnet import config, datasets, dense_overfit_eval, pairing
from stepcovnet.dataset_prep import training_index


def _keras_model_stub(*, predict_return_value=None):
    model = mock.create_autospec(keras.Model, instance=True)
    if predict_return_value is not None:
        model.predict.return_value = predict_return_value
    model.fit.return_value = SimpleNamespace(
        history={"val_loss": [1.0], "loss": [1.0]},
    )
    return model


def test_eval_dense_event_f1_for_pair_normalizes_mel_features(tmp_path) -> None:
    chart = tmp_path / "song.txt"
    chart.write_text(
        "TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n1000 0.1\n",
        encoding="utf-8",
    )
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"")
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    normalized = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    dataset_config = config.OnsetDatasetConfig(
        data_dir=str(tmp_path),
        val_data_dir=str(tmp_path),
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    stub_model = _keras_model_stub(
        predict_return_value=np.zeros((1, 2, 1), dtype=np.float32),
    )

    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "load_onset_features",
            return_value=raw,
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval.datasets,
            "normalize_onset_spectrogram",
            return_value=normalized,
            autospec=True,
        ) as normalize_mock,
    ):
        dense_overfit_eval.eval_dense_event_f1_for_pair(
            stub_model,
            str(audio),
            str(chart),
            dataset_config,
            model_config,
            confidence_threshold=0.5,
            min_onset_distance_ms=50.0,
            tolerance_sec=0.02,
        )
    normalize_mock.assert_called_once()
    assert np.array_equal(normalize_mock.call_args[0][0], raw)
    predict_batch = stub_model.predict.call_args[0][0]
    assert np.array_equal(predict_batch[0], normalized)


def test_eval_dense_event_f1_for_pair_skips_normalize_for_waveform(tmp_path) -> None:
    chart = tmp_path / "song.txt"
    chart.write_text(
        "TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n1000 0.1\n",
        encoding="utf-8",
    )
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"")
    waveform = np.linspace(-1.0, 1.0, num=4410, dtype=np.float32)
    stub_model = _keras_model_stub(
        predict_return_value=np.zeros((1, 10, 1), dtype=np.float32),
    )
    dataset_config = config.OnsetDatasetConfig(
        data_dir=str(tmp_path),
        val_data_dir=str(tmp_path),
        feature_source=config.FeatureSource.WAVEFORM,
    )
    model_config = config.OnsetModelConfig(waveform_frontend_filters=32)

    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "load_onset_features",
            return_value=waveform,
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval.datasets,
            "normalize_onset_spectrogram",
            autospec=True,
        ) as normalize_mock,
    ):
        dense_overfit_eval.eval_dense_event_f1_for_pair(
            stub_model,
            str(audio),
            str(chart),
            dataset_config,
            model_config,
            confidence_threshold=0.5,
            min_onset_distance_ms=50.0,
            tolerance_sec=0.02,
        )
    normalize_mock.assert_not_called()
    predict_batch = stub_model.predict.call_args[0][0]
    np.testing.assert_allclose(predict_batch[0], waveform)


def test_peak_times_and_confidence_finds_spaced_peaks() -> None:
    probs = np.zeros(100, dtype=np.float64)
    probs[10] = 0.9
    probs[11] = 0.6
    probs[30] = 0.8
    times, conf = dense_overfit_eval.peak_times_and_confidence(
        probs,
        confidence_threshold=0.5,
        min_onset_distance_ms=50.0,
        hop_sec=0.01,
    )
    assert times.shape == (2,)
    assert conf.shape == (2,)
    assert conf[0] == pytest.approx(0.9)
    assert conf[1] == pytest.approx(0.8)


def test_peak_times_and_confidence_empty_when_below_threshold() -> None:
    probs = np.full(20, 0.1, dtype=np.float64)
    times, conf = dense_overfit_eval.peak_times_and_confidence(
        probs,
        confidence_threshold=0.5,
        min_onset_distance_ms=50.0,
        hop_sec=0.01,
    )
    assert times.size == 0
    assert conf.size == 0


def test_build_gt_batch_pads_chart_times(tmp_path) -> None:
    chart = tmp_path / "tide.txt"
    chart.write_text(
        "TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n1000 0.1\n0100 0.2\n",
        encoding="utf-8",
    )
    gt_times, gt_mask = dense_overfit_eval.build_gt_batch(str(chart), n_max=8)
    assert gt_times.shape == (1, 8)
    assert gt_mask.shape == (1, 8)
    assert gt_mask[0, :2].tolist() == [1.0, 1.0]
    assert gt_mask[0, 2:].tolist() == [0.0] * 6
    assert gt_times[0, 0] == pytest.approx(0.1)
    assert gt_times[0, 1] == pytest.approx(0.2)


def test_build_gt_batch_uses_uncapped_chart_times(tmp_path) -> None:
    chart = tmp_path / "long.txt"
    chart.write_text(
        "TITLE Test\nBPM 120.0\nNOTES\nDIFFICULTY Challenge\n", encoding="utf-8"
    )
    long_times = np.linspace(0.1, 200.0, num=1100, dtype=np.float32)

    with mock.patch.object(
        dense_overfit_eval.charts,
        "load_onset_times",
        return_value=long_times,
        autospec=True,
    ):
        gt_times, gt_mask = dense_overfit_eval.build_gt_batch(str(chart))

    assert gt_times.shape == (1, 1100)
    assert gt_mask.shape == (1, 1100)
    assert int(gt_mask.sum()) == 1100
    np.testing.assert_allclose(gt_times[0], long_times)


def test_build_gt_batch_passes_chart_index(tmp_path) -> None:
    chart = tmp_path / "song.chart.json"
    chart.write_text("{}", encoding="utf-8")
    with mock.patch.object(
        dense_overfit_eval.charts,
        "load_onset_times",
        return_value=np.array([0.1], dtype=np.float64),
        autospec=True,
    ) as load_mock:
        dense_overfit_eval.build_gt_batch(str(chart), chart_index=2)
    load_mock.assert_called_once_with(str(chart), max_steps=None, chart_index=2)


def test_resolve_dense_eval_samples_uses_manifest() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="",
        val_data_dir="",
        training_index_path="data/final_data/training_index.json",
        data_root="data/final_data",
        feature_source=config.FeatureSource.MERT,
    )
    expected = [("a.ogg", "a.chart.json", 1)]
    index_path = pathlib.Path("data/final_data/training_index.json")
    with (
        mock.patch.object(
            training_index,
            "locate_training_index",
            return_value=(index_path.resolve(), index_path.parent.resolve()),
            autospec=True,
        ),
        mock.patch.object(
            training_index,
            "load_training_index",
            autospec=True,
        ),
        mock.patch.object(
            training_index,
            "resolve_output_dir",
            return_value=pathlib.Path("data/final_data"),
            autospec=True,
        ),
        mock.patch.object(
            datasets,
            "list_dense_onset_samples",
            return_value=expected,
            autospec=True,
        ) as list_mock,
    ):
        samples, root = datasets.resolve_dense_eval_samples(dataset_config)
    list_mock.assert_called_once()
    assert list_mock.call_args.kwargs == {"split": "val"}
    assert (
        pathlib.Path(list_mock.call_args.args[0])
        .as_posix()
        .endswith(
            "data/final_data/training_index.json",
        )
    )
    assert samples == expected
    assert root == "data/final_data"


def test_list_unique_audio_paths_deduplicates_manifest_rows() -> None:
    rows = [
        ("a.ogg", "a.chart.json", 0),
        ("a.ogg", "a.chart.json", 1),
        ("b.ogg", "b.chart.json", 0),
    ]
    with (
        mock.patch.object(
            pairing,
            "list_training_samples",
            return_value=rows,
            autospec=True,
        ),
        mock.patch.object(
            training_index,
            "locate_training_index",
            return_value=(pathlib.Path("index.json"), pathlib.Path("root")),
            autospec=True,
        ),
        mock.patch.object(
            training_index,
            "load_training_index",
            autospec=True,
        ),
        mock.patch.object(
            training_index,
            "resolve_output_dir",
            return_value=pathlib.Path("data/final_data"),
            autospec=True,
        ),
    ):
        audio_paths, root = pairing.list_unique_audio_paths("index.json")
    assert audio_paths == ["a.ogg", "b.ogg"]
    assert pathlib.Path(root).as_posix() == "data/final_data"


def test_eval_dense_val_event_f1_aggregates_per_song() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    stub_model = _keras_model_stub()

    def _fake_pair_metrics(
        _model,
        _audio_path,
        _chart_path,
        _dataset_config,
        _model_config,
        *,
        confidence_threshold,
        min_onset_distance_ms,
        tolerance_sec,
        data_root="",
        chart_index=0,
        corruption="none",
    ) -> dict[str, float]:
        del (
            confidence_threshold,
            min_onset_distance_ms,
            tolerance_sec,
            data_root,
            chart_index,
            corruption,
        )
        if "alpha" in _audio_path:
            return {
                "event_f1": 0.8,
                "event_tp": 8.0,
                "event_fp": 2.0,
                "event_fn": 0.0,
                "num_peaks": 10.0,
                "duration_sec": 30.0,
            }
        return {
            "event_f1": 0.4,
            "event_tp": 2.0,
            "event_fp": 3.0,
            "event_fn": 3.0,
            "num_peaks": 5.0,
            "duration_sec": 30.0,
        }

    pairs = [
        ("data/val/alpha/alpha.ogg", "data/val/alpha/alpha.txt", 0),
        ("data/val/beta/beta.ogg", "data/val/beta/beta.txt", 0),
    ]
    gt_flat = np.linspace(0.5, 25.0, num=8).astype(np.float64)
    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "resolve_dense_eval_samples",
            return_value=(pairs, "data/val"),
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "eval_dense_event_f1_for_pair",
            side_effect=_fake_pair_metrics,
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval.charts,
            "load_onset_times",
            return_value=gt_flat,
            autospec=True,
        ),
    ):
        report = dense_overfit_eval.eval_dense_val_event_f1(
            stub_model,
            dataset_config,
            model_config,
            confidence_threshold=0.5,
        )

    assert report["num_songs"] == 2
    assert report["mean_event_f1"] == pytest.approx(0.6)
    assert report["micro_event_f1"] == pytest.approx(20.0 / 28.0)
    assert report["micro_tp"] == pytest.approx(10.0)
    assert report["micro_fp"] == pytest.approx(5.0)
    assert report["micro_fn"] == pytest.approx(3.0)
    assert report["per_song"]["alpha"]["event_f1"] == pytest.approx(0.8)
    assert report["per_song"]["beta"]["event_f1"] == pytest.approx(0.4)
    assert report["corruption"] == "none"
    null_floors = report["null_floors"]
    assert set(null_floors["by_kind"]) == set(
        dense_overfit_eval.onset_null_baseline.DEFAULT_KINDS
    )
    assert 0.0 <= null_floors["event_f1_floor"] <= 1.0
    assert null_floors["skill_event_f1"] == pytest.approx(
        dense_overfit_eval.onset_null_baseline.skill_over_null(
            report["micro_event_f1"],
            null_floors["event_f1_floor"],
        )
    )


def test_corrupt_features_kinds() -> None:
    features = np.arange(12, dtype=np.float32).reshape(4, 3)
    assert dense_overfit_eval.corrupt_features(features, "none") is features
    zeros = dense_overfit_eval.corrupt_features(features, "zeros")
    assert not zeros.any()
    assert zeros.shape == features.shape
    shuffled = dense_overfit_eval.corrupt_features(features, "shuffle")
    assert shuffled.shape == features.shape
    assert sorted(shuffled[:, 0].tolist()) == features[:, 0].tolist()
    shuffled_again = dense_overfit_eval.corrupt_features(features, "shuffle")
    np.testing.assert_array_equal(shuffled, shuffled_again)
    with pytest.raises(ValueError, match="unsupported corruption kind"):
        dense_overfit_eval.corrupt_features(features, "reverse")


def test_predict_dense_probs_for_pair_applies_corruption(tmp_path) -> None:
    audio = tmp_path / "song.ogg"
    audio.write_bytes(b"")
    raw = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    dataset_config = config.OnsetDatasetConfig(
        data_dir=str(tmp_path),
        val_data_dir=str(tmp_path),
        feature_source=config.FeatureSource.MEL,
    )
    stub_model = _keras_model_stub(
        predict_return_value=np.zeros((1, 2, 1), dtype=np.float32),
    )
    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "load_onset_features",
            return_value=raw,
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval.datasets,
            "normalize_onset_spectrogram",
            side_effect=lambda arr: arr,
            autospec=True,
        ),
    ):
        dense_overfit_eval.predict_dense_probs_for_pair(
            stub_model,
            str(audio),
            dataset_config,
            corruption="zeros",
        )
    predict_batch = stub_model.predict.call_args[0][0]
    assert not np.asarray(predict_batch).any()


def test_oracle_density_for_pair_disabled_returns_none() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data",
        val_data_dir="data",
        density_conditioning=config.DENSITY_CONDITIONING_NONE,
    )
    assert (
        dense_overfit_eval.oracle_density_for_pair(
            "chart.json",
            duration_sec=100.0,
            dataset_config=dataset_config,
        )
        is None
    )


def test_oracle_density_for_pair_scales_onset_rate() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data",
        val_data_dir="data",
        density_conditioning=config.DENSITY_CONDITIONING_ONSET,
        feature_source=config.FeatureSource.MERT,
    )
    with mock.patch.object(
        dense_overfit_eval.charts,
        "load_onset_times",
        return_value=np.linspace(0.0, 9.0, 150, dtype=np.float32),
        autospec=True,
    ):
        value = dense_overfit_eval.oracle_density_for_pair(
            "chart.json",
            duration_sec=10.0,
            dataset_config=dataset_config,
        )
    assert value == pytest.approx(1.0)


def test_gt_onset_times_from_frame_target_recovers_binary_onsets() -> None:
    target = np.zeros((120, 1), dtype=np.float32)
    target[10, 0] = 1.0
    target[40, 0] = 1.0
    times = dense_overfit_eval.gt_onset_times_from_frame_target(target)
    assert times.tolist() == pytest.approx(
        [
            10 * dense_overfit_eval.datasets.HOP_COEFF,
            40 * dense_overfit_eval.datasets.HOP_COEFF,
        ]
    )


def test_sweep_thresholds_dense_val_event_f1_selects_best_threshold() -> None:
    hop = dense_overfit_eval.datasets.HOP_COEFF
    onset_frames = [10, 100, 200, 300, 400]
    probs = np.zeros(1000, dtype=np.float64)
    for frame in onset_frames:
        probs[frame] = 0.9
    # Weak spurious peaks that only survive the low threshold.
    for frame in (50, 150, 250):
        probs[frame] = 0.3
    gt_times = np.array(
        [[frame * hop for frame in onset_frames]],
        dtype=np.float32,
    )
    gt_mask = np.ones((1, len(onset_frames)), dtype=np.float32)
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    stub_model = _keras_model_stub()
    pairs = [("data/val/song/song.ogg", "data/val/song/song.txt", 0)]

    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "resolve_dense_eval_samples",
            return_value=(pairs, "data/val"),
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "predict_dense_probs_for_pair",
            return_value=probs,
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "build_gt_batch",
            return_value=(gt_times, gt_mask),
            autospec=True,
        ),
    ):
        report = dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            stub_model,
            dataset_config,
            model_config,
            thresholds=(0.2, 0.35),
        )

    assert report["num_songs"] == 1
    assert report["eval_split"] == "data/val"
    assert report["best_threshold"] == pytest.approx(0.35)
    assert report["best_micro_event_f1"] == pytest.approx(1.0)
    assert report["best_micro_timing_match"] == pytest.approx(1.0)
    assert report["selection_metric"] == "skill_event_f1"
    assert report["raw_f1_best_threshold"] == pytest.approx(0.35)
    by_threshold = {row["confidence_threshold"]: row for row in report["per_threshold"]}
    # The low threshold adds 3 spurious peaks: Hungarian F1 degrades gently
    # (10/13) while skill_event_f1 still prefers the cleaner operating point.
    assert by_threshold[0.2]["micro_event_f1"] == pytest.approx(10.0 / 13.0)
    assert by_threshold[0.35]["micro_event_f1"] == pytest.approx(1.0)
    assert (
        by_threshold[0.2]["micro_timing_match"]
        < by_threshold[0.35]["micro_timing_match"]
    )
    for row in by_threshold.values():
        assert "null_event_f1_floor" in row
        assert "skill_event_f1" in row
        assert "skill_timing_match" in row


def test_threshold_sweep_selection_prefers_skill_over_raw_f1() -> None:
    """A lower threshold that only inflates onset count must not win selection."""
    summaries = [
        {
            "confidence_threshold": 0.05,
            "micro_event_f1": 0.40,
            "micro_timing_match": 0.10,
            "skill_event_f1": 0.05,
            "skill_timing_match": 0.09,
            "mean_event_f1": 0.40,
        },
        {
            "confidence_threshold": 0.30,
            "micro_event_f1": 0.35,
            "micro_timing_match": 0.30,
            "skill_event_f1": 0.20,
            "skill_timing_match": 0.29,
            "mean_event_f1": 0.35,
        },
    ]
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    pairs = [("data/val/song/song.ogg", "data/val/song/song.txt", 0)]
    with (
        mock.patch.object(
            dense_overfit_eval.datasets,
            "resolve_dense_eval_samples",
            return_value=(pairs, "data/val"),
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "predict_dense_probs_for_pair",
            return_value=np.zeros(10, dtype=np.float64),
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "build_gt_batch",
            return_value=(
                np.zeros((1, 1), dtype=np.float32),
                np.ones((1, 1), dtype=np.float32),
            ),
            autospec=True,
        ),
        mock.patch.object(
            dense_overfit_eval,
            "_threshold_summary_from_cache",
            side_effect=summaries,
            autospec=True,
        ),
    ):
        report = dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            _keras_model_stub(),
            dataset_config,
            model_config,
            thresholds=(0.05, 0.30),
        )

    assert report["best_threshold"] == pytest.approx(0.30)
    assert report["raw_f1_best_threshold"] == pytest.approx(0.05)


def test_sweep_thresholds_dense_val_event_f1_rejects_empty_thresholds() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    with pytest.raises(ValueError, match="thresholds must be non-empty"):
        dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            _keras_model_stub(),
            dataset_config,
            model_config,
            thresholds=(),
        )


def test_dense_event_onset_counts_from_arrays_perfect_match() -> None:
    y_true = np.zeros((1, 50, 1), dtype=np.float32)
    y_pred = np.zeros((1, 50, 1), dtype=np.float32)
    y_true[0, 10, 0] = 1.0
    y_pred[0, 10, 0] = 0.9
    tp, fp, fn = dense_overfit_eval.dense_event_onset_counts_from_arrays(
        y_true,
        y_pred,
        tolerance_sec=0.02,
        confidence_threshold=0.5,
        min_onset_distance_ms=50.0,
    )
    assert tp == pytest.approx(1.0)
    assert fp == pytest.approx(0.0)
    assert fn == pytest.approx(0.0)
    assert dense_overfit_eval.micro_f1_from_counts(tp, fp, fn) == pytest.approx(1.0)


def test_dense_timing_match_from_arrays_perfect_peak_pick() -> None:
    y_true = np.zeros((1, 50, 1), dtype=np.float32)
    y_pred = np.zeros((1, 50, 1), dtype=np.float32)
    y_true[0, 10, 0] = 1.0
    y_true[0, 30, 0] = 1.0
    y_pred[0, 10, 0] = 0.9
    y_pred[0, 30, 0] = 0.85
    n_matched, n_ref, n_pred = dense_overfit_eval.dense_timing_match_from_arrays(
        y_true,
        y_pred,
        tolerance_sec=0.02,
        confidence_threshold=0.5,
        min_onset_distance_ms=50.0,
    )
    assert n_matched == pytest.approx(2.0)
    assert n_ref == pytest.approx(2.0)
    assert n_pred == pytest.approx(2.0)
