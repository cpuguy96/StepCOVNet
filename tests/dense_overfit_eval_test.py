import pathlib
from unittest import mock

import numpy as np
import pytest

from stepcovnet import config, datasets, dense_overfit_eval, pairing
from stepcovnet.dataset_prep import training_index


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
    stub_model = mock.Mock()
    stub_model.predict.return_value = np.zeros((1, 2, 1), dtype=np.float32)

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
    stub_model = mock.Mock()
    stub_model.predict.return_value = np.zeros((1, 10, 1), dtype=np.float32)
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
    stub_model = mock.Mock()

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
    ) -> dict[str, float]:
        del (
            confidence_threshold,
            min_onset_distance_ms,
            tolerance_sec,
            data_root,
            chart_index,
        )
        if "alpha" in _audio_path:
            return {
                "event_f1": 0.8,
                "event_tp": 8.0,
                "event_fp": 2.0,
                "event_fn": 0.0,
                "num_peaks": 10.0,
            }
        return {
            "event_f1": 0.4,
            "event_tp": 2.0,
            "event_fp": 3.0,
            "event_fn": 3.0,
            "num_peaks": 5.0,
        }

    pairs = [
        ("data/val/alpha/alpha.ogg", "data/val/alpha/alpha.txt", 0),
        ("data/val/beta/beta.ogg", "data/val/beta/beta.txt", 0),
    ]
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
    probs = np.zeros(100, dtype=np.float64)
    probs[10] = 0.9
    probs[50] = 0.3
    gt_times = np.array(
        [[10 * dense_overfit_eval.datasets.HOP_COEFF]], dtype=np.float32
    )
    gt_mask = np.array([[1.0]], dtype=np.float32)
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    stub_model = mock.Mock()
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
    by_threshold = {row["confidence_threshold"]: row for row in report["per_threshold"]}
    assert by_threshold[0.2]["micro_event_f1"] == pytest.approx(2.0 / 3.0)
    assert by_threshold[0.35]["micro_event_f1"] == pytest.approx(1.0)


def test_sweep_thresholds_dense_val_event_f1_rejects_empty_thresholds() -> None:
    dataset_config = config.OnsetDatasetConfig(
        data_dir="data/train",
        val_data_dir="data/val",
        feature_source=config.FeatureSource.MEL,
    )
    model_config = config.OnsetModelConfig(input_features=2)
    with pytest.raises(ValueError, match="thresholds must be non-empty"):
        dense_overfit_eval.sweep_thresholds_dense_val_event_f1(
            mock.Mock(),
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
