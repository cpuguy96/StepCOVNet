"""Print event onset F1 and error counts for a saved onset-event model."""

import argparse

import tensorflow as tf

from stepcovnet.onset_events import config
from stepcovnet.onset_events import datasets
from stepcovnet.onset_events import matching
from stepcovnet.onset_events import metrics
from stepcovnet.onset_events import trainers

PARSER = argparse.ArgumentParser(description="Evaluate event_onset_f1 on one batch.")
PARSER.add_argument("--config", type=str, required=True)
PARSER.add_argument("--model_path", type=str, required=True)
ARGS = PARSER.parse_args()


def main() -> None:
    cfg = config.OnsetEventExperimentConfig.from_json(ARGS.config)
    cfg.run.overfit_one_song = True
    pair = trainers._resolve_single_song_pair(cfg.dataset, cfg.run)
    ds = datasets.create_onset_event_dataset_from_pairs(
        [pair],
        batch_size=cfg.dataset.batch_size,
        max_audio_seconds=cfg.dataset.max_audio_seconds,
        n_max_onsets=cfg.dataset.n_max_onsets,
        max_steps_per_chart=cfg.dataset.max_steps_per_chart,
        target_sample_rate=cfg.dataset.target_sample_rate,
        frontend=cfg.model.frontend,
        mert_features_dir=cfg.dataset.mert_features_dir,
        data_root=cfg.dataset.data_root or cfg.dataset.data_dir,
        shuffle=False,
    )
    batch = next(iter(ds.take(1)))
    model = tf.keras.models.load_model(ARGS.model_path, compile=False)
    model_inputs = trainers._model_inputs_from_batch(batch, model)
    out = model(model_inputs, training=False)
    pt = out["pred_times"].numpy()
    pc = out["pred_confidence"].numpy()
    gt = batch["gt_times"].numpy()
    gm = batch["gt_mask"].numpy()
    tp, fp, fn = metrics.count_event_onset_errors_numpy(
        pt,
        pc,
        gt,
        gm,
        cfg.run.tolerance_sec,
        cfg.run.confidence_threshold,
    )
    _p, _r, f1 = metrics.event_onset_f1_numpy(
        pt,
        pc,
        gt,
        gm,
        cfg.run.tolerance_sec,
        cfg.run.confidence_threshold,
    )
    tp_mg, fp_mg, fn_mg = metrics.count_event_onset_errors_numpy(
        pt,
        pc,
        gt,
        gm,
        cfg.run.tolerance_sec,
        cfg.run.confidence_threshold,
        cfg.run.min_onset_distance_ms,
    )
    _p_mg, _r_mg, f1_mg = metrics.event_onset_f1_numpy(
        pt,
        pc,
        gt,
        gm,
        cfg.run.tolerance_sec,
        cfg.run.confidence_threshold,
        cfg.run.min_onset_distance_ms,
    )
    print("raw tp", tp, "fp", fp, "fn", fn, "f1", float(f1))
    print(
        "mingap tp",
        tp_mg,
        "fp",
        fp_mg,
        "fn",
        fn_mg,
        "f1",
        float(f1_mg),
        "min_gap_ms",
        cfg.run.min_onset_distance_ms,
    )
    print("conf>=0.5", int((pc >= cfg.run.confidence_threshold).sum()))
    match = matching.match_onsets_numpy(pt, gt, gm, tolerance_sec=cfg.run.tolerance_sec)
    print("num_matches_tol", int(match.num_matches[0]))


if __name__ == "__main__":
    main()
