from __future__ import annotations

import json
import pathlib
import tempfile
import unittest

from stepcovnet.itgpt import scoreboard


def _write_run(
    root: pathlib.Path,
    name: str,
    *,
    f1_at_05: float = 0.76,
    f1_max: float = 0.77,
    fp: int = 4000,
    fn: int = 4000,
    n_beats: int = 17523,
    weights_file: str = "eval_val_slot48_best.json",
    weights_label: str = "best",
    micro_event_f1: float | None = 0.78,
    drop_shuffle: float | None = 0.14,
    drop_zeros: float | None = 0.72,
    f1_16th: float | None = 0.69,
    f1_24th: float | None = 0.45,
    f1_intro: float | None = 0.76,
    started_utc: str = "",
    framework: str = "pytorch",
) -> pathlib.Path:
    run_dir = root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / weights_file).write_text(
        json.dumps(
            {
                "f1_at_05": f1_at_05,
                "f1_max": f1_max,
                "best_threshold": 0.45,
                "n_charts": 45,
                "n_beats": n_beats,
                "false_positives": fp,
                "false_negatives": fn,
                "weights": weights_label,
                "framework": framework,
            }
        ),
        encoding="utf-8",
    )
    if micro_event_f1 is not None:
        (run_dir / "eval_val_grounding_best.json").write_text(
            json.dumps(
                {
                    "slot_f1_drop_shuffle": drop_shuffle,
                    "slot_f1_drop_zeros": drop_zeros,
                    "conditions": {
                        "none": {
                            "micro_event_f1": micro_event_f1,
                            "skill_event_f1": 0.68,
                        }
                    },
                    "model_path": f"/content/runs/{name}/best.pt",
                }
            ),
            encoding="utf-8",
        )
    if f1_16th is not None:
        (run_dir / "eval_val_breakdown.json").write_text(
            json.dumps(
                {
                    "total_errors": fp + fn,
                    "subdivision_f1": {
                        "16th_off": f1_16th,
                        "24th": f1_24th,
                    },
                    "section_f1": {
                        "intro": f1_intro,
                    },
                }
            ),
            encoding="utf-8",
        )
    if started_utc:
        (run_dir / "run_meta.json").write_text(
            json.dumps({"gpu": "NVIDIA L4", "started_utc": started_utc}),
            encoding="utf-8",
        )
    return run_dir


class ItgptScoreboardTest(unittest.TestCase):
    def test_scan_rank_flags_and_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            colab_dir = root / "experiments" / "colab"
            wsl_dir = root / "models_wsl" / "ddc"
            _write_run(
                colab_dir,
                "itgpt_fraxtil_orig_champ",
                f1_at_05=0.7710,
                f1_max=0.7722,
                fp=4200,
                fn=3900,
                f1_16th=0.70,
                f1_intro=0.76,
                started_utc="2026-11-26T01:00:00+00:00",
            )
            _write_run(
                wsl_dir,
                "itgpt_fraxtil_orig_regressed",
                f1_at_05=0.7500,
                f1_max=0.7550,
                fp=5000,
                fn=4500,
                drop_shuffle=0.04,
                drop_zeros=0.05,
                f1_16th=0.65,
                f1_intro=0.62,
                started_utc="2026-11-26T02:00:00+00:00",
            )
            _write_run(
                colab_dir,
                "itgpt_300t_50v_other",
                f1_at_05=0.6900,
                micro_event_f1=None,
                f1_16th=None,
                weights_file="eval_val_slot48.json",
                weights_label="last",
                framework="",
            )
            # Broken/empty folder should be ignored cleanly
            (colab_dir / "broken_run").mkdir(parents=True, exist_ok=True)
            (colab_dir / "broken_run" / "eval_val_slot48.json").write_text(
                "{bad json", encoding="utf-8"
            )

            rows = scoreboard.scan_runs(
                [colab_dir, wsl_dir, root / "missing"],
                name_filter="fraxtil_orig",
            )
            self.assertEqual(len(rows), 2)
            ranked = scoreboard.rank_rows(rows, sort_by="f1_at_05")
            self.assertEqual(ranked[0].run_name, "itgpt_fraxtil_orig_champ")
            self.assertEqual(ranked[0].source, "colab")
            self.assertEqual(ranked[1].source, "wsl")

            flags = scoreboard.detect_flags(ranked[1], baseline=ranked[0])
            self.assertEqual(
                flags,
                ["LOW_SHUFFLE_DROP", "LOW_ZEROS_DROP", "INTRO_DROP", "16TH_DROP"],
            )

            table = scoreboard.format_markdown_table(
                ranked, baseline_name="champ", top_k=2
            )
            self.assertIn("`itgpt_fraxtil_orig_champ`", table)
            self.assertIn("base", table)
            self.assertIn("-2.10", table)
            self.assertIn("LOW_SHUFFLE_DROP", table)

            # Verify other sort keys
            self.assertEqual(
                scoreboard.rank_rows(rows, sort_by="total_errors")[0].run_name,
                "itgpt_fraxtil_orig_champ",
            )
            self.assertEqual(
                scoreboard.rank_rows(rows, sort_by="started_utc")[0].run_name,
                "itgpt_fraxtil_orig_regressed",
            )
            self.assertEqual(
                scoreboard.rank_rows(rows, sort_by="run_name")[0].run_name,
                "itgpt_fraxtil_orig_champ",
            )
            self.assertEqual(
                scoreboard.rank_rows(rows, sort_by="micro_event_f1")[0].run_name,
                "itgpt_fraxtil_orig_champ",
            )
            self.assertEqual(
                scoreboard.rank_rows(rows, sort_by="f1_max")[0].run_name,
                "itgpt_fraxtil_orig_champ",
            )

    def test_plateau_detection_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp) / "custom_runs"
            _write_run(
                root,
                "run_0",
                f1_at_05=0.7700,
                started_utc="2026-11-26T00:00:00+00:00",
            )
            for idx in range(1, 6):
                _write_run(
                    root,
                    f"run_{idx}",
                    f1_at_05=0.7680 + idx * 0.0002,
                    started_utc=f"2026-11-26T0{idx}:00:00+00:00",
                )
            rows = scoreboard.scan_runs([root])
            self.assertTrue(scoreboard.detect_plateau(rows, window=5, tolerance=0.002))
            md = scoreboard.format_markdown_table(scoreboard.rank_rows(rows))
            self.assertIn("Plateau warning", md)

            # Add a breakout run
            _write_run(
                root,
                "run_6",
                f1_at_05=0.7800,
                started_utc="2026-11-26T09:00:00+00:00",
            )
            rows2 = scoreboard.scan_runs([root])
            self.assertFalse(
                scoreboard.detect_plateau(rows2, window=5, tolerance=0.002)
            )

        self.assertIn("No matching", scoreboard.format_markdown_table([]))
        with self.assertRaises(ValueError):
            scoreboard.scan_runs([], weights="invalid")
        with self.assertRaises(ValueError):
            scoreboard.rank_rows([], sort_by="invalid")


if __name__ == "__main__":
    unittest.main()
