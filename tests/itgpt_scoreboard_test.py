from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

SCRIPTS_DIR = pathlib.Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import itgpt_scoreboard  # noqa: E402


class ItgptScoreboardScriptTest(unittest.TestCase):
    def test_cli_markdown_json_jsonl_and_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            colab = root / "experiments" / "colab"
            run_a = colab / "itgpt_fraxtil_orig_a"
            run_b = colab / "itgpt_fraxtil_orig_b"
            run_a.mkdir(parents=True)
            run_b.mkdir(parents=True)
            (run_a / "eval_val_slot48_best.json").write_text(
                json.dumps(
                    {
                        "f1_at_05": 0.77,
                        "f1_max": 0.78,
                        "n_charts": 45,
                        "n_beats": 17523,
                        "false_positives": 4100,
                        "false_negatives": 3900,
                    }
                ),
                encoding="utf-8",
            )
            (run_b / "eval_val_slot48_best.json").write_text(
                json.dumps(
                    {
                        "f1_at_05": 0.75,
                        "f1_max": 0.76,
                        "n_charts": 45,
                        "n_beats": 17523,
                    }
                ),
                encoding="utf-8",
            )
            out_file = root / "out" / "board.json"

            with mock.patch.object(itgpt_scoreboard, "REPO", root):
                buf_json = io.StringIO()
                with redirect_stdout(buf_json):
                    rc = itgpt_scoreboard.main(
                        [
                            "--roots",
                            "experiments/colab",
                            "--filter",
                            "fraxtil_orig",
                            "--format",
                            "json",
                            "--top",
                            "1",
                            "--output",
                            str(out_file),
                        ]
                    )
                self.assertEqual(rc, 0)
                parsed = json.loads(buf_json.getvalue())
                self.assertEqual(len(parsed["runs"]), 1)
                self.assertEqual(parsed["runs"][0]["run_name"], "itgpt_fraxtil_orig_a")
                self.assertTrue(out_file.is_file())

                buf_jsonl = io.StringIO()
                with redirect_stdout(buf_jsonl):
                    rc2 = itgpt_scoreboard.main(
                        [
                            "--roots",
                            str(colab),
                            "--format",
                            "jsonl",
                        ]
                    )
                self.assertEqual(rc2, 0)
                lines = [
                    json.loads(line)
                    for line in buf_jsonl.getvalue().strip().splitlines()
                ]
                self.assertEqual(len(lines), 2)

                buf_md = io.StringIO()
                with redirect_stdout(buf_md):
                    rc3 = itgpt_scoreboard.main(
                        [
                            "--roots",
                            str(colab),
                            "--format",
                            "markdown",
                            "--baseline",
                            "itgpt_fraxtil_orig_b",
                        ]
                    )
                self.assertEqual(rc3, 0)
                self.assertIn("+2.00", buf_md.getvalue())


if __name__ == "__main__":
    unittest.main()
