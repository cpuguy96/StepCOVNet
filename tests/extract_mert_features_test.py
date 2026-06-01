import io
import os
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet import constants, ssl_features, wsl_gpu

_SCRIPT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "scripts"))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import extract_mert_features  # noqa: E402


class ExtractMertFeaturesScriptTest(unittest.TestCase):
    def test_main_extracts_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            out_dir = os.path.join(tmpdir, "mert")
            os.makedirs(data_dir)
            audio_path = os.path.join(data_dir, "song.mp3")
            chart_path = os.path.join(data_dir, "song.txt")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(chart_path, "w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")

            argv = [
                f"--data_dir={data_dir}",
                f"--output_dir={out_dir}",
            ]
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                ),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                    return_value=os.path.join(out_dir, "song.mert.npy"),
                ) as mock_extract,
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args.kwargs
            self.assertEqual(call_kwargs["device"], "cpu")

    def test_main_exits_when_no_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                f"--data_dir={tmpdir}",
                f"--output_dir={os.path.join(tmpdir, 'out')}",
            ]
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                ),
                mock.patch.object(sys, "argv", argv),
            ):
                with self.assertRaises(SystemExit):
                    extract_mert_features.main(argv)

    def test_main_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "data")
            out_dir = os.path.join(tmpdir, "mert")
            os.makedirs(data_dir)
            audio_path = os.path.join(data_dir, "song.mp3")
            chart_path = os.path.join(data_dir, "song.txt")
            with open(audio_path, "wb") as audio_file:
                audio_file.write(b"audio")
            with open(chart_path, "w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            existing = ssl_features.mert_npy_path(
                audio_path,
                out_dir,
                data_dir,
            )
            os.makedirs(os.path.dirname(existing), exist_ok=True)
            np.save(
                existing, np.zeros((1, constants.MERT_HIDDEN_SIZE), dtype=np.float32)
            )

            argv = [
                f"--data_dir={data_dir}",
                f"--output_dir={out_dir}",
                "--skip_existing",
            ]
            stdout = io.StringIO()
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                ),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                ) as mock_extract,
                mock.patch("sys.stdout", stdout),
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_not_called()
            self.assertIn("skipped 1 existing", stdout.getvalue())

    def test_main_dispatches_cuda_to_wsl(self):
        argv = [
            "--data_dir=C:\\data",
            "--output_dir=C:\\out",
            "--device=cuda",
        ]
        with mock.patch.object(
            wsl_gpu,
            "maybe_dispatch_for_mert_extract",
            side_effect=SystemExit(0),
        ) as mock_dispatch:
            with self.assertRaises(SystemExit) as ctx:
                extract_mert_features.main(argv)
        self.assertEqual(ctx.exception.code, 0)
        mock_dispatch.assert_called_once()
        dispatch_argv = mock_dispatch.call_args[0][1]
        self.assertTrue(dispatch_argv[0].endswith("extract_mert_features.py"))
        self.assertIn("--device=cuda", dispatch_argv[1:])
