import io
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from stepcovnet import constants, pairing, ssl_features, wsl_gpu

_SCRIPT_DIR = str(
    pathlib.Path(pathlib.Path(__file__).resolve()).resolve().parent.parent / "scripts"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import extract_mert_features  # noqa: E402


class ExtractMertFeaturesScriptTest(unittest.TestCase):
    def test_main_extracts_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = pathlib.Path(tmpdir) / "data"
            out_dir = pathlib.Path(tmpdir) / "mert"
            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
            audio_path = pathlib.Path(data_dir) / "song.mp3"
            chart_path = pathlib.Path(data_dir) / "song.txt"
            with pathlib.Path(audio_path).open("wb") as audio_file:
                audio_file.write(b"audio")
            with pathlib.Path(chart_path).open("w") as chart_file:
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
                    autospec=True,
                ),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ssl_features,
                    "_load_mert_model",
                    return_value=(mock.Mock(), mock.Mock()),
                    autospec=True,
                ),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                    return_value=str(pathlib.Path(out_dir) / "song.mert.npy"),
                    autospec=True,
                ) as mock_extract,
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args.kwargs
            self.assertEqual(call_kwargs["device"], "cpu")

    def test_main_extracts_pairs_beside_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = pathlib.Path(tmpdir) / "data"
            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
            audio_path = pathlib.Path(data_dir) / "song.mp3"
            chart_path = pathlib.Path(data_dir) / "song.txt"
            with pathlib.Path(audio_path).open("wb") as audio_file:
                audio_file.write(b"audio")
            with pathlib.Path(chart_path).open("w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")

            argv = [
                f"--data_dir={data_dir}",
                "--beside_audio",
            ]
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                    autospec=True,
                ),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ssl_features,
                    "_load_mert_model",
                    return_value=(mock.Mock(), mock.Mock()),
                    autospec=True,
                ),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                    return_value=str(pathlib.Path(data_dir) / "song.mert.npy"),
                    autospec=True,
                ) as mock_extract,
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_called_once()
            output_path = mock_extract.call_args.args[1]
            self.assertTrue(str(output_path).endswith("song.mert.npy"))

    def test_main_training_index_beside_audio(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = str(pathlib.Path(tmpdir) / "song.ogg")
            argv = [
                f"--training_index_path={tmpdir}/training_index.json",
                "--beside_audio",
            ]
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                    autospec=True,
                ),
                mock.patch.object(
                    pairing,
                    "list_unique_audio_paths",
                    return_value=([audio_path], tmpdir),
                    autospec=True,
                ),
                mock.patch.object(
                    ssl_features,
                    "_load_mert_model",
                    return_value=(mock.Mock(), mock.Mock()),
                    autospec=True,
                ),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                    autospec=True,
                ) as mock_extract,
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args.kwargs
            self.assertEqual(mock_extract.call_args.args[0], audio_path)
            self.assertEqual(
                mock_extract.call_args.args[1],
                ssl_features.mert_npy_path(audio_path, "", tmpdir),
            )
            self.assertEqual(call_kwargs["device"], "cpu")
            self.assertIsNotNone(call_kwargs["model"])
            self.assertIsNotNone(call_kwargs["processor"])

    def test_main_exits_when_no_pairs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                f"--data_dir={tmpdir}",
                f"--output_dir={pathlib.Path(tmpdir) / 'out'}",
            ]
            with (
                mock.patch.object(
                    wsl_gpu,
                    "maybe_dispatch_for_mert_extract",
                    return_value=False,
                    autospec=True,
                ),
                mock.patch.object(sys, "argv", argv),
                self.assertRaises(SystemExit),
            ):
                extract_mert_features.main(argv)

    def test_main_skip_existing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = pathlib.Path(tmpdir) / "data"
            out_dir = pathlib.Path(tmpdir) / "mert"
            pathlib.Path(data_dir).mkdir(parents=True, exist_ok=True)
            audio_path = pathlib.Path(data_dir) / "song.mp3"
            chart_path = pathlib.Path(data_dir) / "song.txt"
            with pathlib.Path(audio_path).open("wb") as audio_file:
                audio_file.write(b"audio")
            with pathlib.Path(chart_path).open("w") as chart_file:
                chart_file.write("TITLE test\nBPM 120\nNOTES\n")
            existing = ssl_features.mert_npy_path(
                audio_path,
                out_dir,
                data_dir,
            )
            pathlib.Path(existing).parent.mkdir(parents=True, exist_ok=True)
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
                    autospec=True,
                ),
                mock.patch.object(sys, "argv", argv),
                mock.patch.object(
                    ssl_features,
                    "extract_and_save_mert_features",
                    autospec=True,
                ) as mock_extract,
                mock.patch.object(sys, "stdout", stdout),
            ):
                extract_mert_features.main(argv)
            mock_extract.assert_not_called()
            self.assertIn("Nothing to do", stdout.getvalue())

    def test_main_dispatches_cuda_to_wsl(self):
        argv = [
            "--data_dir=C:\\data",
            "--output_dir=C:\\out",
            "--device=cuda",
        ]
        with (
            mock.patch.object(
                wsl_gpu,
                "maybe_dispatch_for_mert_extract",
                side_effect=SystemExit(0),
            ) as mock_dispatch,
            self.assertRaises(SystemExit) as ctx,
        ):
            extract_mert_features.main(argv)
        self.assertEqual(ctx.exception.code, 0)
        mock_dispatch.assert_called_once()
        dispatch_argv = mock_dispatch.call_args[0][1]
        self.assertTrue(dispatch_argv[0].endswith("extract_mert_features.py"))
        self.assertIn("--device=cuda", dispatch_argv[1:])
