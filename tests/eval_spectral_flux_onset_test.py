import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

_SCRIPT_DIR = str(
    pathlib.Path(pathlib.Path(__file__).resolve()).resolve().parent.parent / "scripts"
)
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import eval_spectral_flux_onset  # noqa: E402


class EvalSpectralFluxOnsetTest(unittest.TestCase):
    def test_spectral_flux_envelope_normalizes_peak_to_one(self) -> None:
        fake_strength = np.array([0.0, 2.0, 4.0, 1.0], dtype=np.float64)
        with (
            mock.patch(
                "eval_spectral_flux_onset.librosa.load",
                return_value=(np.zeros(4), 44100),
            ),
            mock.patch(
                "eval_spectral_flux_onset.librosa.onset.onset_strength",
                return_value=fake_strength,
            ),
        ):
            envelope = eval_spectral_flux_onset.spectral_flux_envelope("fake.wav")
        np.testing.assert_allclose(envelope, [0.0, 0.5, 1.0, 0.25], rtol=0.0, atol=1e-6)

    def test_main_writes_report(self) -> None:
        fake_report = {
            "eval_split": "data/v2/val",
            "num_songs": 1,
            "mean_event_f1": 0.4,
            "micro_event_f1": 0.45,
            "micro_tp": 10.0,
            "micro_fp": 5.0,
            "micro_fn": 5.0,
            "eval_kwargs": {"confidence_threshold": 0.2},
            "per_song": {},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = pathlib.Path(tmpdir) / "flux.json"
            with mock.patch(
                "eval_spectral_flux_onset.eval_spectral_flux_val",
                return_value=fake_report,
            ):
                exit_code = eval_spectral_flux_onset.main(
                    [
                        "--val_data_dir=data/v2/val",
                        f"--output={output_path}",
                        "--threshold=0.2",
                        "--max_songs=1",
                    ],
                )
            self.assertEqual(exit_code, 0)
            with pathlib.Path(output_path).open(encoding="utf-8") as out_file:
                report = json.load(out_file)
            self.assertEqual(report["micro_event_f1"], 0.45)


if __name__ == "__main__":
    unittest.main()
