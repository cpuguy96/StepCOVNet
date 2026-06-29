import pathlib
import unittest

REPO = pathlib.Path(__file__).resolve().parents[1]


class ArTideIterConfigBuilderTest(unittest.TestCase):
    def test_build_config_merges_run_overrides(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        spec = config_builder.get_experiment("iter31")
        cfg = config_builder.build_config(spec)
        self.assertNotIn("init_model_path", cfg["run"])
        self.assertEqual(cfg["run"]["checkpoint_metric"], "val_overfit_gate")
        self.assertEqual(
            cfg["run"]["model_output_dir"],
            "models_wsl/ar/tide_overfit_iter/iter31",
        )

    def test_build_config_drops_explicit_null_lambda_inc(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        spec = config_builder.get_experiment("iter05")
        cfg = config_builder.build_config(spec)
        self.assertNotIn("lambda_incremental_consistency", cfg["run"])
        self.assertNotIn("incremental_consistency_max_steps", cfg["run"])

    def test_unknown_experiment_raises(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        with self.assertRaises(KeyError):
            config_builder.get_experiment("iter999")

    def test_config_path_for_versions_attempts(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        self.assertEqual(
            config_builder.config_path_for("iter31", 1).name,
            "iter31.json",
        )
        self.assertEqual(
            config_builder.config_path_for("iter31", 2).name,
            "iter31.attempt2.json",
        )

    def test_prepare_experiment_spec_partial_run(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        spec = config_builder.prepare_experiment_spec(
            {
                "id": "iter99",
                "notes": "delta only",
                "run": {"learning_rate": 1e-5},
            },
        )
        cfg = config_builder.build_config(spec)
        self.assertEqual(cfg["run"]["learning_rate"], 1e-5)
        self.assertEqual(cfg["run"]["epochs"], 200)
        self.assertNotIn("init_model_path", cfg["run"])

    def test_prepare_experiment_spec_strips_init_model_path(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        spec = config_builder.prepare_experiment_spec(
            {
                "id": "iter99",
                "notes": "should not warm-start",
                "run": {
                    "init_model_path": "models_wsl/ar/tide_overfit_iter/iter17/ar_onset_model.keras",
                    "learning_rate": 1e-5,
                },
            },
        )
        self.assertNotIn("init_model_path", spec["run"])
        cfg = config_builder.build_config(spec)
        self.assertNotIn("init_model_path", cfg["run"])

    def test_run_blocks_equal_detects_recipe_change(self) -> None:
        sys_path = str(REPO / "scripts" / "ar_tide_iter")
        if sys_path not in __import__("sys").path:
            __import__("sys").path.insert(0, sys_path)
        import config_builder  # noqa: PLC0415

        left = {"run": {"learning_rate": 1e-5, "epochs": 150}}
        right = {"run": {"learning_rate": 2e-5, "epochs": 150}}
        self.assertTrue(config_builder.run_blocks_equal(left, left))
        self.assertFalse(config_builder.run_blocks_equal(left, right))


if __name__ == "__main__":
    unittest.main()
