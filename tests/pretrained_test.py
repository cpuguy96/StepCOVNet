"""Tests for stepcovnet.pretrained (resolve model paths, download from Drive)."""

import os
import pathlib
import tempfile
import unittest
import zipfile
from unittest import mock

from stepcovnet import pretrained


class GetDefaultModelsDirTest(unittest.TestCase):
    """Tests for get_default_models_dir."""

    def test_returns_path_under_home_or_localappdata(self):
        """get_default_models_dir returns a Path ending with models; no crash."""
        path = pretrained.get_default_models_dir()
        self.assertIsInstance(path, pathlib.Path)
        self.assertEqual(path.name, "models")
        self.assertIn("stepcovnet", path.parts)

    def test_on_posix_uses_home(self):
        """When os.name is not nt, path is under Path.home() / .stepcovnet."""
        fake_home = pathlib.Path(tempfile.gettempdir()) / "fake_home"
        with mock.patch("os.name", "posix"):
            with mock.patch.object(pathlib.Path, "home", return_value=fake_home):
                path = pretrained.get_default_models_dir()
        self.assertEqual(path.name, "models")
        self.assertIn(".stepcovnet", path.parts)
        self.assertEqual(path, fake_home / ".stepcovnet" / "models")

    def test_on_windows_uses_localappdata_when_set(self):
        """On Windows, when LOCALAPPDATA is set, path is under it."""
        if os.name != "nt":
            self.skipTest("Windows only")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": "C:\\CustomAppData"}, clear=False):
            path = pretrained.get_default_models_dir()
        self.assertTrue(path.parts[0].startswith("C"), path.parts)
        self.assertIn("CustomAppData", path.parts)
        self.assertIn("stepcovnet", path.parts)
        self.assertEqual(path.name, "models")

    def test_on_windows_falls_back_to_home_when_localappdata_unset(self):
        """On Windows, when LOCALAPPDATA is unset or empty, path uses Path.home()."""
        if os.name != "nt":
            self.skipTest("Windows only")
        with mock.patch.dict(os.environ, {"LOCALAPPDATA": ""}, clear=False):
            path = pretrained.get_default_models_dir()
        self.assertIn("stepcovnet", path.parts)
        self.assertEqual(path.name, "models")


class ExtractDriveFileIdTest(unittest.TestCase):
    """Tests for _extract_drive_file_id."""

    def test_full_share_url_returns_id(self):
        """Full share URL returns the file ID."""
        url = "https://drive.google.com/file/d/1abcXYZ123/view?usp=sharing"
        self.assertEqual(pretrained._extract_drive_file_id(url), "1abcXYZ123")

    def test_uc_id_query_returns_id(self):
        """URL with uc?id= returns the ID."""
        url = "https://drive.google.com/uc?id=1abcXYZ123"
        self.assertEqual(pretrained._extract_drive_file_id(url), "1abcXYZ123")

    def test_plain_id_returns_same(self):
        """Plain file ID is returned as-is."""
        self.assertEqual(pretrained._extract_drive_file_id("1abcXYZ123"), "1abcXYZ123")

    def test_empty_string_returns_empty(self):
        """Empty string returns empty."""
        self.assertEqual(pretrained._extract_drive_file_id(""), "")
        self.assertEqual(pretrained._extract_drive_file_id("   "), "")

    def test_none_returns_empty(self):
        """None is treated as empty."""
        self.assertEqual(pretrained._extract_drive_file_id(None), "")

    def test_malformed_url_returns_as_is(self):
        """URL with no /file/d/ or id= returns input as-is."""
        self.assertEqual(
            pretrained._extract_drive_file_id("https://example.com/other"),
            "https://example.com/other",
        )


class DownloadZipAndExtractKerasTest(unittest.TestCase):
    """Tests for _download_zip_and_extract_keras."""

    def test_zip_with_no_keras_raises_runtimeerror(self):
        """When the zip contains no .keras file, raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = pathlib.Path(tmpdir) / "out.keras"
            with mock.patch("stepcovnet.pretrained.gdown.download") as mock_download:
                def create_zip_without_keras(url, output, **kwargs):
                    with zipfile.ZipFile(output, "w") as zf:
                        zf.writestr("readme.txt", b"no model here")

                mock_download.side_effect = create_zip_without_keras
                with self.assertRaises(RuntimeError) as ctx:
                    pretrained._download_zip_and_extract_keras("id", out_path)
                self.assertIn("contains no .keras file", str(ctx.exception))
                self.assertIn("readme.txt", str(ctx.exception))


class ResolveOnsetModelPathTest(unittest.TestCase):
    """Tests for resolve_onset_model_path."""

    def test_provided_path_exists_returns_it(self):
        """When provided_path exists, returns that path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "onset.keras")
            with open(path, "w") as f:
                f.write("")
            result = pretrained.resolve_onset_model_path(path)
            self.assertEqual(result, os.path.abspath(path))

    def test_provided_path_missing_raises_filenotfounderror(self):
        """When provided_path is set but file does not exist, raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            pretrained.resolve_onset_model_path("/nonexistent/onset.keras")
        self.assertIn("Onset model path does not exist", str(ctx.exception))

    def test_none_and_file_in_default_dir_returns_cached_path(self):
        """When provided_path is None and file exists in default dir, returns that path (no download)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cached = os.path.join(tmpdir, pretrained._ONSET_FILENAME)
            with open(cached, "w") as f:
                f.write("")
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                result = pretrained.resolve_onset_model_path(None)
        self.assertEqual(result, cached)

    def test_none_and_empty_drive_id_raises_valueerror(self):
        """When provided_path is None and DEFAULT_ONSET_DRIVE_ID is empty, raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with self.assertRaises(ValueError) as ctx:
                    pretrained.resolve_onset_model_path(None)
        self.assertIn("DEFAULT_ONSET_DRIVE_ID", str(ctx.exception))
        self.assertIn("--onset_model_path", str(ctx.exception))

    def test_none_and_drive_id_set_downloads_zip_and_returns_path(self):
        """When provided_path is None, no cached file, and Drive ID set, downloads zip and extracts .keras."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with mock.patch.object(pretrained, "DEFAULT_ONSET_DRIVE_ID", "test_drive_id_123"):
                    with mock.patch("stepcovnet.pretrained.gdown.download") as mock_download:
                        out_path = os.path.join(tmpdir, pretrained._ONSET_FILENAME)

                        def create_zip_with_keras(url, output, **kwargs):
                            with zipfile.ZipFile(output, "w") as zf:
                                zf.writestr("model.keras", b"dummy")

                        mock_download.side_effect = create_zip_with_keras
                        result = pretrained.resolve_onset_model_path(None)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.isfile(out_path))
            mock_download.assert_called_once()
            call_args = mock_download.call_args[0]
            self.assertIn("test_drive_id_123", call_args[0])
            self.assertTrue(call_args[1].endswith("_download.zip"))

    def test_empty_string_treated_as_none(self):
        """Empty or whitespace provided_path is treated as None (use default/cache/download)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cached = os.path.join(tmpdir, pretrained._ONSET_FILENAME)
            with open(cached, "w") as f:
                f.write("")
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                for empty in ("", "  ", "\t"):
                    result = pretrained.resolve_onset_model_path(empty)
                    self.assertEqual(result, cached)

    def test_download_fails_to_create_zip_raises_runtimeerror(self):
        """When Drive ID is set but gdown does not create the zip, raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with mock.patch.object(pretrained, "DEFAULT_ONSET_DRIVE_ID", "id123"):
                    with mock.patch("stepcovnet.pretrained.gdown.download"):  # no side_effect: no zip created
                        with self.assertRaises(RuntimeError) as ctx:
                            pretrained.resolve_onset_model_path(None)
                        self.assertIn("Download from Drive did not produce file", str(ctx.exception))


class ResolveArrowModelPathTest(unittest.TestCase):
    """Tests for resolve_arrow_model_path."""

    def test_provided_path_exists_returns_it(self):
        """When provided_path exists, returns that path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "arrow.keras")
            with open(path, "w") as f:
                f.write("")
            result = pretrained.resolve_arrow_model_path(path)
            self.assertEqual(result, os.path.abspath(path))

    def test_provided_path_missing_raises_filenotfounderror(self):
        """When provided_path is set but file does not exist, raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError) as ctx:
            pretrained.resolve_arrow_model_path("/nonexistent/arrow.keras")
        self.assertIn("Arrow model path does not exist", str(ctx.exception))

    def test_none_and_file_in_default_dir_returns_cached_path(self):
        """When provided_path is None and file exists in default dir, returns that path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cached = os.path.join(tmpdir, pretrained._ARROW_FILENAME)
            with open(cached, "w") as f:
                f.write("")
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                result = pretrained.resolve_arrow_model_path(None)
        self.assertEqual(result, cached)

    def test_none_and_empty_drive_id_raises_valueerror(self):
        """When provided_path is None and DEFAULT_ARROW_DRIVE_ID is empty, raises ValueError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with self.assertRaises(ValueError) as ctx:
                    pretrained.resolve_arrow_model_path(None)
        self.assertIn("DEFAULT_ARROW_DRIVE_ID", str(ctx.exception))
        self.assertIn("--arrow_model_path", str(ctx.exception))

    def test_none_and_drive_id_set_downloads_zip_and_returns_path(self):
        """When provided_path is None, no cached file, and Drive ID set, downloads zip and extracts .keras."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with mock.patch.object(pretrained, "DEFAULT_ARROW_DRIVE_ID", "arrow_id_456"):
                    with mock.patch("stepcovnet.pretrained.gdown.download") as mock_download:
                        out_path = os.path.join(tmpdir, pretrained._ARROW_FILENAME)

                        def create_zip_with_keras(url, output, **kwargs):
                            with zipfile.ZipFile(output, "w") as zf:
                                zf.writestr("arrow.keras", b"dummy")

                        mock_download.side_effect = create_zip_with_keras
                        result = pretrained.resolve_arrow_model_path(None)
            self.assertEqual(result, out_path)
            self.assertTrue(os.path.isfile(out_path))
            mock_download.assert_called_once()
            call_args = mock_download.call_args[0]
            self.assertIn("arrow_id_456", call_args[0])
            self.assertTrue(call_args[1].endswith("_download.zip"))

    def test_download_fails_to_create_zip_raises_runtimeerror(self):
        """When Drive ID is set but gdown does not create the zip, raises RuntimeError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with mock.patch.object(pretrained, "get_default_models_dir", return_value=pathlib.Path(tmpdir)):
                with mock.patch.object(pretrained, "DEFAULT_ARROW_DRIVE_ID", "id456"):
                    with mock.patch("stepcovnet.pretrained.gdown.download"):
                        with self.assertRaises(RuntimeError) as ctx:
                            pretrained.resolve_arrow_model_path(None)
                        self.assertIn("Download from Drive did not produce file", str(ctx.exception))
