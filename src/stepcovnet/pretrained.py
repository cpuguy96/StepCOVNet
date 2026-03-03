"""Resolve onset/arrow model paths: use provided path, or download from Google Drive to a cache dir.

Models on Drive are expected as separate zip files containing a single .keras model each.
Paste Google Drive file ID or full share link below; leave empty to require explicit
--onset_model_path / --arrow_model_path.
"""

import os
import pathlib
import re
import zipfile

import gdown

# Paste Google Drive file ID or full share link below; leave empty to require explicit path.
DEFAULT_ONSET_DRIVE_ID: str = "https://drive.google.com/file/d/1p95pj6oJhXS9A60BXsLOl-q6TSht71GY/view?usp=drive_link"
DEFAULT_ARROW_DRIVE_ID: str = "https://drive.google.com/file/d/1DQbvIrYEBGb7zs_CMxlAPGC_mttqVQPH/view?usp=drive_link"

_ONSET_FILENAME = "onset.keras"
_ARROW_FILENAME = "arrow.keras"


def _extract_drive_file_id(url_or_id: str) -> str:
    """Extract Google Drive file ID from a share URL, or return as-is if already an ID.

    Accepts e.g. https://drive.google.com/file/d/XXXX/view?usp=sharing or plain XXXX.
    """
    url_or_id = (url_or_id or "").strip()
    if not url_or_id:
        return ""
    # Match /file/d/<id>/ or /open?id=<id> or uc?id=<id>
    match = re.search(r"/file/d/([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([a-zA-Z0-9_-]+)", url_or_id)
    if match:
        return match.group(1)
    return url_or_id


def get_default_models_dir() -> pathlib.Path:
    """Return the directory where downloaded models are cached. Creates it when writing.

    Returns:
        Path to the cache directory for downloaded models.
    """
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA")
        if not base:
            base = str(pathlib.Path.home())
        root = pathlib.Path(base) / "stepcovnet"
    else:
        root = pathlib.Path.home() / ".stepcovnet"
    return root / "models"


def _download_zip_and_extract_keras(drive_id: str, output_path: pathlib.Path) -> None:
    """Download a zip from Google Drive and extract the single .keras file to output_path.

    The zip is expected to contain exactly one .keras file. If multiple exist, the first
    is used. Raises RuntimeError if the zip contains no .keras file or download fails.
    """
    default_dir = output_path.parent
    default_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://drive.google.com/uc?id={drive_id}"
    zip_path = default_dir / "_download.zip"
    try:
        gdown.download(url, str(zip_path), quiet=False)
        if not zip_path.is_file():
            raise RuntimeError(f"Download from Drive did not produce file: {zip_path}")
        with zipfile.ZipFile(zip_path, "r") as zf:
            keras_members = [n for n in zf.namelist() if n.endswith(".keras")]
            if not keras_members:
                raise RuntimeError(
                    f"Zip from Drive contains no .keras file; got: {zf.namelist()!r}"
                )
            # Extract the first .keras (strip any subdir so we get the file content)
            member = keras_members[0]
            with zf.open(member) as src:
                output_path.write_bytes(src.read())
    finally:
        if zip_path.exists():
            zip_path.unlink(missing_ok=True)


def resolve_onset_model_path(provided_path: str | None) -> str:
    """Return path to onset model: use provided path if set and exists, else cache or download.

    If provided_path is non-empty and the file exists, return it.
    If provided_path is non-empty but file is missing, raise FileNotFoundError.
    If provided_path is empty/None: use file in default dir if present; else if
    DEFAULT_ONSET_DRIVE_ID is set, download via gdown and return path; else raise ValueError.

    Args:
        provided_path: User-supplied path to the onset model, or None to use default/cache.

    Returns:
        Absolute path to the onset model file (.keras).
    """
    if provided_path and provided_path.strip():
        p = pathlib.Path(provided_path.strip())
        if not p.is_file():
            raise FileNotFoundError(f"Onset model path does not exist: {p}")
        return str(p.resolve())

    default_dir = get_default_models_dir()
    cached = default_dir / _ONSET_FILENAME
    if cached.is_file():
        return str(cached)

    drive_id = _extract_drive_file_id(DEFAULT_ONSET_DRIVE_ID)
    if not drive_id:
        raise ValueError(
            "No onset model path provided and DEFAULT_ONSET_DRIVE_ID is not set. "
            "Set it in stepcovnet.pretrained or pass --onset_model_path."
        )

    _download_zip_and_extract_keras(drive_id, cached)
    if not cached.is_file():
        raise RuntimeError(f"Download from Drive did not produce file: {cached}")
    return str(cached)


def resolve_arrow_model_path(provided_path: str | None) -> str:
    """Return path to arrow model: use provided path if set and exists, else cache or download.

    If provided_path is non-empty and the file exists, return it.
    If provided_path is non-empty but file is missing, raise FileNotFoundError.
    If provided_path is empty/None: use file in default dir if present; else if
    DEFAULT_ARROW_DRIVE_ID is set, download via gdown and return path; else raise ValueError.

    Args:
        provided_path: User-supplied path to the arrow model, or None to use default/cache.

    Returns:
        Absolute path to the arrow model file (.keras).
    """
    if provided_path and provided_path.strip():
        p = pathlib.Path(provided_path.strip())
        if not p.is_file():
            raise FileNotFoundError(f"Arrow model path does not exist: {p}")
        return str(p.resolve())

    default_dir = get_default_models_dir()
    cached = default_dir / _ARROW_FILENAME
    if cached.is_file():
        return str(cached)

    drive_id = _extract_drive_file_id(DEFAULT_ARROW_DRIVE_ID)
    if not drive_id:
        raise ValueError(
            "No arrow model path provided and DEFAULT_ARROW_DRIVE_ID is not set. "
            "Set it in stepcovnet.pretrained or pass --arrow_model_path."
        )

    _download_zip_and_extract_keras(drive_id, cached)
    if not cached.is_file():
        raise RuntimeError(f"Download from Drive did not produce file: {cached}")
    return str(cached)
