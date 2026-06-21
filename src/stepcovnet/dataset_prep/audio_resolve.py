"""``#MUSIC`` resolution and audio inference."""

from __future__ import annotations

import dataclasses
import pathlib
import re

from stepcovnet.dataset_prep import constants


@dataclasses.dataclass
class AudioResolveResult:
    """Resolved audio file within a raw pack directory.

    Attributes:
        audio_filename: Resolved source audio basename within the raw pack.
        audio_source: ``music_tag`` or ``inferred``.
        audio_resolved_relpath: Path relative to the pack directory.
        warnings: Machine-readable warning codes for inference heuristics.
    """

    audio_filename: str
    audio_source: str
    audio_resolved_relpath: str
    warnings: list[str]


def _normalize_match_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", name.lower())


def list_audio_files(pack_dir: pathlib.Path) -> list[pathlib.Path]:
    """List audio files directly under a pack directory.

    Args:
        pack_dir: Raw song pack directory.

    Returns:
        Sorted audio file paths (non-recursive).
    """
    files: list[pathlib.Path] = []
    if not pack_dir.is_dir():
        return files
    for path in pack_dir.iterdir():
        if path.is_file() and path.suffix.lower() in constants.AUDIO_EXTENSIONS:
            files.append(path)
    return sorted(files, key=lambda item: item.name.lower())


def _find_music_tag_match(
    music_filename: str,
    audio_files: list[pathlib.Path],
) -> pathlib.Path | None:
    if not music_filename.strip():
        return None
    music_path = pathlib.Path(music_filename)
    target_name = music_path.name
    target_key = _normalize_match_key(target_name)
    for path in audio_files:
        if path.name == target_name:
            return path
    lowered = target_name.lower()
    for path in audio_files:
        if path.name.lower() == lowered:
            return path
    for path in audio_files:
        if _normalize_match_key(path.name) == target_key:
            return path
    return None


def _score_audio_candidate(
    path: pathlib.Path,
    *,
    music_filename: str,
    simfile_stem: str,
    title_slug: str,
) -> int:
    stem_key = _normalize_match_key(path.stem)
    if music_filename:
        music_key = _normalize_match_key(pathlib.Path(music_filename).stem)
        if stem_key == music_key:
            return 400
    if simfile_stem and stem_key == _normalize_match_key(simfile_stem):
        return 300
    if title_slug and stem_key == title_slug:
        return 200
    return 100


def _pick_inferred_audio(
    audio_files: list[pathlib.Path],
    *,
    music_filename: str,
    simfile_stem: str,
    title_slug: str,
) -> tuple[pathlib.Path, list[str]]:
    if len(audio_files) == 1:
        return audio_files[0], ["audio_inferred_single_candidate"]

    scored = sorted(
        (
            (
                _score_audio_candidate(
                    path,
                    music_filename=music_filename,
                    simfile_stem=simfile_stem,
                    title_slug=title_slug,
                ),
                path.stat().st_size,
                path.name.lower(),
                path,
            )
            for path in audio_files
        ),
        key=lambda item: (item[0], item[1], item[2]),
        reverse=True,
    )
    winner = scored[0][3]
    runner_ups = [item[3].name for item in scored[1:4]]
    warning = "audio_inferred_heuristic"
    if runner_ups:
        warning = f"audio_inferred_heuristic:{','.join(runner_ups)}"
    return winner, [warning]


def resolve_audio(
    pack_dir: pathlib.Path,
    *,
    music_filename: str,
    simfile_name: str,
    title: str,
) -> AudioResolveResult | None:
    """Resolve the audio file for a pack using ``#MUSIC`` and inference rules.

    Args:
        pack_dir: Raw song pack directory.
        music_filename: Raw ``#MUSIC`` tag value.
        simfile_name: Parsed simfile basename (for stem matching).
        title: Song title used for slug matching.

    Returns:
        Resolved audio metadata, or ``None`` when no audio file is found.
    """
    audio_files = list_audio_files(pack_dir)
    if not audio_files:
        return None

    music_match = _find_music_tag_match(music_filename, audio_files)
    if music_match is not None:
        return AudioResolveResult(
            audio_filename=music_match.name,
            audio_source=constants.AUDIO_SOURCE_MUSIC_TAG,
            audio_resolved_relpath=music_match.name,
            warnings=[],
        )

    title_slug = _normalize_match_key(title)
    simfile_stem = pathlib.Path(simfile_name).stem
    chosen, warnings = _pick_inferred_audio(
        audio_files,
        music_filename=music_filename,
        simfile_stem=simfile_stem,
        title_slug=title_slug,
    )
    return AudioResolveResult(
        audio_filename=chosen.name,
        audio_source=constants.AUDIO_SOURCE_INFERRED,
        audio_resolved_relpath=chosen.name,
        warnings=warnings,
    )
