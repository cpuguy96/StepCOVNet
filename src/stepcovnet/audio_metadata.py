"""Zero-dependency audio tag and embedded cover-art reader for StepMania metadata.

Extracts ``title``, ``subtitle``, ``artist``, ``genre``, and embedded album art
directly from common audio containers without requiring external tagging
packages:

- **MP3**: ID3v2.2 / ID3v2.3 / ID3v2.4 (``TIT2``, ``TIT3``, ``TPE1``/``TPE2``,
  ``TCON``, ``APIC``/``PIC``) with ID3v1 trailer fallback.
- **FLAC**: ``VORBIS_COMMENT`` and ``PICTURE`` metadata blocks.
- **OGG** (Vorbis / Opus): Vorbis comment header packet and base64
  ``METADATA_BLOCK_PICTURE`` cover art.
- **MP4 / M4A**: ``moov -> udta -> meta -> ilst`` atoms (``©nam``, ``©ART``,
  ``aART``, ``©gen``, ``covr``).
- **WAV**: ``LIST/INFO`` chunks (``INAM``, ``IART``, ``IGNR``) and embedded
  ``id3 `` chunks.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import pathlib
import re
import struct

# Compact standard ID3v1 genre table (indices 0..79) for legacy numeric TCON/ID3v1 tags.
_ID3V1_GENRES: tuple[str, ...] = (
    "Blues",
    "Classic Rock",
    "Country",
    "Dance",
    "Disco",
    "Funk",
    "Grunge",
    "Hip-Hop",
    "Jazz",
    "Metal",
    "New Age",
    "Oldies",
    "Other",
    "Pop",
    "R&B",
    "Rap",
    "Reggae",
    "Rock",
    "Techno",
    "Industrial",
    "Alternative",
    "Ska",
    "Death Metal",
    "Pranks",
    "Soundtrack",
    "Euro-Techno",
    "Ambient",
    "Trip-Hop",
    "Vocal",
    "Jazz+Funk",
    "Fusion",
    "Trance",
    "Classical",
    "Instrumental",
    "Acid",
    "House",
    "Game",
    "Sound Clip",
    "Gospel",
    "Noise",
    "Alternative Rock",
    "Bass",
    "Soul",
    "Punk",
    "Space",
    "Meditative",
    "Instrumental Pop",
    "Instrumental Rock",
    "Ethnic",
    "Gothic",
    "Darkwave",
    "Techno-Industrial",
    "Electronic",
    "Pop-Folk",
    "Eurodance",
    "Dream",
    "Southern Rock",
    "Comedy",
    "Cult",
    "Gangsta",
    "Top 40",
    "Christian Rap",
    "Pop/Funk",
    "Jungle",
    "Native American",
    "Cabaret",
    "New Wave",
    "Psychedelic",
    "Rave",
    "Showtunes",
    "Trailer",
    "Lo-Fi",
    "Tribal",
    "Acid Punk",
    "Acid Jazz",
    "Polka",
    "Retro",
    "Musical",
    "Rock & Roll",
    "Hard Rock",
)

_ID3_GENRE_NUM_RE = re.compile(r"^\((\d+)\)(.*)$")
_MAX_TAG_READ_BYTES = 8 * 1024 * 1024  # 8 MiB safety cap for header/art parsing


@dataclasses.dataclass(frozen=True)
class AudioMetadata:
    """Extracted audio file metadata for populating StepMania ``.sm`` headers.

    Attributes:
        title: Track title tag (empty if absent).
        subtitle: Track subtitle / version tag (empty if absent).
        artist: Track artist / album artist tag (empty if absent).
        genre: Musical genre tag (empty if absent).
        cover_art_bytes: Raw image bytes for embedded album art, if present.
        cover_art_ext: File extension (e.g. ``".jpg"``, ``".png"``) for
            ``cover_art_bytes``, or empty if no valid cover art was found.
    """

    title: str = ""
    subtitle: str = ""
    artist: str = ""
    genre: str = ""
    cover_art_bytes: bytes | None = None
    cover_art_ext: str = ""


def detect_image_extension(data: bytes, mime_type: str = "") -> str:
    """Detect an image file extension from magic bytes (with MIME fallback).

    Args:
        data: Raw image payload bytes.
        mime_type: Optional MIME type string (e.g. ``"image/jpeg"``).

    Returns:
        Lowercase extension (``".jpg"``, ``".png"``, ``".webp"``, ``".gif"``),
        or ``""`` if the payload is not a recognized image format.
    """
    if len(data) < 4:
        return ""
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    if data[:4] in (b"GIF8",):
        return ".gif"
    mime = mime_type.strip().lower()
    if mime in ("image/jpeg", "image/jpg", "jpg"):
        return ".jpg"
    if mime in ("image/png", "png"):
        return ".png"
    if mime == "image/webp":
        return ".webp"
    if mime == "image/gif":
        return ".gif"
    return ""


def _clean_text(value: str) -> str:
    """Strip NULs, control whitespace, and surrounding spaces from a tag value."""
    cleaned = value.replace("\x00", "").replace("\r", " ").replace("\n", " ").strip()
    return re.sub(r"\s+", " ", cleaned)


def _clean_genre(raw: str) -> str:
    """Normalize ID3v1 numeric genre references like ``(17)`` or ``(17)Rock``."""
    text = _clean_text(raw)
    if not text:
        return ""
    match = _ID3_GENRE_NUM_RE.match(text)
    if match:
        idx = int(match.group(1))
        rest = _clean_text(match.group(2))
        if rest:
            return rest
        if 0 <= idx < len(_ID3V1_GENRES):
            return _ID3V1_GENRES[idx]
        return ""
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(_ID3V1_GENRES):
            return _ID3V1_GENRES[idx]
    return text


def _decode_syncsafe_u32(raw: bytes) -> int:
    return (
        ((raw[0] & 0x7F) << 21)
        | ((raw[1] & 0x7F) << 14)
        | ((raw[2] & 0x7F) << 7)
        | (raw[3] & 0x7F)
    )


def _decode_id3_text(encoding_byte: int, payload: bytes) -> str:
    """Decode an ID3v2 text payload according to its encoding byte (0..3)."""
    if encoding_byte == 0:
        return _clean_text(payload.decode("latin-1", errors="replace"))
    if encoding_byte == 1:
        # UTF-16 with BOM; may contain multiple NUL-separated strings.
        return _clean_text(payload.decode("utf-16", errors="replace"))
    if encoding_byte == 2:
        return _clean_text(payload.decode("utf-16-be", errors="replace"))
    if encoding_byte == 3:
        return _clean_text(payload.decode("utf-8", errors="replace"))
    return _clean_text(payload.decode("utf-8", errors="replace"))


def _skip_id3_terminated_string(payload: bytes, pos: int, encoding_byte: int) -> int:
    """Advance ``pos`` past a NUL-terminated string in an ID3v2 frame."""
    if encoding_byte in (1, 2):
        # UTF-16 uses a 2-byte aligned 0x00 0x00 terminator.
        i = pos
        while i + 1 < len(payload):
            if payload[i] == 0 and payload[i + 1] == 0:
                return i + 2
            i += 2
        return len(payload)
    nul = payload.find(b"\x00", pos)
    return (nul + 1) if nul != -1 else len(payload)


def _parse_id3_apic(
    payload: bytes, *, version_major: int
) -> tuple[int, bytes, str] | None:
    """Parse an ID3v2 ``APIC`` (v2.3/v2.4) or ``PIC`` (v2.2) frame."""
    if len(payload) < 6:
        return None
    enc = payload[0]
    if version_major == 2:
        fmt = payload[1:4].decode("latin-1", errors="replace").strip().lower()
        pic_type = payload[4]
        pos = _skip_id3_terminated_string(payload, 5, enc)
        mime = f"image/{fmt}" if fmt else ""
    else:
        mime_end = payload.find(b"\x00", 1)
        if mime_end == -1 or mime_end + 2 > len(payload):
            return None
        mime = payload[1:mime_end].decode("latin-1", errors="replace").strip()
        pic_type = payload[mime_end + 1]
        pos = _skip_id3_terminated_string(payload, mime_end + 2, enc)
    img_bytes = payload[pos:]
    ext = detect_image_extension(img_bytes, mime)
    if not ext:
        return None
    return pic_type, img_bytes, ext


def _parse_id3v2_bytes(data: bytes) -> AudioMetadata:
    """Parse an ID3v2 tag starting at byte 0 of ``data``."""
    if len(data) < 10 or data[:3] != b"ID3":
        return AudioMetadata()
    version_major = data[3]
    if version_major not in (2, 3, 4):
        return AudioMetadata()
    flags = data[5]
    tag_size = _decode_syncsafe_u32(data[6:10])
    tag_body = data[10 : 10 + tag_size]

    # Undo tag-level unsynchronization if flagged.
    if flags & 0x80:
        tag_body = tag_body.replace(b"\xff\x00", b"\xff")

    pos = 0
    # Skip optional extended header (v2.3 / v2.4).
    if version_major in (3, 4) and (flags & 0x40) and len(tag_body) >= 4:
        if version_major == 4:
            ext_size = _decode_syncsafe_u32(tag_body[:4])
            pos += ext_size
        else:
            ext_size = struct.unpack(">I", tag_body[:4])[0]
            pos += 4 + ext_size

    title = ""
    subtitle = ""
    artist = ""
    album_artist = ""
    genre = ""
    best_art: tuple[int, bytes, str] | None = None

    while pos < len(tag_body):
        if tag_body[pos] == 0:
            break
        if version_major == 2:
            if pos + 6 > len(tag_body):
                break
            frame_id = tag_body[pos : pos + 3].decode("latin-1", errors="replace")
            frame_size = int.from_bytes(tag_body[pos + 3 : pos + 6], "big")
            header_len = 6
            frame_flags = 0
        else:
            if pos + 10 > len(tag_body):
                break
            frame_id = tag_body[pos : pos + 4].decode("latin-1", errors="replace")
            raw_size = tag_body[pos + 4 : pos + 8]
            frame_size = (
                _decode_syncsafe_u32(raw_size)
                if version_major == 4
                else struct.unpack(">I", raw_size)[0]
            )
            frame_flags = struct.unpack(">H", tag_body[pos + 8 : pos + 10])[0]
            header_len = 10

        if frame_size <= 0 or pos + header_len + frame_size > len(tag_body):
            break
        payload = tag_body[pos + header_len : pos + header_len + frame_size]
        pos += header_len + frame_size

        # In ID3v2.4, handle per-frame data-length indicator & unsynchronization.
        if version_major == 4:
            if frame_flags & 0x0002:
                payload = payload.replace(b"\xff\x00", b"\xff")
            if frame_flags & 0x0001 and len(payload) >= 4:
                payload = payload[4:]

        if frame_id in ("TIT2", "TT2") and len(payload) >= 1 and not title:
            title = _decode_id3_text(payload[0], payload[1:])
        elif frame_id in ("TIT3", "TT3") and len(payload) >= 1 and not subtitle:
            subtitle = _decode_id3_text(payload[0], payload[1:])
        elif frame_id in ("TPE1", "TP1") and len(payload) >= 1 and not artist:
            artist = _decode_id3_text(payload[0], payload[1:])
        elif frame_id in ("TPE2", "TP2") and len(payload) >= 1 and not album_artist:
            album_artist = _decode_id3_text(payload[0], payload[1:])
        elif frame_id in ("TCON", "TCO") and len(payload) >= 1 and not genre:
            genre = _clean_genre(_decode_id3_text(payload[0], payload[1:]))
        elif frame_id in ("APIC", "PIC"):
            parsed_art = _parse_id3_apic(payload, version_major=version_major)
            if parsed_art is not None and (
                best_art is None or (parsed_art[0] == 3 and best_art[0] != 3)
            ):
                best_art = parsed_art

    return AudioMetadata(
        title=title,
        subtitle=subtitle,
        artist=artist or album_artist,
        genre=genre,
        cover_art_bytes=best_art[1] if best_art is not None else None,
        cover_art_ext=best_art[2] if best_art is not None else "",
    )


def _read_id3v1(path: pathlib.Path) -> AudioMetadata:
    """Read an ID3v1 128-byte trailer from the end of an MP3 file."""
    try:
        file_size = path.stat().st_size
        if file_size < 128:
            return AudioMetadata()
        with path.open("rb") as f:
            f.seek(file_size - 128)
            tail = f.read(128)
    except OSError:
        return AudioMetadata()
    if len(tail) != 128 or tail[:3] != b"TAG":
        return AudioMetadata()
    title = _clean_text(tail[3:33].decode("latin-1", errors="replace"))
    artist = _clean_text(tail[33:63].decode("latin-1", errors="replace"))
    genre_idx = tail[127]
    genre = _ID3V1_GENRES[genre_idx] if 0 <= genre_idx < len(_ID3V1_GENRES) else ""
    return AudioMetadata(title=title, artist=artist, genre=genre)


def _parse_flac_picture_block(data: bytes) -> tuple[int, bytes, str] | None:
    """Parse a FLAC ``METADATA_BLOCK_PICTURE`` payload."""
    if len(data) < 32:
        return None
    pos = 0
    pic_type, mime_len = struct.unpack(">II", data[pos : pos + 8])
    pos += 8
    if pos + mime_len + 4 > len(data):
        return None
    mime = data[pos : pos + mime_len].decode("latin-1", errors="replace")
    pos += mime_len
    desc_len = struct.unpack(">I", data[pos : pos + 4])[0]
    pos += 4 + desc_len
    if pos + 20 > len(data):
        return None
    # Skip width, height, color depth, indexed colors (4 x uint32 = 16 bytes).
    pos += 16
    img_len = struct.unpack(">I", data[pos : pos + 4])[0]
    pos += 4
    if img_len <= 0 or pos + img_len > len(data):
        return None
    img_bytes = data[pos : pos + img_len]
    ext = detect_image_extension(img_bytes, mime)
    if not ext:
        return None
    return pic_type, img_bytes, ext


def _parse_vorbis_comment_block(
    data: bytes,
) -> tuple[dict[str, str], tuple[int, bytes, str] | None]:
    """Parse a Vorbis comment block into ``(tags_by_upper_key, best_picture)``."""
    tags: dict[str, str] = {}
    best_art: tuple[int, bytes, str] | None = None
    if len(data) < 8:
        return tags, best_art
    vendor_len = struct.unpack("<I", data[:4])[0]
    pos = 4 + vendor_len
    if pos + 4 > len(data):
        return tags, best_art
    count = struct.unpack("<I", data[pos : pos + 4])[0]
    pos += 4
    for _ in range(min(count, 4096)):
        if pos + 4 > len(data):
            break
        entry_len = struct.unpack("<I", data[pos : pos + 4])[0]
        pos += 4
        if pos + entry_len > len(data):
            break
        entry = data[pos : pos + entry_len].decode("utf-8", errors="replace")
        pos += entry_len
        if "=" not in entry:
            continue
        key, val = entry.split("=", 1)
        upper_key = key.strip().upper()
        if upper_key == "METADATA_BLOCK_PICTURE":
            with contextlib.suppress(ValueError):
                pic_raw = base64.b64decode(val.strip())
                parsed_art = _parse_flac_picture_block(pic_raw)
                if parsed_art is not None and (
                    best_art is None or (parsed_art[0] == 3 and best_art[0] != 3)
                ):
                    best_art = parsed_art
        elif upper_key not in tags:
            cleaned = _clean_text(val)
            if cleaned:
                tags[upper_key] = cleaned
    return tags, best_art


def _read_flac(path: pathlib.Path) -> AudioMetadata:
    """Parse ``VORBIS_COMMENT`` and ``PICTURE`` blocks from a FLAC file."""
    tags: dict[str, str] = {}
    best_art: tuple[int, bytes, str] | None = None
    try:
        with path.open("rb") as f:
            if f.read(4) != b"fLaC":
                return AudioMetadata()
            bytes_read = 4
            while bytes_read < _MAX_TAG_READ_BYTES:
                hdr = f.read(4)
                if len(hdr) < 4:
                    break
                bytes_read += 4
                is_last = bool(hdr[0] & 0x80)
                block_type = hdr[0] & 0x7F
                length = int.from_bytes(hdr[1:4], "big")
                if block_type in (4, 6) and length <= _MAX_TAG_READ_BYTES:
                    payload = f.read(length)
                    bytes_read += len(payload)
                    if block_type == 4:
                        vc_tags, vc_art = _parse_vorbis_comment_block(payload)
                        tags.update(vc_tags)
                        if vc_art is not None and (
                            best_art is None or (vc_art[0] == 3 and best_art[0] != 3)
                        ):
                            best_art = vc_art
                    elif block_type == 6:
                        parsed_art = _parse_flac_picture_block(payload)
                        if parsed_art is not None and (
                            best_art is None
                            or (parsed_art[0] == 3 and best_art[0] != 3)
                        ):
                            best_art = parsed_art
                else:
                    f.seek(length, 1)
                    bytes_read += length
                if is_last:
                    break
    except OSError:
        return AudioMetadata()

    return AudioMetadata(
        title=tags.get("TITLE", ""),
        subtitle=tags.get("SUBTITLE", "") or tags.get("VERSION", ""),
        artist=tags.get("ARTIST", "") or tags.get("ALBUMARTIST", ""),
        genre=_clean_genre(tags.get("GENRE", "")),
        cover_art_bytes=best_art[1] if best_art is not None else None,
        cover_art_ext=best_art[2] if best_art is not None else "",
    )


def _read_ogg(path: pathlib.Path) -> AudioMetadata:
    """Parse Vorbis/Opus comment packet and optional picture from an Ogg stream."""
    try:
        with path.open("rb") as f:
            buf = f.read(_MAX_TAG_READ_BYTES)
    except OSError:
        return AudioMetadata()

    packets: list[bytes] = []
    cur = bytearray()
    pos = 0
    while pos + 27 <= len(buf) and len(packets) < 3:
        if buf[pos : pos + 4] != b"OggS":
            break
        n_segments = buf[pos + 26]
        if pos + 27 + n_segments > len(buf):
            break
        seg_table = buf[pos + 27 : pos + 27 + n_segments]
        body_pos = pos + 27 + n_segments
        body_len = sum(seg_table)
        if body_pos + body_len > len(buf):
            break
        seg_offset = body_pos
        for seg_len in seg_table:
            cur.extend(buf[seg_offset : seg_offset + seg_len])
            seg_offset += seg_len
            if seg_len < 255:
                packets.append(bytes(cur))
                cur.clear()
                if len(packets) >= 3:
                    break
        pos = body_pos + body_len

    comment_payload: bytes | None = None
    for pkt in packets:
        if pkt.startswith(b"\x03vorbis"):
            comment_payload = pkt[7:]
            break
        if pkt.startswith(b"OpusTags"):
            comment_payload = pkt[8:]
            break
    if comment_payload is None:
        return AudioMetadata()

    tags, best_art = _parse_vorbis_comment_block(comment_payload)
    return AudioMetadata(
        title=tags.get("TITLE", ""),
        subtitle=tags.get("SUBTITLE", "") or tags.get("VERSION", ""),
        artist=tags.get("ARTIST", "") or tags.get("ALBUMARTIST", ""),
        genre=_clean_genre(tags.get("GENRE", "")),
        cover_art_bytes=best_art[1] if best_art is not None else None,
        cover_art_ext=best_art[2] if best_art is not None else "",
    )


def _iter_mp4_atoms(data: bytes, start: int, end: int):
    """Yield ``(atom_type, payload_start, payload_end)`` inside ``data[start:end]``."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        atom_type = data[pos + 4 : pos + 8]
        header = 8
        if size == 1:
            if pos + 16 > end:
                break
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header = 16
        elif size == 0:
            size = end - pos
        if size < header or pos + size > end:
            break
        yield atom_type, pos + header, pos + size
        pos += size


def _find_mp4_ilst(data: bytes, start: int, end: int) -> tuple[int, int] | None:
    """Locate the ``moov -> udta -> meta -> ilst`` byte span in MP4/M4A data."""
    for atom_type, p_start, p_end in _iter_mp4_atoms(data, start, end):
        if atom_type == b"ilst":
            return p_start, p_end
        if atom_type in (b"moov", b"udta"):
            found = _find_mp4_ilst(data, p_start, p_end)
            if found is not None:
                return found
        elif atom_type == b"meta" and p_start + 4 <= p_end:
            found = _find_mp4_ilst(data, p_start + 4, p_end)
            if found is not None:
                return found
    return None


def _read_mp4(path: pathlib.Path) -> AudioMetadata:
    """Parse iTunes-style tags and cover art from an MP4/M4A file."""
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            if file_size <= _MAX_TAG_READ_BYTES:
                buf = f.read()
            else:
                # Scan top-level atoms to locate moov even when placed at EOF.
                buf = b""
                pos = 0
                while pos + 8 <= file_size:
                    f.seek(pos)
                    hdr = f.read(8)
                    if len(hdr) < 8:
                        break
                    size = struct.unpack(">I", hdr[:4])[0]
                    atom_type = hdr[4:8]
                    if size == 1:
                        ext_sz = f.read(8)
                        if len(ext_sz) < 8:
                            break
                        size = struct.unpack(">Q", ext_sz)[0]
                    elif size == 0:
                        size = file_size - pos
                    if size < 8 or pos + size > file_size:
                        break
                    if atom_type == b"moov" and size <= _MAX_TAG_READ_BYTES:
                        f.seek(pos)
                        buf = f.read(size)
                        break
                    pos += size
    except OSError:
        return AudioMetadata()

    ilst = _find_mp4_ilst(buf, 0, len(buf))
    if ilst is None:
        return AudioMetadata()

    title = ""
    artist = ""
    album_artist = ""
    genre = ""
    cover_bytes: bytes | None = None
    cover_ext = ""

    for atom_type, item_start, item_end in _iter_mp4_atoms(buf, ilst[0], ilst[1]):
        for sub_type, d_start, d_end in _iter_mp4_atoms(buf, item_start, item_end):
            if sub_type != b"data" or d_start + 8 > d_end:
                continue
            payload = buf[d_start + 8 : d_end]
            if atom_type == b"\xa9nam" and not title:
                title = _clean_text(payload.decode("utf-8", errors="replace"))
            elif atom_type == b"\xa9ART" and not artist:
                artist = _clean_text(payload.decode("utf-8", errors="replace"))
            elif atom_type == b"aART" and not album_artist:
                album_artist = _clean_text(payload.decode("utf-8", errors="replace"))
            elif atom_type == b"\xa9gen" and not genre:
                genre = _clean_genre(payload.decode("utf-8", errors="replace"))
            elif atom_type == b"covr" and cover_bytes is None:
                ext = detect_image_extension(payload)
                if ext:
                    cover_bytes = payload
                    cover_ext = ext

    return AudioMetadata(
        title=title,
        artist=artist or album_artist,
        genre=genre,
        cover_art_bytes=cover_bytes,
        cover_art_ext=cover_ext,
    )


def _read_wav_info(path: pathlib.Path) -> AudioMetadata:
    """Parse ``LIST/INFO`` and embedded ``id3 `` chunks from a RIFF WAV file."""
    title = ""
    artist = ""
    genre = ""
    id3_meta = AudioMetadata()
    try:
        file_size = path.stat().st_size
        with path.open("rb") as f:
            hdr = f.read(12)
            if len(hdr) < 12 or hdr[:4] != b"RIFF" or hdr[8:12] != b"WAVE":
                return AudioMetadata()
            pos = 12
            while pos + 8 <= file_size:
                f.seek(pos)
                chunk_hdr = f.read(8)
                if len(chunk_hdr) < 8:
                    break
                chunk_id = chunk_hdr[:4]
                chunk_size = struct.unpack("<I", chunk_hdr[4:8])[0]
                if chunk_id == b"LIST" and 4 <= chunk_size <= _MAX_TAG_READ_BYTES:
                    payload = f.read(chunk_size)
                    if payload[:4] == b"INFO":
                        sub_pos = 4
                        while sub_pos + 8 <= len(payload):
                            sub_id = payload[sub_pos : sub_pos + 4]
                            sub_len = struct.unpack(
                                "<I", payload[sub_pos + 4 : sub_pos + 8]
                            )[0]
                            val_bytes = payload[sub_pos + 8 : sub_pos + 8 + sub_len]
                            text = _clean_text(
                                val_bytes.decode("utf-8", errors="replace")
                            )
                            if sub_id == b"INAM" and not title:
                                title = text
                            elif sub_id == b"IART" and not artist:
                                artist = text
                            elif sub_id == b"IGNR" and not genre:
                                genre = _clean_genre(text)
                            sub_pos += 8 + sub_len + (sub_len % 2)
                elif (
                    chunk_id.lower() == b"id3 "
                    and 10 <= chunk_size <= _MAX_TAG_READ_BYTES
                ):
                    id3_meta = _parse_id3v2_bytes(f.read(chunk_size))
                pos += 8 + chunk_size + (chunk_size % 2)
    except OSError:
        return AudioMetadata()

    return AudioMetadata(
        title=title or id3_meta.title,
        subtitle=id3_meta.subtitle,
        artist=artist or id3_meta.artist,
        genre=genre or id3_meta.genre,
        cover_art_bytes=id3_meta.cover_art_bytes,
        cover_art_ext=id3_meta.cover_art_ext,
    )


def _merge_metadata(primary: AudioMetadata, fallback: AudioMetadata) -> AudioMetadata:
    return AudioMetadata(
        title=primary.title or fallback.title,
        subtitle=primary.subtitle or fallback.subtitle,
        artist=primary.artist or fallback.artist,
        genre=primary.genre or fallback.genre,
        cover_art_bytes=(
            primary.cover_art_bytes
            if primary.cover_art_bytes is not None
            else fallback.cover_art_bytes
        ),
        cover_art_ext=(
            primary.cover_art_ext
            if primary.cover_art_bytes is not None
            else fallback.cover_art_ext
        ),
    )


def read_audio_metadata(audio_path: str | pathlib.Path) -> AudioMetadata:
    """Read embedded text tags and cover art from an audio file.

    Detects container format from magic bytes (MP3/ID3v2/ID3v1, FLAC, OGG
    Vorbis/Opus, MP4/M4A, WAV) and safely returns an empty :class:`AudioMetadata`
    if the file is missing, unreadable, or untagged.

    Args:
        audio_path: Path to an audio file.

    Returns:
        Populated :class:`AudioMetadata` instance.
    """
    path = pathlib.Path(audio_path)
    if not path.is_file():
        return AudioMetadata()
    try:
        with path.open("rb") as f:
            head = f.read(16)
    except OSError:
        return AudioMetadata()
    if len(head) < 4:
        return AudioMetadata()

    meta = AudioMetadata()
    if head[:3] == b"ID3":
        try:
            tag_size = _decode_syncsafe_u32(head[6:10]) if len(head) >= 10 else 0
            read_len = min(10 + tag_size, _MAX_TAG_READ_BYTES)
            with path.open("rb") as f:
                raw_tag = f.read(read_len)
            meta = _parse_id3v2_bytes(raw_tag)
        except OSError:
            meta = AudioMetadata()
        meta = _merge_metadata(meta, _read_id3v1(path))
    elif head[:4] == b"fLaC":
        meta = _read_flac(path)
    elif head[:4] == b"OggS":
        meta = _read_ogg(path)
    elif len(head) >= 8 and head[4:8] == b"ftyp":
        meta = _read_mp4(path)
    elif len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        meta = _read_wav_info(path)
    elif path.suffix.lower() == ".mp3":
        meta = _read_id3v1(path)

    return meta


__all__ = [
    "AudioMetadata",
    "detect_image_extension",
    "read_audio_metadata",
]
