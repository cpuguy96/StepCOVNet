"""Unit tests for zero-dependency audio tag and cover-art extraction."""

from __future__ import annotations

import base64
import pathlib
import struct
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from stepcovnet import audio_metadata, beat_tracking, constants, generator, models
from stepcovnet.dataset_prep import simfile_parser

_SCRIPT_DIR = str(pathlib.Path(__file__).resolve().parent.parent / "scripts")
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import generate as generate_script  # noqa: E402

_FAKE_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01"
_FAKE_PNG = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"


def _encode_syncsafe_u32(val: int) -> bytes:
    return bytes(
        [
            (val >> 21) & 0x7F,
            (val >> 14) & 0x7F,
            (val >> 7) & 0x7F,
            val & 0x7F,
        ]
    )


def _build_id3v2_tag(
    frames: list[tuple[str, bytes]], *, version_major: int = 3
) -> bytes:
    body = bytearray()
    for frame_id, payload in frames:
        size_bytes = (
            _encode_syncsafe_u32(len(payload))
            if version_major == 4
            else struct.pack(">I", len(payload))
        )
        body.extend(frame_id.encode("latin-1"))
        body.extend(size_bytes)
        body.extend(b"\x00\x00")
        body.extend(payload)
    header = b"ID3" + bytes([version_major, 0, 0]) + _encode_syncsafe_u32(len(body))
    return header + bytes(body)


def _build_flac_picture_payload(
    img_bytes: bytes, *, mime: str, pic_type: int = 3
) -> bytes:
    mime_b = mime.encode("latin-1")
    return (
        struct.pack(">II", pic_type, len(mime_b))
        + mime_b
        + struct.pack(">I", 0)
        + struct.pack(">IIII", 100, 100, 24, 0)
        + struct.pack(">I", len(img_bytes))
        + img_bytes
    )


def _build_vorbis_comment_payload(comments: list[str]) -> bytes:
    vendor = b"stepcovnet-test"
    out = bytearray(struct.pack("<I", len(vendor)) + vendor)
    out.extend(struct.pack("<I", len(comments)))
    for c in comments:
        cb = c.encode("utf-8")
        out.extend(struct.pack("<I", len(cb)) + cb)
    return bytes(out)


def _build_ogg_page(packet: bytes, *, page_seq: int) -> bytes:
    segs: list[int] = []
    rem = len(packet)
    while rem >= 255:
        segs.append(255)
        rem -= 255
    segs.append(rem)
    header = (
        b"OggS"
        + b"\x00\x00"
        + struct.pack("<Q", 0)
        + struct.pack("<I", 1)
        + struct.pack("<I", page_seq)
        + struct.pack("<I", 0)
        + bytes([len(segs)])
        + bytes(segs)
    )
    return header + packet


def _build_mp4_atom(atom_type: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + atom_type + payload


class _StubDenseOnset(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.zeros((x.shape[0], x.shape[1], 1), dtype=torch.float32)
        out[:, 10::20, :] = 0.9
        return out


class AudioMetadataReaderTest(unittest.TestCase):
    def test_id3v23_utf16_genre_and_apic_cover(self) -> None:
        title_utf16 = b"\x01" + "Starlight; Mix".encode("utf-16")
        subtitle_utf8 = b"\x03" + b"Extended Edit"
        artist_latin = b"\x00" + b"DJ Step"
        genre_num = b"\x00" + b"(52)"  # Electronic
        apic_back = b"\x00image/png\x00\x04back\x00" + _FAKE_PNG
        apic_front = b"\x00image/jpeg\x00\x03front\x00" + _FAKE_JPEG

        raw = _build_id3v2_tag(
            [
                ("TIT2", title_utf16),
                ("TIT3", subtitle_utf8),
                ("TPE1", artist_latin),
                ("TCON", genre_num),
                ("APIC", apic_back),
                ("APIC", apic_front),
            ],
            version_major=3,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3 = pathlib.Path(tmpdir) / "track.mp3"
            mp3.write_bytes(raw + b"\xff\xfb\x90\x00" + b"\x00" * 64)
            meta = audio_metadata.read_audio_metadata(mp3)

        self.assertEqual(meta.title, "Starlight; Mix")
        self.assertEqual(meta.subtitle, "Extended Edit")
        self.assertEqual(meta.artist, "DJ Step")
        self.assertEqual(meta.genre, "Electronic")
        self.assertEqual(meta.cover_art_bytes, _FAKE_JPEG)
        self.assertEqual(meta.cover_art_ext, ".jpg")

    def test_id3v24_syncsafe_and_id3v1_fallback(self) -> None:
        raw_v24 = _build_id3v2_tag(
            [
                ("TIT2", b"\x03Neon Pulse"),
                ("TPE2", b"\x03Synth Artist"),
                ("TCON", b"\x03(18)Hardcore"),
                ("APIC", b"\x03image/png\x00\x03\x00" + _FAKE_PNG),
            ],
            version_major=4,
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            mp3_v24 = pathlib.Path(tmpdir) / "v24.mp3"
            mp3_v24.write_bytes(raw_v24 + b"\x00" * 32)
            meta = audio_metadata.read_audio_metadata(mp3_v24)
            self.assertEqual(meta.title, "Neon Pulse")
            self.assertEqual(meta.artist, "Synth Artist")
            self.assertEqual(meta.genre, "Hardcore")
            self.assertEqual(meta.cover_art_ext, ".png")
            self.assertEqual(meta.cover_art_bytes, _FAKE_PNG)

            # Pure ID3v1 fallback (no ID3v2 header).
            mp3_v1 = pathlib.Path(tmpdir) / "v1.mp3"
            v1_trailer = (
                b"TAG"
                + b"Oldschool Track".ljust(30, b"\x00")
                + b"Retro Artist".ljust(30, b"\x00")
                + b"Album".ljust(30, b"\x00")
                + b"1999"
                + b"Comment".ljust(30, b"\x00")
                + bytes([3])  # Dance
            )
            mp3_v1.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 64 + v1_trailer)
            meta_v1 = audio_metadata.read_audio_metadata(mp3_v1)
            self.assertEqual(meta_v1.title, "Oldschool Track")
            self.assertEqual(meta_v1.artist, "Retro Artist")
            self.assertEqual(meta_v1.genre, "Dance")

    def test_flac_vorbis_comment_and_picture(self) -> None:
        vc = _build_vorbis_comment_payload(
            [
                "TITLE=Lossless Anthem",
                "SUBTITLE=VIP",
                "ARTIST=FLAC Producer",
                "GENRE=Drum & Bass",
            ]
        )
        pic = _build_flac_picture_payload(_FAKE_PNG, mime="image/png", pic_type=3)
        block_vc = bytes([4]) + len(vc).to_bytes(3, "big") + vc
        block_pic = bytes([0x80 | 6]) + len(pic).to_bytes(3, "big") + pic

        with tempfile.TemporaryDirectory() as tmpdir:
            flac = pathlib.Path(tmpdir) / "song.flac"
            flac.write_bytes(b"fLaC" + block_vc + block_pic)
            meta = audio_metadata.read_audio_metadata(flac)

        self.assertEqual(meta.title, "Lossless Anthem")
        self.assertEqual(meta.subtitle, "VIP")
        self.assertEqual(meta.artist, "FLAC Producer")
        self.assertEqual(meta.genre, "Drum & Bass")
        self.assertEqual(meta.cover_art_ext, ".png")
        self.assertEqual(meta.cover_art_bytes, _FAKE_PNG)

    def test_ogg_vorbis_comments_and_base64_picture(self) -> None:
        pic_b64 = base64.b64encode(
            _build_flac_picture_payload(_FAKE_JPEG, mime="image/jpeg", pic_type=3)
        ).decode("ascii")
        comment_pkt = b"\x03vorbis" + _build_vorbis_comment_payload(
            [
                "TITLE=Ogg Groove",
                "ARTIST=Vorbis Crew",
                "GENRE=House",
                f"METADATA_BLOCK_PICTURE={pic_b64}",
            ]
        )
        id_pkt = b"\x01vorbis" + b"\x00" * 23
        ogg_bytes = _build_ogg_page(id_pkt, page_seq=0) + _build_ogg_page(
            comment_pkt, page_seq=1
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            ogg = pathlib.Path(tmpdir) / "song.ogg"
            ogg.write_bytes(ogg_bytes)
            meta = audio_metadata.read_audio_metadata(ogg)

        self.assertEqual(meta.title, "Ogg Groove")
        self.assertEqual(meta.artist, "Vorbis Crew")
        self.assertEqual(meta.genre, "House")
        self.assertEqual(meta.cover_art_ext, ".jpg")
        self.assertEqual(meta.cover_art_bytes, _FAKE_JPEG)

    def test_mp4_ilst_tags_and_cover_art(self) -> None:
        def data_atom(payload: bytes) -> bytes:
            return _build_mp4_atom(
                b"data", b"\x00\x00\x00\x01\x00\x00\x00\x00" + payload
            )

        ilst = _build_mp4_atom(
            b"ilst",
            _build_mp4_atom(b"\xa9nam", data_atom(b"M4A Hit"))
            + _build_mp4_atom(b"\xa9ART", data_atom(b"AAC Singer"))
            + _build_mp4_atom(b"\xa9gen", data_atom(b"Pop"))
            + _build_mp4_atom(b"covr", data_atom(_FAKE_JPEG)),
        )
        meta_atom = _build_mp4_atom(b"meta", b"\x00\x00\x00\x00" + ilst)
        udta = _build_mp4_atom(b"udta", meta_atom)
        moov = _build_mp4_atom(b"moov", udta)
        ftyp = _build_mp4_atom(b"ftyp", b"M4A \x00\x00\x00\x00M4A ")

        with tempfile.TemporaryDirectory() as tmpdir:
            m4a = pathlib.Path(tmpdir) / "song.m4a"
            m4a.write_bytes(ftyp + moov)
            meta = audio_metadata.read_audio_metadata(m4a)

        self.assertEqual(meta.title, "M4A Hit")
        self.assertEqual(meta.artist, "AAC Singer")
        self.assertEqual(meta.genre, "Pop")
        self.assertEqual(meta.cover_art_ext, ".jpg")
        self.assertEqual(meta.cover_art_bytes, _FAKE_JPEG)

    def test_wav_list_info_tags_and_corrupt_files_safe(self) -> None:
        def subchunk(cid: bytes, text: bytes) -> bytes:
            padded = text + (b"\x00" if len(text) % 2 else b"")
            return cid + struct.pack("<I", len(text)) + padded

        info_body = (
            b"INFO"
            + subchunk(b"INAM", b"Wave Beat\x00")
            + subchunk(b"IART", b"PCM Artist\x00")
            + subchunk(b"IGNR", b"Techno\x00")
        )
        list_chunk = b"LIST" + struct.pack("<I", len(info_body)) + info_body
        riff = b"RIFF" + struct.pack("<I", 4 + len(list_chunk)) + b"WAVE" + list_chunk

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            wav = tmp / "song.wav"
            wav.write_bytes(riff)
            meta = audio_metadata.read_audio_metadata(wav)
            self.assertEqual(meta.title, "Wave Beat")
            self.assertEqual(meta.artist, "PCM Artist")
            self.assertEqual(meta.genre, "Techno")

            # Missing or corrupt files never raise.
            self.assertEqual(
                audio_metadata.read_audio_metadata(tmp / "missing.mp3"),
                audio_metadata.AudioMetadata(),
            )
            bad = tmp / "corrupt.mp3"
            bad.write_bytes(b"ID3\x03\x00\x00\x00\x00\x00\x10TRUNC")
            self.assertEqual(
                audio_metadata.read_audio_metadata(bad),
                audio_metadata.AudioMetadata(),
            )

    def test_generator_auto_metadata_displaybpm_jacket_export_and_cli(self) -> None:
        raw_mp3 = _build_id3v2_tag(
            [
                ("TIT2", b"\x03Auto; Title"),
                ("TIT3", b"\x03Radio Mix"),
                ("TPE1", b"\x03Auto Artist"),
                ("TCON", b"\x03Eurodance"),
                ("APIC", b"\x00image/jpeg\x00\x03\x00" + _FAKE_JPEG),
            ],
            version_major=3,
        )
        onset_model = _StubDenseOnset()
        arrow_model = models.build_arrow_model()
        fake_spec = np.zeros((constants.N_MELS, 120), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = pathlib.Path(tmpdir)
            mp3 = tmp / "raw_track.mp3"
            mp3.write_bytes(raw_mp3 + b"\x00" * 64)

            with (
                mock.patch.object(
                    generator.datasets,
                    "audio_to_spectrogram",
                    return_value=fake_spec,
                    autospec=True,
                ),
                mock.patch.object(
                    generator,
                    "_safe_estimate_sample_preview",
                    return_value=(30.0, 15.0),
                    autospec=True,
                ),
            ):
                # Omitting song_title auto-resolves from the MP3's TIT2 tag!
                out = generator.generate_output_data(
                    audio_path=str(mp3),
                    bpm=132,
                    onset_model=onset_model,
                    arrow_model=arrow_model,
                )

            self.assertEqual(out.title, "Auto; Title")
            self.assertEqual(out.subtitle, "Radio Mix")
            self.assertEqual(out.artist, "Auto Artist")
            self.assertEqual(out.genre, "Eurodance")
            self.assertEqual(out.cover_art_bytes, _FAKE_JPEG)
            self.assertEqual(out.cover_art_ext, ".jpg")

            exported = generator.export_stepmania_song_folder(
                out,
                audio_path=mp3,
                output_dir=tmp / "pack",
            )
            self.assertIsNotNone(exported.jacket_path)
            assert exported.jacket_path is not None
            self.assertTrue(exported.jacket_path.is_file())
            self.assertEqual(exported.jacket_path.name, "jacket.jpg")
            self.assertEqual(exported.jacket_path.read_bytes(), _FAKE_JPEG)

            sm_text = exported.sm_path.read_text(encoding="utf-8")
            # Semicolon in title sanitized to comma so .sm syntax stays valid!
            self.assertIn("#TITLE:Auto, Title;", sm_text)
            self.assertIn("#SUBTITLE:Radio Mix;", sm_text)
            self.assertIn("#ARTIST:Auto Artist;", sm_text)
            self.assertIn("#GENRE:Eurodance;", sm_text)
            self.assertIn("#JACKET:jacket.jpg;", sm_text)
            self.assertIn("#DISPLAYBPM:132.000;", sm_text)
            parsed = simfile_parser.parse_simfile_text(sm_text)
            self.assertEqual(parsed.properties["TITLE"], "Auto, Title")
            self.assertEqual(parsed.properties["ARTIST"], "Auto Artist")
            self.assertEqual(parsed.properties["GENRE"], "Eurodance")
            self.assertEqual(parsed.properties["JACKET"], "jacket.jpg")
            self.assertEqual(parsed.properties["DISPLAYBPM"], "132.000")

            # Variable-tempo grid emits min:max #DISPLAYBPM range.
            var_beats = np.concatenate(
                [
                    np.arange(0.0, 12.0, 0.5, dtype=np.float64),  # 120 BPM
                    12.0 + np.arange(0.0, 9.6, 0.4, dtype=np.float64),  # 150 BPM
                ]
            )
            var_grid = beat_tracking.BeatGrid(
                beat_times=var_beats,
                downbeat_times=var_beats[::4],
                bpm=135.0,
                tracker=beat_tracking.TRACKER_BEAT_THIS,
            )
            var_out = generator.OutputData(
                title="Tempo Shift",
                bpm=135,
                notes={"Challenge": [("0.0", "1000"), ("12.0", "0001")]},
                beat_grid=var_grid,
            )
            var_sm = var_out.generate_sm_output(music_filename="var.ogg")
            self.assertIn("#DISPLAYBPM:120.000:150.000;", var_sm)

        # CLI allows omitting --song_title.
        parsed_cli = generate_script.PARSER.parse_args(
            ["--audio_path", "song.mp3", "--output_dir", "out_pack"]
        )
        self.assertEqual(parsed_cli.song_title, "")


if __name__ == "__main__":
    unittest.main()
