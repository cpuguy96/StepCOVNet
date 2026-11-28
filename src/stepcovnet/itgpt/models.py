"""PyTorch hierarchical transformer placement from ITGPT (`omalley2026itgpt`).

Re-exports hierarchical transformer blocks (:mod:`stepcovnet.itgpt.transformer`),
beat encoders/heads (:mod:`stepcovnet.itgpt.encoders`), and placement losses
(:mod:`stepcovnet.itgpt.losses`) alongside :class:`ItgptPlacementModel` and
:func:`build_itgpt_placement_model`.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.ddcl import constants as ddcl_constants
from stepcovnet.itgpt import constants
from stepcovnet.itgpt.encoders import (
    ItgptBeatAuxHead,
    ItgptBeatCNNEncoder,
    ItgptBeatMertEncoder,
    ItgptConditionProj,
    ItgptConvHead,
)
from stepcovnet.itgpt.losses import (
    ItgptPlacementLoss,
    compute_beat_aux_loss,
    compute_bernoulli_rdrop_kl,
    grid_importance_weights,
)
from stepcovnet.itgpt.transformer import (
    ItgptEncoderBlock,
    ItgptFFN,
    ItgptGatedCompression,
    ItgptHierarchicalAttnBlock,
    ItgptMultiHeadSelfAttention,
    ItgptNormlessEncoderBlock,
    ItgptRmsNorm,
    _init_linear_or_conv,
)

__all__ = [
    "ItgptBeatAuxHead",
    "ItgptBeatCNNEncoder",
    "ItgptBeatMertEncoder",
    "ItgptConditionProj",
    "ItgptConvHead",
    "ItgptEncoderBlock",
    "ItgptFFN",
    "ItgptGatedCompression",
    "ItgptHierarchicalAttnBlock",
    "ItgptMultiHeadSelfAttention",
    "ItgptNormlessEncoderBlock",
    "ItgptPlacementLoss",
    "ItgptPlacementModel",
    "ItgptRmsNorm",
    "build_itgpt_placement_model",
    "compute_beat_aux_loss",
    "compute_bernoulli_rdrop_kl",
    "grid_importance_weights",
]


class ItgptPlacementModel(nn.Module):
    """PyTorch ITGPT hierarchical placement network (dynamic beat length).

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Positional table length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat (``32`` for log-mel, ``24``
            for MERT).
        feature_dim: Per-frame feature width when ``audio_feature_type == "mert"``.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).
        mert_time_mask_max_frames: SpecAugment maximum contiguous MERT frames
            zeroed per beat during training (``0`` disables).
        mert_channel_dropout: Channel-wise dropout probability across the MERT
            feature axis per beat during training (``0.0`` disables).
        mert_time_mask_preserve_downbeat: If True, keep beat frame ``0``
            unmasked when sampling MERT SpecAugment time masks.
        beat_aux_head: If True, add an auxiliary per-beat activity/density head
            on post-``conv_head`` features with a zero-initialized residual slot
            modulation.
        difficulty_source: Scalar conditioning signal passed as ``difficulty``
            (``"meter"`` uses ``[1, 50]`` bounds; ``"density"`` uses ``[0, 6]``
            mean steps per beat bounds).

    Raises:
        ValueError: If frontend or width arguments are invalid.
    """

    def __init__(
        self,
        *,
        d_model: int = constants.D_MODEL,
        n_heads: int = constants.N_HEADS,
        n_enc_layers: int = constants.N_ENC_LAYERS,
        cnn_hidden: int = constants.CNN_HIDDEN,
        dropout_rate: float = constants.DROPOUT_RATE,
        max_beats: int = constants.MAX_BEATS,
        audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
        n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
        feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
        n_mert_layers: int = 1,
        mert_time_mask_max_frames: int = 0,
        mert_channel_dropout: float = 0.0,
        mert_time_mask_preserve_downbeat: bool = False,
        beat_aux_head: bool = False,
        difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
    ) -> None:
        super().__init__()
        if d_model < 1 or n_heads < 1 or d_model % n_heads != 0:
            raise ValueError(f"n_heads must divide d_model, got {n_heads}, {d_model}")
        if n_enc_layers < 1:
            raise ValueError(f"n_enc_layers must be at least 1, got {n_enc_layers}")
        if audio_feature_type not in ddcl_constants.AUDIO_FEATURE_TYPES:
            raise ValueError(
                "audio_feature_type must be one of "
                f"{ddcl_constants.AUDIO_FEATURE_TYPES}, got {audio_feature_type!r}"
            )
        if (
            audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT
            and n_frames_per_beat != constants.CNN_FRAMES_OUT
        ):
            raise ValueError(
                "MERT ITGPT requires n_frames_per_beat == "
                f"{constants.CNN_FRAMES_OUT}, got {n_frames_per_beat}"
            )
        if feature_dim < 1:
            raise ValueError(f"feature_dim must be positive, got {feature_dim}")
        if n_mert_layers < 1:
            raise ValueError(f"n_mert_layers must be positive, got {n_mert_layers}")
        if mert_time_mask_max_frames < 0:
            raise ValueError(
                "mert_time_mask_max_frames must be >= 0, "
                f"got {mert_time_mask_max_frames}"
            )
        if not 0.0 <= mert_channel_dropout < 1.0:
            raise ValueError(
                f"mert_channel_dropout must be in [0, 1), got {mert_channel_dropout}"
            )
        if difficulty_source not in constants.DIFFICULTY_SOURCES:
            raise ValueError(
                "difficulty_source must be one of "
                f"{constants.DIFFICULTY_SOURCES}, got {difficulty_source!r}"
            )

        self.d_model = int(d_model)
        self.n_heads = int(n_heads)
        self.n_enc_layers = int(n_enc_layers)
        self.cnn_hidden = int(cnn_hidden)
        self.dropout_rate = float(dropout_rate)
        self.max_beats = int(max_beats)
        self.audio_feature_type = str(audio_feature_type)
        self.n_frames_per_beat = int(n_frames_per_beat)
        self.feature_dim = int(feature_dim)
        self.n_mert_layers = int(n_mert_layers)
        self.mert_time_mask_max_frames = int(mert_time_mask_max_frames)
        self.mert_channel_dropout = float(mert_channel_dropout)
        self.mert_time_mask_preserve_downbeat = bool(mert_time_mask_preserve_downbeat)
        self.beat_aux_head = bool(beat_aux_head)
        self.difficulty_source = str(difficulty_source)

        self.beat_cnn: ItgptBeatCNNEncoder | None = None
        self.beat_mert: ItgptBeatMertEncoder | None = None
        if self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            self.beat_mert = ItgptBeatMertEncoder(
                self.feature_dim,
                self.d_model,
                dropout_rate=self.dropout_rate,
                n_mert_layers=self.n_mert_layers,
                mert_time_mask_max_frames=self.mert_time_mask_max_frames,
                mert_channel_dropout=self.mert_channel_dropout,
                mert_time_mask_preserve_downbeat=self.mert_time_mask_preserve_downbeat,
            )
        else:
            self.beat_cnn = ItgptBeatCNNEncoder(self.d_model, self.cnn_hidden)

        diff_min, diff_max = (
            (constants.MIN_DENSITY, constants.MAX_DENSITY)
            if self.difficulty_source == constants.DIFFICULTY_SOURCE_DENSITY
            else (constants.MIN_DIFFICULTY, constants.MAX_DIFFICULTY)
        )
        self.diff_proj = ItgptConditionProj(
            self.d_model,
            diff_min,
            diff_max,
        )
        self.bpm_proj = ItgptConditionProj(
            self.d_model,
            constants.MIN_BPM,
            constants.MAX_BPM,
        )

        self.level1 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=1,
        )
        self.level2 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=4,
        )
        self.level3 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=16,
        )
        self.level4 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=constants.CNN_FRAMES_OUT,
            beats_per_chunk=32,
        )
        self.frame_compress = ItgptGatedCompression(4 * self.d_model, self.d_model)
        self.level5 = ItgptHierarchicalAttnBlock(
            self.d_model,
            self.n_heads,
            self.dropout_rate,
            frames_per_beat=6,
            beats_per_chunk=64,
        )
        self.beat_compress = ItgptGatedCompression(6 * self.d_model, self.d_model)
        self.level6_pos = nn.Embedding(self.max_beats + 200, self.d_model)
        nn.init.normal_(self.level6_pos.weight, mean=0.0, std=0.02)

        self.level6 = nn.ModuleList(
            [
                ItgptEncoderBlock(self.d_model, self.n_heads, self.dropout_rate)
                for _ in range(self.n_enc_layers)
            ]
        )
        self.conv_head = ItgptConvHead(self.d_model, self.dropout_rate)
        self.onset_head = nn.Linear(self.d_model, constants.N_SLOTS)
        _init_linear_or_conv(self.onset_head)
        self.beat_aux: ItgptBeatAuxHead | None = (
            ItgptBeatAuxHead(self.d_model, constants.N_SLOTS)
            if self.beat_aux_head
            else None
        )

    def forward_with_aux(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Run the hierarchical encoder and return float32 logits plus optional aux.

        Args:
            inputs: Dict with ``audio``, ``bpm``, and ``difficulty`` tensors.

        Returns:
            Tuple ``(logits, aux)`` where ``logits`` has shape
            ``(batch, beats, 48)`` in float32 and ``aux`` is a dict with
            ``beat_active_logit`` and ``beat_count_pred`` when
            ``beat_aux_head`` is True, else None.

        Raises:
            RuntimeError: If the active beat encoder was not initialized.
        """
        audio = inputs["audio"]
        batch_size = audio.shape[0]
        n_beats = audio.shape[1]
        n_beat_rows = batch_size * n_beats

        if self.audio_feature_type == ddcl_constants.AUDIO_FEATURE_MERT:
            if self.beat_mert is None:
                raise RuntimeError("beat_mert encoder is not initialized")
            tail = (
                (self.n_mert_layers, self.feature_dim)
                if self.n_mert_layers > 1
                else (self.feature_dim,)
            )
            mert_in = audio.reshape(n_beat_rows, self.n_frames_per_beat, *tail)
            hidden = self.beat_mert(mert_in)
        else:
            if self.beat_cnn is None:
                raise RuntimeError("beat_cnn encoder is not initialized")
            conv_in = audio.reshape(
                n_beat_rows,
                constants.N_FRAMES_PER_BEAT,
                constants.N_MELS,
                constants.N_CHANNELS,
            )
            hidden = self.beat_cnn(conv_in)

        hidden = self.level1(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level2(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level3(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = self.level4(hidden, batch_size=batch_size, n_beats=n_beats)
        hidden = hidden.reshape(n_beat_rows, 6, 4 * self.d_model)
        hidden = self.frame_compress(hidden)
        hidden = self.level5(hidden, batch_size=batch_size, n_beats=n_beats)
        global_h = self.beat_compress(hidden.reshape(n_beat_rows, 6 * self.d_model))
        global_h = global_h.reshape(batch_size, n_beats, self.d_model)

        beat_idx = torch.arange(n_beats, dtype=torch.long, device=global_h.device)
        pos = self.level6_pos(beat_idx)
        global_h = global_h + pos.to(global_h.dtype).unsqueeze(0)
        diff_tok = self.diff_proj(inputs["difficulty"]).to(global_h.dtype)
        bpm_tok = self.bpm_proj(inputs["bpm"]).to(global_h.dtype)
        global_h = torch.cat([diff_tok, bpm_tok, global_h], dim=1)
        for block in self.level6:
            global_h = block(global_h)
        global_h = global_h[:, 2:, :]
        global_h = self.conv_head(global_h)
        # Keep onset head in fp32 for numerical stability under autocast.
        logits = nn_func.linear(
            global_h.float(),
            self.onset_head.weight.float(),
            self.onset_head.bias.float() if self.onset_head.bias is not None else None,
        )
        aux: dict[str, torch.Tensor] | None = None
        if self.beat_aux is not None:
            beat_active_logit, beat_count_pred, slot_delta = self.beat_aux(global_h)
            logits = logits + slot_delta
            aux = {
                "beat_active_logit": beat_active_logit,
                "beat_count_pred": beat_count_pred,
            }
        return logits, aux

    def forward_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the hierarchical encoder and return float32 slot logits.

        Args:
            inputs: Dict with ``audio``, ``bpm``, and ``difficulty`` tensors.

        Returns:
            Float32 logits tensor of shape ``(batch, beats, 48)``.
        """
        logits, _ = self.forward_with_aux(inputs)
        return logits

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Run the hierarchical encoder and return float32 slot probabilities.

        Args:
            inputs: Dict with ``audio``, ``bpm``, and ``difficulty`` tensors.

        Returns:
            Float32 probabilities tensor of shape ``(batch, beats, 48)``.
        """
        return torch.sigmoid(self.forward_logits(inputs))

    def model_config(self) -> dict[str, object]:
        """Return constructor kwargs for checkpoint serialization.

        Returns:
            JSON-serializable constructor kwargs mapping.
        """
        return {
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_enc_layers": self.n_enc_layers,
            "cnn_hidden": self.cnn_hidden,
            "dropout_rate": self.dropout_rate,
            "max_beats": self.max_beats,
            "audio_feature_type": self.audio_feature_type,
            "n_frames_per_beat": self.n_frames_per_beat,
            "feature_dim": self.feature_dim,
            "n_mert_layers": self.n_mert_layers,
            "mert_time_mask_max_frames": self.mert_time_mask_max_frames,
            "mert_channel_dropout": self.mert_channel_dropout,
            "mert_time_mask_preserve_downbeat": self.mert_time_mask_preserve_downbeat,
            "beat_aux_head": self.beat_aux_head,
            "difficulty_source": self.difficulty_source,
        }


def build_itgpt_placement_model(
    *,
    d_model: int = constants.D_MODEL,
    n_heads: int = constants.N_HEADS,
    n_enc_layers: int = constants.N_ENC_LAYERS,
    cnn_hidden: int = constants.CNN_HIDDEN,
    dropout_rate: float = constants.DROPOUT_RATE,
    max_beats: int = constants.MAX_BEATS,
    audio_feature_type: str = ddcl_constants.AUDIO_FEATURE_LOGMEL,
    n_frames_per_beat: int = constants.N_FRAMES_PER_BEAT,
    feature_dim: int = ddcl_constants.MERT_FEATURE_DIM,
    n_mert_layers: int = 1,
    mert_time_mask_max_frames: int = 0,
    mert_channel_dropout: float = 0.0,
    mert_time_mask_preserve_downbeat: bool = False,
    beat_aux_head: bool = False,
    difficulty_source: str = constants.DIFFICULTY_SOURCE_METER,
) -> ItgptPlacementModel:
    """Build the PyTorch ITGPT placement network.

    Args:
        d_model: Transformer width.
        n_heads: Attention heads.
        n_enc_layers: Global encoder depth.
        cnn_hidden: Beat CNN base channels.
        dropout_rate: Dropout.
        max_beats: Maximum padded beat length.
        audio_feature_type: Beat audio frontend (``"logmel"`` or ``"mert"``).
        n_frames_per_beat: Input frames per beat.
        feature_dim: Per-frame width when ``audio_feature_type == "mert"``.
        n_mert_layers: Stacked MERT layer count (``1`` disables layer mixing).
        mert_time_mask_max_frames: SpecAugment maximum contiguous MERT frames
            zeroed per beat during training (``0`` disables).
        mert_channel_dropout: Channel-wise dropout probability across the MERT
            feature axis per beat during training (``0.0`` disables).
        mert_time_mask_preserve_downbeat: If True, keep beat frame ``0``
            unmasked when sampling MERT SpecAugment time masks.
        beat_aux_head: If True, add an auxiliary per-beat activity/density head
            on post-``conv_head`` features with a zero-initialized residual slot
            modulation.
        difficulty_source: Scalar conditioning signal (``"meter"`` or
            ``"density"``).

    Returns:
        Instantiated :class:`ItgptPlacementModel`.
    """
    return ItgptPlacementModel(
        d_model=d_model,
        n_heads=n_heads,
        n_enc_layers=n_enc_layers,
        cnn_hidden=cnn_hidden,
        dropout_rate=dropout_rate,
        max_beats=max_beats,
        audio_feature_type=audio_feature_type,
        n_frames_per_beat=n_frames_per_beat,
        feature_dim=feature_dim,
        n_mert_layers=n_mert_layers,
        mert_time_mask_max_frames=mert_time_mask_max_frames,
        mert_channel_dropout=mert_channel_dropout,
        mert_time_mask_preserve_downbeat=mert_time_mask_preserve_downbeat,
        beat_aux_head=beat_aux_head,
        difficulty_source=difficulty_source,
    )
