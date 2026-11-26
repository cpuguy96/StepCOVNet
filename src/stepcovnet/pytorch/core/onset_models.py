"""PyTorch dense frame-wise onset detection architectures (TCN, BiLSTM, Transformer, U-Net WaveNet)."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet import config, constants


def _init_keras_style(module: nn.Module) -> None:
    """Initialize linear/conv layers with Xavier uniform weights and zero bias.

    Args:
        module: Layer to initialize in-place.
    """
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class CausalConv1d(nn.Module):
    """1D convolution with left-only causal padding matching Keras ``padding='causal'``.

    Args:
        in_channels: Input channel width.
        out_channels: Output channel width.
        kernel_size: Temporal filter width.
        dilation: Dilation factor.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.left_pad = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=int(kernel_size),
            dilation=int(dilation),
            padding=0,
        )
        _init_keras_style(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply causal 1D convolution to ``(B, C, T)``.

        Args:
            x: Channels-first tensor ``(B, C, T)``.

        Returns:
            Output tensor ``(B, out_channels, T)``.
        """
        if self.left_pad > 0:
            x = nn_func.pad(x, (self.left_pad, 0))
        return self.conv(x)


class WaveNetResidualBlock(nn.Module):
    """Single gated WaveNet residual block operating on channels-last ``(B, T, C)``.

    Args:
        residual_channels: Width of the residual stream.
        skip_channels: Width of the skip connection output.
        dilation_rate: Causal convolution dilation rate.
        kernel_size: Temporal kernel size.
    """

    def __init__(
        self,
        residual_channels: int,
        skip_channels: int,
        dilation_rate: int,
        kernel_size: int,
    ) -> None:
        super().__init__()
        self.residual_channels = int(residual_channels)
        self.dilated_conv = CausalConv1d(
            residual_channels,
            residual_channels * 2,
            kernel_size,
            dilation=dilation_rate,
        )
        self.res_conv = nn.Conv1d(residual_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, kernel_size=1)
        self.ln = nn.LayerNorm(residual_channels, eps=1e-3)
        _init_keras_style(self.res_conv)
        _init_keras_style(self.skip_conv)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply gated dilated conv, residual connection, and LayerNorm.

        Args:
            x: Input tensor ``(B, T, residual_channels)``.

        Returns:
            Tuple ``(residual_out, skip_out)`` in channels-last layout.
        """
        x_cf = x.transpose(1, 2)
        h = self.dilated_conv(x_cf)
        h_tanh = torch.tanh(h[:, : self.residual_channels, :])
        h_sig = torch.sigmoid(h[:, self.residual_channels :, :])
        gated = h_tanh * h_sig
        res = self.res_conv(gated).transpose(1, 2)
        skip = self.skip_conv(gated).transpose(1, 2)
        return self.ln(x + res), skip


class DenseOnsetOutputHead(nn.Module):
    """Shared per-frame onset logit/probability head matching Keras dense models.

    Args:
        in_channels: Input feature width.
        dropout_rate: Pre-head dropout rate.
    """

    def __init__(self, in_channels: int, dropout_rate: float = 0.0) -> None:
        super().__init__()
        self.drop = nn.Dropout(dropout_rate)
        self.proj1 = nn.Linear(in_channels, 16)
        self.proj2 = nn.Linear(16, 1)
        _init_keras_style(self.proj1)
        _init_keras_style(self.proj2)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return float32 per-frame logits ``(B, T, 1)``.

        Args:
            x: Hidden sequence ``(B, T, C)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        h = nn_func.gelu(self.proj1(self.drop(x)))
        return self.proj2(h).float()


class _CoreTransformerBlock(nn.Module):
    """Post-LN Transformer encoder block matching ``stepcovnet.models._transformer_encoder``.

    Args:
        d_model: Hidden width.
        num_heads: Attention head count.
        ff_dim: FFN hidden dimension.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ff_dim: int,
        dropout_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.head_dim = max(1, self.d_model // self.num_heads)
        inner = self.num_heads * self.head_dim
        self.q_proj = nn.Linear(self.d_model, inner)
        self.k_proj = nn.Linear(self.d_model, inner)
        self.v_proj = nn.Linear(self.d_model, inner)
        self.out_proj = nn.Linear(inner, self.d_model)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.ln1 = nn.LayerNorm(self.d_model, eps=1e-6)
        self.ffn1 = nn.Linear(self.d_model, ff_dim)
        self.ffn2 = nn.Linear(ff_dim, self.d_model)
        self.ffn_drop = nn.Dropout(dropout_rate)
        self.ln2 = nn.LayerNorm(self.d_model, eps=1e-6)
        for lin in (
            self.q_proj,
            self.k_proj,
            self.v_proj,
            self.out_proj,
            self.ffn1,
            self.ffn2,
        ):
            _init_keras_style(lin)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention + ReLU FFN with residual LayerNorm.

        Args:
            x: Sequence ``(B, T, d_model)``.

        Returns:
            Encoded sequence ``(B, T, d_model)``.
        """
        b, t, _ = x.shape
        q = self.q_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        attn = nn_func.scaled_dot_product_attention(q, k, v)
        attn = (
            attn.transpose(1, 2)
            .contiguous()
            .reshape(b, t, self.num_heads * self.head_dim)
        )
        out1 = self.ln1(x + self.attn_drop(self.out_proj(attn)))
        ffn = self.ffn_drop(self.ffn2(nn_func.relu(self.ffn1(out1))))
        return self.ln2(out1 + ffn)


class _UNetWaveNetStack(nn.Module):
    """Multi-scale U-Net WaveNet backbone over ``(B, T, in_features)``.

    Args:
        in_features: Input channel count.
        initial_filters: Width at level 0 (doubles each level).
        depth: Number of encoder/decoder levels.
        dilation_rates: Dilation rates per level.
        kernel_size: WaveNet convolution kernel width.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        in_features: int,
        *,
        initial_filters: int = 16,
        depth: int = 2,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.enc_projs = nn.ModuleList()
        self.enc_blocks = nn.ModuleList()
        self.enc_downs = nn.ModuleList()

        in_ch = int(in_features)
        for i in range(self.depth):
            cur_f = initial_filters * (2**i)
            next_f = initial_filters * (2 ** (i + 1))
            proj = nn.Linear(in_ch, cur_f)
            _init_keras_style(proj)
            self.enc_projs.append(proj)
            self.enc_blocks.append(
                nn.ModuleList(
                    [
                        WaveNetResidualBlock(cur_f, cur_f, rate, kernel_size)
                        for rate in dilation_rates
                    ]
                )
            )
            down = nn.Conv1d(cur_f, next_f, kernel_size=3, stride=2, padding=1)
            _init_keras_style(down)
            self.enc_downs.append(down)
            in_ch = next_f

        bn_f = initial_filters * (2**self.depth)
        self.bn_proj = nn.Linear(in_ch, bn_f)
        _init_keras_style(self.bn_proj)
        self.bn_blocks = nn.ModuleList(
            [
                WaveNetResidualBlock(bn_f, bn_f, rate, kernel_size)
                for rate in dilation_rates
            ]
        )
        self.bn_drop = nn.Dropout(dropout_rate)

        self.dec_ups = nn.ModuleList()
        self.dec_projs = nn.ModuleList()
        self.dec_drops = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        up_in = bn_f
        for i in reversed(range(self.depth)):
            cur_f = initial_filters * (2**i)
            up = nn.ConvTranspose1d(
                up_in,
                cur_f,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
            _init_keras_style(up)
            self.dec_ups.append(up)
            post_proj = nn.Linear(cur_f * 2, cur_f)
            _init_keras_style(post_proj)
            self.dec_projs.append(post_proj)
            self.dec_drops.append(nn.Dropout(dropout_rate))
            self.dec_blocks.append(
                nn.ModuleList(
                    [
                        WaveNetResidualBlock(cur_f, cur_f, rate, kernel_size)
                        for rate in dilation_rates
                    ]
                )
            )
            up_in = cur_f

        self.head = DenseOnsetOutputHead(initial_filters, dropout_rate=dropout_rate)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame float32 onset logits ``(B, T, 1)``.

        Args:
            x: Input sequence ``(B, T, in_features)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        enc_skips: list[torch.Tensor] = []
        for i in range(self.depth):
            x = self.enc_projs[i](x)
            blocks = self.enc_blocks[i]
            assert isinstance(blocks, nn.ModuleList)
            for blk in blocks:
                x, _ = blk(x)
            enc_skips.append(x)
            x = self.enc_downs[i](x.transpose(1, 2)).transpose(1, 2)

        x = self.bn_proj(x)
        for blk in self.bn_blocks:
            x, _ = blk(x)
        x = self.bn_drop(x)

        for dec_idx, skip_idx in enumerate(reversed(range(self.depth))):
            skip = enc_skips[skip_idx]
            target_len = int(skip.shape[1])
            x = self.dec_ups[dec_idx](x.transpose(1, 2)).transpose(1, 2)
            if x.shape[1] > target_len:
                x = x[:, :target_len, :]
            elif x.shape[1] < target_len:
                x = nn_func.pad(x, (0, 0, 0, target_len - int(x.shape[1])))
            x = torch.cat([x, skip], dim=-1)
            x = self.dec_drops[dec_idx](self.dec_projs[dec_idx](x))
            blocks = self.dec_blocks[dec_idx]
            assert isinstance(blocks, nn.ModuleList)
            for blk in blocks:
                x, _ = blk(x)

        return self.head.forward_logits(x)


class TcnOnsetModel(nn.Module):
    """Temporal Convolution Network for dense frame onset detection.

    Args:
        input_features: Input channel width per frame.
        initial_filters: Residual channel width.
        dilation_rates: Dilation rates per block.
        kernel_size: Causal kernel width.
        dropout_rate: Dropout probability.
        tcn_blocks: Number of stacked dilation cycles.
    """

    def __init__(
        self,
        *,
        input_features: int = constants.N_MELS,
        initial_filters: int = 16,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
        tcn_blocks: int = 4,
    ) -> None:
        super().__init__()
        self._cfg = {
            "arch": "tcn",
            "input_features": int(input_features),
            "initial_filters": int(initial_filters),
            "dilation_rates": list(dilation_rates),
            "kernel_size": int(kernel_size),
            "dropout_rate": float(dropout_rate),
            "tcn_blocks": int(tcn_blocks),
        }
        self.input_proj = nn.Linear(input_features, initial_filters)
        _init_keras_style(self.input_proj)
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        WaveNetResidualBlock(
                            initial_filters, initial_filters, rate, kernel_size
                        )
                        for rate in dilation_rates
                    ]
                )
                for _ in range(tcn_blocks)
            ]
        )
        self.block_drops = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(tcn_blocks)]
        )
        self.head = DenseOnsetOutputHead(initial_filters, dropout_rate=dropout_rate)

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Config dict.
        """
        return dict(self._cfg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute float32 per-frame onset logits ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        h = self.input_proj(x)
        for blk_group, drop in zip(self.blocks, self.block_drops, strict=True):
            assert isinstance(blk_group, nn.ModuleList)
            for blk in blk_group:
                h, _ = blk(h)
            h = drop(h)
        return self.head.forward_logits(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Sigmoid probabilities ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(x))


class BiLstmOnsetModel(nn.Module):
    """Bidirectional LSTM stack for dense frame onset detection.

    Args:
        input_features: Input channel width per frame.
        initial_filters: Input projection width.
        depth: Number of BiLSTM layers.
        dropout_rate: Dropout probability after each BiLSTM layer.
        recurrent_units: Units per direction in each LSTM layer.
    """

    def __init__(
        self,
        *,
        input_features: int = constants.N_MELS,
        initial_filters: int = 16,
        depth: int = 2,
        dropout_rate: float = 0.0,
        recurrent_units: int = 128,
    ) -> None:
        super().__init__()
        self._cfg = {
            "arch": "bilstm",
            "input_features": int(input_features),
            "initial_filters": int(initial_filters),
            "depth": int(depth),
            "dropout_rate": float(dropout_rate),
            "recurrent_units": int(recurrent_units),
        }
        self.input_proj = nn.Linear(input_features, initial_filters)
        _init_keras_style(self.input_proj)
        self.lstms = nn.ModuleList()
        self.drops = nn.ModuleList()
        in_dim = initial_filters
        for _ in range(depth):
            self.lstms.append(
                nn.LSTM(in_dim, recurrent_units, batch_first=True, bidirectional=True)
            )
            self.drops.append(nn.Dropout(dropout_rate))
            in_dim = recurrent_units * 2
        self.head = DenseOnsetOutputHead(in_dim, dropout_rate=dropout_rate)

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Config dict.
        """
        return dict(self._cfg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute float32 per-frame onset logits ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        h = self.input_proj(x)
        for lstm, drop in zip(self.lstms, self.drops, strict=True):
            h, _ = lstm(h)
            h = drop(h)
        return self.head.forward_logits(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Sigmoid probabilities ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(x))


class TransformerOnsetModel(nn.Module):
    """Transformer encoder over per-frame features for dense onset detection.

    Args:
        input_features: Input channel width per frame.
        initial_filters: Transformer ``d_model`` width.
        transformer_layers: Number of encoder blocks.
        transformer_heads: Number of attention heads.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        *,
        input_features: int = constants.N_MELS,
        initial_filters: int = 64,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._cfg = {
            "arch": "transformer",
            "input_features": int(input_features),
            "initial_filters": int(initial_filters),
            "transformer_layers": int(transformer_layers),
            "transformer_heads": int(transformer_heads),
            "dropout_rate": float(dropout_rate),
        }
        self.input_proj = nn.Linear(input_features, initial_filters)
        _init_keras_style(self.input_proj)
        ff_dim = max(initial_filters * 4, 128)
        self.layers = nn.ModuleList(
            [
                _CoreTransformerBlock(
                    initial_filters, transformer_heads, ff_dim, dropout_rate
                )
                for _ in range(transformer_layers)
            ]
        )
        self.head = DenseOnsetOutputHead(initial_filters, dropout_rate=dropout_rate)

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Config dict.
        """
        return dict(self._cfg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute float32 per-frame onset logits ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.head.forward_logits(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Sigmoid probabilities ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(x))


class UNetWaveNetOnsetModel(nn.Module):
    """U-Net style WaveNet for dense frame onset detection.

    Args:
        input_features: Input channel width per frame.
        initial_filters: Level-0 filter count.
        depth: U-Net downsampling/upsampling depth.
        dilation_rates: WaveNet dilations per level.
        kernel_size: Temporal convolution kernel width.
        dropout_rate: Dropout probability.
    """

    def __init__(
        self,
        *,
        input_features: int = constants.N_MELS,
        initial_filters: int = 16,
        depth: int = 2,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        self._cfg = {
            "arch": "unet_wavenet",
            "input_features": int(input_features),
            "initial_filters": int(initial_filters),
            "depth": int(depth),
            "dilation_rates": list(dilation_rates),
            "kernel_size": int(kernel_size),
            "dropout_rate": float(dropout_rate),
        }
        self.stack = _UNetWaveNetStack(
            input_features,
            initial_filters=initial_filters,
            depth=depth,
            dilation_rates=dilation_rates,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Config dict.
        """
        return dict(self._cfg)

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Compute float32 per-frame onset logits ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        return self.stack.forward_logits(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)``.

        Args:
            x: Frame features ``(B, T, input_features)``.

        Returns:
            Sigmoid probabilities ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(x))


class UNetWaveNetWaveformModel(nn.Module):
    """End-to-end mono waveform to dense onset probability U-Net WaveNet.

    Args:
        initial_filters: Level-0 U-Net filter count.
        depth: U-Net depth.
        dilation_rates: WaveNet dilations per level.
        kernel_size: WaveNet kernel size.
        dropout_rate: Dropout probability.
        frontend_filters: Output channels of the strided waveform frontend.
    """

    def __init__(
        self,
        *,
        initial_filters: int = 16,
        depth: int = 2,
        dilation_rates: tuple[int, ...] = (1, 2, 4, 8),
        kernel_size: int = 3,
        dropout_rate: float = 0.0,
        frontend_filters: int = 32,
    ) -> None:
        super().__init__()
        self._cfg = {
            "arch": "unet_wavenet_waveform",
            "initial_filters": int(initial_filters),
            "depth": int(depth),
            "dilation_rates": list(dilation_rates),
            "kernel_size": int(kernel_size),
            "dropout_rate": float(dropout_rate),
            "frontend_filters": int(frontend_filters),
        }
        hop = constants.WAVEFORM_SAMPLES_PER_FRAME
        self.frame_embed = nn.Conv1d(1, frontend_filters, kernel_size=hop, stride=hop)
        self.frame_refine = nn.Conv1d(
            frontend_filters, frontend_filters, kernel_size=3, padding=1
        )
        _init_keras_style(self.frame_embed)
        _init_keras_style(self.frame_refine)
        self.stack = _UNetWaveNetStack(
            frontend_filters,
            initial_filters=initial_filters,
            depth=depth,
            dilation_rates=dilation_rates,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
        )

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Config dict.
        """
        return dict(self._cfg)

    def forward_logits(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute float32 per-frame onset logits ``(B, T, 1)`` from mono waveform.

        Args:
            waveform: Mono audio ``(B, samples)``.

        Returns:
            Float32 logits ``(B, T, 1)``.
        """
        x = waveform.unsqueeze(1)
        x = nn_func.gelu(self.frame_embed(x))
        x = nn_func.gelu(self.frame_refine(x)).transpose(1, 2)
        return self.stack.forward_logits(x)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)`` from mono waveform.

        Args:
            waveform: Mono audio ``(B, samples)``.

        Returns:
            Sigmoid probabilities ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(waveform))


DenseOnsetTorchModel = (
    TcnOnsetModel
    | BiLstmOnsetModel
    | TransformerOnsetModel
    | UNetWaveNetOnsetModel
    | UNetWaveNetWaveformModel
)


def build_tcn_onset_model(
    *,
    initial_filters: int = 16,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    tcn_blocks: int = 4,
    input_features: int | None = None,
) -> TcnOnsetModel:
    """Build a PyTorch :class:`TcnOnsetModel`.

    Args:
        initial_filters: Residual channel width.
        dilation_rates: Dilation rates per block (defaults to ``[1, 2, 4, 8]``).
        kernel_size: Causal convolution kernel size.
        dropout_rate: Dropout probability.
        tcn_blocks: Number of dilation cycles.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`TcnOnsetModel`.
    """
    return TcnOnsetModel(
        input_features=constants.N_MELS if input_features is None else input_features,
        initial_filters=initial_filters,
        dilation_rates=tuple(dilation_rates or [1, 2, 4, 8]),
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        tcn_blocks=tcn_blocks,
    )


def build_bilstm_onset_model(
    *,
    initial_filters: int = 16,
    depth: int = 2,
    dropout_rate: float = 0.0,
    recurrent_units: int = 128,
    input_features: int | None = None,
) -> BiLstmOnsetModel:
    """Build a PyTorch :class:`BiLstmOnsetModel`.

    Args:
        initial_filters: Input projection width.
        depth: Number of BiLSTM layers.
        dropout_rate: Dropout probability.
        recurrent_units: Hidden units per direction.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`BiLstmOnsetModel`.
    """
    return BiLstmOnsetModel(
        input_features=constants.N_MELS if input_features is None else input_features,
        initial_filters=initial_filters,
        depth=depth,
        dropout_rate=dropout_rate,
        recurrent_units=recurrent_units,
    )


def build_transformer_onset_model(
    *,
    initial_filters: int = 64,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    dropout_rate: float = 0.0,
    input_features: int | None = None,
) -> TransformerOnsetModel:
    """Build a PyTorch :class:`TransformerOnsetModel`.

    Args:
        initial_filters: Transformer ``d_model`` width.
        transformer_layers: Number of encoder layers.
        transformer_heads: Attention head count.
        dropout_rate: Dropout probability.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`TransformerOnsetModel`.
    """
    return TransformerOnsetModel(
        input_features=constants.N_MELS if input_features is None else input_features,
        initial_filters=initial_filters,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout_rate=dropout_rate,
    )


def build_unet_wavenet_model(
    *,
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    input_features: int | None = None,
) -> UNetWaveNetOnsetModel:
    """Build a PyTorch :class:`UNetWaveNetOnsetModel`.

    Args:
        initial_filters: Level-0 filter count.
        depth: U-Net depth.
        dilation_rates: Dilations per level (defaults to ``[1, 2, 4, 8]``).
        kernel_size: Temporal kernel size.
        dropout_rate: Dropout probability.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`UNetWaveNetOnsetModel`.
    """
    return UNetWaveNetOnsetModel(
        input_features=constants.N_MELS if input_features is None else input_features,
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=tuple(dilation_rates or [1, 2, 4, 8]),
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
    )


def build_unet_wavenet_from_waveform_model(
    *,
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    frontend_filters: int = 32,
) -> UNetWaveNetWaveformModel:
    """Build a PyTorch :class:`UNetWaveNetWaveformModel`.

    Args:
        initial_filters: Level-0 U-Net filter width.
        depth: U-Net depth.
        dilation_rates: Dilations per level.
        kernel_size: WaveNet kernel size.
        dropout_rate: Dropout probability.
        frontend_filters: Waveform frontend channel width.

    Returns:
        Initialized :class:`UNetWaveNetWaveformModel`.
    """
    return UNetWaveNetWaveformModel(
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=tuple(dilation_rates or [1, 2, 4, 8]),
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        frontend_filters=frontend_filters,
    )


def build_onset_dense_model(
    model_config: config.OnsetModelConfig,
    *,
    input_features: int | None = None,
) -> DenseOnsetTorchModel:
    """Dispatch and build a PyTorch dense onset model from ``OnsetModelConfig``.

    Args:
        model_config: Core onset model config.
        input_features: Optional override for input feature width.

    Returns:
        Instantiated PyTorch dense onset model.
    """
    arch = model_config.onset_architecture
    n_features = (
        input_features if input_features is not None else model_config.input_features
    )
    if arch == config.OnsetArchitecture.TCN:
        return build_tcn_onset_model(
            initial_filters=model_config.initial_filters,
            dilation_rates=model_config.dilation_rates,
            kernel_size=model_config.kernel_size,
            dropout_rate=model_config.dropout_rate,
            tcn_blocks=model_config.tcn_blocks,
            input_features=n_features,
        )
    if arch == config.OnsetArchitecture.BILSTM:
        return build_bilstm_onset_model(
            initial_filters=model_config.initial_filters,
            depth=model_config.depth,
            dropout_rate=model_config.dropout_rate,
            recurrent_units=model_config.recurrent_units,
            input_features=n_features,
        )
    if arch == config.OnsetArchitecture.TRANSFORMER:
        return build_transformer_onset_model(
            initial_filters=model_config.initial_filters,
            transformer_layers=model_config.transformer_layers,
            transformer_heads=model_config.transformer_heads,
            dropout_rate=model_config.dropout_rate,
            input_features=n_features,
        )
    return build_unet_wavenet_model(
        initial_filters=model_config.initial_filters,
        depth=model_config.depth,
        dilation_rates=model_config.dilation_rates,
        kernel_size=model_config.kernel_size,
        dropout_rate=model_config.dropout_rate,
        input_features=n_features,
    )
