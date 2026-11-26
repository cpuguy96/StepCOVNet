"""PyTorch arrow sequence models (Transformer, MLP, LSTM, GRU, TCN, CNN1D)."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet import config, constants
from stepcovnet.pytorch.core.onset_models import (
    CausalConv1d,
    _CoreTransformerBlock,
    _init_keras_style,
)


@dataclasses.dataclass
class ArrowInputOptions:
    """Options controlling auxiliary inputs for PyTorch arrow models.

    Attributes:
        snippet_half_frames: Half-window of mel frames per step (0 disables snippets).
        use_interval: Include inter-onset interval feature(s).
        interval_encoding: Interval encoding variant (``DEFAULT``, ``LOG``, ``MULTI``).
        use_step_index: Include normalized step index feature.
        use_beat_phase: Include beat phase feature.
    """

    snippet_half_frames: int = 0
    use_interval: bool = False
    interval_encoding: config.IntervalEncoding = config.IntervalEncoding.DEFAULT
    use_step_index: bool = False
    use_beat_phase: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Serialize options to a plain dictionary.

        Returns:
            JSON-compatible dictionary.
        """
        return {
            "snippet_half_frames": int(self.snippet_half_frames),
            "use_interval": bool(self.use_interval),
            "interval_encoding": self.interval_encoding.value,
            "use_step_index": bool(self.use_step_index),
            "use_beat_phase": bool(self.use_beat_phase),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArrowInputOptions:
        """Deserialize options from a dictionary.

        Args:
            data: Input dictionary.

        Returns:
            Initialized :class:`ArrowInputOptions`.
        """
        enc = data.get("interval_encoding", config.IntervalEncoding.DEFAULT.value)
        if not isinstance(enc, config.IntervalEncoding):
            enc = config.IntervalEncoding(str(enc))
        return cls(
            snippet_half_frames=int(data.get("snippet_half_frames", 0)),
            use_interval=bool(data.get("use_interval", False)),
            interval_encoding=enc,
            use_step_index=bool(data.get("use_step_index", False)),
            use_beat_phase=bool(data.get("use_beat_phase", False)),
        )


@dataclasses.dataclass
class ArrowOutputOptions:
    """Output options for PyTorch arrow models.

    Attributes:
        use_aux_interval: Include auxiliary next-interval regression head.
        model_name: Optional descriptive model name.
    """

    use_aux_interval: bool = False
    model_name: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Serialize options to a plain dictionary.

        Returns:
            JSON-compatible dictionary.
        """
        return {
            "use_aux_interval": bool(self.use_aux_interval),
            "model_name": str(self.model_name),
        }


class SnippetCNN(nn.Module):
    """2D CNN over each ``(n_frames, n_mels)`` audio snippet per onset step.

    Args:
        n_frames: Number of frames per snippet.
        n_mels: Number of mel bins per snippet.
        filters: Number of Conv2D output channels.
    """

    def __init__(self, n_frames: int, n_mels: int, filters: int = 32) -> None:
        super().__init__()
        self.n_frames = int(n_frames)
        self.n_mels = int(n_mels)
        self.filters = int(filters)
        self.conv1 = nn.Conv2d(1, self.filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=3, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.zeros_(self.conv1.bias)  # type: ignore[arg-type]
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)  # type: ignore[arg-type]

    def forward(self, snippets: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, S, n_frames, n_mels)`` into ``(B, S, filters)``.

        Args:
            snippets: Per-step mel snippets ``(B, S, n_frames, n_mels)``.

        Returns:
            Pooled snippet embeddings ``(B, S, filters)``.
        """
        b, s = int(snippets.shape[0]), int(snippets.shape[1])
        x = snippets.reshape(b * s, 1, self.n_frames, self.n_mels)
        x = nn_func.relu(self.conv1(x))
        x = nn_func.relu(self.conv2(x))
        x = x.mean(dim=(2, 3))
        return x.reshape(b, s, self.filters)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding matching ``stepcovnet.models.PositionalEncoding``.

    Args:
        position: Maximum supported sequence length.
        d_model: Even embedding dimension.

    Attributes:
        pos_encoding: Precomputed ``(1, position, d_model)`` float32 buffer.

    Raises:
        ValueError: If ``d_model`` is odd.
    """

    pos_encoding: torch.Tensor

    def __init__(self, position: int, d_model: int) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError("PositionalEncoding requires an even d_model.")
        self.position = int(position)
        self.d_model = int(d_model)
        pos = np.arange(self.position, dtype=np.float32)[:, np.newaxis]
        idx = np.arange(self.d_model, dtype=np.float32)[np.newaxis, :]
        angles = pos / np.power(10000.0, (2.0 * (idx // 2.0)) / float(self.d_model))
        table = np.zeros((self.position, self.d_model), dtype=np.float32)
        table[:, 0::2] = np.sin(angles[:, 0::2])
        table[:, 1::2] = np.cos(angles[:, 1::2])
        self.register_buffer(
            "pos_encoding", torch.from_numpy(table).unsqueeze(0), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sinusoidal positional encodings to ``(B, S, d_model)``.

        Args:
            x: Input sequence ``(B, S, d_model)``.

        Returns:
            Position-encoded tensor ``(B, S, d_model)``.
        """
        seq_len = int(x.shape[1])
        return x + self.pos_encoding[:, :seq_len, :].to(x.dtype)


class ArrowInputEncoder(nn.Module):
    """Fuses timing, optional intervals, step index, beat phase, and mel snippets.

    Args:
        embed_dim: Output embedding width.
        input_options: Active input feature flags.
        scale_timing: Multiply timing embedding by ``sqrt(embed_dim)``.

    Raises:
        ValueError: If ``interval_encoding`` is unrecognized.
    """

    def __init__(
        self,
        embed_dim: int,
        input_options: ArrowInputOptions,
        *,
        scale_timing: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.input_options = input_options
        self.scale_timing = bool(scale_timing)

        self.timing_proj = nn.Linear(1, self.embed_dim)
        _init_keras_style(self.timing_proj)

        self.interval_proj: nn.Linear | None = None
        self.interval_log_proj: nn.Linear | None = None
        self.interval_next_proj: nn.Linear | None = None
        if input_options.use_interval:
            enc = input_options.interval_encoding
            if enc == config.IntervalEncoding.DEFAULT:
                self.interval_proj = nn.Linear(1, self.embed_dim)
                _init_keras_style(self.interval_proj)
            elif enc == config.IntervalEncoding.LOG:
                self.interval_log_proj = nn.Linear(1, self.embed_dim)
                _init_keras_style(self.interval_log_proj)
            elif enc == config.IntervalEncoding.MULTI:
                self.interval_log_proj = nn.Linear(1, self.embed_dim)
                self.interval_next_proj = nn.Linear(1, self.embed_dim)
                _init_keras_style(self.interval_log_proj)
                _init_keras_style(self.interval_next_proj)
            else:
                raise ValueError(f"Invalid interval encoding: {enc}")

        self.step_index_proj: nn.Linear | None = None
        if input_options.use_step_index:
            self.step_index_proj = nn.Linear(1, self.embed_dim)
            _init_keras_style(self.step_index_proj)

        self.beat_phase_proj: nn.Linear | None = None
        if input_options.use_beat_phase:
            self.beat_phase_proj = nn.Linear(1, self.embed_dim)
            _init_keras_style(self.beat_phase_proj)

        self.snippet_cnn: SnippetCNN | None = None
        self.snippet_proj: nn.Linear | None = None
        if input_options.snippet_half_frames > 0:
            n_frames = 2 * int(input_options.snippet_half_frames) + 1
            self.snippet_cnn = SnippetCNN(n_frames, constants.N_MELS, filters=32)
            self.snippet_proj = nn.Linear(32, self.embed_dim)
            _init_keras_style(self.snippet_proj)

    def required_input_names(self) -> list[str]:
        """Return ordered list of expected input dictionary keys.

        Returns:
            Input key names matching Keras arrow model inputs.
        """
        names = ["timing_input"]
        if self.input_options.use_interval:
            enc = self.input_options.interval_encoding
            if enc == config.IntervalEncoding.DEFAULT:
                names.append("interval_input")
            elif enc == config.IntervalEncoding.LOG:
                names.append("interval_log_input")
            elif enc == config.IntervalEncoding.MULTI:
                names.extend(["interval_log_input", "interval_next_input"])
        if self.input_options.use_step_index:
            names.append("step_index_input")
        if self.input_options.use_beat_phase:
            names.append("beat_phase_input")
        if self.input_options.snippet_half_frames > 0:
            names.append("snippet_input")
        return names

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode arrow inputs into ``(fused_embedding, timing_tensor)``.

        Args:
            inputs: Single ``timing_input`` tensor ``(B, S, 1)`` or input dict.

        Returns:
            Tuple ``(x, timing_tensor)`` with ``x`` shaped ``(B, S, embed_dim)``.
        """
        if isinstance(inputs, torch.Tensor):
            feed: dict[str, torch.Tensor] = {"timing_input": inputs}
        else:
            feed = inputs

        timing = feed["timing_input"]
        x = self.timing_proj(timing)
        if self.scale_timing:
            x = x * math.sqrt(float(self.embed_dim))
        if self.interval_proj is not None:
            x = x + self.interval_proj(feed["interval_input"])
        if self.interval_log_proj is not None:
            x = x + self.interval_log_proj(feed["interval_log_input"])
        if self.interval_next_proj is not None:
            x = x + self.interval_next_proj(feed["interval_next_input"])
        if self.step_index_proj is not None:
            x = x + self.step_index_proj(feed["step_index_input"])
        if self.beat_phase_proj is not None:
            x = x + self.beat_phase_proj(feed["beat_phase_input"])
        if self.snippet_cnn is not None and self.snippet_proj is not None:
            x = x + self.snippet_proj(self.snippet_cnn(feed["snippet_input"]))
        return x, timing


class ArrowSequenceModel(nn.Module):
    """Unified PyTorch arrow sequence model supporting all 6 StepCOVNet backbones.

    Args:
        model_type: Backbone type (``transformer``, ``mlp``, ``lstm``, ``gru``, ``tcn``, ``cnn1d``).
        input_options: Input feature flags.
        output_options: Output head configuration.
        params: Backbone-specific hyperparameter dataclass.

    Raises:
        ValueError: If ``model_type`` is unsupported.
    """

    def __init__(
        self,
        *,
        model_type: str,
        input_options: ArrowInputOptions,
        output_options: ArrowOutputOptions,
        params: Any,
    ) -> None:
        super().__init__()
        self.model_type = str(model_type)
        self.input_options = input_options
        self.output_options = output_options
        self._params_dict = dataclasses.asdict(params)

        self.timing_pos_bias: nn.Linear | None = None
        self.pos_enc: PositionalEncoding | None = None
        self.in_drop = nn.Dropout(getattr(params, "dropout_rate", 0.1))

        # Backbone-specific attributes initialized below.
        self.transformer_layers: nn.ModuleList | None = None
        self.mlp_layers: nn.ModuleList | None = None
        self.mlp_drops: nn.ModuleList | None = None
        self.rnn_layers: nn.ModuleList | None = None
        self.gru_attn_proj: nn.Linear | None = None
        self.gru_attn: _CoreTransformerBlock | None = None
        self.tcn_convs: nn.ModuleList | None = None
        self.tcn_drops: nn.ModuleList | None = None
        self.cnn_convs: nn.ModuleList | None = None
        self.head_dense: nn.Linear | None = None
        self.head_drop: nn.Dropout | None = None

        if self.model_type == "transformer":
            d_model = int(params.d_model)
            self.encoder = ArrowInputEncoder(d_model, input_options, scale_timing=True)
            if params.use_timing_position:
                self.timing_pos_bias = nn.Linear(1, d_model)
                _init_keras_style(self.timing_pos_bias)
            else:
                self.pos_enc = PositionalEncoding(constants.MAX_STEPS, d_model)
            self.transformer_layers = nn.ModuleList(
                [
                    _CoreTransformerBlock(
                        d_model, params.num_heads, params.ff_dim, params.dropout_rate
                    )
                    for _ in range(params.num_layers)
                ]
            )
            out_dim = d_model
        elif self.model_type == "mlp":
            hidden_dims = list(params.hidden_dims or [256, 128])
            self.encoder = ArrowInputEncoder(hidden_dims[0], input_options)
            self.mlp_layers = nn.ModuleList()
            self.mlp_drops = nn.ModuleList()
            cur = hidden_dims[0]
            for dim in hidden_dims[1:]:
                lin = nn.Linear(cur, dim)
                _init_keras_style(lin)
                self.mlp_layers.append(lin)
                self.mlp_drops.append(nn.Dropout(params.dropout_rate))
                cur = dim
            out_dim = cur
        elif self.model_type == "lstm":
            units = int(params.units)
            self.encoder = ArrowInputEncoder(units, input_options)
            self.rnn_layers = nn.ModuleList()
            cur = units
            for _ in range(params.num_layers):
                self.rnn_layers.append(
                    nn.LSTM(
                        cur,
                        units,
                        batch_first=True,
                        bidirectional=bool(params.bidirectional),
                    )
                )
                cur = units * (2 if params.bidirectional else 1)
            head_w = max(units // 2, constants.N_ARROW_TYPES)
            self.head_dense = nn.Linear(cur, head_w)
            _init_keras_style(self.head_dense)
            self.head_drop = nn.Dropout(params.dropout_rate)
            out_dim = head_w
        elif self.model_type == "gru":
            units = int(params.units)
            self.encoder = ArrowInputEncoder(units, input_options)
            self.rnn_layers = nn.ModuleList()
            cur = units
            for _ in range(params.num_layers):
                self.rnn_layers.append(
                    nn.GRU(
                        cur,
                        units,
                        batch_first=True,
                        bidirectional=bool(params.bidirectional),
                    )
                )
                cur = units * (2 if params.bidirectional else 1)
            if params.add_attention_layer:
                attn_dim = int(params.attention_dim)
                if cur != attn_dim:
                    self.gru_attn_proj = nn.Linear(cur, attn_dim)
                    _init_keras_style(self.gru_attn_proj)
                self.gru_attn = _CoreTransformerBlock(
                    attn_dim,
                    int(params.attention_heads),
                    max(attn_dim * 2, 64),
                    params.dropout_rate,
                )
                cur = attn_dim
                head_w = max(attn_dim // 2, constants.N_ARROW_TYPES)
            else:
                head_w = max(units // 2, constants.N_ARROW_TYPES)
            self.head_dense = nn.Linear(cur, head_w)
            _init_keras_style(self.head_dense)
            self.head_drop = nn.Dropout(params.dropout_rate)
            out_dim = head_w
        elif self.model_type == "tcn":
            filters = int(params.filters)
            self.encoder = ArrowInputEncoder(filters, input_options)
            self.tcn_convs = nn.ModuleList(
                [
                    CausalConv1d(
                        filters,
                        filters,
                        int(params.kernel_size),
                        dilation=int(params.dilation_base) ** i,
                    )
                    for i in range(int(params.num_layers))
                ]
            )
            self.tcn_drops = nn.ModuleList(
                [nn.Dropout(params.dropout_rate) for _ in range(int(params.num_layers))]
            )
            head_w = max(filters // 2, constants.N_ARROW_TYPES)
            self.head_dense = nn.Linear(filters, head_w)
            _init_keras_style(self.head_dense)
            self.head_drop = nn.Dropout(params.dropout_rate)
            out_dim = head_w
        elif self.model_type == "cnn1d":
            filters = int(params.filters)
            kernel_sizes = list(params.kernel_sizes or [3, 3, 3])
            self.encoder = ArrowInputEncoder(filters, input_options)
            self.cnn_convs = nn.ModuleList(
                [CausalConv1d(filters, filters, int(k)) for k in kernel_sizes]
            )
            head_w = max(filters // 2, constants.N_ARROW_TYPES)
            self.head_dense = nn.Linear(filters, head_w)
            _init_keras_style(self.head_dense)
            self.head_drop = nn.Dropout(params.dropout_rate)
            out_dim = head_w
        else:
            raise ValueError(f"Unsupported arrow model_type {model_type!r}")

        self.arrow_head = nn.Linear(out_dim, constants.N_ARROW_TYPES)
        _init_keras_style(self.arrow_head)
        self.aux_interval_head: nn.Linear | None = None
        if output_options.use_aux_interval:
            self.aux_interval_head = nn.Linear(out_dim, 1)
            _init_keras_style(self.aux_interval_head)

    def model_config(self) -> dict[str, Any]:
        """Return checkpoint serialization metadata.

        Returns:
            Dictionary describing model type, options, and backbone params.
        """
        return {
            "model_type": self.model_type,
            "input_options": self.input_options.as_dict(),
            "output_options": self.output_options.as_dict(),
            "params": dict(self._params_dict),
        }

    def required_input_names(self) -> list[str]:
        """Return expected input key names for this model.

        Returns:
            Ordered list of input names.
        """
        return self.encoder.required_input_names()

    def _encode_hidden(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor:
        x, timing = self.encoder(inputs)
        if self.model_type == "transformer":
            if self.timing_pos_bias is not None:
                x = x + self.timing_pos_bias(timing)
            elif self.pos_enc is not None:
                x = self.pos_enc(x)
            x = self.in_drop(x)
            assert self.transformer_layers is not None
            for layer in self.transformer_layers:
                x = layer(x)
        elif self.model_type == "mlp":
            assert self.mlp_layers is not None and self.mlp_drops is not None
            for lin, drop in zip(self.mlp_layers, self.mlp_drops, strict=True):
                x = drop(nn_func.relu(lin(x)))
        elif self.model_type in ("lstm", "gru"):
            assert self.rnn_layers is not None
            for rnn in self.rnn_layers:
                x, _ = rnn(x)
                x = self.in_drop(x)
            if self.gru_attn_proj is not None:
                x = self.gru_attn_proj(x)
            if self.gru_attn is not None:
                x = self.gru_attn(x)
        elif self.model_type == "tcn":
            assert self.tcn_convs is not None and self.tcn_drops is not None
            for conv, drop in zip(self.tcn_convs, self.tcn_drops, strict=True):
                x = drop(nn_func.relu(conv(x.transpose(1, 2)).transpose(1, 2)))
        elif self.model_type == "cnn1d":
            assert self.cnn_convs is not None
            for conv in self.cnn_convs:
                x = nn_func.relu(conv(x.transpose(1, 2)).transpose(1, 2))
            x = self.in_drop(x)

        if self.head_dense is not None and self.head_drop is not None:
            x = self.head_drop(nn_func.relu(self.head_dense(x)))
        return x

    def forward_logits(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return float32 arrow logits (and optional ``aux_interval``).

        Args:
            inputs: Single timing tensor or dict of arrow feature tensors.

        Returns:
            Logits tensor ``(B, S, 256)`` or dict with ``arrow_logits`` and ``aux_interval``.
        """
        h = self._encode_hidden(inputs)
        logits = self.arrow_head(h).float()
        if self.aux_interval_head is not None:
            return {
                "arrow_logits": logits,
                "aux_interval": self.aux_interval_head(h).float(),
            }
        return logits

    def forward(
        self,
        inputs: torch.Tensor | dict[str, torch.Tensor],
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return softmax arrow probabilities (and optional ``aux_interval``).

        Args:
            inputs: Single timing tensor or dict of arrow feature tensors.

        Returns:
            Probability tensor ``(B, S, 256)`` or dict with ``output_probabilities``
            and ``aux_interval``.
        """
        out = self.forward_logits(inputs)
        if isinstance(out, dict):
            return {
                "output_probabilities": torch.softmax(out["arrow_logits"], dim=-1),
                "aux_interval": out["aux_interval"],
            }
        return torch.softmax(out, dim=-1)


def build_arrow_model(
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
    params: config.TransformerArrowParams,
) -> ArrowSequenceModel:
    """Build a Transformer-based PyTorch arrow model.

    Args:
        input_options: Arrow input feature options.
        output_options: Arrow output head options.
        params: Transformer hyperparameters.

    Returns:
        Initialized :class:`ArrowSequenceModel`.
    """
    return ArrowSequenceModel(
        model_type="transformer",
        input_options=input_options,
        output_options=output_options,
        params=params,
    )


def build_arrow_model_from_config(
    model_config: config.ArrowModelConfig,
    input_options: ArrowInputOptions,
    output_options: ArrowOutputOptions,
) -> ArrowSequenceModel:
    """Build any configured PyTorch arrow backbone from ``ArrowModelConfig``.

    Args:
        model_config: Core arrow model configuration.
        input_options: Arrow input feature options.
        output_options: Arrow output head options.

    Returns:
        Initialized :class:`ArrowSequenceModel`.

    Raises:
        ValueError: If ``model_type`` is unsupported or missing params.
    """
    params = model_config.get_active_params_block()
    if params is None:
        raise ValueError(
            f"Arrow model_type {model_config.model_type!r} has no params block set"
        )
    return ArrowSequenceModel(
        model_type=model_config.model_type,
        input_options=input_options,
        output_options=output_options,
        params=params,
    )
