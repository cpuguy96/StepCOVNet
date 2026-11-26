"""PyTorch C-LSTM step-placement model from Dance Dance Convolution (`donahue2017ddc`)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as nn_func

from stepcovnet.ddc import constants


def _init_keras_style(module: nn.Module) -> None:
    """Apply Xavier/Glorot uniform + zero bias matching Keras defaults.

    Args:
        module: Layer to initialize if linear or conv.
    """
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class DdcPerFrameCNN(nn.Module):
    """Apply the DDC 2-layer CNN independently to each 15x80x3 frame window.

    Accepts channels-last ``(batch, time, 15, 80, 3)`` directly from
    :mod:`stepcovnet.ddc.datasets`, permutes internally for PyTorch ``Conv2d``,
    and permutes back before flattening so the 1120-d feature vector matches
    Keras ordering 1-to-1.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(
            constants.N_CHANNELS,
            constants.CNN_FILTERS[0],
            kernel_size=constants.CNN_KERNELS[0],
            padding=0,
        )
        self.pool0 = nn.MaxPool2d(
            kernel_size=constants.CNN_POOL,
            stride=constants.CNN_POOL,
        )
        self.conv1 = nn.Conv2d(
            constants.CNN_FILTERS[0],
            constants.CNN_FILTERS[1],
            kernel_size=constants.CNN_KERNELS[1],
            padding=0,
        )
        self.pool1 = nn.MaxPool2d(
            kernel_size=constants.CNN_POOL,
            stride=constants.CNN_POOL,
        )
        _init_keras_style(self.conv0)
        _init_keras_style(self.conv1)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode each 15x80x3 window.

        Args:
            audio: Audio context windows ``(batch, time, 15, 80, 3)``.

        Returns:
            Flattened CNN features ``(batch, time, 1120)``.
        """
        batch_size, seq_len = int(audio.shape[0]), int(audio.shape[1])
        frames = audio.reshape(
            batch_size * seq_len,
            constants.CONTEXT_FRAMES,
            constants.N_MELS,
            constants.N_CHANNELS,
        )
        hidden = frames.permute(0, 3, 1, 2).contiguous()
        hidden = self.pool0(nn_func.relu(self.conv0(hidden)))
        hidden = self.pool1(nn_func.relu(self.conv1(hidden)))
        # (BT, 20, 7, 8) -> (BT, 7, 8, 20) -> (B, T, 1120) matching Keras Flatten.
        hidden = hidden.permute(0, 2, 3, 1).contiguous()
        return hidden.reshape(batch_size, seq_len, -1)


class DdcClstmPlacementModel(nn.Module):
    """PyTorch DDC C-LSTM placement network.

    Args:
        lstm_units: Hidden size of each LSTM layer (paper: 200).
        lstm_layers: Number of stacked LSTM layers (paper: 2).
        dropout_rate: Dropout after each LSTM and dense layer (paper: 0.5).
        dnn_sizes: Hidden widths of the fully-connected stack (paper: 256, 128).

    Raises:
        ValueError: If ``lstm_layers`` or ``lstm_units`` is less than 1.
    """

    def __init__(
        self,
        *,
        lstm_units: int = constants.LSTM_UNITS,
        lstm_layers: int = constants.LSTM_LAYERS,
        dropout_rate: float = constants.DROPOUT_RATE,
        dnn_sizes: tuple[int, ...] = constants.DNN_SIZES,
    ) -> None:
        super().__init__()
        if lstm_layers < 1:
            raise ValueError(f"lstm_layers must be at least 1, got {lstm_layers}")
        if lstm_units < 1:
            raise ValueError(f"lstm_units must be at least 1, got {lstm_units}")
        self.lstm_units = int(lstm_units)
        self.lstm_layers = int(lstm_layers)
        self.dropout_rate = float(dropout_rate)
        self.dnn_sizes = tuple(int(w) for w in dnn_sizes)

        self.ddc_cnn = DdcPerFrameCNN()
        cnn_out_dim = 7 * 8 * constants.CNN_FILTERS[1]  # 1120
        self.rnn_proj = nn.Linear(
            cnn_out_dim + constants.N_DIFFICULTIES, self.lstm_units
        )
        _init_keras_style(self.rnn_proj)

        self.lstms = nn.ModuleList(
            [
                nn.LSTM(
                    input_size=self.lstm_units,
                    hidden_size=self.lstm_units,
                    num_layers=1,
                    batch_first=True,
                )
                for _ in range(self.lstm_layers)
            ]
        )
        self.lstm_dropouts = nn.ModuleList(
            [nn.Dropout(self.dropout_rate) for _ in range(self.lstm_layers)]
        )

        dnn_layers: list[nn.Module] = []
        in_dim = self.lstm_units
        for width in self.dnn_sizes:
            lin = nn.Linear(in_dim, width)
            _init_keras_style(lin)
            dnn_layers.extend([lin, nn.ReLU(), nn.Dropout(self.dropout_rate)])
            in_dim = width
        self.dnn = nn.Sequential(*dnn_layers)
        self.onset = nn.Linear(in_dim, 1)
        _init_keras_style(self.onset)

    def forward_logits(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute per-frame float32 onset logits.

        Args:
            inputs: Dict with ``audio`` ``(B, T, 15, 80, 3)`` and
                ``difficulty`` ``(B, T, 5)``.

        Returns:
            Float32 logits tensor of shape ``(B, T, 1)``.
        """
        conv_flat = self.ddc_cnn(inputs["audio"])
        merged = torch.cat(
            [conv_flat, inputs["difficulty"].to(conv_flat.dtype)], dim=-1
        )
        hidden = self.rnn_proj(merged)
        for lstm, drop in zip(self.lstms, self.lstm_dropouts, strict=True):
            hidden, _ = lstm(hidden)
            hidden = drop(hidden)
        hidden = self.dnn(hidden)
        return nn_func.linear(
            hidden.float(),
            self.onset.weight.float(),
            self.onset.bias.float() if self.onset.bias is not None else None,
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute per-frame onset probabilities ``(B, T, 1)``.

        Args:
            inputs: Dict with ``audio`` and ``difficulty`` tensors.

        Returns:
            Float32 probabilities tensor ``(B, T, 1)``.
        """
        return torch.sigmoid(self.forward_logits(inputs))

    def model_config(self) -> dict[str, object]:
        """Return constructor kwargs for checkpoint serialization.

        Returns:
            JSON-serializable constructor kwargs mapping.
        """
        return {
            "lstm_units": self.lstm_units,
            "lstm_layers": self.lstm_layers,
            "dropout_rate": self.dropout_rate,
            "dnn_sizes": list(self.dnn_sizes),
        }


def build_clstm_placement_model(
    *,
    lstm_units: int = constants.LSTM_UNITS,
    lstm_layers: int = constants.LSTM_LAYERS,
    dropout_rate: float = constants.DROPOUT_RATE,
    dnn_sizes: tuple[int, ...] = constants.DNN_SIZES,
) -> DdcClstmPlacementModel:
    """Build the PyTorch DDC C-LSTM placement network.

    Args:
        lstm_units: Hidden size per LSTM layer.
        lstm_layers: Number of stacked LSTM layers.
        dropout_rate: Dropout after each LSTM and dense layer.
        dnn_sizes: Fully-connected hidden widths.

    Returns:
        Instantiated :class:`DdcClstmPlacementModel`.
    """
    return DdcClstmPlacementModel(
        lstm_units=lstm_units,
        lstm_layers=lstm_layers,
        dropout_rate=dropout_rate,
        dnn_sizes=tuple(dnn_sizes),
    )
