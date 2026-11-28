"""PyTorch model architectures for dense onset detection and arrow classification."""

from __future__ import annotations

from stepcovnet import config, constants
from stepcovnet.core import arrow_models, onset_models

ArrowInputOptions = arrow_models.ArrowInputOptions
ArrowOutputOptions = arrow_models.ArrowOutputOptions
SnippetCNN = arrow_models.SnippetCNN
PositionalEncoding = arrow_models.PositionalEncoding
ArrowInputEncoder = arrow_models.ArrowInputEncoder
ArrowSequenceModel = arrow_models.ArrowSequenceModel

DenseOnsetTorchModel = onset_models.DenseOnsetTorchModel
TcnOnsetModel = onset_models.TcnOnsetModel
BiLstmOnsetModel = onset_models.BiLstmOnsetModel
TransformerOnsetModel = onset_models.TransformerOnsetModel
UNetWaveNetOnsetModel = onset_models.UNetWaveNetOnsetModel
UNetWaveNetWaveformModel = onset_models.UNetWaveNetWaveformModel
WaveNetResidualBlock = onset_models.WaveNetResidualBlock
CausalConv1d = onset_models.CausalConv1d


def _onset_model_name(model_name: str, default_name: str) -> str:
    return model_name if model_name else default_name


def build_tcn_onset_model(
    *,
    model_name: str = "",
    initial_filters: int = 16,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    tcn_blocks: int = 4,
    input_features: int | None = None,
) -> TcnOnsetModel:
    """Build a PyTorch TCN dense onset model.

    Args:
        model_name: Optional model identifier attached to ``model.name``.
        initial_filters: Residual channel width.
        dilation_rates: Dilation rates per block.
        kernel_size: Causal convolution kernel size.
        dropout_rate: Dropout probability.
        tcn_blocks: Number of dilation cycles.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`TcnOnsetModel`.
    """
    model = onset_models.build_tcn_onset_model(
        initial_filters=initial_filters,
        dilation_rates=dilation_rates,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        tcn_blocks=tcn_blocks,
        input_features=input_features,
    )
    model.name = _onset_model_name(model_name, "tcn_onset")  # type: ignore[attr-defined]
    return model


def build_bilstm_onset_model(
    *,
    model_name: str = "",
    initial_filters: int = 16,
    depth: int = 2,
    dropout_rate: float = 0.0,
    recurrent_units: int = 128,
    input_features: int | None = None,
) -> BiLstmOnsetModel:
    """Build a PyTorch BiLSTM dense onset model.

    Args:
        model_name: Optional model identifier attached to ``model.name``.
        initial_filters: Input projection width.
        depth: Number of BiLSTM layers.
        dropout_rate: Dropout probability.
        recurrent_units: Hidden units per direction.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`BiLstmOnsetModel`.
    """
    model = onset_models.build_bilstm_onset_model(
        initial_filters=initial_filters,
        depth=depth,
        dropout_rate=dropout_rate,
        recurrent_units=recurrent_units,
        input_features=input_features,
    )
    model.name = _onset_model_name(model_name, "bilstm_onset")  # type: ignore[attr-defined]
    return model


def build_transformer_onset_model(
    *,
    model_name: str = "",
    initial_filters: int = 64,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    dropout_rate: float = 0.0,
    input_features: int | None = None,
) -> TransformerOnsetModel:
    """Build a PyTorch Transformer dense onset model.

    Args:
        model_name: Optional model identifier attached to ``model.name``.
        initial_filters: Transformer ``d_model`` width.
        transformer_layers: Number of encoder layers.
        transformer_heads: Attention head count.
        dropout_rate: Dropout probability.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`TransformerOnsetModel`.
    """
    model = onset_models.build_transformer_onset_model(
        initial_filters=initial_filters,
        transformer_layers=transformer_layers,
        transformer_heads=transformer_heads,
        dropout_rate=dropout_rate,
        input_features=input_features,
    )
    model.name = _onset_model_name(model_name, "transformer_onset")  # type: ignore[attr-defined]
    return model


def build_unet_wavenet_model(
    *,
    model_name: str = "",
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    input_features: int | None = None,
) -> UNetWaveNetOnsetModel:
    """Build a PyTorch U-Net WaveNet dense onset model.

    Args:
        model_name: Optional model identifier attached to ``model.name``.
        initial_filters: Level-0 filter count.
        depth: U-Net depth.
        dilation_rates: Dilations per level.
        kernel_size: Temporal kernel size.
        dropout_rate: Dropout probability.
        input_features: Input feature width (defaults to ``N_MELS``).

    Returns:
        Initialized :class:`UNetWaveNetOnsetModel`.
    """
    model = onset_models.build_unet_wavenet_model(
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=dilation_rates,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        input_features=input_features,
    )
    model.name = _onset_model_name(model_name, "unet_wavenet_onset")  # type: ignore[attr-defined]
    return model


def build_unet_wavenet_from_waveform_model(
    *,
    model_name: str = "",
    initial_filters: int = 16,
    depth: int = 2,
    dilation_rates: list[int] | None = None,
    kernel_size: int = 3,
    dropout_rate: float = 0.0,
    frontend_filters: int = 32,
) -> UNetWaveNetWaveformModel:
    """Build a PyTorch waveform-input U-Net WaveNet dense onset model.

    Args:
        model_name: Optional model identifier attached to ``model.name``.
        initial_filters: Level-0 U-Net filter width.
        depth: U-Net depth.
        dilation_rates: Dilations per level.
        kernel_size: WaveNet kernel size.
        dropout_rate: Dropout probability.
        frontend_filters: Waveform frontend channel width.

    Returns:
        Initialized :class:`UNetWaveNetWaveformModel`.
    """
    model = onset_models.build_unet_wavenet_from_waveform_model(
        initial_filters=initial_filters,
        depth=depth,
        dilation_rates=dilation_rates,
        kernel_size=kernel_size,
        dropout_rate=dropout_rate,
        frontend_filters=frontend_filters,
    )
    model.name = _onset_model_name(model_name, "unet_wavenet_waveform_onset")  # type: ignore[attr-defined]
    return model


def build_onset_dense_model(
    model_config: config.OnsetModelConfig,
    *,
    model_name: str = "",
    input_features: int | None = None,
) -> DenseOnsetTorchModel:
    """Dispatch and build a PyTorch dense onset model from ``OnsetModelConfig``.

    Args:
        model_config: Core onset model config.
        model_name: Optional model identifier attached to ``model.name``.
        input_features: Optional override for input feature width.

    Returns:
        Instantiated PyTorch dense onset model.
    """
    n_features = (
        input_features
        if input_features is not None
        else (model_config.input_features or constants.N_MELS)
    )
    model = onset_models.build_onset_dense_model(
        model_config,
        input_features=n_features,
    )
    model.name = _onset_model_name(  # type: ignore[attr-defined]
        model_name, f"{model_config.onset_architecture.value}_onset"
    )
    return model


def build_arrow_model(
    input_options: ArrowInputOptions | None = None,
    output_options: ArrowOutputOptions | None = None,
    params: config.TransformerArrowParams | None = None,
    *,
    num_attention_layers: int | None = None,
    num_heads: int | None = None,
    key_dim: int | None = None,
    ff_dim: int | None = None,
    dropout_rate: float | None = None,
    model_name: str = "",
) -> ArrowSequenceModel:
    """Build a Transformer-based PyTorch arrow model.

    Args:
        input_options: Arrow input feature options.
        output_options: Arrow output head options.
        params: Optional Transformer hyperparameters block.
        num_attention_layers: Optional override for attention layers.
        num_heads: Optional override for attention heads.
        key_dim: Optional override for key dimension.
        ff_dim: Optional override for feed-forward dimension.
        dropout_rate: Optional override for dropout rate.
        model_name: Optional model name override.

    Returns:
        Initialized :class:`ArrowSequenceModel`.
    """
    in_opts = input_options or ArrowInputOptions()
    out_opts = output_options or ArrowOutputOptions(model_name=model_name)
    if params is None:
        params = config.TransformerArrowParams(
            num_layers=1 if num_attention_layers is None else num_attention_layers,
            d_model=64 if key_dim is None else key_dim,
            num_heads=4 if num_heads is None else num_heads,
            ff_dim=128 if ff_dim is None else ff_dim,
            dropout_rate=0.1 if dropout_rate is None else dropout_rate,
        )
    model = arrow_models.build_arrow_model(in_opts, out_opts, params)
    model.name = out_opts.model_name or model_name or "arrow_transformer"  # type: ignore[attr-defined]
    return model


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
    """
    model = arrow_models.build_arrow_model_from_config(
        model_config, input_options, output_options
    )
    model.name = output_options.model_name or f"arrow_{model_config.model_type}"  # type: ignore[attr-defined]
    return model


__all__ = [
    "ArrowInputEncoder",
    "ArrowInputOptions",
    "ArrowOutputOptions",
    "ArrowSequenceModel",
    "BiLstmOnsetModel",
    "CausalConv1d",
    "DenseOnsetTorchModel",
    "PositionalEncoding",
    "SnippetCNN",
    "TcnOnsetModel",
    "TransformerOnsetModel",
    "UNetWaveNetOnsetModel",
    "UNetWaveNetWaveformModel",
    "WaveNetResidualBlock",
    "build_arrow_model",
    "build_arrow_model_from_config",
    "build_bilstm_onset_model",
    "build_onset_dense_model",
    "build_tcn_onset_model",
    "build_transformer_onset_model",
    "build_unet_wavenet_from_waveform_model",
    "build_unet_wavenet_model",
]
