# Complete PyTorch Port (`stepcovnet.pytorch`)

Isolated, **zero-TensorFlow / zero-Keras** PyTorch port of every StepCOVNet model track, designed so TensorFlow can be cleanly swapped to PyTorch without breaking existing configs or NumPy data/eval stacks.

- **Zero TensorFlow/Keras imports** — every package passes clean-subprocess import checks without `tensorflow` or `keras` installed (supports native Windows/Linux CUDA, MPS, and CPU).
- **Zero edits to existing tracked TF/Keras code** — shares StepCOVNet's pure-NumPy configs (`configs/`), data prep, beat-grid slots, and onset timing/F1 metrics.
- **Built-in performance wins:** PyTorch SDPA (FlashAttention-2 / cuDNN SDPA), architecture-aware `bf16` (SM 8.0+) / `fp16` (SM 7.x) autocast + GradScaler, optional `torch.compile(dynamic=True)`, `ModelEMA`, true $O(L)$ per-layer KV caching for AR onset decoding, and ONNX export helpers.

## Ported packages

| Package | Mirrors | Highlights |
| --- | --- | --- |
| `stepcovnet.pytorch.runtime` | Shared GPU/AMP/EMA/LR runtime | `resolve_device`, SM 8.0+ `bf16` / Turing `fp16` `resolve_amp_dtype`, `ModelEMA`, `compute_warmup_cosine_lr` |
| `stepcovnet.pytorch.itgpt` | `stepcovnet.itgpt` (Champion ITGPT-MERT) | SDPA Transformer, logmel & multi-layer SUPERB MERT mixer, `early_film`, `cyclic_measure_pos`, `continuous_beat_conv`, Keras ↔ PyTorch numerical parity (`< 1e-5`), audio grounding ablations |
| `stepcovnet.pytorch.ddc` | `stepcovnet.ddc` (`donahue2017ddc`) | C-LSTM 10 ms frame placement model, peak picking & `M-ddc-20ms` F-score eval |
| `stepcovnet.pytorch.ddcl` | `stepcovnet.ddcl` (`omalley2025ddcl`) | Bidirectional `ConvLSTM2d` + MERT beat-slot models, focal loss, `M-slot48` eval, resume |
| `stepcovnet.pytorch.onset_ar` | `stepcovnet.onset_ar` | Encoder-decoder AR Transformer (content pointer & relative gap heads, density conditioning, STE/soft-alpha losses) with **true $O(L)$ per-layer KV cache** verified `< 1e-5` against full parallel forward |
| `stepcovnet.pytorch.core` | `stepcovnet.{models,losses,generator}` | Dense onset (`tcn`, `bilstm`, `transformer`, `unet_wavenet`, `unet_wavenet_waveform`), all 6 arrow backbones (`transformer`, `mlp`, `lstm`, `gru`, `tcn`, `cnn1d`), differentiable StepMania chart validity/diversity losses, PyTorch `generator`, ONNX export |

## Measured Colab GPU benchmarks (`ITGPT-MERT`, 256 beats, `d_model=256, 8L`)

| GPU | TensorFlow `mixed_float16` | PyTorch Eager (SDPA) | PyTorch `torch.compile` | Peak VRAM (PyTorch vs TF) |
| --- | --- | --- | --- | --- |
| **Tesla T4** | `172.85 ms/step` | **`78.68 ms/step` (2.20×, fp16)** | — | `707.8 MB` vs `1440.3 MB` |
| **NVIDIA L4** | `119.66 ms/step` | `48.43 ms/step` (2.47×, fp32) | **`28.05 ms/step` (4.27×, bf16)** | `542.0 MB` vs `1420.9 MB` |

## CLI entrypoints

```bash
# ITGPT-MERT placement (champion track)
python -m stepcovnet.pytorch.itgpt.train \
  --config configs/ddc/itgpt_mert_placement_fraxtil_orig_cosine64_l2_6_10_16_ema.json \
  --model_output_dir models_wsl/ddc/itgpt_mert_l2_6_10_16_ema_torch \
  --precision auto --compile --fresh

python -m stepcovnet.pytorch.itgpt.eval \
  --config configs/ddc/itgpt_mert_placement_fraxtil_orig_cosine64_l2_6_10_16_ema.json \
  --model_path models_wsl/ddc/itgpt_mert_l2_6_10_16_ema_torch/best_slot_f1.pt \
  --diagnose_grounding

# DDC C-LSTM placement
python -m stepcovnet.pytorch.ddc.train --config configs/ddc/placement_fraxtil.json --precision auto
python -m stepcovnet.pytorch.ddc.eval --config configs/ddc/placement_fraxtil.json --model best.pt

# DDCL ConvLSTM / MERT placement
python -m stepcovnet.pytorch.ddcl.train --config configs/ddc/ddcl_placement_fraxtil.json --precision auto
python -m stepcovnet.pytorch.ddcl.eval --config configs/ddc/ddcl_placement_fraxtil.json --model best_slot_f1.pt

# AR Onset Transformer (with O(L) KV-cached evaluation)
python -m stepcovnet.pytorch.onset_ar.train --config configs/ar/tide_overfit.json --precision auto
python -m stepcovnet.pytorch.onset_ar.eval --config configs/ar/tide_overfit.json --model best.pt
```

## Verification

```bash
ruff format --check src/stepcovnet/pytorch tests/pytorch
ruff check src/stepcovnet/pytorch tests/pytorch
pydoclint src/stepcovnet/pytorch
pyright src/stepcovnet/pytorch
pytest tests/pytorch/ -v
```
