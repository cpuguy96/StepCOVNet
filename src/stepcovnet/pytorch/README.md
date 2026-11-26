# PyTorch Backend (`stepcovnet.pytorch`)

PyTorch implementation powering every StepCOVNet model track (`itgpt`, `ddc`, `ddcl`, `onset_ar`, and `core` dense onset / arrow models).

- **Sole deep learning backend** — all canonical packages (`stepcovnet.{models,losses,metrics,datasets,generator,trainers}`, `stepcovnet.{itgpt,ddc,ddcl,onset_ar}`) and `scripts/` run natively on PyTorch.
- **Native cross-platform acceleration** — supports CUDA, Apple MPS, and CPU out of the box.
- **Built-in performance wins:** PyTorch SDPA (FlashAttention-2 / cuDNN SDPA), architecture-aware `bf16` (SM 8.0+) / `fp16` (SM 7.x) autocast + GradScaler, optional `torch.compile(dynamic=True)`, `ModelEMA`, true $O(L)$ per-layer KV caching for AR onset decoding, and ONNX export helpers.

## Packages

| Package | Mirrors | Highlights |
| --- | --- | --- |
| `stepcovnet.pytorch.runtime` | Shared GPU/AMP/EMA/LR runtime | `resolve_device`, SM 8.0+ `bf16` / Turing `fp16` `resolve_amp_dtype`, `ModelEMA`, `compute_warmup_cosine_lr` |
| `stepcovnet.pytorch.itgpt` | `stepcovnet.itgpt` (Champion ITGPT-MERT) | SDPA Transformer, logmel & multi-layer SUPERB MERT mixer, `early_film`, `cyclic_measure_pos`, `continuous_beat_conv`, label smoothing, audio grounding ablations |
| `stepcovnet.pytorch.ddc` | `stepcovnet.ddc` (`donahue2017ddc`) | C-LSTM 10 ms frame placement model, peak picking & `M-ddc-20ms` F-score eval |
| `stepcovnet.pytorch.ddcl` | `stepcovnet.ddcl` (`omalley2025ddcl`) | Bidirectional `ConvLSTM2d` + MERT beat-slot models, focal loss, `M-slot48` eval, resume |
| `stepcovnet.pytorch.onset_ar` | `stepcovnet.onset_ar` | Encoder-decoder AR Transformer (content pointer & relative gap heads, density conditioning, STE/soft-alpha losses) with **true $O(L)$ per-layer KV cache** verified `< 1e-5` against full parallel forward |
| `stepcovnet.pytorch.core` | `stepcovnet.{models,losses,generator}` | Dense onset (`tcn`, `bilstm`, `transformer`, `unet_wavenet`, `unet_wavenet_waveform`), all 6 arrow backbones (`transformer`, `mlp`, `lstm`, `gru`, `tcn`, `cnn1d`), differentiable StepMania chart validity/diversity losses, PyTorch `generator`, ONNX export |

## Measured Colab GPU benchmarks (`ITGPT-MERT`, 256 beats, `d_model=256, 8L`)

| GPU | PyTorch Eager (SDPA) | PyTorch `torch.compile` | Peak VRAM |
| --- | --- | --- | --- |
| **Tesla T4** | **`78.68 ms/step` (fp16)** | — | `707.8 MB` |
| **NVIDIA L4** | `48.43 ms/step` (fp32) | **`28.05 ms/step` (bf16)** | `542.0 MB` |

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
