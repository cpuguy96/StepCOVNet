"""Tests for shared PyTorch runtime helpers."""

from __future__ import annotations

import math
import unittest
from unittest import mock

import torch
from torch import nn

from stepcovnet import runtime


class RuntimeTest(unittest.TestCase):
    def test_resolve_device_explicit_cpu(self):
        self.assertEqual(runtime.resolve_device("cpu"), torch.device("cpu"))
        self.assertEqual(
            runtime.resolve_device(torch.device("cpu")),
            torch.device("cpu"),
        )

    def test_set_seed_is_reproducible(self):
        runtime.set_seed(123)
        a = torch.randn(4, 4)
        runtime.set_seed(123)
        b = torch.randn(4, 4)
        self.assertTrue(torch.equal(a, b))

    def test_resolve_amp_dtype_validation(self):
        cpu = torch.device("cpu")
        self.assertIsNone(runtime.resolve_amp_dtype("fp32", cpu))
        self.assertIsNone(runtime.resolve_amp_dtype("fp16", cpu))
        self.assertIsNone(runtime.resolve_amp_dtype("auto", cpu))
        self.assertEqual(runtime.resolve_amp_dtype("bf16", cpu), torch.bfloat16)
        with self.assertRaises(ValueError):
            runtime.resolve_amp_dtype("fp8", cpu)

    def test_configure_cuda_performance_tf32(self):
        self.assertFalse(runtime.configure_cuda_performance(torch.device("cpu")))
        orig_matmul = torch.backends.cuda.matmul.allow_tf32
        orig_cudnn = torch.backends.cudnn.allow_tf32
        try:
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.get_device_capability", return_value=(7, 5)),
            ):
                self.assertFalse(
                    runtime.configure_cuda_performance(torch.device("cuda"))
                )
            with (
                mock.patch("torch.cuda.is_available", return_value=True),
                mock.patch("torch.cuda.get_device_capability", return_value=(8, 9)),
            ):
                self.assertTrue(
                    runtime.configure_cuda_performance(torch.device("cuda"))
                )
                self.assertTrue(torch.backends.cuda.matmul.allow_tf32)
                self.assertTrue(torch.backends.cudnn.allow_tf32)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = orig_matmul
            torch.backends.cudnn.allow_tf32 = orig_cudnn

    def test_model_ema_update_swap_restore_and_state_dict(self):
        model = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        ema = runtime.ModelEMA(model, momentum=0.9)
        with torch.no_grad():
            model.weight.fill_(2.0)
        ema.update(model)
        # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        with ema.average_parameters(model):
            self.assertTrue(
                torch.allclose(model.weight, torch.full_like(model.weight, 1.1))
            )
        # Restored back to live weights (2.0).
        self.assertTrue(
            torch.allclose(model.weight, torch.full_like(model.weight, 2.0))
        )

        # Round-trip state_dict
        state = ema.state_dict()
        ema2 = runtime.ModelEMA(model, momentum=0.5)
        ema2.load_state_dict(state)
        self.assertAlmostEqual(ema2.momentum, 0.9)
        self.assertTrue(torch.allclose(ema2.shadow["weight"], ema.shadow["weight"]))

    def test_model_ema_rejects_invalid_momentum(self):
        model = nn.Linear(2, 2)
        with self.assertRaises(ValueError):
            runtime.ModelEMA(model, momentum=0.0)
        with self.assertRaises(ValueError):
            runtime.ModelEMA(model, momentum=1.0)

    def test_compute_warmup_cosine_lr(self):
        lr0 = runtime.compute_warmup_cosine_lr(
            0, peak_lr=1e-4, min_lr=1e-6, warmup_steps=4, total_steps=16
        )
        lr2 = runtime.compute_warmup_cosine_lr(
            2, peak_lr=1e-4, min_lr=1e-6, warmup_steps=4, total_steps=16
        )
        lr4 = runtime.compute_warmup_cosine_lr(
            4, peak_lr=1e-4, min_lr=1e-6, warmup_steps=4, total_steps=16
        )
        lr10 = runtime.compute_warmup_cosine_lr(
            10, peak_lr=1e-4, min_lr=1e-6, warmup_steps=4, total_steps=16
        )
        lr16 = runtime.compute_warmup_cosine_lr(
            16, peak_lr=1e-4, min_lr=1e-6, warmup_steps=4, total_steps=16
        )
        self.assertAlmostEqual(lr0, 1e-6)
        self.assertAlmostEqual(lr2, 0.5 * (1e-6 + 1e-4))
        self.assertAlmostEqual(lr4, 1e-4)
        self.assertAlmostEqual(
            lr10, 1e-6 + (1e-4 - 1e-6) * 0.5 * (1.0 + math.cos(math.pi * 0.5))
        )
        self.assertAlmostEqual(lr16, 1e-6)

    def test_supports_fused_adam(self):
        self.assertFalse(runtime.supports_fused_adam([]))
        cpu_param = nn.Parameter(torch.zeros(2, 2, dtype=torch.float32))
        self.assertFalse(runtime.supports_fused_adam([cpu_param]))
        int_param = nn.Parameter(
            torch.zeros(2, 2, dtype=torch.int64), requires_grad=False
        )
        self.assertFalse(runtime.supports_fused_adam([int_param]))
        if torch.cuda.is_available():
            cuda_param = nn.Parameter(
                torch.zeros(2, 2, dtype=torch.float32, device="cuda")
            )
            self.assertTrue(runtime.supports_fused_adam([cuda_param]))
