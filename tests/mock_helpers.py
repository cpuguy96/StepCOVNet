"""Shared test stand-ins — real minimal objects / create_autospec before MagicMock."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest import mock

import keras


class KerasHistoryStub:
    """Keras ``History``-like object returned from ``model.fit``."""

    def __init__(self, history: dict[str, list[float]]) -> None:
        self.history = history


class GenerateOutputDataStub:
    def __init__(self, text: str) -> None:
        self._text = text

    def generate_txt_output(self) -> str:
        return self._text


def completed_process(returncode: int) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(
        args=[],
        returncode=returncode,
        stdout=b"",
        stderr=b"",
    )


def keras_model_stub(
    *,
    predict_return_value: Any = None,
    predict_side_effect: Callable[..., Any] | None = None,
    inputs: list[Any] | None = None,
    fit_history: dict[str, list[float]] | None = None,
) -> mock.NonCallableMagicMock:
    """``create_autospec(keras.Model)`` with common train/predict hooks."""
    model = mock.create_autospec(keras.Model, instance=True)
    if predict_side_effect is not None:
        model.predict.side_effect = predict_side_effect
    elif predict_return_value is not None:
        model.predict.return_value = predict_return_value
    if inputs is not None:
        model.inputs = inputs
    history = fit_history or {"val_loss": [1.0], "loss": [1.0]}
    model.fit.return_value = KerasHistoryStub(history)
    return model


def keras_history_stub(history: dict[str, list[float]]) -> KerasHistoryStub:
    return KerasHistoryStub(history)


def tk_event(**attrs: Any) -> SimpleNamespace:
    return SimpleNamespace(**attrs)


def keras_input_tensor(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=f"{name}:0")


def win32_kernel32_stub(*, last_error: int = 0, mutex_handle: int = 12345) -> mock.Mock:
    kernel = mock.Mock(spec=["CreateMutexW", "GetLastError", "CloseHandle"])
    kernel.CreateMutexW.return_value = mutex_handle
    kernel.GetLastError.return_value = last_error
    return kernel


def win32_user32_stub() -> mock.Mock:
    return mock.Mock(spec=["MessageBoxW"])


def opaque_handle() -> mock.Mock:
    """Untyped stand-in for opaque third-party objects (torch MERT, etc.)."""
    return mock.Mock(spec=[])


def mert_model_and_processor() -> tuple[mock.Mock, mock.Mock]:
    return opaque_handle(), opaque_handle()
