"""Skip PyTorch test collection cleanly when ``torch`` is not installed (e.g. TF-only CI)."""

from __future__ import annotations

import importlib.util

collect_ignore_glob: list[str] = (
    ["*_test.py"] if importlib.util.find_spec("torch") is None else []
)
