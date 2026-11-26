"""Guardrail tests verifying zero TensorFlow/Keras imports in src/ and scripts/."""

from __future__ import annotations

import ast
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
FORBIDDEN_TOP_MODULES = frozenset({"tensorflow", "keras", "tf_keras"})


def _find_forbidden_imports(py_file: pathlib.Path) -> list[str]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                if top in FORBIDDEN_TOP_MODULES:
                    hits.append(
                        f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: import {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".", 1)[0]
            if top in FORBIDDEN_TOP_MODULES:
                hits.append(
                    f"{py_file.relative_to(REPO_ROOT)}:{node.lineno}: from {node.module} import ..."
                )
    return hits


def test_no_tensorflow_or_keras_imports_in_src_or_scripts() -> None:
    violations: list[str] = []
    for root in (REPO_ROOT / "src" / "stepcovnet", REPO_ROOT / "scripts"):
        for py_file in sorted(root.rglob("*.py")):
            violations.extend(_find_forbidden_imports(py_file))
    assert not violations, "Forbidden TF/Keras imports remain:\n" + "\n".join(
        violations
    )
