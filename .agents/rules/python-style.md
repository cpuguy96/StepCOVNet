---
description: Python code style — imports, third-party submodules, and docstrings
globs: "**/*.py"
trigger: glob
---

# Python style

Applies to all Python in `src/`, `scripts/`, and `tests/`. Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) for formatting and docstrings unless this file or `pyproject.toml` says otherwise.

**Lint:** After editing Python, run Ruff from repo root using the active project venv (see [python-env.md](python-env.md)) at the same CI scope (`ruff check .`):

```text
python -m ruff check .
python -m ruff format src scripts tests
python -m pydoclint src/stepcovnet/<package>/
python -m pyright src/stepcovnet/<package>/
```

Before commit/push, run `python pre_submit.py --fast` (or `--skip-install` for the full CI mirror). Config: `[tool.ruff]`, `[tool.pydoclint]`, and `[tool.pyright]` in `pyproject.toml`.

---

## Google Style Guide mapping

Follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) unless this file or `pyproject.toml` says otherwise.

| Google topic                        | Enforced by                          | Notes / deviations                                                                                                             |
| ----------------------------------- | ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------ |
| Line length (80 cols)               | Ruff format (`line-length = 88`)     | **88 columns** — Black-compatible; do not wrap to 80 in new code.                                                              |
| Imports                             | Ruff `I` + `PLC0415` + rules below   | **Module-level only** in `src/`; see exceptions table in Imports section.                                                      |
| Naming (`N`)                        | Ruff `N`                             | **`N806` ignored** for ML locals (`X`, `y`, …).                                                                                |
| Pathlib (`PTH`)                     | Ruff `PTH`                           | **Full repo:** `ruff check .` at repo root (same scope as CI). Fix violations in touched files; do not commit with a red tree. |
| Simplify (`SIM`)                    | Ruff `SIM`                           | Prefer combined `with` contexts where Ruff suggests.                                                                           |
| Docstrings (§3.8)                   | pydoclint + rules below              | Google-style **Args / Returns / Raises** on **public and `_private` functions**; see docstring section.                        |
| Type hints (§2.22)                  | pyright (`typeCheckingMode = basic`) | Run on packages you change; fix missing imports and obvious type errors.                                                       |
| `main()` guard, exceptions, strings | Agent review                         | No dedicated linter — match surrounding module style.                                                                          |

---

## Imports

Use **module-level imports**. Do not import individual classes or functions from modules.

Place **all imports at the top of the file** (after the module docstring, if any). **Do not add imports inside functions, methods, or other blocks** unless one of the exceptions below applies.

### Exceptions (late imports allowed)

| Case | Pattern | Notes |
| ---- | ------- | ----- |
| **Import order** | `sys.path` / env setup before a script import | Use `# noqa: E402` on that import line only. |
| **Optional extras** | `transformers` (in `[ssl]`), not in core dependencies | Import inside a dedicated `_require_*()` helper; raise `ImportError` with install hint. |
| **Scripts / tests** | `scripts/*.py`, `tests/**/*.py` | `E402` and `PLC0415` ignored in `pyproject.toml` for bootstrap and mock-before-import patterns. |

Refactor to break **circular imports** (extract a small shared module) instead of adding function-level imports when possible. Ruff enforces this in `src/` via rule **`PLC0415`**.

### Rules

- **Use:** `import some_module` then `some_module.ClassName`, `some_module.function_name`, etc.
- **Use (stdlib):** `import io` then `io.StringIO()` — not `from io import StringIO`.
- **Use (submodules):** `from parent import submodule` when the imported name is a **submodule** (another module), not a class or function. Then use `submodule.thing`.
- **Avoid:** `import parent.submodule` (dotted import of a submodule). Use `from parent import submodule` instead.
- **Avoid:** `from some_module import ClassName`, `from some_module import function_name`, or `from parent.submodule import ClassName`.

**Exception (pathlib):** `import pathlib as pl` is allowed when the module is used once or the name is long.

### Examples

```python
# ✅ GOOD – top-level and submodule imports
import numpy as np
import torch
from stepcovnet import models
from tkinter import filedialog, messagebox

model = models.build_model(config)
path = filedialog.askopenfilename(title="Select file")
```

```python
# ✅ GOOD – simfile (third-party with subpackages)
import simfile
from simfile import notes
from simfile.notes import timed
from simfile import timing

note_data = notes.NoteData(chart)
rows = list(timed.time_notes(note_data, timing.TimingData(sim, chart)))
```

```python
# ❌ BAD – dotted submodule import or importing symbols
import simfile.notes
from simfile.notes import NoteType, NoteData
from simfile.notes.timed import time_notes
from numpy import array
from stepcovnet.models import build_model
```

### Naming

Do not use a local variable name that shadows an imported submodule (e.g. if you `from simfile import notes`, do not name a parameter `notes`).

### Import order and formatting

**Ruff** enforces import order and grouping via `[tool.ruff.lint]` / `[tool.ruff.lint.isort]` in `pyproject.toml` (rule `I001`). Do not hand-sort imports — run the linter after edits (see above).

---

## Docstrings

Use [Google-style](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods) sections. Match the guide’s formatting: summary line (≤80 chars when practical), blank line before sections, section headers with trailing colon, hanging indent of 2 or 4 spaces (be consistent within a file).

Use one imperative or descriptive style per file (e.g. `"""Load …"""` or `"""Loads …"""`) — do not mix.

Reference names in docstrings with double backticks: `name_map.json`.

### Classes (non-test code)

For every **class** defined outside `tests/`:

1. **Summary** — one line describing what an instance represents.
2. **Attributes** — public attributes (one line each). Dataclasses/config classes: document every field.

**Exclusion:** Test classes (`unittest.TestCase` subclasses).

### Functions and methods

Required for every **function and method** at module or class scope, including **`_private` helpers**, except `main()` under `if __name__ == "__main__"`.

Same section rules (omit any section the guide allows to omit):

| Section     | When required                                                                                                                                                                                                                                                                 |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Summary** | Always (one line; extra paragraph only if needed).                                                                                                                                                                                                                            |
| **Args**    | When parameters are not obvious from name + type hints alone. If present, list **every** parameter.                                                                                                                                                                           |
| **Returns** | When return semantics are not fully covered by a summary that **starts with** `Return`, `Returns`, `Yield`, or `Yields`. **Do not** add **Returns** for `-> None` side-effect-only functions. **Do not** duplicate a sufficient opening `Return(s)…` sentence in **Returns**. |
| **Yields**  | Generators/async generators — document `next()` value, not the generator object.                                                                                                                                                                                              |
| **Raises**  | Every exception relevant to callers (or maintainers, for private helpers). Omit only when the function truly cannot raise.                                                                                                                                                    |

`@property` accessors: summary only (attribute-style, not `Returns the …`).

### Examples

**Returns in summary — omit Returns section:**

```python
def name_map_path(output_dir: str | os.PathLike[str]) -> pathlib.Path:
    """Return ``name_map.json`` path under ``output_dir``.

    Args:
        output_dir: Preprocess output root.
    """
```

**Private helper — same rules:**

```python
def _pack_folder_name(pack_relpath: str) -> str:
    """Return the basename of a pack path relative to the input root.

    Args:
        pack_relpath: Pack path relative to ``input_dir``.

    Returns:
        Final path segment (raw folder name).
    """
```

**Returns not in summary — include Returns:**

```python
def load_name_map(path: str | os.PathLike[str]) -> NameMap:
    """Load ``name_map.json`` from disk.

    Args:
        path: Path to a name map JSON file.

    Returns:
        Parsed name map with legacy ``source_pack`` paths canonicalized.

    Raises:
        ValueError: If ``schema_version`` is missing or unsupported.
    """
```

**Side-effect only (`-> None`) — no Returns:**

```python
def validate_prep_config(config: PrepConfig) -> None:
    """Validate prep config before a batch run.

    Args:
        config: Prep settings to check.

    Raises:
        ValueError: If any field is out of range or unsupported in v1.
    """
```

### Agent checklist (before finishing Python edits)

1. Run **`ruff check .`** at repo root (CI scope), then format changed paths.
2. Run **pydoclint** and **pyright** on changed `src/stepcovnet/` packages (and modified scripts when adding types/docstrings).
3. Before push: run `python pre_submit.py --fast` (or `--skip-install` before PR).
4. For each new/changed function or method (public **or** `_private`): summary line; **Args** if not obvious; **Raises** when applicable; **Returns** only per table above.
