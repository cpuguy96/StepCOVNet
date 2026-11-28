---
description: Tests and coverage when changing src/, scripts/, or tests/
globs: "{src,scripts,tests}/**/*"
trigger: glob
---

# Tests and coverage

When you add or modify code under `src/`, `scripts/`, or `tests/`:

1. **Update tests** in `tests/` (one test module per source module: `stepcovnet/foo.py` -> `tests/foo_test.py`; `scripts/bar.py` -> `tests/bar_test.py` when the script rule below applies).
2. **Delta coverage** — aim for **100%** of changed/new code via **public API** tests; high 90s only as last resort. Do not test private helpers solely for coverage.
3. **Lint & type checks:** Run Ruff, pydoclint, and pyright per [python-style.md](python-style.md) (`pyproject.toml` — import order, Google docstrings, pathlib, types).
4. **Verify** from repo root using the project venv (see [python-env.md](python-env.md)):

```text
python -m pytest tests/ -m "not slow" --cov=stepcovnet --cov=scripts
```

Use the relevant test file when scope is narrow. Import conventions beyond Ruff: [python-style.md](python-style.md).

## Scripts (`scripts/`)

**Default:** add `tests/<script>_test.py` in the **same change** when the script contains non-trivial **parsing, discovery, path resolution, or CLI argument assembly** (even if helpers are `_private`).

| Script shape | Tests required? | Where logic is covered |
| ------------ | --------------- | ---------------------- |
| Parsing / discovery / path logic (`argparse`, manifest walks, logdir specs, JSON config reads) | **Yes** — `tests/<script>_test.py` | Script tests + temp dirs; patch `REPO` when paths are repo-relative |
| Thin CLI that only calls `src/stepcovnet` with no local logic | No dedicated script test | Existing `tests/` for the imported package |
| Mixed (local helpers + `src/` calls) | Yes for the local helpers | Script tests for discovery/paths; package tests for core behavior |

Examples: `audit_agent_brain.py` → `audit_agent_brain_test.py`; `project_status.py` → `project_status_test.py`. Pattern: import the script module from `tests/` via `sys.path` insert on `scripts/` (see `tests/conftest.py` — set `STEPCOVNET_NO_WSL=1` is already global).

Agents: do **not** skip script tests for “ops helpers” when they match the first row — that is the default, not optional.

## Test patterns

**Mocks:** `from unittest import mock`.

- **Stand-in objects:** prefer a **real minimal instance** (small class with only the fields the code touches) when no behavior needs stubbing. Otherwise use `mock.create_autospec(SpecClass, instance=True)` — pass a **live instance** (`create_autospec(_Model(), instance=True)`) when attributes must keep real values (e.g. `stop_training: bool`). Avoid bare `mock.MagicMock()`.
- **`patch.object` / `@patch`:** pass `autospec=True` when replacing callables (functions, methods, bound hooks). Patch where imported, not dotted strings.
- **Skip `autospec`** only when replacing a module attribute with a plain value (`sys.argv`, env constants, `Path` flags) or when `create=True` adds a non-existent attribute (e.g. `sys.frozen`).
- **`patch.dict`:** no autospec (not applicable).
- **Opaque third-party handles** (torch models, Tk events): `create_autospec` when a real type is available; otherwise `mock.Mock(spec=[...])` or `SimpleNamespace` for attribute bags.

**Schedules and knobs:** test that the knob **changes behavior**, not just that its schedule computes the right number. A ramp helper covered only by arithmetic assertions can be green while the value it produces is ignored at run time — that is how scheduled sampling stayed inert for months ([EXP-20260802-05](../../docs/research/EXPERIMENT_LOG.md#exp-20260802-05-scheduled-sampling-on-r2-is-a-no-op--the-branch-is-compiled-out-of-train_step)).

**Values mutated mid-training:** when schedules or callbacks mutate training state (e.g., scheduled sampling `p`, loss weights, EMA shadow parameters), assert that the compiled or eager training step actually observes the updated value by running one step first, changing the state, and verifying the next step's output changes.

**Temp files:** `tempfile.TemporaryDirectory()` or `NamedTemporaryFile` inside `with` (`delete=True`); never `delete=False` + manual unlink.

**Duplication:** shared helpers, `@pytest.mark.parametrize`, fixtures — avoid copy-pasted bodies.

**Flat tests:** `with mock.patch("a"), mock.patch("b"):` or stacked `@mock.patch` — avoid deep nesting.

**Test module docstrings:** omit unless documenting non-obvious setup.
