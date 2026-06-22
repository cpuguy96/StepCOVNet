"""Dispatch GPU workloads to WSL on Windows."""

from __future__ import annotations

import os
import pathlib
import shlex
import subprocess
import sys


def is_windows() -> bool:
    """Return True when running on native Windows (not inside WSL).

    Returns:
        True for win32 platform outside WSL.
    """
    return sys.platform == "win32"


def is_running_in_wsl() -> bool:
    """Return True when the current process is running inside WSL.

    Returns:
        True if STEPCOVNET_IN_WSL is set or /proc/version mentions Microsoft.
    """
    if os.environ.get("STEPCOVNET_IN_WSL") == "1":
        return True
    if sys.platform != "linux":
        return False
    try:
        with pathlib.Path("/proc/version").open(encoding="utf-8") as version_file:
            return "microsoft" in version_file.read().lower()
    except OSError:
        return False


def wsl_gpu_available() -> bool:
    """Return True when WSL is installed and reports at least one NVIDIA GPU.

    Returns:
        True if ``wsl nvidia-smi -L`` succeeds and lists a GPU.
    """
    if not is_windows():
        return False
    try:
        result = subprocess.run(
            ["wsl", "-e", "bash", "-lc", "nvidia-smi -L"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and "GPU" in result.stdout


def find_repo_root(start: str | None = None) -> pathlib.Path:
    """Locate the repository root containing ``pyproject.toml``.

    Args:
        start: Optional starting path (file or directory).

    Returns:
        Absolute path to the repo root.

    Raises:
        RuntimeError: If no ``pyproject.toml`` is found in parents.
    """
    start_path = pathlib.Path(start or pathlib.Path.cwd()).resolve()
    if start_path.is_file():
        start_path = start_path.parent
    for candidate in (start_path, *start_path.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RuntimeError("Could not find repo root (pyproject.toml)")


def windows_path_to_wsl(path: str) -> str:
    """Convert a Windows absolute path to a WSL ``/mnt/<drive>/...`` path.

    Args:
        path: Windows or already-Unix path string.

    Returns:
        WSL path when input is a Windows drive path; otherwise unchanged.
    """
    cleaned = path.strip().strip('"')
    if len(cleaned) >= 2 and cleaned[1] == ":":
        drive = cleaned[0].lower()
        rest = cleaned[2:].replace("\\", "/")
        if not rest.startswith("/"):
            rest = f"/{rest}"
        return f"/mnt/{drive}{rest}"
    return cleaned.replace("\\", "/")


def translate_arg_for_wsl(arg: str) -> str:
    """Translate one CLI argument path segment for WSL when needed.

    Args:
        arg: Single argv element.

    Returns:
        Argument with Windows paths converted inside ``--key=value`` forms.
    """
    if arg.startswith("--") and "=" in arg:
        key, _, value = arg.partition("=")
        return f"{key}={windows_path_to_wsl(value)}"
    if len(arg) >= 2 and arg[1] == ":":
        return windows_path_to_wsl(arg)
    if "\\" in arg:
        return windows_path_to_wsl(arg)
    return arg


def translate_argv_for_wsl(argv: list[str]) -> list[str]:
    """Translate script argv for execution inside WSL.

    Args:
        argv: Full argv including script path at index 0.

    Returns:
        New argv with Windows paths converted (script path unchanged).
    """
    if not argv:
        return []
    return [argv[0], *[translate_arg_for_wsl(arg) for arg in argv[1:]]]


def parse_device_from_argv(argv: list[str] | None) -> str:
    """Parse ``--device`` from an argument vector (defaults to cpu).

    Args:
        argv: Argument vector without the executable name.

    Returns:
        Device string, e.g. ``cpu`` or ``cuda``.
    """
    args = argv or []
    for index, arg in enumerate(args):
        if arg == "--device" and index + 1 < len(args):
            return args[index + 1].lower()
        if arg.startswith("--device="):
            return arg.split("=", 1)[1].lower()
    return "cpu"


def device_requests_gpu(device: str) -> bool:
    """Return True when a torch device string requests GPU execution.

    Args:
        device: Device string such as ``cuda`` or ``cpu``.

    Returns:
        True for cuda/cuda:N device strings.
    """
    return device == "cuda" or device.startswith("cuda:")


def wsl_disabled() -> bool:
    """Return True when WSL dispatch is explicitly disabled via environment.

    Returns:
        True if STEPCOVNET_NO_WSL=1.
    """
    return os.environ.get("STEPCOVNET_NO_WSL") == "1"


def default_wsl_venv() -> pathlib.Path:
    """Return the WSL GPU virtualenv path.

    Honors ``WSL_VENV`` when set; otherwise ``~/stepcovnet-venv-wsl``.
    """
    env_venv = os.environ.get("WSL_VENV")
    if env_venv:
        return pathlib.Path(env_venv)
    return pathlib.Path.home() / "stepcovnet-venv-wsl"


def nvidia_library_dirs(venv_path: pathlib.Path | None = None) -> list[pathlib.Path]:
    """Return sorted nvidia wheel ``lib*`` directories under a WSL venv.

    Matches ``find "$VENV" -path '*/nvidia/*/lib*' -type d`` from
    ``scripts/wsl_gpu_env.sh``.

    Args:
        venv_path: Virtualenv root; defaults to ``~/stepcovnet-venv-wsl``.

    Returns:
        Existing directories under ``*/nvidia/*/lib*`` beneath the venv.
    """
    root = venv_path or default_wsl_venv()
    if not root.is_dir():
        return []
    lib_dirs: set[pathlib.Path] = set()
    for path in root.rglob("*"):
        if not path.is_dir():
            continue
        rel_parts = path.relative_to(root).parts
        for index, part in enumerate(rel_parts):
            if part != "nvidia":
                continue
            if index + 2 < len(rel_parts) and rel_parts[index + 2].startswith("lib"):
                lib_dirs.add(path)
                break
    return sorted(lib_dirs)


def apply_tensorflow_gpu_library_path(venv_path: pathlib.Path | None = None) -> bool:
    """Prepend NVIDIA wheel library paths for TensorFlow GPU inside WSL.

    Mirrors ``scripts/wsl_gpu_env.sh``. Must run **before the Python process
    starts** (e.g. ``source scripts/wsl_gpu_env.sh`` in the shell). Changing
    ``LD_LIBRARY_PATH`` from Python after startup does not affect TensorFlow's
    GPU loader; use :func:`reexec_with_tensorflow_gpu_env_if_needed` instead.

    Args:
        venv_path: Virtualenv root; defaults to ``~/stepcovnet-venv-wsl``.

    Returns:
        True when ``LD_LIBRARY_PATH`` was updated with at least one directory.
    """
    lib_dirs = nvidia_library_dirs(venv_path)
    if not lib_dirs:
        return False
    joined = ":".join(str(path) for path in lib_dirs)
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if current:
        os.environ["LD_LIBRARY_PATH"] = f"{joined}:{current}"
    else:
        os.environ["LD_LIBRARY_PATH"] = joined
    return True


def reexec_with_tensorflow_gpu_env_if_needed(argv: list[str]) -> None:
    """Re-exec the current script under ``wsl_gpu_env.sh`` when inside WSL.

    TensorFlow only picks up NVIDIA wheel libraries when ``LD_LIBRARY_PATH`` is
    set before Python starts. When ``STEPCOVNET_WSL_GPU_ENV=1`` is already set,
    this is a no-op.

    Args:
        argv: Full argv including script path at index 0.
    """
    if not is_running_in_wsl() or os.environ.get("STEPCOVNET_WSL_GPU_ENV") == "1":
        return
    repo_root = find_repo_root(argv[0] if argv else None)
    gpu_env = repo_root / "scripts" / "wsl_gpu_env.sh"
    if not gpu_env.is_file():
        return
    python_exe = shlex.quote(sys.executable)
    script_and_args = " ".join(shlex.quote(arg) for arg in argv)
    command = (
        f"source {shlex.quote(str(gpu_env))} && "
        f"export STEPCOVNET_WSL_GPU_ENV=1 && "
        f"export STEPCOVNET_IN_WSL=1 && "
        f"cd {shlex.quote(str(repo_root))} && "
        f"exec {python_exe} {script_and_args}"
    )
    os.execvp("bash", ["bash", "-lc", command])


def require_tensorflow_gpu() -> None:
    """Exit with an error when TensorFlow cannot see a GPU in WSL training.

    Call after :func:`apply_tensorflow_gpu_library_path` and before training.
    """
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"TensorFlow GPU devices: {gpus}", flush=True)
        return
    if is_running_in_wsl():
        print(
            "TensorFlow found no GPU in WSL. Run "
            "`source scripts/wsl_gpu_env.sh` before training, or use "
            "`wsl_gpu.apply_tensorflow_gpu_library_path()` before importing TensorFlow.",
            flush=True,
        )
        raise SystemExit(1)
    print("Warning: training without a visible TensorFlow GPU device.", flush=True)


def _wsl_bash_command(
    repo_root_wsl: str, python_cmd: str, script_rel: str, args: list[str]
) -> str:
    ensure = f"{repo_root_wsl}/scripts/wsl_ensure_env.sh"
    gpu_env = f"{repo_root_wsl}/scripts/wsl_gpu_env.sh"
    quoted_args = " ".join(shlex.quote(arg) for arg in args)
    script_path = f"{repo_root_wsl}/{script_rel}"
    return (
        f"bash {shlex.quote(ensure)} && "
        f"source {shlex.quote(gpu_env)} && "
        f"export STEPCOVNET_WSL_GPU_ENV=1 && "
        f"cd {shlex.quote(repo_root_wsl)} && "
        f"export STEPCOVNET_IN_WSL=1 && "
        f"{python_cmd} {shlex.quote(script_path)} {quoted_args}"
    )


def run_script_in_wsl(script_rel: str, argv: list[str]) -> int:
    """Run a repo script inside WSL using the project GPU venv.

    Args:
        script_rel: Script path relative to repo root (e.g. ``scripts/train_onset.py``).
        argv: Full process argv (``sys.argv``), including script path at index 0.

    Returns:
        Subprocess exit code from WSL.
    """
    repo_root = find_repo_root(argv[0] if argv else None)
    repo_root_wsl = windows_path_to_wsl(str(repo_root))
    wsl_argv = translate_argv_for_wsl(argv)
    wsl_args = wsl_argv[1:]
    python_cmd = "${STEPCOVNET_WSL_PYTHON:-$HOME/stepcovnet-venv-wsl/bin/python}"
    command = _wsl_bash_command(repo_root_wsl, python_cmd, script_rel, wsl_args)
    print(f"Dispatching to WSL GPU: {script_rel}", flush=True)
    completed = subprocess.run(
        ["wsl", "-e", "bash", "-lc", command],
        check=False,
    )
    return int(completed.returncode)


def maybe_dispatch_for_mert_extract(script_rel: str, argv: list[str]) -> bool:
    """Dispatch MERT extraction to WSL when GPU is requested on Windows.

    Args:
        script_rel: Script path relative to repo root.
        argv: Full process argv.

    Returns:
        True if the call was dispatched to WSL (caller should exit).
    """
    if wsl_disabled() or is_running_in_wsl() or not is_windows():
        return False
    device = parse_device_from_argv(argv[1:])
    if not device_requests_gpu(device):
        return False
    if not wsl_gpu_available():
        raise SystemExit(
            "CUDA device requested but WSL GPU is unavailable. "
            "Install WSL + NVIDIA drivers, or pass --device=cpu."
        )
    code = run_script_in_wsl(script_rel, argv)
    raise SystemExit(code)


def maybe_dispatch_for_training(script_rel: str, argv: list[str]) -> bool:
    """Dispatch TensorFlow training to WSL on Windows when a GPU is available.

    Args:
        script_rel: Script path relative to repo root.
        argv: Full process argv.

    Returns:
        True if the call was dispatched to WSL (caller should exit).
    """
    if wsl_disabled() or is_running_in_wsl() or not is_windows():
        return False
    if not wsl_gpu_available():
        return False
    code = run_script_in_wsl(script_rel, argv)
    raise SystemExit(code)
