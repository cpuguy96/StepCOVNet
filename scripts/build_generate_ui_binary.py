r"""Build the standalone Generator UI executable with PyInstaller.

Run from anywhere; uses the project root (parent of scripts/). Requires
stepcovnet and pyinstaller installed (e.g. pip install -e . && pip install .[build]).

Usage:
    python scripts/build_generate_ui_binary.py
"""

import os
import subprocess
import sys


def main() -> None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    spec_path = os.path.join(script_dir, "generate_ui.spec")

    if not os.path.isfile(spec_path):
        print(f"Spec file not found: {spec_path}", file=sys.stderr)
        sys.exit(1)

    os.chdir(project_root)
    result = subprocess.run(
        [sys.executable, "-m", "PyInstaller", spec_path],
        cwd=project_root,
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
