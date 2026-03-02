# PyInstaller spec for StepCOVNet Generator UI.
# Run from project root: pyinstaller scripts/generate_ui.spec
# Requires stepcovnet installed (e.g. pip install -e .) and PyInstaller (pip install pyinstaller).
# Note: PyInstaller exec()s this file without __file__; we use getcwd() which is project root.

import os

project_root = os.path.abspath(os.getcwd())
src_path = os.path.join(project_root, "src")

a = Analysis(
    [os.path.join(project_root, "scripts", "generate_ui.py")],
    pathex=[project_root, src_path],
    hiddenimports=[
        "stepcovnet",
        "stepcovnet.generator",
        "stepcovnet.models",
        "stepcovnet.datasets",
        "stepcovnet.constants",
    ],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="generate_ui",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    onefile=True,
)
