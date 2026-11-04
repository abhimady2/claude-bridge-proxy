# -*- mode: python ; coding: utf-8 -*-
import sys

# pystray's backend module is platform-specific: pystray._win32 does not exist
# on macOS and pystray._darwin does not exist on Windows. Importing the wrong
# one is a runtime crash, so pick the backend for the platform being built.
IS_MAC = sys.platform == 'darwin'
PYSTRAY_BACKEND = 'pystray._darwin' if IS_MAC else 'pystray._win32'

a = Analysis(
    ['claude_bridge_app.py'],
    pathex=[],
    binaries=[],
    datas=[('assets', 'assets')],
    hiddenimports=[PYSTRAY_BACKEND, 'anthropic', 'PIL.ImageTk', 'socksio'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['test_proxy'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

if IS_MAC:
    # A macOS .app must be one-dir wrapped by COLLECT/BUNDLE -- a onefile exe
    # re-extracts to temp on every launch and drops everything into the
    # bundle's root, defeating the layout. PyInstaller converts the .ico to
    # .icns itself, so the Windows icon asset works for both.
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name='ClaudeBridge',
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        upx_exclude=[],
        console=False,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
        icon='assets/icon.ico',
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name='ClaudeBridge',
    )
    app = BUNDLE(
        coll,
        name='ClaudeBridge.app',
        icon='assets/icon.ico',
        bundle_identifier='com.claudebridge.app',
        info_plist={
            'CFBundleName': 'Claude Bridge',
            'CFBundleDisplayName': 'Claude Bridge',
            'NSHighResolutionCapable': True,
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name='ClaudeBridge',
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
        icon='assets/icon.ico',
    )
