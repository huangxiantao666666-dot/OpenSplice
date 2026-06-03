"""
Build standalone executable with PyInstaller.

Usage:
    pip install pyinstaller
    python build_exe.py

Output: dist/OpenSplice.exe  (single file, ~3 GB with SAM3 checkpoint)
"""

import os
import sys
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"

def build():
    print("=== OpenSplice Standalone Builder ===\n")

    # 1. Ensure checkpoints exist in the package
    sam3_pt = ROOT / "checkpoints" / "sam3.pt"
    simopa_pt = ROOT / "checkpoints" / "simopa.pth"
    if not sam3_pt.exists():
        print(f"ERROR: {sam3_pt} not found. Download SAM3 checkpoint first.")
        print("  See README.md for download instructions.")
        sys.exit(1)

    # 2. Clean old build
    for d in ["build", "dist"]:
        path = ROOT / d
        if path.exists():
            shutil.rmtree(path)
            print(f"Cleaned {d}/")

    # 2.5 Ensure outputs directory
    (ROOT / "outputs").mkdir(exist_ok=True)

    # 3. Create launcher script (auto-downloads weights on first run)
    launcher = ROOT / "_launcher.py"
    launcher.write_text(r'''
import os, sys, webbrowser, threading, time, pathlib

os.environ.setdefault("SAM3_DEVICE", "cpu")
os.environ.setdefault("OUTPUT_DIR", "./outputs")

# ── Auto-download pretrained weights ──────────────────────────────────────

CKPT_DIR = pathlib.Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

def download_sam3():
    """Download SAM3 checkpoint (~3.4 GB) — only if not already bundled."""
    dest = CKPT_DIR / "sam3.pt"
    if dest.exists():
        sz = dest.stat().st_size
        print(f"[OK] SAM3 checkpoint: {dest} ({sz/1e9:.1f} GB)")
        return True

    print("\n" + "=" * 60)
    print("  SAM 3 checkpoint not found (not bundled or moved).")
    print("  Auto-downloading (~3.4 GB, one-time only)...")
    print("=" * 60)

    # Try HuggingFace first
    try:
        from huggingface_hub import hf_hub_download
        print("  Source: HuggingFace (huggingface.co)")
        hf_hub_download("facebook/sam3", "sam3.pt", local_dir=str(CKPT_DIR))
        if dest.exists():
            print(f"  [OK] Downloaded to {dest}")
            return True
    except Exception as e:
        print(f"  HuggingFace failed: {e}")

    # Try ModelScope (faster in China)
    try:
        from modelscope import snapshot_download
        print("  Source: ModelScope (modelscope.cn)")
        snapshot_download("facebook/sam3", cache_dir=str(CKPT_DIR))
        # Find the .pt file
        for f in CKPT_DIR.rglob("*.pt"):
            if f.name == "sam3.pt" or "sam3" in f.name.lower():
                if f != dest:
                    import shutil
                    shutil.copy(f, dest)
                print(f"  [OK] Downloaded to {dest}")
                return True
    except Exception as e:
        print(f"  ModelScope failed: {e}")

    print(f"\n  Auto-download failed. Please download manually to:")
    print(f"  {dest.resolve()}")
    print(f"  See: https://github.com/facebookresearch/sam3")
    return False

def download_simopa():
    """Download simOPA checkpoint (~45 MB). Shows progress."""
    dest = CKPT_DIR / "simopa.pth"
    if dest.exists():
        print(f"[OK] simOPA checkpoint found ({dest.stat().st_size/1e6:.0f} MB)")
        return True

    print("  Auto-downloading simOPA weights (~45 MB)...")
    try:
        from huggingface_hub import hf_hub_download
        hf_hub_download(
            "BCMIZB/Libcom_pretrained_models",
            "SimOPA.pth",
            local_dir=str(CKPT_DIR),
        )
        # Rename if needed
        for f in CKPT_DIR.glob("*.pth"):
            if f.name.lower() == "simopa.pth" and f != dest:
                import shutil; shutil.move(str(f), str(dest))
                break
        if dest.exists():
            print(f"  [OK] simOPA ready")
            return True
    except Exception as e:
        print(f"  HuggingFace failed: {e}")

    print(f"  simOPA scoring will be unavailable.")
    return False

# Run downloads
ok_sam3 = download_sam3()
ok_simopa = download_simopa()
if ok_sam3:
    os.environ["SAM3_CHECKPOINT"] = str(CKPT_DIR / "sam3.pt")

print(f"\n  SAM3:  {'READY' if ok_sam3 else 'MISSING'}")
print(f"  simOPA: {'READY' if ok_simopa else 'UNAVAILABLE'}")
print(f"  Starting OpenSplice...\n")

def open_browser():
    time.sleep(3)
    webbrowser.open("http://127.0.0.1:7860")

threading.Thread(target=open_browser, daemon=True).start()

from image_stitch_agent.app import main
main()
''', encoding="utf-8")
    print("Created _launcher.py (SAM3 + simOPA auto-download)")

    # 4. Run PyInstaller — bundle existing checkpoints, skip if missing
    ckpt = ROOT / "checkpoints"
    sam3_pt = ckpt / "sam3.pt"
    simopa_pt = ckpt / "simopa.pth"

    cmd_parts = [
        sys.executable, "-m", "PyInstaller",
        "--onefile",
        "--name", "OpenSplice",
        "--add-data", f"image_stitch_agent{os.pathsep}image_stitch_agent",
        "--add-data", f".env{os.pathsep}.",
    ]

    # Bundle checkpoints that exist (avoid re-download in .exe)
    bundled = []
    if sam3_pt.exists():
        cmd_parts += ["--add-data", f"{sam3_pt}{os.pathsep}checkpoints"]
        bundled.append(f"SAM3 ({sam3_pt.stat().st_size/1e9:.1f} GB)")
    else:
        print("NOTE: sam3.pt not found — will auto-download on first run")

    if simopa_pt.exists():
        cmd_parts += ["--add-data", f"{simopa_pt}{os.pathsep}checkpoints"]
        bundled.append(f"simOPA ({simopa_pt.stat().st_size/1e6:.0f} MB)")
    else:
        print("NOTE: simopa.pth not found — scoring will be unavailable")

    for b in bundled:
        print(f"Bundling: {b}")

    cmd_parts += [
        "--hidden-import", "sam3",
        "--hidden-import", "sam3.model_builder",
        "--hidden-import", "sam3.model.sam3_image_processor",
        "--hidden-import", "gradio",
        "--hidden-import", "cv2",
        "--hidden-import", "numpy",
        "--hidden-import", "PIL",
        "--hidden-import", "dashscope",
        "--hidden-import", "torch",
        "--hidden-import", "torchvision",
        "--hidden-import", "requests",
        "--hidden-import", "huggingface_hub",
        "--collect-all", "sam3",
        "--collect-all", "gradio",
        "--collect-all", "torchvision",
        "--collect-submodules", "torch",
    ]
    cmd_parts.append(str(launcher))

    print(f"Running PyInstaller...\n")
    import subprocess
    result = subprocess.run(cmd_parts, check=False)
    if result.returncode != 0:
        print("\nBuild failed. Check PyInstaller output above.")
        sys.exit(1)

    # 5. Clean up
    launcher.unlink(missing_ok=True)

    # 6. Result
    exe = DIST / "OpenSplice.exe"
    if exe.exists():
        size_mb = exe.stat().st_size / (1024*1024)
        print(f"\n{'='*60}")
        print(f"  Built: {exe}")
        print(f"  Size:  {size_mb:.0f} MB")
        print(f"\n  Double-click OpenSplice.exe to run.")
        print(f"  Browser opens automatically at http://127.0.0.1:7860")
        print(f"{'='*60}")
    else:
        print("\nBuild failed. Check PyInstaller output above.")


if __name__ == "__main__":
    build()
