"""Rebuild every cached selection-strip thumbnail.

Run after adding images to input/car_env or input/car_angles:

    cd C:/Users/syeda/EZlaunch-Minimax-H3/ComfyUI
    ../venv/Scripts/python.exe custom_nodes/car_placement_canvas/regen_thumbs.py

Writes input/car_thumbs/<folder>/<stem>.webp (176px wide). The node also rebuilds a thumb
on demand through /car_placement_canvas/thumb once the server has been restarted.
"""

import importlib.util
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))  # ComfyUI root, for folder_paths

spec = importlib.util.spec_from_file_location("cpc_thumbs", HERE / "thumbs.py")
thumbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(thumbs)

input_dir = Path(thumbs.folder_paths.get_input_directory())
total = 0
started = time.time()
for sub in thumbs.SUBFOLDERS:
    src_dir = input_dir / sub
    if not src_dir.is_dir():
        print(f"{sub}: (no such folder)")
        continue
    files = sorted(p for p in src_dir.iterdir() if p.is_file() and thumbs.is_image(str(p)))
    for p in files:
        out = thumbs.ensure_thumb(sub, p.name)
        print(f"{sub}/{p.name} -> {out.name if out else 'FAILED'} ({out.stat().st_size if out else 0} bytes)")
        total += 1
print(f"{total} thumbnails in {time.time() - started:.1f}s -> {input_dir / 'car_thumbs'}")
