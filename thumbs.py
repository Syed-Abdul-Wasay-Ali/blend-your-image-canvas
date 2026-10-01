"""Cached thumbnails for the CarPlacementCanvas selection strips.

ComfyUI's /view re-encodes the full-resolution source on every request: an input plate PNG
can be ~25 MB, so each 73px tile cost ~1 MB of on-the-fly webp and about a second of CPU.
Thirteen tiles therefore took ~20s to paint (measured). This serves a 176px webp from a
disk cache instead, regenerated only when the source file changes.

Route: GET /car_placement_canvas/thumb?folder=car_env&file=01_mountain_road.png
"""

import asyncio
import importlib
import os
from pathlib import Path

from PIL import Image, ImageFile

import folder_paths

# a plate whose upload was interrupted still deserves a tile: decode what is there
ImageFile.LOAD_TRUNCATED_IMAGES = True

# optional decoders, same ones nodes.py registers, so a heic/avif source gets a tile too
for _mod, _fn in (("pillow_heif", "register_heif_opener"),
                  ("pillow_avif", "register_avif_opener")):
    try:
        getattr(importlib.import_module(_mod), _fn)()
    except Exception:
        pass

MAX_W = 176
QUALITY = 82
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".dib", ".tif", ".tiff",
             ".gif", ".tga", ".ico", ".ppm", ".pgm", ".pbm", ".pcx", ".dds", ".jp2",
             ".j2k", ".psd", ".sgi", ".im", ".msp", ".xbm", ".avif", ".heic", ".heif")
SUBFOLDERS = ("car_env", "car_angles")


def is_image(path) -> bool:
    """Any decodable image counts as a layer, not just the extensions above."""
    try:
        with Image.open(path) as im:
            return bool(im.format)
    except Exception:
        return False


def _cache_root() -> Path:
    d = Path(folder_paths.get_input_directory()) / "car_thumbs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def thumb_path(sub: str, name: str) -> Path:
    if sub not in SUBFOLDERS:
        raise ValueError("bad subfolder")
    base = os.path.basename(name)
    if not base.lower().endswith(IMAGE_EXT) and not is_image(
            Path(folder_paths.get_input_directory()) / sub / base):
        raise ValueError("bad file")
    return _cache_root() / sub / (Path(base).stem + ".webp")


def ensure_thumb(sub: str, name: str):
    """Return the cached thumb path, (re)building it when the source is newer. Blocking."""
    src = Path(folder_paths.get_input_directory()) / sub / os.path.basename(name)
    if not src.is_file():
        return None
    out = thumb_path(sub, name)
    try:
        if out.is_file() and out.stat().st_mtime >= src.stat().st_mtime:
            return out
    except OSError:
        pass
    out.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(src) as im:
        if im.mode not in ("RGBA", "RGB"):
            im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
        if im.width > MAX_W:
            im = im.resize((MAX_W, max(1, round(im.height * MAX_W / im.width))), Image.LANCZOS)
        tmp = out.with_suffix(".tmp.webp")
        im.save(tmp, "WEBP", quality=QUALITY, method=4)
    os.replace(tmp, out)
    return out


def register_route() -> bool:
    """Attach the thumbnail route to the running ComfyUI server (no-op when headless)."""
    if getattr(register_route, "_done", False):
        return True
    try:
        from aiohttp import web
        from server import PromptServer

        @PromptServer.instance.routes.get("/car_placement_canvas/thumb")
        async def _cpc_thumb(request):
            sub = request.query.get("folder", "")
            name = request.query.get("file", "")
            loop = asyncio.get_event_loop()
            try:
                path = await loop.run_in_executor(None, ensure_thumb, sub, name)
            except Exception:
                path = None
            if not path:
                raise web.HTTPNotFound()
            return web.FileResponse(path, headers={"Cache-Control": "max-age=86400"})

    except Exception:
        return False
    register_route._done = True
    return True
