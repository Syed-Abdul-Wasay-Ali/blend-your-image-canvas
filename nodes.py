"""Car Placement Canvas - interactive drag/scale cutout placement, server-side composite.

Drop a car cutout on a background plate inside the node (drag, resize, rotate), and the
server composites the exact same frame so it can feed a klein / relight / LoRA stage.

Front-end: web/car_canvas.js renders the live canvas inside the node.
Back-end:  this file owns the pixel math, so a headless /prompt run produces byte-for-byte
           the placement the canvas showed.
"""

import importlib
import json
import math
import os
import re

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFile, ImageFilter

import folder_paths

# an upload that was cut short still composites (PIL fills the missing rows from the last
# complete one) instead of throwing at the composite stage
ImageFile.LOAD_TRUNCATED_IMAGES = True

CATEGORY = "car placement"

# Layer formats come from the decoder, not this tuple: a layer may be any image Pillow reads,
# so the list only documents the common ones and is not used to gate anything. Non-car photos,
# 16-bit TIFFs, palette GIFs, JP2/PCX/DDS/TGA and extension-less exports all list as layers.
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".jfif", ".webp", ".bmp", ".dib", ".tif", ".tiff",
             ".gif", ".tga", ".ico", ".ppm", ".pgm", ".pbm", ".pcx", ".dds", ".jp2",
             ".j2k", ".psd", ".sgi", ".im", ".msp", ".xbm", ".avif", ".heic", ".heif")

# Leading bytes that identify a format Pillow can decode here. Deliberately narrow: a text
# file renamed .png matches none of them, and a format nothing can read (psd without a psd
# decoder, heic without pillow_heif) is not advertised as a layer just because it sniffed.
_MAGIC_PREFIX = (b"\x89PNG\r\n\x1a\n", b"\xff\xd8\xff", b"GIF87a", b"GIF89a", b"BM",
                 b"II*\x00", b"MM\x00*", b"\x00\x00\x01\x00", b"\x00\x00\x02\x00",
                 b"\x00\x00\x0a\x00", b"DDS ", b"\x00\x00\x00\x0cjP", b"\xff\x4f\xff\x51",
                 b"qoif")
_PPM_MAGIC = (b"P1", b"P2", b"P3", b"P4", b"P5", b"P6")

# optional decoders: when these are installed they widen "any format" to heic/avif too
for _mod, _fn in (("pillow_heif", "register_heif_opener"),
                  ("pillow_avif", "register_avif_opener")):
    try:
        getattr(importlib.import_module(_mod), _fn)()
    except Exception:
        pass


def _sniff(path):
    """Cheap verdict from the first bytes: True, False, or None when the decoder must decide.

    A file that is not an image is rejected here even when it is named .png - the picker must
    only ever offer layers that will actually composite.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(512)
    except OSError:
        return False
    if not head:
        return False
    if head.startswith(b"RIFF") and head[8:12] == b"WEBP":
        return True
    if head[:2] in _PPM_MAGIC and head[2:3] in (b"\n", b"\r", b" ", b"\t"):
        return True
    if any(head.startswith(m) for m in _MAGIC_PREFIX):
        return True
    return None


def _is_image(path):
    """True when the file is a usable image: byte sniff first, Pillow header probe second."""
    v = _sniff(path)
    if v is not None:
        return v
    try:
        with Image.open(path) as im:
            return bool(im.format)
    except Exception:
        return False


def _list_images(d):
    """Every decodable image in a folder, whatever its extension."""
    try:
        entries = os.listdir(d)
    except OSError:
        return []
    names = []
    for f in entries:
        p = os.path.join(d, f)
        try:
            if not os.path.isfile(p):
                continue
        except OSError:
            continue
        if _is_image(p):
            names.append(f)
    return sorted(names)


def _input_images():
    return ["(none)"] + _list_images(folder_paths.get_input_directory())


def _folder_images(sub):
    """Dropdown options for a hard-coded selector folder (input/<sub>)."""
    return ["(none)"] + _list_images(os.path.join(folder_paths.get_input_directory(), sub))


def _first(sub):
    names = _folder_images(sub)
    return names[1] if len(names) > 1 else "(none)"


def _resolve(name, subfolder=""):
    """input/<subfolder>/<name>, falling back to the input root for dropped files."""
    if not name or name == "(none)":
        return None
    root = folder_paths.get_input_directory()
    for cand in ([os.path.join(root, subfolder, name)] if subfolder else []) + [os.path.join(root, name)]:
        if os.path.isfile(cand):
            return cand
    return None


def _open_layer(path, mode):
    """Open any image the decoder understands and normalize it to `mode`.

    Tolerates a truncated file and odd source modes (palette, CMYK, 16-bit, LA, I;16), so a
    layer is not limited to well-formed RGB PNG/JPEG. Fails loudly with the file named rather
    than silently falling back to an empty plate.
    """
    try:
        im = Image.open(path)
        im.load()
        return im if mode is None or im.mode == mode else im.convert(mode)
    except Exception as e:
        raise RuntimeError("cannot decode layer %r: %s" % (os.path.basename(path), e)) from e


def _t2pil(t):
    """torch (H,W,C) float 0..1 -> PIL, keeping alpha when present."""
    a = (t.detach().cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
    if a.ndim == 2:
        return Image.fromarray(a, "L")
    if a.shape[-1] == 4:
        return Image.fromarray(a, "RGBA")
    if a.shape[-1] == 3:
        return Image.fromarray(a, "RGB")
    return Image.fromarray(a[..., 0], "L")


def _mask2pil(m):
    """MASK (H,W) or (1,H,W) float -> PIL L."""
    a = m
    if hasattr(a, "detach"):
        a = a.detach().cpu().numpy()
    a = np.asarray(a, dtype=np.float32)
    while a.ndim > 2:
        a = a[0]
    return Image.fromarray((a * 255.0).clip(0, 255).astype(np.uint8), "L")


def _pil2t(img):
    a = np.asarray(img.convert("RGB") if img.mode == "L" else img).astype(np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    return torch.from_numpy(a)[None, ...]


def _pil2mask(img):
    a = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    return torch.from_numpy(a)[None, ...]


def _trim_alpha(img):
    """Crop to the alpha bounding box so `scale` means the size of the car itself."""
    if img.mode != "RGBA":
        return img
    bbox = img.getchannel("A").getbbox()
    return img.crop(bbox) if bbox else img


SAT_BG = 0.14
VAL_BG = 0.62


def _reach_from_border(seed):
    """True where a seed pixel is 4-connected to the image border."""
    from collections import deque
    h, w = seed.shape
    out = np.zeros_like(seed, dtype=bool)
    dq = deque()
    for x in range(w):
        for y in (0, h - 1):
            if seed[y, x] and not out[y, x]:
                out[y, x] = True
                dq.append((y, x))
    for y in range(h):
        for x in (0, w - 1):
            if seed[y, x] and not out[y, x]:
                out[y, x] = True
                dq.append((y, x))
    while dq:
        y, x = dq.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h and 0 <= nx < w and seed[ny, nx] and not out[ny, nx]:
                out[ny, nx] = True
                dq.append((ny, nx))
    return out


def _key_studio_bg(img, max_side=1400):
    """Near-white studio sweep -> alpha.

    The fill starts at the border so enclosed achromatic areas (glass, grille, wheel
    wells, shadowed underside) stay opaque, and saturated paint is forced opaque again
    afterwards. Only used when the selected plate is fully opaque.
    """
    rgb = img.convert("RGB")
    small, scaled = rgb, False
    if max(rgb.size) > max_side:
        k = max_side / float(max(rgb.size))
        small = rgb.resize((max(1, int(rgb.width * k)), max(1, int(rgb.height * k))), Image.LANCZOS)
        scaled = True
    a = np.asarray(small).astype(np.float32) / 255.0
    mx = a.max(axis=2)
    mn = a.min(axis=2)
    sat = np.where(mx > 1e-6, (mx - mn) / np.maximum(mx, 1e-6), 0.0)
    keep = sat > 0.28
    bg = _reach_from_border((sat < SAT_BG) & (mx > VAL_BG))
    alpha = np.where(bg, 0.0, 255.0)
    alpha = np.asarray(
        Image.fromarray(alpha.astype(np.uint8), "L").filter(ImageFilter.GaussianBlur(1.1))
    ).astype(np.float32)
    alpha[keep] = 255.0
    alpha[alpha < 12] = 0.0
    am = Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8), "L")
    if scaled:
        am = am.resize(rgb.size, Image.LANCZOS)
    out = rgb.convert("RGBA")
    out.putalpha(am)
    bbox = out.getchannel("A").getbbox()
    return out.crop(bbox) if bbox else out


def _layer_on(ui_state, key):
    """Read a UI layer switch. Absent means on, so older workflows keep every layer."""
    try:
        return json.loads(ui_state or "{}").get(key, True) is not False
    except ValueError:
        return True


def _pretty(name):
    """'01_front_3q_left.png' -> 'front three-quarter left' (for auto prompt text)."""
    if not name:
        return ""
    base = os.path.splitext(os.path.basename(str(name)))[0]
    base = re.sub(r"^\d+[\s_\-]+", "", base)
    words = [w for w in re.split(r"[\s_\-,]+", base) if w]
    fix = {"3q": "three-quarter", "34": "three-quarter", "fq": "front-quarter"}
    return " ".join(fix.get(w.lower(), w.lower()) for w in words)


class CarPlacementCanvas:
    """Drag a car cutout across a background plate; output the frozen frame + its mask."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # Hard-coded selectors. Picking an environment loads that plate as the
                # canvas background; picking a car angle drops that cutout on top as the
                # movable layer. Both read from the node's own input folders:
                #   input/car_env/     <- environment plates
                #   input/car_angles/  <- keyed RGBA car-angle cutouts
                "env_file": (_folder_images("car_env"), {"default": _first("car_env")}),
                "angle_file": (_folder_images("car_angles"), {"default": _first("car_angles")}),
                # an opaque studio plate gets alpha-keyed on the way in
                "key_studio_bg": ("BOOLEAN", {"default": True}),
                # LoadImage hands out the cutout as RGB + a 1-alpha MASK, so the mask is
                # transparency by default. Segmentation-style masks are coverage instead.
                "mask_mode": (["transparency (LoadImage)", "coverage (segmentation mask)"],
                              {"default": "transparency (LoadImage)"}),
                # x/y are the cutout CENTRE, normalised to the canvas (0..1, y down).
                "x": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.0005}),
                "y": ("FLOAT", {"default": 0.62, "min": -1.0, "max": 2.0, "step": 0.0005}),
                # scale = cutout width as a fraction of the canvas width.
                "scale": ("FLOAT", {"default": 0.45, "min": 0.01, "max": 5.0, "step": 0.001}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.05}),
                "flip_h": ("BOOLEAN", {"default": False}),
                # contact shadow strength 0 = none; 1 = heavy darkening under the cutout.
                "shadow": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_offset": ("FLOAT", {"default": 0.012, "min": -0.1, "max": 0.1, "step": 0.001}),
                "feather": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 64.0, "step": 0.5}),
                "canvas_width": ("INT", {"default": 1376, "min": 64, "max": 8192, "step": 8}),
                "canvas_height": ("INT", {"default": 768, "min": 64, "max": 8192, "step": 8}),
                "ui_state": ("STRING", {"default": "{}", "multiline": False}),
            },
            "optional": {
                "background": ("IMAGE",),
                "car_image": ("IMAGE",),
                "car_mask": ("MASK",),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "STRING")
    RETURN_NAMES = ("frame", "car_mask", "car_name", "env_name")
    FUNCTION = "place"
    CATEGORY = CATEGORY

    # ------------------------------------------------------------------ helpers
    def _temp_dir(self):
        try:
            return folder_paths.get_temp_directory()
        except Exception:
            import tempfile
            return tempfile.gettempdir()

    def _stash(self, uid, tag, img):
        """Park the incoming plate/cutout in temp so the node UI can preview them."""
        if uid is None:
            return
        # the UI only needs a preview: a 5500 px plate would be a 25 MB round trip
        view = img
        if max(img.size) > 1600:
            ratio = 1600.0 / max(img.size)
            view = img.resize((max(1, int(img.width * ratio)), max(1, int(img.height * ratio))), Image.LANCZOS)
        try:
            view.save(os.path.join(self._temp_dir(), "cpc_%s_%s.png" % (uid, tag)))
        except Exception:
            pass

    # ------------------------------------------------------------------ main
    def place(self, env_file, angle_file, key_studio_bg, mask_mode, x, y, scale, rotation,
              flip_h, shadow, shadow_offset, feather, canvas_width, canvas_height,
              ui_state="{}", background=None, car_image=None, car_mask=None, unique_id=None):

        uid = str(unique_id) if unique_id is not None else None
        # the UI can delete a layer: that has to hold even when an IMAGE is wired in
        env_layer = _layer_on(ui_state, "env_layer")
        car_layer = _layer_on(ui_state, "car_layer")

        # --- background plate (environment selector) --------------------------
        bg = None
        src_bg = None
        if not env_layer:
            src_bg = None
        elif background is not None and len(background) > 0:
            src_bg = _t2pil(background[0]).convert("RGB")
        else:
            p = _resolve(env_file, "car_env")
            if p:
                src_bg = _open_layer(p, "RGB")
        if src_bg is not None:
            bg = src_bg.copy()
            self._stash(uid, "bg", src_bg)
        else:
            bg = Image.new("RGB", (int(canvas_width), int(canvas_height)), (24, 24, 26))

        W, H = bg.size

        # --- car cutout (angle selector) --------------------------------------
        car = None
        if car_layer and car_image is not None and len(car_image) > 0:
            ci = _t2pil(car_image[0])
            if ci.mode == "RGBA":
                car = ci
            else:
                car = ci.convert("RGB")
                if car_mask is not None and len(car_mask) > 0:
                    m = _mask2pil(car_mask[0]).resize(car.size, Image.LANCZOS)
                    if str(mask_mode).startswith("transparency"):
                        m = ImageChops.invert(m)
                    car.putalpha(m)
        elif car_layer:
            p = _resolve(angle_file, "car_angles")
            if p:
                car = _open_layer(p, "RGBA")
        # an opaque source (studio sweep shot) becomes a real cutout here, so the
        # shipped angle plates and a hand-dropped one behave the same way
        if car is not None and car.mode != "RGBA":
            car = car.convert("RGBA")
        if car is not None and bool(key_studio_bg) and car.getchannel("A").getextrema() == (255, 255):
            car = _key_studio_bg(car.convert("RGB"))
        if car is None:
            self._stash(uid, "out", bg)
            return (_pil2t(bg), _pil2mask(Image.new("L", (W, H), 0)))

        self._stash(uid, "car", car)
        car = _trim_alpha(car)

        # --- transform --------------------------------------------------------
        target_w = max(1, int(round(float(scale) * W)))
        target_h = max(1, int(round(car.height * (target_w / float(car.width)))))
        cut = car.resize((target_w, target_h), Image.LANCZOS)
        if flip_h:
            cut = cut.transpose(Image.FLIP_LEFT_RIGHT)
        if abs(float(rotation)) > 0.01:
            # PIL rotates counter-clockwise for positive angles; the UI dial is CW positive.
            cut = cut.rotate(-float(rotation), resample=Image.BICUBIC, expand=True)
        if float(feather) > 0:
            a = cut.getchannel("A").filter(ImageFilter.GaussianBlur(float(feather)))
            cut.putalpha(a)

        cx = int(round(float(x) * W))
        cy = int(round(float(y) * H))
        px = cx - cut.width // 2
        py = cy - cut.height // 2

        # --- contact shadow ---------------------------------------------------
        if float(shadow) > 0.001:
            blur = max(6.0, W * 0.012)
            # contact only: fade the silhouette shadow in over the lower part of the cutout
            grad = Image.new("L", cut.size, 0)
            gd = ImageDraw.Draw(grad)
            top = int(cut.height * 0.55)
            for i, yy in enumerate(range(top, cut.height)):
                val = int(255 * (i / max(1, cut.height - top)))
                gd.line([(0, yy), (cut.width, yy)], fill=val)
            contact = ImageChops.multiply(cut.getchannel("A"), grad)
            s_alpha = contact.filter(ImageFilter.GaussianBlur(blur))
            smask = Image.new("L", (W, H), 0)
            smask.paste(s_alpha, (px, py + int(round(float(shadow_offset) * H))))
            # pool a soft ellipse under the bottom edge as well
            ell = Image.new("L", (W, H), 0)
            d = ImageDraw.Draw(ell)
            ew = int(cut.width * 0.62)
            eh = max(6, int(cut.height * 0.075))
            ex = px + cut.width // 2
            ey = py + cut.height - int(cut.height * 0.03) + int(round(float(shadow_offset) * H))
            d.ellipse([ex - ew // 2, ey - eh // 2, ex + ew // 2, ey + eh // 2], fill=255)
            ell = ell.filter(ImageFilter.GaussianBlur(max(4.0, eh * 1.2)))
            smask = Image.composite(ell, smask, ell)
            strength = float(shadow)
            smask = smask.point(lambda v: int(v * strength * 0.8))
            bg = Image.composite(Image.new("RGB", (W, H), (0, 0, 0)), bg, smask)

        # --- final composite --------------------------------------------------
        frame = bg.copy()
        frame.paste(cut, (px, py), cut)

        cover = Image.new("L", (W, H), 0)
        cover.paste(cut.getchannel("A"), (px, py))

        # what the frozen frame actually shows, for the auto prompt box downstream:
        # only named when it came from the node's own selectors (a wired IMAGE has no name)
        car_name = _pretty(angle_file) if (car is not None and not car_image) else ""
        env_name = _pretty(env_file) if src_bg is not None else ""

        self._stash(uid, "out", frame)
        return (_pil2t(frame), _pil2mask(cover), car_name, env_name)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # any widget tweak must re-run: the placement IS the input.
        return float("NaN")

    @classmethod
    def VALIDATE_INPUTS(cls, **kwargs):
        return True


class FusionPromptBox:
    """Auto prompt box: fills the klein/fusion retouch prompt from the canvas picks."""

    TEMPLATE = (
        "Photoreal automotive retouch of this exact frame. Blend the car into the environment it "
        "stands in: match the ambient light direction, colour temperature and intensity across the "
        "bodywork, carry the scene's reflections over the paint and glass, keep the tyres grounded "
        "with a soft contact shadow, and leave the car's shape, position and angle exactly as placed."
        "\nSubject: {car}. Environment: {env}."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # the box itself: edit freely, {car} and {env} are filled from the canvas
                "shot": ("STRING", {"multiline": True, "default": cls.TEMPLATE}),
                "extra": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "car_name": ("STRING", {"forceInput": True}),
                "env_name": ("STRING", {"forceInput": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "compose"
    CATEGORY = CATEGORY

    def compose(self, shot, extra="", car_name="", env_name=""):
        car = (car_name or "").strip() or "the car"
        env = (env_name or "").strip() or "the environment it stands in"
        # replace, not str.format: stray braces in the user's text must not raise
        text = (shot or self.TEMPLATE).replace("{car}", car).replace("{env}", env)
        extra = (extra or "").strip()
        if extra:
            text = text.rstrip() + " " + extra
        print("[FusionPromptBox] %s" % text.replace("\n", " "))
        return (text,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # text is the input: any edit (or a new canvas pick) must re-run downstream
        return float("NaN")


NODE_CLASS_MAPPINGS = {"CarPlacementCanvas": CarPlacementCanvas, "FusionPromptBox": FusionPromptBox}
NODE_DISPLAY_NAME_MAPPINGS = {
    "CarPlacementCanvas": "Blend Your Image Canvas",
    "FusionPromptBox": "Fusion Prompt Box (auto from canvas)",
}

# Serve the selection-strip thumbnails from a disk cache; /view would re-encode the
# full-resolution plate on every tile request (measured ~20s for the two strips).
def _load_thumbs():
    import importlib.util
    spec = importlib.util.spec_from_file_location("cpc_thumbs", os.path.join(os.path.dirname(os.path.abspath(__file__)), "thumbs.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


try:
    _load_thumbs().register_route()
except Exception:
    pass


