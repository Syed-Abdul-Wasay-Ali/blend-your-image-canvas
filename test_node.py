"""Standalone check for CarPlacementCanvas: no server, no GPU.

Run with the serving install's python, from the ComfyUI root:
  ../venv/Scripts/python.exe custom_nodes/car_placement_canvas/test_node.py
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from PIL import Image  # noqa: E402

from custom_nodes.car_placement_canvas.nodes import CarPlacementCanvas  # noqa: E402

OUT = os.path.join(os.environ.get("TMPDIR") or os.environ.get("TEMP") or ".", "cpc_test")
os.makedirs(OUT, exist_ok=True)

n = CarPlacementCanvas()


def run(tag, **kw):
    args = dict(
        env_file="04_valley_road.png",
        angle_file="01_front_3q_left.png",
        key_studio_bg=True,
        mask_mode="transparency (LoadImage)",
        x=0.5,
        y=0.62,
        scale=0.45,
        rotation=0.0,
        flip_h=False,
        shadow=0.45,
        shadow_offset=0.012,
        feather=1.5,
        canvas_width=1376,
        canvas_height=768,
        ui_state="{}",
        unique_id="999",
    )
    args.update(kw)
    img, mask = n.place(**args)
    print(tag, "frame", tuple(img.shape), "mask", tuple(mask.shape),
          "mask_max", float(mask.max()), "range", float(img.min()), float(img.max()))
    p = os.path.join(OUT, f"{tag}.png")
    Image.fromarray((img[0].numpy() * 255).astype("uint8")).save(p)
    Image.fromarray((mask[0].numpy() * 255).astype("uint8")).save(os.path.join(OUT, f"{tag}_mask.png"))
    return p


# 1. file-driven, the settings the node ships with
run("a_default")
# 2. moved + resized + rotated backwards, no shadow
run("b_moved", x=0.72, y=0.55, scale=0.62, rotation=-7.5, shadow=0.0)
# 3. tensor-driven (IMAGE + MASK inputs), like a connected graph
bg = Image.open("input/car_env/04_valley_road.png").convert("RGB").resize((1376, 768), Image.LANCZOS)
car = Image.open("input/car_angles/01_front_3q_left.png").convert("RGBA")
run(
    "c_tensor",
    env_file="(none)",
    angle_file="(none)",
    background=torch.from_numpy(np.asarray(bg).astype("float32") / 255.0)[None],
    # exactly what LoadImage emits: RGB image + (1 - alpha) mask
    car_image=torch.from_numpy(np.asarray(car.convert("RGB")).astype("float32") / 255.0)[None],
    car_mask=torch.from_numpy(1.0 - np.asarray(car.getchannel("A")).astype("float32") / 255.0)[None],
    x=0.3,
    y=0.72,
    scale=0.3,
    shadow=0.6,
)
# 4. delete layer: the cutout is dropped even though an angle file is still set
run("d_car_deleted", ui_state='{"car_layer": false}')
# 5. clear all: no plate either, so the frame falls back to the empty canvas colour
run("e_all_deleted", ui_state='{"car_layer": false, "env_layer": false}')
# 6. a layer deleted in the UI also wins over a wired-in IMAGE input
run(
    "f_deleted_tensor",
    env_file="(none)",
    angle_file="(none)",
    background=torch.from_numpy(np.asarray(bg).astype("float32") / 255.0)[None],
    car_image=torch.from_numpy(np.asarray(car.convert("RGB")).astype("float32") / 255.0)[None],
    car_mask=torch.from_numpy(1.0 - np.asarray(car.getchannel("A")).astype("float32") / 255.0)[None],
    ui_state='{"car_layer": false}',
)

print("wrote", OUT)
