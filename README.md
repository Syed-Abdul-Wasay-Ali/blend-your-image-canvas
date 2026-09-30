# Blend Your Image Canvas

An interactive canvas inside a ComfyUI node: drop a car cutout on a background plate, drag it,
resize it from the corner handles, rotate it, lock the placement, then run the graph and the
server composites the exact frame you placed. The frozen frame goes straight into a FLUX.2 klein
9B + fusion LoRA pass that blends the car into the environment, with the prompt filled
automatically from the tiles you picked on the canvas.

Two nodes ship here, `CarPlacementCanvas` (the canvas) and `FusionPromptBox` (the auto prompt box
that feeds the klein pass).

## Nodes

### `Blend Your Image Canvas` (class `CarPlacementCanvas`) - category `car placement`

Outputs

- `frame` - the composite (background + placed cutout, with optional contact shadow)
- `car_mask` - the cutout's coverage in frame space, if a downstream stage wants to mask
- `car_name` - label of the ticked car-angle tile (clean text, e.g. `front 3q left`)
- `env_name` - label of the ticked environment tile (clean text, e.g. `valley road`)

Inputs

- `background` (IMAGE, optional) - the plate. Whatever is wired here wins over `env_file`.
- `car_image` (IMAGE, optional) - the cutout. Feeds either RGBA directly or RGB + `car_mask`.
- `car_mask` (MASK, optional) - used when `car_image` is RGB.

Widgets

| widget | meaning |
| --- | --- |
| `env_file` | plate from `input/car_env/`, used when nothing is wired to `background` |
| `angle_file` | cutout from `input/car_angles/`, used when nothing is wired to `car_image` |
| `key_studio_bg` | key a near-white studio sweep out of an opaque plate into alpha |
| `mask_mode` | `transparency (LoadImage)` inverts ComfyUI's 1-alpha mask; `coverage` uses it as-is |
| `x`, `y` | cutout centre, normalised to the frame (0..1, y down) |
| `scale` | cutout width as a fraction of the frame width |
| `rotation` | degrees, positive turns clockwise |
| `flip_h` | mirror the cutout |
| `shadow`, `shadow_offset` | contact shadow strength and its downward offset |
| `feather` | edge softening in pixels on the cutout alpha |
| `canvas_width`, `canvas_height` | frame size when no background is available |
| `ui_state` | JSON the canvas round-trips (transform, `locked`, `car_layer`, `env_layer`); hidden in the UI |

### `Fusion Prompt Box` (class `FusionPromptBox`) - category `car placement`

The auto prompt box. It holds the retouch prompt as an editable multiline widget and fills
`{car}` and `{env}` from the canvas picks, so the klein pass is driven by the tiles ticked on the
canvas instead of a prompt typed by hand.

- `shot` - the prompt, multiline, editable. `{car}` and `{env}` are replaced at run time.
- `extra` - optional extra text appended to the prompt.
- `car_name`, `env_name` (STRING, optional inputs) - wire these from the canvas outputs.
- Output `prompt` (STRING) - goes into the positive `CLIP Text Encode`.

Braces in your own text are safe: the box does a plain replace, not `str.format`. It prints the
composed prompt to the console and always reports changed, so editing it or picking a new tile
re-runs the klein pass.

## Canvas controls

- Two galleries under the canvas: `car environment` (plates from `input/car_env/`) and
  `car angles` (cutouts from `input/car_angles/`). Click a tile to tick it; the tick drives the
  matching widget and the name outputs.
- `+ upload` in a gallery header uploads into that gallery's folder and selects the new file;
  dropping a file on the node does the same (left half = environment, right half = car angle).
- drag the body to move, corner squares to resize proportionally, the blue dot to rotate
  (Shift = 15 degree steps), mouse wheel over the cutout to scale.
- `lock` freezes the placement: the canvas stops taking drags, resizes, rotations and wheel
  zoom, so a finished composite cannot be nudged by accident. It draws a dashed grey frame with
  a `locked` badge and survives a reload.
- `del layer` deletes the car cutout layer (the plate stays); picking a tile brings it back.
  `clear all` deletes both layers. Deleted layers are recorded in `ui_state` and honoured by the
  server, so they hold even when an IMAGE is wired into `background` / `car_image`.
- `result` shows the frame the server produced on the last run, `reset` returns the transform to
  the defaults.

Before the first run the canvas previews the plate and cutout from the dropdowns; after a run the
node shows what actually reached it (the server parks a preview of the incoming plate, cutout and
final frame in `temp/` as `cpc_<node id>_{bg,car,out}.png`), so a plate coming out of an upstream
node appears here with no re-uploading.

Gallery tiles load a cached 176px webp instead of re-encoding the full-resolution plate
(`thumbs.py` serves `/car_placement_canvas/thumb`; the client also reads the cache straight out of
`input/car_thumbs/<folder>/`). Rebuild the cache after adding images with

```
python custom_nodes/blend_your_image_canvas/regen_thumbs.py
```

or restart the server, after which missing thumbs are generated on demand.

## The blend workflow

`examples/blend_your_image_canvas_klein.json` is the shipped graph:

- `Blend Your Image Canvas` freezes the frame and emits `car_name` and `env_name`.
- `Fusion Prompt Box` turns those two names into the retouch prompt.
- `Scale Image to Total Pixels` (1.0 MP) then `VAE Encode` turns the frame into a latent, and
  `Set Reference Latent` hands it to the sampler (this is what makes it an edit, not a fresh gen).
- `Load Diffusion Model` (flux-2-klein-9b) into `Load LoRA` (fusion2250, weight 1.0) is the blend.
- `Get Image Size+` sizes an `Empty Flux2 Latent Image` to the frame.
- `KSampler`: 8 steps, cfg 1.0, euler, simple, denoise 1.0.
- `VAE Decode` into `Save Image` (prefix `car_blend/02_klein_fusion`) and a preview.

Model files, in the matching `models/` folders:

| file | folder |
| --- | --- |
| `flux-2-klein-9b.safetensors` | `models/diffusion_models/` |
| `fusion2250.safetensors` | `models/loras/` |
| `qwen_3_8b_fp8mixed.safetensors` | `models/text_encoders/` |
| `flux2-vae.safetensors` | `models/vae/` |

Load the example workflow and it comes up with the canvas placement, the prompt and the klein
pass already wired.

## Install

```
cd ComfyUI/custom_nodes
git clone https://github.com/Jackloid1/blend-your-image-canvas.git
```

Restart ComfyUI. Put environment plates in `ComfyUI/input/car_env/` and keyed car cutouts in
`ComfyUI/input/car_angles/` (the folders are created on first run).

## Notes

- `IS_CHANGED` always reports changed, so a placement tweak re-runs the node even when nothing
  else in the graph moved.
- The compositing is server-side PIL, so a headless `/prompt` run reproduces the canvas placement
  exactly at full resolution.
- Node code, canvas JS and the thumbnail route are self-contained; there are no third-party node
  dependencies.

## Test

```
python custom_nodes/blend_your_image_canvas/test_node.py
```

composites six frames from `input/car_env/04_valley_road.png` + `input/car_angles/01_front_3q_left.png`
(default settings, moved/rotated, the tensor path with a LoadImage-style mask, `car_layer: false`,
`env_layer: false`, and a deleted layer with an IMAGE wired in) into `%TEMP%/cpc_test`.
