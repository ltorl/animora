# Animora documentation

**Command-line 3D animation and physically based path-traced rendering. No GUI, ever.**

Animora is an animation package you drive entirely from a terminal. A scene is a plain text file
(JSON, YAML or Python). You describe the geometry, materials, lights, cameras and animation in it,
and Animora produces image sequences and finished videos. Previews are drawn directly in the terminal
using true-colour block characters, so no window is ever opened.

```bash
animora new shot.json --template bounce     # write a starter scene
animora preview shot.json -f 12             # look at frame 12 in the terminal
animora play shot.json                      # low-res flipbook in the terminal
animora render shot.json                    # final frames + shot.mp4
```

What's inside:

* **Renderer.** A native C++17 unidirectional path tracer, plus an Apple Metal **GPU** port of the
  same integrator (`--device gpu`, about 4× faster on an M2).
  * Principled BSDF: diffuse, sheen, GGX metal and dielectric, clearcoat, rough and thin glass with
    Beer absorption, random-walk subsurface scattering, emission and alpha.
  * Next-event estimation with MIS; HDRI importance sampling; physical sky.
  * Volumetric fog; depth of field with polygonal bokeh; motion blur (transform and deformation).
  * Adaptive sampling and an AOV-guided denoiser.
  * AgX, ACES and Filmic tone mapping, plus a full post stack.
* **Animation.** Keyframes with Maya-style tangents, about 40 easing curves, infinity cycling,
  additive and multiplicative layers, and a sandboxed expression language (`"=sin(t*2)*noise(t)"`).
* **Rigging.** Hierarchies, joints, IK (two-bone analytic, FABRIK, CCD; with pole vectors,
  stretch, soft IK and twist), linear and dual-quaternion skinning with automatic weights, blend
  shapes, **scripted weight painting** (brush strokes as JSON, mirror, smooth, prune, heat-map
  previews, weight import/export), **HumanIK-style retargeting** of BVH or scene motion onto any
  biped, and Maya's main constraints (aim, point, orient, parent, scale, motion path, limit,
  copy transforms).
* **Modelling.** 18 parametric primitives, OBJ/PLY import and export, and a non-destructive
  stack of 21 modifiers and deformers (Catmull-Clark subdivision, bend, twist, lattice, wave,
  noise and more).
* **Dynamics.** Rigid bodies, PBD cloth with pins, wind, colliders and **tearing**, particle
  systems with force fields, smoke and fire (native multithreaded solver with **exact mesh
  colliders**), liquids, hair, and jiggle (secondary motion). The solver is deterministic and its results can be
  baked to a cache.
* **Output.** PNG (8 or 16 bit), JPEG, TIFF, BMP, OpenEXR (with AOV layers), Radiance HDR and PFM
  images. Videos in H.264, H.265, ProRes / ProRes 4444, WebM, AV1 or GIF, with audio muxing
  (via ffmpeg).

---

## Table of contents

1. [Installation](#1-installation)
2. [Quick start](#2-quick-start)
3. [Command reference](#3-command-reference)
4. [Scene file format](#4-scene-file-format)
5. [Settings](#5-settings)
6. [Nodes and transforms](#6-nodes-and-transforms)
7. [Geometry: primitives, files, inline meshes, curves](#7-geometry)
8. [Modifiers and deformers](#8-modifiers-and-deformers)
9. [Materials and textures](#9-materials-and-textures)
10. [Cameras](#10-cameras)
11. [Lights](#11-lights)
12. [Environment, fog](#12-environment-and-fog)
13. [Post-processing and colour](#13-post-processing-and-colour)
14. [Animation: keys, tangents, easing, layers](#14-animation)
15. [Expressions](#15-expressions)
16. [Constraints](#16-constraints)
17. [Rigging: joints, IK, skinning, blend shapes, weight painting, retargeting](#17-rigging)
18. [Dynamics: rigid bodies, cloth, particles, volumes, smoke, liquids, hair](#18-dynamics)
19. [Rendering in depth](#19-rendering-in-depth)
20. [Video and image output](#20-video-and-image-output)
21. [Python API](#21-python-api)
22. [Examples](#22-examples)
23. [Comparison with Autodesk Maya](#23-comparison-with-autodesk-maya)
24. [Limitations and tips](#24-limitations-and-tips)
25. [Project layout and tests](#25-project-layout-and-tests)

---

## 1. Installation

Requirements:

* Python ≥ 3.9 with `numpy` and `pillow`.
* A C++17 compiler, such as `clang++` (Xcode command-line tools) or `g++`.
* `ffmpeg` (optional) for video encoding.
* `pyyaml` (optional) for `.yaml` scenes.

One-line install (macOS and Linux). It sets up a private environment in `~/.animora`, links
`animora` into `~/.local/bin` and compiles the renderer; see [README.md](README.md) for its options:

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh
```

Or install from a clone:

```bash
git clone https://github.com/ltorl/animora.git && cd animora
pip install -e .            # installs the `animora` command
animora build               # compile the native renderer (also happens automatically on first render)
animora doctor              # check everything
animora doctor --bench      # ...and run a quick render benchmark
```

Without installing, run it as a module from the repo root:

```bash
python3 -m animora <command> ...
```

`animora doctor` reports the Python, numpy and Pillow versions, the compiler, the renderer status,
ffmpeg and its available encoders, and the CPU thread count.

The renderer is compiled once into `animora/native/build/` with `-O3 -march=native`. It is rebuilt
automatically when `pathtracer.cpp` changes; `animora build --force` forces a rebuild.

---

## 2. Quick start

```bash
animora new ball.json --template bounce
animora info ball.json                  # summary + hierarchy
animora preview ball.json -f 10         # draw frame 10 in the terminal
animora play ball.json --loops 3        # flipbook
animora render ball.json --preview      # fast draft render of the whole range
animora render ball.json                # final quality frames + video (settings.video)
```

A complete minimal scene:

```json
{
  "settings": {"resolution": [1280, 720], "frame_range": [1, 48], "samples": 128,
               "output": "render/ball_####.png", "video": "render/ball.mp4"},
  "environment": {"type": "sky", "sun_elevation": 40},
  "materials": {"red": {"preset": "glossy_plastic", "base_color": "#d63a2f"}},
  "nodes": [
    {"name": "cam",   "type": "camera", "translate": [0, 2, 9], "look_at": "ball", "focal_length": 40},
    {"name": "floor", "type": "mesh", "primitive": "plane", "size": 30, "material": "studio_floor"},
    {"name": "ball",  "type": "mesh", "primitive": "uv_sphere", "radius": 0.5, "material": "red",
     "translate": {"keys": [[1, [0, 3, 0], "ease_in"], [24, [0, 0.5, 0], "ease_out"], [48, [0, 3, 0]]],
                   "post": "cycle"},
     "modifiers": [{"type": "squash", "amount": "=1 - 0.35*pulse(frame, 21, 27)"}]}
  ]
}
```

---

## 3. Command reference

Run `animora <command> --help` for the authoritative list of flags.

### Common render-setting flags

These flags are accepted by `render`, `preview`, `play`, `info`, `validate`, `eval`, `export` and
`bake`. Each one overrides the scene's `settings` without editing the file.

| flag | meaning |
|---|---|
| `--set KEY=VALUE` | Override any settings key. VALUE is parsed as JSON. Repeatable, e.g. `--set samples=256 --set caustics=true --set 'resolution=[640,360]'`. |
| `--res RES` | `WxH` (`1920x1080`) or a preset: `720p`, `1080p`, `4k`, `square`, `vertical`, and so on. |
| `--samples N`, `-s N` | Samples per pixel. |
| `--threads N`, `-t N` | Render threads (0 = all cores). |
| `--device D`, `-d D` | `cpu` (default), `gpu` (Apple Metal) or `auto` (GPU when available). See [§19](#gpu-rendering). |
| `--integrator NAME` | `path` (full GI), `direct`, `ao`, `clay`, `normals`, `albedo`, `depth`, `uv` or `wireframe`. |
| `--max-bounces N` | Path depth. |
| `--seed N` | Sampling seed. |
| `--tonemap NAME` | `agx`, `agx_punchy`, `aces`, `filmic`, `reinhard`, `neutral` or `standard`. |
| `--exposure EV` | Exposure in stops. |
| `--motion-blur` | Enable motion blur. |
| `--no-denoise` | Disable the denoiser. |
| `--transparent` | Transparent background (alpha channel). |

### `animora render SCENE`

Renders a frame range to images and, optionally, a video.

| flag | meaning |
|---|---|
| `-f, --frames` | `12`, `1-48`, `1-48x2` (every 2nd), `1,5,9`, `first,last`. Defaults to `settings.frame_range` and `frame_step`. |
| `-o, --output` | Image pattern, e.g. `out/shot_####.exr` (`#` padding or `%04d`). The extension picks the format. |
| `--video PATH` | Encode a video to PATH (overrides `settings.video`). |
| `--codec C` | `h264`, `h265`, `prores`, `prores4444`, `webm`, `av1`, `gif` or `png`. Defaults to the extension's codec. |
| `--audio FILE` | Mux an audio track into the video. |
| `--no-video` | Do not encode even if `settings.video` is set. |
| `--preview` | Fast draft that uses `preview_scale` and `preview_samples`. |
| `--scale X` | Resolution multiplier (0.5 = half). |
| `--resume` | Skip frames whose output already exists, so an interrupted render can be continued. |
| `--show` | Draw each finished frame in the terminal; `--show-width` sets its column width. |
| `--log FILE` | Write a JSON render log (per-frame timings and statistics). |
| `-q` / `-v` | Quiet / verbose. |

```bash
animora render shot.json -f 1-120 --res 1080p -s 256 -o out/shot_####.png --video out/shot.mov --codec prores
animora render shot.json -f 1-48x4 --preview --show      # quick blocking pass, see frames as they finish
animora render shot.json --resume                        # continue after Ctrl-C
```

Ctrl-C stops cleanly after the current frame.

### `animora preview SCENE`

Renders a single frame and draws it in the terminal. By default it uses preview scale and samples.

* `-f, --frame N` selects the frame (default: the first).
* `--full` uses full resolution and full samples.
* `--scale X` sets the resolution multiplier.
* `-w, --width COLS` sets how many terminal columns to use.
* `--save PATH` also writes the image to disk.

### `animora play SCENE`

Renders a low-resolution flipbook, then plays it in the terminal at the scene fps.
Flags: `-f/--frames`, `--fps`, `--scale`, `-w/--width`, `--loops N`.

### `animora view IMAGE [IMAGE ...]`

Draws image files (PNG, JPEG, TIFF, EXR, HDR, PFM and others) in the terminal.
HDR images are tone mapped with `--tonemap` and `--exposure`. `-w` sets the width.

### `animora info SCENE`

Prints the scene summary (settings, cameras, lights, materials and node counts) and the node
hierarchy. `--json` prints machine-readable output. `--stats` also builds frame 1 and reports
triangle, instance and light counts.

### `animora validate SCENE`

Loads the scene and evaluates a few frames, or the frames given with `-f`. It reports errors such
as unknown materials, bad references, broken expressions or IK problems. The exit code is non-zero
on failure, which makes it usable in CI or before a long render.

### `animora eval SCENE CHANNEL [CHANNEL ...]`

Prints animated values per frame, which is the command-line equivalent of a Graph Editor.

```bash
animora eval shot.json ball.translate.y ball.wpos cam.focus_distance -f 1-24
animora eval shot.json arm.world_rotation --json --precision 6
```

Any `node.attribute[.component]` path works; see [Reading values](#reading-values-with-get).

### `animora new PATH`

Writes a starter scene. `--template` is one of `basic` (default), `bounce`, `turntable` or
`empty`. `--force` overwrites an existing file. The output format follows the extension
(`.json`, `.yaml` or `.py`).

### `animora encode PATTERN OUTPUT`

Encodes an existing image sequence into a video with ffmpeg.

```bash
animora encode "render/frame_####.png" out.mp4 --fps 24 --quality high
animora encode "render/frame_####.exr" out.mov --codec prores4444 --alpha
```

| flag | meaning |
|---|---|
| `--fps` | Frame rate (default 24). |
| `--start N` | First frame number (auto-detected by default). |
| `--codec` | `h264`, `h265`, `prores`, `prores4444`, `webm`, `av1`, `gif` or `png`. |
| `--quality` | `draft` (CRF 28), `low` (26), `medium` (22), `high` (18, default), `best` (14) or `lossless` (0). |
| `--crf N` | Explicit CRF (overrides `--quality`). |
| `--audio FILE` | Mux audio; the video is trimmed to the shortest stream. |
| `--alpha` | Keep alpha (ProRes 4444, WebM, PNG). |
| `--scale` | Output width (`1280`) or an ffmpeg scale expression. |
| `-v` | Show ffmpeg output. |

Sequences with gaps (for example frames rendered with `1-48x4`) are handled automatically. They are
linked into a contiguous temporary sequence before encoding.

### `animora build`

Compiles the native renderer and, on macOS, the Metal GPU renderer (it prints the Metal device
it found). `--force` rebuilds; `-v` shows the compiler command. Both libraries are also built
automatically the first time they are needed, and rebuilt whenever their sources change.

### `animora retarget SOURCE TARGET`

HumanIK-style retargeting: transfers motion from a BVH file (or a scene's skeleton) onto the joint
hierarchy of another scene, and writes a new scene with the target joints keyed. See
[§17 Retargeting](#retargeting-humanik-style).

| flag | meaning |
|---|---|
| `-o, --output` | Output scene (default `TARGET_retargeted.json`). |
| `--target-root` / `--source-root` | Root joint of the target / source skeleton (when a scene holds several). |
| `--map` | JSON object or file `{target_joint: source_joint}` overriding parts of the automatic mapping. |
| `--list-map` | Print the joint mapping (with the HumanIK slot of each joint) and exit. |
| `--scale` | BVH unit scale (default 0.01: centimetres to metres). |
| `--start` | First frame of the retargeted motion. |
| `--yaw` | Rotate the source about +Y by this many degrees (default: detect the facing automatically). |
| `--height-scale` | Root-motion scale (default: target hip height / source hip height). |
| `--in-place` | Remove horizontal root travel. |
| `--no-root-motion` | Keep the hips at their rest position. |
| `--no-match-rest` | Do not compensate for different rest poses (T-pose vs A-pose). |
| `--foot-lock` | Pin planted feet with two-bone leg IK to remove foot sliding. |

```bash
animora retarget examples/data/walk_cycle.bvh mannequin.json --list-map
animora retarget examples/data/walk_cycle.bvh mannequin.json -o walk_on_mannequin.json --foot-lock
```

### `animora weights SCENE [NODE]`

Inspects skin weights after automatic weights and every `weight_paint` stroke have been applied.
NODE defaults to the first skinned mesh. With no other action it prints per-joint statistics.

| flag | meaning |
|---|---|
| `-j, --joint` | Joint to visualise as a blue-to-red heat map (default: every influence in its own colour). |
| `-o, --output` | Write the heat map image (PNG, EXR, ...). |
| `--show` | Draw the heat map in the terminal (`--width` columns). |
| `--frame` | Show the mesh posed at this frame (default: the bind pose). |
| `--res` | Image size `WxH` (default 640x480). |
| `--camera` | View through a scene camera instead of the automatic framing. |
| `--view` | Automatic camera direction: `front`, `back`, `left`, `right` or `top`. |
| `--export` | Write the final weights to JSON, ready to load back with `"weights": "file.json"`. |
| `--list` | Per-joint statistics: vertices influenced, maximum and total weight. |
| `--set` | Override a setting (`key=value`). |

```bash
animora weights examples/weight_paint.json --list
animora weights examples/weight_paint.json -j l_elbow --show
animora weights examples/weight_paint.json -j l_elbow -o elbow_weights.png --res 1280x720
animora weights examples/weight_paint.json --frame 16 -o posed.png        # all influences, posed
animora weights examples/weight_paint.json --export span_weights.json
```

### `animora export SCENE OUT` (alias `export-obj`)

The output extension picks what is exported.

**Geometry (`.obj` with `.mtl`, `.ply`).** Writes the fully evaluated geometry: all modifiers,
skinning, blend shapes, cloth, particles, hair and liquid surfaces.

* `-f, --frame N` exports one frame.
* `--frames 1-96` (or `1-96x2`) with `####` in the name writes a numbered sequence:
  `animora export shot.json cache/cloth_####.obj --frames 1-96`.
* `--nodes a b c` limits the export to those nodes and their children.
* `--separate` writes one file per node, with OUT as a directory. `--format .obj|.ply` sets the
  format for `--separate`.
* `--hidden` includes invisible nodes.

**Animated scene (`.glb`, `.gltf`, `.usda`).** Writes the whole shot for other DCCs and game
engines (Blender, Maya, Houdini, Unreal, Unity, Omniverse, three.js, usdview).

* `--frames 1-120` sets the sampled range. The default is the scene's frame range, sampled every
  `--step` frames (default 1).
* `-q` silences progress output.

| content | glTF / GLB | USD (`.usda`) |
|---|---|---|
| Hierarchy and transforms | TRS channels sampled per frame | `xformOp:transform` time samples (flat world matrices) |
| Rigid meshes | Mesh, UVs and normals, written once | `Mesh` prim with `primvars:st`, normals and material `subsets` |
| Deforming meshes (skin, blend shapes, cloth, hair) | One morph target per frame, driven by weights | `points` / `normals` time samples |
| Changing topology (liquid) | One mesh per frame, switched by step-scaled keys | One child per frame, switched by `visibility` |
| Materials | PBR metallic-roughness, KHR transmission/IOR/clearcoat/sheen/emissive strength, embedded base-colour textures | `UsdPreviewSurface` and `UsdUVTexture` |
| Cameras and lights | Perspective cameras, `KHR_lights_punctual` | `Camera` (focal length, aperture, focus), `UsdLux` Sphere/Disk/Rect/Distant lights |

USD files validate with Pixar's `usdchecker`. Units are metres, Y is up, and USD time codes are
Animora frame numbers.

```bash
animora export shot.json shot.glb              # whole frame range
animora export shot.json shot.usda --frames 1-48x2
```

### `animora import FILE`

Converts another format into an Animora scene file (`-o out.json`; by default the file is written
next to the input).

| input | result |
|---|---|
| `.glb` / `.gltf` | Node hierarchy, meshes (referenced as `file.glb#Node`), materials, cameras, keyframed TRS animation. Morph targets become blend shapes. |
| `.bvh` | Motion capture: one `joint` per BVH joint with the file's rotation order, keyed root translation and joint rotations, plus display bones (a sphere per joint, a cylinder per bone). |
| `.obj` / `.ply` | A scene that references the mesh. |

| flag | meaning |
|---|---|
| `--fps N` | Scene frame rate. The default is 24, or 1 / `Frame Time` for BVH. BVH motion is retimed to the scene rate. |
| `--scale S` | BVH unit scale (default 0.01: centimetres to metres). |
| `--joint-radius R` | BVH display-bone radius (0 = joints only, no display geometry). |
| `--camera` | Adds a framing camera, plus a sun and a ground plane if the scene has no lights, so the result renders immediately. |

```bash
animora import walk.bvh -o walk.json --camera && animora play walk.json
animora import robot.glb --camera && animora render robot.json --preview
```

The imported scene is ordinary JSON. Add a skinned character mesh whose `skin.root` is the BVH
hips joint, and the mocap drives it.

### `animora bake SCENE`

Runs every simulation and writes the caches:

* rigid bodies, cloth, particles and jiggle go into one `.npz` file. `-o` sets the path (default
  `<scene>.cache.npz`) and `--end` the last frame. Point `dynamics.cache` at that file.
* smoke `volume` nodes and `liquid` nodes are simulated over the frame range and written to the
  `cache` path in their `smoke` block or liquid node. When no `cache` is set, the file is
  `<scene>.<node>.<smoke|liquid>.npz`; add it as `cache` to reuse it at render time.

Renders then load the caches instead of re-simulating, so renders on many machines or in many
processes share an identical simulation.

### `animora graph SCENE [CHANNELS ...]`

A graph editor without a window. It plots animation curves in the terminal (colour braille
plots with key markers, a value axis and a frame ruler), or writes a PNG with a grid, curves,
keys, tangent handles and a current-frame line.

```bash
animora graph shot.json                          # every animated transform
animora graph shot.json ball.translate.y ball.scale -f 1-48 --keys
animora graph shot.json arm.rotate -o arm_curves.png --size 1600x800 --show
```

| flag | meaning |
|---|---|
| `-f 1-120` | Frame range. |
| `-o FILE.png` | Write a PNG instead of a terminal plot. Add `--terminal` to get both. |
| `--size WxH` | PNG size. |
| `-w`, `--height` | Terminal plot size. |
| `--normalize` | Scale every curve to 0..1, like Maya's normalize view. |
| `--keys` | Also print the key values. |
| `--hide-flat` | Hide components that never change. |
| `--current F` | Draw a current-frame marker. |
| `--min`, `--max` | Fix the value axis. |
| `--light` | Light PNG theme. |
| `--title` | Plot title. |
| `--show` | Draw the PNG in the terminal. |

Channels are `node.attr` or `node.attr.x|y|z`, any animated attribute (material parameters,
modifier amounts and so on included).

### `animora key ACTION SCENE CHANNEL [VALUE]`

Edits keyframes in a `.json` scene from the command line. Only the edited channel changes; the
rest of the file stays exactly as written.

| action | effect |
|---|---|
| `set` | Add or replace a key at `-t F`. VALUE defaults to the channel's current value, so this works like Maya's "set key". `-i` sets its interpolation. |
| `delete` | Remove the key at `-t F`. |
| `move` | Move the key at `-t F` to `--to F2`. |
| `interp` | Set the interpolation (`-i linear|step|bezier|auto|ease_in_out|...`) of the key at `-t F`, or of every key when `-t` is omitted. |
| `tangent` | Set the `--in` / `--out` slopes (value per frame) of the key at `-t F`. |
| `infinity` | Set `--pre` / `--post` to `constant`, `linear`, `cycle`, `cycle_offset` or `oscillate`. |
| `clear` | Remove the keys, keeping the value at `-t F` (default: the first key) as a constant. |

```bash
animora key set shot.json ball.translate.y 2.5 -t 12 -i ease_in_out
animora key move shot.json ball.translate.y -t 12 --to 14
animora key infinity shot.json wheel.rotate.z --post cycle_offset
```

### `animora keys SCENE [CHANNELS ...]`

Lists the keyframes (frame, value and interpolation) of the given channels, or of every animated
transform. It also shows each channel's kind (curve, per-component, driven) and, for driven keys,
the driver.

### `animora colorspaces`

Lists the supported colour spaces (see [section 13](#13-post-processing-and-colour)).
`--matrix --from lin_srgb --to acescg` prints the 3×3 conversion matrix, and
`--convert 1 0.5 0.2 --from srgb --to acescg` converts one colour.

### `animora turntable MODEL`

Builds and renders a 360° studio turntable for an `.obj` or `.ply` model. The model is
auto-framed, placed on a floor with a three-point studio environment, and encoded as a video.

| flag | meaning |
|---|---|
| `--out-dir DIR` | Output directory (default `turntable/`). |
| `--frames N` | Frames per revolution. |
| `--fps N` | Frame rate. |
| `--samples N` | Samples per pixel. |
| `--res RES` | Resolution. |
| `--material NAME` | Material preset for the model (default `clay`). |
| `--keep-materials` | Use the model's `.mtl` instead. |
| `--transparent` | Alpha background. |
| `--no-render` | Only write the generated scene file. |

### `animora materials [NAME]`

Lists all material presets, or prints the full parameters of one.

### `animora primitives`

Lists every mesh primitive with its parameters and defaults.

### `animora curves [NAME ...]`

Lists easing curves. `--plot` draws them as ASCII graphs; `--width` and `--height` set the plot
size.

### `animora doctor`

Checks the environment (numpy, Pillow, compiler, native renderer, Metal GPU, ffmpeg).
`--bench` also times a small render on the CPU and, when available, on the GPU.

---

## 4. Scene file format

A scene is a dictionary with these top-level keys (all optional):

| key | purpose |
|---|---|
| `settings` | Render and timeline settings ([§5](#5-settings)). |
| `nodes` | The scene graph: a list of node specs, or a `{name: spec}` dict. Children can nest under `children`. |
| `materials` | Named materials ([§9](#9-materials-and-textures)). |
| `environment` | World lighting: sky, HDRI, gradient, studio or colour ([§12](#12-environment-and-fog)). |
| `fog` | Volumetric fog. |
| `post` | Colour grading and lens effects ([§13](#13-post-processing-and-colour)). |
| `camera` | Name of the active camera. |
| `camera_switches` | `[[frame, "camName"], ...]`: cut between cameras over time. |
| `ik` | IK handles ([§17](#17-rigging)). |
| `dynamics` | Global simulation settings ([§18](#18-dynamics)). |
| `variables` | User variables available in every expression, e.g. `{"speed": 2.5}`. Values may themselves be animated. |
| `include` / `includes` | A path or list of paths to other scene files merged in first. Use it for shared material libraries, rigs or lighting setups. |

Keys called `comment` are ignored anywhere, so you can annotate freely.

### File formats

* **JSON** (`.json`): the canonical format.
* **YAML** (`.yaml` / `.yml`): the same structure; needs `pyyaml`.
* **Python** (`.py`): the file defines a dict `scene = {...}`, or a function `build()` that returns
  a dict or a `Scene`. The namespace pre-imports `animora` (also as `an`), and `math` and `numpy`
  are importable. This is the procedural / "MEL-script" route: loops, generated crowds, data-driven
  animation.

```python
# crowd.py
import math
nodes = [{"name": "cam", "type": "camera", "translate": [0, 6, 18], "look_at": [0, 0, 0]}]
for i in range(60):
    a = i / 60 * math.tau
    nodes.append({"name": f"c{i}", "type": "mesh", "primitive": "capsule", "radius": 0.2, "height": 1,
                  "translate": [6 * math.cos(a), f"=0.5 + abs(sin(t*3 + {i}))", 6 * math.sin(a)],
                  "material": f"hsv({i/60}, 0.7, 0.9)"})
scene = {"settings": {"frame_range": [1, 72]}, "environment": {"type": "studio"}, "nodes": nodes}
```

### Units and axes

The coordinate system is right-handed and Y-up. Units are scene units (think metres); gravity is
-9.81. Rotations are in degrees and use Euler order `xyz` by default (configurable per node).
Cameras and lights look down their local −Z axis.

---

## 5. Settings

| key | default | description |
|---|---|---|
| `resolution` | `[960, 540]` | Output size in pixels. |
| `fps` | `24` | Frames per second (also drives `t` in expressions). |
| `frame_range` | `[1, 48]` | First and last frame. |
| `frame_step` | `1` | Render every n-th frame by default. |
| `samples` | `64` | Samples per pixel (maximum when adaptive). |
| `min_samples` | `0` | Minimum samples before adaptive termination. |
| `adaptive_threshold` | `0` | Adaptive-sampling noise threshold (0 = off; try 0.01). |
| `max_bounces` | `8` | Maximum path depth. |
| `clamp_indirect` | `10` | Firefly clamp on indirect contributions (0 = off). |
| `caustics` | `false` | Allow caustic paths (diffuse → specular → light). Off uses path regularisation for clean renders. |
| `filter_width` | `1.5` | Gaussian pixel filter width in pixels. |
| `integrator` | `"path"` | `path`, `direct`, `ao`, `clay`, `normals`, `albedo`, `depth`, `uv` or `wireframe`. |
| `ao_distance` | `1.0` | Ray length for the `ao` integrator. |
| `wire_width` | `1.0` | Line width for the `wireframe` integrator. |
| `tonemap` | `"agx"` | View transform ([§13](#13-post-processing-and-colour)). |
| `exposure` | `0` | Exposure in stops, or `"auto"`. |
| `film_transparent` | `false` | Render the background as alpha 0. |
| `seed` | `1` | Sampling seed. |
| `threads` | `0` | 0 = all cores. |
| `device` | `"cpu"` | `cpu`, `gpu` (Apple Metal) or `auto`. Scenes with volumes (fog boxes, smoke, fire) always render on the CPU. |
| `motion_blur` | `false` | Transform and deformation motion blur. |
| `shutter` | `0.5` | Shutter length in frames (0.5 = 180° shutter). |
| `shutter_position` | `"center"` | `center`, `start` or `end`. |
| `denoise` | `true` | Edge-aware, AOV-guided denoiser. |
| `denoise_iterations` | `5` | À-trous passes. |
| `denoise_strength` | `1.0` | Blend between noisy (0) and denoised (1). |
| `bit_depth` | `8` | 8 or 16 for PNG and TIFF. |
| `output` | `"render/frame_####.png"` | Image pattern (relative to the scene file). |
| `video` | `null` | If set, encode a video after rendering (`"render/shot.mp4"`). |
| `video` (dict form) | – | `{"path": "render/shot.mov", "codec": "prores", "quality": "best", "crf": 16, "audio": "music.wav", "alpha": true, "scale": 1280, "fps": 24}`. Every key except `path` is optional. |
| `aovs` | `[]` | Extra passes: `albedo`, `normal`, `depth`, `variance`, `spp`, `alpha`, `noisy`, `position`. Written as EXR layers, or as a sidecar `.aovs.exr` for LDR outputs. |
| `preview_scale` | `0.5` | Resolution factor for `--preview`, `preview` and `play`. |
| `preview_samples` | `8` | Samples for previews. |
| `unit_scale` | `1.0` | Global scale applied to the scene. |

Any setting can be overridden from the command line with `--set key=value`.

---

## 6. Nodes and transforms

Every object is a node. Common keys:

| key | description |
|---|---|
| `name` | Unique name (auto-generated if omitted). |
| `type` | `group` / `locator` (default), `mesh`, `camera`, `light`, `joint`, `curve`, `particles`, `cloth` or `instance`. |
| `parent` | Parent node name. Nested `children: [...]` also works. |
| `translate` / `position` | `[x, y, z]`, animatable. |
| `rotate` / `rotation` | Euler degrees, animatable. |
| `scale` | Scalar or `[x, y, z]`, animatable. |
| `pivot` | Rotate and scale pivot in local space. |
| `rotate_order` | `xyz` (default), `xzy`, `yxz`, `yzx`, `zxy` or `zyx`. |
| `visible` | `true`/`false` or animated (step), e.g. `{"keys": [[1, 1], [30, 0, "step"]]}`. |
| `camera_visible` | `false` hides the object from the camera while it still casts shadows and reflects. |
| `inherit_transform` | `false` ignores the parent's transform. |
| `look_at` | A node name or point: shorthand for an aim constraint. A dict gives aim-constraint options. |
| `constraints` | A list of constraints ([§16](#16-constraints)). |
| `attributes` | Custom animatable attributes, e.g. `{"happiness": {"keys": [[1, 0], [48, 1]]}}`. Read them with `get("node.happiness")` and use them to drive anything. |
| `shake` | Procedural noise shake: `{"amplitude": 0.5, "frequency": 1.5, "seed": 0, "translate": 0.0}` (degrees of rotation noise, plus optional translation noise). Great for handheld cameras. |
| `rigid_body`, `jiggle` | Dynamics ([§18](#18-dynamics)). |
| `tags` | Free-form labels. |

The transform order is translate · pivot · rotate · scale · −pivot, multiplied by the parent's
world matrix (as in Maya). After that come constraints, then IK, then dynamics.

**Instances.** A node of type `instance` with `"source": "otherNode"` reuses another mesh with its
own transform and an optional material override. It is cheap in memory and handy for forests and
crowds.

---

## 7. Geometry

### Primitives

A mesh node sets `"primitive"`. Its parameters go inline or in `"params"`, and every numeric
parameter is animatable.

| primitive | parameters (defaults) |
|---|---|
| `box` / `cube` | `size` 1, `width`, `height`, `depth`, `smooth_angle` 30 |
| `sphere` / `uv_sphere` | `radius` 1, `segments` 32, `rings` 16 |
| `ico_sphere` / `icosphere` | `radius` 1, `subdivisions` 3 |
| `plane` | `size` 1, `subdivisions` 1, `width`, `depth`, `uv_scale` 1 |
| `grid` | `size` 10, `subdivisions` 10 |
| `disc` | `radius` 1, `segments` 32, `rings` 1 |
| `cylinder` | `radius` 0.5, `height` 1, `segments` 32, `caps` true, `height_segments` 1, `radius_top`, `smooth_angle` 40 |
| `cone` | `radius` 0.5, `height` 1, `segments` 32, `caps` true |
| `capsule` | `radius` 0.5, `height` 2, `segments` 32, `rings` 8 |
| `torus` | `major_radius` 1, `minor_radius` 0.25, `major_segments` 48, `minor_segments` 16 |
| `terrain` | `size` 10, `resolution` 64, `height` 1.5, `frequency` 0.25, `octaves` 5, `seed` 0, `ridged` false |
| `lathe` | `profile` [[r, y], ...], `segments` 32, `angle` 360, `smooth_angle` 60: a surface of revolution (vases, bottles) |
| `extrude` | `points2d` [[x, y], ...], `depth` 0.2: an extruded 2D outline |
| `tube` | `path` [[x, y, z], ...], `radius` 0.1, `segments` 12, `closed`, `caps`: a pipe along a path |
| `heart` | `size` 1 |

`animora primitives` prints this table from the code.

### Mesh files

```json
{"name": "statue", "type": "mesh", "file": "assets/statue.obj", "smooth_angle": 50}
```

Supported formats:

* OBJ, with `.mtl` materials, UVs, normals and n-gons.
* PLY, ASCII or binary.
* glTF 2.0 (`.gltf` with embedded or external buffers, `.glb`). Without a selector, every mesh in
  the default scene is merged in world space.

File materials become material slots; override them with `material` or `materials`.

**Selecting one glTF mesh.** `"file": "robot.glb#Arm"` loads only the node (or mesh) named `Arm`,
in its own local space, so it can be parented and animated like any node.
`"robot.glb#Face@2"` loads morph target 2 of that mesh as absolute positions; `animora import`
uses this to turn glTF morph targets into blend shapes.

**File sequences (Alembic-style caches).** A run of `#` in the file name is replaced by the
frame number, padded to the run's length:

```json
{"name": "sim", "type": "mesh", "file": "cache/cloth_####.obj", "file_offset": -100, "file_range": [101, 196]}
```

* `file_offset` shifts the frame that is loaded.
* `file_range: [first, last]` clamps it, holding the first or last file outside the range.

Point caches exported from other packages, or Animora's own
`export --frames ... cloth_####.obj`, play back this way, with motion blur.

### Inline meshes

```json
{"type": "mesh", "mesh": {"verts": [[0,0,0],[1,0,0],[1,1,0],[0,1,0]], "faces": [[0,1,2,3]],
                          "uvs": [[0,0],[1,0],[1,1],[0,1]], "uv_indices": [[0,1,2,3]],
                          "face_materials": [0], "smooth_angle": 30}}
```

### Shading

* `smooth_angle` (degrees) splits normals at edges sharper than the angle; 180 gives fully smooth
  shading and 0 gives faceted.
* `material` assigns one material.
* `materials: ["a", "b"]` assigns per-slot materials, using face material indices from the OBJ or
  `face_materials`.

### Procedural modelling

These primitives build new shapes from code. Their results are ordinary meshes, so they take
materials, modifiers, skinning, cloth and export like any other.

#### Signed-distance CSG (`primitive: "sdf"`)

```json
{"name": "part", "type": "mesh", "primitive": "sdf", "resolution": 96, "material": "aluminium",
 "shapes": [
   {"type": "box", "size": [2, 1, 1], "round": 0.08},
   {"type": "cylinder", "radius": 0.3, "height": 2, "rotate": [90, 0, 0], "op": "subtract"},
   {"type": "sphere", "radius": 0.45, "translate": [1, 0.5, 0], "blend": 0.25},
   {"type": "text", "text": "A1", "size": 0.3, "depth": 0.1, "translate": [-0.6, 0.1, 0.5], "op": "subtract"}
 ]}
```

`shapes` is evaluated in order. Each shape combines with everything before it:

* `op`: `union` (default), `subtract` or `intersect`.
* `blend`: a smooth-blend radius (0 gives sharp CSG). Use it for fillets, organic joins and
  melted shapes.

Each shape takes `translate`, `rotate` (degrees) and `scale`.

| shape | parameters |
|---|---|
| `sphere` | `radius` |
| `box` | `size` [x, y, z] |
| `torus` | `major_radius`, `minor_radius` |
| `capsule` | `radius`, `height` |
| `cylinder` | `radius`, `height` |
| `cone` | `radius`, `height`, `radius_top` |
| `ellipsoid` | `radii` [x, y, z] |
| `octahedron` | `radius` |
| `plane` | `normal`, `offset` (a half-space, useful for cutting) |
| `text` | `text`, `size`, `depth`, `font`, `bevel`, `align` |
| `mesh` | `node` (another mesh node, converted to an exact SDF), or `verts` and `tris` |
| `metaballs` | `balls`, `threshold` |
| `group` | A nested `shapes` list, which gives a full CSG tree |

Shaping keys, which work on any shape:

| key | effect |
|---|---|
| `round` | Inflate the shape and round its edges by this radius. |
| `shell` | Hollow the shape, with this wall thickness. |
| `displace` | `{amplitude, frequency, octaves, seed}`: noise displacement. |
| `twist` | Degrees per unit along Y. |
| `bend` | Degrees per unit along X, around Z. |
| `mirror` | `"x"`, `"xz"`, ...: mirror the shape across those axes. |
| `repeat` | Cell spacing [x, y, z]; 0 means no repetition on that axis. `count` limits the number of copies. |

The surface is extracted with surface nets on a grid of `resolution` cells along the longest
axis (default 64). Each vertex is then projected exactly onto the surface, so curved shapes stay
smooth at moderate resolutions. Other keys:

* `bounds`: a fixed [[min], [max]] meshing box (automatic by default).
* `smooth_angle` (default 60).
* `project`: the number of Newton projection steps (default 3).

#### Other procedural primitives

| primitive | parameters (defaults) |
|---|---|
| `metaballs` | `balls` [{`center`, `radius`, `strength`}] (negative strength subtracts), `threshold` 0.5, `resolution` 64 |
| `text` | `text`, `size` 1 (cap height), `depth` 0.2, `bevel` 0.02, `font` (font file or system font name), `align` (left, center or right), `line_spacing` 1.2. The text faces +Z; multi-line text is supported. |
| `sweep` | `path` [[x,y,z], ...], `profile` (`"circle"`, `"square"`, `"star"` or a 2D point list), `radius` 0.1, `segments` 16, `samples` 8, `smooth`, `closed`, `caps`, `twist` (degrees over the path), `scale_start` / `scale_end` (taper), `up`. Frames are parallel-transported, so the profile never flips. |
| `loft` | `sections`: lists of 3D points, or `{"profile", "radius", "translate", "rotate", "scale"}` dicts. The sections are resampled to equal point counts and skinned with Catmull-Rom (`smooth`, `samples` rings between sections, `closed`, `caps`). |

```json
{"name": "logo", "type": "mesh", "primitive": "text", "text": "ANIMORA", "size": 0.5, "depth": 0.15, "material": "gold"},
{"name": "horn", "type": "mesh", "primitive": "sweep", "profile": "circle", "radius": 0.2, "scale_end": 0.05,
 "twist": 180, "path": [[0,0,0],[0,1,0.3],[0.4,1.6,1]]},
{"name": "vase", "type": "mesh", "primitive": "loft", "sections": [
   {"profile": "circle", "radius": 0.4}, {"radius": 0.7, "translate": [0, 0.6, 0]},
   {"radius": 0.25, "translate": [0, 1.4, 0]}, {"radius": 0.35, "translate": [0, 1.7, 0]}]}
```

The modelling modifiers `boolean`, `remesh`, `decimate` and `extrude` are listed in
[section 8](#8-modifiers-and-deformers).

### Curves

```json
{"name": "rail", "type": "curve", "points": [[0,0,0],[2,1,0],[4,0,2],"handleNode"],
 "smooth": true, "closed": false, "samples": 16, "radius": 0.02, "segments": 8, "material": "chrome"}
```

* Points may be vectors or node names, so the curve follows animated nodes.
* `smooth` makes a Catmull-Rom curve.
* `radius > 0` renders the curve as a tube; `radius: 0` makes it an invisible guide, for example
  for motion paths.

---

## 8. Modifiers and deformers

A mesh takes a `"modifiers"` list that is evaluated top to bottom every frame. Every numeric
parameter is animatable. Axis parameters take `"x"`, `"y"` or `"z"`.

| type | parameters (defaults) |
|---|---|
| `subdivide` | `levels` (1). Catmull-Clark subdivision. `render_levels` applies only to final renders. |
| `transform` | `translate`, `rotate`, `scale`: bakes a transform into the points. |
| `bend` | `angle` (0°), `axis` ("y"), `direction` ("x"), `center`, `limits` [lo, hi]. |
| `twist` | `angle` (0°), `axis` ("y"), `center`, `limits`. |
| `taper` | `factor` (0), `axis` ("y"), `curve` ("linear" or "smooth"), `limits`. |
| `squash` / `squash_stretch` | `amount` (1 = none, 0.5 = half height), `axis` ("y"), `pivot`. Volume-preserving. |
| `wave` | `amplitude` 0.1, `wavelength` 1, `speed` 1, `direction` ("x" or "radial"), `axis` ("y"), `falloff` 0, `phase` 0. |
| `noise` | `amplitude` 0.1, `scale` 1, `octaves` 3, `seed`, `speed` (0 = static), `mode` ("normal" or "vector"). |
| `inflate` | `amount` 0.05: push along normals. |
| `spherify` | `amount` 1, `radius` (auto), `center`. |
| `smooth` | `iterations` 5, `factor` 0.5: Laplacian smoothing. |
| `array` | `count` 3, `offset` [1,0,0], `rotate` (per copy), `scale` (per copy). |
| `mirror` | `axis` ("x"), `merge` true, `tolerance` 1e-4. |
| `solidify` | `thickness` 0.05, `offset` (−1 inward … 1 outward). |
| `ripple` | `amplitude` 0.1, `wavelength` 0.5, `speed` 1, `center`, `decay` 0.5. |
| `displace` | `amplitude`, `texture` ("noise", "voronoi" or "ridged"), `scale`, `seed`. |
| `lattice` | `resolution` [2,2,2], `offsets` {"i,j,k": [dx,dy,dz]}, `bounds` (auto): free-form deformation. |
| `flip` | Flips the normals and winding. |
| `weld` | `tolerance` 1e-5. |
| `smooth_shading` | `angle` 180. |
| `boolean` | `object` (another mesh node), `op` (`difference` (default), `union` or `intersect`), `resolution` 96, `blend` 0 (smooth fillet radius). Volumetric: both meshes become exact signed-distance fields, so even open or self-intersecting inputs work. Animated cutters work too. |
| `remesh` | `resolution` 64, `offset` 0 (grow + / shrink −), `smooth_angle`. Rebuilds an even, watertight surface; fills holes and fuses intersecting parts. |
| `decimate` | `ratio` 0.5 (fraction of vertices to keep) or `cell` (cluster size). Vertex-clustering reduction. |
| `extrude` | `distance` 0.1, `faces` (indices, or `{"normal": [x,y,z], "angle": deg}` to pick faces facing a direction; default all), `direction`, `inset` 0. |

Any modifier can have `"after_deform": true` to run after skinning and blend shapes (subdivision
does this by default).

```json
"modifiers": [
  {"type": "twist", "angle": "=sin(t*2)*90", "axis": "y"},
  {"type": "bend",  "angle": {"keys": [[1, 0], [24, 120, "ease_in_out"]]}},
  {"type": "subdivide", "levels": 1, "render_levels": 2}
]
```

---

## 9. Materials and textures

Define materials in the top-level `materials` block and reference them by name. A node can also
use a preset name or a colour string directly: `"material": "gold"`, `"material": "#ff8800"`.

```json
"materials": {
  "paint":  {"preset": "car_paint", "base_color": "#0b3d91"},
  "glass":  {"preset": "glass"},
  "ground": {"base_color": {"type": "checker", "scale": 8, "color1": "#ddd", "color2": "#333"},
             "roughness": 0.6},
  "brick":  {"base_color_map": "textures/brick_albedo.jpg",
             "normal_map": {"file": "textures/brick_normal.png", "colorspace": "linear"},
             "roughness_map": {"file": "textures/brick_rough.png", "colorspace": "linear"}}
}
```

### Principled BSDF parameters

| parameter | default | meaning |
|---|---|---|
| `base_color` | 0.8 grey | Diffuse or metal colour. |
| `metallic` | 0 | 0 = dielectric, 1 = metal. |
| `roughness` | 0.5 | Microfacet roughness (GGX). |
| `specular` | 0.5 | Dielectric specular level (0.5 ≈ 4% F0). |
| `specular_tint` | 0 | Tint the specular by the base colour. |
| `ior` | 1.5 | Index of refraction (glass and transmission). |
| `transmission` | 0 | 1 = fully refractive (glass, water). |
| `transmission_roughness` | – | Separate roughness for refraction. |
| `attenuation_color`, `attenuation_distance` | – | Beer-Lambert volume absorption (tinted glass, liquids). |
| `thin_walled` | false | Thin shell such as a soap bubble, leaf or sheet. |
| `subsurface` | 0 | Random-walk subsurface amount. |
| `subsurface_radius` | [1, 0.2, 0.1] | Per-channel mean free path. |
| `subsurface_scale` | 0.1 | Scales the radius. |
| `clearcoat`, `clearcoat_roughness` | 0, 0.03 | Lacquer layer. |
| `sheen`, `sheen_tint` | 0, 0.5 | Cloth rim. |
| `emission`, `emission_strength` | black, 1 | Emissive surface; it acts as a real light source with NEE. |
| `emission_one_sided` | false | Emit only from the front face. |
| `alpha` | 1 | Cut-out opacity (leaves, decals). |
| `normal_map`, `normal_strength` | – | Tangent-space normal map. |
| `bump_map`, `bump_strength` | – | Height-based bump (texture or procedural). |
| `shadow_catcher` | false | Invisible except for received shadows, for compositing over plates. |

### Textures

Any colour or scalar parameter accepts a texture. Write it as `"<param>": {texture spec}` or as
`"<param>_map": spec`, with an optional `"<param>_factor"` multiplier.

**Image textures.** A path string, or a spec with these keys:

* `file`
* `colorspace` (`srgb` or `linear`; maps for roughness, metallic and normal should be linear)
* `wrap` (`repeat`, `clamp` or `mirror`)
* `interpolation` (`linear` or `closest`)
* `scale` [u, v], `offset` [u, v], `rotation` (degrees)

All supported image formats can be used, including EXR and HDR.

**Procedural textures.** These take `type`, `color1`, `color2`, `scale`, and `space` (`uv`,
`object` or `world`). Object space is evaluated on the rest pose, so the texture sticks to
deforming meshes.

| type | extra parameters |
|---|---|
| `checker` | – |
| `noise` | `octaves` 5, `gain` 0.5, `distortion` 0, `contrast` 1 |
| `marble` | `octaves` 6, `power` 1, `distortion` 5 |
| `wood` | `rings` 8, `turbulence` 1.5 |
| `grid` | `line_width` 0.05 |
| `gradient` | `axis` (x/y/z), `low` 0, `high` 1 |
| `voronoi` | `mode` (`cells`, `edges` or `distance`), `sharpness` 4 |
| `brick` | `mortar` 0.03, `ratio` 2, `variation` 0 |
| `stripes` | `axis`, `softness` 0 |
| `constant` | `color1` |

### Colours

Colours can be written in any of these forms:

* `[r, g, b]`: linear values.
* `"#rrggbb"`: sRGB.
* CSS colour names: `"tomato"`.
* `"kelvin:3200"` or `"3200K"`: blackbody.
* `"srgb(255,128,0)"`.
* `"hsv(0.6, 0.8, 1)"`.
* A single number, meaning grey.

### Presets

Use `animora materials` to list them and `animora materials NAME` to see one preset's parameters.

| family | presets |
|---|---|
| basic | default, matte, clay, plastic, glossy_plastic, rubber, ceramic, porcelain, toon, checker, studio_floor |
| metals | metal, chrome, mirror, gold, silver, copper, brass, bronze, aluminum/aluminium, brushed_metal, iron, titanium |
| glass and liquids | glass, frosted_glass, tinted_glass, water, ice, diamond, bubble, honey |
| subsurface | skin, wax, jade, milk, marble |
| other | wood, car_paint, velvet, fabric, leather, concrete |
| emission | emissive, light, neon |
| compositing | shadow_catcher, holdout |

A preset plus overrides is the usual pattern: `{"preset": "gold", "roughness": 0.35}`.

---

## 10. Cameras

```json
{"name": "cam", "type": "camera", "translate": [4, 2, 8], "look_at": "hero",
 "focal_length": 35, "fstop": 2.8, "focus_target": "hero", "blades": 6}
```

| key | default | meaning |
|---|---|---|
| `focal_length` | 50 | mm (animatable, so zooms and dolly-zooms work). |
| `fov` / `fov_horizontal` | – | Field of view in degrees, as an alternative to focal length. |
| `sensor_width`, `sensor_height`, `sensor_fit` | 36 mm, –, auto | Film back. |
| `projection` | perspective | `perspective` or `ortho`. |
| `ortho_height` / `ortho_width` | – | Orthographic frame size. |
| `dof` | true if `fstop` or `aperture_radius` is set | Depth of field. |
| `fstop` | – | Physical aperture. |
| `aperture_radius` | – | Explicit lens radius (scene units). |
| `dof_scale` | 1 | Exaggerate or reduce the bokeh. |
| `focus_distance` | 10 | Focus distance. |
| `focus_target` | – | A node to keep in focus (automatic rack focus). |
| `blades`, `blade_rotation` | 0 (round), 0 | Polygonal bokeh. |
| `shift_x`, `shift_y` | 0 | Lens shift (architectural framing). |
| `near_clip` | 0.001 | Near clipping distance. |

* Camera cuts use `"camera_switches": [[1, "wide"], [49, "close"], [97, "wide"]]`.
* The active camera comes from `"camera": "name"`, otherwise the first camera.
* With no camera, an auto-framing camera is created.
* For a handheld camera, add `"shake": {"amplitude": 0.4, "frequency": 1.2}`.

---

## 11. Lights

| `light` | keys (defaults) |
|---|---|
| `point` / `sphere` / `omni` | `intensity` (or `power` in W-like units, where intensity = power / 4π), `radius` (soft shadows), `camera_visible` false |
| `spot` | Point keys plus `cone_angle` 40°, `penumbra` / `blend` 0.15 |
| `sun` / `distant` / `directional` | `intensity` / `strength` 3, `angle` 0.53° (shadow softness), `camera_visible` true |
| `area` / `rect` / `quad` / `disk` / `disc` | `size` [w, h], `intensity` (radiance) or `power`, `two_sided` false, `camera_visible` false, `cast_shadows` false |

Shared keys:

* `color`
* `temperature` (Kelvin, multiplied into the colour)
* `multiplier`
* `exposure` (stops)
* `cast_shadows`

Every value is animatable. Lights aim down −Z: use `rotate`, `look_at` or an aim constraint.
Many-light scenes use light selection that mixes power and solid angle, which keeps noise low.

```json
{"name": "key", "type": "light", "light": "area", "size": [2, 1], "power": 400,
 "temperature": 5200, "translate": [3, 4, 3], "look_at": "hero"}
```

Emissive materials on meshes are also lights: they are sampled with MIS.

---

## 12. Environment and fog

```json
"environment": {"type": "sky", "sun_elevation": 25, "sun_azimuth": 140, "turbidity": 3}
```

| type | keys (defaults) |
|---|---|
| `sky` | Physical sky with sun disc. `sun_elevation` 35°, `sun_azimuth` 0°, `turbidity` 2.5, `sun_strength` 4, `sun_size`, `sun` (true), `ground_color`, `ozone`, `resolution`. |
| `hdri` | `file` (.hdr, .exr or any image), `strength`, `rotation` (°). Importance-sampled. |
| `gradient` | `zenith`, `horizon`, `ground`, `sharpness`. |
| `studio` | `color`, `key` 6, `fill` 2, `rim` 4, `floor`: a soft three-point studio dome. |
| `color` | `color`: a uniform background. |

Common keys:

* `strength`
* `rotation`
* `visible` (`false` lights the scene without showing the environment in camera)
* `background` (a colour shown to the camera instead)

The sun of the `sky` environment is a real distant light with correct angular size.

**Fog.** `"fog": {"density": 0.05, "color": [1,1,1], "anisotropy": 0.3, "height_min": 0,
"height_max": 4}` creates homogeneous participating media with a Henyey-Greenstein phase function.
It gives god rays through spot and sun light, and every key is animatable.

---

## 13. Post-processing and colour

### Colour management

Animora has an OCIO-style colour pipeline built in, with no OCIO dependency. Rendering happens
in a scene-linear **working space**. Every colour entering the renderer is converted from its own
space into the working space; the display transform converts back out. That covers material and
light colours, colour textures, HDRIs and the procedural sky.

| setting | default | meaning |
|---|---|---|
| `working_space` | `lin_srgb` | Rendering space: `lin_srgb` (linear Rec.709), `acescg` (ACES AP1, the VFX standard), `lin_rec2020` or `lin_p3`. Wide-gamut spaces give more saturated, more physically plausible bounce light. |
| `color_input` | `lin_srgb` | Space assumed for plain numeric `[r, g, b]` colours. Hex strings and CSS names are always sRGB. |
| `display` | `srgb` | Output encoding for PNG, JPEG and video: `srgb`, `rec709` (BT.1886), `display_p3`, `rec2020` or `adobe_rgb`. |
| `exr_space` | working space | Space of linear HDR outputs (EXR, HDR, PFM), e.g. `aces2065` for ACES interchange files. |

Per-value spaces:

```json
"base_color": {"rgb": [0.8, 0.2, 0.1], "space": "acescg"},
"base_color": {"type": "image", "file": "albedo.png", "colorspace": "srgb"},
"roughness":  {"type": "image", "file": "rough.png",  "colorspace": "raw"},
"environment": {"type": "hdri", "file": "studio.exr", "colorspace": "acescg"}
```

8-bit colour textures default to sRGB, HDR images to linear, and data maps (roughness, normal,
bump) to raw.

| space | primaries | white | encoding | aliases and typical use |
|---|---|---|---|---|
| `lin_srgb` | Rec.709 | D65 | linear | `linear`, `lin_rec709`; scene-linear default |
| `srgb` | Rec.709 | D65 | sRGB curve | `srgb_texture`; 8-bit textures, web images |
| `rec709` | Rec.709 | D65 | gamma 2.4 | broadcast SDR display |
| `acescg` | ACES AP1 | D60 | linear | VFX working space |
| `acescct`, `acescc` | ACES AP1 | D60 | log | grading |
| `aces2065` | ACES AP0 | D60 | linear | `aces`, `aces2065-1`; archival and interchange |
| `lin_rec2020` / `rec2020` | Rec.2020 | D65 | linear / gamma 2.4 | wide gamut |
| `lin_p3` / `display_p3` | P3 | D65 | linear / sRGB curve | Apple displays |
| `lin_adobe_rgb` / `adobe_rgb` | Adobe 1998 | D65 | linear / gamma 2.2 | print |
| `raw` | – | – | none | `data`, `non-color`; data textures |

White points are adapted with the Bradford transform. `animora colorspaces` lists the spaces,
prints conversion matrices and converts values.

### View transform and grading

The view transform is chosen with `settings.tonemap`. It is applied after converting from the
working space to the display primaries, with soft gamut compression of out-of-gamut colours:

| tonemap | look |
|---|---|
| `agx` (default) | Filmic, graceful highlight desaturation; Blender 4 style. |
| `agx_punchy` | AgX with more contrast. |
| `aces` | ACES RRT/ODT fit. |
| `filmic` | Hable/Uncharted-style filmic curve. |
| `reinhard` | Extended Reinhard. |
| `neutral` | Khronos PBR Neutral. |
| `standard` | Plain sRGB clip. |

The `post` block (every key is animatable):

| key | meaning |
|---|---|
| `exposure` | Stops, or `"auto"`. |
| `white_balance` | Kelvin, e.g. 5000. |
| `tint` | Green-magenta shift. |
| `contrast`, `saturation` | 1 = neutral. |
| `lift`, `gamma`, `gain` | Colour-grading wheels (scalar or RGB). |
| `lut` | A look LUT applied after the view transform: `.cube` (3D or 1D; Resolve/Adobe), `.spi3d` / `.spi1d` (Sony Imageworks, OCIO's native formats) or `.3dl` (Autodesk/Lustre). Trilinear interpolation. |
| `bloom`, `bloom_threshold`, `bloom_radius` | Glow around bright areas. |
| `vignette`, `vignette_softness` | Lens vignette. |
| `chromatic_aberration` | Lateral CA in pixels. |
| `grain`, `grain_size` | Film grain. |
| `dither` | true by default for 8-bit output (prevents banding). |

HDR outputs (EXR, HDR, PFM) store linear radiance **before** tone mapping, in `exr_space`, ready
for compositing.

---

## 14. Animation

Every numeric, vector or colour value in a scene can be animated, including transforms, material
parameters, light intensities, modifier amounts, camera focus and constraint weights. Values can be
given in any of these forms:

| form | example |
|---|---|
| Constant | `0.5`, `[0, 1, 0]`, `"#ff0000"` |
| Keyframes | `{"keys": [[1, 0], [24, 10, "ease_out"], [48, 0]]}` |
| Bare key list | `[[1, 0], [24, 10]]` (a list of `[frame, value]` pairs) |
| Expression | `"=sin(t*tau)*2"` or `{"expr": "..."}` |
| Per-component mix | `[0, "=abs(sin(t*4))", {"keys": [[1,0],[48,5]]}]` |
| Layered | `{"keys": [...], "add": "=noise(t*3)*0.1"}` or `"mul": ...` |
| Driven key | `{"driver": "ctrl.translate.y", "keys": [[0, 0], [1, 90]]}` (below) |

### Keys and tangents

A key is `[frame, value]`, `[frame, value, interp]`, or a dict
`{"frame": f, "value": v, "interp": ..., "in": slope, "out": slope}`. The interpolation applies
to the segment **after** the key.

| interp | behaviour |
|---|---|
| `auto` (default) | Smooth spline that does not overshoot (Maya "auto"). |
| `spline` | Catmull-Rom style smooth tangents (may overshoot). |
| `clamped` | Spline tangents, flattened where neighbouring values are equal. |
| `flat` | Zero tangents (ease in/out at every key). |
| `linear` | Straight lines. |
| `step` / `constant` | Hold the value until the next key. |
| `bezier` | Explicit `in` / `out` slopes (value units per frame). |
| any easing name | Robert-Penner style ease over the segment (below). |
| `cubic-bezier(x1,y1,x2,y2)` | CSS-style custom ease. |

**Easings** (`animora curves --plot` draws them):

* `linear`, `ease`, `ease_in`, `ease_out`, `ease_in_out`, `smoothstep`, `smootherstep`
* `sine_*`, `quad_*`, `cubic_*`, `quart_*`, `quint_*`, `expo_*`, `circ_*`, `back_*`, `elastic_*`,
  `bounce_*`, each in `_in`, `_out` and `_in_out` variants

### Infinity (before and after the keys)

Set `"pre"` and `"post"` on a key block to one of:

* `constant` (default)
* `linear`: extrapolate the end tangent.
* `cycle`: repeat.
* `cycle_offset`: repeat while accumulating the offset, which suits walk cycles.
* `oscillate`: ping-pong.

```json
"rotate": {"keys": [[1, [0,0,0]], [24, [0,360,0], "linear"]], "post": "cycle_offset"}
```

### Driven keys (set driven key)

A key block with a `driver` is a **driven key**, like Maya's Set Driven Key. The curve's
horizontal axis is the value of the driver channel, not the frame. Every interpolation, tangent
and infinity option above works the same way.

```json
{"name": "ctrl", "type": "locator", "translate": {"keys": [[1, [0,0,0]], [48, [0,1,0]]]}},
{"name": "finger", "type": "joint",
 "rotate": [0, 0, {"driver": "ctrl.translate.y", "keys": [[0, 0], [0.5, 40, "flat"], [1, 90]], "post": "linear"}]}
```

The driver is any path accepted by `get()` (see [Reading values with `get`](#reading-values-with-get)),
for example `ctrl.translate.y`, `ctrl.rotate.x`, `ctrl.world_position.y` or a custom attribute
`ctrl.curl`. Driven keys can drive vectors and colours as well as scalars; the keys'
values are then vectors. Chains of driven keys are allowed, but a cycle evaluates the cyclic
driver as 0. `animora keys` labels a driven channel with "driven by …", and `animora graph` plots
the resulting value over time.

### Animating from Python

```python
ball = scene.node("ball")
ball.key("translate", 1,  [0, 3, 0])
ball.key("translate", 12, [0, 0.5, 0], "ease_in")
ball.set("scale", "=1 + 0.1*sin(t*8)")
```

---

## 15. Expressions

A string starting with `=` is an expression. Expressions are evaluated every frame (including
motion-blur sub-frames) in a safe sandbox: an AST whitelist with no imports and no attribute
access.

**Variables:**

| name | meaning |
|---|---|
| `t` | Time in seconds. |
| `frame` / `f` | Current frame (fractional at sub-frames). |
| `fps`, `start`, `end` | Timeline values. |
| `u` | Normalised time 0 → 1 over the frame range. |
| `pi`, `tau`, `e` | Constants. |
| user variables | Everything in the scene's `variables` block. |

**Functions:**

| group | functions |
|---|---|
| math | `sin cos tan asin acos atan atan2 sqrt exp log log10 pow abs floor ceil round min max sign radians degrees hypot` |
| shaping | `clamp(x,a,b)`, `lerp(a,b,t)`, `smoothstep(a,b,x)`, `linstep(a,b,x)`, `remap(x,a,b,c,d)`, `step(edge,x)`, `pulse(x,a,b)`, `fract`, `mod`, `ease(name, u)` |
| noise | `noise(x[,y,z])` (smooth −1…1), `fbm(x, octaves)`, `turbulence(x)`, `wiggle(freq, amp[, seed])`, `rand(seed)` (stable random 0…1), `choose(seed, a, b, ...)` |
| motion | `bounce(t, height, period[, damping])`, `spring(t, freq, damping)`, `saw(x)`, `tri(x)`, `square(x)` |
| vectors | `vec(x,y,z)`, `length`, `normalize`, `dot`, `cross`, `mix(a,b,t)` |
| scene | `get("node.attr")` (value now), `at(frame, "node.attr")` (value at another frame) |

Lists `[a, b, c]`, conditionals `a if cond else b`, comparisons, and `and`, `or` and `not` all work.

```json
"translate": ["=sin(t*2)*3", "=abs(sin(t*6))*1.5 + 0.5", 0],
"rotate":    [0, "=degrees(atan2(get('target.wpos.x') - get('self.wpos.x'), 1))", 0],
"intensity": "=50 + 30*noise(t*10)",
"visible":   "=frame >= 24",
"translate": "=at(frame - 6, 'leader.world_position')"
```

### Reading values with `get`

The path format is `node.attribute[.component]`. Attributes:

* `translate`, `rotate`, `scale`, `pivot`
* `world_position` / `wpos`, `world_rotation` / `wrot`, `world_scale`
* `visible`, any primitive parameter (`radius`, ...), any custom `attributes` entry
* blend-shape weights

Components are `x`/`y`/`z`, `r`/`g`/`b`, or an index. The same paths work in `animora eval`.

---

## 16. Constraints

Constraints go in a node's `"constraints"` list and are applied in order after the node's own
transform. Every constraint takes these keys:

* `target` or `targets`: a node name, a point, or `{"node": ..., "weight": ...}`.
* `weight`: animatable, so constraints can be blended in and out.
* `enabled`.

| type | keys |
|---|---|
| `aim` / `look_at` | `aim_axis` ("-z" for cameras and lights, otherwise "x"), `up_axis` ("y"), `world_up_type` (`vector`, `object` or `object_rotation`), `world_up`, `world_up_object`, `offset` (°) |
| `point` | `offset`, `maintain_offset`, `skip` (e.g. "xz") |
| `orient` | `offset`, `maintain_offset` |
| `parent` | `maintain_offset` (true), `bind_frame`. Weight-animated parent switching (pick up / put down props). |
| `scale` | `maintain_offset`, `skip` |
| `path` / `motion_path` | `curve` (curve node) or `points`, `smooth`, `closed`, `samples`, `u` (animatable 0…1), or `start_frame` / `end_frame`, `loop` / `cycle`, `arc_length` (constant speed), `offset`, `follow` (true), `aim_axis`, `up_axis`, `world_up`, `bank`, `bank_limit` (60°) |
| `limit` | `translate_min`, `translate_max`, `rotate_min`, `rotate_max`, `scale_min`, `scale_max` (null skips an axis) |
| `copy_transforms` | Copies the world transform of the target. |

Multiple weighted targets are blended, as with Maya's multi-target constraints.

```json
{"name": "car", "type": "mesh", "file": "car.obj",
 "constraints": [{"type": "path", "curve": "road", "start_frame": 1, "end_frame": 120,
                  "arc_length": true, "follow": true, "bank": 0.5}]}
```

---

## 17. Rigging

### Joints

`"type": "joint"` nodes form skeletons through `parent` or `children`. A joint is a transform
node: animate it with FK keys like any node, or drive it with IK.

### IK

IK handles go in the top-level `ik` list:

```json
"ik": [{"chain": ["shoulder", "elbow", "wrist"], "target": "hand_ctrl", "pole": "elbow_pole",
        "solver": "two_bone", "stretch": true, "soft": 0.05, "orient_end": true}]
```

| key | meaning |
|---|---|
| `chain` or `start` + `end` | The joints to solve. |
| `target` | Goal node. |
| `pole` | Pole-vector node (controls the bending plane). |
| `solver` | `auto` (two-bone for 3 joints, FABRIK otherwise), `two_bone` (analytic), `fabrik` or `ccd`. |
| `weight` | IK/FK blend, animatable. |
| `enabled` | Animatable on/off. |
| `twist` | Twist around the chain axis in degrees. |
| `stretch` | Joints stretch when the target is out of reach. |
| `soft` | Soft-IK distance; removes the snap at full extension. |
| `iterations` | 20 (iterative solvers). |
| `bend_axis` | Preferred bend axis when there is no pole. |
| `orient_end` | Make the end joint match the target's rotation. |

### Skinning

```json
{"name": "arm", "type": "mesh", "primitive": "cylinder", "height": 4, "height_segments": 24,
 "skin": {"joints": ["shoulder", "elbow", "wrist"], "method": "dq", "weights": "auto",
          "max_influences": 4, "smooth": 2, "falloff": 4, "bind_frame": 1}}
```

* `joints` lists the influences; alternatively, `root` takes every joint under it.
* `method` is `linear` (LBS) or `dq` (dual-quaternion, no candy-wrapper collapse).
* `weights: "auto"` computes weights from bone-segment distance, then smooths them over the mesh
  (`smooth` iterations, `falloff` exponent). Explicit weights are also accepted: `{"joint": [w per
  vertex]}` or per-vertex lists.
* `weights: "file.json"` loads weights written by `animora weights --export` (or any file in the
  same format). `weights_match` picks how vertices are matched: `index` (same topology),
  `position` (nearest vertex in the bind pose, so weights transfer onto a re-modelled or
  re-subdivided mesh) or `auto` (index when the vertex counts agree, otherwise position).
* `copy_weights_from: "other_mesh"` copies the final weights of another skinned mesh by nearest
  bind-pose position (Maya's Copy Skin Weights), e.g. from a body onto a shirt.
* `weight_paint` and `locked_joints` edit the weights; see [Weight painting](#weight-painting).
* `bind_frame` is the frame whose pose counts as the bind pose.

### Weight painting

Maya's Paint Skin Weights tool is interactive. In Animora each brush stroke is one JSON object in
the skin's `weight_paint` list, so painting is reproducible, diffable and versionable. Strokes run
in order, after the automatic, loaded or copied weights.

```json
"skin": {
  "root": "chest", "method": "dq", "locked_joints": ["chest"],
  "weight_paint": [
    {"joint": "l_elbow", "op": "replace", "value": 1.0, "center": [1.12, 1.4, 0], "radius": 0.2,
     "surface": true, "mirror": "x"},
    {"joint": "l_shoulder", "op": "add", "value": 0.3, "bone": "l_shoulder", "strength": 0.6, "mirror": "x"},
    {"op": "smooth", "bone": "l_elbow", "radius": 0.15, "iterations": 3},
    {"op": "mirror", "axis": "x", "direction": "+"},
    {"op": "prune", "threshold": 0.02},
    {"op": "max_influences", "count": 3}
  ]
}
```

**Brush operations** (need `joint`, except `smooth` and `hammer`):

| op | effect |
|---|---|
| `add` | Raise the joint's weight by `value` (default 0.1). |
| `subtract` | Lower it by `value` (default 0.1). |
| `replace` | Set it to `value` (default 1.0). |
| `scale` | Multiply it by `value` (default 0.9). |
| `smooth` | Blend every influence with the mesh neighbours (`iterations`, `strength` default 0.5). |
| `hammer` | Full-strength smooth; fixes stray, spiky vertices. |

Normalisation works like Maya's interactive mode: when a brush changes one joint, the other
unlocked influences of that vertex absorb the difference in proportion (if none can, the joint's
parent takes it). Weights always sum to 1.

**Where a brush acts** (pick one; with none, the stroke floods the whole mesh):

| key | region |
|---|---|
| `center` | A bind-pose world point, with `radius`. |
| `at` | The position of a joint, with `radius`. |
| `bone` | The segment from a joint to its child, with `radius`. |
| `points` | A polyline stroke `[[x,y,z], ...]`, with `radius`. |
| `vertices` | Explicit vertex indices. |
| `box` | An axis-aligned box `[[min], [max]]`. |

**Brush settings:**

| key | default | meaning |
|---|---|---|
| `radius` | 0.1 | Brush radius in scene units. |
| `falloff` | `smooth` | `smooth`, `linear`, `constant`, `gaussian` or `needle`. |
| `strength` | 1.0 | Opacity of the stroke (0.5 for `smooth`). |
| `surface` | false | Measure distance along the mesh surface instead of straight-line, so a brush on one leg never bleeds onto the other. |
| `mirror` | false | `true` / `"x"` / `"y"` / `"z"`: repeat the stroke on the other side with the mirrored joint (`l_elbow` ↔ `r_elbow`). `mirror_origin` moves the mirror plane. |
| `lock` | [] | Extra joints locked for this stroke only. |

Mirrored joint names are found for all common conventions: `Left`/`Right`, `l_`/`r_`, `_l`/`_r`,
`.L`/`.R`, `_L_`, `LHip`/`RHip` and CMU-style `lhumerus`/`rhumerus`, with or without namespaces.

**Global operations** (no brush):

| op | keys | effect |
|---|---|---|
| `prune` | `threshold` (0.01) | Drop weights below the threshold and renormalise (Maya's Prune Small Weights). |
| `normalize` | – | Rescale every vertex to sum to 1. |
| `max_influences` | `count` (4) | Keep only the strongest N influences per vertex. |
| `mirror` | `axis` (`x`), `direction` (`+`: copy the + side onto the − side), `origin`, `tolerance` | Mirror Skin Weights: copies one side's weights to the other with joint names swapped. |
| `move` | `from`, `to`, optional brush region | Move a joint's weight onto another joint. |
| `copy` | `from`, `to`, optional brush region | Copy a joint's weight column onto another joint. |

`locked_joints` on the skin (Maya's Hold Weights) protects joints from every brush; `prune` and
`max_influences` still apply to them. Inspect the result with `animora weights` (heat maps, posed
previews, statistics) and save it with `--export`.

### Blend shapes

```json
"blend_shapes": [
  {"name": "smile", "weight": {"keys": [[1, 0], [20, 1], [40, 0]]},
   "modifiers": [{"type": "lattice", "offsets": {"0,1,1": [0, 0.2, 0.1]}}]},
  {"name": "blink", "weight": "=pulse(frame % 72, 60, 64)", "file": "head_blink.obj"}
]
```

Each target is defined by one of:

* `modifiers`: a modifier stack applied to the base mesh.
* `file`: a mesh with the same topology.
* `verts`: absolute positions.
* `deltas`: offsets.

Weights are animatable, are evaluated before skinning, and are exposed as node attributes
(`get("face.smile")`).

### Retargeting (HumanIK-style)

Retargeting transfers motion between skeletons with different proportions, rest poses, joint
names and facing directions, which is what Maya's HumanIK "characterize and retarget" does. Add a
top-level `retarget` block to a scene that contains the target skeleton; the target joints are
keyed when the scene loads:

```json
"retarget": {"source": "data/walk_cycle.bvh", "foot_lock": true}
```

| key | default | meaning |
|---|---|---|
| `source` | – | A `.bvh` file (relative to the scene). Omit it and set `source_root` to retarget between two skeletons inside the same scene. |
| `source_root` / `target_root` | – | Root joint of the source / target skeleton, when a scene holds several. |
| `map` | – | `{target_joint: source_joint}` overrides for the automatic mapping. |
| `scale` | 0.01 | BVH unit scale (centimetres to metres). |
| `start_frame` | first frame | Frame where the motion starts. |
| `source_yaw` | `"auto"` | Degrees to turn the source about +Y; `auto` detects both characters' facing from hips and shoulders. |
| `match_rest` | true | Compensate for different rest poses, so a T-pose capture drives an A-pose character. |
| `height_scale` | `"auto"` | Root-motion scale; `auto` uses the ratio of hip heights so the stride fits the legs. |
| `root_motion` | true | Transfer the hips translation. |
| `in_place` | false | Remove horizontal travel (treadmill motion). |
| `foot_lock` | false | Detect planted feet and pin them with two-bone leg IK, removing foot sliding. |
| `lock_height`, `lock_speed` | 0.06, 0.5 | Contact thresholds, as fractions of the leg length (height above the ground; speed per second). |
| `rest_frame` | first frame | Frame whose pose counts as the rest pose. |
| `set_frame_range` | true | Set `settings.frame_range` to the motion's length when the scene has none (`true` always overrides it). |

`retarget` can also be a list of blocks, for example one per character.

How it works:

1. **Characterisation.** Every joint of both skeletons is mapped onto the HumanIK slots (`Hips`,
   `Spine`…`Spine3`, `Neck`, `Head`, and `Shoulder`, `Arm`, `ForeArm`, `Hand`, `UpLeg`, `Leg`,
   `Foot`, `ToeBase` on each side). The matcher understands HumanIK / Mixamo
   (`mixamorig:LeftForeArm`), Unreal (`upperarm_l`, `calf_r`, `spine_02`), Blender / Rigify
   (`upper_arm.L`, `shin.R`), CMU / ASF (`lhumerus`, `rfemur`, `lowerback`), SMPL (`left_collar`,
   `left_knee`) and generic `L_` / `_R` / `Left` spellings. Spine chains of different lengths are
   distributed proportionally, and joints with identical names (fingers, for example) map directly.
2. **Facing alignment** removes the yaw between the characters.
3. **Rest-pose matching** rotates each source bone onto the target's rest bone direction.
4. **Rotation transfer** applies world-space rotation deltas to the target rest pose and converts
   them to local Euler keys in each joint's own `rotate_order`. Unmapped joints keep their rest pose.
5. **Root motion** is scaled by leg length and measured from each skeleton's ground level.
6. **Foot locking** (optional) pins each contact to one world position with leg IK.

From the command line, `animora retarget` does the same and writes a new scene
([§3](#animora-retarget-source-target)). From Python:

```python
from animora.retarget import retarget, canonical_slots
keys, info = retarget("walk.bvh", target_scene, foot_lock=True)   # {joint: {"rotate": keys, ...}}
print(info["map"], info["height_scale"], info["source_yaw"])
```

---

## 18. Dynamics

Simulations are deterministic. Integer frames are cached, and sub-frames (motion blur) are
interpolated. Global settings go in the top-level `dynamics` block:

| key | default |
|---|---|
| `gravity` | [0, −9.81, 0] |
| `ground` | true (infinite ground plane) |
| `ground_height` | 0 |
| `ground_friction` | 0.6 |
| `ground_restitution` | 0.4 |
| `substeps` | 8 |
| `cache` | null: path to a baked `.npz` (`animora bake`) |

### Rigid bodies

Add `"rigid_body": {...}` to any mesh:

| key | default | meaning |
|---|---|---|
| `type` | `active` | `active` / `dynamic` are simulated; `passive` is a static or keyframed collider. |
| `mass` | 1 | Mass. |
| `shape` | auto | `sphere`, `box`, `plane`, `capsule`, `cylinder` (both Y-axis; `radius`, `height`), `mesh` (aliases `convex`, `hull`, `sdf`: the exact mesh as a signed-distance field, so concave shapes such as bowls and torus rings collide correctly) or `auto` (from the primitive, otherwise the bounding box). `radius`, `size` and `height` override. |
| `velocity` | [0,0,0] | Initial linear velocity. |
| `angular_velocity` | [0,0,0] | Initial angular velocity in °/s. |
| `restitution` / `bounce` | 0.4 | Bounciness. |
| `friction` | 0.5 | Friction. |
| `linear_damping` | 0.02 | Linear damping. |
| `angular_damping` | 0.05 | Angular damping. |
| `start_frame` | first frame | Before this frame the body follows its animation, then it inherits that velocity. |

### Cloth

`"type": "cloth"` nodes are simulated with position-based dynamics. By default the cloth is a
`size` 2 plane with `resolution` 24; set `primitive` or `file` to use any mesh.

| key | default | meaning |
|---|---|---|
| `stiffness` | 1.0 | Stretch resistance. |
| `bend` / `bend_stiffness` | 0.2 | Bending resistance. |
| `damping` | 0.01 | Velocity damping. |
| `drag` | 0 | Air drag (needed for wind). |
| `thickness` | 0.01 | Collision offset. |
| `friction` | 0.4 | Collision friction. |
| `iterations` | 25 | Constraint iterations. |
| `substeps` | max(4, global) | Substeps per frame. |
| `gravity_scale` | 1 | Gravity multiplier. |
| `wind` | – | Wind vector [x, y, z]. |
| `turbulence` | 0 | Noisy wind gusts; `seed` varies them. |
| `pin` | – | Pinned vertices: `"corners"`, `"top"`, `"bottom"`, `"center"`, `"edge_min_x"`/`"edge_max_z"` (any min/max and axis), a vertex index, `{"near": [x,y,z], "radius": r}`, or a list of these. Pinned vertices follow the node's animated transform. |
| `colliders` | [] | Mesh nodes to collide with. Primitives use analytic shapes (sphere, box, plane, capsule, cylinder); any other mesh uses an exact signed-distance grid of its surface. Animated and deforming colliders work. Entries are names or `{"node": name, "shape": ...}`. |
| `self_collision` | false | Stop the cloth passing through itself (folds, crumples, draping over itself). Uses a spatial hash on vertices. |
| `self_thickness` | 0.9 × average edge length | Minimum distance kept between non-neighbouring cloth vertices. |
| `self_rest_ignore` | max(2 × edge, 1.5 × thickness) | Vertex pairs closer than this at rest are never pushed apart, so neighbours do not fight. |
| `self_collision_every` | 5 | Run the self-collision pass every N constraint iterations (always on the last one). |
| `ground` | global | Collide with the ground plane. |
| `start_frame` | first frame | Start of the simulation. |
| `tear` | false | Let the cloth rip. `true`, or a number used as `tear_strain`. |
| `tear_strain` | 1.5 | An edge tears when stretched beyond this multiple of its rest length. |
| `tear_variation` | 0.15 | Random ± variation of each edge's threshold, so rips look ragged instead of straight. |
| `tear_max` | 6 | Maximum vertex splits per substep (limits how fast a rip runs). |
| `tear_start` | `start_frame` | No tearing before this frame (lets the cloth settle first). |

Tearing splits vertices along overstretched edges, so the mesh topology changes during the
simulation. UVs, materials, collisions, subdivision, rendering, baking (`animora bake`) and export
all follow the torn mesh. See `examples/cloth_tear.json`.

Subdivision modifiers on a cloth run **after** the simulation, which keeps it fast while the
render stays smooth.

### Particles

`"type": "particles"` nodes emit, simulate and render particles.

| key | default | meaning |
|---|---|---|
| `emitter` | point | `{"shape": "point"}`, `"sphere"` (`radius`, `surface`), `"box"` (`size`), `"disc"` (`radius`), or `"mesh"` (`node`: emit from another mesh's surface). |
| `rate` | 100 | Particles per second (animatable). |
| `burst` | – | `{"frame": f, "count": n}`: a one-shot burst (count defaults to 100). |
| `start_frame`, `end_frame` | timeline | Emission window. |
| `speed` | 2 | Initial speed. |
| `speed_random` | 0.2 | Speed variation. |
| `direction` | [0,1,0] (`"normal"` for sphere and mesh emitters) | Emission direction. |
| `spread` | 15° | Cone angle around the direction. |
| `lifetime` | 2 s | Lifetime. |
| `lifetime_random` | 0.3 | Lifetime variation. |
| `size` | 0.05 | Particle size. |
| `size_random` | 0.3 | Size variation. |
| `size_over_life` | – | Curve `[[0, 1], [1, 0]]`. |
| `drag` | 0 | Air drag. |
| `gravity_scale` | 1 | Gravity multiplier. |
| `bounce` | 0.3 | Restitution on collision. |
| `friction` | 0.2 | Collision friction. |
| `collide_ground` | global | Collide with the ground plane. |
| `colliders` | [] | Collider mesh nodes: analytic for primitives, exact signed-distance grid for any other mesh. |
| `fields` | [] | Force fields (below). |
| `max_particles` | 5000 | Particle cap. |
| `seed` | 1 | Random seed. |
| `substeps` | 2 | Substeps per frame. |
| `render` | sphere | `{"shape": "sphere" (with `subdivisions`), "cube", "quad" or "tetra"}`, or `{"instance": "node"}` to copy any mesh. |

Force fields (`fields` entries; `strength` is animatable):

| type | keys |
|---|---|
| `turbulence` / `noise` | `strength`, `frequency`, `speed`, `seed` |
| `curl` | Divergence-free curl noise (beautiful swirls); `strength`, `frequency`, `speed`, `seed` |
| `wind` | `direction`, `strength`, `drag` |
| `vortex` | `center`, `axis`, `strength`, `pull` |
| `attract` / `repel` | `target` (node or point), `strength`, `falloff` |
| `drag` | `strength` |
| `gravity` | `direction`, `strength` |

Particles get motion blur automatically when `motion_blur` is on.

### Volumes

`"type": "volume"` nodes are participating media: clouds, smoke, fire, dust and fog with god
rays. A volume is a box of `size`, centred on the node origin and transformed like any node. It
is rendered unbiased in the path tracer, using delta tracking for camera and bounce rays and
ratio tracking for shadows. It has single and multiple scattering, receives light from every
light type and the environment, and casts volumetric shadows.

```json
{"name": "cloud", "type": "volume", "size": [4, 2, 3], "translate": [0, 3, 0],
 "density": 6, "albedo": 0.95, "anisotropy": 0.5,
 "shape": "sphere", "falloff": 0.4, "frequency": 1.5, "octaves": 5, "threshold": 0.35,
 "offset": {"keys": [[1, [0,0,0]], [96, [0.8,0.1,0]]]}}
```

| key | default | meaning |
|---|---|---|
| `size` | [2, 2, 2] | Box size (local units). |
| `density` | 1 | Extinction coefficient multiplier (animatable). |
| `albedo` / `color` | 0.8 | Scattering colour; 1 is white cloud, low values are dark soot. |
| `anisotropy` | 0 | Henyey-Greenstein g: > 0 scatters forward (silver lining), < 0 backward. |
| `emission`, `emission_strength` | white, 0 | Emitted colour and strength (per unit length, scaled by density). |
| `blackbody` / `blackbody_strength` | 0 | Physically based fire glow from heat (Planck colour, brightness ∝ T⁴). |
| `temperature` | [800, 2600] | Kelvin range that heat 0 to 1 maps onto. |
| `heat_from_density` | 0 | Heat = normalised density × this, for procedural volumes and grids without a heat channel (glowing nebulae, simple fireballs). |

**Procedural source** (the default). `shape` is `sphere`/`ball`, `box`/`cube` or `cylinder`,
softened by `falloff` (0.3) and carved by fBm noise: `frequency` 3, `octaves` 5 (0 = no noise),
`gain` 0.5, `lacunarity` 2, `threshold` 0.35, `contrast` 2.5. `offset` is an animatable
vector; animate it to make clouds drift or boil.

**Grid source.** Set `grid` to a `.npy` (density only) or `.npz` file, or to an inline nested list.
An `.npz` file is read for `density` (rename with `density_channel`) and an optional heat array
(`heat`, `temperature`, `flame` or `fuel`, or `heat_channel`). `grid_order` is `zyx`
(default, numpy `[z][y][x]`) or `xyz`; `density_scale` and `heat_scale` multiply the channels.

### Smoke and fire

Add `"smoke": {...}` to a volume to simulate smoke or fire inside its box. The solver is a stable-fluids /
Bridson-style grid solver with buoyancy, wind, turbulence, vorticity confinement, pressure projection and
MacCormack (BFECC) advection. The volume then renders the simulated density and heat. Combine
with `blackbody` for fire.

```json
{"name": "fire", "type": "volume", "size": [2, 4, 2], "translate": [0, 2, 0],
 "density": 8, "albedo": 0.3, "blackbody": 8, "temperature": [700, 2400],
 "smoke": {"resolution": 48, "buoyancy": 4, "cooling": 1.5, "dissipation": 0.2,
           "vorticity": 2, "turbulence": 1,
           "sources": [{"shape": "sphere", "center": [0, -1.7, 0], "radius": 0.35,
                        "density": 2, "heat": 1, "velocity": [0, 1, 0], "noise": 0.5}]}}
```

| key | default | meaning |
|---|---|---|
| `resolution` | 40 | Voxels along the longest box axis (the others are proportional). |
| `substeps` | 1 | Solver steps per frame. |
| `buoyancy` | 3.0 | Upward force from heat. |
| `weight` | 0.05 | Downward force from density (soot). |
| `wind` | [0, 0, 0] | Constant wind (animatable). |
| `turbulence`, `turbulence_scale` | 0.5, 0.5 | Curl-noise forcing strength and feature size. |
| `vorticity` | 1.0 | Vorticity confinement: keeps small swirls alive. |
| `cooling` | 1.0 | Heat loss per second. |
| `dissipation` | 0.1 | Density loss per second. |
| `max_heat` | 1.0 | Heat clamp. |
| `pressure_iterations` | 60 (native) / 40 (numpy) | Pressure-solve iterations for incompressibility. |
| `sor` | 1.7 | Over-relaxation factor of the native red-black SOR pressure solver. |
| `native` | true | Use the multithreaded C++ solver (advection, vorticity, pressure); `false` forces the numpy fallback. |
| `floor` | true | The bottom face of the domain is a wall, so smoke spreads along the ground. |
| `start_frame`, `seed` | first frame, 1 | Simulation start and noise seed. |
| `colliders` | [] | Obstacles: node names or `{"node": name, "shape": "mesh" \| "box" \| "sphere", "thickness": t}`. |
| `cache` | – | `.npz` path, written after simulating and reused when it exists. |
| `sources` | [{}] | Emitters, below. |

Source keys: `shape` (`sphere` or `box`), `center`, `radius`, `size` (local to the box centre),
animatable `density`, `heat` and `velocity`, `follow` (a node name; the source tracks its world
position), `start` / `end` frames, and `noise` (0 to 1; breaks up the emission for flickering
flames).

**Colliders.** The default `shape: "mesh"` voxelises the collider's actual geometry every frame
with a signed-distance field (winding-number sign, so open meshes work; open surfaces get a shell
`thickness`, default half a voxel). Concave shapes such as rings, bowls and characters block smoke
correctly, and animated, deforming and skinned colliders push it: their per-vertex velocity is
written into the solid cells and used as a moving-wall boundary in the pressure solve. `box` and
`sphere` use the collider's bounding box or sphere and are cheaper.

**Performance.** The native solver runs 20 to 60× faster than the numpy fallback and makes
100³–150³ grids practical. See `examples/smoke_colliders.json`: smoke rising through a torus ring
and around a swinging paddle.

### Liquids

`"type": "liquid"` nodes simulate water and other liquids with Position-Based Fluids (Macklin &
Müller 2013) in native code. That covers incompressibility through density constraints,
tensile-instability correction, XSPH viscosity, vorticity confinement and adaptive CFL substeps.
Every frame the particles are meshed into a smooth watertight surface (kernel splatting plus surface
nets with sub-voxel projection). The surface renders with the node's `material`; the `water` preset
gives glass-like refraction.

```json
{"name": "water", "type": "liquid", "size": [4, 2.2, 1.6], "translate": [0, 1.1, 0],
 "spacing": 0.035, "material": "water", "viscosity": 0.02,
 "emitters": [{"type": "fill", "shape": "box", "center": [-1.5, -0.45, 0], "size": [1.0, 1.3, 1.6]},
              {"type": "stream", "center": [1.85, 0.6, 0.3], "radius": 0.1,
               "velocity": [-3, 0.5, 0], "start": 12, "end": 50}],
 "colliders": ["rock"], "cache": "water.npz"}
```

| key | default | meaning |
|---|---|---|
| `size` | [2, 2, 2] | Domain box (particles collide with its walls). |
| `spacing` | 0.05 | Particle spacing, the main resolution control. Particle count grows as 1/spacing³. |
| `substeps` | max(1, ⌈240/fps⌉) | Minimum steps per frame (more are added for fast flow). |
| `iterations` | 4 | Density-constraint iterations per step. |
| `viscosity` | 0.01 | XSPH viscosity: 0.01 for water, 0.1 to 0.5 for syrup. |
| `vorticity` | 0 | Vorticity confinement. |
| `tension` | 0.001 | Artificial pressure (surface-tension-like clustering fix). |
| `gravity` | [0, −9.81, 0] | Gravity. |
| `friction`, `damping` | 0.1, 0 | Wall/collider friction, global velocity damping. |
| `open_top` | false | Let particles leave through the top of the domain. |
| `max_particles` | 300000 | Particle cap. |
| `cfl`, `max_speed`, `max_correction` | 0.3, auto, 0.25 | Stability controls (adaptive step size, velocity clamp, per-iteration correction clamp). |
| `colliders` | [] | Mesh nodes; exact signed-distance grids, rebuilt when they move or deform. |
| `surface_radius` | 2.5 | Splat radius in spacings (larger is smoother and blobbier). |
| `surface_resolution` | 2 (1 in preview) | Voxels per particle spacing for meshing. |
| `iso`, `smooth_iterations`, `smooth_angle` | 0.5, 3, 180 | Surface threshold, field smoothing and normal smoothing angle. |
| `start_frame`, `seed` | first frame, 7 | Start and jitter seed. |
| `cache` | – | `.npz` particle cache, written after simulating and reused when it exists. |

Emitters (positions are local to the domain):

* `type: "fill"` fills a volume once with particles. `shape` is `box` (`center`, `size`),
  `sphere` (`center`, `radius`) or `mesh` (`object`: the inside of another mesh node). An
  optional `velocity` gives the initial motion.
* `type: "stream"` (alias `nozzle` / `inflow`) is a disc of radius `radius` at `center` that
  emits continuously along `velocity` between `start` and `end`. All values are animatable.

### Hair and fur

`"type": "hair"` nodes grow strands on the surface of a `source` mesh. Roots are stored as
triangle and barycentric coordinates, so the hair follows any deformation of the source: skinning,
blend shapes, cloth and modifiers. Each frame the strands are styled, optionally simulated, and
output as thin tapered tube (or ribbon) geometry that renders like any mesh.

```json
{"name": "fur", "type": "hair", "source": "ball", "material": "fur",
 "count": 12000, "length": 0.22, "length_variation": 0.4, "segments": 6, "width": 0.006,
 "gravity": 0.25, "curl": 0.4, "frizz": 0.3, "clump": 0.35, "clump_count": 600,
 "comb": [0, -1, 0], "comb_strength": 0.3,
 "groom": [{"center": [0, 1.22, 0], "radius": 0.5, "length": 1.6, "comb": [0, 0, 1]}],
 "dynamics": true, "stiffness": 0.12, "colliders": [{"node": "floor"}]}
```

| key | default | meaning |
|---|---|---|
| `source` | – | Mesh node to grow on (required). |
| `count` | 2000 | Number of strands (area-weighted distribution). |
| `length`, `length_variation` | 0.3, 0.3 | Strand length and random shortening (0 to 1). |
| `segments` | 8 | Segments per strand. |
| `width`, `tip_width` | 0.004, 0.2 | Root width; tip width as a fraction of the root width. |
| `shape`, `sides` | `tube`, 3 | `tube` (with `sides`) or `ribbon` (camera-independent flat strips). |
| `gravity` | 0.3 | Droop of the styled shape. |
| `comb`, `comb_strength` | [0,0,0], 0.5 if `comb` is set | Direction strands lean along the surface. |
| `curl`, `curl_frequency` | 0, 3 | Helical curl amplitude (relative to length) and turns per strand. |
| `frizz`, `frizz_scale` | 0, 8 | Noise displacement increasing toward the tip. |
| `clump`, `clump_count`, `clump_shape` | 0, count/40, 1 | Pull strands toward clump guides; `clump_shape` > 1 keeps roots looser. |
| `groom` | [] | Regions `{"center", "radius", "weight", "length" (scale), "comb", "comb_strength", "gravity"}` with smooth falloff, for partings, cowlicks and longer manes. |
| `mask` | – | Where strands grow: `{"axis": "y", "min": .., "max": ..}`, `{"center": [...], "radius": r}`, or a list; `"invert": true` flips it. |
| `dynamics` | false | Strand dynamics: Verlet with inextensible segments and root-to-tip stiffness toward the styled shape. |
| `stiffness`, `damping`, `substeps` | 0.15, 0.05, 4 | Dynamics controls. |
| `colliders` | [] | Collider nodes (names or `{"node": ...}`); analytic for primitives, exact signed-distance grid otherwise. |
| `ground` | global `dynamics.ground` | Collide strands with the ground plane. |
| `start_frame` | first frame | Dynamics start. |
| `preview_fraction` | 0.25 | Fraction of strands drawn in previews (`animora preview`, `render --preview`). |
| `seed` | 0 | Random seed. |

The strand mesh carries `uv`: `u` is a per-strand random value and `v` goes from 0 at the root to 1
at the tip. The `hair` and `fur` material presets use a gradient on `v` (dark roots, lighter
tips) with sheen. Use your own `gradient` texture on `v` for dyed tips.

### Jiggle (secondary motion)

Add `"jiggle": true` or `{"stiffness": 120, "damping": 8, "amount": 1.0, "gravity": 0}` to any
node. A damped spring makes it lag behind and overshoot its animated position. Use it for
antennae, bellies, ponytails and props.

### Baking

```bash
animora bake scene.json -o scene.cache.npz
```

Then set `"dynamics": {"cache": "scene.cache.npz"}`. Renders load the cache instead of
re-simulating, which is useful for distributed rendering or for frames rendered out of order.

`bake` also simulates every smoke volume and liquid and writes each one's own `cache` file. When
a node has no `cache`, the file is `<scene>.<node>.<kind>.npz`; add that path as the node's
`cache` to reuse it.

---

## 19. Rendering in depth

* **Algorithm.** Unidirectional path tracing.
  * Next-event estimation for every light type and emissive triangle, combined with BSDF sampling
    through multiple importance sampling (power heuristic). Russian roulette after 3 bounces.
  * Owen-scrambled Sobol sequences provide low-discrepancy sampling.
* **Acceleration.** A binned-SAH BVH, rebuilt per frame. It stores two time samples per vertex for
  deformation motion blur.
* **Light selection.** Lights are picked by a mix of their power and their solid-angle estimate,
  so dim lights close to a surface are not starved and big suns do not dominate.
* **Clean images.**
  * Path regularisation (when `caustics` is false) roughens glossy lobes after diffuse bounces,
    which removes caustic fireflies.
  * `clamp_indirect` clamps outliers.
  * Adaptive sampling (`adaptive_threshold`, `min_samples`) spends samples where noise is.
* **Denoiser.** An edge-avoiding à-trous wavelet filter. It is guided by albedo, normal, depth and
  per-pixel variance AOVs and demodulates albedo so textures stay sharp.
* **Glass and SSS.** Rough and smooth dielectrics with Fresnel, Beer absorption inside volumes,
  thin-walled mode, and random-walk subsurface scattering with per-channel radius.
* **Motion blur.** Transforms and deformation are sampled at shutter open and close and
  interpolated per ray time. This works for skinned, simulated and particle geometry too.
* **Depth of field.** A thin lens with polygonal (bladed) apertures; focus can follow a node.
* **Debug integrators.** `direct`, `ao`, `clay` (everything grey; good for checking lighting),
  `normals`, `albedo`, `depth`, `uv` and `wireframe`.
* **Performance.** The renderer is multithreaded C++ called through ctypes, with tile-based
  progress. On an Apple M2, a 1280×720 scene at 64 spp with GI typically takes from a few seconds
  to under a minute per frame on the CPU, depending on glass, SSS and light complexity. The GPU is
  about 4× faster (`animora doctor --bench`).

### GPU rendering

On macOS, Animora also has an Apple Metal GPU path tracer: a line-by-line port of the CPU
integrator (the same BSDF, light sampling and MIS, Sobol sampler, fog, random-walk SSS, motion blur,
depth of field, adaptive sampling and AOVs). The BVH and light tables are built by the same C++ code
and shared with the GPU, so the CPU and GPU converge to the same image; only the noise pattern
differs.

```bash
animora render shot.json --device gpu          # or -d gpu
animora render shot.json --device auto         # GPU when available, otherwise CPU
animora render shot.json --set device=gpu      # same thing through settings
animora doctor --bench                         # check the GPU and compare CPU vs GPU speed
```

* Put `"device": "gpu"` in `settings` to make it the scene default.
* The Metal library is compiled automatically on first use (or with `animora build`); the kernel is
  compiled by the Metal driver when the first frame starts.
* Scenes containing volumes (fog boxes, smoke, fire, clouds) fall back to the CPU automatically
  with a warning, as does any machine without Metal. The render log and `info["device"]` record
  which device actually rendered.
* From Python: `animora.render.native.gpu_info()` returns `{"available", "device", "error"}`.

---

## 20. Video and image output

**Image formats** are chosen by the output extension:

| format | notes |
|---|---|
| `.png` | 8- or 16-bit (`bit_depth`), RGBA with `film_transparent`. |
| `.jpg` | 8-bit. |
| `.tif` | 8- or 16-bit. |
| `.bmp` | – |
| `.exr` | 32-bit float linear, with AOV layers. Written by Animora's own EXR codec (no OpenEXR library needed); EXR files can also be read. |
| `.hdr` | Radiance RGBE. |
| `.pfm` | Portable float map. |

**Video.** Set `settings.video` (`"render/shot.mp4"`) or pass `render --video`. The codec is
inferred from the extension: `.mp4` → h264, `.mov` → prores, `.webm` → VP9, `.gif`, `.mkv`.
Override with `--codec`.

| codec | use case |
|---|---|
| h264 | Universal playback. |
| h265 | Smaller files. |
| prores | Editing. |
| prores4444 | Editing with alpha. |
| webm | VP9, web, supports alpha. |
| av1 | Modern, small. |
| gif | Palette-optimised. |
| png | PNG-in-MOV, lossless with alpha. |

Audio can be added with `--audio` or the `audio` key of a `video` dict. Pixel formats are chosen for maximum player
compatibility (yuv420p for h264 and h265).

`render --log FILE` writes a JSON log with the settings, the frame list, per-frame timings and
statistics, and the native renderer version.

---

## 21. Python API

Everything the CLI does is available as a library.

```python
import animora as an

# load and render
scene = an.load_scene("examples/bouncing_ball.json", overrides={"samples": 32})
res = an.render_frame(scene, 12)                 # dict: rgba (linear), display (uint8), aovs, info
an.save_frame(res, "ball_12.exr", scene, aovs=["albedo", "normal", "depth"])
paths = an.render_sequence(scene, frames=range(1, 49), output="out/ball_####.png")
an.encode("out/ball_####.png", "out/ball.mp4", fps=scene.fps, quality="high")

# build a scene in code
from animora import Scene
s = Scene({"settings": {"resolution": [1280, 720], "frame_range": [1, 24]}})
s.add_material("red", preset="plastic", base_color="#d33")
s.add_mesh("floor", primitive="plane", size=20, material="clay")
ball = s.add_mesh("ball", primitive="uv_sphere", radius=0.5, material="red")
ball.key("translate", 1,  [0, 3, 0])
ball.key("translate", 12, [0, 0.5, 0], "ease_in")
ball.key("translate", 24, [0, 3, 0], "ease_out")
s.add_camera("cam", translate=[0, 2, 8], look_at="ball")
s.add_light("sun", light="sun", rotate=[-50, 30, 0], intensity=3)
print(s.get_value("ball.translate.y", 12))      # 0.5
s.save("ball.json")                              # round-trips to a scene file
```

**`Scene` methods:**

| method | purpose |
|---|---|
| `Scene(data=None, base_dir=".")` | Create a scene from a dict. |
| `add(type, name, parent, **spec)` | Add any node; returns the `Node`. |
| `add_mesh(name, primitive, material, **spec)` | Add a mesh. |
| `add_camera(name, **spec)` | Add a camera. |
| `add_light(name, light, **spec)` | Add a light. |
| `add_joint(name, parent, **spec)` | Add a joint. |
| `add_ik(start, end, target, pole, solver, **kw)` | Add an IK handle. |
| `add_material(name, spec=None, **kw)` | Add a material. |
| `remove(name)` | Remove a node. |
| `node(name)` / `scene[name]` | Look up a node. |
| `children(name)` | List a node's children. |
| `frames()` | The frame list. |
| `fps`, `frame_start`, `frame_end` | Timeline values. |
| `state(frame)` | Evaluated `FrameState`: `.world(name)`, `.position(name)`, `.local(name)`. |
| `world_matrix(name, frame)` | 4×4 world matrix. |
| `get_value("node.attr.x", frame)` | Read any attribute. |
| `world_mesh(name, frame)`, `object_mesh(name, frame)` | Evaluated geometry (`Mesh`). |
| `validate(frames)` | Errors list. |
| `summary()`, `tree()` | Text reports. |
| `to_dict()`, `save(path)` | Serialise. |
| `invalidate()` | Clear caches after editing specs. |

**`Node` methods:**

* `key(attr, frame, value, interp=None)` sets a keyframe.
* `set(attr, value)` sets a constant, an expression or a key spec.
* `p(key, default, frame)` evaluates any parameter.

**Other exports:**

* `make_primitive(name, **params)` and `merge(meshes)`.
* `Mesh`, with `verts`, `faces`, `uvs`, `transformed()` and `to_triangles()`.
* `read_mesh(path)` (returns `(mesh, materials)`) and `write_mesh(path, mesh)`.
* `read_image` and `write_image`.
* `AnimCurve`, `Key`, `get_easing(name)` and `EASINGS`.
* `Param(spec).evaluate(frame, ctx)`.
* `Material` and `MATERIAL_PRESETS`.
* `build_packed(scene, frame)`, the renderer input.
* `probe(path)`, which wraps ffprobe.

---

## 22. Examples

The `examples/` directory contains:

| file | shows |
|---|---|
| `bouncing_ball.json` | Keyframes with easing, cycling, squash & stretch via expressions, motion blur. |
| `materials_showcase.json` | A grid of presets (glass, gold, SSS, car paint, velvet, ...) under an HDRI-like sky. |
| `constraints_expressions.json` | Aim and path constraints, expression-driven animation, camera follow. |
| `ik_arm.json` | A 3-joint arm with two-bone IK, pole vector, a dual-quaternion skinned cylinder, animated target. |
| `cloth_drape.json` | A silk sheet draping over a sphere; a pinned flag in turbulent wind. |
| `particles_fountain.json` | An emissive spark fountain with curl noise, ground collisions and bloom. |
| `rigid_bodies.json` | A crate stack knocked over by a wrecking ball; bouncing spheres. |
| `fur_ball.json` | A bouncing ball covered in 12k groomed, clumped, curly fur strands with strand dynamics. |
| `liquid_dam_break.json` | A PBF dam break plus a side stream crashing around a rock collider, meshed and rendered as water. |
| `modeling.json` | Procedural modelling: an SDF CSG part, 3D text, a swept horn and a lofted vase. |
| `smoke_colliders.json` | Smoke from the native solver rising through a torus ring and pushed by a swinging paddle (exact mesh colliders). |
| `cloth_tear.json` | A heavy ball smashing through a sheet stretched between two bars; the fabric rips open. |
| `retarget_walk.json` | HumanIK-style retargeting: a T-pose BVH walk (HumanIK names, cm) drives an A-pose mannequin (Unreal names, m) with foot locking. |
| `weight_paint.json` | Scripted weight painting on a two-arm span: mirrored brush strokes, smoothing, mirror, prune and locked joints; elbows bend cleanly with DQ skinning. |

```bash
animora render examples/materials_showcase.json -f 36 --show
animora render examples/cloth_drape.json --preview
animora play examples/ik_arm.json
animora bake examples/liquid_dam_break.json      # simulate the water once, then render frames in any order
animora render examples/fur_ball.json -f 24 --preview
animora render examples/cloth_tear.json --device gpu      # GPU path tracing
animora render examples/retarget_walk.json --preview
animora weights examples/weight_paint.json -j l_elbow --show

# Motion capture: import a BVH, render it, or export it to glTF for other tools
animora import walk.bvh -o walk.json --scale 0.01
animora render walk.json --preview
animora export walk.json walk.glb
```

---

## 23. Comparison with Autodesk Maya

Animora targets the part of Maya most people use to make a finished animated shot: modelling,
keyframe animation, rigging, simulation, and a photorealistic renderer. It replaces the GUI with
text files and a terminal. Below is an honest assessment.

| area | Maya (+ Arnold, Bifrost, XGen) | Animora |
|---|---|---|
| Interface | Full interactive GUI, viewport, Graph Editor, Outliner | **Command line only, by design**: text scenes, terminal previews and flipbooks. `graph` draws curves in the terminal or to PNG; `key` / `keys` edit and list keys. |
| Scripting / automation | MEL, Python (maya.cmds, PyMEL), headless `mayapy` and `Render` | Scenes **are** code (JSON / YAML / Python); full Python API; everything is scriptable and diffable in git |
| Rendering | Arnold: unidirectional PT, MIS, SSS, volumes, hair, AOVs, OIDN/OptiX denoisers, CPU + GPU | Unidirectional PT with MIS, principled BSDF, random-walk SSS, glass with absorption, **heterogeneous volumes** (delta and ratio tracking, multiple scattering, blackbody fire), physical sky, HDRI importance sampling, fog, DOF, motion blur, AOVs, guided denoiser. **Comparable image quality**; multithreaded CPU and **Apple Metal GPU** (about 4× faster on an M2). No CUDA / OptiX. |
| Colour pipeline | OCIO / ACES configs | Built-in OCIO-style management: `lin_srgb`, `acescg`, `lin_rec2020` and `lin_p3` working spaces; per-texture, per-colour and per-HDRI input spaces; display spaces (sRGB, Rec.709, Display P3, Rec.2020, Adobe RGB); Bradford adaptation; ACES2065-1 EXR output; AgX, ACES, Filmic and Khronos Neutral view transforms; `.cube`, `.spi3d`, `.spi1d` and `.3dl` LUTs. It cannot load an OCIO `config.ocio` file. |
| Keyframe animation | Graph Editor, tangents, layers, Time Editor | Auto, spline, clamped, flat, linear, step and Bézier tangents; 40 easings; infinity cycling; additive and multiplicative layers; **driven keys**. Curves are edited from the CLI (`key set/move/tangent/infinity`), viewed with `graph`. |
| Expressions | MEL expressions, node graph | Sandboxed expression language with noise, wiggle, spring, bounce and `get`/`at` cross-references |
| Constraints | Aim, point, orient, parent, scale, motion path, pole, ... | Aim, point, orient, parent, scale, path, limit and copy transforms, all multi-target and weight-animated |
| Rigging | HumanIK, IK solvers, skin clusters, blend shapes, set driven keys, deformers, node editor | Two-bone, FABRIK and CCD IK with pole, stretch, soft IK and twist; LBS and DQ skinning with auto weights; blend shapes; driven keys; lattice. BVH mocap import with **HumanIK-style retargeting** (auto-characterisation of HumanIK, Mixamo, Unreal, Rigify, CMU and SMPL names, rest-pose matching, foot locking). **Scripted weight painting** (replace / add / scale / smooth brushes, mirror, prune, hold weights, heat maps, import / export). No node editor, and painting is scripted rather than interactive. |
| Modelling | Full polygon, NURBS and subdivision toolset with interactive tools | 20+ parametric primitives, **SDF CSG modelling** (smooth union, subtract, intersect, shell, round, twist, bend), metaballs, 3D text, sweeps and lofts, mesh **booleans**, remesh, decimate, extrude, and 25+ procedural modifiers. Procedural, not interactive: detailed sculpting and hand-built topology still belong in a dedicated modeller, imported by OBJ, PLY or glTF. |
| Dynamics | Bullet, nCloth, nParticles, Bifrost fluids, XGen hair | Rigid bodies (sphere, box, plane, capsule, cylinder, **exact mesh**); PBD cloth with **self-collision** and **exact mesh colliders**; particles with fields and mesh colliders; **grid smoke and fire** (multithreaded native solver, exact moving mesh colliders); **cloth tearing**; **PBF liquids** with meshed surfaces; **hair and fur** with grooming and strand dynamics; jiggle. All deterministic and bakeable. Bifrost's sparse FLIP/APIC and aero solvers scale further, and nCloth has more constraint types. |
| Pipeline | USD, Alembic, FBX, referencing | Import: glTF/GLB (meshes, skins, animation, morph targets, cameras, lights), BVH, OBJ, PLY. Export: animated **glTF/GLB** and **USD (`.usda`)**, OBJ/PLY sequences. Scene includes for referencing; EXR AOVs. **No Alembic, FBX or USD import.** |
| Cost / footprint | Subscription; multi-GB install | Free (MIT); about 23k lines of Python, C++ and Metal; numpy and Pillow only |

**Where Animora is at or above Maya's level:**

* Final-pixel quality of product, motion-graphics, effects and stylised character shots: path
  tracing with SSS, glass, volumes and fire, DOF, motion blur, AgX and ACES.
* Reproducibility and automation: every shot is a text file, so it can be versioned, diffed,
  templated and rendered on any headless machine with one command.
* Procedural animation and modelling through expressions, SDFs and Python-generated scenes.
* Out-of-the-box effects: smoke, fire, liquids and fur need no plugins or separate renderer setup.

**Where Maya is still ahead:**

* Interactive work: posing, brush-by-hand weight painting, sculpting and hand-editing curves all
  depend on a GUI. Animora deliberately has none; weights are painted with scripted strokes.
* Production scale: Bifrost-class sparse solvers for very large sims, CUDA / OptiX GPU rendering on
  Linux and Windows (Animora's GPU path is Metal only), Alembic, FBX and USD layering and
  referencing, and the plugin ecosystem.

Choose Animora for automated, scripted, reproducible animation, simulation and rendering. Choose
Maya for interactive character work at film-pipeline scale.

---

## 24. Limitations and tips

**Limitations:**

* No GUI or interactive viewport, by design. Terminal previews are low resolution; use
  `render --preview` or `--show` for a proper look.
* GPU rendering uses Apple Metal, so it runs on macOS only; elsewhere, and for scenes with
  volumes, rendering uses the CPU.
* Simulations are dense-grid or particle based on the CPU. Smoke uses a multithreaded native
  solver, so 100³–150³ grids are practical, but there is no sparse grid, so very large domains need
  a lot of memory. Liquids are practical up to a few hundred thousand particles.
* Pipeline: no Alembic or FBX; USD is export-only (`.usda` text). OCIO `config.ocio` files cannot be
  loaded; the built-in spaces and LUT files cover the common ACES and wide-gamut workflows.
* Retargeting covers the body (spine, neck, head, arms, legs); fingers transfer only when the joint
  names match.

**Tips:**

* Iterate with `preview` or `play`. Use `render --preview` for blocking passes, and
  `--integrator clay` to judge lighting without materials.
* Run `validate` before long renders; it catches typos in node, material and attribute names.
* Glass and caustics look clean with `caustics: false` (the default). Turn caustics on and raise
  `samples` only when you need light focused through glass.
* Noisy interiors: raise `samples`, enable `adaptive_threshold: 0.01`, or use area lights with
  `power` instead of tiny point lights.
* Use `--resume` and frame lists (`-f 1-240x10`, then fill in) for long shots.
* Bake dynamics, smoke and liquids (`animora bake`) before splitting a render across machines.
* Liquids: start with a coarse `spacing` (0.06 to 0.08) to block the motion, then refine. Each
  halving of `spacing` gives 8× the particles.
* Fur: block grooming with `preview` (it shows a quarter of the strands), then render the full count.
* Put shared materials and rigs in a separate file and use `include`.
* Weight painting: check every stroke with `animora weights SCENE -j JOINT --show` (a heat map),
  then bend the joint with `preview`. Use `mirror` strokes so you only paint one side.
* Retargeting: run `animora retarget SRC TARGET --list-map` first, then fix any wrong joints with
  `map`.
* Use `--device gpu` for look development on a Mac; the GPU and CPU converge to the same image.

---

## 25. Project layout and tests

```
animora/
  cli.py            command-line interface (all commands above)
  cli_extra.py      import / graph / key / keys / colorspaces / retarget / weights commands
  scene.py          scene graph, evaluation, settings, cameras/lights, packing for the renderer
  params.py         animatable parameter values (constants / keys / expressions / layers)
  curves.py         keyframe curves, tangents, easing library, infinity modes
  expressions.py    sandboxed expression evaluator
  constraints.py    aim/point/orient/parent/scale/path/limit/copy constraints
  rig.py            IK solvers, skinning (LBS/DQ, auto weights), blend shapes
  weightpaint.py    scripted weight painting: brushes, mirror, smooth, prune, heat maps, import/export
  retarget.py       HumanIK-style characterisation and motion retargeting, foot locking
  dynamics.py       rigid bodies, cloth (self-collision, tearing), particles, jiggle, mesh SDF colliders, bake/cache
  volumes.py        heterogeneous volumes (procedural / grid / smoke) packed for the renderer
  smoke.py          smoke and fire grid solver (native or numpy), mesh-voxelised moving colliders
  liquid.py         PBF liquid simulation and surface meshing
  hair.py           hair and fur: growth, grooming, strand dynamics, strand geometry
  geometry.py       Mesh class, primitives, subdivision, normals
  modeling.py       SDF / CSG, metaballs, text, sweep, loft, surface nets, boolean / remesh / decimate
  modifiers.py      modifier / deformer stack
  materials.py      principled material model, presets, textures, colour parsing
  color.py          colour management: colour spaces, working / display spaces, LUT loaders
  graph.py          terminal and PNG curve plots, key editing
  noise.py          Perlin/simplex/fbm/curl noise
  mathutil.py       matrices, quaternions, Euler orders
  output.py         ffmpeg encoding, sequence handling
  terminal.py       true-colour terminal image drawing and flipbook player
  io/images.py      PNG/JPEG/TIFF/BMP/EXR/HDR/PFM read and write
  io/obj.py         OBJ/MTL and PLY read and write
  io/gltf.py        glTF 2.0 / GLB read, scene import and animated export
  io/usd.py         USD (.usda) animated export
  io/exchange.py    animated-scene sampling shared by the glTF and USD exporters
  io/bvh.py         BVH motion-capture import
  render/renderer.py  render_frame / render_sequence / save_frame
  render/native.py    builds and binds the C++ renderer and the Metal GPU library (ctypes)
  render/env.py       sky / HDRI / gradient / studio environments
  render/tonemap.py   view transforms and post-processing
  native/pathtracer.cpp  the CPU path tracer and the native smoke solver
  native/gpu_metal.mm    Metal host code (buffers, dispatch, progressive accumulation)
  native/gpu_kernel.msl  the GPU path-tracing kernel (port of the CPU integrator)
examples/           sample scenes
tests/              unit tests
```

Run the tests (64 tests, about 12 seconds):

```bash
python3 -m unittest discover -s tests -v
```

The tests cover curves and easings, expressions, the parameter forms, transforms and constraints,
IK reach and pole behaviour, skinning, modifiers, primitives, OBJ round-trips, image I/O
(including EXR), dynamics determinism, scene save/load round-trips, and native renders against
reference statistics. `test_improvements.py` covers driven keys, the graph and key commands,
colour management and LUTs, volumes, smoke, liquids, hair, procedural modelling and booleans,
cloth self-collision, rigid capsule, cylinder and mesh shapes, glTF round trips (including morph
targets), USD export (checked with `usdchecker` when installed), BVH import and file sequences. `test_features.py` covers the GPU renderer against the CPU (skipped
without Metal), the native smoke solver against numpy, mesh-voxelised smoke colliders, cloth
tearing, retargeting (name matching for every convention, exact pose transfer onto a rotated,
renamed skeleton, the BVH scene block) and weight painting (strokes, mirror, locks, global
operations, import / export and transfer, heat maps).

License: MIT (see [LICENSE](LICENSE)).
