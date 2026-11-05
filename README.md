# Animora

**Command-line 3D animation, simulation and path-traced rendering. No GUI.**

You write a scene as a text file (JSON, YAML or Python), preview it in the terminal and render
finished frames and video from the shell.

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh
```

That's it. The script:

1. downloads Animora into `~/.animora`,
2. creates a private Python environment there (numpy, Pillow, PyYAML),
3. links the `animora` command into `~/.local/bin`,
4. compiles the native renderer (and the Metal GPU renderer on macOS).

Then check the install:

```bash
animora doctor
```

**Requirements:** macOS or Linux, Python 3.9+, and a C++ compiler (`xcode-select --install` on
macOS, `sudo apt install g++` on Debian/Ubuntu). `ffmpeg` is optional and only needed for video
files.

**Options** (environment variables, set before `sh`):

| variable | default | meaning |
|---|---|---|
| `ANIMORA_HOME` | `~/.animora` | Install location |
| `ANIMORA_BIN_DIR` | `~/.local/bin` | Where the `animora` command is linked |
| `ANIMORA_REF` | `main` | Branch or tag to install |
| `ANIMORA_PYTHON` | `python3` | Python interpreter to use |
| `ANIMORA_NO_BUILD` | – | Set to `1` to skip compiling the renderer during install |

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | ANIMORA_HOME=/opt/animora sh
```

**Update:** run the install command again.

**Uninstall:**

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh -s -- --uninstall
```

**From a clone** (for development):

```bash
git clone https://github.com/ltorl/animora.git && cd animora
pip install -e .
python3 -m unittest discover -s tests
```

## Quick start

```bash
animora new ball.json --template bounce   # write a starter scene
animora preview ball.json -f 12           # draw frame 12 in the terminal
animora play ball.json                    # flipbook in the terminal
animora render ball.json --preview        # fast draft render
animora render ball.json                  # final frames + video
animora render ball.json --device gpu     # render on the GPU (macOS / Metal)
```

The installer also downloads the example scenes into `~/.animora/src/examples`:

```bash
animora render ~/.animora/src/examples/cloth_tear.json --preview
```

## Features

* **Renderer:** a C++ path tracer plus an Apple Metal GPU port of it. Principled BSDF,
  subsurface scattering, glass, volumes, fire, depth of field, motion blur, AOVs, a denoiser, and
  AgX and ACES colour.
* **Animation:** keyframes with Maya-style tangents, about 40 easing curves, layers, driven keys,
  expressions and constraints.
* **Rigging:** IK, linear and dual-quaternion skinning, blend shapes, scripted weight painting, and
  HumanIK-style retargeting of BVH motion capture.
* **Modelling:** parametric primitives, SDF modelling (CSG), mesh booleans, sweeps, lofts, 3D text,
  and a modifier stack.
* **Simulation:** rigid bodies, cloth (with tearing), particles, smoke and fire with mesh colliders,
  liquids, and hair and fur.
* **Input and output:** glTF, BVH, OBJ and PLY import; glTF, USD, OBJ and PLY export. Images in
  PNG, EXR, HDR and more; video in H.264, H.265, ProRes, WebM, AV1 or GIF.

## Documentation

[**DOCS.md**](DOCS.md) is the full manual: every command, scene key, node type, material, modifier,
simulation and render setting, with examples. It also has a comparison with Autodesk Maya.

## License

MIT. See [LICENSE](LICENSE).
