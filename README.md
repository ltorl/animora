<div align="center">

<img width="1900" height="560" src="https://github.com/user-attachments/assets/cf92f664-74bd-4307-b3ea-2403c572a6a5" />

<br>
<br>

**Extremely high quality Command-line 3D animation, simulation and path-traced rendering. No GUI.**

Ai can write a scene as a text file (JSON, YAML or Python), preview it in the terminal and render
finished frames and video from the shell.

![No GUI](https://img.shields.io/badge/NO-GUI-d62828?style=for-the-badge)
![Terminal](https://img.shields.io/badge/FULLY-TERMINAL-e94525?style=for-the-badge)
![AI Optimized](https://img.shields.io/badge/AI-OPTIMIZED-f26522?style=for-the-badge)
![Text Based](https://img.shields.io/badge/SCENES-CODE_BASED-ff9d00?style=for-the-badge)

</div>

<br>
<br>
<br>

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/ltorl/animora/main/install.sh | sh
```

To check the install:

```bash
animora doctor
```

<br>
<br>

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

<br>
<br>

## <!-- AI AGENTS LOOK HERE: --> Documentation

[**DOCS.md**](DOCS.md) is the full manual: every command, scene key, node type, material, modifier,
simulation and render setting, with examples.

<br>
<br>

## License

MIT. See [LICENSE](LICENSE).
