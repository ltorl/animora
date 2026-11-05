"""Heterogeneous volumes (``type: "volume"`` nodes): clouds, smoke, fire, dust, god-ray media.

A volume is a box (``size``, centred on the node origin, transformed by the node's world matrix)
filled with a density field from one of these sources:

* **procedural** (default): a ``shape`` (box / sphere / cylinder) with a soft ``falloff`` edge,
  carved by fBm noise (``frequency``, ``octaves``, ``gain``, ``lacunarity``, ``threshold``,
  ``contrast``, animatable ``offset`` for drifting / boiling clouds);
* **grid**: a dense voxel grid from ``grid`` = ``.npy`` / ``.npz`` file (``density`` + optional
  ``heat`` / ``temperature`` / ``flame`` arrays, z-y-x or x-y-z order via ``grid_order``) or an
  inline nested list;
* **smoke**: ``smoke: {...}`` runs the built-in smoke / fire solver (:mod:`animora.smoke`) and
  renders its density and heat at the current frame.

Rendering is unbiased delta tracking (camera / bounce rays) and ratio tracking (shadow rays)
in the native path tracer, with single + multiple scattering, Henyey-Greenstein ``anisotropy``,
coloured ``albedo``, constant ``emission`` and physically-based **blackbody** emission driven by
heat mapped onto ``temperature`` = [K_min, K_max].
"""
from __future__ import annotations

import os

import numpy as np

from . import mathutil as mu

SHAPES = {"box": 0, "cube": 0, "sphere": 1, "ball": 1, "cylinder": 2}

VOLUME_KEYS = ("size", "density", "albedo", "color", "anisotropy", "emission", "emission_strength",
               "blackbody", "blackbody_strength", "temperature", "heat_from_density", "shape", "falloff",
               "frequency", "octaves", "gain", "lacunarity", "threshold", "contrast", "offset", "grid",
               "grid_order", "density_channel", "heat_channel", "heat_scale", "density_scale", "smoke")


def blackbody_lut(t_min, t_max, n=64):
    """RGB (linear Rec.709, converted to the working space by the caller) for heat 0..1.

    Colour follows Planck's locus; brightness follows Stefan-Boltzmann (T^4) relative to
    ``t_max`` so cool smoke glows dull red and hot cores are white-yellow."""
    from .materials import kelvin_to_rgb
    out = np.zeros((n, 3), np.float64)
    lumw = np.array([0.2126, 0.7152, 0.0722])
    for i in range(n):
        h = i / (n - 1)
        T = t_min + (t_max - t_min) * h
        if h <= 0 or T <= 0:
            continue
        rgb = kelvin_to_rgb(max(T, 1000.0))
        rgb = rgb / max(float(rgb @ lumw), 1e-9)
        # fade below ~1000K where kelvin_to_rgb clamps
        out[i] = rgb * (T / t_max) ** 4 * min(1.0, h * 8)
    return out


def _load_grid(spec, base_dir):
    """Return (density, heat_or_None) float32 arrays indexed [z, y, x]."""
    g = spec.get("grid")
    order = str(spec.get("grid_order", "zyx")).lower()
    dch = spec.get("density_channel", "density")
    hch = spec.get("heat_channel")
    heat = None
    if isinstance(g, str):
        path = g if os.path.isabs(g) else os.path.join(base_dir or ".", g)
        if path.endswith(".npz"):
            z = np.load(path)
            if dch not in z:
                raise ValueError(f"volume grid {path}: no array {dch!r} (have {list(z.keys())})")
            dens = z[dch]
            for k in ([hch] if hch else ["heat", "temperature", "flame", "fuel"]):
                if k and k in z:
                    heat = z[k]
                    break
        else:
            dens = np.load(path)
    else:
        dens = np.asarray(g, np.float32)
    dens = np.asarray(dens, np.float32)
    if dens.ndim != 3:
        raise ValueError(f"volume grid must be 3-D, got shape {dens.shape}")
    if order == "xyz":
        dens = np.ascontiguousarray(dens.transpose(2, 1, 0))
        heat = None if heat is None else np.ascontiguousarray(np.asarray(heat).transpose(2, 1, 0))
    return dens, (None if heat is None else np.asarray(heat, np.float32))


_GRID_CACHE = {}


def pack_volume(scene, name, frame, ctx, data_offset):
    """Build an ``AnVolume`` for node ``name``. Returns (AnVolume, list_of_float_arrays)."""
    from .render import native as nat
    from .materials import parse_color
    from . import color as cm
    node = scene.nodes[name]
    sp = node.spec
    p = lambda k, d: node.p(k, d, frame, ctx)  # noqa: E731
    v = nat.AnVolume()
    size = np.asarray(p("size", [2.0, 2.0, 2.0]), float)
    if size.ndim == 0:
        size = np.full(3, float(size))
    W = scene.world_matrix(name, frame)
    box = W @ mu.translate(-size / 2) @ mu.scale(size)  # unit cube -> world
    inv = np.linalg.inv(box)
    for i in range(12):
        v.world_to_local[i] = float(inv[i // 4, i % 4])
    alb = parse_color(p("albedo", p("color", [0.8, 0.8, 0.8])))
    nat.set_vec(v.albedo, np.clip(alb, 0, 1))
    v.density = float(p("density", 1.0))
    nat.set_vec(v.emission, parse_color(p("emission", [1.0, 1.0, 1.0])))
    v.emission_strength = float(p("emission_strength", 0.0))
    v.anisotropy = float(np.clip(p("anisotropy", 0.0), -0.95, 0.95))
    v.blackbody_strength = float(p("blackbody_strength", p("blackbody", 0.0)))
    v.heat_from_density = float(p("heat_from_density", 0.0))
    v.shape = SHAPES.get(str(sp.get("shape", "sphere")).lower(), 1)
    v.falloff = float(p("falloff", 0.3))
    v.frequency = float(p("frequency", 3.0))
    v.octaves = int(p("octaves", 5 if "grid" not in sp and "smoke" not in sp else 0))
    v.gain = float(p("gain", 0.5))
    v.lacunarity = float(p("lacunarity", 2.0))
    v.threshold = float(p("threshold", 0.35))
    v.contrast = float(p("contrast", 2.5))
    nat.set_vec(v.offset, np.asarray(p("offset", [0.0, 0.0, 0.0]), float))
    v.heat_offset = -1
    arrays = []
    off = data_offset
    dens = heat = None
    if "smoke" in sp:
        from .smoke import smoke_frame
        dens, heat = smoke_frame(scene, name, frame)
    elif "grid" in sp:
        key = (name, str(sp.get("grid")) if isinstance(sp.get("grid"), str) else id(sp), sp.get("grid_order"))
        if key not in _GRID_CACHE:
            _GRID_CACHE[key] = _load_grid(sp, scene.base_dir)
        dens, heat = _GRID_CACHE[key]
    if dens is not None:
        dens = np.asarray(dens, np.float32) * float(p("density_scale", 1.0))
        v.type = 0
        v.nz, v.ny, v.nx = dens.shape
        v.density_offset = off
        arrays.append(dens.ravel())
        off += dens.size
        if heat is not None:
            heat = np.clip(np.asarray(heat, np.float32) * float(p("heat_scale", 1.0)), 0, None)
            v.heat_offset = off
            arrays.append(heat.ravel())
            off += heat.size
        v.majorant = float(v.density * max(float(dens.max()), 1e-6) * 1.0001)
        v.octaves = 0
    else:
        v.type = 1
        v.majorant = float(max(v.density, 1e-6))
    # blackbody table
    if v.blackbody_strength > 0:
        tr = p("temperature", [800.0, 2600.0])
        tr = np.atleast_1d(np.asarray(tr, float))
        t_min, t_max = (float(tr[0]), float(tr[-1])) if tr.size > 1 else (500.0, float(tr[0]))
        lut = cm.to_working(blackbody_lut(t_min, t_max), "lin_srgb")
        v.bb_offset = off
        v.bb_size = lut.shape[0]
        arrays.append(np.asarray(lut, np.float32).ravel())
        off += lut.size
    else:
        v.bb_offset, v.bb_size = 0, 0
    return v, arrays
