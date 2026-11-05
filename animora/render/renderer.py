"""High-level rendering: one frame -> image(s), or a whole frame range -> image sequence (+ video).

Pipeline per frame::

    Scene --build_packed--> PackedScene --native path tracer--> linear RGBA + AOVs
          --denoise (edge-aware, AOV guided)--> --post / tonemap--> display image --> file(s)

Everything is driven from ``scene.settings`` (see :data:`animora.scene.DEFAULT_SETTINGS`) and
the optional top-level ``post`` dictionary (grading keys of :func:`animora.render.tonemap.process`).
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np

from . import native as nat
from . import tonemap
from ..io.images import write_image, HDR_EXTS

AOV_NAMES = ("albedo", "normal", "depth", "variance", "spp", "alpha", "noisy", "position")

_PAD_RE = re.compile(r"(#+)")
_PRINTF_RE = re.compile(r"%(0?\d*)d")
_FMT_RE = re.compile(r"\{frame(?::([^}]*))?\}")


def frame_path(pattern, frame):
    """Expand a frame-number pattern.

    Supports ``####`` (padding = number of #), ``%04d`` and ``{frame:04d}``. Fractional
    frames (sub-frame steps) are written as ``0012.500``. A pattern without any placeholder
    gets ``_####`` inserted before the extension.
    """
    pattern = str(pattern)
    whole = float(frame).is_integer()
    fi = int(math.floor(frame))

    def fmt(width):
        s = f"{fi:0{width}d}" if fi >= 0 else f"-{abs(fi):0{max(1, width - 1)}d}"
        if not whole:
            s += f"{frame - fi:.3f}"[1:]
        return s

    if _PAD_RE.search(pattern):
        return _PAD_RE.sub(lambda m: fmt(len(m.group(1))), pattern, count=1)
    if _PRINTF_RE.search(pattern):
        return _PRINTF_RE.sub(lambda m: fmt(int(m.group(1) or 0)), pattern, count=1)
    if _FMT_RE.search(pattern):
        def rep(m):
            spec = m.group(1) or "d"
            w = re.match(r"0?(\d+)", spec)
            return fmt(int(w.group(1)) if w else 0)
        return _FMT_RE.sub(rep, pattern, count=1)
    root, ext = os.path.splitext(pattern)
    return f"{root}_{fmt(4)}{ext}"


def ffmpeg_pattern(pattern):
    """Convert a ``####`` pattern into ffmpeg's ``%04d`` form."""
    pattern = str(pattern)
    if _PAD_RE.search(pattern):
        return _PAD_RE.sub(lambda m: f"%0{len(m.group(1))}d", pattern, count=1)
    if _PRINTF_RE.search(pattern):
        return pattern
    if _FMT_RE.search(pattern):
        def rep(m):
            w = re.match(r"0?(\d+)", m.group(1) or "")
            return f"%0{w.group(1)}d" if w else "%d"
        return _FMT_RE.sub(rep, pattern, count=1)
    root, ext = os.path.splitext(pattern)
    return f"{root}_%04d{ext}"


def post_settings(scene):
    """Merge the grading keys from ``settings`` and the top-level ``post`` block."""
    s = dict(scene.settings)
    s.update(scene.data.get("post", {}) or {})
    return s


def native_settings(scene, width, height, spp, preview=False, overrides=None):
    """Build :class:`native.AnSettings` from scene settings (``overrides`` wins)."""
    st = dict(scene.settings)
    st.update(overrides or {})
    s = nat.default_settings(int(width), int(height), int(spp))
    s.min_spp = int(st.get("min_samples", 0) or 0)
    s.adaptive_threshold = float(st.get("adaptive_threshold", 0.0) or 0.0)
    s.max_depth = int(st.get("max_bounces", 8)) if not preview else min(4, int(st.get("max_bounces", 8)))
    s.threads = int(st.get("threads", 0) or 0)
    s.seed = int(st.get("seed", 1)) & 0xFFFFFFFF
    s.clamp_indirect = float(st.get("clamp_indirect", 10.0) or 0.0)
    integ = st.get("integrator", "path")
    if integ not in nat.INTEGRATORS:
        raise ValueError(f"unknown integrator {integ!r}; available: {sorted(nat.INTEGRATORS)}")
    s.integrator = nat.INTEGRATORS[integ]
    s.film_transparent = 1 if st.get("film_transparent") else 0
    s.caustics = 1 if st.get("caustics") else 0
    s.filter_width = float(st.get("filter_width", 1.5))
    s.ao_distance = float(st.get("ao_distance", 1.0))
    s.wire_width = float(st.get("wire_width", 1.0))
    s.rr_depth = int(st.get("rr_depth", 3))
    s.tile_size = int(st.get("tile_size", 16))
    region = st.get("region")
    if region:
        x0, y0, x1, y1 = region
        if all(isinstance(v, float) and 0 <= v <= 1 for v in region):
            x0, x1 = x0 * width, x1 * width
            y0, y1 = y0 * height, y1 * height
        s.region_x0, s.region_y0 = int(x0), int(y0)
        s.region_x1, s.region_y1 = int(x1), int(y1)
    return s


def resolution(scene, preview=False, scale=None):
    w, h = scene.settings.get("resolution", [960, 540])
    sc = scale if scale is not None else (float(scene.settings.get("preview_scale", 0.5)) if preview else 1.0)
    return max(1, int(round(w * sc))), max(1, int(round(h * sc)))


def render_frame(scene, frame, preview=False, samples=None, scale=None, denoise=None, progress=None,
                 overrides=None):
    """Render one frame. Returns a dict::

        rgba      (H,W,4) linear, denoised radiance + alpha
        display   (H,W,4) tone-mapped / graded display values 0-1
        aovs      {albedo, normal, depth, variance, spp, noisy}
        info      {triangles, objects, lights, camera, time_build, time_render, time_total, spp}
        cancelled bool
    """
    t0 = time.time()
    st = dict(scene.settings)
    st.update(overrides or {})
    W, H = resolution(scene, preview, scale)
    if samples is None:
        samples = int(st.get("preview_samples", 8)) if preview else int(st.get("samples", 64))
    packed, info = build(scene, frame, preview, st, W / H)
    t1 = time.time()
    s = native_settings(scene, W, H, samples, preview, overrides)
    out = nat.render(packed, s, progress=progress, aovs=True, device=str(st.get("device", "cpu") or "cpu"))
    t2 = time.time()
    rgba = out["rgba"]
    aovs = {k: out[k] for k in ("albedo", "normal", "depth", "variance", "spp")}
    aovs["noisy"] = rgba[..., :3].copy()
    do_dn = st.get("denoise", True) if denoise is None else denoise
    if do_dn and st.get("integrator", "path") in ("path", "direct", "ao") and samples < 4096 and packed.num_tris:
        rgb = nat.denoise(rgba[..., :3], aovs["albedo"], aovs["normal"], aovs["depth"], aovs["variance"],
                          iterations=int(st.get("denoise_iterations", 5)), threads=s.threads)
        strength = float(st.get("denoise_strength", 1.0))
        if strength < 1.0:
            rgb = rgb * strength + rgba[..., :3] * (1 - strength)
        rgba = np.concatenate([rgb, rgba[..., 3:]], -1)
    ps = post_settings(scene)
    ps.update(overrides or {})
    if st.get("integrator", "path") in ("normals", "uv", "depth", "albedo"):
        display = np.clip(rgba, 0, 1)
        if st.get("integrator") == "albedo":
            display[..., :3] = tonemap.srgb_encode(display[..., :3])
    else:
        display = tonemap.process(rgba, ps, frame=frame, base_dir=scene.base_dir)
    t3 = time.time()
    info.update(device=out.get("device", "cpu"), time_build=t1 - t0, time_render=t2 - t1, time_total=t3 - t0, spp=samples, width=W, height=H,
                mean_spp=float(np.mean(out["spp"])) if out.get("spp") is not None else samples)
    return {"rgba": rgba, "display": display, "aovs": aovs, "info": info, "cancelled": out["cancelled"]}


def build(scene, frame, preview, st, aspect):
    from ..scene import build_packed
    mb = st.get("motion_blur", False) and not preview
    return build_packed(scene, frame, preview=preview, motion_blur=mb, aspect=aspect)


def save_frame(result, path, scene, aovs=None, bit_depth=None, frame=0):
    """Write a render result. LDR formats get the display image; ``.exr/.hdr/.pfm`` get linear
    radiance. ``aovs`` (list) adds layers to an EXR, or writes a sidecar ``<name>.aovs.exr``."""
    path = Path(path)
    ext = path.suffix.lower()
    st = scene.settings
    bit_depth = int(bit_depth or st.get("bit_depth", 8))
    alpha = bool(st.get("film_transparent"))
    aovs = list(aovs if aovs is not None else st.get("aovs", []) or [])
    for a in aovs:
        if a not in AOV_NAMES:
            raise ValueError(f"unknown AOV {a!r}; available: {AOV_NAMES}")
    layers = {}
    for a in aovs:
        if a == "alpha":
            layers["alpha"] = result["rgba"][..., 3]
        elif a == "position":
            continue
        else:
            layers[a] = np.asarray(result["aovs"][a], np.float32)
    if ext in HDR_EXTS:
        from .. import color as cm
        img = result["rgba"] if (alpha or ext == ".exr") else result["rgba"][..., :3]
        work = cm.canonical(st.get("working_space") or "lin_srgb")
        out_space = cm.canonical(st.get("exr_space") or work)
        if out_space != work:
            img = np.asarray(cm.convert(img, work, out_space), np.float32)
        write_image(path, img, layers=layers if ext == ".exr" else None,
                    chromaticities=cm.exr_chromaticities(out_space) if ext == ".exr" else None)
    else:
        img = result["display"] if alpha and ext in (".png", ".tif", ".tiff", ".webp") else result["display"][..., :3]
        if bit_depth == 8 and st.get("dither", True):
            img = tonemap.to_uint8(img, dither=True, seed=int(frame * 7 + 3)).astype(np.float32) / 255.0
        write_image(path, img, bit_depth=bit_depth)
        if layers:
            side = path.with_suffix("").as_posix() + ".aovs.exr"
            write_image(side, result["rgba"], layers=layers)
    return path


def render_sequence(scene, frames=None, output=None, preview=False, samples=None, scale=None, resume=False,
                    progress=None, on_frame=None, overrides=None, aovs=None, stop=None):
    """Render a list of frames to an image sequence. Returns a list of written paths.

    ``resume`` skips frames whose output file already exists. ``progress(frame_idx, n, frac)``
    is called during rendering; ``on_frame(frame, path, result)`` after each frame is saved.
    ``stop()`` returning True aborts cleanly after the current frame.
    """
    frames = list(frames if frames is not None else scene.frames())
    pattern = output or scene.settings.get("output", "render/frame_####.png")
    pattern = scene.resolve_path(pattern) if not os.path.isabs(str(pattern)) else str(pattern)
    written = []
    n = len(frames)
    for i, f in enumerate(frames):
        path = frame_path(pattern, f)
        if resume and os.path.exists(path) and os.path.getsize(path) > 0:
            written.append(path)
            if on_frame:
                on_frame(f, path, None)
            continue
        cb = (lambda frac, i=i: progress(i, n, frac)) if progress else None
        res = render_frame(scene, f, preview=preview, samples=samples, scale=scale, progress=cb, overrides=overrides)
        if res["cancelled"]:
            raise KeyboardInterrupt
        tmp = path + ".partial" + os.path.splitext(path)[1]
        save_frame(res, tmp, scene, aovs=aovs, frame=f)
        os.replace(tmp, path)
        side_tmp = Path(tmp).with_suffix("").as_posix() + ".aovs.exr"
        if os.path.exists(side_tmp):
            os.replace(side_tmp, Path(path).with_suffix("").as_posix() + ".aovs.exr")
        written.append(path)
        if on_frame:
            on_frame(f, path, res)
        if stop and stop():
            break
    return written


def write_metadata(path, scene, frames, stats):
    """Write a small JSON log next to a sequence (render settings + per-frame timings)."""
    d = {"settings": {k: v for k, v in scene.settings.items() if k != "output"}, "frames": frames,
         "stats": stats, "native": nat.version() if nat.available() else None}
    Path(path).write_text(json.dumps(d, indent=2, default=str))
