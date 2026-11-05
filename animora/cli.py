"""``animora`` command-line interface.

Run ``animora --help`` or ``animora <command> --help``. Every command works headless; previews
are drawn directly in the terminal with ANSI colour.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from . import terminal as term


# ======================================================================================
# helpers
# ======================================================================================
def parse_frames(spec, scene=None):
    """Parse a frame specification.

    ``12``  ``1-48``  ``1-48x2`` (every 2nd)  ``1,5,9``  ``1-10,20-30``  ``all`` / empty (scene range)
    ``first`` / ``last`` / ``mid``. Fractional values are allowed (``1-2x0.25``).
    """
    if spec is None or str(spec).strip() in ("", "all"):
        return list(scene.frames()) if scene is not None else []
    out = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if scene is not None and part in ("first", "last", "mid"):
            out.append({"first": scene.frame_start, "last": scene.frame_end,
                        "mid": round((scene.frame_start + scene.frame_end) / 2)}[part])
            continue
        step = 1.0
        if "x" in part:
            part, s = part.split("x", 1)
            step = float(s)
        if "-" in part[1:]:
            i = part.index("-", 1)
            a, b = float(part[:i]), float(part[i + 1:])
            n = int(math.floor((b - a) / step + 1e-9)) + 1
            out += [a + k * step for k in range(max(0, n))]
        else:
            out.append(float(part))
    return [int(f) if float(f).is_integer() else f for f in out]


def parse_value(s):
    """Parse a ``--set`` value: JSON if possible (numbers, lists, true/false/null), else a string."""
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        low = str(s).lower()
        if low in ("on", "yes"):
            return True
        if low in ("off", "no"):
            return False
        return s


def parse_overrides(pairs):
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--set expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = parse_value(v.strip())
    return out


def parse_res(s):
    if s is None:
        return None
    presets = {"480p": [854, 480], "540p": [960, 540], "720p": [1280, 720], "hd": [1280, 720], "1080p": [1920, 1080],
               "fhd": [1920, 1080], "1440p": [2560, 1440], "2k": [2048, 1080], "4k": [3840, 2160],
               "uhd": [3840, 2160], "square": [1080, 1080], "vertical": [1080, 1920], "insta": [1080, 1350]}
    if s.lower() in presets:
        return presets[s.lower()]
    for sep in ("x", "X", "*", ","):
        if sep in s:
            w, h = s.split(sep)
            return [int(w), int(h)]
    raise SystemExit(f"bad resolution {s!r}; use WIDTHxHEIGHT or one of {sorted(presets)}")


def load(args, extra=None):
    from .scene import load_scene, SceneError
    ov = parse_overrides(getattr(args, "set", None))
    if getattr(args, "res", None):
        ov["resolution"] = parse_res(args.res)
    for k in ("samples", "threads", "seed", "integrator", "tonemap", "exposure", "fps", "bit_depth", "device"):
        v = getattr(args, k, None)
        if v is not None:
            ov[k] = v
    if getattr(args, "motion_blur", None):
        ov["motion_blur"] = True
    if getattr(args, "no_denoise", False):
        ov["denoise"] = False
    if getattr(args, "transparent", False):
        ov["film_transparent"] = True
    if getattr(args, "max_bounces", None) is not None:
        ov["max_bounces"] = args.max_bounces
    ov.update(extra or {})
    try:
        return load_scene(args.scene, overrides=ov)
    except SceneError as e:
        raise SystemExit(term.red(f"scene error: {e}"))


def _human(n):
    for u in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.0f}{u}" if u == "" else f"{n:.1f}{u}"
        n /= 1000.0
    return f"{n:.1f}T"


def _ensure_native():
    from .render import native as nat
    if not nat.available():
        term.log(term.yellow("building native path tracer (first run)..."))
        nat.build(verbose=False)
        if not nat.available():
            raise SystemExit(term.red("could not build the native renderer; run `animora doctor`"))


# ======================================================================================
# commands
# ======================================================================================
def cmd_render(args):
    from .render import renderer
    from . import output
    _ensure_native()
    sc = load(args)
    frames = parse_frames(args.frames, sc)
    if not frames:
        raise SystemExit("no frames to render")
    pattern = args.output or sc.settings.get("output", "render/frame_####.png")
    if args.output and not os.path.isabs(args.output):
        pattern = os.path.abspath(args.output)
    else:
        pattern = sc.resolve_path(pattern)
    Path(pattern).parent.mkdir(parents=True, exist_ok=True)
    if sc.dynamics is not None:
        t = time.time()
        bar = term.Progress("simulating")
        sc.dynamics.simulate(max(frames), progress=lambda f, e: bar.update(f / max(1, e), f"simulating frame {f}"))
        bar.done(term.dim(f"dynamics simulated in {time.time() - t:.1f}s") if not args.quiet else None)
    n = len(frames)
    W, H = renderer.resolution(sc, args.preview, args.scale)
    spp = args.samples or (sc.settings.get("preview_samples", 8) if args.preview else sc.settings.get("samples", 64))
    if not args.quiet:
        term.log(term.bold(f"Animora {__version__}") + f"  {Path(args.scene).name}: {n} frame(s) "
                 f"{frames[0]}..{frames[-1]}  {W}x{H}  {spp} spp  -> {pattern}")
    bar = term.Progress(enabled=not args.quiet)
    t_start = time.time()
    stats = []

    def prog(i, total, frac):
        bar.update((i + frac) / total, f"frame {frames[i]} ({i + 1}/{total})")

    def on_frame(f, path, res):
        if res is None:
            if not args.quiet:
                bar.done(term.dim(f"  frame {f}: exists, skipped"))
            return
        inf = res["info"]
        stats.append({"frame": f, "time": round(inf["time_total"], 3), "triangles": inf.get("triangles"),
                      "mean_spp": round(inf.get("mean_spp", spp), 1)})
        if not args.quiet:
            bar.done(f"  frame {f}: {inf['time_total']:.2f}s  ({_human(inf.get('triangles', 0))} tris, "
                     f"{inf.get('lights', 0)} lights)  {os.path.relpath(path)}")
        if args.show:
            term.show_image(res["display"], width=args.show_width)

    try:
        written = renderer.render_sequence(sc, frames, pattern, preview=args.preview, samples=args.samples,
                                           scale=args.scale, resume=args.resume, progress=prog,
                                           on_frame=on_frame)
    except KeyboardInterrupt:
        bar.done(term.yellow("interrupted - rerun with --resume to continue"))
        return 130
    total = time.time() - t_start
    if not args.quiet:
        term.log(term.green(f"rendered {len(written)} frame(s) in {term.fmt_time(total)}") +
                 (f"  ({total / max(1, len(stats)):.2f}s/frame)" if stats else ""))
    if args.log:
        renderer.write_metadata(args.log, sc, frames, stats)
    # video
    video = None
    if args.no_video:
        return 0
    if args.video:
        v = sc.settings.get("video") or {}
        v = {"path": v} if isinstance(v, str) else dict(v)
        v["path"] = os.path.abspath(args.video)
        if args.codec:
            v["codec"] = args.codec
        if args.audio:
            v["audio"] = os.path.abspath(args.audio)
        sc.settings["video"] = v
    elif args.codec and sc.settings.get("video"):
        v = sc.settings["video"]
        v = {"path": v} if isinstance(v, str) else dict(v)
        v["codec"] = args.codec
        sc.settings["video"] = v
    if sc.settings.get("video") and len(frames) > 1:
        if any(not float(f).is_integer() for f in frames):
            term.log(term.yellow("sub-frame steps: skipping video encode"))
            return 0
        steps = np.diff(frames)
        if len(steps) and not np.allclose(steps, steps[0]):
            term.log(term.yellow("non-contiguous frame list: skipping video encode"))
            return 0
        if len(steps) and steps[0] != 1:
            sc.settings["frame_step"] = steps[0]
            renum = _renumber_for_video(pattern, frames)
            pat, start = renum, 1
        else:
            pat, start = pattern, frames[0]
        try:
            video = output.encode_from_settings(sc, pat, [start], verbose=args.verbose)
        except output.EncodeError as e:
            term.log(term.red(str(e)))
            return 1
        if video and not args.quiet:
            info = output.probe(video) or {}
            term.log(term.green(f"video: {video}") + term.dim(
                f"  {info.get('codec_name', '')} {info.get('width', '')}x{info.get('height', '')} "
                f"{info.get('nb_read_frames', '')} frames"))
    return 0


def _renumber_for_video(pattern, frames):
    """Hard-link (or copy) a stepped sequence to a contiguous temporary sequence for ffmpeg."""
    import shutil
    import tempfile
    from .render.renderer import frame_path
    d = tempfile.mkdtemp(prefix="animora_seq_")
    ext = os.path.splitext(pattern)[1]
    for i, f in enumerate(frames):
        src = frame_path(pattern, f)
        dst = os.path.join(d, f"f_{i + 1:06d}{ext}")
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy(src, dst)
    return os.path.join(d, f"f_######{ext}")


def cmd_preview(args):
    from .render import renderer
    from .io.images import write_image
    _ensure_native()
    sc = load(args)
    frame = parse_frames(args.frame, sc)[0] if args.frame else sc.frame_start
    if sc.dynamics is not None:
        sc.dynamics.simulate(frame)
    bar = term.Progress(f"frame {frame}")
    scale = args.scale if args.scale is not None else (None if not args.full else 1.0)
    res = renderer.render_frame(sc, frame, preview=not args.full, samples=args.samples, scale=scale,
                                progress=lambda f: bar.update(f))
    bar.done()
    term.show_image(res["display"], width=args.width)
    inf = res["info"]
    term.log(term.dim(f"frame {frame}  {inf['width']}x{inf['height']}  {inf['spp']} spp  "
                      f"{_human(inf.get('triangles', 0))} tris  {inf['time_total']:.2f}s"))
    if args.save:
        renderer.save_frame(res, args.save, sc, frame=frame)
        term.log(f"saved {args.save}")
    return 0


def cmd_play(args):
    """Render low-res preview frames and play them as a flipbook in the terminal."""
    from .render import renderer
    _ensure_native()
    sc = load(args)
    frames = parse_frames(args.frames, sc)
    if sc.dynamics is not None:
        sc.dynamics.simulate(max(frames))
    imgs = []
    bar = term.Progress("rendering flipbook")
    for i, f in enumerate(frames):
        r = renderer.render_frame(sc, f, preview=True, samples=args.samples, scale=args.scale)
        imgs.append(r["display"])
        bar.update((i + 1) / len(frames))
    bar.done()
    term.play(imgs, fps=args.fps or sc.fps, width=args.width, loops=args.loops)
    return 0


def cmd_view(args):
    from .io.images import read_image
    from .render import tonemap
    for p in args.images:
        img = read_image(p, linear=False)
        ext = Path(p).suffix.lower()
        if ext in (".exr", ".hdr", ".pfm"):
            img = tonemap.process(img if img.shape[-1] == 4 else img[..., :3],
                                  {"tonemap": args.tonemap, "exposure": args.exposure})
        print(term.bold(p))
        term.show_image(img, width=args.width)
    return 0


def cmd_info(args):
    sc = load(args)
    s = sc.summary()
    if args.json:
        print(json.dumps({"summary": s, "settings": sc.settings}, indent=2, default=str))
        return 0
    print(term.bold(Path(args.scene).name))
    for k, v in s.items():
        print(f"  {k:<14} {v}")
    print(term.bold("\nsettings"))
    for k in ("resolution", "fps", "frame_range", "samples", "max_bounces", "integrator", "motion_blur",
              "denoise", "tonemap", "output", "video"):
        print(f"  {k:<14} {sc.settings.get(k)}")
    print(term.bold("\nhierarchy"))
    print("  " + sc.tree().replace("\n", "\n  "))
    if sc.materials:
        print(term.bold("\nmaterials"))
        print("  " + ", ".join(sc.materials))
    if args.stats:
        from .scene import build_packed
        f = sc.frame_start
        t = time.time()
        packed, info = build_packed(sc, f)
        print(term.bold("\nframe stats") + f" (frame {f}, built in {time.time() - t:.2f}s)")
        for k, v in info.items():
            print(f"  {k:<14} {v}")
    return 0


def cmd_validate(args):
    sc = load(args)
    frames = parse_frames(args.frames, sc) if args.frames else None
    probs = sc.validate(frames)
    if not probs:
        print(term.green("OK") + f"  {Path(args.scene).name}: {len(sc.nodes)} nodes, frames "
                                 f"{sc.frame_start:g}-{sc.frame_end:g}")
        return 0
    errors = [p for p in probs if not p.startswith("no camera")]
    for p in probs:
        print((term.yellow("warning: ") if p.startswith("no camera") else term.red("error: ")) + p)
    return 1 if errors else 0


def cmd_eval(args):
    """Print evaluated channel values over a frame range (``node.attr[.component]``)."""
    sc = load(args)
    frames = parse_frames(args.frames, sc)
    rows = []
    for f in frames:
        vals = []
        for path in args.channels:
            v = sc.get_value(path, f)
            v = np.asarray(v, float).round(args.precision).tolist() if not isinstance(v, (bool, str)) else v
            vals.append(v)
        rows.append((f, vals))
    if args.json:
        print(json.dumps([{"frame": f, **dict(zip(args.channels, v))} for f, v in rows]))
        return 0
    print("frame  " + "  ".join(args.channels))
    for f, vals in rows:
        print(f"{f:>5}  " + "  ".join(str(v) for v in vals))
    return 0


TEMPLATES = {}


def _template(kind):
    base = {
        "settings": {"resolution": [1280, 720], "fps": 24, "frame_range": [1, 48], "samples": 64,
                     "output": "render/frame_####.png", "video": "render/animation.mp4"},
        "environment": {"type": "sky", "sun_elevation": 35, "sun_azimuth": 140, "strength": 1.0},
        "materials": {
            "floor": {"preset": "studio_floor"},
            "hero": {"preset": "plastic", "base_color": "#e0452b"},
        },
        "nodes": [
            {"name": "camera", "type": "camera", "translate": [0, 1.6, 7], "look_at": [0, 0.8, 0],
             "focal_length": 40},
            {"name": "key", "type": "light", "light": "area", "translate": [-3, 4, 3], "look_at": [0, 0.5, 0],
             "size": 2.0, "intensity": 60, "temperature": 5200},
            {"name": "floor", "type": "mesh", "primitive": "plane", "size": 30, "material": "floor"},
        ],
    }
    if kind == "empty":
        base["nodes"] = base["nodes"][:1]
        base["materials"] = {}
    elif kind == "bounce":
        base["nodes"].append({
            "name": "ball", "type": "mesh", "primitive": "uv_sphere", "radius": 0.5, "material": "hero",
            "translate": [0, "=0.5 + bounce(t, 1.8, 1.0)", 0],
            "comment": "absolute-sine hop: 1.8 units high, one bounce per second"})
        base["nodes"][0].update(translate=[0, 1.8, 9], look_at=[0, 1.2, 0], focal_length=35)
    elif kind == "turntable":
        base["nodes"][0] = {"name": "camera", "type": "camera", "parent": "orbit", "translate": [0, 1.4, 6],
                            "look_at": [0, 0.8, 0], "focal_length": 50}
        base["nodes"].insert(0, {"name": "orbit", "type": "group", "rotate": [0, "=u*360", 0]})
        base["nodes"].append({"name": "model", "type": "mesh", "primitive": "torus", "major_radius": 0.8,
                              "minor_radius": 0.3, "translate": [0, 1.1, 0], "rotate": [70, 0, 0],
                              "material": "gold", "modifiers": [{"type": "subdivide", "levels": 1}]})
        base["materials"]["gold"] = {"preset": "gold"}
        base["settings"]["frame_range"] = [1, 96]
    else:  # basic
        base["nodes"].append({
            "name": "cube", "type": "mesh", "primitive": "cube", "size": 1.0, "material": "hero",
            "translate": {"keys": [[1, [-2, 0.5, 0]], [48, [2, 0.5, 0], "ease_in_out"]]},
            "rotate": {"keys": [[1, [0, 0, 0]], [48, [0, 0, -360], "ease_in_out"]]}})
    return base


def cmd_new(args):
    path = Path(args.path)
    if path.suffix == "":
        path = path / "scene.json" if not path.exists() or path.is_dir() else path.with_suffix(".json")
    if path.exists() and not args.force:
        raise SystemExit(f"{path} exists (use --force to overwrite)")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".py":
        path.write_text(_PY_TEMPLATE)
    else:
        path.write_text(json.dumps(_template(args.template), indent=2) + "\n")
    print(f"created {path}\n  render it:  animora render {path}\n  preview:    animora preview {path}")
    return 0


_PY_TEMPLATE = '''"""Animora Python scene. Render with: animora render scene.py"""
import animora as an

scene = an.Scene({"settings": {"resolution": [1280, 720], "frame_range": [1, 48], "samples": 64,
                               "video": "render/animation.mp4"},
                  "environment": {"type": "sky", "sun_elevation": 35, "sun_azimuth": 140}})
scene.add_material("floor", "studio_floor")
scene.add_material("hero", preset="glossy_plastic", base_color="#2b7de0")
scene.add_camera("camera", translate=[0, 1.6, 7], look_at=[0, 0.8, 0], focal_length=40)
scene.add_light("key", light="area", translate=[-3, 4, 3], look_at=[0, 0.5, 0], size=2, intensity=60)
scene.add_mesh("floor", primitive="plane", size=30, material="floor")
ball = scene.add_mesh("ball", primitive="ico_sphere", radius=0.5, material="hero")
for frame, y in [(1, 0.5), (12, 2.5), (24, 0.5), (36, 2.5), (48, 0.5)]:
    ball.key("translate", frame, [0, y, 0], "ease_out" if y > 1 else "ease_in")
'''


def cmd_encode(args):
    from . import output
    from .render.renderer import frame_path
    pattern = args.pattern
    start = args.start
    if start is None:
        # find the first existing frame
        start = next((f for f in range(0, 100000) if os.path.exists(frame_path(pattern, f))), None)
        if start is None:
            raise SystemExit(f"no frames match {pattern}")
    kw = {"codec": args.codec, "quality": args.quality, "audio": args.audio, "alpha": args.alpha,
          "scale": args.scale, "crf": args.crf, "loop": 0}
    try:
        out = output.encode(pattern, args.out, fps=args.fps, start_number=start, verbose=args.verbose,
                            **{k: v for k, v in kw.items() if v is not None})
    except output.EncodeError as e:
        raise SystemExit(term.red(str(e)))
    info = output.probe(out) or {}
    print(term.green(f"wrote {out}") + term.dim(f"  {info.get('codec_name', '')} {info.get('width', '')}x"
                                                f"{info.get('height', '')} {info.get('nb_read_frames', '')} frames"))
    return 0


def cmd_build(args):
    from .render import native as nat
    t = time.time()
    path = nat.build(force=args.force, verbose=args.verbose)
    print(term.green(f"native renderer ready ({time.time() - t:.1f}s): {path}"))
    print(f"  {nat.version()}")
    if nat.gpu_supported():
        t = time.time()
        try:
            gp = nat.build_gpu(force=args.force, verbose=args.verbose)
            info = nat.gpu_info()
            if info["available"]:
                print(term.green(f"GPU renderer ready ({time.time() - t:.1f}s): {gp}"))
                print(f"  Metal device: {info['device']}")
            else:
                print(term.yellow(f"GPU renderer built but unusable: {info['error']}"))
        except Exception as e:  # noqa: BLE001
            print(term.yellow(f"GPU renderer not built: {e}"))
    return 0


def cmd_export(args):
    """Export geometry: .obj/.ply (one frame, or a numbered sequence with --frames), or a full
    animated scene as .glb/.gltf (glTF 2.0) or .usda (USD)."""
    from .geometry import merge
    from .io.obj import write_mesh
    sc = load(args)
    ext = Path(args.out).suffix.lower()
    if ext in (".glb", ".gltf", ".usda", ".usd"):
        frames = parse_frames(args.frames, sc) if args.frames else (
            parse_frames(args.frame, sc) if args.frame else None)
        if frames is None:
            frames = list(np.arange(sc.frame_start, sc.frame_end + 1e-9, args.step))
        bar = term.Progress("sampling scene", enabled=not args.quiet)
        prog = lambda i, n: bar.update(i / max(n, 1))
        if ext in (".glb", ".gltf"):
            from .io.gltf import export_gltf
            info = export_gltf(sc, args.out, frames, args.nodes, args.hidden, progress=prog)
        else:
            from .io.usd import export_usd
            out = args.out if ext == ".usda" else str(Path(args.out).with_suffix(".usda"))
            info = export_usd(sc, out, frames, args.nodes, args.hidden, progress=prog)
            args.out = out
        if bar:
            bar.done()
        print(f"wrote {args.out}: " + ", ".join(f"{k} {v}" for k, v in info.items()))
        return 0
    if ext not in (".obj", ".ply", "") and not args.separate:
        raise SystemExit(f"unsupported export format {ext!r} (.obj .ply .glb .gltf .usda)")
    names = args.nodes or [n for n in sc.order if sc.nodes[n].type in ("mesh", "instance", "cloth", "curve",
                                                                           "liquid", "hair")]
    frames = parse_frames(args.frames, sc) if args.frames else [
        parse_frames(args.frame, sc)[0] if args.frame else sc.frame_start]
    seq = bool(args.frames)
    if sc.dynamics is not None:
        sc.dynamics.simulate(int(np.ceil(max(frames))))
    from .scene import _node_visible
    total = 0
    for frame in frames:
        meshes = []
        for n in names:
            if n not in sc.nodes:
                raise SystemExit(f"no node {n!r}")
            if not args.hidden and not _node_visible(sc, n, frame):
                continue
            m = sc.world_mesh(n, frame)
            if m is not None and m.num_verts:
                m.name = n
                meshes.append(m)
        if not meshes:
            if seq:
                continue
            raise SystemExit("nothing to export")
        fnum = int(round(frame))
        if args.separate:
            out = Path(args.out)
            out.mkdir(parents=True, exist_ok=True)
            for m in meshes:
                write_mesh(out / (f"{m.name}_{fnum:04d}{args.format}" if seq else f"{m.name}{args.format}"), m)
            total += len(meshes)
        else:
            target = args.out
            if seq:
                target = _frame_path(args.out, fnum)
                Path(target).parent.mkdir(parents=True, exist_ok=True)
            merged = merge(meshes, name=Path(args.out).stem)
            write_mesh(target, merged)
            total += 1
            if not seq:
                print(f"wrote {target}: {merged.num_verts} verts, {merged.num_faces} faces (frame {frame:g})")
    if args.separate or seq:
        print(f"wrote {total} files to {Path(args.out).parent if not args.separate else args.out}"
              f" ({len(frames)} frame(s))")
    return 0


def _frame_path(pattern, frame):
    """``out/sim_####.obj`` -> ``out/sim_0012.obj``; without ``#`` the frame is appended."""
    import re
    p = Path(pattern)
    m = re.search(r"#+", p.name)
    if m:
        return str(p.with_name(p.name[:m.start()] + str(frame).zfill(m.end() - m.start()) + p.name[m.end():]))
    return str(p.with_name(f"{p.stem}_{frame:04d}{p.suffix}"))


def cmd_import(args):
    """Convert a glTF/GLB (meshes, hierarchy, cameras, lights, materials, animation) or BVH
    (motion capture skeleton) into an Animora scene file."""
    out = Path(args.output or Path(args.file).with_suffix(".json"))
    ext = Path(args.file).suffix.lower()
    if ext in (".glb", ".gltf"):
        from .io.gltf import gltf_to_scene
        data = gltf_to_scene(args.file, fps=args.fps, scene_path=out)
    elif ext == ".bvh":
        from .io.bvh import bvh_to_scene
        data = bvh_to_scene(args.file, fps=args.fps, scale=args.scale, joint_radius=args.joint_radius,
                            scene_path=out)
    elif ext in (".obj", ".ply"):
        rel = os.path.relpath(Path(args.file).resolve(), out.resolve().parent)
        data = {"settings": {"fps": args.fps or 24, "frame_range": [1, 1]}, "environment": {"type": "sky"},
                "nodes": [{"name": Path(args.file).stem, "type": "mesh", "file": rel}]}
    else:
        raise SystemExit(f"unsupported import format {ext!r} (.glb .gltf .bvh .obj .ply)")
    nodes = data["nodes"]
    if args.camera and not any(n.get("type") == "camera" for n in nodes):
        from .scene import Scene
        try:
            tmp = Scene(json.loads(json.dumps(data)), base_dir=str(out.resolve().parent))
            lo, hi = _scene_bounds(tmp)
        except Exception:
            lo, hi = np.array([-1.0, 0, -1]), np.array([1.0, 2, 1])
        c = (lo + hi) / 2
        r = float(np.linalg.norm(hi - lo)) / 2 + 1e-3
        # fit the bounding sphere inside the vertical field of view of a 35 mm lens (24 mm gate)
        fov_y = 2 * math.atan(12.0 / 35.0)
        dist = r / math.sin(fov_y / 2) * 1.08
        d = np.array([1.2, 0.6, 2.2]) / np.linalg.norm([1.2, 0.6, 2.2])
        nodes.insert(0, {"name": "camera", "type": "camera", "focal_length": 35,
                         "translate": [round(float(x), 4) for x in c + d * dist],
                         "look_at": [round(float(x), 4) for x in c]})
        if not any(n.get("type") == "light" for n in nodes):
            nodes.insert(1, {"name": "key", "type": "light", "light": "sun", "rotate": [-45, 30, 0], "intensity": 3})
            nodes.append({"name": "ground", "type": "mesh", "primitive": "plane", "size": round(r * 20, 3),
                          "translate": [0, round(float(lo[1]), 4), 0]})
    out.write_text(json.dumps(data, indent=1))
    kinds = {}
    for n in nodes:
        kinds[n.get("type", "?")] = kinds.get(n.get("type", "?"), 0) + 1
    print(f"wrote {out}: " + ", ".join(f"{v} {k}" for k, v in sorted(kinds.items()))
          + f", frames {data['settings']['frame_range'][0]}-{data['settings']['frame_range'][1]}")
    return 0


def _scene_bounds(sc, samples=5):
    """World bounds of the scene's geometry and joints, unioned over a few frames of its range
    (so an imported walk cycle or a falling object stays in frame)."""
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    fs, fe = sc.frame_start, sc.frame_end
    frames = sorted({round(fs + (fe - fs) * k / max(samples - 1, 1)) for k in range(samples)}) if fe > fs else [fs]
    for n in sc.order:
        node = sc.nodes[n]
        if node.type in ("mesh", "instance", "curve"):
            for f in frames:
                m = sc.world_mesh(n, f, True)
                if m is not None and m.num_verts:
                    lo, hi = np.minimum(lo, m.verts.min(0)), np.maximum(hi, m.verts.max(0))
        elif node.type == "joint":
            for f in frames:
                p = sc.state(f).world(n)[:3, 3]
                lo, hi = np.minimum(lo, p), np.maximum(hi, p)
    if not np.isfinite(lo).all():
        return np.array([-1.0, 0, -1]), np.array([1.0, 2, 1])
    return lo, hi


def cmd_bake(args):
    sc = load(args)
    from .cli_extra import bake_grid_sims
    n_grid = bake_grid_sims(sc, args)
    if sc.dynamics is None:
        if n_grid:
            return 0
        raise SystemExit("scene has nothing to simulate (rigid bodies, jiggle, cloth, particles, smoke, liquid)")
    out = args.output or sc.dynamics.cfg.get("cache") or str(Path(args.scene).with_suffix(".sim.npz"))
    out = sc.resolve_path(out) if not os.path.isabs(out) else out
    end = args.end or sc.frame_end
    bar = term.Progress("simulating")
    t = time.time()
    sc.dynamics.bake(out, end=end, progress=lambda f, e: bar.update(f / max(1, e), f"simulating frame {f}"))
    bar.done(term.green(f"baked frames {sc.frame_start}-{end} to {out} in {time.time() - t:.1f}s"))
    return 0


def cmd_turntable(args):
    """Build (and optionally render) a turntable scene for a mesh file."""
    from .io.obj import read_mesh
    m, _mats = read_mesh(args.model)
    lo, hi = m.bounds()
    size = float(np.max(hi - lo))
    s = 1.6 / max(size, 1e-9)
    center = (lo + hi) / 2
    mid_y = float((hi[1] - lo[1]) * s / 2)
    model = os.path.abspath(args.model)
    frames = args.frames
    data = {
        "settings": {"resolution": parse_res(args.res) if args.res else [1280, 720], "fps": args.fps,
                     "frame_range": [1, frames], "samples": args.samples,
                     "output": os.path.abspath(os.path.join(args.out_dir, "frame_####.png")),
                     "video": os.path.abspath(os.path.join(args.out_dir, Path(args.model).stem + "_turntable.mp4"))},
        "environment": {"type": "studio", "strength": 1.0},
        "materials": {"floor": {"preset": "shadow_catcher" if args.transparent else "studio_floor"},
                      "model": {"preset": args.material}},
        "nodes": [
            {"name": "turn", "type": "group", "rotate": [0, f"=(frame-1)/{frames}*360", 0], "children": [
                {"name": "model", "type": "mesh", "file": model, "scale": s,
                 "translate": [float(-center[0] * s), float(-lo[1] * s), float(-center[2] * s)],
                 "material": None if args.keep_materials else "model"}]},
            {"name": "floor", "type": "mesh", "primitive": "disc", "radius": 20, "material": "floor"},
            {"name": "camera", "type": "camera", "translate": [0, mid_y + 0.45, 5.6], "look_at": [0, mid_y, 0],
             "focal_length": 50},
            {"name": "key", "type": "light", "light": "area", "translate": [-3, 4, 3], "look_at": [0, 0.8, 0],
             "size": 2.5, "intensity": 50},
            {"name": "rim", "type": "light", "light": "area", "translate": [3, 3, -3], "look_at": [0, 1, 0],
             "size": 2.0, "intensity": 40, "temperature": 7500},
        ],
    }
    if args.keep_materials:
        data["nodes"][0]["children"][0].pop("material")
    if args.transparent:
        data["settings"]["film_transparent"] = True
    scene_path = os.path.join(args.out_dir, "turntable.json")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(scene_path).write_text(json.dumps(data, indent=2))
    print(f"wrote {scene_path}")
    if args.no_render:
        return 0
    ns = argparse.Namespace(scene=scene_path, frames=None, output=None, preview=False, samples=None, scale=None,
                            resume=False, quiet=False, show=False, show_width=None, video=None, no_video=False,
                            codec=None, audio=None, log=None, verbose=False, set=[], res=None)
    return cmd_render(ns)


def cmd_materials(args):
    from .materials import PRESETS, Material
    if args.name:
        if args.name not in PRESETS:
            raise SystemExit(f"no preset {args.name!r}")
        from .materials import _DEFAULTS
        merged = dict(_DEFAULTS)
        merged.update(PRESETS[args.name])
        print(json.dumps({"preset": args.name, "resolved": merged}, indent=2, default=str))
        return 0
    for k, v in PRESETS.items():
        desc = ", ".join(f"{a}={b if not isinstance(b, dict) else b.get('type', 'tex')}" for a, b in v.items())
        print(f"  {term.bold(k):<24} {term.dim(desc[:100])}")
    return 0


def cmd_primitives(args):
    import inspect
    from .geometry import PRIMITIVES
    for k, fn in sorted(PRIMITIVES.items()):
        sig = ", ".join(f"{p.name}={p.default!r}" for p in inspect.signature(fn).parameters.values()
                        if p.default is not inspect.Parameter.empty and p.name != "name")
        print(f"  {term.bold(k):<20} {term.dim(sig)}")
    return 0


def cmd_curves(args):
    from .curves import EASINGS, get_easing
    names = args.names or sorted(EASINGS)
    if not args.plot:
        print("  " + "\n  ".join(names))
        print(term.dim("\nalso: linear, constant/step, bezier, auto, cubic-bezier(x1,y1,x2,y2)"))
        return 0
    W, H = args.width, args.height
    for n in names:
        f = get_easing(n)
        if f is None:
            print(term.red(f"unknown easing {n!r}"))
            continue
        xs = np.linspace(0, 1, W)
        ys = np.array([float(f(x)) for x in xs])
        lo, hi = min(0.0, ys.min()), max(1.0, ys.max())
        grid = [[" "] * W for _ in range(H)]
        for x, y in enumerate(ys):
            r = int(round((1 - (y - lo) / (hi - lo)) * (H - 1)))
            grid[r][x] = "●"
        print(term.bold(n))
        for row in grid:
            print("  │" + "".join(row))
        print("  └" + "─" * W)
    return 0


def cmd_doctor(args):
    import platform
    import shutil
    from .render import native as nat
    from . import output
    ok = True
    print(term.bold(f"Animora {__version__}"))
    print(f"  python      {sys.version.split()[0]} ({platform.machine()}, {platform.system()} {platform.release()})")
    print(f"  numpy       {np.__version__}")
    try:
        import PIL
        print(f"  pillow      {PIL.__version__}")
    except ImportError:
        print(term.red("  pillow      missing (pip install pillow) - needed for PNG/JPEG I/O"))
        ok = False
    cxx = os.environ.get("CXX") or shutil.which("clang++") or shutil.which("g++")
    print(f"  c++         {cxx or term.red('not found - needed once to build the path tracer')}")
    try:
        if not nat.available():
            nat.build()
        print(f"  renderer    {term.green('ok')}  {nat.version()}")
    except Exception as e:  # noqa: BLE001
        print(term.red(f"  renderer    FAILED: {e}"))
        ok = False
    if nat.gpu_supported():
        info = nat.gpu_info()
        if info["available"]:
            print(f"  gpu         {term.green('ok')}  Metal: {info['device']}  (--device gpu)")
        else:
            print(term.yellow(f"  gpu         unavailable ({info['error']}) - CPU rendering only"))
    else:
        print("  gpu         no GPU backend on this platform (Metal/macOS only) - CPU rendering")
    ff = output.find_ffmpeg()
    if ff:
        enc = output.ffmpeg_encoders()
        have = [c for c in ("libx264", "libx265", "prores_ks", "libvpx-vp9", "libsvtav1", "libaom-av1") if c in enc]
        print(f"  ffmpeg      {ff}  ({', '.join(have)})")
    else:
        print(term.yellow("  ffmpeg      not found - image sequences work, video encoding needs ffmpeg"))
    print(f"  cpu threads {os.cpu_count()}")
    if args.bench:
        from .scene import Scene
        from .render import renderer
        sc = Scene(_template("basic"))
        sc.settings.update(resolution=[480, 270], samples=32)
        t = time.time()
        r = renderer.render_frame(sc, 24)
        dt = time.time() - t
        mrays = 480 * 270 * 32 / dt / 1e6
        print(f"  benchmark   480x270 @32spp in {dt:.2f}s  (~{mrays:.1f} M camera samples/s)  [cpu]")
        if nat.gpu_available():
            sc.settings["device"] = "gpu"
            renderer.render_frame(sc, 24, samples=1)  # warm up (driver shader cache)
            t = time.time()
            renderer.render_frame(sc, 24)
            dt = time.time() - t
            print(f"  benchmark   480x270 @32spp in {dt:.2f}s  (~{480 * 270 * 32 / dt / 1e6:.1f} M camera "
                  "samples/s)  [gpu]")
    print(term.green("all good") if ok else term.red("problems found"))
    return 0 if ok else 1


# ======================================================================================
# parser
# ======================================================================================
def _add_scene_overrides(p):
    p.add_argument("--set", action="append", metavar="KEY=VALUE",
                   help="override a settings key (JSON value), repeatable: --set samples=256 --set caustics=true")
    p.add_argument("--res", help="resolution WxH or preset (720p, 1080p, 4k, square, vertical, ...)")
    p.add_argument("--samples", "-s", type=int, help="samples per pixel")
    p.add_argument("--threads", "-t", type=int, help="render threads (0 = all cores)")
    p.add_argument("--device", "-d", choices=["cpu", "gpu", "auto"],
                   help="render device: cpu, gpu (Apple Metal) or auto (GPU when available)")
    p.add_argument("--integrator", help="path, direct, ao, clay, normals, albedo, depth, uv, wireframe")
    p.add_argument("--max-bounces", type=int, dest="max_bounces")
    p.add_argument("--seed", type=int)
    p.add_argument("--tonemap", help="agx, agx_punchy, aces, filmic, reinhard, neutral, standard")
    p.add_argument("--exposure", type=float, help="exposure in stops")
    p.add_argument("--motion-blur", action="store_true", dest="motion_blur")
    p.add_argument("--no-denoise", action="store_true", dest="no_denoise")
    p.add_argument("--transparent", action="store_true", help="transparent background (alpha)")


def build_parser():
    ap = argparse.ArgumentParser(prog="animora", description="Animora - command-line 3D animation and "
                                 "path-traced rendering. No GUI: scenes are text files, output is images/video.")
    ap.add_argument("--version", action="version", version=f"animora {__version__}")
    sub = ap.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("render", help="render a frame range to images (+ video)")
    p.add_argument("scene")
    p.add_argument("-f", "--frames", help="frames: 12 | 1-48 | 1-48x2 | 1,5,9 | first,last (default: scene range)")
    p.add_argument("-o", "--output", help="image pattern, e.g. out/shot_####.exr (default: settings.output)")
    p.add_argument("--video", help="encode a video to this path (overrides settings.video)")
    p.add_argument("--codec", help="h264, h265, prores, prores4444, webm, av1, gif, png")
    p.add_argument("--audio", help="audio file muxed into the video")
    p.add_argument("--no-video", action="store_true", dest="no_video", help="skip video encoding")
    p.add_argument("--preview", action="store_true", help="fast draft: preview_scale & preview_samples")
    p.add_argument("--scale", type=float, help="resolution multiplier, e.g. 0.5")
    p.add_argument("--resume", action="store_true", help="skip frames that already exist")
    p.add_argument("--show", action="store_true", help="draw each finished frame in the terminal")
    p.add_argument("--show-width", type=int, dest="show_width")
    p.add_argument("--log", help="write a JSON render log")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("preview", help="render one frame and draw it in the terminal")
    p.add_argument("scene")
    p.add_argument("-f", "--frame", help="frame (default: first)")
    p.add_argument("--full", action="store_true", help="full resolution / full samples")
    p.add_argument("--scale", type=float)
    p.add_argument("-w", "--width", type=int, help="terminal columns to use")
    p.add_argument("--save", help="also save the image to this path")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_preview)

    p = sub.add_parser("play", help="render a low-res flipbook and play it in the terminal")
    p.add_argument("scene")
    p.add_argument("-f", "--frames")
    p.add_argument("--fps", type=float)
    p.add_argument("--scale", type=float, default=0.25)
    p.add_argument("-w", "--width", type=int)
    p.add_argument("--loops", type=int, default=2)
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_play)

    p = sub.add_parser("view", help="draw image file(s) in the terminal (PNG, JPEG, EXR, HDR, ...)")
    p.add_argument("images", nargs="+")
    p.add_argument("-w", "--width", type=int)
    p.add_argument("--tonemap", default="agx")
    p.add_argument("--exposure", type=float, default=0.0)
    p.set_defaults(func=cmd_view)

    p = sub.add_parser("info", help="print scene summary, settings, hierarchy")
    p.add_argument("scene")
    p.add_argument("--json", action="store_true")
    p.add_argument("--stats", action="store_true", help="also build frame 1 and print triangle/light counts")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("validate", help="check a scene for errors (evaluates a few frames)")
    p.add_argument("scene")
    p.add_argument("-f", "--frames")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("eval", help="print animated channel values: animora eval s.json ball.translate.y")
    p.add_argument("scene")
    p.add_argument("channels", nargs="+", help="node.attr[.component], e.g. ball.world_position, cam.focal_length")
    p.add_argument("-f", "--frames")
    p.add_argument("--precision", type=int, default=4)
    p.add_argument("--json", action="store_true")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("new", help="create a starter scene file")
    p.add_argument("path", help="file (.json or .py) or directory")
    p.add_argument("--template", default="basic", choices=["basic", "bounce", "turntable", "empty"])
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_new)

    p = sub.add_parser("encode", help="encode an image sequence into a video with ffmpeg")
    p.add_argument("pattern", help="e.g. render/frame_####.png")
    p.add_argument("out", help="e.g. render/anim.mp4")
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--start", type=int, help="first frame number (default: auto-detect)")
    p.add_argument("--codec")
    p.add_argument("--quality", default="high", choices=["draft", "low", "medium", "high", "best", "lossless"])
    p.add_argument("--crf", type=int)
    p.add_argument("--audio")
    p.add_argument("--alpha", action="store_true")
    p.add_argument("--scale", help="output width or ffmpeg scale expr, e.g. 1280 or 1280:-2")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("build", help="compile the native path tracer")
    p.add_argument("--force", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("export", aliases=["export-obj"],
                       help="export geometry (.obj/.ply, frame sequences) or the animated scene (.glb/.gltf/.usda)")
    p.add_argument("scene")
    p.add_argument("out", help="output file; .glb/.gltf/.usda export the animated scene; use #### in .obj/.ply "
                               "names with --frames for a numbered sequence")
    p.add_argument("-f", "--frame", help="single frame (.obj/.ply)")
    p.add_argument("--frames", help="frame list/range, e.g. 1-96 or 1-96x2: sequence export (.obj/.ply) or "
                                    "the sampled range (.glb/.usda)")
    p.add_argument("--step", type=float, default=1.0, help="sampling step for .glb/.usda when --frames is not given")
    p.add_argument("-q", "--quiet", action="store_true")
    p.add_argument("--nodes", nargs="*")
    p.add_argument("--separate", action="store_true", help="one file per node; OUT is a directory")
    p.add_argument("--format", default=".obj", choices=[".obj", ".ply"])
    p.add_argument("--hidden", action="store_true", help="include invisible nodes")
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("import", help="convert a .glb/.gltf/.bvh/.obj/.ply file into an Animora scene")
    p.add_argument("file")
    p.add_argument("-o", "--output", help="scene .json to write (default: next to the input)")
    p.add_argument("--fps", type=float, default=None, help="frames per second for sampled animation (default 24, "
                                                           "BVH: the file's frame time)")
    p.add_argument("--scale", type=float, default=0.01, help="BVH unit scale (default 0.01: cm -> m)")
    p.add_argument("--joint-radius", type=float, default=0.03, help="BVH: joint display radius")
    p.add_argument("--camera", action="store_true", help="add a framing camera, a sun and a ground plane")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("bake", help="simulate dynamics and write a cache file")
    p.add_argument("scene")
    p.add_argument("-o", "--output")
    p.add_argument("--end", type=int)
    _add_scene_overrides(p)
    p.set_defaults(func=cmd_bake)

    p = sub.add_parser("turntable", help="make and render a 360° turntable of a .obj/.ply model")
    p.add_argument("model")
    p.add_argument("--out-dir", default="turntable", dest="out_dir")
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--fps", type=float, default=24)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--res")
    p.add_argument("--material", default="clay")
    p.add_argument("--keep-materials", action="store_true", dest="keep_materials", help="use the file's .mtl")
    p.add_argument("--transparent", action="store_true")
    p.add_argument("--no-render", action="store_true", dest="no_render")
    p.set_defaults(func=cmd_turntable)

    p = sub.add_parser("materials", help="list material presets (or print one)")
    p.add_argument("name", nargs="?")
    p.set_defaults(func=cmd_materials)

    p = sub.add_parser("primitives", help="list mesh primitives and their parameters")
    p.set_defaults(func=cmd_primitives)

    p = sub.add_parser("curves", help="list easing curves (--plot draws them)")
    p.add_argument("names", nargs="*")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--width", type=int, default=48)
    p.add_argument("--height", type=int, default=12)
    p.set_defaults(func=cmd_curves)

    p = sub.add_parser("doctor", help="check the installation (compiler, renderer, ffmpeg)")
    p.add_argument("--bench", action="store_true", help="run a quick render benchmark")
    p.set_defaults(func=cmd_doctor)

    from . import cli_extra
    cli_extra.register(sub, _add_scene_overrides)
    return ap


def main(argv=None):
    ap = build_parser()
    args = ap.parse_args(argv)
    if not getattr(args, "func", None):
        ap.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        term.log(term.yellow("\ninterrupted"))
        return 130
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
