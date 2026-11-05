"""Additional CLI commands (graph editor, key editing, ...). Registered from :mod:`animora.cli`."""
from __future__ import annotations

import json
import sys

import numpy as np

from . import terminal as term


def _cli():
    from . import cli
    return cli


# ----------------------------------------------------------------------------- graph
def cmd_graph(args):
    from . import graph
    cli = _cli()
    sc = cli.load(args)
    frames = cli.parse_frames(args.frames, sc) if args.frames else None
    if frames is None:
        f0, f1 = sc.frame_start, sc.frame_end
        frames = list(np.linspace(f0, f1, int(max(2, min(600, (f1 - f0) * 4 + 1)))))
    elif len(frames) < 2:
        frames = [frames[0], frames[0] + 1]
    else:
        f0, f1 = float(frames[0]), float(frames[-1])
        frames = list(np.linspace(f0, f1, int(max(2, min(600, (f1 - f0) * 4 + 1)))))
    channels = args.channels or _default_channels(sc)
    if not channels:
        print(term.yellow("no animated channels found"))
        return 1
    series = graph.sample(sc, channels, frames)
    if not args.channels or args.hide_flat:  # drop components that never change
        moving = [s for s in series if np.nanmax(s[1]) - np.nanmin(s[1]) > 1e-9]
        series = moving or series
    if args.normalize:
        out = []
        for name, v, k in series:
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            s = (hi - lo) or 1.0
            out.append((name, (v - lo) / s, [(f, (val - lo) / s, i) for f, val, i in k]))
        series = out
    if args.output:
        p = graph.plot_png(series, frames, args.output, size=cli.parse_res(args.size) if args.size else (1400, 700),
                           vmin=args.min, vmax=args.max, current=args.current,
                           title=args.title or ", ".join(channels), dark=not args.light)
        print(f"wrote {p}")
        if args.show:
            from .io.images import read_image
            term.show_image(read_image(p)[..., :3], width=args.width)
    if not args.output or args.show_terminal:
        print(graph.plot_terminal(series, frames, args.width, args.height, color=term.supports_color(sys.stdout),
                                  vmin=args.min, vmax=args.max, current=args.current))
    if args.keys:
        for name, _, ks in series:
            if ks:
                print(term.bold(name))
                for f, v, i in ks:
                    print(f"  {f:>8g}  {v:>12.5g}  {i}")
    return 0


def _default_channels(sc):
    out = []
    for n in sc.order:
        node = sc.nodes[n]
        for a in ("translate", "rotate", "scale"):
            if getattr(node, a).is_animated():
                out.append(f"{n}.{a}")
        for k, p in list(node.params.items()) + list(node.attributes.items()):
            if hasattr(p, "is_animated") and p.is_animated() and p.kind in ("curve", "parts", "driven"):
                try:
                    v = np.asarray(sc.get_value(f"{n}.{k}", sc.frame_start), float)
                except Exception:  # noqa: BLE001
                    continue
                if v.size <= 4:
                    out.append(f"{n}.{k}")
    return out[:8]


# ----------------------------------------------------------------------------- key
def cmd_key(args):
    from . import graph
    defaults = {"scale": 1.0, "visible": 1.0}
    val = None
    if args.value is not None:
        val = json.loads(args.value) if args.value.strip().startswith("[") else float(args.value)
    op = args.op
    frame = None if args.frame is None else float(args.frame)
    if op in ("set", "delete", "move", "tangent") and frame is None:
        print(term.red(f"key {op} needs a frame (-t/--frame)"))
        return 2
    try:
        spec = graph.edit_keys(args.scene, args.channel, op, frame=frame, value=val, interp=args.interp,
                               new_frame=args.to, in_slope=args.in_slope, out_slope=args.out_slope,
                               pre=args.pre, post=args.post, defaults=defaults)
    except (ValueError, KeyError) as e:
        print(term.red(f"key: {e}"))
        return 2
    print(f"{args.channel} = {json.dumps(spec)}")
    return 0


def cmd_keys_list(args):
    cli = _cli()
    from . import graph
    sc = cli.load(args)
    chans = args.channels or _default_channels(sc)
    for ch in chans:
        prm, comp = graph.find_param(sc, ch)
        if prm is None:
            print(term.yellow(f"{ch}: not found"))
            continue
        print(term.bold(ch) + term.dim(f"  ({prm.kind})"))
        if prm.kind == "parts" and comp is None:
            for c, p in enumerate(prm.parts):
                for f, v, i in graph.param_keys(p):
                    print(f"  .{graph.COMP_NAMES[c] if c < 4 else c}  {f:>8g}  {v!s:>14}  {i}")
        else:
            for f, v, i in graph.param_keys(prm, comp):
                vv = np.round(v, 5).tolist() if isinstance(v, np.ndarray) else round(v, 5)
                print(f"  {f:>8g}  {vv!s:>14}  {i}")
        if prm.kind == "driven":
            print(term.dim(f"  driven by {prm.driver}"))
    return 0


# ----------------------------------------------------------------------------- colour
def cmd_colorspaces(args):
    from . import color as cm
    if args.convert:
        vals = [float(v) for v in args.convert]
        out = cm.convert(np.asarray(vals[:3]), args.src, args.dst)
        print(" ".join(f"{v:.6f}" for v in np.asarray(out).ravel()))
        return 0
    print(term.bold(f"{'name':<15}{'primaries':<11}{'white':<7}transfer"))
    for n, p, w, t in cm.describe():
        print(f"{n:<15}{p:<11}{w:<7}{t}")
    al = {}
    for a, n in cm.ALIASES.items():
        al.setdefault(n, []).append(a)
    print(term.dim("\naliases: " + "; ".join(f"{n}: {', '.join(v)}" for n, v in sorted(al.items()))))
    if args.matrix:
        m = cm.matrix(args.src, args.dst)
        print(term.bold(f"\n{args.src} -> {args.dst}"))
        for r in m:
            print("  " + "  ".join(f"{v:+.6f}" for v in r))
    return 0


# ----------------------------------------------------------------------------- retarget
def cmd_retarget(args):
    """Retarget a BVH (or a scene skeleton) onto the joints of a target scene and write the
    result with the motion baked into the target joints."""
    import os
    from pathlib import Path
    from .retarget import retarget, canonical_slots, _joint_order
    from .scene import load_scene, Scene
    tpath = Path(args.target)
    data = json.loads(tpath.read_text())
    data.pop("retarget", None)
    base = str(tpath.resolve().parent)
    tsc = Scene(data, base_dir=base)
    mapping = None
    if args.map:
        mapping = json.loads(Path(args.map).read_text()) if os.path.exists(args.map) else json.loads(args.map)
    src = args.source
    if src.lower().endswith(".json"):
        source = load_scene(src)
    else:
        source = src
    if args.list_map:
        from .io.bvh import read_bvh
        if isinstance(source, str):
            sj = [j["name"] for j in read_bvh(source)[0]]
        else:
            sj = _joint_order(source, args.source_root)
        from .retarget import auto_map
        tj = _joint_order(tsc, args.target_root)
        m = auto_map(sj, tj, overrides=mapping)
        ts = {v: k for k, v in canonical_slots(tj).items()}
        width = max(len(t) for t in tj)
        for t in tj:
            print(f"  {t:<{width}}  <- {m.get(t, term.dim('(unmapped)'))}  {term.dim(ts.get(t, ''))}")
        print(term.dim(f"{sum(t in m for t in tj)}/{len(tj)} target joints mapped"))
        return 0
    keys, info = retarget(source, tsc, target_root=args.target_root, source_root=args.source_root,
                          mapping=mapping, bvh_scale=args.scale,
                          start_frame=args.start, source_yaw=("auto" if args.yaw is None else args.yaw),
                          match_rest=not args.no_match_rest, in_place=args.in_place,
                          foot_lock=args.foot_lock, root_motion=not args.no_root_motion,
                          height_scale=("auto" if args.height_scale is None else args.height_scale))
    from .retarget import _walk_specs
    nodes = data.get("nodes", [])
    if isinstance(nodes, dict):
        data["nodes"] = nodes = [dict(v, name=k) for k, v in nodes.items()]
    for spec in _walk_specs(nodes):
        if spec.get("name") in keys:
            spec.update(keys[spec["name"]])
    data.setdefault("settings", {})["frame_range"] = [int(info["frames"][0]), int(np.ceil(info["frames"][1]))]
    out = Path(args.output or tpath.with_name(tpath.stem + "_retargeted.json"))
    out.write_text(json.dumps(data, indent=1))
    print(term.green(f"wrote {out}") + f": {len(keys)} joints keyed, frames {info['frames'][0]:g}-"
          f"{info['frames'][1]:g}, {len(info['map'])} mapped, height scale {info['height_scale']:.3f}, "
          f"yaw {info['source_yaw']:.1f}°" + (f", foot locks {info['foot_locks']}" if info["foot_locks"] else ""))
    if info["unmapped_target"]:
        print(term.dim("unmapped target joints (kept at rest): " + ", ".join(info["unmapped_target"])))
    return 0


# ----------------------------------------------------------------------------- weights
def cmd_weights(args):
    """Inspect, preview (heat map), export and import skin weights of a skinned node."""
    from .rig import skin_data, _skin_spec
    from . import weightpaint as wp
    cli = _cli()
    sc = cli.load(args)
    if args.node is None or args.node not in sc.nodes:
        skinned = [n for n in sc.order if "skin" in sc.nodes[n].spec]
        if args.node is not None:
            raise SystemExit(term.red(f"no node {args.node!r}") + f"; skinned nodes: {', '.join(skinned) or 'none'}")
        if not skinned:
            raise SystemExit(term.red("scene has no skinned nodes"))
        args.node = skinned[0]
    node = sc.nodes[args.node]
    if "skin" not in node.spec:
        raise SystemExit(term.red(f"{args.node!r} has no skin"))
    sk = _skin_spec(node)
    bf = float(sk.get("bind_frame", sc.frame_start))
    rest = sc.object_mesh(args.node, bf)
    data = skin_data(sc, node, rest)
    W, joints = data["weights"], data["joints"]
    if args.list or not (args.output or args.export or args.show):
        nz = (W > 1e-4).sum(axis=1)
        print(term.bold(args.node) + f": {len(W)} verts, {len(joints)} joints, "
              f"influences/vert max {nz.max()} mean {nz.mean():.2f}, sum error {np.abs(W.sum(1) - 1).max():.1e}")
        width = max(len(j) for j in joints)
        for k, j in enumerate(joints):
            col = W[:, k]
            n = int((col > 1e-4).sum())
            bar = "█" * int(round(20 * col.sum() / max(1e-9, W.sum())))
            print(f"  {j:<{width}}  {n:6d} verts  max {col.max():.3f}  {term.dim(bar)}")
    if args.export:
        wp.export_weights(args.export, W, joints, data.get("rest_world"), node=args.node)
        print(term.green(f"wrote {args.export}") + f" ({len(W)} verts, {len(joints)} joints)")
    if args.output or args.show:
        if args.joint and args.joint not in joints:
            raise SystemExit(term.red(f"{args.joint!r} is not a skin joint") + f": {', '.join(joints)}")
        w, h = cli.parse_res(args.res) if args.res else (640, 480)
        frame = float(args.frame) if args.frame is not None else None
        cam = args.camera if args.camera else "auto"
        img = wp.render_heatmap(sc, args.node, args.joint, frame, w, h, camera=cam, view=args.view,
                                posed=frame is not None)
        if args.output:
            from .io.images import write_image
            write_image(args.output, img)
            print(term.green(f"wrote {args.output}") + f"  ({args.joint or 'all influences'})")
        if args.show:
            term.show_image(img, width=args.width)
    return 0


# ----------------------------------------------------------------------------- registration
def register(sub, add_overrides):
    p = sub.add_parser("graph", help="graph editor: plot animation curves in the terminal or to PNG")
    p.add_argument("scene")
    p.add_argument("channels", nargs="*", help="node.attr[.comp] (default: every animated transform)")
    p.add_argument("-f", "--frames", help="frame range, e.g. 1-120")
    p.add_argument("-o", "--output", help="write a PNG graph (curves, keys, tangent handles)")
    p.add_argument("--size", help="PNG size, e.g. 1600x800")
    p.add_argument("-w", "--width", type=int, help="terminal columns")
    p.add_argument("--height", type=int, help="terminal rows")
    p.add_argument("--min", type=float, help="value axis minimum")
    p.add_argument("--max", type=float, help="value axis maximum")
    p.add_argument("--current", type=float, help="draw a current-frame marker")
    p.add_argument("--normalize", action="store_true", help="scale every curve to 0..1 (Maya 'normalize')")
    p.add_argument("--keys", action="store_true", help="also list key values")
    p.add_argument("--hide-flat", action="store_true", dest="hide_flat", help="hide components that never change")
    p.add_argument("--title")
    p.add_argument("--light", action="store_true", help="light PNG theme")
    p.add_argument("--show", action="store_true", help="draw the PNG in the terminal")
    p.add_argument("--terminal", action="store_true", dest="show_terminal", help="terminal plot even with -o")
    add_overrides(p)
    p.set_defaults(func=cmd_graph)

    p = sub.add_parser("key", help="edit keyframes in a .json scene: set/delete/move/interp/tangent/infinity")
    p.add_argument("op", choices=["set", "delete", "move", "interp", "tangent", "infinity", "clear"])
    p.add_argument("scene")
    p.add_argument("channel", help="node.attr[.comp], e.g. ball.translate.y")
    p.add_argument("value", nargs="?", help="value for 'set' (number or JSON list); default = current value")
    p.add_argument("-t", "--frame", help="key frame")
    p.add_argument("--to", type=float, help="new frame for 'move'")
    p.add_argument("-i", "--interp", help="linear, step, bezier, auto, ease_in_out, ...")
    p.add_argument("--in", type=float, dest="in_slope", help="in tangent slope (value/frame)")
    p.add_argument("--out", type=float, dest="out_slope", help="out tangent slope (value/frame)")
    p.add_argument("--pre", help="pre-infinity: constant, linear, cycle, cycle_offset, oscillate")
    p.add_argument("--post", help="post-infinity")
    p.set_defaults(func=cmd_key)

    p = sub.add_parser("colorspaces", help="list colour spaces, print conversion matrices, convert values")
    p.add_argument("--from", dest="src", default="lin_srgb", help="source space (default lin_srgb)")
    p.add_argument("--to", dest="dst", default="acescg", help="target space (default acescg)")
    p.add_argument("--matrix", action="store_true", help="print the 3x3 linear matrix --from -> --to")
    p.add_argument("--convert", nargs=3, metavar=("R", "G", "B"), help="convert one colour --from -> --to")
    p.set_defaults(func=cmd_colorspaces)

    p = sub.add_parser("keys", help="list keyframes of channels")
    p.add_argument("scene")
    p.add_argument("channels", nargs="*")
    add_overrides(p)
    p.set_defaults(func=cmd_keys_list)

    p = sub.add_parser("retarget", help="HumanIK-style retargeting of a BVH / skeleton onto a target rig")
    p.add_argument("source", help=".bvh file, or a .json scene holding the source skeleton")
    p.add_argument("target", help="target scene .json containing the joint hierarchy")
    p.add_argument("-o", "--output", help="output scene (default TARGET_retargeted.json)")
    p.add_argument("--target-root", dest="target_root", help="root joint of the target skeleton")
    p.add_argument("--source-root", dest="source_root", help="root joint of the source skeleton (.json source)")
    p.add_argument("--map", help="JSON object or file {target_joint: source_joint} overriding the auto mapping")
    p.add_argument("--list-map", action="store_true", dest="list_map", help="print the joint mapping and exit")
    p.add_argument("--scale", type=float, default=0.01, help="BVH unit scale (default 0.01: cm -> m)")
    p.add_argument("--start", type=int, default=None, help="first frame of the retargeted motion")
    p.add_argument("--yaw", type=float, default=None, help="rotate the source about +Y (default: auto facing)")
    p.add_argument("--height-scale", type=float, default=None, dest="height_scale",
                   help="root motion scale (default: target/source hip height)")
    p.add_argument("--in-place", action="store_true", dest="in_place", help="remove horizontal root travel")
    p.add_argument("--no-root-motion", action="store_true", dest="no_root_motion", help="keep the hips at rest")
    p.add_argument("--no-match-rest", action="store_true", dest="no_match_rest",
                   help="do not align source rest bones to the target rest pose")
    p.add_argument("--foot-lock", action="store_true", dest="foot_lock", help="pin planted feet with leg IK")
    p.set_defaults(func=cmd_retarget)

    p = sub.add_parser("weights", help="skin weights: list, heat-map preview, export / import")
    p.add_argument("scene")
    p.add_argument("node", nargs="?", help="skinned node (default: the first one)")
    p.add_argument("-j", "--joint", help="joint to visualise (default: every influence in its own colour)")
    p.add_argument("-o", "--output", help="write a heat-map PNG/EXR")
    p.add_argument("--frame", help="pose the mesh at this frame (default: bind pose)")
    p.add_argument("--res", help="heat-map resolution WxH (default 640x480)")
    p.add_argument("--camera", help="scene camera to view through (default: auto framing)")
    p.add_argument("--view", default="front", choices=["front", "back", "left", "right", "top"],
                   help="auto-camera direction")
    p.add_argument("--export", help="write the final weights as JSON (use as \"weights\": \"file.json\")")
    p.add_argument("--list", action="store_true", help="print per-joint statistics")
    p.add_argument("--show", action="store_true", help="draw the heat map in the terminal")
    p.add_argument("--width", type=int, default=None, help="terminal preview width")
    p.add_argument("--set", action="append", help="override settings key=value")
    p.set_defaults(func=cmd_weights)


# ----------------------------------------------------------------------------- grid sims (smoke / liquid)
def bake_grid_sims(sc, args):
    """Simulate every smoke volume and liquid node and write their caches. Returns the count."""
    import os
    import time
    from pathlib import Path
    n = 0
    end = int(getattr(args, "end", None) or sc.frame_end)
    for name in sc.order:
        node = sc.nodes[name]
        kind = None
        if node.type == "volume" and "smoke" in node.spec:
            from . import smoke as mod
            kind = "smoke"
        elif node.type == "liquid":
            from . import liquid as mod
            kind = "liquid"
        if kind is None:
            continue
        sim = mod.get_sim(sc, name)
        if not sim.cache:
            sim.cache = str(Path(args.scene).with_suffix(f".{name}.{kind}.npz"))
            print(term.dim(f"{name}: no 'cache' path in the {kind} block, writing {sim.cache} "
                           f"(add \"cache\": \"{os.path.basename(sim.cache)}\" to reuse it at render time)"))
        bar = term.Progress(f"{kind} {name}")
        t = time.time()
        f0 = int(sim.frame) + 1
        for f in range(f0, end + 1):
            sim.advance_to(f)
            bar.update((f - f0 + 1) / max(1, end - f0 + 1), f"{kind} {name} frame {f}")
        sim.save(sim.cache)
        bar.done(term.green(f"baked {kind} '{name}' to frame {end} -> {sim.cache} in {time.time() - t:.1f}s"))
        n += 1
    return n
