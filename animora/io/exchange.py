"""Scene sampling shared by the interchange writers (glTF / GLB, USD).

:func:`collect` evaluates a scene over a list of frames and returns plain data:

* mesh items: object-space geometry (static, a per-frame list for deforming meshes), per-frame
  world matrices, per-frame visibility and material slots;
* camera items: per-frame world matrices and vertical field of view;
* light items: type, colour, intensity, cone, per-frame world matrices.

Deforming geometry (cloth, skinned meshes, blend shapes, animated modifiers, liquid, hair) is
sampled every frame in world space with an identity transform; rigidly animated geometry is
written once in object space and animated through its transform.
"""
from __future__ import annotations

import math

import numpy as np

MESH_TYPES = ("mesh", "instance", "cloth", "curve", "liquid", "hair")


def material_pbr(scene, ref, frame=None):
    """Evaluate a material reference into a flat PBR dict for exporters."""
    from ..materials import Material, material_from_spec, parse_color
    f = scene.frame_start if frame is None else frame
    if ref is None:
        mat = Material(name="default")
    elif isinstance(ref, str) and ref in scene.materials:
        mat = scene.materials[ref]
    else:
        mat = material_from_spec(ref, name=ref if isinstance(ref, str) else "material")
    v = mat.evaluate(f)
    sc = lambda k: float(np.asarray(v[k], float).ravel()[0])
    base = np.asarray(parse_color(v["base_color"]), float)
    tex = mat.textures.get("base_color")
    tex_file = None
    if isinstance(tex, str):
        tex_file = tex
    elif isinstance(tex, dict):
        if tex.get("type", "image") == "image" and "file" in tex:
            tex_file = tex["file"]
        elif "color1" in tex:  # procedural: export the average of its two colours
            c1 = np.asarray(parse_color(tex.get("color1", [0, 0, 0])), float)
            c2 = np.asarray(parse_color(tex.get("color2", c1)), float)
            base = base * 0.5 * (c1 + c2)
    if tex_file is not None:
        tex_file = str(scene.resolve_path(tex_file))
    em = np.asarray(parse_color(v["emission"]), float) * max(sc("emission_strength"), 0.0 if not np.any(
        np.asarray(parse_color(v["emission"]))) else 1.0)
    return {
        "name": getattr(mat, "name", "material") if not isinstance(ref, str) else ref,
        "base_color": base.tolist(), "alpha": sc("alpha"), "metallic": sc("metallic"),
        "roughness": sc("roughness"), "ior": sc("ior"), "transmission": sc("transmission"),
        "clearcoat": sc("clearcoat"), "clearcoat_roughness": sc("clearcoat_roughness"), "sheen": sc("sheen"),
        "emission": em.tolist(), "base_color_texture": tex_file, "specular": sc("specular"),
    }


def _same_topology(a, b):
    return (a.num_verts == b.num_verts and len(a.indices) == len(b.indices)
            and np.array_equal(a.indices, b.indices) and np.array_equal(a.counts, b.counts))


def collect(scene, frames, nodes=None, hidden=False, preview=False, progress=None):
    from ..scene import _node_visible, camera_params, _material_slots, light_color
    frames = [float(f) for f in frames]
    if scene.dynamics is not None:
        scene.dynamics.simulate(int(math.ceil(max(frames))))
    names = list(nodes) if nodes else [n for n in scene.order
                                       if scene.nodes[n].type in MESH_TYPES + ("camera", "light")]
    res = scene.settings.get("resolution", [960, 540])
    aspect = float(res[0]) / float(res[1])
    items, used_mats = [], {}
    for i, n in enumerate(names):
        if n not in scene.nodes:
            raise ValueError(f"no node {n!r}")
        node = scene.nodes[n]
        vis = [bool(_node_visible(scene, n, f)) for f in frames]
        if not hidden and not any(vis):
            continue
        worlds = [scene.state(f).world(n) for f in frames]
        if node.type == "camera":
            cps = [camera_params(scene, n, f, aspect) for f in frames]
            items.append({"kind": "camera", "name": n, "world": worlds, "visible": vis, "aspect": aspect,
                          "fov_y": [float(c["fov_y"]) for c in cps],
                          "focal_length": [float(c["focal_length"]) for c in cps],
                          "near": float(cps[0].get("near_clip") or 0.01)})
            continue
        if node.type == "light":
            st = scene.state(frames[0])
            g = lambda k, d: node.p(k, d, frames[0], st.ctx)
            kind = node.spec.get("light", "point")
            if node.has("intensity"):
                inten = float(g("intensity", 10.0))
            else:
                inten = float(g("power", 100.0)) / (4 * math.pi)
            items.append({"kind": "light", "name": n, "world": worlds, "visible": vis,
                          "light": kind, "color": light_color(node, frames[0], st.ctx).tolist(),
                          "intensity": inten * float(g("multiplier", 1.0)) * 2.0 ** float(g("exposure", 0.0)),
                          "cone_angle": float(g("cone_angle", 40.0)), "size": g("size", 1.0)})
            continue
        world_space = node.type in ("cloth", "liquid", "hair") or "skin" in node.spec
        deforming = len(frames) > 1 and (world_space or node.geometry_animated())

        def local(f):
            if node.type == "cloth" or "skin" in node.spec:
                return scene.world_mesh(n, f, preview), np.eye(4)
            m = scene.object_mesh(n, f, preview)
            return m, scene.state(f).world(n)

        if deforming:
            meshes, mats = [], []
            for f in frames:
                m, W = local(f)
                if node.type in ("liquid", "hair") and m is not None:
                    m, W = m.transformed(W), np.eye(4)
                meshes.append(m)
                mats.append(W)
            if all(m is None or m.num_verts == 0 for m in meshes):
                continue
            ref = next(m for m in meshes if m is not None and m.num_verts)
            meshes = [m if (m is not None and m.num_verts) else None for m in meshes]
            const = all(m is not None and _same_topology(ref, m) for m in meshes)
            item = {"kind": "mesh", "name": n, "world": mats, "visible": vis,
                    "mode": "morph" if const else "sequence", "meshes": meshes, "mesh": ref}
        else:
            m, W = local(frames[0])
            if m is None or m.num_verts == 0:
                continue
            item = {"kind": "mesh", "name": n, "mode": "static", "mesh": m, "visible": vis,
                    "world": worlds if not (node.type == "cloth" or "skin" in node.spec) else [np.eye(4)] * len(frames)}
        slots = _material_slots(scene, node, item["mesh"])
        keys = []
        for r in slots:
            key = r if isinstance(r, str) else (repr(r) if r is not None else "__default__")
            if key not in used_mats:
                used_mats[key] = material_pbr(scene, r)
                if not isinstance(r, str):
                    used_mats[key]["name"] = f"{n}_mat{len(keys)}" if r is not None else "default"
            keys.append(key)
        item["materials"] = keys
        items.append(item)
        if progress:
            progress(i + 1, len(names))
    return items, used_mats


def matrix_to_trs(M):
    """Decompose an affine matrix to (translation, quaternion xyzw, scale) for glTF / USD."""
    from .. import mathutil as mu
    t, R, s = mu.decompose(M)
    q = np.asarray(mu.matrix_to_quat(R), float)  # (w, x, y, z)
    return t, np.array([q[1], q[2], q[3], q[0]]), s
