"""USD export (``.usda`` text layer).

Writes a self-contained USD stage readable by usdview, Houdini/Solaris, Maya-USD, Omniverse,
Blender and Unreal:

* ``/World/<node>`` Xform prims with ``xformOp:transform`` time samples (world matrices, so the
  stage is flat and exact regardless of rotate order / pivots);
* ``Mesh`` prims (points, faceVertexCounts/Indices, ``primvars:st`` face-varying UVs, normals,
  ``subsets`` per material slot); deforming meshes with constant topology get ``points`` /
  ``normals`` time samples; meshes with changing topology (liquid) get one child per frame switched
  with ``visibility`` time samples;
* ``Camera`` prims (focal length, apertures, clipping, focus);
* ``UsdLux`` lights (Sphere / Disk / Rect / Distant with ``shaping:cone:angle`` for spots);
* ``UsdPreviewSurface`` materials (+ ``UsdUVTexture`` for image base colours).

Units: metersPerUnit = 1, upAxis = Y, timeCodesPerSecond = scene fps; time codes equal Animora
frame numbers.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from .exchange import collect


def _ident(name, used):
    s = re.sub(r"[^A-Za-z0-9_]", "_", str(name)) or "prim"
    if s[0].isdigit():
        s = "_" + s
    base, k = s, 1
    while s in used:
        k += 1
        s = f"{base}_{k}"
    used.add(s)
    return s


def _f(x):
    x = float(x)
    if not math.isfinite(x):
        x = 0.0
    r = f"{x:.6g}"
    return r


def _vec(v):
    return "(" + ", ".join(_f(x) for x in v) + ")"


def _arr(a, per=None):
    a = np.asarray(a)
    if a.ndim == 1:
        if a.dtype.kind in "iu":
            return "[" + ", ".join(str(int(x)) for x in a) + "]"
        return "[" + ", ".join(_f(x) for x in a) + "]"
    return "[" + ", ".join(_vec(r) for r in a) + "]"


def _matrix(M):
    # USD matrices are row-major with row vectors: transpose of our column-vector matrices.
    T = np.asarray(M, float).T
    return "( " + ", ".join("(" + ", ".join(_f(x) for x in row) + ")" for row in T) + " )"


class _W:
    def __init__(self):
        self.lines = []
        self.ind = 0

    def __call__(self, s=""):
        self.lines.append("    " * self.ind + s if s else "")

    def open(self, s):
        self(s)
        self("{")
        self.ind += 1

    def close(self):
        self.ind -= 1
        self("}")


def _samples(w, decl, frames, vals, fmt, constant_check=True):
    if constant_check and all(np.array_equal(np.asarray(vals[0]), np.asarray(v)) for v in vals[1:]):
        w(f"{decl} = {fmt(vals[0])}")
        return
    w(f"{decl}.timeSamples = {{")
    w.ind += 1
    for f, v in zip(frames, vals):
        w(f"{_f(f)}: {fmt(v)},")
    w.ind -= 1
    w("}")


def _vertex_normals(m):
    """Per-vertex normals (USD's default ``vertex`` interpolation, usable with time samples)."""
    n = np.zeros((m.num_verts, 3))
    np.add.at(n, np.asarray(m.indices, int), m.corner_normals())
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def _mesh_body(w, mesh, mat_paths, frames=None, morph=None):
    counts = np.asarray(mesh.counts, int)
    idx = np.asarray(mesh.indices, int)
    w('uniform token subdivisionScheme = "none"')
    w('uniform token orientation = "rightHanded"')
    w(f"int[] faceVertexCounts = {_arr(counts)}")
    w(f"int[] faceVertexIndices = {_arr(idx)}")
    if morph:
        _samples(w, "point3f[] points", frames, [m.verts for m in morph], _arr, constant_check=False)
        _samples(w, "normal3f[] normals", frames, [_vertex_normals(m) for m in morph], _arr, constant_check=False)
    else:
        w(f"point3f[] points = {_arr(mesh.verts)}")
        w(f"normal3f[] normals = {_arr(mesh.corner_normals())} (")
        w('    interpolation = "faceVarying"')
        w(")")
    lo, hi = mesh.verts.min(0), mesh.verts.max(0)
    w(f"float3[] extent = [{_vec(lo)}, {_vec(hi)}]")
    if mesh.uvs is not None:
        uv = np.asarray(mesh.uvs, float).copy()
        w(f"texCoord2f[] primvars:st = {_arr(uv)} (")
        w('    interpolation = "faceVarying"')
        w(")")
    fm = np.asarray(mesh.face_mat if mesh.face_mat is not None else np.zeros(len(counts), int), int)
    slots = np.unique(fm)
    if len(slots) <= 1 or len(mat_paths) <= 1:
        if mat_paths:
            w(f"rel material:binding = <{mat_paths[min(int(slots[0]) if len(slots) else 0, len(mat_paths) - 1)]}>")
        return
    w('uniform token subsetFamily:materialBind:familyType = "partition"')
    for s in slots:
        w.open(f'def GeomSubset "slot{int(s)}" (prepend apiSchemas = ["MaterialBindingAPI"])')
        w('uniform token elementType = "face"')
        w('uniform token familyName = "materialBind"')
        w(f"int[] indices = {_arr(np.nonzero(fm == s)[0])}")
        w(f"rel material:binding = <{mat_paths[min(int(s), len(mat_paths) - 1)]}>")
        w.close()


def export_usd(scene, path, frames=None, nodes=None, hidden=False, step=1, progress=None):
    """Write ``scene`` as a ``.usda`` stage. Returns a summary dict."""
    path = Path(path)
    if frames is None:
        frames = list(np.arange(scene.frame_start, scene.frame_end + 1e-9, step))
    frames = [float(f) for f in frames]
    items, mats = collect(scene, frames, nodes, hidden, progress=progress)
    w = _W()
    w("#usda 1.0")
    w("(")
    w('    defaultPrim = "World"')
    w('    doc = "Exported by Animora"')
    w("    metersPerUnit = 1")
    w('    upAxis = "Y"')
    w(f"    startTimeCode = {_f(frames[0])}")
    w(f"    endTimeCode = {_f(frames[-1])}")
    w(f"    timeCodesPerSecond = {_f(scene.fps)}")
    w(f"    framesPerSecond = {_f(scene.fps)}")
    w(")")
    w()
    w.open('def Xform "World" (kind = "assembly")')
    used = set()
    # materials
    mat_path = {}
    if mats:
        w.open('def Scope "Looks"')
        mused = set()
        for key, pb in mats.items():
            mn = _ident(pb["name"], mused)
            mp = f"/World/Looks/{mn}"
            mat_path[key] = mp
            w.open(f'def Material "{mn}"')
            w(f"token outputs:surface.connect = <{mp}/Surface.outputs:surface>")
            w.open('def Shader "Surface"')
            w('uniform token info:id = "UsdPreviewSurface"')
            tex = pb.get("base_color_texture")
            if tex:
                w(f"color3f inputs:diffuseColor.connect = <{mp}/BaseTex.outputs:rgb>")
            else:
                w(f"color3f inputs:diffuseColor = {_vec(np.clip(pb['base_color'], 0, None))}")
            w(f"float inputs:metallic = {_f(pb['metallic'])}")
            w(f"float inputs:roughness = {_f(pb['roughness'])}")
            w(f"float inputs:ior = {_f(pb['ior'])}")
            w(f"float inputs:opacity = {_f(min(pb['alpha'], 1.0 - 0.9 * pb['transmission']))}")
            w(f"float inputs:clearcoat = {_f(pb['clearcoat'])}")
            w(f"float inputs:clearcoatRoughness = {_f(pb['clearcoat_roughness'])}")
            w(f"color3f inputs:emissiveColor = {_vec(pb['emission'])}")
            w("token outputs:surface")
            w.close()
            if tex:
                w.open('def Shader "BaseTex"')
                w('uniform token info:id = "UsdUVTexture"')
                w(f"asset inputs:file = @{tex}@")
                w(f"float2 inputs:st.connect = <{mp}/UV.outputs:result>")
                w('token inputs:wrapS = "repeat"')
                w('token inputs:wrapT = "repeat"')
                w(f"float4 inputs:scale = ({_f(pb['base_color'][0])}, {_f(pb['base_color'][1])}, "
                  f"{_f(pb['base_color'][2])}, 1)")
                w("float3 outputs:rgb")
                w.close()
                w.open('def Shader "UV"')
                w('uniform token info:id = "UsdPrimvarReader_float2"')
                w('string inputs:varname = "st"')
                w("float2 outputs:result")
                w.close()
            w.close()
        w.close()
    summary = {"meshes": 0, "cameras": 0, "lights": 0, "frames": len(frames)}
    res = scene.settings.get("resolution", [960, 540])
    for it in items:
        nm = _ident(it["name"], used)
        vis_tok = ['"inherited"' if v else '"invisible"' for v in it["visible"]]
        if it["kind"] == "camera":
            w.open(f'def Camera "{nm}"')
            w('uniform token[] xformOpOrder = ["xformOp:transform"]')
            _samples(w, "matrix4d xformOp:transform", frames, it["world"], _matrix)
            # vertical aperture from the fov and focal length (mm, tenths of scene units in USD)
            fl = it["focal_length"]
            vap = [2 * f_ * math.tan(fy / 2) for f_, fy in zip(fl, it["fov_y"])]
            _samples(w, "float focalLength", frames, fl, _f)
            _samples(w, "float verticalAperture", frames, vap, _f)
            _samples(w, "float horizontalAperture", frames, [v * it["aspect"] for v in vap], _f)
            w(f"float2 clippingRange = ({_f(it['near'])}, 100000)")
            w.close()
            summary["cameras"] += 1
            continue
        if it["kind"] == "light":
            kind = it["light"]
            typ = {"sun": "DistantLight", "distant": "DistantLight", "directional": "DistantLight",
                   "area": "RectLight", "rect": "RectLight", "quad": "RectLight", "disk": "DiskLight"}.get(kind, "SphereLight")
            schemas = ' (prepend apiSchemas = ["ShapingAPI"])' if kind == "spot" else ""
            w.open(f'def {typ} "{nm}"{schemas}')
            w('uniform token[] xformOpOrder = ["xformOp:transform"]')
            _samples(w, "matrix4d xformOp:transform", frames, it["world"], _matrix)
            w(f"color3f inputs:color = {_vec(it['color'])}")
            w(f"float inputs:intensity = {_f(it['intensity'])}")
            size = float(np.asarray(it["size"], float).ravel()[0])
            if typ == "RectLight":
                w(f"float inputs:width = {_f(size)}")
                w(f"float inputs:height = {_f(size)}")
            elif typ in ("DiskLight", "SphereLight"):
                w(f"float inputs:radius = {_f(size * 0.5 if typ == 'DiskLight' else 0.05)}")
            if kind == "spot":
                w(f"float inputs:shaping:cone:angle = {_f(it['cone_angle'] * 0.5)}")
                w("float inputs:shaping:cone:softness = 0.2")
            _samples(w, "token visibility", frames, vis_tok, str)
            w.close()
            summary["lights"] += 1
            continue
        mps = [mat_path[k] for k in it["materials"]]
        api = ' (prepend apiSchemas = ["MaterialBindingAPI"])'
        if it["mode"] == "sequence":
            w.open(f'def Xform "{nm}"')
            for fi, (m, W) in enumerate(zip(it["meshes"], it["world"])):
                if m is None:
                    continue
                w.open(f'def Mesh "f{fi:04d}"{api}')
                w('uniform token[] xformOpOrder = ["xformOp:transform"]')
                w(f"matrix4d xformOp:transform = {_matrix(W)}")
                _samples(w, "token visibility", frames,
                         ['"inherited"' if (k == fi and it["visible"][k]) else '"invisible"' for k in range(len(frames))],
                         str)
                _mesh_body(w, m, mps)
                w.close()
            w.close()
        else:
            w.open(f'def Mesh "{nm}"{api}')
            w('uniform token[] xformOpOrder = ["xformOp:transform"]')
            _samples(w, "matrix4d xformOp:transform", frames, it["world"], _matrix)
            _samples(w, "token visibility", frames, vis_tok, str)
            if it["mode"] == "morph":
                _mesh_body(w, it["meshes"][0], mps, frames, it["meshes"])
            else:
                _mesh_body(w, it["mesh"], mps)
            w.close()
        summary["meshes"] += 1
    w.close()
    path.write_text("\n".join(l for l in w.lines if l is not None) + "\n")
    summary["bytes"] = path.stat().st_size
    return summary
