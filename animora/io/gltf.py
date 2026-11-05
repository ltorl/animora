"""glTF 2.0 / GLB export and import.

Export (:func:`export_gltf`) writes meshes, PBR materials (with KHR extensions for transmission,
IOR, clearcoat, sheen and emissive strength), embedded base-colour textures, cameras,
KHR_lights_punctual lights and animation:

* rigidly animated nodes get translation / rotation / scale channels sampled every frame;
* deforming meshes with constant topology (cloth, skinning, blend shapes, hair) become one morph
  target per frame, driven by a weight channel (linear blend between consecutive frames);
* meshes whose topology changes (liquid) are written as one mesh per frame, switched with step
  scale keys;
* hidden frames are encoded as zero scale.

Import (:func:`read_gltf`) reads ``.gltf`` (embedded or external buffers) and ``.glb``: triangle
primitives (lists, strips, fans), node hierarchies, normals, UVs, materials and textures.
``file.glb#Name`` selects one node's mesh in its own local space; :func:`gltf_to_scene` turns a
glTF file into an Animora scene dict with hierarchy, cameras and keyframed TRS animation.
"""
from __future__ import annotations

import base64
import io
import json
import math
import struct
from pathlib import Path

import numpy as np

from .exchange import collect, matrix_to_trs

FLOAT, UINT32, UINT16, UINT8 = 5126, 5125, 5123, 5121
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}
_DTYPE = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8, 5122: np.int16, 5120: np.int8}


# ======================================================================================
# writer
# ======================================================================================
class _Builder:
    def __init__(self):
        self.bin = bytearray()
        self.doc = {"asset": {"version": "2.0", "generator": "Animora"}, "buffers": [], "bufferViews": [],
                    "accessors": [], "meshes": [], "nodes": [], "materials": [], "scenes": [{"nodes": []}],
                    "scene": 0}
        self.ext_used = set()

    def view(self, data: bytes, target=None):
        while len(self.bin) % 4:
            self.bin.append(0)
        bv = {"buffer": 0, "byteOffset": len(self.bin), "byteLength": len(data)}
        if target:
            bv["target"] = target
        self.bin.extend(data)
        self.doc["bufferViews"].append(bv)
        return len(self.doc["bufferViews"]) - 1

    def accessor(self, arr, typ, ctype=FLOAT, target=None, minmax=False):
        arr = np.ascontiguousarray(arr, dtype=_DTYPE[ctype])
        acc = {"bufferView": self.view(arr.tobytes(), target), "componentType": ctype,
               "count": int(arr.shape[0]), "type": typ}
        if minmax and len(arr):
            a2 = arr.reshape(len(arr), -1)
            acc["min"] = [float(x) for x in a2.min(0)]
            acc["max"] = [float(x) for x in a2.max(0)]
        self.doc["accessors"].append(acc)
        return len(self.doc["accessors"]) - 1

    def finish(self, path: Path):
        doc = self.doc
        for k in ("meshes", "materials", "nodes"):
            if not doc[k]:
                doc.pop(k)
        if self.ext_used:
            doc["extensionsUsed"] = sorted(self.ext_used)
        while len(self.bin) % 4:
            self.bin.append(0)
        doc["buffers"] = [{"byteLength": len(self.bin)}]
        if path.suffix.lower() == ".glb":
            js = json.dumps(doc, separators=(",", ":")).encode()
            js += b" " * (-len(js) % 4)
            out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(js) + 8 + len(self.bin))
            out += struct.pack("<I4s", len(js), b"JSON") + js
            out += struct.pack("<I4s", len(self.bin), b"BIN\0") + bytes(self.bin)
            path.write_bytes(out)
        else:
            doc["buffers"][0]["uri"] = "data:application/octet-stream;base64," + base64.b64encode(
                bytes(self.bin)).decode()
            path.write_text(json.dumps(doc, indent=1))


def _weld(mesh):
    """Triangulate and split vertices by (vertex, normal, uv). Returns dict of arrays."""
    tri = mesh.to_triangles()
    vidx = tri["vidx"].reshape(-1)
    N = tri["N"].reshape(-1, 3)
    UV = tri["UV"].reshape(-1, 2)
    key = np.concatenate([vidx[:, None].astype(np.float64), np.round(N, 4), np.round(UV, 5)], axis=1)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    return {"vidx": vidx[first], "corner": first, "N": N[first], "UV": UV[first],
            "tris": inv.reshape(-1, 3), "mat": tri["mat"]}


def _material_json(b, pbr, base_dir_cache):
    m = {"name": pbr["name"], "pbrMetallicRoughness": {
        "baseColorFactor": [*(float(np.clip(c, 0, 1)) for c in pbr["base_color"]), float(np.clip(pbr["alpha"], 0, 1))],
        "metallicFactor": float(np.clip(pbr["metallic"], 0, 1)),
        "roughnessFactor": float(np.clip(pbr["roughness"], 0, 1))}, "doubleSided": True}
    if pbr["alpha"] < 0.999:
        m["alphaMode"] = "BLEND"
    ext = {}
    em = np.asarray(pbr["emission"], float)
    if em.max() > 0:
        s = float(em.max())
        if s > 1:
            ext["KHR_materials_emissive_strength"] = {"emissiveStrength": s}
            em = em / s
        m["emissiveFactor"] = [float(x) for x in em]
    if pbr["transmission"] > 0:
        ext["KHR_materials_transmission"] = {"transmissionFactor": float(pbr["transmission"])}
    if abs(pbr["ior"] - 1.5) > 1e-6:
        ext["KHR_materials_ior"] = {"ior": float(pbr["ior"])}
    if pbr["clearcoat"] > 0:
        ext["KHR_materials_clearcoat"] = {"clearcoatFactor": float(pbr["clearcoat"]),
                                          "clearcoatRoughnessFactor": float(pbr["clearcoat_roughness"])}
    if pbr["sheen"] > 0:
        ext["KHR_materials_sheen"] = {"sheenColorFactor": [float(min(pbr["sheen"], 1))] * 3,
                                      "sheenRoughnessFactor": 0.5}
    tex = pbr.get("base_color_texture")
    if tex and Path(tex).exists():
        if tex not in base_dir_cache:
            p = Path(tex)
            data, mime = p.read_bytes(), "image/png"
            if p.suffix.lower() in (".jpg", ".jpeg"):
                mime = "image/jpeg"
            elif p.suffix.lower() != ".png":
                from .images import read_image, linear_to_srgb
                from PIL import Image
                img = np.clip(linear_to_srgb(read_image(p)[..., :3]), 0, 1)
                bio = io.BytesIO()
                Image.fromarray((img * 255 + 0.5).astype(np.uint8)).save(bio, "PNG")
                data = bio.getvalue()
            d = b.doc
            d.setdefault("images", []).append({"bufferView": b.view(data), "mimeType": mime})
            d.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}])
            d.setdefault("textures", []).append({"source": len(d["images"]) - 1, "sampler": 0})
            base_dir_cache[tex] = len(d["textures"]) - 1
        m["pbrMetallicRoughness"]["baseColorTexture"] = {"index": base_dir_cache[tex]}
    if ext:
        m["extensions"] = ext
        b.ext_used.update(ext)
    return m


def _mesh_json(b, name, mesh, mat_ids, morph_meshes=None):
    w = _weld(mesh)
    P = mesh.verts[w["vidx"]]
    uv = w["UV"].copy()
    uv[:, 1] = 1.0 - uv[:, 1]
    attrs = {"POSITION": b.accessor(P, "VEC3", target=34962, minmax=True),
             "NORMAL": b.accessor(w["N"], "VEC3", target=34962)}
    if mesh.uvs is not None:
        attrs["TEXCOORD_0"] = b.accessor(uv, "VEC2", target=34962)
    targets = []
    if morph_meshes:
        for mm in morph_meshes:
            dP = mm.verts[w["vidx"]] - P
            dN = mm.to_triangles()["N"].reshape(-1, 3)[w["corner"]] - w["N"]
            targets.append({"POSITION": b.accessor(dP, "VEC3", target=34962, minmax=True),
                            "NORMAL": b.accessor(dN, "VEC3", target=34962)})
    big = len(P) > 65535
    prims = []
    for slot in np.unique(w["mat"]):
        t = w["tris"][w["mat"] == slot].reshape(-1)
        prim = {"attributes": attrs, "mode": 4,
                "indices": b.accessor(t, "SCALAR", UINT32 if big else UINT16, target=34963)}
        if mat_ids:
            prim["material"] = mat_ids[min(int(slot), len(mat_ids) - 1)]
        if targets:
            prim["targets"] = targets
        prims.append(prim)
    mj = {"name": name, "primitives": prims}
    if targets:
        mj["weights"] = [0.0] * len(targets)
    b.doc["meshes"].append(mj)
    return len(b.doc["meshes"]) - 1


def _trs_channels(b, anim, node_idx, times_acc, worlds, visible, force_step_scale=False):
    T, Q, S = [], [], []
    for W in worlds:
        t, q, s = matrix_to_trs(W)
        if Q and np.dot(Q[-1], q) < 0:
            q = -q
        T.append(t), Q.append(q), S.append(s)
    T, Q, S = np.array(T), np.array(Q), np.array(S)
    vis = np.asarray(visible, bool)
    S = S * vis[:, None]
    node = b.doc["nodes"][node_idx]
    node["translation"], node["rotation"], node["scale"] = T[0].tolist(), Q[0].tolist(), S[0].tolist()
    if len(worlds) < 2:
        return
    for path, vals, typ in (("translation", T, "VEC3"), ("rotation", Q, "VEC4"), ("scale", S, "VEC3")):
        if np.allclose(vals, vals[0], atol=1e-7):
            continue
        interp = "STEP" if (path == "scale" and (force_step_scale or not vis.all())) else "LINEAR"
        anim["samplers"].append({"input": times_acc, "output": b.accessor(vals, typ), "interpolation": interp})
        anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": node_idx, "path": path}})


def export_gltf(scene, path, frames=None, nodes=None, hidden=False, step=1, progress=None):
    """Write ``scene`` to ``path`` (.glb or .gltf). ``frames``: list of frames (default: the
    scene range, every ``step`` frames). Returns a summary dict."""
    path = Path(path)
    if frames is None:
        frames = list(np.arange(scene.frame_start, scene.frame_end + 1e-9, step))
    frames = [float(f) for f in frames]
    items, mats = collect(scene, frames, nodes, hidden, progress=progress)
    b = _Builder()
    tex_cache = {}
    mat_index = {}
    for key, pbr in mats.items():
        b.doc["materials"].append(_material_json(b, pbr, tex_cache))
        mat_index[key] = len(b.doc["materials"]) - 1
    times = np.array([(f - frames[0]) / scene.fps for f in frames], np.float32)
    anim = {"name": "animation", "samplers": [], "channels": []}
    times_acc = b.accessor(times, "SCALAR", minmax=True) if len(frames) > 1 else None
    roots = b.doc["scenes"][0]["nodes"]
    lights = []
    summary = {"meshes": 0, "morph_targets": 0, "cameras": 0, "lights": 0, "frames": len(frames)}
    for it in items:
        name = it["name"]
        if it["kind"] == "camera":
            cam = {"type": "perspective", "name": name, "perspective": {
                "yfov": float(it["fov_y"][0]), "aspectRatio": float(it["aspect"]), "znear": max(it["near"], 1e-3),
                "zfar": 10000.0}}
            b.doc.setdefault("cameras", []).append(cam)
            b.doc["nodes"].append({"name": name, "camera": len(b.doc["cameras"]) - 1})
            idx = len(b.doc["nodes"]) - 1
            roots.append(idx)
            _trs_channels(b, anim, idx, times_acc, it["world"], [True] * len(frames))
            summary["cameras"] += 1
            continue
        if it["kind"] == "light":
            kind = it["light"]
            L = {"name": name, "color": [float(c) for c in it["color"]], "intensity": float(it["intensity"])}
            if kind == "spot":
                L["type"] = "spot"
                outer = math.radians(it["cone_angle"] * 0.5)
                L["spot"] = {"innerConeAngle": outer * 0.8, "outerConeAngle": outer}
            elif kind in ("sun", "distant", "directional"):
                L["type"] = "directional"
            else:
                L["type"] = "point"
            lights.append(L)
            b.doc["nodes"].append({"name": name, "extensions": {"KHR_lights_punctual": {"light": len(lights) - 1}}})
            idx = len(b.doc["nodes"]) - 1
            roots.append(idx)
            _trs_channels(b, anim, idx, times_acc, it["world"], it["visible"])
            summary["lights"] += 1
            continue
        mids = [mat_index[k] for k in it["materials"]]
        if it["mode"] == "sequence":
            b.doc["nodes"].append({"name": name, "children": []})
            gidx = len(b.doc["nodes"]) - 1
            roots.append(gidx)
            for fi, (m, W) in enumerate(zip(it["meshes"], it["world"])):
                if m is None:
                    continue
                mi = _mesh_json(b, f"{name}_{fi:04d}", m, mids)
                b.doc["nodes"].append({"name": f"{name}_{fi:04d}", "mesh": mi})
                idx = len(b.doc["nodes"]) - 1
                b.doc["nodes"][gidx]["children"].append(idx)
                vis = [it["visible"][fi] and k == fi for k in range(len(frames))]
                _trs_channels(b, anim, idx, times_acc, [W] * len(frames), vis, force_step_scale=True)
                summary["meshes"] += 1
            continue
        morph = it["meshes"][1:] if it["mode"] == "morph" else None
        base = it["meshes"][0] if it["mode"] == "morph" else it["mesh"]
        mi = _mesh_json(b, name, base, mids, morph)
        b.doc["nodes"].append({"name": name, "mesh": mi})
        idx = len(b.doc["nodes"]) - 1
        roots.append(idx)
        _trs_channels(b, anim, idx, times_acc, it["world"], it["visible"])
        summary["meshes"] += 1
        if morph:
            nt = len(morph)
            Wt = np.zeros((len(frames), nt), np.float32)
            for k in range(1, len(frames)):
                Wt[k, k - 1] = 1.0
            anim["samplers"].append({"input": times_acc, "output": b.accessor(Wt.reshape(-1), "SCALAR"),
                                     "interpolation": "LINEAR"})
            anim["channels"].append({"sampler": len(anim["samplers"]) - 1, "target": {"node": idx, "path": "weights"}})
            summary["morph_targets"] += nt
    if lights:
        b.doc["extensions"] = {"KHR_lights_punctual": {"lights": lights}}
        b.ext_used.add("KHR_lights_punctual")
    if anim["channels"]:
        b.doc["animations"] = [anim]
    b.finish(path)
    summary["bytes"] = path.stat().st_size
    return summary


# ======================================================================================
# reader
# ======================================================================================
def _load(path):
    path = Path(path)
    raw = path.read_bytes()
    bin_chunk = None
    if raw[:4] == b"glTF":
        _, _, total = struct.unpack_from("<4sII", raw, 0)
        off = 12
        doc = None
        while off < total:
            ln, typ = struct.unpack_from("<I4s", raw, off)
            data = raw[off + 8: off + 8 + ln]
            if typ == b"JSON":
                doc = json.loads(data.decode())
            elif typ == b"BIN\0":
                bin_chunk = data
            off += 8 + ln
    else:
        doc = json.loads(raw.decode())
    buffers = []
    for i, buf in enumerate(doc.get("buffers", [])):
        uri = buf.get("uri")
        if uri is None:
            buffers.append(bin_chunk)
        elif uri.startswith("data:"):
            buffers.append(base64.b64decode(uri.split(",", 1)[1]))
        else:
            from urllib.parse import unquote
            buffers.append((path.parent / unquote(uri)).read_bytes())
    return doc, buffers


def _view_bytes(doc, buffers, bv_idx):
    bv = doc["bufferViews"][bv_idx]
    buf = buffers[bv["buffer"]]
    o = bv.get("byteOffset", 0)
    return buf[o: o + bv["byteLength"]], bv.get("byteStride")


def _accessor(doc, buffers, idx):
    acc = doc["accessors"][idx]
    nc = _NCOMP[acc["type"]]
    dt = np.dtype(_DTYPE[acc["componentType"]])
    cnt = acc["count"]
    if "bufferView" in acc:
        data, stride = _view_bytes(doc, buffers, acc["bufferView"])
        off = acc.get("byteOffset", 0)
        es = dt.itemsize * nc
        if stride and stride != es:
            a = np.ndarray((cnt, nc), dt, buffer=data, offset=off, strides=(stride, dt.itemsize)).copy()
        else:
            a = np.frombuffer(data, dt, cnt * nc, off).reshape(cnt, nc).copy()
    else:
        a = np.zeros((cnt, nc), dt)
    sp = acc.get("sparse")
    if sp:
        ii = sp["indices"]
        idt = np.dtype(_DTYPE[ii["componentType"]])
        d, _ = _view_bytes(doc, buffers, ii["bufferView"])
        ind = np.frombuffer(d, idt, sp["count"], ii.get("byteOffset", 0)).astype(np.int64)
        vv = sp["values"]
        d, _ = _view_bytes(doc, buffers, vv["bufferView"])
        a = a.astype(dt)
        a[ind] = np.frombuffer(d, dt, sp["count"] * nc, vv.get("byteOffset", 0)).reshape(-1, nc)
    if acc.get("normalized"):
        a = a.astype(np.float64) / float(np.iinfo(dt).max)
    return a if nc > 1 else a[:, 0]


def _node_local(nd):
    from .. import mathutil as mu
    if "matrix" in nd:
        return np.asarray(nd["matrix"], float).reshape(4, 4).T
    t = nd.get("translation", [0, 0, 0])
    q = nd.get("rotation", [0, 0, 0, 1])
    s = nd.get("scale", [1, 1, 1])
    M = mu.quat_to_matrix([q[3], q[0], q[1], q[2]])
    M[:3, :3] = M[:3, :3] * np.asarray(s, float)[None, :]
    M[:3, 3] = t
    return M


def _gltf_materials(doc, buffers, path):
    """glTF materials -> Animora material specs (textures extracted next to the file)."""
    out = []
    tex_dir = None
    for i, m in enumerate(doc.get("materials", [])):
        pb = m.get("pbrMetallicRoughness", {})
        bc = pb.get("baseColorFactor", [1, 1, 1, 1])
        spec = {"name": m.get("name", f"material{i}"), "base_color": [float(x) for x in bc[:3]],
                "metallic": float(pb.get("metallicFactor", 1.0)), "roughness": float(pb.get("roughnessFactor", 1.0))}
        if bc[3] < 1 and m.get("alphaMode", "OPAQUE") != "OPAQUE":
            spec["alpha"] = float(bc[3])
        em = m.get("emissiveFactor")
        ext = m.get("extensions", {})
        if em and max(em) > 0:
            spec["emission"] = [float(x) for x in em]
            spec["emission_strength"] = float(ext.get("KHR_materials_emissive_strength", {}).get("emissiveStrength", 1.0))
        if "KHR_materials_transmission" in ext:
            spec["transmission"] = float(ext["KHR_materials_transmission"].get("transmissionFactor", 0))
        if "KHR_materials_ior" in ext:
            spec["ior"] = float(ext["KHR_materials_ior"].get("ior", 1.5))
        if "KHR_materials_clearcoat" in ext:
            spec["clearcoat"] = float(ext["KHR_materials_clearcoat"].get("clearcoatFactor", 0))
        if "KHR_materials_sheen" in ext:
            spec["sheen"] = float(max(ext["KHR_materials_sheen"].get("sheenColorFactor", [0, 0, 0])))
        bt = pb.get("baseColorTexture")
        if bt is not None and "textures" in doc:
            try:
                img = doc["images"][doc["textures"][bt["index"]]["source"]]
                if "uri" in img and not img["uri"].startswith("data:"):
                    tfile = str((Path(path).parent / img["uri"]).resolve())
                else:
                    if "uri" in img:
                        data = base64.b64decode(img["uri"].split(",", 1)[1])
                        mime = img["uri"][5:].split(";")[0]
                    else:
                        data, _ = _view_bytes(doc, buffers, img["bufferView"])
                        mime = img.get("mimeType", "image/png")
                    if tex_dir is None:
                        tex_dir = Path(path).with_suffix(".textures")
                        tex_dir.mkdir(exist_ok=True)
                    tfile = str(tex_dir / f"tex{bt['index']}{'.jpg' if 'jpeg' in mime else '.png'}")
                    Path(tfile).write_bytes(bytes(data))
                spec["base_color"] = {"type": "image", "file": tfile, "colorspace": "srgb"}
                if bc[:3] != [1, 1, 1]:
                    spec["base_color_factor"] = [float(x) for x in bc[:3]]
            except (KeyError, IndexError, OSError):
                pass
        out.append(spec)
    return out


def _prim_tris(prim, n):
    mode = prim.get("mode", 4)
    return mode


def _node_mesh(doc, buffers, mesh_idx, nmat, morph=None):
    """Mesh of glTF mesh ``mesh_idx`` (local space). Returns (verts, tri idx, uvs per corner, normals per vertex, face mats)."""
    V, F, UV, N, FM = [], [], [], [], []
    base = 0
    for prim in doc["meshes"][mesh_idx]["primitives"]:
        a = prim["attributes"]
        if "POSITION" not in a:
            continue
        P = _accessor(doc, buffers, a["POSITION"]).astype(np.float64)
        if morph is not None:
            tg = prim.get("targets", [])
            if morph >= len(tg):
                raise ValueError(f"morph target {morph} out of range ({len(tg)} targets)")
            if "POSITION" in tg[morph]:
                P = P + _accessor(doc, buffers, tg[morph]["POSITION"]).astype(np.float64)
        n = len(P)
        if "indices" in prim:
            idx = _accessor(doc, buffers, prim["indices"]).astype(np.int64)
        else:
            idx = np.arange(n)
        mode = prim.get("mode", 4)
        if mode == 5:  # strip
            t = np.array([[idx[i], idx[i + 1], idx[i + 2]] if i % 2 == 0 else [idx[i + 1], idx[i], idx[i + 2]]
                          for i in range(len(idx) - 2)])
        elif mode == 6:  # fan
            t = np.array([[idx[0], idx[i], idx[i + 1]] for i in range(1, len(idx) - 1)])
        elif mode == 4:
            t = idx[: len(idx) // 3 * 3].reshape(-1, 3)
        else:
            continue  # points / lines: no surface
        if len(t) == 0:
            continue
        uv = _accessor(doc, buffers, a["TEXCOORD_0"]).astype(np.float64) if "TEXCOORD_0" in a else None
        if uv is not None:
            uv = uv.copy()
            uv[:, 1] = 1.0 - uv[:, 1]
        nrm = _accessor(doc, buffers, a["NORMAL"]).astype(np.float64) if "NORMAL" in a else None
        V.append(P)
        F.append(t + base)
        UV.append(uv[t.reshape(-1)] if uv is not None else np.zeros((t.size, 2)))
        N.append(nrm if nrm is not None else np.zeros((n, 3)))
        mi = prim.get("material", -1)
        FM.append(np.full(len(t), mi if mi >= 0 else nmat))
        base += n
    if not V:
        return None
    return np.vstack(V), np.vstack(F), np.vstack(UV), np.vstack(N), np.concatenate(FM)


def read_gltf(path, select=None):
    """Read a .gltf/.glb. Returns ``(Mesh, [material spec])``.

    Without ``select`` every mesh in the default scene is merged in world space. With ``select``
    (a node name) only that node's own mesh is returned, in its local space."""
    from ..geometry import Mesh
    doc, buffers = _load(path)
    mats = _gltf_materials(doc, buffers, path)
    nmat = len(mats)
    nodes = doc.get("nodes", [])
    parts = []
    morph = None
    if select is not None and "@" in select:
        select, mk = select.rsplit("@", 1)
        morph = int(mk)
    if select is not None:
        hits = [i for i, nd in enumerate(nodes) if nd.get("name") == select]
        if not hits:
            mh = [i for i, m in enumerate(doc.get("meshes", [])) if m.get("name") == select]
            if not mh:
                raise ValueError(f"{path}: no node or mesh named {select!r}")
            parts.append((mh[0], np.eye(4)))
        else:
            nd = nodes[hits[0]]
            if "mesh" not in nd:
                raise ValueError(f"{path}: node {select!r} has no mesh")
            parts.append((nd["mesh"], np.eye(4)))
    else:
        scn = doc.get("scenes", [{}])[doc.get("scene", 0)] if doc.get("scenes") else {
            "nodes": [i for i in range(len(nodes))
                      if not any(i in nd.get("children", []) for nd in nodes)]}
        stack = [(i, np.eye(4)) for i in scn.get("nodes", [])]
        while stack:
            i, PM = stack.pop()
            nd = nodes[i]
            M = PM @ _node_local(nd)
            if "mesh" in nd:
                parts.append((nd["mesh"], M))
            stack.extend((c, M) for c in nd.get("children", []))
        if not parts and doc.get("meshes"):
            parts = [(i, np.eye(4)) for i in range(len(doc["meshes"]))]
    Vs, Fs, UVs, FMs, Ns = [], [], [], [], []
    base = 0
    for mi, M in parts:
        r = _node_mesh(doc, buffers, mi, nmat, morph)
        if r is None:
            continue
        V, F, UV, N, FM = r
        V = V @ M[:3, :3].T + M[:3, 3]
        if np.linalg.det(M[:3, :3]) < 0:
            F = F[:, ::-1]
            UV = UV.reshape(-1, 3, 2)[:, ::-1].reshape(-1, 2)
        Vs.append(V), Fs.append(F + base), UVs.append(UV), FMs.append(FM)
        base += len(V)
    if not Vs:
        raise ValueError(f"{path}: no triangle meshes")
    F = np.vstack(Fs)
    FM = np.concatenate(FMs)
    if (FM == nmat).any():
        mats = mats + [{"name": "default"}]
    m = Mesh(np.vstack(Vs), np.full(len(F), 3), F.reshape(-1), uvs=np.vstack(UVs), smooth_angle=60.0,
             face_mat=FM, name=Path(path).stem)
    return m, mats


def gltf_to_scene(path, fps=None, scene_path=None):
    """Convert a glTF file into an Animora scene dict (hierarchy, meshes referenced as
    ``file#node``, cameras, lights, materials and keyframed TRS animation of the first clip)."""
    from .. import mathutil as mu
    fps = float(fps or 24.0)
    doc, buffers = _load(path)
    nodes = doc.get("nodes", [])
    rel = str(path) if scene_path is None else str(Path(path).resolve().relative_to(Path(scene_path).resolve().parent)) \
        if Path(path).resolve().is_relative_to(Path(scene_path).resolve().parent) else str(Path(path).resolve())
    names, used = [], set()
    for i, nd in enumerate(nodes):
        nm = nd.get("name") or f"node{i}"
        base, k = nm, 1
        while nm in used:
            k += 1
            nm = f"{base}_{k}"
        used.add(nm)
        names.append(nm)
    parent = {}
    for i, nd in enumerate(nodes):
        for c in nd.get("children", []):
            parent[c] = i
    # animation (first clip): node -> path -> (times, values, interp)
    anim = {}
    t_end = 0.0
    if doc.get("animations"):
        a = doc["animations"][0]
        for ch in a["channels"]:
            tg = ch["target"]
            if "node" not in tg or tg["path"] not in ("translation", "rotation", "scale", "weights"):
                continue
            smp = a["samplers"][ch["sampler"]]
            t = _accessor(doc, buffers, smp["input"]).astype(float).reshape(-1)
            v = _accessor(doc, buffers, smp["output"]).astype(float)
            interp = smp.get("interpolation", "LINEAR")
            if interp == "CUBICSPLINE":
                v = v.reshape(len(t), 3, -1)[:, 1]
            if tg["path"] == "weights":
                anim.setdefault(tg["node"], {})["weights"] = (t, v.reshape(len(t), -1), interp)
                t_end = max(t_end, float(t[-1]) if len(t) else 0)
                continue
            anim.setdefault(tg["node"], {})[tg["path"]] = (t, v.reshape(len(t), -1), interp)
            t_end = max(t_end, float(t[-1]) if len(t) else 0)
    mats = _gltf_materials(doc, buffers, path)
    out_nodes = []
    lights = doc.get("extensions", {}).get("KHR_lights_punctual", {}).get("lights", [])
    for i, nd in enumerate(nodes):
        L = _node_local(nd)
        t, R, s = mu.decompose(L)
        e = mu.matrix_to_euler(np.vstack([np.hstack([R, np.zeros((3, 1))]), [0, 0, 0, 1]]), "xyz")
        spec = {"name": names[i]}
        if i in parent:
            spec["parent"] = names[parent[i]]
        if "mesh" in nd:
            spec["type"] = "mesh"
            spec["file"] = f"{rel}#{nd.get('name') or names[i]}"
            prims = doc["meshes"][nd["mesh"]]["primitives"]
            slots = [p.get("material") for p in prims]
            if any(s_ is not None for s_ in slots):
                spec["materials"] = [{k: v for k, v in mats[s_].items() if k != "name"} if s_ is not None else None
                                     for s_ in sorted(set(x for x in slots if x is not None))]
        elif "camera" in nd:
            cam = doc["cameras"][nd["camera"]]
            spec["type"] = "camera"
            if cam.get("type") == "perspective":
                spec["fov"] = math.degrees(cam["perspective"]["yfov"])
        elif "KHR_lights_punctual" in nd.get("extensions", {}):
            Lg = lights[nd["extensions"]["KHR_lights_punctual"]["light"]]
            spec["type"] = "light"
            spec["light"] = {"directional": "sun", "spot": "spot"}.get(Lg.get("type"), "point")
            spec["color"] = Lg.get("color", [1, 1, 1])
            spec["intensity"] = Lg.get("intensity", 1.0)
            if Lg.get("type") == "spot":
                spec["cone_angle"] = math.degrees(2 * Lg.get("spot", {}).get("outerConeAngle", math.pi / 4))
        else:
            spec["type"] = "group"
        spec["translate"] = [round(float(x), 6) for x in t]
        spec["rotate"] = [round(float(x), 5) for x in e]
        spec["scale"] = [round(float(x), 6) for x in s]
        ch = anim.get(i, {})
        if "mesh" in nd:
            gm = doc["meshes"][nd["mesh"]]
            nt = max((len(p_.get("targets", [])) for p_ in gm["primitives"]), default=0)
            if nt:
                w0 = list(nd.get("weights", gm.get("weights", [0.0] * nt)))
                shapes = []
                for k in range(nt):
                    bs = {"name": (gm.get("extras", {}).get("targetNames") or [None] * nt)[k] or f"target{k}",
                          "file": f"{spec['file']}@{k}"}
                    if "weights" in ch:
                        tt, vv, interp = ch["weights"]
                        ip = "step" if interp == "STEP" else "linear"
                        bs["weight"] = {"keys": [[round(1 + float(x) * fps, 4), round(float(v[k]), 6), ip]
                                                 for x, v in zip(tt, vv)]}
                    else:
                        bs["weight"] = float(w0[k]) if k < len(w0) else 0.0
                    shapes.append(bs)
                spec["blend_shapes"] = shapes
        for path_, key in (("translation", "translate"), ("scale", "scale"), ("rotation", "rotate")):
            if path_ not in ch:
                continue
            tt, vv, interp = ch[path_]
            if path_ == "rotation":
                eul = np.array([mu.matrix_to_euler(mu.quat_to_matrix([q[3], q[0], q[1], q[2]]), "xyz") for q in vv])
                vv = np.degrees(np.unwrap(np.radians(eul), axis=0))
            ip = "step" if interp == "STEP" else "linear"
            spec[key] = {"keys": [[round(1 + float(x) * fps, 4), [round(float(c), 6) for c in v], ip]
                                  for x, v in zip(tt, vv)]}
        out_nodes.append(spec)
    last = max(1, int(math.ceil(1 + t_end * fps)))
    return {"comment": f"imported from {Path(path).name}",
            "settings": {"fps": fps, "frame_range": [1, last], "resolution": [1280, 720]},
            "environment": {"type": "sky"}, "nodes": out_nodes}
