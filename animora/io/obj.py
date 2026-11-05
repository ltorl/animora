"""Wavefront OBJ (+MTL) and PLY mesh import / export.

``read_mesh(path)`` dispatches on the extension and returns ``(Mesh, material_specs)`` where
``material_specs`` is a list of material spec dicts (one per ``face_mat`` slot) built from the
MTL file when present.
"""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from ..geometry import Mesh

_MESH_CACHE: dict = {}


# ======================================================================================
# OBJ
# ======================================================================================
def _parse_mtl(path: Path):
    mats, cur = {}, None
    if not path.exists():
        return mats
    for line in path.read_text(errors="replace").splitlines():
        parts = line.strip().split()
        if not parts or parts[0].startswith("#"):
            continue
        tag, args = parts[0], parts[1:]
        if tag == "newmtl":
            cur = {}
            mats[" ".join(args)] = cur
            continue
        if cur is None:
            continue
        try:
            if tag == "Kd":
                cur["base_color"] = [float(a) for a in args[:3]]
            elif tag == "Ke":
                e = [float(a) for a in args[:3]]
                if max(e) > 0:
                    cur["emission"] = e
                    cur["emission_strength"] = 1.0
            elif tag == "Ns":
                ns = float(args[0])
                cur["roughness"] = float(np.clip((2.0 / (ns + 2.0)) ** 0.25, 0.02, 1.0))
            elif tag == "Pr":
                cur["roughness"] = float(args[0])
            elif tag == "Pm":
                cur["metallic"] = float(args[0])
            elif tag == "Ni":
                cur["ior"] = float(args[0])
            elif tag in ("d",):
                cur["alpha"] = float(args[0])
            elif tag == "Tr":
                cur["alpha"] = 1.0 - float(args[0])
            elif tag == "map_Kd":
                cur["base_color_map"] = str((path.parent / args[-1]).resolve())
                cur["base_color"] = [1.0, 1.0, 1.0]
            elif tag in ("map_Bump", "map_bump", "bump"):
                cur["bump_map"] = str((path.parent / args[-1]).resolve())
            elif tag == "norm":
                cur["normal_map"] = str((path.parent / args[-1]).resolve())
            elif tag == "map_Pr":
                cur["roughness_map"] = str((path.parent / args[-1]).resolve())
            elif tag == "map_d":
                cur["alpha_map"] = str((path.parent / args[-1]).resolve())
        except (ValueError, IndexError):
            continue
    return mats


def read_obj(path):
    """Read an OBJ file. Returns ``(Mesh, [material spec, ...])``.

    Supports v / vt / vn / f (n-gons, negative indices), ``usemtl`` + ``mtllib`` and ``s`` groups
    (normals are taken from the file when every face provides them).
    """
    path = Path(path)
    V, VT, VN = [], [], []
    counts, vi, ti, ni, fmat = [], [], [], [], []
    mat_names, mat_index = [], {}
    cur_mat = 0
    mtl = {}
    all_have_uv, all_have_n = True, True
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            if not line or line[0] == "#":
                continue
            parts = line.split()
            if not parts:
                continue
            tag = parts[0]
            if tag == "v":
                V.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "vt":
                VT.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif tag == "vn":
                VN.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif tag == "f":
                corners = parts[1:]
                if len(corners) < 3:
                    continue
                counts.append(len(corners))
                fmat.append(cur_mat)
                for c in corners:
                    s = c.split("/")
                    a = int(s[0])
                    vi.append(a - 1 if a > 0 else len(V) + a)
                    if len(s) > 1 and s[1]:
                        b = int(s[1])
                        ti.append(b - 1 if b > 0 else len(VT) + b)
                    else:
                        ti.append(-1)
                        all_have_uv = False
                    if len(s) > 2 and s[2]:
                        n = int(s[2])
                        ni.append(n - 1 if n > 0 else len(VN) + n)
                    else:
                        ni.append(-1)
                        all_have_n = False
            elif tag == "usemtl":
                name = " ".join(parts[1:]) or "default"
                if name not in mat_index:
                    mat_index[name] = len(mat_names)
                    mat_names.append(name)
                cur_mat = mat_index[name]
            elif tag == "mtllib":
                mtl.update(_parse_mtl(path.parent / " ".join(parts[1:])))
    if not counts:
        raise ValueError(f"{path}: no faces found")
    uvs = np.asarray(VT, np.float64)[ti] if (all_have_uv and VT) else None
    normals = np.asarray(VN, np.float64)[ni] if (all_have_n and VN) else None
    mesh = Mesh(np.asarray(V, np.float64), counts, vi, uvs=uvs, normals=normals, face_mat=fmat, name=path.stem)
    specs = [dict(mtl.get(n, {}), name=n) for n in mat_names] or [{"name": "default"}]
    return mesh, specs


def write_obj(path, mesh: Mesh, material_names=None, write_normals=True):
    """Write a :class:`Mesh` as OBJ (with per-corner UVs / normals)."""
    path = Path(path)
    lines = [f"# Animora OBJ export: {mesh.name}", f"o {mesh.name}"]
    lines += [f"v {x:.6g} {y:.6g} {z:.6g}" for x, y, z in mesh.verts]
    has_uv = mesh.uvs is not None
    if has_uv:
        lines += [f"vt {u:.6g} {v:.6g}" for u, v in mesh.uvs]
    N = mesh.corner_normals() if write_normals else None
    if N is not None:
        lines += [f"vn {x:.5g} {y:.5g} {z:.5g}" for x, y, z in N]
    o = mesh.offsets
    cur = None
    for f in range(mesh.num_faces):
        if material_names is not None:
            m = int(mesh.face_mat[f])
            if m != cur:
                cur = m
                lines.append(f"usemtl {material_names[m] if m < len(material_names) else m}")
        cs = []
        for c in range(o[f], o[f + 1]):
            v = mesh.indices[c] + 1
            t = f"{c + 1}" if has_uv else ""
            n = f"{c + 1}" if N is not None else ""
            cs.append(f"{v}/{t}/{n}" if (t or n) else f"{v}")
        lines.append("f " + " ".join(cs))
    path.write_text("\n".join(lines) + "\n")
    return path


# ======================================================================================
# PLY
# ======================================================================================
_PLY_TYPES = {"char": "b", "int8": "b", "uchar": "B", "uint8": "B", "short": "h", "int16": "h", "ushort": "H",
              "uint16": "H", "int": "i", "int32": "i", "uint": "I", "uint32": "I", "float": "f", "float32": "f",
              "double": "d", "float64": "d"}


def read_ply(path):
    """Read an ASCII or binary (little / big endian) PLY. Returns ``(Mesh, [spec])``."""
    path = Path(path)
    data = path.read_bytes()
    end = data.find(b"end_header")
    if end < 0:
        raise ValueError(f"{path}: not a PLY file")
    header = data[:end].decode("ascii", "replace").splitlines()
    body_start = data.index(b"\n", end) + 1
    fmt = "ascii"
    elements = []
    for line in header:
        p = line.split()
        if not p:
            continue
        if p[0] == "format":
            fmt = p[1]
        elif p[0] == "element":
            elements.append({"name": p[1], "count": int(p[2]), "props": []})
        elif p[0] == "property":
            if p[1] == "list":
                elements[-1]["props"].append((p[4], "list", p[2], p[3]))
            else:
                elements[-1]["props"].append((p[2], p[1]))
    verts, faces, colors, uvs_v = None, [], None, None
    if fmt == "ascii":
        tokens = data[body_start:].split()
        pos = 0
        for el in elements:
            rows = []
            for _ in range(el["count"]):
                row = {}
                for pr in el["props"]:
                    if pr[1] == "list":
                        n = int(tokens[pos]); pos += 1
                        row[pr[0]] = [float(t) for t in tokens[pos:pos + n]]; pos += n
                    else:
                        row[pr[0]] = float(tokens[pos]); pos += 1
                rows.append(row)
            el["rows"] = rows
    else:
        endian = "<" if fmt == "binary_little_endian" else ">"
        pos = body_start
        for el in elements:
            rows = []
            simple = all(pr[1] != "list" for pr in el["props"])
            if simple:
                st = endian + "".join(_PLY_TYPES[pr[1]] for pr in el["props"])
                size = struct.calcsize(st)
                arr = np.frombuffer(data, dtype=np.dtype([(pr[0], endian + _PLY_TYPES[pr[1]]) for pr in el["props"]]),
                                    count=el["count"], offset=pos)
                pos += size * el["count"]
                el["array"] = arr
                continue
            for _ in range(el["count"]):
                row = {}
                for pr in el["props"]:
                    if pr[1] == "list":
                        ct = endian + _PLY_TYPES[pr[2]]
                        n = struct.unpack_from(ct, data, pos)[0]; pos += struct.calcsize(ct)
                        it = endian + _PLY_TYPES[pr[3]] * n
                        row[pr[0]] = list(struct.unpack_from(it, data, pos)); pos += struct.calcsize(it)
                    else:
                        t = endian + _PLY_TYPES[pr[1]]
                        row[pr[0]] = struct.unpack_from(t, data, pos)[0]; pos += struct.calcsize(t)
                rows.append(row)
            el["rows"] = rows
    for el in elements:
        if el["name"] == "vertex":
            if "array" in el:
                a = el["array"]
                verts = np.stack([a["x"], a["y"], a["z"]], axis=1).astype(np.float64)
                names = a.dtype.names
                if "red" in names:
                    colors = np.stack([a["red"], a["green"], a["blue"]], axis=1) / 255.0
                for un, vn in (("u", "v"), ("s", "t"), ("texture_u", "texture_v")):
                    if un in names:
                        uvs_v = np.stack([a[un], a[vn]], axis=1).astype(np.float64)
            else:
                rows = el["rows"]
                verts = np.array([[r["x"], r["y"], r["z"]] for r in rows], np.float64)
                if rows and "u" in rows[0]:
                    uvs_v = np.array([[r["u"], r["v"]] for r in rows], np.float64)
        elif el["name"] == "face":
            key = el["props"][0][0]
            faces = [r[key] for r in el["rows"]] if "rows" in el else []
    if verts is None:
        raise ValueError(f"{path}: no vertex element")
    counts = [len(f) for f in faces]
    idx = [int(i) for f in faces for i in f]
    uvs = uvs_v[np.asarray(idx, np.int64)] if uvs_v is not None and idx else None
    mesh = Mesh(verts, counts, idx, uvs=uvs, name=path.stem)
    spec = {"name": "default"}
    if colors is not None:
        spec["base_color"] = colors.mean(axis=0).tolist()
    return mesh, [spec]


def write_ply(path, mesh: Mesh, binary=True):
    """Write a :class:`Mesh` as PLY (binary little endian by default)."""
    path = Path(path)
    head = ["ply", "format " + ("binary_little_endian 1.0" if binary else "ascii 1.0"),
            "comment Animora export", f"element vertex {mesh.num_verts}",
            "property float x", "property float y", "property float z",
            f"element face {mesh.num_faces}", "property list uchar int vertex_indices", "end_header"]
    o = mesh.offsets
    if binary:
        out = bytearray(("\n".join(head) + "\n").encode())
        out += mesh.verts.astype("<f4").tobytes()
        for f in range(mesh.num_faces):
            ids = mesh.indices[o[f]:o[f + 1]]
            out += struct.pack("<B", len(ids)) + ids.astype("<i4").tobytes()
        path.write_bytes(bytes(out))
    else:
        lines = head + [f"{x:.6g} {y:.6g} {z:.6g}" for x, y, z in mesh.verts]
        for f in range(mesh.num_faces):
            ids = mesh.indices[o[f]:o[f + 1]]
            lines.append(f"{len(ids)} " + " ".join(str(int(i)) for i in ids))
        path.write_text("\n".join(lines) + "\n")
    return path


# ======================================================================================
def read_mesh(path, cache=True):
    """Read OBJ, PLY, glTF or GLB (cached by path + mtime). Returns ``(Mesh, [material spec, ...])``.

    ``model.glb#NodeName`` selects one node's mesh of a glTF file (in the node's local space)."""
    import re
    sel = None
    mt = re.match(r"^(.*\.(?:glb|gltf))#(.+)$", str(path), re.I)
    if mt:
        path, sel = mt.group(1), mt.group(2)
    p = Path(path).resolve()
    key = (str(p), p.stat().st_mtime, sel)
    if cache and key in _MESH_CACHE:
        m, s = _MESH_CACHE[key]
        return m.copy(), [dict(x) for x in s]
    ext = p.suffix.lower()
    if ext == ".obj":
        res = read_obj(p)
    elif ext == ".ply":
        res = read_ply(p)
    elif ext in (".glb", ".gltf"):
        from .gltf import read_gltf
        res = read_gltf(p, sel)
    else:
        raise ValueError(f"unsupported mesh format {ext!r} (supported: .obj, .ply, .glb, .gltf)")
    if cache:
        _MESH_CACHE[key] = res
    return res[0].copy(), [dict(x) for x in res[1]]


def write_mesh(path, mesh, **kw):
    ext = Path(path).suffix.lower()
    if ext == ".obj":
        return write_obj(path, mesh, **kw)
    if ext == ".ply":
        return write_ply(path, mesh, **kw)
    raise ValueError(f"unsupported mesh format {ext!r}")
