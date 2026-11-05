"""Scene description, node graph, per-frame evaluation and packing for the renderer.

A scene is a plain dictionary (usually stored as JSON) with the top-level keys::

    settings      render / output settings (see DEFAULT_SETTINGS)
    camera        name of the active camera node (or omitted: first camera)
    camera_switches  [[frame, "camName"], ...]  -- cut between cameras
    environment   world lighting (sky / hdri / gradient / studio / color / none)
    fog           volumetric fog {density, color, anisotropy, height_min, height_max}
    materials     {name: material spec}
    variables     {name: value}  -- constants usable in every expression
    nodes         list (or {name: spec} dict) of nodes; nodes may nest via "children"
    ik            list of IK handles
    include       list of other scene files whose materials / nodes are merged in

See DOCS.md for every node key. All animatable values accept constants, keyframes or
expressions (see :mod:`animora.params`).
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path

import numpy as np

from . import mathutil as mu
from .expressions import make_context
from .geometry import Mesh, make_primitive, merge, PRIMITIVES
from .materials import Material, TextureTable, material_from_spec, parse_color, kelvin_to_rgb
from .modifiers import apply_stack, TOPOLOGY_CHANGING
from .params import Param, TreeParam

TREE_PARAMS = ("profile", "path", "points2d", "shapes", "balls", "sections", "bounds")

DEFAULT_SETTINGS = {
    "resolution": [960, 540],
    "fps": 24,
    "frame_range": [1, 48],
    "frame_step": 1,
    "samples": 64,
    "min_samples": 0,
    "adaptive_threshold": 0.0,
    "max_bounces": 8,
    "clamp_indirect": 10.0,
    "caustics": False,
    "filter_width": 1.5,
    "integrator": "path",
    "ao_distance": 1.0,
    "wire_width": 1.0,
    "film_transparent": False,
    "seed": 1,
    "threads": 0,
    "device": "cpu",
    "motion_blur": False,
    "shutter": 0.5,
    "shutter_position": "center",
    "denoise": True,
    "denoise_iterations": 5,
    "denoise_strength": 1.0,
    "output": "render/frame_####.png",
    "video": None,
    "aovs": [],
    "tonemap": "agx",
    "exposure": 0.0,
    "bit_depth": 8,
    "preview_scale": 0.5,
    "preview_samples": 8,
    "unit_scale": 1.0,
}

NODE_TYPES = ("group", "locator", "mesh", "camera", "light", "joint", "curve", "particles", "cloth", "instance",
              "volume", "liquid", "hair")
TRANSFORM_KEYS = ("translate", "rotate", "scale", "pivot", "shear")
COMMON_KEYS = {"name", "type", "parent", "children", "translate", "position", "rotate", "rotation", "scale",
               "pivot", "rotate_order", "visible", "constraints", "look_at", "attributes", "comment", "description",
               "rigid_body", "jiggle", "shake", "tags", "inherit_transform"}

_AXES = {"x": [1, 0, 0], "+x": [1, 0, 0], "-x": [-1, 0, 0], "y": [0, 1, 0], "+y": [0, 1, 0], "-y": [0, -1, 0],
         "z": [0, 0, 1], "+z": [0, 0, 1], "-z": [0, 0, -1]}


def axis_vector(v, default="x"):
    if v is None:
        v = default
    if isinstance(v, str):
        if v.lower() not in _AXES:
            raise ValueError(f"bad axis {v!r}; use x, y, z, -x, -y, -z or a vector")
        return np.array(_AXES[v.lower()], np.float64)
    return mu.normalize(np.asarray(v, np.float64))


def file_for_frame(node, frame):
    """Substitute a run of ``#`` in a mesh node's ``file`` with the (padded) frame number:
    ``cache/sim_####.obj`` at frame 12 -> ``cache/sim_0012.obj``. ``file_offset`` shifts the frame;
    ``file_range: [first, last]`` clamps it (hold first / last file outside the range)."""
    import re
    f = node.spec["file"]
    base = os.path.basename(str(f))
    sel = ""
    mt = re.match(r"^(.*\.(?:glb|gltf))(#.+)$", str(f), re.I)
    if mt:
        f, sel = mt.group(1), mt.group(2)
        base = os.path.basename(f)
    run = re.search(r"#+", base)
    if not run:
        return str(f) + sel
    n = int(round(float(frame))) + int(node.spec.get("file_offset", 0))
    rng = node.spec.get("file_range")
    if rng:
        n = max(int(rng[0]), min(int(rng[1]), n))
    width = run.end() - run.start()
    d = os.path.dirname(str(f))
    return os.path.join(d, base[:run.start()] + str(n).zfill(width) + base[run.end():]) + sel


class SceneError(ValueError):
    pass


# ======================================================================================
# Node
# ======================================================================================
class Node:
    """One object in the scene graph."""

    def __init__(self, spec, scene=None):
        spec = dict(spec)
        self.scene = scene
        self.name = str(spec.get("name") or "")
        if not self.name:
            raise SceneError(f"node without a name: {spec}")
        self.type = spec.get("type", "mesh" if ("primitive" in spec or "file" in spec or "mesh" in spec)
                             else "group")
        if self.type == "locator":
            self.type = "group"
        if self.type not in NODE_TYPES:
            raise SceneError(f"node {self.name!r}: unknown type {self.type!r}; use one of {NODE_TYPES}")
        self.parent = spec.get("parent")
        self.translate = Param(spec.get("translate", spec.get("position", [0.0, 0.0, 0.0])))
        self.rotate = Param(spec.get("rotate", spec.get("rotation", [0.0, 0.0, 0.0])))
        sc = spec.get("scale", [1.0, 1.0, 1.0])
        if isinstance(sc, (int, float)):
            sc = [sc] * 3
        self.scale = Param(sc)
        self.pivot = Param(spec.get("pivot", [0.0, 0.0, 0.0]))
        self.rotate_order = spec.get("rotate_order", "xyz")
        if self.rotate_order not in mu.ROTATE_ORDERS:
            raise SceneError(f"node {self.name!r}: rotate_order must be one of {mu.ROTATE_ORDERS}")
        self.visible = Param(spec.get("visible", True))
        self.inherit_transform = bool(spec.get("inherit_transform", True))
        self.constraints = list(spec.get("constraints", []))
        if "look_at" in spec and spec["look_at"] is not None:
            la = spec["look_at"]
            c = la if isinstance(la, dict) else {"target": la}
            c = dict(c)
            c["type"] = "aim"
            self.constraints.append(c)
        self.attributes = {k: Param(v) for k, v in (spec.get("attributes") or {}).items()}
        bs = spec.get("blend_shapes") or []
        if isinstance(bs, dict):
            bs = [dict(v, name=k) for k, v in bs.items()]
        for i, b in enumerate(bs):
            self.attributes.setdefault(b.get("name", f"shape{i}"), Param(b.get("weight", 0.0)))
        self.tags = list(spec.get("tags", []))
        self.spec = spec
        # every other key becomes an animatable parameter where possible
        self.params: dict[str, Param] = {}
        for k, v in spec.items():
            if k in COMMON_KEYS or k in ("modifiers", "material", "materials", "skin", "blend_shapes", "mesh",
                                         "file", "primitive", "params", "points", "source", "emitter",
                                         "colliders", "pin", "fields", "layers", "smoke", "grid", "shape",
                                         "emitters", "guides", "groom"):
                continue
            try:
                self.params[k] = Param(v)
            except Exception:
                pass
        self.prim_params = {}
        if self.type == "mesh" and "primitive" in spec:
            prim = spec["primitive"]
            if prim not in PRIMITIVES:
                raise SceneError(f"node {self.name!r}: unknown primitive {prim!r}; available {sorted(PRIMITIVES)}")
            import inspect
            sig = inspect.signature(PRIMITIVES[prim]).parameters
            pp = dict(spec.get("params") or {})
            for k in spec:
                if k in sig and k not in pp and k not in COMMON_KEYS and k not in ("primitive", "params"):
                    pp[k] = spec[k]
            for k, v in pp.items():
                if k not in sig:
                    raise SceneError(f"node {self.name!r}: primitive {prim!r} has no parameter {k!r}; "
                                     f"parameters: {list(sig)}")
                self.prim_params[k] = (TreeParam(v) if k in TREE_PARAMS else Param(v))
        self.modifiers = []
        for m in spec.get("modifiers", []) or []:
            m = dict(m)
            if "type" not in m:
                raise SceneError(f"node {self.name!r}: modifier without 'type'")
            self.modifiers.append({k: (Param(v) if k not in ("type", "name", "enabled", "after_deform", "limits",
                                                              "offsets", "bounds", "axis", "direction", "mode",
                                                              "texture", "curve", "faces", "object")
                                       else v) for k, v in m.items()})

    # ------------------------------------------------------------------
    def p(self, key, default=None, frame=0.0, ctx=None):
        """Evaluate a type-specific parameter."""
        prm = self.params.get(key)
        if prm is None:
            return default
        return prm.evaluate(frame, ctx)

    def has(self, key):
        return key in self.params

    def local_matrix(self, frame, ctx):
        t = self.translate.evaluate(frame, ctx)
        r = self.rotate.evaluate(frame, ctx)
        s = self.scale.evaluate(frame, ctx)
        if np.ndim(s) == 0:
            s = [s] * 3
        pv = self.pivot.evaluate(frame, ctx)
        pv = None if not np.any(np.asarray(pv)) else pv
        return mu.compose(t, r, s, self.rotate_order, pv)

    def is_visible(self, frame, ctx):
        v = self.visible.evaluate(frame, ctx)
        return bool(float(np.asarray(v, float).ravel()[0]) > 0.5) if not isinstance(v, bool) else v

    def geometry_animated(self):
        if self.type in ("liquid", "hair"):
            return True
        if self.type == "curve" and any(isinstance(p, str) or isinstance(p, dict) for p in self.spec.get("points", [])):
            return True  # control points follow other nodes or are animated
        if "#" in os.path.basename(str(self.spec.get("file", "")).split(".gl")[0]):
            return True  # frame-numbered mesh sequence (geometry cache)
        if any(hasattr(v, "is_animated") and v.is_animated() for v in self.prim_params.values()):
            return True
        for m in self.modifiers:
            if m.get("type") in ("wave", "ripple", "noise_deform") and float(
                    m["speed"].evaluate(0) if isinstance(m.get("speed"), Param) else m.get("speed", 1.0)) != 0:
                return True
            if any(isinstance(v, Param) and v.is_animated() for v in m.values()):
                return True
        if any(m.get("type") == "boolean" for m in self.modifiers):
            return True  # depends on another node (results are cached by content in modeling.boolean)
        return "skin" in self.spec or "blend_shapes" in self.spec

    def key(self, attr, frame, value, interp=None):
        """Python API: set a keyframe on ``translate``/``rotate``/``scale``/``visible`` or a parameter."""
        target = getattr(self, attr, None) if attr in ("translate", "rotate", "scale", "pivot", "visible") else None
        if target is None:
            target = self.params.setdefault(attr, Param(value))
        target.key(frame, value, interp)
        self.spec[attr] = target.to_spec()
        return self

    def set(self, attr, value):
        if attr in ("translate", "rotate", "scale", "pivot", "visible"):
            getattr(self, attr).set(value)
        else:
            self.params[attr] = Param(value)
        self.spec[attr] = value
        return self

    def to_spec(self):
        d = dict(self.spec)
        for a in ("translate", "rotate", "scale", "visible"):
            d[a] = getattr(self, a).to_spec()
        d.pop("position", None)
        d.pop("rotation", None)
        d.pop("children", None)
        for k, prm in self.params.items():
            d[k] = prm.to_spec()
        return d

    def __repr__(self):
        return f"<Node {self.name!r} type={self.type}>"


# ======================================================================================
# Frame state (memoised evaluation of one instant)
# ======================================================================================
class FrameState:
    def __init__(self, scene, frame):
        self.scene = scene
        self.frame = float(frame)
        self.ctx = scene.context(frame)
        self._local = {}
        self._world = {}
        self._raw = {}
        self._visiting = set()
        self._ik_done = set()
        self._ik_active = set()
        self.ik_local = {}

    # FK local matrix (no IK)
    def fk_local(self, name):
        if name not in self._local:
            self._local[name] = self.scene.nodes[name].local_matrix(self.frame, self.ctx)
        return self._local[name]

    def local(self, name):
        h = self.scene._ik_by_joint.get(name)
        if h is not None:
            if h not in self._ik_done and h not in self._ik_active:
                self._solve_ik(h)
            if name in self.ik_local:
                return self.ik_local[name]
        return self.fk_local(name)

    def _solve_ik(self, handle_index):
        from .rig import solve_ik
        self._ik_active.add(handle_index)
        try:
            res = solve_ik(self, self.scene.ik[handle_index])
            self.ik_local.update(res)
            # invalidate cached worlds of the chain and everything below it
            if res:
                self._world.clear()
                self._raw.clear()
        finally:
            self._ik_active.discard(handle_index)
            self._ik_done.add(handle_index)

    def parent_world(self, name, dynamics=True):
        node = self.scene.nodes[name]
        if node.parent and node.inherit_transform:
            return self.world(node.parent, dynamics)
        return np.eye(4)

    def world(self, name, dynamics=True):
        cache = self._world if dynamics else self._raw
        m = cache.get(name)
        if m is not None:
            return m
        key = (name, dynamics)
        if key in self._visiting:
            raise SceneError(f"dependency cycle while evaluating {name!r} "
                             f"(involving {sorted(n for n, _ in self._visiting)})")
        self._visiting.add(key)
        try:
            node = self.scene.nodes[name]
            pw = self.parent_world(name, dynamics)
            W = pw @ self.local(name)
            if node.constraints:
                from .constraints import apply_constraints
                W = apply_constraints(self, node, W, pw)
            if "shake" in node.spec:
                W = _apply_shake(node, W, self.frame, self.scene.fps)
            if dynamics and self.scene.dynamics is not None:
                ov = self.scene.dynamics.world_override(name, self.frame, W)
                if ov is not None:
                    W = ov
        finally:
            self._visiting.discard(key)
        cache[name] = W
        return W

    def position(self, name):
        return self.world(name)[:3, 3].copy()


def _apply_shake(node, W, frame, fps):
    from .noise import wiggle
    sh = node.spec["shake"]
    if not isinstance(sh, dict):
        sh = {"amplitude": float(sh)}
    amp = float(sh.get("amplitude", 0.5))
    freq = float(sh.get("frequency", 1.5))
    seed = int(sh.get("seed", 0))
    tamp = float(sh.get("translate", 0.0))
    t = frame / fps
    rx = wiggle(t, freq, amp, 3, seed)
    ry = wiggle(t, freq, amp, 3, seed + 17)
    rz = wiggle(t, freq, amp * 0.3, 2, seed + 31)
    R = mu.euler_matrix([rx, ry, rz], "xyz")
    out = W.copy()
    out[:3, :3] = W[:3, :3] @ R[:3, :3]
    if tamp:
        out[:3, 3] += np.array([wiggle(t, freq, tamp, 2, seed + 41 + i) for i in range(3)])
    return out


# ======================================================================================
# Scene
# ======================================================================================
class Scene:
    """In-memory scene. Build from a dict (``Scene(data)``), a file (``load_scene(path)``) or
    programmatically with :meth:`add_mesh`, :meth:`add_camera`, :meth:`add_light`, ..."""

    def __init__(self, data=None, base_dir="."):
        data = copy.deepcopy(data or {})
        if data.get("retarget"):
            # HumanIK-style retargeting: bake source motion into the target joints first
            from .retarget import apply_retarget_to_data
            data = apply_retarget_to_data(data, base_dir)
        self.base_dir = str(base_dir)
        self.data = data
        self.settings = dict(DEFAULT_SETTINGS)
        self.settings.update(data.get("settings", {}))
        self.variables = dict(data.get("variables", {}))
        self.materials: dict[str, Material] = {}
        self.material_specs = {}
        for name, ms in (data.get("materials") or {}).items():
            self.add_material(name, ms)
        self.nodes: dict[str, Node] = {}
        self.order: list[str] = []
        nodes = data.get("nodes", [])
        if isinstance(nodes, dict):
            nodes = [dict(v, name=k) for k, v in nodes.items()]
        for spec in nodes:
            self._add_spec(spec, None)
        self.camera_name = data.get("camera")
        self.camera_switches = [(float(f), n) for f, n in data.get("camera_switches", [])]
        self.environment_spec = data.get("environment", {"type": "sky"})
        self.fog = {k: Param(v) for k, v in (data.get("fog") or {}).items()}
        self.ik = [dict(h) for h in data.get("ik", [])]
        self.dynamics = None
        self._states = {}
        self._mesh_cache = {}
        self._rest_cache = {}
        self._finalize()

    # ------------------------------------------------------------------ building
    def _add_spec(self, spec, parent):
        spec = dict(spec)
        children = spec.pop("children", []) or []
        if parent is not None and not spec.get("parent"):
            spec["parent"] = parent
        node = Node(spec, self)
        if node.name in self.nodes:
            raise SceneError(f"duplicate node name {node.name!r}")
        self.nodes[node.name] = node
        self.order.append(node.name)
        for c in children:
            self._add_spec(c, node.name)
        return node

    def _finalize(self):
        for n in self.nodes.values():
            if n.parent and n.parent not in self.nodes:
                raise SceneError(f"node {n.name!r}: parent {n.parent!r} does not exist")
        # parent cycles
        for n in self.nodes.values():
            seen, p = {n.name}, n.parent
            while p:
                if p in seen:
                    raise SceneError(f"parent cycle at node {n.name!r}")
                seen.add(p)
                p = self.nodes[p].parent
        self._ik_by_joint = {}
        from .rig import ik_chain
        for i, h in enumerate(self.ik):
            chain = ik_chain(self, h)
            h["_chain"] = chain
            for j in chain:
                self._ik_by_joint[j] = i
        self._children = {n: [] for n in self.nodes}
        for n in self.order:
            p = self.nodes[n].parent
            if p:
                self._children[p].append(n)
        if any(n.type in ("particles", "cloth") or "rigid_body" in n.spec or "jiggle" in n.spec
               for n in self.nodes.values()):
            from .dynamics import Dynamics
            self.dynamics = Dynamics(self)
        else:
            self.dynamics = None
        self._states.clear()
        self._mesh_cache.clear()
        self._rest_cache.clear()

    def invalidate(self):
        """Call after editing nodes programmatically."""
        self._finalize()

    def add_material(self, name, spec=None, **kw):
        if isinstance(spec, Material):
            mat = spec
            mat.name = name
        else:
            if isinstance(spec, str):
                spec = {"preset": spec} if spec in __import__("animora.materials", fromlist=["PRESETS"]).PRESETS \
                    else {"base_color": spec}
            spec = dict(spec or {})
            spec.update(kw)
            mat = Material(spec, name=name)
        self.materials[name] = mat
        self.material_specs[name] = mat.to_spec()
        return mat

    def add(self, type="group", name=None, parent=None, **spec):
        """Add a node of any type. Returns the :class:`Node`."""
        if name is None:
            i = 1
            while f"{type}{i}" in self.nodes:
                i += 1
            name = f"{type}{i}"
        spec.update(type=type, name=name)
        if parent is not None:
            spec["parent"] = parent if isinstance(parent, str) else parent.name
        node = self._add_spec(spec, None)
        self._finalize()
        return node

    def add_mesh(self, name=None, primitive="cube", material=None, **spec):
        if material is not None:
            spec["material"] = material.name if isinstance(material, Material) else material
        if "file" not in spec and "mesh" not in spec:
            spec["primitive"] = primitive
        return self.add("mesh", name, **spec)

    def add_camera(self, name="camera", **spec):
        node = self.add("camera", name, **spec)
        if self.camera_name is None:
            self.camera_name = node.name
        return node

    def add_light(self, name=None, light="point", **spec):
        spec["light"] = light
        return self.add("light", name, **spec)

    def add_joint(self, name=None, parent=None, **spec):
        return self.add("joint", name, parent=parent, **spec)

    def add_ik(self, start, end, target, pole=None, solver="auto", **kw):
        h = dict(start=start, end=end, target=target, solver=solver, **kw)
        if pole is not None:
            h["pole"] = pole
        self.ik.append(h)
        self._finalize()
        return h

    def remove(self, name):
        for c in list(self._children.get(name, [])):
            self.remove(c)
        self.nodes.pop(name, None)
        if name in self.order:
            self.order.remove(name)
        self._finalize()

    def node(self, name) -> Node:
        try:
            return self.nodes[name]
        except KeyError:
            raise SceneError(f"no node named {name!r}") from None

    __getitem__ = node

    def children(self, name):
        return list(self._children.get(name, []))

    # ------------------------------------------------------------------ serialisation
    def to_dict(self):
        d = {"settings": dict(self.settings)}
        if self.camera_name:
            d["camera"] = self.camera_name
        if self.camera_switches:
            d["camera_switches"] = [[f, n] for f, n in self.camera_switches]
        d["environment"] = self.environment_spec
        if self.fog:
            d["fog"] = {k: p.to_spec() for k, p in self.fog.items()}
        if self.variables:
            d["variables"] = self.variables
        d["materials"] = {k: m.to_spec() for k, m in self.materials.items()}
        d["nodes"] = [self.nodes[n].to_spec() for n in self.order]
        if self.ik:
            d["ik"] = [{k: v for k, v in h.items() if not k.startswith("_")} for h in self.ik]
        return _jsonable(d)

    def save(self, path):
        path = Path(path)
        d = self.to_dict()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            path.write_text(yaml.safe_dump(d, sort_keys=False))
        else:
            path.write_text(json.dumps(d, indent=2))
        return path

    # ------------------------------------------------------------------ time
    @property
    def fps(self):
        return float(self.settings.get("fps", 24))

    @property
    def frame_start(self):
        return float(self.settings.get("frame_range", [1, 48])[0])

    @property
    def frame_end(self):
        return float(self.settings.get("frame_range", [1, 48])[1])

    def frames(self):
        step = float(self.settings.get("frame_step", 1)) or 1
        out, f = [], self.frame_start
        while f <= self.frame_end + 1e-9:
            out.append(int(f) if float(f).is_integer() else f)
            f += step
        return out

    def context(self, frame):
        return make_context(frame, self.fps, self.frame_start, self.frame_end, self.variables, self.get_value)

    def state(self, frame) -> FrameState:
        key = round(float(frame), 6)
        st = self._states.get(key)
        if st is None:
            if len(self._states) > 16:
                self._states.clear()
            st = FrameState(self, key)
            self._states[key] = st
        return st

    def world_matrix(self, name, frame, dynamics=True):
        return self.state(frame).world(name, dynamics)

    # ------------------------------------------------------------------ attribute access
    def get_value(self, path, frame):
        """Resolve ``"node.attr[.component]"`` at ``frame`` (used by ``get()`` in expressions).

        attr: translate, rotate, scale, world_position (wpos), world_rotation, visible, any node
        parameter (e.g. focal_length, intensity) or custom attribute. component: x/y/z, r/g/b or index.
        """
        parts = str(path).split(".")
        if parts[0] not in self.nodes:
            if parts[0] in self.variables and len(parts) == 1:
                return self.variables[parts[0]]
            raise SceneError(f"get({path!r}): no node {parts[0]!r}")
        node = self.nodes[parts[0]]
        attr = parts[1] if len(parts) > 1 else "world_position"
        comp = parts[2] if len(parts) > 2 else None
        st = self.state(frame)
        if attr in ("translate", "rotate", "scale", "pivot"):
            v = getattr(node, attr).evaluate(frame, st.ctx)
        elif attr in ("world_position", "wpos", "worldPosition"):
            v = st.world(node.name)[:3, 3]
        elif attr in ("world_rotation", "wrot"):
            v = mu.matrix_to_euler(st.world(node.name), "xyz")
        elif attr in ("world_scale",):
            v = np.linalg.norm(st.world(node.name)[:3, :3], axis=0)
        elif attr == "visible":
            v = 1.0 if node.is_visible(frame, st.ctx) else 0.0
        elif attr in node.attributes:
            v = node.attributes[attr].evaluate(frame, st.ctx)
        elif attr in node.params:
            v = node.params[attr].evaluate(frame, st.ctx)
        else:
            raise SceneError(f"get({path!r}): node {node.name!r} has no attribute {attr!r}")
        if comp is not None:
            idx = {"x": 0, "y": 1, "z": 2, "r": 0, "g": 1, "b": 2, "w": 3}.get(comp)
            if idx is None:
                idx = int(comp)
            v = np.asarray(v, float).ravel()[idx]
        if isinstance(v, np.ndarray):
            return v.copy()
        return v

    # ------------------------------------------------------------------ cameras
    def cameras(self):
        return [n for n in self.order if self.nodes[n].type == "camera"]

    def active_camera(self, frame):
        name = self.camera_name
        for f, n in sorted(self.camera_switches):
            if frame >= f:
                name = n
        if name is None:
            cams = self.cameras()
            if not cams:
                return None
            name = cams[0]
        if name not in self.nodes or self.nodes[name].type != "camera":
            raise SceneError(f"active camera {name!r} is not a camera node")
        return name

    # ------------------------------------------------------------------ geometry
    def _base_mesh(self, node, frame, ctx):
        """Object-space mesh before deformation (primitive / file / inline / curve)."""
        spec = node.spec
        if node.type == "liquid":
            from .liquid import liquid_mesh
            return liquid_mesh(self, node.name, frame, getattr(self, "_preview_flag", False))
        if node.type == "hair":
            from .hair import hair_mesh
            return hair_mesh(self, node.name, frame, getattr(self, "_preview_flag", False))
        if node.type == "mesh":
            if "primitive" in spec:
                kw = {k: (v.evaluate(frame, ctx) if hasattr(v, "evaluate") else v) for k, v in node.prim_params.items()}
                for k, v in list(kw.items()):
                    if isinstance(v, np.ndarray) and v.ndim == 0:
                        kw[k] = float(v)
                    if k in ("segments", "rings", "subdivisions", "major_segments", "minor_segments",
                             "height_segments", "resolution", "octaves", "seed") and not isinstance(v, (list, tuple)):
                        kw[k] = int(round(float(np.asarray(v).ravel()[0])))
                m = make_primitive(spec["primitive"], **kw)
            elif "file" in spec:
                from .io.obj import read_mesh
                m, mats = read_mesh(self.resolve_path(file_for_frame(node, frame)))
                node._file_materials = mats
            elif "mesh" in spec:
                d = spec["mesh"]
                faces = d["faces"]
                counts = [len(f) for f in faces]
                idx = [i for f in faces for i in f]
                uvs = None
                if "uvs" in d:
                    uv = np.asarray(d["uvs"], float)
                    uvs = uv[np.asarray(d["uv_indices"], int).ravel()] if "uv_indices" in d else uv
                m = Mesh(d["verts"], counts, idx, uvs=uvs, smooth_angle=d.get("smooth_angle", 30.0),
                         face_mat=d.get("face_materials"), name=node.name)
            else:
                raise SceneError(f"mesh node {node.name!r} needs 'primitive', 'file' or 'mesh'")
            if "smooth_angle" in node.params:
                m.smooth_angle = float(node.p("smooth_angle", 30.0, frame, ctx))
            return m
        if node.type == "curve":
            pts = self.curve_points(node.name, frame, world=False)
            r = float(node.p("radius", 0.02, frame, ctx))
            if r <= 0:
                return None
            return _tube_mesh(pts, r, int(node.p("segments", 8, frame, ctx)), bool(node.p("closed", False)))
        return None

    def curve_points(self, name, frame, world=True, samples=None):
        """Polyline of a ``curve`` node (Catmull-Rom smoothed when ``smooth`` is true)."""
        node = self.nodes[name]
        st = self.state(frame)
        if "points" not in node.spec:
            raise SceneError(f"curve {name!r} has no 'points'")
        pts_spec = node.spec["points"]
        pts = []
        for p in pts_spec:
            if isinstance(p, str) and p in self.nodes:
                pts.append(st.world(p)[:3, 3])  # world space control point (node reference)
            else:
                pts.append(np.asarray(Param(p).evaluate(frame, st.ctx), float))
        pts = np.asarray(pts, float).reshape(-1, 3)
        closed = bool(node.p("closed", False, frame, st.ctx))
        if node.p("smooth", True, frame, st.ctx) and len(pts) > 2:
            pts = catmull_rom(pts, int(samples or node.p("samples", 16, frame, st.ctx)), closed)
        elif closed:
            pts = np.vstack([pts, pts[:1]])
        if world:
            refs = any(isinstance(p, str) for p in pts_spec)
            if not refs:
                pts = mu.transform_points(st.world(name), pts)
        return pts

    def object_mesh(self, name, frame, preview=False):
        """Mesh in object space after modifiers / blend shapes (no skinning). Returns Mesh or None."""
        node = self.nodes[name]
        st = self.state(frame)
        animated = node.geometry_animated()
        ckey = (name, preview)
        if not animated and ckey in self._mesh_cache:
            return self._mesh_cache[ckey]
        if node.type == "instance":
            src = node.spec.get("source")
            if src not in self.nodes:
                raise SceneError(f"instance {name!r}: source {src!r} not found")
            return self.object_mesh(src, frame, preview)
        m = self._base_mesh(node, frame, st.ctx)
        if m is None:
            return None
        has_skin = "skin" in node.spec
        mods = node.modifiers
        if any(md.get("type") == "boolean" for md in mods):
            mods = [self._resolve_boolean(node, md, frame) if md.get("type") == "boolean" else md for md in mods]
        m = apply_stack(m, mods, frame, st.ctx, self.fps, preview, stage="pre" if has_skin else None)
        if "blend_shapes" in node.spec:
            from .rig import apply_blend_shapes
            m = apply_blend_shapes(self, node, m, frame, st.ctx)
        if not animated:
            self._mesh_cache[ckey] = m
        return m

    def _resolve_boolean(self, node, md, frame):
        """Inject the other operand of a ``boolean`` modifier (in this node's object space)."""
        other = md.get("object")
        other = other.evaluate(frame) if isinstance(other, Param) else other
        if other not in self.nodes:
            raise SceneError(f"node {node.name!r}: boolean modifier needs 'object' = an existing mesh node "
                             f"(got {other!r})")
        if other == node.name:
            raise SceneError(f"node {node.name!r}: boolean modifier cannot reference itself")
        om = self.world_mesh(other, frame)
        md = dict(md)
        md.pop("object", None)
        if om is not None:
            om = om.transformed(np.linalg.inv(self.world_matrix(node.name, frame)))
        md["_other"] = om
        return md

    def world_mesh(self, name, frame, preview=False):
        """Final world-space mesh of a node at ``frame`` (after skinning / simulation)."""
        node = self.nodes[name]
        st = self.state(frame)
        if node.type == "cloth" and self.dynamics is not None:
            base = self.cloth_rest_mesh(name, preview)
            verts = self.dynamics.cloth_verts(name, frame)
            topo = self.dynamics.cloth_topology(name, frame)
            if topo is not None and len(topo[0]) == len(base.indices):
                idx, vsrc = topo
                n = int(idx.max()) + 1 if len(idx) else 0
                if len(verts) < n:  # (baked cache shorter than topology) pad from parents
                    verts = self.dynamics._cloth(name)._extend(verts, n)
                m = Mesh(verts[:n], base.counts, idx, base.uvs,
                         None, base.smooth_angle, base.face_mat, base.name)
                m.vsrc = vsrc[:len(m.verts)]
                return m
            m = base.with_verts(verts)
            return m
        m = self.object_mesh(name, frame, preview)
        if m is None:
            return None
        if "skin" in node.spec:
            from .rig import skin_mesh
            m = skin_mesh(self, node, m, frame)  # world space
            post = [mod for mod in node.modifiers if mod.get("after_deform",
                                                               mod["type"] in ("subdivide", "subdiv", "smooth_mesh"))]
            if post:
                m = apply_stack(m, node.modifiers, frame, st.ctx, self.fps, preview, stage="post")
            return m
        return m.transformed(st.world(name))

    def cloth_rest_mesh(self, name, preview=False):
        key = ("cloth", name, preview)
        if key not in self._rest_cache:
            node = self.nodes[name]
            st = self.state(self.frame_start)
            if "primitive" in node.spec or "file" in node.spec or "mesh" in node.spec:
                fake = copy.copy(node)
                fake.type = "mesh"
                m = self._base_mesh(fake, self.frame_start, st.ctx)
            else:
                m = make_primitive("plane", size=float(node.p("size", 2.0)),
                                   subdivisions=int(node.p("resolution", 24)))
            m = apply_stack(m, [md for md in node.modifiers if md["type"] not in ("subdivide", "subdiv")],
                            self.frame_start, st.ctx, self.fps, preview)
            self._rest_cache[key] = m
        return self._rest_cache[key]

    def rest_mesh(self, name, preview=False):
        """Mesh used for PREF (object-space rest coordinates of procedural textures)."""
        key = ("rest", name, preview)
        if key not in self._rest_cache:
            node = self.nodes[name]
            if node.type == "cloth":
                self._rest_cache[key] = self.cloth_rest_mesh(name, preview)
            elif node.type in ("liquid", "hair"):
                self._rest_cache[key] = None
            else:
                self._rest_cache[key] = self.object_mesh(name, self.frame_start, preview)
        return self._rest_cache[key]

    # ------------------------------------------------------------------ files
    def resolve_path(self, p):
        p = os.path.expanduser(str(p))
        if os.path.isabs(p):
            return p
        return os.path.normpath(os.path.join(self.base_dir, p))

    # ------------------------------------------------------------------ info
    def summary(self):
        counts = {}
        for n in self.nodes.values():
            counts[n.type] = counts.get(n.type, 0) + 1
        return {
            "nodes": len(self.nodes), "types": counts, "materials": len(self.materials),
            "frame_range": [self.frame_start, self.frame_end], "fps": self.fps,
            "resolution": self.settings["resolution"], "camera": self.camera_name or (self.cameras() or [None])[0],
            "ik_handles": len(self.ik), "environment": (self.environment_spec or {}).get("type", "sky")
            if isinstance(self.environment_spec, dict) else self.environment_spec,
        }

    def tree(self):
        lines = []

        def rec(n, depth):
            node = self.nodes[n]
            extra = ""
            if node.type == "mesh":
                extra = node.spec.get("primitive") or os.path.basename(str(node.spec.get("file", "inline")))
            elif node.type == "light":
                extra = node.spec.get("light", "point")
            lines.append("  " * depth + f"{n} [{node.type}{': ' + extra if extra else ''}]")
            for c in self._children[n]:
                rec(c, depth + 1)

        for n in self.order:
            if not self.nodes[n].parent:
                rec(n, 0)
        return "\n".join(lines)

    def validate(self, frames=None):
        """Evaluate the scene at a few frames and return a list of problems (empty == OK)."""
        problems = []
        frames = frames or sorted({self.frame_start, (self.frame_start + self.frame_end) / 2, self.frame_end})
        if not self.cameras():
            problems.append("no camera node (a default camera will be generated)")
        for n in self.nodes.values():
            if n.type in ("mesh", "instance") or n.type == "curve":
                mats = n.spec.get("material")
                if isinstance(mats, str) and mats not in self.materials and not _is_inline_material(mats):
                    problems.append(f"node {n.name!r}: material {mats!r} is not defined")
        for f in frames:
            try:
                build_packed(self, f, preview=True, motion_blur=False)
            except Exception as e:  # noqa: BLE001
                problems.append(f"frame {f}: {type(e).__name__}: {e}")
                break
        return problems


def _is_inline_material(s):
    from .materials import PRESETS, is_color_string
    return s in PRESETS or is_color_string(s)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items() if not str(k).startswith("_")}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, Param):
        return o.to_spec()
    return o


def catmull_rom(pts, samples=16, closed=False):
    pts = np.asarray(pts, float)
    if closed:
        P = np.vstack([pts[-1:], pts, pts[:2]])
        nseg = len(pts)
    else:
        P = np.vstack([2 * pts[0] - pts[1], pts, 2 * pts[-1] - pts[-2]])
        nseg = len(pts) - 1
    out = []
    t = np.linspace(0, 1, samples, endpoint=False)[:, None]
    for i in range(nseg):
        p0, p1, p2, p3 = P[i], P[i + 1], P[i + 2], P[i + 3]
        out.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    out.append((P[nseg + 1])[None])
    return np.vstack(out)


def _tube_mesh(pts, radius, segments, closed):
    from .geometry import tube
    return tube(pts, radius=radius, segments=max(3, segments), closed=False, caps=not closed)


# ======================================================================================
# loading
# ======================================================================================
def _merge_include(data, base_dir):
    inc = data.pop("include", None) or data.pop("includes", None)
    if not inc:
        return data
    if isinstance(inc, str):
        inc = [inc]
    merged_nodes, merged_mats = [], {}
    for p in inc:
        sub = load_scene_data(os.path.join(base_dir, p))
        merged_mats.update(sub.get("materials", {}))
        n = sub.get("nodes", [])
        merged_nodes += [dict(v, name=k) for k, v in n.items()] if isinstance(n, dict) else list(n)
        for k in ("environment", "fog", "settings", "variables"):
            if k in sub and k not in data:
                data[k] = sub[k]
        data.setdefault("ik", [])
        data["ik"] = list(sub.get("ik", [])) + list(data["ik"])
    merged_mats.update(data.get("materials", {}))
    data["materials"] = merged_mats
    own = data.get("nodes", [])
    own = [dict(v, name=k) for k, v in own.items()] if isinstance(own, dict) else list(own)
    data["nodes"] = merged_nodes + own
    return data


def load_scene_data(path):
    """Load a scene file (.json, .yaml/.yml or .py) and return the raw dict."""
    path = Path(path)
    if not path.exists():
        raise SceneError(f"scene file not found: {path}")
    ext = path.suffix.lower()
    if ext == ".json":
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise SceneError(f"{path}: invalid JSON at line {e.lineno} col {e.colno}: {e.msg}") from None
    elif ext in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise SceneError("YAML scene files need PyYAML (pip install pyyaml); use .json instead") from None
        data = yaml.safe_load(path.read_text())
    elif ext == ".py":
        sc = _exec_py_scene(path)
        data = sc.to_dict() if isinstance(sc, Scene) else sc
    else:
        raise SceneError(f"unsupported scene format {ext!r} (use .json, .yaml or .py)")
    if not isinstance(data, dict):
        raise SceneError(f"{path}: top level must be an object")
    return _merge_include(data, str(path.parent))


def _exec_py_scene(path):
    import animora
    ns = {"__file__": str(path), "__name__": "__animora_scene__", "animora": animora, "an": animora}
    code = compile(path.read_text(), str(path), "exec")
    old = os.getcwd()
    try:
        os.chdir(path.parent)
        exec(code, ns)
    finally:
        os.chdir(old)
    if "build" in ns and callable(ns["build"]):
        return ns["build"]()
    if "scene" in ns:
        return ns["scene"]
    raise SceneError(f"{path}: python scene must define `scene` or `build()`")


def load_scene(path, overrides=None) -> Scene:
    """Load a scene file and return a :class:`Scene`. ``overrides`` updates ``settings``."""
    path = Path(path)
    data = load_scene_data(path)
    if overrides:
        data.setdefault("settings", {}).update(overrides)
    return Scene(data, base_dir=str(path.parent.resolve()))


# ======================================================================================
# conversion to native renderer structures
# ======================================================================================
def camera_params(scene, name, frame, aspect):
    """Evaluate camera lens parameters. Returns dict (fov_y radians, aperture radius, ...)."""
    node = scene.nodes[name]
    st = scene.state(frame)
    ctx = st.ctx
    g = lambda k, d: node.p(k, d, frame, ctx)
    sensor_w = float(g("sensor_width", 36.0))
    sensor_h = float(g("sensor_height", sensor_w / aspect))
    fit = node.spec.get("sensor_fit", "horizontal")
    if node.has("fov"):  # vertical field of view in degrees
        fov_y = math.radians(float(g("fov", 40.0)))
        focal = sensor_h / 2 / math.tan(fov_y / 2) if fit != "horizontal" else sensor_w / aspect / 2 / math.tan(fov_y / 2)
    elif node.has("fov_horizontal"):
        fh = math.radians(float(g("fov_horizontal", 60.0)))
        fov_y = 2 * math.atan(math.tan(fh / 2) / aspect)
        focal = sensor_w / 2 / math.tan(fh / 2)
    else:
        focal = float(g("focal_length", 50.0))
        if fit == "vertical":
            fov_y = 2 * math.atan(sensor_h / 2 / focal)
        else:
            fov_y = 2 * math.atan(sensor_w / aspect / 2 / focal)
    W = st.world(name)
    pos = W[:3, 3]
    fwd = mu.normalize(-W[:3, 2])
    # focus
    if "focus_target" in node.spec and node.spec["focus_target"]:
        tgt = node.spec["focus_target"]
        tp = st.world(tgt)[:3, 3] if isinstance(tgt, str) else np.asarray(Param(tgt).evaluate(frame, ctx), float)
        fd = max(1e-3, float(np.dot(tp - pos, fwd)))
    else:
        fd = float(g("focus_distance", 10.0))
    aperture = 0.0
    dof = g("dof", None)
    if node.has("aperture_radius"):
        aperture = float(g("aperture_radius", 0.0))
    elif node.has("fstop") and (dof is None or bool(dof)):
        N = max(0.1, float(g("fstop", 5.6)))
        aperture = focal / 1000.0 / (2 * N) / float(scene.settings.get("unit_scale", 1.0))
        aperture *= float(g("dof_scale", 1.0))
    if dof is not None and not bool(dof):
        aperture = 0.0
    return {
        "fov_y": fov_y, "focal_length": focal, "aperture": aperture, "focus_distance": fd,
        "projection": node.spec.get("projection", "perspective"),
        "ortho_height": float(g("ortho_height", g("ortho_width", 10.0) / aspect if node.has("ortho_width") else 10.0)),
        "blades": int(g("blades", 0)), "blade_rotation": math.radians(float(g("blade_rotation", 0.0))),
        "shift_x": float(g("shift_x", 0.0)), "shift_y": float(g("shift_y", 0.0)),
        "near_clip": float(g("near_clip", 0.0)), "matrix": W,
    }


def _camera_struct(scene, name, f0, f1, aspect):
    from .render import native as nat
    cam = nat.AnCamera()
    a = camera_params(scene, name, f0, aspect)
    b = camera_params(scene, name, f1, aspect) if f1 != f0 else a
    for which, cp in (("0", a), ("1", b)):
        W = cp["matrix"]
        nat.set_vec(getattr(cam, "pos" + which), W[:3, 3])
        nat.set_vec(getattr(cam, "fwd" + which), mu.normalize(-W[:3, 2]))
        nat.set_vec(getattr(cam, "up" + which), mu.normalize(W[:3, 1]))
    cam.fov_y = a["fov_y"]
    cam.projection = nat.PROJECTIONS.get(a["projection"], 0)
    cam.ortho_height = a["ortho_height"]
    cam.aperture_radius = a["aperture"]
    cam.focus_distance = a["focus_distance"]
    cam.blades = a["blades"]
    cam.blade_rotation = a["blade_rotation"]
    cam.shift_x = a["shift_x"]
    cam.shift_y = a["shift_y"]
    cam.near_clip = a["near_clip"]
    return cam


def default_camera(scene, frame, aspect):
    """Auto-framing camera used when the scene has none: looks at the bounding box."""
    from .render import native as nat
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for n in scene.order:
        node = scene.nodes[n]
        if node.type in ("mesh", "instance", "cloth", "liquid", "hair"):
            try:
                m = scene.world_mesh(n, frame, preview=True)
            except Exception:  # noqa: BLE001
                continue
            if m is not None and m.num_verts:
                a, b = m.bounds()
                lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    if not np.all(np.isfinite(lo)):
        lo, hi = -np.ones(3), np.ones(3)
    c = (lo + hi) / 2
    r = max(1e-3, np.linalg.norm(hi - lo) / 2)
    fov = math.radians(35)
    d = r / math.sin(fov / 2) * 1.05
    eye = c + mu.normalize(np.array([0.6, 0.45, 1.0])) * d
    W = mu.look_at_matrix(eye, c)
    cam = nat.AnCamera()
    for w in ("0", "1"):
        nat.set_vec(getattr(cam, "pos" + w), W[:3, 3])
        nat.set_vec(getattr(cam, "fwd" + w), -W[:3, 2])
        nat.set_vec(getattr(cam, "up" + w), W[:3, 1])
    cam.fov_y = fov
    cam.focus_distance = d
    return cam


def light_color(node, frame, ctx):
    col = parse_color(node.p("color", [1.0, 1.0, 1.0], frame, ctx))
    if node.has("temperature"):
        col = col * parse_color(f'{float(node.p("temperature", 6500, frame, ctx))}k')
    return np.asarray(col, float)


def light_structs(scene, name, frame, f1=None):
    """Convert a light node to native lights (and, for area lights, emissive geometry).

    Returns ``(lights, area)`` where area is ``None`` or ``(Mesh_world, AnMaterial)``.
    """
    from .render import native as nat
    node = scene.nodes[name]
    st = scene.state(frame)
    ctx = st.ctx
    g = lambda k, d: node.p(k, d, frame, ctx)
    kind = node.spec.get("light", "point")
    W = st.world(name)
    pos = W[:3, 3]
    direction = mu.normalize(-W[:3, 2])
    col = light_color(node, frame, ctx)
    mult = float(g("multiplier", 1.0)) * 2.0 ** float(g("exposure", 0.0))
    if kind in ("point", "sphere", "omni", "spot"):
        L = nat.AnLight()
        L.type = 0 if kind != "spot" else 1
        if node.has("intensity"):
            I = float(g("intensity", 10.0))
        else:
            I = float(g("power", 100.0)) / (4 * math.pi)
        L.intensity = I * mult
        nat.set_vec(L.position, pos)
        nat.set_vec(L.direction, direction)
        nat.set_vec(L.color, col)
        L.radius = float(g("radius", 0.1 if kind == "sphere" else 0.0))
        if kind == "spot":
            cone = float(g("cone_angle", 40.0))
            L.angle = math.radians(cone * 0.5)
            pen = float(g("penumbra", g("blend", 0.15)))
            L.blend = float(np.clip(pen, 0.0, 1.0))
        L.visible = 1 if bool(g("camera_visible", False)) else 0
        L.cast_shadows = 1 if bool(g("cast_shadows", True)) else 0
        return [L], None
    if kind in ("sun", "distant", "directional"):
        L = nat.AnLight()
        L.type = 2
        L.intensity = float(g("intensity", g("strength", 3.0))) * mult
        nat.set_vec(L.direction, direction)
        nat.set_vec(L.color, col)
        L.angle = math.radians(float(g("angle", 0.53)) * 0.5)
        L.visible = 1 if bool(g("camera_visible", True)) else 0
        L.cast_shadows = 1 if bool(g("cast_shadows", True)) else 0
        return [L], None
    if kind in ("area", "rect", "quad", "disk", "disc"):
        size = g("size", [1.0, 1.0])
        size = np.broadcast_to(np.asarray(size, float).ravel(), (2,)) if np.size(size) <= 2 else np.asarray(size)[:2]
        sw, sh = float(size[0]), float(size[1])
        if kind in ("disk", "disc"):
            m = make_primitive("disc", radius=sw / 2, segments=32)
            m = m.transformed(mu.rot_x(90))   # disc faces +Y -> rotate to face -Z
            area = math.pi * (sw / 2) ** 2
        else:
            m = make_primitive("plane", width=sw, depth=sh)
            m = m.transformed(mu.rot_x(90))   # plane faces +Y -> faces -Z after +90 about X
            area = sw * sh
        # make sure it emits along local -Z
        fn = m.face_normals()
        if fn[0][2] > 0:
            m = m.flipped()
        if node.has("intensity"):
            radiance = float(g("intensity", 10.0))
        else:
            radiance = float(g("power", 100.0)) / (math.pi * max(area, 1e-8))
        mat = nat.default_material()
        nat.set_vec(mat.base_color, (0, 0, 0))
        mat.specular = 0.0
        mat.roughness = 1.0
        nat.set_vec(mat.emission, col)
        mat.emission_strength = radiance * mult
        mat.emission_one_sided = 0 if bool(g("two_sided", False)) else 1
        mat.flags = (0 if bool(g("camera_visible", False)) else 1) | (0 if bool(g("cast_shadows", False)) else 2)
        return [], (m.transformed(W), mat)
    raise SceneError(f"light {name!r}: unknown light type {kind!r} (point, sphere, spot, sun, area, disk)")


# ======================================================================================
# packing
# ======================================================================================
def shutter_frames(scene, frame, motion_blur=None):
    mb = scene.settings.get("motion_blur", False) if motion_blur is None else motion_blur
    if not mb:
        return float(frame), float(frame)
    s = float(scene.settings.get("shutter", 0.5))
    pos = scene.settings.get("shutter_position", "center")
    if pos == "start":
        return float(frame), float(frame) + s
    if pos == "end":
        return float(frame) - s, float(frame)
    return float(frame) - s / 2, float(frame) + s / 2


class _MatTable:
    def __init__(self, scene, frame, textable, preview):
        self.scene, self.frame, self.textable, self.preview = scene, frame, textable, preview
        self.ctx = scene.state(frame).ctx
        self.index = {}
        self.list = []
        self._inline = {}

    def get(self, ref, flags=0, owner=""):
        if isinstance(ref, Material):
            mat, key = ref, ("obj", id(ref), flags)
        elif ref is None:
            key = ("__default__", flags)
            mat = None
        elif isinstance(ref, str) and ref in self.scene.materials:
            mat, key = self.scene.materials[ref], (ref, flags)
        elif isinstance(ref, (str, dict)):
            k = json.dumps(ref, sort_keys=True, default=str)
            if k not in self._inline:
                self._inline[k] = material_from_spec(ref, name=f"{owner}_inline")
            mat, key = self._inline[k], ("inline", k, flags)
        else:
            raise SceneError(f"bad material reference {ref!r}")
        if key in self.index:
            return self.index[key]
        from .render import native as nat
        if mat is None:
            am = nat.default_material()
        else:
            am = mat.to_native(self.textable, self.frame, self.ctx, self.scene.base_dir)
        am.flags |= flags
        self.index[key] = len(self.list)
        self.list.append(am)
        return self.index[key]

    def add_raw(self, am):
        self.list.append(am)
        return len(self.list) - 1


def _node_visible(scene, name, frame):
    st = scene.state(frame)
    n = name
    while n:
        node = scene.nodes[n]
        if not node.is_visible(frame, st.ctx):
            return False
        n = node.parent
    return True


def _material_slots(scene, node, mesh):
    """Return list of material references for face_mat slots."""
    if "materials" in node.spec:
        return list(node.spec["materials"])
    if "material" in node.spec:
        return [node.spec["material"]]
    if node.type == "instance" and node.spec.get("source") in scene.nodes:
        src = scene.nodes[node.spec["source"]]
        return _material_slots(scene, src, mesh)
    fm = getattr(node, "_file_materials", None)
    if fm:
        return [{k: v for k, v in s.items() if k != "name"} for s in fm]
    return [None]


def _tris_with_pref(mesh, rest):
    tri = mesh.to_triangles()
    vsrc = getattr(mesh, "vsrc", None)
    if rest is not None and rest.num_verts == mesh.num_verts and rest.num_faces == mesh.num_faces:
        tri["PREF"] = rest.verts[tri["vidx"]]
    elif (rest is not None and vsrc is not None and len(vsrc) == mesh.num_verts
          and rest.num_faces == mesh.num_faces and len(vsrc) and rest.num_verts > int(vsrc.max())):
        tri["PREF"] = rest.verts[vsrc[tri["vidx"]]]   # torn cloth: map split verts to their rest vertex
    else:
        tri["PREF"] = None
    return tri


def build_packed(scene, frame, preview=False, motion_blur=None, aspect=None, extra=None):
    """Evaluate the scene at ``frame`` and return ``(PackedScene, info)``."""
    from .render import native as nat
    from .render.env import Environment
    from . import color as cm
    cm.set_working_space(scene.settings.get("working_space"), scene.settings.get("color_input"))
    scene._preview_flag = bool(preview)   # liquid / hair meshing detail
    f0, f1 = shutter_frames(scene, frame, motion_blur)
    moving = f1 != f0
    st = scene.state(frame)
    ctx = st.ctx
    res = scene.settings.get("resolution", [960, 540])
    if aspect is None:
        aspect = float(res[0]) / float(res[1])
    packed = nat.PackedScene()
    textable = TextureTable()
    mats = _MatTable(scene, frame, textable, preview)
    P0s, P1s, N0s, N1s, UVs, PREFs, MATs, OBJs = [], [], [], [], [], [], [], []
    lights = []
    info = {"objects": 0, "triangles": 0, "lights": 0}
    obj_id = 0

    def push(tri0, tri1, slots_idx, oid):
        T = len(tri0["P"])
        if T == 0:
            return
        P0s.append(tri0["P"])
        N0s.append(tri0["N"])
        UVs.append(tri0["UV"])
        pref = tri0.get("PREF")
        PREFs.append(pref if pref is not None else tri0["P"])
        P1s.append(tri1["P"] if tri1 is not None else tri0["P"])
        N1s.append(tri1["N"] if tri1 is not None else tri0["N"])
        fm = np.clip(tri0["mat"], 0, len(slots_idx) - 1)
        MATs.append(np.asarray(slots_idx, np.int32)[fm])
        OBJs.append(np.full(T, oid, np.int32))
        info["triangles"] += T

    for name in scene.order:
        node = scene.nodes[name]
        if node.type not in ("mesh", "instance", "curve", "cloth", "particles", "light", "liquid", "hair"):
            continue
        if not _node_visible(scene, name, frame):
            continue
        if node.type == "light":
            ls, area = light_structs(scene, name, f0)
            if area is not None:
                m0, am = area
                tri0 = m0.to_triangles()
                tri1 = None
                if moving:
                    _, area1 = light_structs(scene, name, f1)
                    tri1 = area1[0].to_triangles()
                idx = mats.add_raw(am)
                tri0["PREF"] = None
                push(tri0, tri1, [idx], obj_id)
                obj_id += 1
            if moving and ls:
                # analytic lights don't support motion; use shutter-centre state
                ls, _ = light_structs(scene, name, frame)
            lights += ls
            info["lights"] += 1
            continue
        flags = 0 if node.p("camera_visible", True, frame, ctx) else 1
        if node.type == "particles":
            if scene.dynamics is None:
                continue
            from .dynamics import particle_geometry
            res_p = particle_geometry(scene, node, frame, f0, f1, preview)
            if res_p is None:
                continue
            tri0, tri1 = res_p
            slots = [mats.get(r, flags, name) for r in _material_slots(scene, node, None)]
            push(tri0, tri1, slots, obj_id)
            obj_id += 1
            info["objects"] += 1
            continue
        m0 = scene.world_mesh(name, f0, preview)
        if m0 is None:
            continue
        m1 = scene.world_mesh(name, f1, preview) if moving else None
        rest = scene.rest_mesh(name if node.type != "instance" else node.spec["source"], preview)
        tri0 = _tris_with_pref(m0, rest)
        tri1 = None
        if m1 is not None and m1.num_verts == m0.num_verts and m1.num_faces == m0.num_faces:
            tri1 = m1.to_triangles()
        slots = [mats.get(r, flags, name) for r in _material_slots(scene, node, m0)]
        push(tri0, tri1, slots, obj_id)
        obj_id += 1
        info["objects"] += 1

    if extra:
        for tri0, matref in extra:
            push(tri0, None, [mats.get(matref)], obj_id)
            obj_id += 1

    if P0s:
        packed.P0 = np.ascontiguousarray(np.concatenate(P0s), np.float32)
        packed.N0 = np.ascontiguousarray(np.concatenate(N0s), np.float32)
        packed.UV = np.ascontiguousarray(np.concatenate(UVs), np.float32)
        packed.PREF = np.ascontiguousarray(np.concatenate(PREFs), np.float32)
        packed.tri_mat = np.ascontiguousarray(np.concatenate(MATs), np.int32)
        packed.tri_obj = np.ascontiguousarray(np.concatenate(OBJs), np.int32)
        if moving:
            P1 = np.ascontiguousarray(np.concatenate(P1s), np.float32)
            if np.any(np.abs(P1 - packed.P0) > 1e-7):
                packed.P1 = P1
                packed.N1 = np.ascontiguousarray(np.concatenate(N1s), np.float32)

    # heterogeneous volumes
    vols, vdata, voff = [], [], 0
    for name in scene.order:
        node = scene.nodes[name]
        if node.type != "volume" or not _node_visible(scene, name, frame):
            continue
        from .volumes import pack_volume
        v, arrs = pack_volume(scene, name, frame, ctx, voff)
        vols.append(v)
        for a in arrs:
            vdata.append(a)
            voff += a.size
    if vols:
        packed.volumes = vols
        packed.voldata = np.ascontiguousarray(np.concatenate(vdata) if vdata else np.zeros(1), np.float32)
        info["volumes"] = len(vols)

    # environment
    env = Environment(scene.environment_spec, scene.base_dir)
    lights += env.apply(packed, textable, frame, ctx, preview)
    # fog
    if scene.fog:
        gf = lambda k, d: scene.fog[k].evaluate(frame, ctx) if k in scene.fog else d
        packed.fog_density = float(gf("density", 0.0))
        packed.fog_color = tuple(parse_color(gf("color", [1.0, 1.0, 1.0])))
        packed.fog_anisotropy = float(gf("anisotropy", 0.0))
        packed.fog_height_min = float(gf("height_min", 0.0))
        packed.fog_height_max = float(gf("height_max", 0.0))
    # camera
    cam_name = scene.active_camera(frame)
    if cam_name is None:
        packed.camera = default_camera(scene, frame, aspect)
    else:
        packed.camera = _camera_struct(scene, cam_name, f0, f1, aspect)
    info["camera"] = cam_name or "<auto>"
    packed.materials = mats.list
    packed.lights = lights
    packed.textures, packed.texdata = textable.textures, textable.data()
    return packed, info
