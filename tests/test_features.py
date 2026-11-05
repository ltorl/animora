"""Tests for GPU rendering, native smoke with mesh colliders, cloth tearing, HumanIK-style
retargeting and skin weight painting."""
import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from animora.scene import Scene  # noqa: E402


def _scene(nodes, **settings):
    st = {"frame_range": [1, 20], "resolution": [64, 36]}
    st.update(settings)
    return Scene({"settings": st, "nodes": nodes})


# ============================================================================== GPU
class TestGPU(unittest.TestCase):
    def test_gpu_matches_cpu(self):
        from animora.render import native as nat
        if not nat.available() or not nat.gpu_available():
            self.skipTest("no Metal GPU renderer on this machine")
        from animora.render import renderer
        nodes = [
            {"name": "cam", "type": "camera", "translate": [0, 1, 4], "look_at": [0, 0.5, 0]},
            {"name": "floor", "type": "mesh", "primitive": "plane", "size": 10},
            {"name": "ball", "type": "mesh", "primitive": "uv_sphere", "radius": 0.5, "translate": [0, 0.5, 0],
             "material": {"base_color": [0.8, 0.2, 0.1], "roughness": 0.4}},
            {"name": "key", "type": "light", "light": "area", "translate": [2, 3, 2], "look_at": [0, 0, 0],
             "intensity": 30, "size": 1},
        ]
        imgs = {}
        for dev in ("cpu", "gpu"):
            sc = _scene(nodes, resolution=[96, 54], samples=64, device=dev, denoise=False)
            res = renderer.render_frame(sc, 1)
            self.assertEqual(res["info"]["device"], dev)
            imgs[dev] = np.asarray(res["display"], float)[..., :3]
        # different random streams: compare means of the whole image and of coarse blocks
        a, b = imgs["cpu"], imgs["gpu"]
        self.assertLess(abs(a.mean() - b.mean()), 0.03)
        blk = lambda x: x[:54 // 6 * 6, :96 // 8 * 8].reshape(6, 9, 12, 8, 3).mean(axis=(1, 3))  # noqa: E731
        self.assertLess(np.abs(blk(a) - blk(b)).mean(), 0.05)


# ============================================================================== smoke
def _smoke_scene(native, colliders=None, res=20):
    nodes = [
        {"name": "smoke", "type": "volume", "size": [2, 2, 2], "translate": [0, 1, 0],
         "smoke": {"resolution": res, "native": native, "buoyancy": 4, "seed": 3,
                   "sources": [{"shape": "sphere", "center": [0, -0.6, 0], "radius": 0.25, "density": 1, "heat": 1}],
                   "colliders": colliders or []}},
    ]
    return nodes


class TestSmoke(unittest.TestCase):
    def test_native_matches_numpy(self):
        from animora import smoke
        if not smoke._native_ok():
            self.skipTest("native library unavailable")
        out = {}
        for native in (True, False):
            sc = _scene(_smoke_scene(native))
            d, _ = smoke.smoke_frame(sc, "smoke", 8)
            out[native] = np.asarray(d, float)
        a, b = out[True], out[False]
        self.assertGreater(a.sum(), 0)
        # same physics, different advection/projection implementations: total mass and the
        # height of the plume's centre of mass agree closely
        self.assertLess(abs(a.sum() - b.sum()) / b.sum(), 0.25)
        zc = lambda d: (d.sum(axis=(0, 2)) * np.arange(d.shape[1])).sum() / d.sum()  # noqa: E731
        self.assertLess(abs(zc(a) - zc(b)), 1.5)

    def test_mesh_collider_voxelised_solid(self):
        from animora import smoke
        nodes = _smoke_scene(True, colliders=[{"node": "ring", "shape": "mesh"}]) + [
            {"name": "ring", "type": "mesh", "primitive": "torus", "major_radius": 0.5, "minor_radius": 0.15,
             "translate": [0, 1.2, 0]}]
        sc = _scene(nodes)
        sim = smoke.get_sim(sc, "smoke")
        sim._update_colliders(1)
        self.assertIsNotNone(sim.solid)
        sol = sim.solid.astype(bool)
        # the torus hole (centre) is open, the tube is solid
        def cell(p):
            l = np.asarray(p) - np.array([0, 1, 0]) + sim.size / 2
            i = (l / sim.h).astype(int)
            return sol[i[2], i[1], i[0]]
        self.assertTrue(cell([0.5, 1.2, 0.0]))
        self.assertFalse(cell([0.0, 1.2, 0.0]))
        self.assertFalse(cell([0.0, 0.4, 0.0]))
        # smoke never ends up inside the obstacle
        d, _ = smoke.smoke_frame(sc, "smoke", 10)
        self.assertLess(float(np.asarray(d)[sol].max(initial=0)), 1e-6)


# ============================================================================== cloth tearing
class TestClothTear(unittest.TestCase):
    def _sheet(self, tear):
        nodes = [
            {"name": "ball", "type": "mesh", "primitive": "uv_sphere", "radius": 0.3,
             "translate": {"keys": [[1, [0, 2.0, 0]], [6, [0, 1.0, 0]], [16, [0, -0.5, 0]]], "interp": "linear"}},
            {"name": "sheet", "type": "cloth", "size": 1.6, "resolution": 16, "translate": [0, 1.2, 0],
             "pin": ["edge_min_x", "edge_max_x"], "colliders": ["ball"], "tear": tear, "tear_strain": 1.3,
             "start_frame": 1},
        ]
        sc = Scene({"settings": {"frame_range": [1, 16]}, "dynamics": {"substeps": 6}, "nodes": nodes})
        sc.dynamics.simulate(16)
        return sc

    def test_tear_splits_vertices(self):
        sc = self._sheet(True)
        m0 = sc.world_mesh("sheet", 1)
        m1 = sc.world_mesh("sheet", 16)
        self.assertGreater(m1.num_verts, m0.num_verts)
        self.assertIsNotNone(sc.dynamics.cloth_topology("sheet", 16))
        self.assertTrue(np.all(np.isfinite(m1.verts)))

    def test_no_tear_keeps_topology(self):
        sc = self._sheet(False)
        self.assertEqual(sc.world_mesh("sheet", 16).num_verts, sc.world_mesh("sheet", 1).num_verts)
        self.assertIsNone(sc.dynamics.cloth_topology("sheet", 16))


# ============================================================================== retargeting
HIK = ["Hips", "Spine", "Spine1", "Spine2", "Neck", "Head", "LeftShoulder", "LeftArm", "LeftForeArm",
       "LeftHand", "RightShoulder", "RightArm", "RightForeArm", "RightHand", "LeftUpLeg", "LeftLeg",
       "LeftFoot", "LeftToeBase", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"]


class TestRetargetNames(unittest.TestCase):
    def check(self, names, expect):
        from animora.retarget import canonical_slots
        slots = canonical_slots(names)
        for slot, joint in expect.items():
            self.assertEqual(slots.get(slot), joint, f"{slot} in {names[:4]}...")

    def test_mixamo(self):
        names = ["mixamorig:" + n for n in HIK]
        self.check(names, {"Hips": "mixamorig:Hips", "LeftForeArm": "mixamorig:LeftForeArm",
                           "RightUpLeg": "mixamorig:RightUpLeg", "Head": "mixamorig:Head"})

    def test_unreal(self):
        names = ["pelvis", "spine_01", "spine_02", "neck_01", "head", "clavicle_l", "upperarm_l", "lowerarm_l",
                 "hand_l", "clavicle_r", "upperarm_r", "lowerarm_r", "hand_r", "thigh_l", "calf_l", "foot_l",
                 "ball_l", "thigh_r", "calf_r", "foot_r", "ball_r"]
        self.check(names, {"Hips": "pelvis", "LeftArm": "upperarm_l", "RightForeArm": "lowerarm_r",
                           "LeftUpLeg": "thigh_l", "RightLeg": "calf_r", "LeftToeBase": "ball_l",
                           "LeftShoulder": "clavicle_l"})

    def test_cmu(self):
        names = ["hip", "lowerback", "upperback", "thorax", "lowerneck", "head", "lclavicle", "lhumerus",
                 "lradius", "lhand", "rclavicle", "rhumerus", "rradius", "rhand", "lfemur", "ltibia", "lfoot",
                 "ltoes", "rfemur", "rtibia", "rfoot", "rtoes"]
        self.check(names, {"LeftArm": "lhumerus", "LeftForeArm": "lradius", "RightUpLeg": "rfemur",
                           "RightLeg": "rtibia", "LeftFoot": "lfoot"})

    def test_smpl(self):
        names = ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2", "left_ankle",
                 "right_ankle", "spine3", "left_foot", "right_foot", "neck", "left_collar", "right_collar", "head",
                 "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist"]
        self.check(names, {"LeftArm": "left_shoulder", "LeftShoulder": "left_collar", "LeftFoot": "left_ankle",
                           "LeftToeBase": "left_foot", "RightLeg": "right_knee", "RightHand": "right_wrist"})

    def test_rigify(self):
        names = ["spine", "spine.001", "spine.002", "spine.003", "spine.004", "spine.006", "shoulder.L",
                 "upper_arm.L", "forearm.L", "hand.L", "shoulder.R", "upper_arm.R", "forearm.R", "hand.R",
                 "thigh.L", "shin.L", "foot.L", "toe.L", "thigh.R", "shin.R", "foot.R", "toe.R"]
        self.check(names, {"LeftArm": "upper_arm.L", "RightForeArm": "forearm.R", "LeftLeg": "shin.L",
                           "RightToeBase": "toe.R"})


def _biped(prefix="", yaw=0.0, anim=False):
    """Small biped (HumanIK names) facing +Z rotated by ``yaw`` degrees about Y."""
    def k(name, parent, t, rot=None):
        n = {"name": prefix + name, "type": "joint", "translate": t}
        if parent:
            n["parent"] = prefix + parent
        if rot is not None:
            n["rotate"] = rot
        return n
    a = (lambda keys: {"keys": keys}) if anim else (lambda keys: [0, 0, 0])
    return [
        k("Hips", None, {"keys": [[1, [0, 1, 0]], [10, [0.2, 1.05, 0.5]]]} if anim else [0, 1, 0],
          [0, yaw, 0] if not anim else {"keys": [[1, [0, yaw, 0]], [10, [5, yaw + 20, 0]]]}),
        k("Spine", "Hips", [0, 0.2, 0], a([[1, [0, 0, 0]], [10, [15, 0, 5]]])),
        k("Neck", "Spine", [0, 0.4, 0]),
        k("Head", "Neck", [0, 0.15, 0], a([[1, [0, 0, 0]], [10, [0, 30, 0]]])),
        k("LeftArm", "Spine", [0.2, 0.35, 0], a([[1, [0, 0, 0]], [10, [0, 20, -60]]])),
        k("LeftForeArm", "LeftArm", [0.3, 0, 0], a([[1, [0, 0, 0]], [10, [0, -70, 0]]])),
        k("LeftHand", "LeftForeArm", [0.25, 0, 0]),
        k("RightArm", "Spine", [-0.2, 0.35, 0], a([[1, [0, 0, 0]], [10, [10, 0, 50]]])),
        k("RightForeArm", "RightArm", [-0.3, 0, 0], a([[1, [0, 0, 0]], [10, [0, 40, 0]]])),
        k("RightHand", "RightForeArm", [-0.25, 0, 0]),
        k("LeftUpLeg", "Hips", [0.1, -0.05, 0], a([[1, [0, 0, 0]], [10, [-40, 0, 0]]])),
        k("LeftLeg", "LeftUpLeg", [0, -0.45, 0], a([[1, [0, 0, 0]], [10, [60, 0, 0]]])),
        k("LeftFoot", "LeftLeg", [0, -0.45, 0]),
        k("RightUpLeg", "Hips", [-0.1, -0.05, 0], a([[1, [0, 0, 0]], [10, [30, 0, 0]]])),
        k("RightLeg", "RightUpLeg", [0, -0.45, 0], a([[1, [0, 0, 0]], [10, [20, 0, 0]]])),
        k("RightFoot", "RightLeg", [0, -0.45, 0]),
    ]


class TestRetargetTransfer(unittest.TestCase):
    def test_identity_retarget_recovers_pose(self):
        """Retargeting onto the same skeleton (renamed and turned 90°) reproduces the poses."""
        from animora.retarget import retarget
        src = Scene({"settings": {"frame_range": [1, 10]}, "nodes": _biped(anim=True)})
        tgt_nodes = _biped(prefix="mixamorig:", yaw=90.0)
        tgt = Scene({"settings": {"frame_range": [1, 10]}, "nodes": tgt_nodes})
        keys, info = retarget(src, tgt, source_yaw="auto", height_scale=1.0)
        self.assertAlmostEqual(abs(info["source_yaw"]), 90.0, places=3)
        self.assertEqual(len(info["map"]), 16)
        for n in tgt_nodes:
            if n["name"] in keys:
                n.update(keys[n["name"]])
        out = Scene({"settings": {"frame_range": [1, 10]}, "nodes": tgt_nodes})
        R = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], float)  # +90° about Y
        for f in (1, 5, 10):
            ss, so = src.state(f), out.state(f)
            hs = ss.world("Hips")[:3, 3]
            ho = so.world("mixamorig:Hips")[:3, 3]
            for j in ("Head", "LeftHand", "RightHand", "LeftFoot", "RightFoot"):
                ps = ss.world(j)[:3, 3] - hs
                po = so.world("mixamorig:" + j)[:3, 3] - ho
                np.testing.assert_allclose(po, R @ ps, atol=1e-4, err_msg=f"{j} @ {f}")

    def test_bvh_scene_block(self):
        """The bundled HumanIK BVH retargets onto the UE-style mannequin through the scene block."""
        with open(os.path.join(ROOT, "examples", "retarget_walk.json")) as fh:
            data = json.load(fh)
        sc = Scene(data,
                   os.path.join(ROOT, "examples"))
        self.assertTrue(sc.nodes["thigh_l"].spec.get("rotate"))
        p0 = sc.state(sc.frame_start).world("pelvis")[:3, 3]
        p1 = sc.state(sc.frame_end).world("pelvis")[:3, 3]
        self.assertGreater(np.linalg.norm((p1 - p0)[[0, 2]]), 0.5)   # it walks
        for f in (sc.frame_start, (sc.frame_start + sc.frame_end) / 2):
            foot = min(sc.state(f).world("foot_l")[1, 3], sc.state(f).world("foot_r")[1, 3])
            self.assertLess(abs(foot), 0.2)                            # a foot stays near the floor


# ============================================================================== weight painting
def _arm(paint=None, **skin):
    nodes = [
        {"name": "root", "type": "joint", "translate": [0, 0, 0]},
        {"name": "l_arm", "type": "joint", "parent": "root", "translate": [0.2, 0, 0]},
        {"name": "l_hand", "type": "joint", "parent": "l_arm", "translate": [0.8, 0, 0]},
        {"name": "r_arm", "type": "joint", "parent": "root", "translate": [-0.2, 0, 0]},
        {"name": "r_hand", "type": "joint", "parent": "r_arm", "translate": [-0.8, 0, 0]},
        {"name": "span", "type": "mesh", "primitive": "cylinder", "radius": 0.08, "height": 2.4,
         "height_segments": 48, "segments": 12, "rotate": [0, 0, 90],
         "skin": dict({"root": "root", "weight_paint": paint or []}, **skin)},
    ]
    return nodes


def _weights(nodes):
    from animora.rig import skin_data
    sc = _scene(nodes)
    d = skin_data(sc, sc.nodes["span"], sc.object_mesh("span", 1))
    return d["weights"], d["joints"], d["rest_world"]


class TestWeightPaint(unittest.TestCase):
    def test_mirror_name(self):
        from animora.weightpaint import mirror_name
        for a, b in [("LeftArm", "RightArm"), ("l_elbow", "r_elbow"), ("upperarm_l", "upperarm_r"),
                     ("hand.L", "hand.R"), ("LHip", "RHip"), ("lhumerus", "rhumerus"), ("spine", "spine"),
                     ("mixamorig:RightHand", "mixamorig:LeftHand"), ("thigh_R_jnt", "thigh_L_jnt")]:
            self.assertEqual(mirror_name(a), b)
            self.assertEqual(mirror_name(b), a)

    def test_replace_stroke_and_normalisation(self):
        W0, J, P = _weights(_arm())
        W1, _, _ = _weights(_arm([{"joint": "l_hand", "op": "replace", "value": 1.0, "center": [0.6, 0, 0],
                                   "radius": 0.15, "falloff": "constant"}]))
        np.testing.assert_allclose(W1.sum(axis=1), 1.0, atol=1e-9)
        k = J.index("l_hand")
        inside = np.linalg.norm(P - [0.6, 0, 0], axis=1) < 0.149
        outside = np.linalg.norm(P - [0.6, 0, 0], axis=1) > 0.151
        np.testing.assert_allclose(W1[inside, k], 1.0, atol=1e-9)
        np.testing.assert_allclose(W1[outside], W0[outside], atol=1e-12)

    def test_add_respects_locks(self):
        W0, J, P = _weights(_arm())
        W1, _, _ = _weights(_arm([{"joint": "l_hand", "op": "add", "value": 0.5, "center": [0.25, 0, 0],
                                   "radius": 0.3}], locked_joints=["root"]))
        np.testing.assert_allclose(W1[:, J.index("root")], W0[:, J.index("root")], atol=1e-12)
        np.testing.assert_allclose(W1.sum(axis=1), 1.0, atol=1e-9)
        self.assertGreater(W1[:, J.index("l_hand")].sum(), W0[:, J.index("l_hand")].sum())

    def test_mirrored_stroke_is_symmetric(self):
        stroke = {"joint": "l_hand", "op": "replace", "value": 0.9, "center": [0.5, 0.02, 0], "radius": 0.2,
                  "mirror": "x"}
        W, J, P = _weights(_arm([stroke]))
        from animora.weightpaint import nearest_indices
        Pm = P * [-1, 1, 1]
        nn = nearest_indices(P, Pm)
        np.testing.assert_allclose(W[nn, J.index("r_hand")], W[:, J.index("l_hand")], atol=1e-3)

    def test_global_ops(self):
        W, J, P = _weights(_arm([{"op": "prune", "threshold": 0.1}, {"op": "max_influences", "count": 2},
                                 {"op": "smooth", "iterations": 2, "strength": 0.3},
                                 {"op": "max_influences", "count": 2}]))
        self.assertLessEqual(int((W > 0).sum(axis=1).max()), 2)
        np.testing.assert_allclose(W.sum(axis=1), 1.0, atol=1e-9)
        # mirror op: copy the +x side to the -x side
        Wm, _, _ = _weights(_arm([{"joint": "l_hand", "op": "replace", "value": 1, "center": [0.9, 0, 0],
                                   "radius": 0.3}, {"op": "mirror", "axis": "x", "direction": "+"}]))
        from animora.weightpaint import nearest_indices
        nn = nearest_indices(P, P * [-1, 1, 1])
        np.testing.assert_allclose(Wm[nn, J.index("r_hand")], Wm[:, J.index("l_hand")], atol=1e-3)
        # move all of l_hand's weight onto l_arm
        Wv, _, _ = _weights(_arm([{"op": "move", "from": "l_hand", "to": "l_arm"}]))
        self.assertLess(Wv[:, J.index("l_hand")].max(), 1e-12)

    def test_export_import_roundtrip_and_transfer(self):
        from animora.weightpaint import export_weights
        nodes = _arm([{"joint": "l_hand", "op": "add", "value": 0.4, "at": "l_arm", "radius": 0.3}])
        W, J, P = _weights(nodes)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.json")
            export_weights(path, W, J, P, node="span")
            n2 = _arm()
            n2[-1]["skin"] = {"root": "root", "weights": path}
            W2, _, _ = _weights(n2)
            np.testing.assert_allclose(W2, W, atol=2e-4)   # export drops weights < 1e-4
            n3 = _arm()
            n3[-1]["height_segments"] = 64          # different vertex count -> position transfer
            n3[-1]["skin"] = {"root": "root", "weights": path}
            W3, _, P3 = _weights(n3)
            self.assertEqual(len(W3), len(P3))
            np.testing.assert_allclose(W3.sum(axis=1), 1.0, atol=1e-9)

    def test_heatmap_renders(self):
        from animora.weightpaint import render_heatmap
        sc = _scene(_arm())
        img = render_heatmap(sc, "span", "l_hand", None, 80, 60)
        self.assertEqual(img.shape, (60, 80, 3))
        self.assertGreater(img.std(), 0.02)
        img2 = render_heatmap(sc, "span", None, 1, 80, 60)
        self.assertEqual(img2.shape, (60, 80, 3))


if __name__ == "__main__":
    unittest.main()
