"""Tests for the 'Maya gap' improvements: driven keys, graph editor, colour management, volumes,
simulations, modelling, pipeline formats, ..."""
import json
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from animora.scene import Scene  # noqa: E402


def _scene(nodes, **settings):
    st = {"frame_range": [1, 20], "resolution": [64, 36]}
    st.update(settings)
    return Scene({"settings": st, "nodes": nodes})


def _load(path):
    with open(path) as fh:
        return Scene(json.load(fh))


class TestDrivenKeys(unittest.TestCase):
    def test_driven(self):
        sc = _scene([
            {"name": "ctrl", "type": "locator", "translate": {"keys": [[1, [0, 0, 0]], [20, [0, 1, 0]]]}},
            {"name": "door", "type": "locator",
             "rotate": [0, {"driver": "ctrl.translate.y", "keys": [[0, 0], [1, 90]]}, 0]},
        ])
        self.assertAlmostEqual(float(sc.get_value("door.rotate", 1)[1]), 0.0, places=4)
        self.assertAlmostEqual(float(sc.get_value("door.rotate", 20)[1]), 90.0, places=4)


class TestGraph(unittest.TestCase):
    def test_sample_and_plot(self):
        from animora import graph
        sc = _scene([{"name": "b", "type": "locator",
                      "translate": {"keys": [[1, [0, 0, 0]], [10, [0, 5, 0]], [20, [0, 0, 0]]]}}])
        frames = list(np.linspace(1, 20, 39))
        series = graph.sample(sc, ["b.translate.y"], frames)
        self.assertEqual(len(series), 1)
        self.assertAlmostEqual(float(np.max(series[0][1])), 5.0, places=3)
        txt = graph.plot_terminal(series, frames, 60, 12, color=False)
        self.assertIn("◆", txt)
        with tempfile.TemporaryDirectory() as d:
            p = graph.plot_png(series, frames, os.path.join(d, "g.png"), size=(400, 200))
            self.assertTrue(os.path.getsize(p) > 1000)

    def test_edit_keys(self):
        from animora import graph
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "s.json")
            with open(path, "w") as fh:
                json.dump({"settings": {"frame_range": [1, 10]},
                           "nodes": [{"name": "b", "type": "locator", "translate": [0, 0, 0]}]}, fh)
            graph.edit_keys(path, "b.translate.y", "set", frame=1, value=0.0)
            graph.edit_keys(path, "b.translate.y", "set", frame=10, value=4.0, interp="linear")
            sc = _load(path)
            self.assertAlmostEqual(float(sc.get_value("b.translate", 10)[1]), 4.0, places=4)
            graph.edit_keys(path, "b.translate.y", "move", frame=10, new_frame=5)
            sc = _load(path)
            self.assertAlmostEqual(float(sc.get_value("b.translate", 5)[1]), 4.0, places=4)


class TestColor(unittest.TestCase):
    def test_roundtrip(self):
        from animora import color as cm
        x = np.random.RandomState(1).rand(20, 3)
        for sp in ("srgb", "acescg", "aces2065", "acescct", "acescc", "display_p3", "rec2020", "adobe_rgb"):
            y = cm.convert(cm.convert(x, "lin_srgb", sp), sp, "lin_srgb")
            self.assertLess(float(np.abs(y - x).max()), 1e-6, sp)

    def test_known_matrix(self):
        from animora import color as cm
        m = cm.matrix("lin_srgb", "aces2065")
        self.assertAlmostEqual(m[0, 0], 0.4397, places=3)
        self.assertAlmostEqual(m[1, 1], 0.8134, places=3)
        # white stays white within the same white point
        np.testing.assert_allclose(cm.convert([1, 1, 1], "lin_srgb", "lin_p3"), [1, 1, 1], atol=1e-9)

    def test_parse_color_spaces(self):
        from animora import color as cm
        from animora.materials import parse_color
        cm.set_working_space("lin_srgb")
        a = parse_color({"rgb": [0.2, 0.3, 0.4], "space": "acescg"})
        np.testing.assert_allclose(a, cm.convert([0.2, 0.3, 0.4], "acescg", "lin_srgb"), atol=1e-9)
        cm.set_working_space("acescg")
        try:
            np.testing.assert_allclose(parse_color([1, 0, 0]), cm.convert([1, 0, 0], "lin_srgb", "acescg"), atol=1e-9)
        finally:
            cm.set_working_space("lin_srgb")

    def test_display_and_luts(self):
        from animora.render import tonemap
        img = np.full((2, 2, 3), 0.18, np.float32)
        a = tonemap.process(img, {"working_space": "acescg", "display": "display_p3"})
        b = tonemap.process(img, {})
        np.testing.assert_allclose(a, b, atol=2e-3)  # grey is grey in every space
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "inv.spi1d")
            with open(p, "w") as fh:
                fh.write("Version 1\nFrom 0.0 1.0\nLength 2\nComponents 1\n{\n1.0\n0.0\n}\n")
            out = tonemap.apply_lut(np.full((1, 1, 3), 0.25, np.float32), tonemap.read_lut(p))
            np.testing.assert_allclose(out, 0.75, atol=1e-6)


class TestVolumesAndSims(unittest.TestCase):
    def test_procedural_volume_packs(self):
        from animora.scene import build_packed
        sc = _scene([
            {"name": "cam", "type": "camera", "translate": [0, 0, 6]},
            {"name": "cloud", "type": "volume", "size": [2, 2, 2], "density": 4, "shape": "sphere"},
        ])
        build_packed(sc, 1, preview=True)  # must not raise

    def test_smoke_rises(self):
        sc = _scene([{"name": "fire", "type": "volume", "size": [1, 2, 1], "translate": [0, 1, 0], "blackbody": 4,
                      "smoke": {"resolution": 16, "buoyancy": 4,
                                "sources": [{"shape": "sphere", "center": [0, -0.8, 0], "radius": 0.2,
                                             "density": 2, "heat": 1}]}}])
        from animora.smoke import smoke_frame
        d, T = smoke_frame(sc, "fire", 12)
        self.assertGreater(float(d.sum()), 0.0)
        ny = d.shape[1]
        self.assertGreater(float(d[:, ny // 2:, :].sum()), 0.0)  # smoke has reached the upper half
        self.assertGreater(float(T.max()), 0.0)

    def test_liquid_settles_in_domain(self):
        sc = _scene([{"name": "water", "type": "liquid", "size": [1, 1, 0.4], "spacing": 0.08,
                      "emitters": [{"type": "fill", "shape": "box", "center": [-0.25, 0, 0], "size": [0.4, 0.8, 0.4]}]}],
                    frame_range=[1, 12])
        from animora.liquid import get_sim, liquid_mesh
        P0 = get_sim(sc, "water").at(1).copy()
        P = get_sim(sc, "water").at(12)
        self.assertGreater(len(P), 20)
        self.assertTrue(np.all(np.abs(P) <= np.array([0.5, 0.5, 0.2]) + 1e-3))  # stays in the box
        self.assertLess(float(P[:, 1].mean()), float(P0[:, 1].mean()))      # falls and spreads
        self.assertGreater(len(liquid_mesh(sc, "water", 12, preview=True).verts), 0)


class TestModelling(unittest.TestCase):
    def test_sdf_text_sweep_loft(self):
        sc = _scene([
            {"name": "csg", "type": "mesh", "primitive": "sdf", "resolution": 32,
             "shapes": [{"type": "box", "size": [1, 1, 1]}, {"type": "sphere", "radius": 0.6, "op": "subtract"}]},
            {"name": "txt", "type": "mesh", "primitive": "text", "text": "Hi", "size": 0.5, "depth": 0.1},
            {"name": "sw", "type": "mesh", "primitive": "sweep", "path": [[0, 0, 0], [0, 1, 0], [1, 1, 0]], "radius": 0.1},
            {"name": "lf", "type": "mesh", "primitive": "loft",
             "sections": [{"profile": "circle", "radius": 0.3}, {"profile": "star", "radius": 0.2, "translate": [0, 1, 0]}]},
        ])
        for n in ("csg", "txt", "sw", "lf"):
            m = sc.object_mesh(n, 1)
            self.assertGreater(len(m.verts), 10, n)
        # box minus a centred sphere: the centre is hollow but the corners remain
        V = sc.object_mesh("csg", 1).verts
        self.assertGreater(float(np.abs(V).max()), 0.45)
        self.assertGreater(float(np.linalg.norm(V, axis=1).min()), 0.5)

    def test_boolean_and_decimate(self):
        from animora import modeling
        from animora.geometry import make_primitive as primitive
        cube = primitive("cube", size=1.0)
        cutter = primitive("uv_sphere", radius=0.5)
        cutter.verts = cutter.verts + np.array([0.5, 0.5, 0.5])
        out = modeling.boolean(cube, op="difference", _other=cutter, resolution=40)
        self.assertGreater(len(out.verts), 0)
        # the cut corner is gone, the opposite one remains
        self.assertLess(float(np.linalg.norm(out.verts - [-0.5, -0.5, -0.5], axis=1).min()), 0.1)
        self.assertGreater(float(np.linalg.norm(out.verts - [0.5, 0.5, 0.5], axis=1).min()), 0.3)
        dense = primitive("uv_sphere", radius=1.0, segments=64, rings=32)
        dec = modeling.decimate(dense, ratio=0.25)
        self.assertLess(len(dec.counts), len(dense.counts) * 0.6)


class TestDynamicsImprovements(unittest.TestCase):
    def _rest_y(self, shape, prim, frames=48, **prim_kw):
        sc = _scene([{"name": "b", "type": "mesh", "primitive": prim, "translate": [0, 2, 0],
                      "rigid_body": {"shape": shape}, **prim_kw}], frame_range=[1, frames])
        return sc, float(sc.world_matrix("b", frames)[1, 3])

    def test_rigid_shapes_rest_height(self):
        _, y = self._rest_y("capsule", "capsule", radius=0.25, height=1.0)
        self.assertGreater(y, 0.15)
        self.assertLess(y, 0.8)
        _, y = self._rest_y("cylinder", "cylinder", radius=0.4, height=0.5)
        self.assertGreater(y, 0.15)
        self.assertLess(y, 0.6)
        _, y = self._rest_y("mesh", "cube", size=1)
        self.assertAlmostEqual(y, 0.5, delta=0.08)

    def test_cloth_self_collision(self):
        def run(sc_flag):
            sc = _scene([{"name": "c", "type": "cloth", "size": 1.0, "resolution": 14, "translate": [0, 1.5, 0],
                          "self_collision": sc_flag, "stiffness": 1.0}], frame_range=[1, 40])
            return sc.dynamics.cloth_verts("c", 40)
        V = run(True)
        self.assertTrue(np.all(np.isfinite(V)))
        self.assertGreaterEqual(float(V[:, 1].min()), -0.02)  # ground still respected

    def test_hair_ground(self):
        sc = _scene([
            {"name": "skin", "type": "mesh", "primitive": "plane", "size": 0.5, "translate": [0, 0.05, 0]},
            {"name": "h", "type": "hair", "source": "skin", "count": 40, "length": 0.4, "segments": 4,
             "dynamics": True, "gravity": 1.0, "comb": [0, -1, 0], "comb_strength": 1.0},
        ], frame_range=[1, 20])
        from animora.hair import hair_curves
        C = hair_curves(sc, "h", 20)
        self.assertEqual(C.shape[1], 5)
        self.assertGreaterEqual(float(C[..., 1].min()), -0.01)


class TestPipeline(unittest.TestCase):
    def _anim_scene(self):
        return _scene([
            {"name": "cam", "type": "camera", "translate": [0, 1, 5]},
            {"name": "ball", "type": "mesh", "primitive": "uv_sphere", "radius": 0.5,
             "translate": {"keys": [[1, [0, 0, 0]], [10, [0, 2, 0]]]}},
            {"name": "bent", "type": "mesh", "primitive": "cube", "size": 1,
             "modifiers": [{"type": "twist", "angle": {"keys": [[1, 0], [10, 90]]}}]},
        ], frame_range=[1, 10])

    def test_gltf_round_trip(self):
        from animora.io.gltf import export_gltf, read_gltf, gltf_to_scene
        sc = self._anim_scene()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.glb")
            summ = export_gltf(sc, p)
            self.assertGreaterEqual(summ["meshes"], 2)
            m, _ = read_gltf(p, "ball")
            self.assertEqual(len(m.verts) > 0, True)
            data = gltf_to_scene(p)
            back = Scene(data)
            back.scene_path = os.path.join(d, "a.json")
            self.assertAlmostEqual(float(back.world_matrix("ball", 10)[1, 3]), 2.0, places=3)
            self.assertAlmostEqual(float(back.world_matrix("ball", 1)[1, 3]), 0.0, places=3)
            if summ.get("morph_targets"):
                base, _ = read_gltf(p, "bent")
                tgt, _ = read_gltf(p, "bent@%d" % (summ["morph_targets"] - 1))
                self.assertGreater(float(np.abs(tgt.verts - base.verts).max()), 1e-3)

    def test_usd_export(self):
        from animora.io.usd import export_usd
        import shutil
        import subprocess
        sc = self._anim_scene()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.usda")
            export_usd(sc, p)
            with open(p) as fh:
                txt = fh.read()
            self.assertTrue(txt.startswith("#usda 1.0"))
            self.assertIn("timeSamples", txt)
            if shutil.which("usdchecker"):
                r = subprocess.run(["usdchecker", p], capture_output=True, text=True, timeout=60)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_bvh_import(self):
        from animora.io.bvh import bvh_to_scene
        bvh = """HIERARCHY
ROOT Hips
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Spine
  {
    OFFSET 0 10 0
    CHANNELS 3 Zrotation Xrotation Yrotation
    End Site
    {
      OFFSET 0 10 0
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0416667
0 90 0 0 0 0 0 0 0
0 90 0 0 0 0 90 0 0
"""
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.bvh")
            with open(p, "w") as fh:
                fh.write(bvh)
            sc = Scene(bvh_to_scene(p, scale=0.01, display=False))
            self.assertEqual(sc.fps, 24.0)
            # frame 1: spine straight up from hips at y=0.9; frame 2: spine rotated 90 deg about Z
            self.assertAlmostEqual(float(sc.world_matrix("Spine", 1)[1, 3]), 1.0, places=4)
            M2 = sc.world_matrix("Spine", 2)
            tip = M2 @ np.array([0, 0.1, 0, 1.0])
            np.testing.assert_allclose(tip[:3], [-0.1, 1.0, 0.0], atol=1e-4)

    def test_file_for_frame(self):
        from animora.scene import file_for_frame

        class N:
            spec = {"file": "cache/sim_####.obj", "file_range": [1, 5]}
        self.assertEqual(file_for_frame(N, 3), "cache/sim_0003.obj")
        self.assertEqual(file_for_frame(N, 9), "cache/sim_0005.obj")
        N.spec = {"file": "a.glb#Body"}
        self.assertEqual(file_for_frame(N, 3), "a.glb#Body")


if __name__ == "__main__":
    unittest.main()
