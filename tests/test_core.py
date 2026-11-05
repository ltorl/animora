"""Core unit tests for Animora (stdlib unittest; run with ``python3 -m unittest discover -s tests``)."""
import math
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import animora as an  # noqa: E402
from animora import geometry as geo  # noqa: E402
from animora import mathutil as mu  # noqa: E402
from animora import modifiers  # noqa: E402
from animora.cli import parse_frames  # noqa: E402
from animora.expressions import evaluate, make_context  # noqa: E402
from animora.scene import Scene  # noqa: E402


class TestFrames(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(parse_frames("12"), [12])
        self.assertEqual(parse_frames("1-5"), [1, 2, 3, 4, 5])
        self.assertEqual(parse_frames("1-9x4"), [1, 5, 9])
        self.assertEqual(parse_frames("1,3,7"), [1, 3, 7])
        self.assertEqual(parse_frames("1-2x0.5"), [1, 1.5, 2])
        self.assertEqual(parse_frames("1-2,10-11"), [1, 2, 10, 11])

    def test_scene_range(self):
        sc = Scene({"settings": {"frame_range": [5, 9]}, "nodes": []})
        self.assertEqual(parse_frames("all", sc), [5, 6, 7, 8, 9])
        self.assertEqual(parse_frames("last", sc), [9])
        self.assertEqual(parse_frames("mid", sc), [7])


class TestCurves(unittest.TestCase):
    def test_linear(self):
        c = an.AnimCurve([[0, 0.0], [10, 10.0]], default_interp="linear")
        self.assertAlmostEqual(c(5), 5.0)
        self.assertAlmostEqual(c(-5), 0.0)   # constant pre-infinity
        self.assertAlmostEqual(c(20), 10.0)  # constant post-infinity

    def test_step_and_keys_pass_through(self):
        c = an.AnimCurve([[0, 1.0], [10, 3.0], [20, -2.0]])
        for f, v in [(0, 1.0), (10, 3.0), (20, -2.0)]:
            self.assertAlmostEqual(c(f), v, places=6)
        s = an.AnimCurve([[0, 1.0], [10, 3.0]], default_interp="step")
        self.assertAlmostEqual(s(9.9), 1.0)

    def test_cycle(self):
        c = an.AnimCurve([[0, 0.0], [10, 1.0]], post="cycle", default_interp="linear")
        self.assertAlmostEqual(c(15), 0.5, places=6)

    def test_easings(self):
        for name in an.EASINGS:
            f = an.get_easing(name)
            self.assertAlmostEqual(f(0.0), 0.0, places=4, msg=name)
            self.assertAlmostEqual(f(1.0), 1.0, places=4, msg=name)

    def test_vector_param(self):
        p = an.Param({"keys": [[1, [0, 0, 0]], [11, [10, 20, 30]]], "interp": "linear"})
        np.testing.assert_allclose(p.evaluate(6), [5, 10, 15], atol=1e-6)


class TestExpressions(unittest.TestCase):
    def test_basic(self):
        self.assertAlmostEqual(evaluate("2 + 3*4"), 14)
        ctx = make_context(48, fps=24)
        self.assertAlmostEqual(evaluate("t", **ctx), 2.0)
        self.assertAlmostEqual(evaluate("sin(tau*t)", **ctx), 0.0, places=6)

    def test_param_expression(self):
        p = an.Param("=frame*2")
        self.assertAlmostEqual(float(p.evaluate(7, make_context(7))), 14.0)
        v = an.Param("=[cos(0), 2, frame]")
        np.testing.assert_allclose(v.evaluate(3, make_context(3)), [1, 2, 3])

    def test_rejects_unsafe(self):
        with self.assertRaises(Exception):
            evaluate("__import__('os').system('echo hi')")


class TestMath(unittest.TestCase):
    def test_compose_decompose(self):
        M = mu.compose([1, 2, 3], [10, 20, 30], [2, 2, 2])
        t, R, s = mu.decompose(M)
        np.testing.assert_allclose(t, [1, 2, 3], atol=1e-9)
        np.testing.assert_allclose(s, [2, 2, 2], atol=1e-9)

    def test_quat_roundtrip(self):
        q = mu.quat_axis_angle([0, 1, 0], 90)
        v = mu.rotate_vector(q, np.array([1.0, 0, 0]))
        np.testing.assert_allclose(v, [0, 0, -1], atol=1e-9)
        q2 = mu.matrix_to_quat(mu.quat_to_matrix(q))
        self.assertAlmostEqual(abs(float(np.dot(q, q2))), 1.0, places=9)


class TestGeometry(unittest.TestCase):
    def test_primitives_valid(self):
        for kind in ("plane", "cube", "sphere", "ico_sphere", "cylinder", "cone", "disc", "torus", "capsule"):
            m = an.make_primitive(kind)
            self.assertGreater(len(m.counts), 0, kind)
            self.assertTrue(np.isfinite(m.verts).all(), kind)

    def test_cube_vector_size(self):
        m = geo.cube(size=[2, 4, 6])
        np.testing.assert_allclose(m.verts.max(0) - m.verts.min(0), [2, 4, 6], atol=1e-9)
        m = geo.cube(size=3)
        np.testing.assert_allclose(m.verts.max(0) - m.verts.min(0), [3, 3, 3], atol=1e-9)

    def test_sphere_radius(self):
        m = geo.uv_sphere(radius=2.0)
        r = np.linalg.norm(m.verts, axis=1)
        self.assertAlmostEqual(float(r.max()), 2.0, places=6)

    def test_array_modifier(self):
        m = geo.cube(1.0)
        a = modifiers.array(m, count=4, offset=(2, 0, 0))
        self.assertEqual(len(a.verts), 4 * len(m.verts))
        self.assertAlmostEqual(float(a.verts[:, 0].max()), 6.5, places=6)

    def test_obj_roundtrip(self):
        m = geo.torus()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.obj")
            an.write_mesh(p, m)
            m2, _ = an.read_mesh(p, cache=False)
        self.assertEqual(len(m2.verts), len(m.verts))


class TestScene(unittest.TestCase):
    def test_hierarchy_and_pivot(self):
        sc = Scene({"nodes": [
            {"name": "parent", "type": "group", "translate": [1, 0, 0]},
            {"name": "child", "type": "group", "parent": "parent", "translate": [0, 2, 0]},
            {"name": "piv", "type": "group", "scale": [2, 2, 2], "pivot": [0, -1, 0]},
        ]})
        np.testing.assert_allclose(sc.world_matrix("child", 1)[:3, 3], [1, 2, 0], atol=1e-9)
        # scaling about a pivot keeps the pivot point fixed
        W = sc.world_matrix("piv", 1)
        np.testing.assert_allclose(mu.transform_points(W, np.array([[0, -1, 0.0]]))[0], [0, -1, 0], atol=1e-9)

    def test_constraints(self):
        sc = Scene({"nodes": [
            {"name": "tgt", "type": "locator", "translate": [0, 0, -5]},
            {"name": "aimer", "type": "locator", "constraints": [{"type": "aim", "target": "tgt",
                                                                  "aim_axis": "-z"}]},
            {"name": "follower", "type": "locator", "constraints": [{"type": "point", "target": "tgt"}]},
        ]})
        W = sc.world_matrix("aimer", 1)
        fwd = -W[:3, 2] / np.linalg.norm(W[:3, 2])
        np.testing.assert_allclose(fwd, [0, 0, -1], atol=1e-6)
        np.testing.assert_allclose(sc.world_matrix("follower", 1)[:3, 3], [0, 0, -5], atol=1e-9)

    def test_example_scenes_validate(self):
        root = os.path.join(os.path.dirname(__file__), "..", "examples")
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".json"):
                sc = an.load_scene(os.path.join(root, fn))
                problems = sc.validate(frames=[sc.frame_start])
                errors = [p for p in (problems or []) if str(p).lower().startswith("error")]
                self.assertEqual(errors, [], fn)

    def test_save_roundtrip(self):
        sc = Scene({"nodes": [{"name": "ball", "type": "mesh", "primitive": "sphere",
                               "translate": {"keys": [[1, [0, 0, 0]], [10, [0, 5, 0]]]}}]})
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s.json")
            sc.save(p)
            sc2 = an.load_scene(p)
        np.testing.assert_allclose(sc2.world_matrix("ball", 10)[:3, 3], [0, 5, 0], atol=1e-9)


class TestRig(unittest.TestCase):
    SCENE = {"nodes": [
        {"name": "shoulder", "type": "joint", "translate": [0, 0, 0]},
        {"name": "elbow", "type": "joint", "parent": "shoulder", "translate": [0, 1.5, 0]},
        {"name": "wrist", "type": "joint", "parent": "elbow", "translate": [0, 1.5, 0]},
        {"name": "target", "type": "locator", "translate": [1.2, 1.4, 0.5]},
        {"name": "pole", "type": "locator", "translate": [0, 1, 3]},
        {"name": "arm", "type": "mesh", "primitive": "cylinder", "radius": 0.15, "height": 3.0,
         "height_segments": 24, "translate": [0, 1.5, 0], "skin": {"root": "shoulder", "method": "dq"}},
    ], "ik": [{"name": "arm_ik", "start": "shoulder", "end": "wrist", "target": "target", "pole": "pole"}]}

    def test_ik_reaches_target(self):
        for solver in ("two_bone", "fabrik", "ccd"):
            data = dict(self.SCENE)
            data["ik"] = [dict(self.SCENE["ik"][0], solver=solver)]
            sc = Scene(data)
            st = sc.state(1)
            np.testing.assert_allclose(st.world("wrist")[:3, 3], [1.2, 1.4, 0.5], atol=2e-3, err_msg=solver)

    def test_skin_follows_joints(self):
        sc = Scene(self.SCENE)
        mesh = sc.world_mesh("arm", 1)
        # the top of the arm ends near the wrist (the IK target)
        top = mesh.verts[np.argmax(np.linalg.norm(mesh.verts - [0, 0, 0], axis=1))]
        self.assertLess(np.linalg.norm(top - np.array([1.2, 1.4, 0.5])), 0.4)


class TestDynamics(unittest.TestCase):
    def test_box_stack_stable(self):
        nodes = [{"name": "ground", "type": "mesh", "primitive": "plane", "size": 20}]
        for i in range(3):
            nodes.append({"name": f"box{i}", "type": "mesh", "primitive": "cube", "size": 1,
                          "translate": [0, 0.5 + i * 1.0, 0], "rigid_body": {"mass": 1}})
        sc = Scene({"settings": {"frame_range": [1, 72]}, "nodes": nodes})
        top = sc.world_matrix("box2", 72)[:3, 3]
        self.assertAlmostEqual(float(top[1]), 2.5, delta=0.1)
        self.assertLess(float(np.hypot(top[0], top[2])), 0.1)

    def test_falling_ball(self):
        sc = Scene({"settings": {"frame_range": [1, 48]}, "nodes": [
            {"name": "ball", "type": "mesh", "primitive": "sphere", "radius": 0.5, "translate": [0, 5, 0],
             "rigid_body": {"shape": "sphere", "restitution": 0.0}}]})
        y12 = sc.world_matrix("ball", 12)[1, 3]
        self.assertLess(y12, 5.0)
        y48 = sc.world_matrix("ball", 48)[1, 3]
        self.assertAlmostEqual(float(y48), 0.5, delta=0.1)


@unittest.skipUnless(os.environ.get("ANIMORA_SKIP_RENDER") is None, "render tests disabled")
class TestRender(unittest.TestCase):
    def test_small_render(self):
        sc = Scene({"settings": {"resolution": [64, 36], "samples": 8},
                    "nodes": [
                        {"name": "cam", "type": "camera", "translate": [0, 1, 5], "look_at": [0, 0.5, 0]},
                        {"name": "ball", "type": "mesh", "primitive": "sphere", "radius": 0.5,
                         "translate": [0, 0.5, 0], "material": {"preset": "plastic", "base_color": [1, 0.2, 0.1]}},
                        {"name": "floor", "type": "mesh", "primitive": "plane", "size": 10},
                        {"name": "key", "type": "light", "light": "sun", "rotate": [-45, 30, 0]}]})
        r = an.render_frame(sc, 1)
        img = r["display"]
        self.assertEqual(img.shape[:2], (36, 64))
        self.assertTrue(np.isfinite(img).all())
        # the ball (centre) is redder than it is blue
        c = img[18, 32]
        self.assertGreater(c[0], c[2])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "f.png")
            an.save_frame(r, p, sc)
            self.assertTrue(os.path.getsize(p) > 100)


if __name__ == "__main__":
    unittest.main()
