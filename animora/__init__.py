"""Animora - a command-line 3D animation and path-traced rendering package.

Scenes are plain JSON (or Python) files. Everything runs headless: renders go to image
sequences and videos, previews are drawn in the terminal.

Quick start (Python)::

    import animora as an
    scene = an.load_scene("examples/bouncing_ball.json")
    result = an.render_frame(scene, 12)
    an.save_frame(result, "frame12.png", scene)
    an.render_sequence(scene, output="out/ball_####.png")
    an.encode("out/ball_####.png", "out/ball.mp4", fps=scene.fps)
"""
__version__ = "1.0.0"

from .scene import Scene, Node, SceneError, load_scene, load_scene_data, build_packed  # noqa: E402
from .materials import Material, PRESETS as MATERIAL_PRESETS  # noqa: E402
from .geometry import Mesh, make_primitive, merge, PRIMITIVES  # noqa: E402
from .curves import AnimCurve, Key, get_easing, EASINGS  # noqa: E402
from .params import Param  # noqa: E402
from .render.renderer import (render_frame, render_sequence, save_frame, frame_path,  # noqa: E402
                              resolution)
from .output import encode, encode_from_settings, probe  # noqa: E402
from .io.images import read_image, write_image  # noqa: E402
from .io.obj import read_mesh, write_mesh  # noqa: E402

__all__ = [
    "__version__", "Scene", "Node", "SceneError", "load_scene", "load_scene_data", "build_packed",
    "Material", "MATERIAL_PRESETS", "Mesh", "make_primitive", "merge", "PRIMITIVES",
    "AnimCurve", "Key", "get_easing", "EASINGS", "Param",
    "render_frame", "render_sequence", "save_frame", "frame_path", "resolution",
    "encode", "encode_from_settings", "probe", "read_image", "write_image", "read_mesh", "write_mesh",
]
