"""Materials, colours, textures and presets.

A material is a *principled* (Disney/OpenPBR-style) surface. Every scalar / colour attribute
is animatable and most accept a texture instead of a constant.

Material attributes (defaults)
------------------------------
=======================  ===========  ==========================================================
attribute                default      meaning
=======================  ===========  ==========================================================
``base_color``           0.8 grey     diffuse albedo / metal reflectance (colour or texture)
``metallic``             0            0 dielectric ... 1 metal (scalar or texture)
``roughness``            0.5          microfacet roughness (scalar or texture)
``specular``             0.5          dielectric specular level (0.5 = 4% F0)
``specular_tint``        0            tint dielectric specular towards base colour
``ior``                  1.5          index of refraction (glass 1.5, water 1.33, diamond 2.42)
``transmission``         0            0 opaque ... 1 fully refractive (glass)
``transmission_roughness`` = roughness  separate roughness for refraction (frosted glass)
``attenuation_color``    white        Beer-Lambert absorption colour for transmissive volumes
``attenuation_distance`` 0 (off)      distance at which light is tinted to attenuation_color
``thin_walled``          false        treat transmissive surfaces as thin sheets (bubbles, leaves)
``subsurface``           0            subsurface scattering weight (skin, wax, marble, milk)
``subsurface_radius``    [1,.2,.1]    per-channel mean free path (relative)
``subsurface_scale``     0.05         multiplier for subsurface_radius in scene units
``clearcoat``            0            secondary glossy coat (car paint, varnish)
``clearcoat_roughness``  0.03
``sheen``                0            grazing retro-reflection for cloth/velvet
``sheen_tint``           0.5
``emission``             black        emitted radiance colour (or texture)
``emission_strength``    0            multiplier - an emissive surface becomes a mesh light
``emission_one_sided``   false        emit from the front face only
``alpha``                1            cut-out opacity (scalar or texture; RGBA images use alpha)
``normal_map``           none         tangent-space normal map texture
``normal_strength``      1
``bump_map``             none         height texture
``bump_strength``        1            bump amplitude
``shadow_catcher``       false        invisible surface that only receives shadows (compositing)
=======================  ===========  ==========================================================

Texture slots: ``base_color``, ``roughness``, ``metallic``, ``emission``, ``alpha`` and
``transmission`` can each be given a texture spec instead of a value (a constant factor can
then be supplied as ``<name>_factor``), or use the explicit ``<name>_map`` keys.
``normal_map`` and ``bump_map`` are texture-only.

Texture specs
-------------
* ``"textures/brick.png"`` - an image file path (``.png/.jpg/.exr/.hdr/...``)
* ``{"type": "image", "file": "x.png", "colorspace": "srgb"|"linear", "wrap": "repeat"|"clamp",
  "interpolation": "linear"|"closest", "scale": [su, sv], "offset": [ou, ov], "rotation": deg}``
* procedural: ``{"type": T, "color1": c, "color2": c, "scale": s, "space": "uv"|"object"|"world",
  ...type-specific params}``

=============  =============================================================================
type           parameters
=============  =============================================================================
``checker``    ``scale`` (checks per unit)
``noise``      ``octaves`` (5), ``gain`` (0.5), ``distortion`` (0), ``contrast`` (1)
``marble``     ``octaves`` (6), ``power`` (1), ``distortion`` (5)
``wood``       ``rings`` (8 ring frequency), ``turbulence`` (1.5)
``grid``       ``line_width`` (0.05)
``gradient``   ``axis`` (0 x, 1 y, 2 z), ``low`` (0), ``high`` (1)
``voronoi``    ``mode`` ("cells" | "edges" | "distance"), ``sharpness`` (4)
``brick``      ``mortar`` (0.03), ``ratio`` (2 width/height), ``variation`` (0)
``stripes``    ``axis`` (0), ``softness`` (0)
``constant``   ``color1``
=============  =============================================================================

Colours
-------
Colours may be ``[r, g, b]`` (linear, 0-1), ``"#rrggbb"`` / ``"#rgb"`` (sRGB hex, converted to
linear), a CSS colour name (``"tomato"``, ``"skyblue"`` ...), ``"kelvin:3200"`` / ``"3200K"``
(black-body temperature), ``"srgb(255, 128, 0)"``, ``"hsv(h, s, v)"`` (h in degrees) or a single
number (grey).
"""
from __future__ import annotations

import colorsys
import math
import re
from pathlib import Path

import numpy as np

from .params import Param
from .render import native as nat

# ----------------------------------------------------------------------------- colours
NAMED_COLORS = {
    "black": "#000000", "white": "#ffffff", "red": "#ff0000", "green": "#00ff00", "blue": "#0000ff",
    "yellow": "#ffff00", "cyan": "#00ffff", "magenta": "#ff00ff", "grey": "#808080", "gray": "#808080",
    "lightgrey": "#d3d3d3", "lightgray": "#d3d3d3", "darkgrey": "#404040", "darkgray": "#404040",
    "orange": "#ffa500", "purple": "#800080", "pink": "#ffc0cb", "brown": "#8b4513", "gold": "#ffd700",
    "silver": "#c0c0c0", "navy": "#000080", "teal": "#008080", "olive": "#808000", "maroon": "#800000",
    "lime": "#32cd32", "coral": "#ff7f50", "tomato": "#ff6347", "salmon": "#fa8072", "crimson": "#dc143c",
    "skyblue": "#87ceeb", "steelblue": "#4682b4", "royalblue": "#4169e1", "turquoise": "#40e0d0",
    "violet": "#ee82ee", "indigo": "#4b0082", "beige": "#f5f5dc", "ivory": "#fffff0", "khaki": "#f0e68c",
    "chocolate": "#d2691e", "tan": "#d2b48c", "forestgreen": "#228b22", "seagreen": "#2e8b57",
    "midnightblue": "#191970", "slategray": "#708090", "slategrey": "#708090", "lavender": "#e6e6fa",
    "mint": "#98ff98", "peach": "#ffcba4", "charcoal": "#36454f", "cream": "#fffdd0", "sand": "#c2b280",
    "rust": "#b7410e", "emerald": "#50c878", "ruby": "#e0115f", "sapphire": "#0f52ba", "amber": "#ffbf00",
}


def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def kelvin_to_rgb(k):
    """Approximate linear RGB (normalised to max 1) of a black body at ``k`` Kelvin (1000-40000)."""
    t = float(np.clip(k, 1000, 40000)) / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
        b = 0.0 if t <= 19 else 138.5177312231 * math.log(t - 10) - 305.0447927307
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
        b = 255.0
    rgb = _srgb_to_lin(np.clip([r, g, b], 0, 255) / 255.0)
    return rgb / max(rgb.max(), 1e-9)


def parse_color(v, default=(0.8, 0.8, 0.8), working=True):
    """Parse any supported colour notation into a linear RGB numpy array (3,) in the current
    working space (see :mod:`animora.color`). ``working=False`` returns linear Rec.709 instead.

    Plain numeric lists are interpreted in ``settings.color_input`` (default ``lin_srgb``);
    ``{"rgb": [...], "space": "acescg"}`` names the space explicitly; hex / CSS names / ``srgb()``
    / ``hsv()`` / kelvin are sRGB-referred.
    """
    from . import color as cm
    if isinstance(v, dict) and ("rgb" in v or "value" in v or "color" in v):
        raw = v.get("rgb", v.get("value", v.get("color")))
        sp = v.get("space", v.get("colorspace", "lin_srgb"))
        a = _parse_color_raw(raw, default)
        if isinstance(raw, str):  # strings are always sRGB-referred: already lin_srgb
            sp = "lin_srgb"
        a = np.asarray(cm.convert(a, sp, "lin_srgb"), float)
        return cm.to_working(a, "lin_srgb") if working else a
    a = _parse_color_raw(v, default)
    if isinstance(v, (list, tuple, np.ndarray, int, float)) and not isinstance(v, bool) and v is not None:
        src = cm.default_input_space()
        if src != "lin_srgb":
            a = np.asarray(cm.convert(a, src, "lin_srgb"), float)
    if not working:
        return a
    return np.asarray(cm.to_working(a, "lin_srgb"), float)


def _parse_color_raw(v, default=(0.8, 0.8, 0.8)):
    """parse_color without colour management (result: linear Rec.709 / raw numbers)."""
    if v is None:
        return np.asarray(default, float)
    if isinstance(v, np.ndarray):
        a = v.astype(float).ravel()
        return np.repeat(a[:1], 3) if a.size == 1 else a[:3].copy()
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return np.full(3, float(v))
    if isinstance(v, (list, tuple)):
        a = np.asarray(v, float).ravel()
        return np.repeat(a[:1], 3) if a.size == 1 else a[:3].copy()
    if isinstance(v, str):
        s = v.strip().lower().replace(" ", "")
        if s in NAMED_COLORS:
            s = NAMED_COLORS[s]
        if s.startswith("#"):
            h = s[1:]
            if len(h) in (3, 4):
                h = "".join(ch * 2 for ch in h[:3])
            rgb = np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], float) / 255.0
            return _srgb_to_lin(rgb)
        m = re.fullmatch(r"(?:kelvin:|k:)?([0-9.]+)k?", s)
        if m and (s.endswith("k") or s.startswith("k")):
            return kelvin_to_rgb(float(m.group(1)))
        m = re.fullmatch(r"srgb\(([^)]*)\)", s)
        if m:
            vals = np.array([float(x) for x in m.group(1).split(",")], float)
            if vals.max() > 1.0:
                vals = vals / 255.0
            return _srgb_to_lin(vals[:3])
        m = re.fullmatch(r"hsv\(([^)]*)\)", s)
        if m:
            h, sat, val = (float(x) for x in m.group(1).split(","))
            return _srgb_to_lin(np.array(colorsys.hsv_to_rgb((h / 360.0) % 1.0, sat, val)))
        m = re.fullmatch(r"(?:rgb|linear)\(([^)]*)\)", s)
        if m:
            return np.array([float(x) for x in m.group(1).split(",")][:3], float)
    raise ValueError(f"cannot parse colour {v!r}")


def is_color_string(v):
    try:
        _parse_color_raw(v)
        return isinstance(v, str)
    except Exception:
        return False


# ----------------------------------------------------------------------------- presets
#: Physically-motivated material presets. ``preset`` values are a starting point; any attribute
#: given alongside overrides it. (Metal colours are measured F0 reflectances, linear.)
PRESETS = {
    "default": {},
    "matte": {"roughness": 0.9, "specular": 0.3},
    "clay": {"base_color": [0.62, 0.6, 0.58], "roughness": 0.85, "specular": 0.25},
    "plastic": {"base_color": [0.6, 0.05, 0.05], "roughness": 0.3, "specular": 0.5},
    "glossy_plastic": {"base_color": [0.05, 0.2, 0.6], "roughness": 0.08, "clearcoat": 0.5},
    "rubber": {"base_color": [0.03, 0.03, 0.03], "roughness": 0.75, "specular": 0.35},
    "ceramic": {"base_color": [0.9, 0.9, 0.88], "roughness": 0.15, "clearcoat": 1.0, "clearcoat_roughness": 0.02},
    "porcelain": {"base_color": [0.92, 0.92, 0.9], "roughness": 0.2, "subsurface": 0.3,
                  "subsurface_radius": [1, 0.9, 0.8], "clearcoat": 1.0},
    "metal": {"base_color": [0.91, 0.92, 0.92], "metallic": 1.0, "roughness": 0.25},
    "chrome": {"base_color": [0.55, 0.56, 0.55], "metallic": 1.0, "roughness": 0.03},
    "mirror": {"base_color": [0.97, 0.97, 0.97], "metallic": 1.0, "roughness": 0.0},
    "gold": {"base_color": [1.0, 0.71, 0.29], "metallic": 1.0, "roughness": 0.18},
    "silver": {"base_color": [0.97, 0.96, 0.91], "metallic": 1.0, "roughness": 0.15},
    "copper": {"base_color": [0.95, 0.64, 0.54], "metallic": 1.0, "roughness": 0.2},
    "brass": {"base_color": [0.91, 0.78, 0.42], "metallic": 1.0, "roughness": 0.22},
    "bronze": {"base_color": [0.8, 0.5, 0.3], "metallic": 1.0, "roughness": 0.3},
    "aluminum": {"base_color": [0.91, 0.92, 0.92], "metallic": 1.0, "roughness": 0.35},
    "aluminium": {"base_color": [0.91, 0.92, 0.92], "metallic": 1.0, "roughness": 0.35},
    "brushed_metal": {"base_color": [0.9, 0.9, 0.9], "metallic": 1.0, "roughness": 0.45},
    "iron": {"base_color": [0.56, 0.57, 0.58], "metallic": 1.0, "roughness": 0.45},
    "titanium": {"base_color": [0.54, 0.5, 0.45], "metallic": 1.0, "roughness": 0.3},
    "glass": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.0, "ior": 1.5, "specular": 0.5},
    "frosted_glass": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.25, "ior": 1.5},
    "tinted_glass": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.0, "ior": 1.52,
                     "attenuation_color": [0.2, 0.6, 0.35], "attenuation_distance": 0.5},
    "water": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.0, "ior": 1.333,
              "attenuation_color": [0.75, 0.92, 0.95], "attenuation_distance": 3.0},
    "ice": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.08, "ior": 1.31,
            "attenuation_color": [0.85, 0.95, 1.0], "attenuation_distance": 1.0},
    "diamond": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.0, "ior": 2.42},
    "bubble": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.0, "ior": 1.33, "thin_walled": True},
    "honey": {"base_color": [1, 1, 1], "transmission": 1.0, "roughness": 0.05, "ior": 1.5,
              "attenuation_color": [0.9, 0.45, 0.05], "attenuation_distance": 0.15},
    "skin": {"base_color": [0.8, 0.55, 0.45], "roughness": 0.45, "subsurface": 0.6,
             "subsurface_radius": [1.0, 0.35, 0.2], "subsurface_scale": 0.03, "sheen": 0.1},
    "wax": {"base_color": [0.9, 0.82, 0.65], "roughness": 0.35, "subsurface": 0.8,
            "subsurface_radius": [1.0, 0.7, 0.4], "subsurface_scale": 0.05},
    "jade": {"base_color": [0.25, 0.65, 0.4], "roughness": 0.15, "subsurface": 0.7,
             "subsurface_radius": [0.4, 1.0, 0.6], "subsurface_scale": 0.05, "clearcoat": 0.6},
    "milk": {"base_color": [0.95, 0.95, 0.92], "roughness": 0.15, "subsurface": 1.0,
             "subsurface_radius": [1.0, 0.9, 0.7], "subsurface_scale": 0.08},
    "marble": {"base_color": {"type": "marble", "color1": [0.93, 0.92, 0.9], "color2": [0.35, 0.35, 0.38],
                              "scale": 1.5, "space": "object"},
               "roughness": 0.12, "subsurface": 0.4, "subsurface_radius": [1, 0.95, 0.9], "clearcoat": 0.4},
    "wood": {"base_color": {"type": "wood", "color1": [0.45, 0.25, 0.12], "color2": [0.22, 0.1, 0.04],
                            "scale": 1.0, "space": "object"},
             "roughness": 0.45, "clearcoat": 0.25, "clearcoat_roughness": 0.1},
    "car_paint": {"base_color": [0.6, 0.02, 0.02], "metallic": 0.6, "roughness": 0.35, "clearcoat": 1.0,
                  "clearcoat_roughness": 0.02},
    "velvet": {"base_color": [0.35, 0.02, 0.1], "roughness": 0.9, "sheen": 1.0, "sheen_tint": 0.6, "specular": 0.2},
    "fabric": {"base_color": [0.4, 0.42, 0.5], "roughness": 0.85, "sheen": 0.6, "specular": 0.25},
    "hair": {"base_color": {"type": "gradient", "space": "uv", "axis": 1, "color1": [0.05, 0.025, 0.012],
                            "color2": [0.22, 0.12, 0.05]}, "roughness": 0.35, "specular": 0.6, "sheen": 0.4},
    "fur": {"base_color": {"type": "gradient", "space": "uv", "axis": 1, "color1": [0.12, 0.07, 0.04],
                           "color2": [0.75, 0.6, 0.42]}, "roughness": 0.55, "specular": 0.35, "sheen": 0.8,
            "sheen_tint": 0.5},
    "leather": {"base_color": [0.25, 0.12, 0.06], "roughness": 0.55, "clearcoat": 0.2,
                "bump_map": {"type": "noise", "scale": 40, "space": "object"}, "bump_strength": 0.3},
    "concrete": {"base_color": {"type": "noise", "color1": [0.45, 0.44, 0.42], "color2": [0.62, 0.61, 0.58],
                                "scale": 6, "space": "object", "octaves": 6},
                 "roughness": 0.9, "bump_map": {"type": "noise", "scale": 30, "space": "object"},
                 "bump_strength": 0.4},
    "checker": {"base_color": {"type": "checker", "color1": [0.9, 0.9, 0.9], "color2": [0.1, 0.1, 0.1],
                               "scale": 8}, "roughness": 0.5},
    "studio_floor": {"base_color": [0.18, 0.18, 0.19], "roughness": 0.35, "specular": 0.5},
    "emissive": {"base_color": [0, 0, 0], "emission": [1, 1, 1], "emission_strength": 10.0},
    "light": {"base_color": [0, 0, 0], "emission": [1, 1, 1], "emission_strength": 10.0},
    "neon": {"base_color": [0, 0, 0], "emission": [1.0, 0.1, 0.6], "emission_strength": 25.0},
    "shadow_catcher": {"shadow_catcher": True, "base_color": [0.8, 0.8, 0.8], "roughness": 1.0},
    "holdout": {"shadow_catcher": True, "base_color": [0.8, 0.8, 0.8], "roughness": 1.0},
    "toon": {"base_color": [0.9, 0.5, 0.2], "roughness": 1.0, "specular": 0.0},
}

SCALAR_ATTRS = ["metallic", "roughness", "ior", "transmission", "specular", "specular_tint", "clearcoat",
                "clearcoat_roughness", "sheen", "sheen_tint", "subsurface", "subsurface_scale",
                "emission_strength", "alpha", "normal_strength", "bump_strength", "transmission_roughness",
                "attenuation_distance"]
COLOR_ATTRS = ["base_color", "subsurface_radius", "emission", "attenuation_color"]
BOOL_ATTRS = ["thin_walled", "emission_one_sided", "shadow_catcher"]
TEXTURE_SLOTS = {"base_color": "tex_base_color", "roughness": "tex_roughness", "metallic": "tex_metallic",
                 "normal": "tex_normal", "emission": "tex_emission", "alpha": "tex_alpha", "bump": "tex_bump",
                 "transmission": "tex_transmission"}

_DEFAULTS = {"base_color": [0.8, 0.8, 0.8], "metallic": 0.0, "roughness": 0.5, "ior": 1.5, "transmission": 0.0,
             "specular": 0.5, "specular_tint": 0.0, "clearcoat": 0.0, "clearcoat_roughness": 0.03, "sheen": 0.0,
             "sheen_tint": 0.5, "subsurface": 0.0, "subsurface_radius": [1.0, 0.2, 0.1], "subsurface_scale": 0.05,
             "emission": [0.0, 0.0, 0.0], "emission_strength": 0.0, "alpha": 1.0, "normal_strength": 1.0,
             "bump_strength": 1.0, "transmission_roughness": -1.0, "attenuation_color": [1.0, 1.0, 1.0],
             "attenuation_distance": 0.0}


def _is_texture_spec(v):
    if isinstance(v, dict):
        return "type" in v or "file" in v
    if isinstance(v, str) and not v.startswith("="):
        return Path(v).suffix.lower() in {".png", ".jpg", ".jpeg", ".exr", ".hdr", ".tif", ".tiff", ".bmp",
                                          ".tga", ".pfm", ".webp"}
    return False


class Material:
    """A principled surface material. Construct from a spec dict or keywords::

        Material(preset="gold", roughness=0.3)
        Material({"base_color": "#ff8800", "roughness": {"keys": [[1, 0.1], [48, 0.8]]}})
    """

    def __init__(self, spec=None, name="material", **kw):
        spec = dict(spec or {})
        spec.update(kw)
        self.name = spec.pop("name", name)
        self.preset = spec.pop("preset", None)
        base = {}
        if self.preset is not None:
            if self.preset not in PRESETS:
                raise ValueError(f"unknown material preset {self.preset!r}; available: {sorted(PRESETS)}")
            base.update(PRESETS[self.preset])
        base.update(spec)
        self.values: dict[str, Param] = {}
        self.textures: dict[str, object] = {}
        self.flags: dict[str, bool] = {}
        self._spec = {"preset": self.preset, **spec} if self.preset else dict(spec)
        for k, v in base.items():
            self._set(k, v)

    def _set(self, k, v):
        if k in BOOL_ATTRS:
            self.flags[k] = bool(v)
            return
        if k.endswith("_map"):
            slot = k[:-4]
            if slot not in TEXTURE_SLOTS:
                raise ValueError(f"unknown texture slot {k!r}; available: {[s + '_map' for s in TEXTURE_SLOTS]}")
            self.textures[slot] = v
            return
        if k.endswith("_factor"):
            k = k[:-7]
        if k in TEXTURE_SLOTS and _is_texture_spec(v):
            self.textures[k] = v
            if k in COLOR_ATTRS and k not in self.values:
                self.values[k] = Param([1.0, 1.0, 1.0])
            elif k == "emission" or k in ("roughness", "metallic", "alpha", "transmission"):
                self.values.setdefault(k, Param(1.0))
            return
        if k in COLOR_ATTRS:
            if isinstance(v, str) and not v.startswith("="):
                v = parse_color(v).tolist()
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                v = [float(v)] * 3
            self.values[k] = Param(v)
            return
        if k in SCALAR_ATTRS:
            self.values[k] = Param(v)
            return
        if k in ("description", "comment"):
            return
        raise ValueError(f"unknown material attribute {k!r} in material {self.name!r}")

    # ------------------------------------------------------------------
    def is_animated(self):
        return any(p.is_animated() for p in self.values.values())

    def evaluate(self, frame=0.0, ctx=None):
        out = {}
        for k in SCALAR_ATTRS + COLOR_ATTRS:
            p = self.values.get(k)
            out[k] = p.evaluate(frame, ctx) if p is not None else _DEFAULTS[k]
        return out

    def to_native(self, textable, frame=0.0, ctx=None, base_dir="."):
        """Build an :class:`AnMaterial`, registering textures in ``textable``."""
        m = nat.default_material()
        vals = self.evaluate(frame, ctx)
        for k in COLOR_ATTRS:
            nat.set_vec(getattr(m, k), parse_color(vals[k]))
        for k in SCALAR_ATTRS:
            setattr(m, k, float(np.asarray(vals[k], float).ravel()[0]))
        for k in BOOL_ATTRS:
            setattr(m, k, 1 if self.flags.get(k) else 0)
        for slot, spec in self.textures.items():
            linear = slot not in ("base_color", "emission")
            setattr(m, TEXTURE_SLOTS[slot], textable.add(spec, linear_default=linear, frame=frame, ctx=ctx,
                                                         base_dir=base_dir))
        return m

    def to_spec(self):
        return dict(self._spec)

    def __repr__(self):
        return f"Material({self.name!r}, preset={self.preset!r})"


def material_from_spec(spec, name="material"):
    if isinstance(spec, Material):
        return spec
    if isinstance(spec, str):
        if spec in PRESETS:
            return Material(preset=spec, name=name)
        if spec.startswith("#") or spec.lower().replace(" ", "") in NAMED_COLORS:
            return Material(base_color=spec, name=name)
        raise ValueError(f"unknown material {spec!r}")
    return Material(spec, name=name)


# ----------------------------------------------------------------------------- textures
_SPACE = {"uv": 0, "object": 1, "orco": 1, "generated": 1, "local": 1, "world": 2}
_VORONOI_MODE = {"cells": 0, "cell": 0, "edges": 1, "edge": 1, "distance": 2, "f1": 2}
_IMAGE_CACHE: dict = {}


def load_texture_image(path, linear, space=None):
    """Load a texture as float RGB(A). ``linear`` = data texture (no conversion). ``space`` names the
    file's colour space (default sRGB for 8-bit colour textures, lin_srgb for HDR files); colour
    textures are converted into the current working space."""
    from . import color as cm
    work = cm.working_space()
    key = (str(Path(path).resolve()), bool(linear), space, work)
    img = _IMAGE_CACHE.get(key)
    if img is None:
        from .io.images import read_image
        if linear or space is None or cm.canonical(space) == "srgb":
            img = read_image(path, linear=not linear)  # colour textures are decoded sRGB -> linear
            src = "raw" if linear else "lin_srgb"
        else:
            img = read_image(path, linear=True)
            src = cm.canonical(space)
        if img.shape[2] == 2:
            img = np.concatenate([np.repeat(img[..., :1], 3, 2), img[..., 1:]], 2)
        if src != "raw" and img.shape[2] >= 3 and src != work:
            img = np.ascontiguousarray(cm.convert(img, src, work), np.float32)
        _IMAGE_CACHE[key] = img
    return img


class TextureTable:
    """Accumulates :class:`AnTexture` records and their pixel data for one render."""

    def __init__(self):
        self.textures = []
        self.chunks = []
        self.size = 0
        self._image_offsets = {}
        self._dedupe = {}

    def add_image_data(self, img, key=None):
        img = np.ascontiguousarray(img, np.float32)
        if key is not None and key in self._image_offsets:
            return self._image_offsets[key]
        off = self.size
        self.chunks.append(img.ravel())
        self.size += img.size
        if key is not None:
            self._image_offsets[key] = off
        return off

    def add(self, spec, linear_default=False, frame=0.0, ctx=None, base_dir="."):
        """Register a texture spec, returning its index."""
        if isinstance(spec, str):
            spec = {"type": "image", "file": spec}
        spec = dict(spec)
        if "file" in spec and "type" not in spec:
            spec["type"] = "image"
        t = nat.AnTexture()
        typ = spec.get("type", "image")
        if typ not in nat.TEXTURE_TYPES:
            raise ValueError(f"unknown texture type {typ!r}; available: {sorted(nat.TEXTURE_TYPES)}")
        t.type = nat.TEXTURE_TYPES[typ]
        ev = lambda v: Param(v).evaluate(frame, ctx) if not isinstance(v, (int, float)) else v
        t.space = _SPACE[spec.get("space", "uv" if typ in ("image", "checker", "grid", "brick") else "object")]
        t.wrap = 0 if spec.get("wrap", "repeat") == "repeat" else 1
        t.interp = 1 if spec.get("interpolation", "linear") in ("closest", "nearest") else 0
        nat.set_vec(t.color1, parse_color(ev(spec.get("color1", [0.0, 0.0, 0.0] if typ != "constant" else 1.0))))
        nat.set_vec(t.color2, parse_color(ev(spec.get("color2", [1.0, 1.0, 1.0]))))
        sc = ev(spec.get("scale", 1.0))
        sca = np.asarray(sc, float).ravel()
        if t.space == 0:
            uvs = np.resize(sca, 2) if typ == "image" else np.resize(sca, 2)
            nat.set_vec(t.uv_scale, uvs)
            t.scale = float(sca[0]) if typ != "image" else 1.0
            if typ != "image":
                # procedural in UV space: scale applied via t.scale only
                nat.set_vec(t.uv_scale, (1.0, 1.0))
        else:
            nat.set_vec(t.uv_scale, (1.0, 1.0))
            t.scale = float(sca[0])
        nat.set_vec(t.uv_offset, np.resize(np.asarray(ev(spec.get("offset", [0.0, 0.0])), float), 2))
        t.uv_rotation = math.radians(float(ev(spec.get("rotation", 0.0))))
        p = [0.0] * 6
        if typ == "noise":
            p = [spec.get("octaves", 5), spec.get("gain", 0.5), spec.get("distortion", 0.0), spec.get("contrast", 1.0), 0, 0]
        elif typ == "marble":
            p = [spec.get("octaves", 6), spec.get("power", 1.0), spec.get("distortion", 5.0), 0, 0, 0]
        elif typ == "wood":
            p = [spec.get("rings", 8.0), 0, spec.get("turbulence", 1.5), 0, 0, 0]
        elif typ == "grid":
            p = [spec.get("line_width", 0.05), 0, 0, 0, 0, 0]
        elif typ == "gradient":
            ax = spec.get("axis", 0)
            p = [{"x": 0, "y": 1, "z": 2}.get(ax, ax), spec.get("low", 0.0), spec.get("high", 1.0), 0, 0, 0]
        elif typ == "voronoi":
            md = spec.get("mode", "cells")
            p = [_VORONOI_MODE.get(md, md), spec.get("sharpness", 4.0), 0, 0, 0, 0]
        elif typ == "brick":
            p = [spec.get("mortar", 0.03), spec.get("ratio", 2.0), spec.get("variation", 0.0), 0, 0, 0]
        elif typ == "stripes":
            ax = spec.get("axis", 0)
            p = [{"x": 0, "y": 1, "z": 2}.get(ax, ax), spec.get("softness", 0.0), 0, 0, 0, 0]
        for i, v in enumerate(p):
            t.params[i] = float(ev(v))
        if typ == "image":
            f = Path(spec["file"])
            if not f.is_absolute():
                f = Path(base_dir) / f
            if not f.exists():
                raise FileNotFoundError(f"texture image not found: {f}")
            from . import color as cm
            cs = spec.get("colorspace")
            hdr = f.suffix.lower() in (".exr", ".hdr", ".pfm")
            space = None
            if cs is None:
                linear = linear_default
                if hdr and not linear:
                    space = "lin_srgb"
            else:
                space = cm.canonical(cs)
                linear = space == "raw"
            if hdr and space == "srgb":
                space = "lin_srgb"
            if space is None and not linear and hdr:
                space = "lin_srgb"
            if linear:
                space = None
            img = load_texture_image(f, linear, space)
            t.width, t.height, t.channels = img.shape[1], img.shape[0], img.shape[2]
            t.offset = self.add_image_data(img, key=(str(f.resolve()), linear, space, cm.working_space()))
        self.textures.append(t)
        return len(self.textures) - 1

    def add_image_array(self, img, wrap="repeat", interp="linear"):
        """Register a raw float image (H,W,C), e.g. an environment map. Returns the index."""
        t = nat.AnTexture()
        t.type = 0
        t.width, t.height, t.channels = img.shape[1], img.shape[0], img.shape[2]
        t.offset = self.add_image_data(img, key=("array", id(img)))
        t.wrap = 0 if wrap == "repeat" else 1
        t.interp = 1 if interp in ("closest", "nearest") else 0
        nat.set_vec(t.uv_scale, (1, 1))
        t.scale = 1.0
        self.textures.append(t)
        return len(self.textures) - 1

    def add_constant(self, color):
        t = nat.AnTexture()
        t.type = nat.TEXTURE_TYPES["constant"]
        nat.set_vec(t.color1, parse_color(color))
        nat.set_vec(t.uv_scale, (1, 1))
        t.scale = 1.0
        self.textures.append(t)
        return len(self.textures) - 1

    def data(self):
        if not self.chunks:
            return np.zeros(1, np.float32)
        return np.concatenate(self.chunks).astype(np.float32)
