"""Colour management (an OCIO-style pipeline without the OCIO dependency).

Animora renders in a configurable **working space** (``settings.working_space``):
``lin_srgb`` (linear Rec.709 primaries, default) or ``acescg`` (ACES AP1, the VFX standard),
``lin_rec2020`` or ``lin_p3``. Every colour that enters the renderer -- material colours,
light colours, colour textures, HDRIs, the procedural sky -- is converted from its own
colour space into the working space; the display transform converts out again.

Colour spaces (``colorspace`` on textures / HDRIs, ``space`` on colour values,
``settings.display`` / ``settings.exr_space`` on output)::

    name              primaries   white  encoding        typical use
    lin_srgb          Rec.709     D65    linear          scene-linear default ("linear", "lin_rec709")
    srgb              Rec.709     D65    sRGB curve      8-bit textures, web images ("srgb_texture")
    rec709            Rec.709     D65    gamma 2.4       broadcast SDR display (BT.1886)
    acescg            ACES AP1    D60    linear          VFX working space
    acescct           ACES AP1    D60    ACEScct log     grading
    acescc            ACES AP1    D60    ACEScc log      grading
    aces2065          ACES AP0    D60    linear          archival / interchange EXR ("aces", "aces2065-1")
    lin_rec2020       Rec.2020    D65    linear
    rec2020           Rec.2020    D65    gamma 2.4       wide gamut SDR display
    lin_p3            P3          D65    linear
    display_p3        P3          D65    sRGB curve      Apple displays
    adobe_rgb         Adobe 1998  D65    gamma 2.2
    lin_adobe_rgb     Adobe 1998  D65    linear
    raw               --          --     none            data textures ("data", "non-color")

Chromatic adaptation between white points uses the Bradford transform.
"""
from __future__ import annotations

import numpy as np

_WHITES = {"D65": (0.3127, 0.3290), "D60": (0.32168, 0.33767), "D50": (0.3457, 0.3585)}
_PRIMARIES = {
    "rec709": ((0.64, 0.33), (0.30, 0.60), (0.15, 0.06), "D65"),
    "rec2020": ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046), "D65"),
    "p3": ((0.680, 0.320), (0.265, 0.690), (0.150, 0.060), "D65"),
    "ap1": ((0.713, 0.293), (0.165, 0.830), (0.128, 0.044), "D60"),
    "ap0": ((0.7347, 0.2653), (0.0, 1.0), (0.0001, -0.0770), "D60"),
    "adobe": ((0.64, 0.33), (0.21, 0.71), (0.15, 0.06), "D65"),
}

#: name -> (primaries, transfer)
SPACES = {
    "lin_srgb": ("rec709", "linear"), "srgb": ("rec709", "srgb"), "rec709": ("rec709", "gamma24"),
    "acescg": ("ap1", "linear"), "acescct": ("ap1", "acescct"), "acescc": ("ap1", "acescc"),
    "aces2065": ("ap0", "linear"),
    "lin_rec2020": ("rec2020", "linear"), "rec2020": ("rec2020", "gamma24"),
    "lin_p3": ("p3", "linear"), "display_p3": ("p3", "srgb"),
    "adobe_rgb": ("adobe", "gamma22"), "lin_adobe_rgb": ("adobe", "linear"),
    "raw": (None, "linear"),
}
ALIASES = {
    "linear": "lin_srgb", "lin_rec709": "lin_srgb", "scene_linear": "lin_srgb", "linear_srgb": "lin_srgb",
    "scene_linear_rec709": "lin_srgb", "srgb_texture": "srgb", "srgb_display": "srgb", "sRGB": "srgb",
    "aces": "aces2065", "aces2065-1": "aces2065", "aces2065_1": "aces2065", "ap0": "aces2065",
    "ap1": "acescg", "aces_cg": "acescg", "lin_ap1": "acescg",
    "rec.709": "rec709", "bt709": "rec709", "bt1886": "rec709",
    "rec.2020": "rec2020", "bt2020": "rec2020", "linear_rec2020": "lin_rec2020",
    "p3": "display_p3", "displayp3": "display_p3", "p3_d65": "display_p3", "lin_display_p3": "lin_p3",
    "linear_p3": "lin_p3", "adobergb": "adobe_rgb",
    "data": "raw", "non-color": "raw", "non_color": "raw", "noncolor": "raw", "utility_raw": "raw",
}
WORKING_SPACES = ("lin_srgb", "acescg", "lin_rec2020", "lin_p3", "lin_adobe_rgb", "aces2065")
DISPLAYS = ("srgb", "display_p3", "rec709", "rec2020", "adobe_rgb")


def canonical(name):
    """Normalise a colour-space name (raises ValueError for unknown names)."""
    if name is None:
        return "lin_srgb"
    n = str(name).strip()
    key = ALIASES.get(n, ALIASES.get(n.lower(), n.lower().replace(" ", "_").replace("-", "_")))
    key = ALIASES.get(key, key)
    if key not in SPACES:
        raise ValueError(f"unknown colour space {name!r}; available: {sorted(SPACES)}")
    return key


# ----------------------------------------------------------------------------- matrices
def _xy_to_xyz(x, y):
    return np.array([x / y, 1.0, (1 - x - y) / y])


def rgb_to_xyz_matrix(prim):
    r, g, b, w = _PRIMARIES[prim]
    P = np.stack([_xy_to_xyz(*r), _xy_to_xyz(*g), _xy_to_xyz(*b)], 1)
    W = _xy_to_xyz(*_WHITES[w])
    S = np.linalg.solve(P, W)
    return P * S[None, :]


_BRADFORD = np.array([[0.8951, 0.2664, -0.1614], [-0.7502, 1.7135, 0.0367], [0.0389, -0.0685, 1.0296]])


def adaptation_matrix(src_white, dst_white):
    if src_white == dst_white:
        return np.eye(3)
    ws = _BRADFORD @ _xy_to_xyz(*_WHITES[src_white])
    wd = _BRADFORD @ _xy_to_xyz(*_WHITES[dst_white])
    return np.linalg.inv(_BRADFORD) @ np.diag(wd / ws) @ _BRADFORD


_MCACHE: dict = {}


def primaries_matrix(src_prim, dst_prim):
    """3x3 matrix converting linear RGB in ``src_prim`` to linear RGB in ``dst_prim``."""
    if src_prim is None or dst_prim is None or src_prim == dst_prim:
        return np.eye(3)
    key = (src_prim, dst_prim)
    m = _MCACHE.get(key)
    if m is None:
        a = adaptation_matrix(_PRIMARIES[src_prim][3], _PRIMARIES[dst_prim][3])
        m = np.linalg.inv(rgb_to_xyz_matrix(dst_prim)) @ a @ rgb_to_xyz_matrix(src_prim)
        _MCACHE[key] = m
    return m


def matrix(src, dst):
    """Linear-to-linear 3x3 matrix between the primaries of colour spaces ``src`` and ``dst``."""
    return primaries_matrix(SPACES[canonical(src)][0], SPACES[canonical(dst)][0])


# ----------------------------------------------------------------------------- transfer functions
def _srgb_enc(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def _srgb_dec(x):
    x = np.maximum(x, 0.0)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


def _acescct_enc(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.0078125, 10.5402377416545 * x + 0.0729055341958355,
                    (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52)


def _acescct_dec(y):
    y = np.asarray(y, np.float64)
    return np.where(y <= 0.155251141552511, (y - 0.0729055341958355) / 10.5402377416545,
                    np.minimum(np.power(2.0, y * 17.52 - 9.72), 65504.0))


def _acescc_enc(x):
    x = np.asarray(x, np.float64)
    lo = (np.log2(2.0 ** -16) + 9.72) / 17.52
    mid = (np.log2(2.0 ** -16 + np.maximum(x, 0) * 0.5) + 9.72) / 17.52
    hi = (np.log2(np.maximum(x, 1e-10)) + 9.72) / 17.52
    return np.where(x <= 0, lo, np.where(x < 2.0 ** -15, mid, hi))


def _acescc_dec(y):
    y = np.asarray(y, np.float64)
    a = (2.0 ** (y * 17.52 - 9.72) - 2.0 ** -16) * 2.0
    b = np.minimum(2.0 ** (y * 17.52 - 9.72), 65504.0)
    return np.where(y < (9.72 - 15) / 17.52, a, b)


def _gamma(g):
    return (lambda x: np.power(np.maximum(x, 0.0), 1.0 / g), lambda x: np.power(np.maximum(x, 0.0), g))


_G22 = _gamma(563 / 256)
_G24 = _gamma(2.4)
TRANSFERS = {
    "linear": (lambda x: x, lambda x: x),
    "srgb": (_srgb_enc, _srgb_dec),
    "gamma22": _G22,
    "gamma24": _G24,
    "acescct": (_acescct_enc, _acescct_dec),
    "acescc": (_acescc_enc, _acescc_dec),
}


def encode(lin, space):
    """Apply the transfer function (OETF) of ``space`` to linear values."""
    return TRANSFERS[SPACES[canonical(space)][1]][0](lin)


def decode(enc, space):
    """Invert the transfer function of ``space`` (encoded -> linear)."""
    return TRANSFERS[SPACES[canonical(space)][1]][1](enc)


def convert(rgb, src, dst):
    """Convert an array (..., 3[+alpha]) from colour space ``src`` to ``dst``."""
    src, dst = canonical(src), canonical(dst)
    a = np.asarray(rgb, np.float64)
    if src == dst or src == "raw" or dst == "raw":
        return a.copy()
    extra = None
    if a.shape[-1] > 3:
        extra, a = a[..., 3:], a[..., :3]
    lin = decode(a, src)
    m = primaries_matrix(SPACES[src][0], SPACES[dst][0])
    if not np.allclose(m, np.eye(3)):
        lin = lin @ m.T
    out = encode(lin, dst)
    if extra is not None:
        out = np.concatenate([out, extra], -1)
    return out


def is_linear(space):
    return SPACES[canonical(space)][1] == "linear"


# ----------------------------------------------------------------------------- working space state
_WORKING = "lin_srgb"
_INPUT = "lin_srgb"


def set_working_space(name, input_space=None):
    """Set the process-wide rendering space (called when a scene is loaded / built)."""
    global _WORKING, _INPUT
    w = canonical(name or "lin_srgb")
    if not is_linear(w) or w == "raw":
        raise ValueError(f"working space must be a linear colour space, not {name!r}; use one of {WORKING_SPACES}")
    _WORKING = w
    _INPUT = canonical(input_space or "lin_srgb")


def working_space():
    return _WORKING


def default_input_space():
    """Colour space assumed for plain numeric ``[r, g, b]`` colours (``settings.color_input``)."""
    return _INPUT


def to_working(rgb, src="lin_srgb"):
    """Convert colour(s) from ``src`` into the current working space."""
    src = canonical(src)
    if src == _WORKING or src == "raw":
        return np.asarray(rgb, np.float64)
    return convert(rgb, src, _WORKING)


def working_to(rgb, dst):
    return convert(rgb, _WORKING, dst)


# ----------------------------------------------------------------------------- display
def display_encode_from_srgb(display_srgb, display):
    """Re-encode sRGB-encoded display values (as produced by the tonemappers, whose input was
    already in ``display`` primaries) with the transfer curve of ``display``."""
    d = canonical(display)
    if SPACES[d][1] == "srgb":
        return display_srgb
    return encode(_srgb_dec(display_srgb), d)


def exr_chromaticities(space):
    """(rx, ry, gx, gy, bx, by, wx, wy) for an OpenEXR ``chromaticities`` attribute."""
    prim = SPACES[canonical(space)][0]
    if prim is None:
        prim = "rec709"
    r, g, b, w = _PRIMARIES[prim]
    return (*r, *g, *b, *_WHITES[w])


def describe():
    """Rows ``(name, primaries, white, transfer)`` for listing."""
    rows = []
    for n, (p, t) in SPACES.items():
        rows.append((n, p or "-", _PRIMARIES[p][3] if p else "-", t))
    return rows
