"""Display transforms, colour grading and lens/film post effects (all numpy).

Pipeline order applied by :func:`process` (each step optional)::

    linear HDR  -> exposure (EV) -> white balance -> bloom / glare -> chromatic aberration
                -> view transform (tonemap) -> look (contrast / saturation / lift-gamma-gain / LUT)
                -> vignette -> film grain -> dither -> display (sRGB-encoded, 0-1)

``settings.tonemap`` / ``post`` keys
------------------------------------
=====================  =================================================================================
``tonemap``            ``agx`` (default, Blender 4 style filmic), ``agx_punchy``, ``aces`` (ACES fitted RRT+ODT),
                       ``filmic`` (Hable/Uncharted 2), ``reinhard``, ``neutral`` (Khronos PBR Neutral),
                       ``standard`` / ``linear`` (clip, then sRGB)
``exposure``           exposure in stops (EV, 0); ``"auto"`` meters the image (18% grey key)
``white_balance``      Kelvin temperature of the light that should appear white (6500 = neutral)
``tint``               green-magenta shift (-1 ... 1)
``contrast``           1 = neutral, applied around mid-grey in log space
``saturation``         1 = neutral
``lift / gamma / gain`` per-channel (or scalar) ASC-CDL-ish grade: ``out = gain*(x + lift*(1-x))^(1/gamma)``
``lut``                path to a ``.cube`` 3D LUT applied on display values
``bloom``              strength (0 off, ~0.05 subtle, 0.3 dreamy); ``bloom_threshold`` (1.0), ``bloom_radius`` (1.0)
``vignette``           0..1 corner darkening; ``vignette_softness`` (0.5)
``chromatic_aberration`` lateral CA in pixels at the corners (0)
``grain``              film-grain amplitude (0 ... 0.1); ``grain_size`` (1)
``dither``             true - adds +-0.5 LSB noise to kill 8-bit banding (default true)
=====================  =================================================================================
"""
from __future__ import annotations

import numpy as np

# ------------------------------------------------------------------ helpers
_LUM = np.array([0.2126, 0.7152, 0.0722], np.float32)


def luminance(rgb):
    return rgb[..., :3] @ _LUM


def srgb_encode(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def srgb_decode(x):
    x = np.clip(x, 0.0, None)
    return np.where(x <= 0.04045, x / 12.92, np.power((x + 0.055) / 1.055, 2.4))


# ------------------------------------------------------------------ view transforms
_AGX_IN = np.array([[0.842479062253094, 0.0423282422610123, 0.0423756549057051],
                    [0.0784335999999992, 0.878468636469772, 0.0784336],
                    [0.0792237451477643, 0.0791661274605434, 0.879142973793104]], np.float32)
_AGX_OUT = np.array([[1.19687900512017, -0.0528968517574562, -0.0529716355144438],
                     [-0.0980208811401368, 1.15190312990417, -0.0980434501171241],
                     [-0.0990297440797205, -0.0989611768448433, 1.15107367264116]], np.float32)
_AGX_MIN_EV, _AGX_MAX_EV = -12.47393, 4.026069


def agx(rgb, look="none"):
    """AgX view transform (Troy Sobotka) - returns display-encoded values (~sRGB) in 0-1."""
    x = np.maximum(rgb[..., :3], 1e-10) @ _AGX_IN
    x = np.clip(np.log2(np.maximum(x, 1e-10)), _AGX_MIN_EV, _AGX_MAX_EV)
    x = (x - _AGX_MIN_EV) / (_AGX_MAX_EV - _AGX_MIN_EV)
    x2 = x * x
    x4 = x2 * x2
    x = (15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x - 0.00232)
    if look == "punchy":
        lum = (x @ _LUM)[..., None]
        x = np.power(np.maximum(x, 0), 1.35)
        lum = np.power(np.maximum(lum, 0), 1.35)
        x = lum + 1.4 * (x - lum)
    x = x @ _AGX_OUT
    # AgX base output is in a 2.2 display encoding; re-encode to the sRGB curve
    lin = np.power(np.clip(x, 0, 1), 2.2)
    return srgb_encode(lin)


def aces_fitted(rgb):
    """Stephen Hill's ACES fitted RRT+ODT (sRGB primaries). Returns linear display values."""
    a_in = np.array([[0.59719, 0.35458, 0.04823], [0.07600, 0.90834, 0.01566], [0.02840, 0.13383, 0.83777]],
                    np.float32)
    a_out = np.array([[1.60475, -0.53108, -0.07367], [-0.10208, 1.10813, -0.00605], [-0.00327, -0.07276, 1.07602]],
                     np.float32)
    v = rgb[..., :3] @ a_in.T
    a = v * (v + 0.0245786) - 0.000090537
    b = v * (0.983729 * v + 0.4329510) + 0.238081
    v = (a / b) @ a_out.T
    return np.clip(v, 0, 1)


def hable(rgb, white=11.2):
    def f(x):
        A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
        return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
    return np.clip(f(2.0 * rgb[..., :3]) / f(np.float32(white)), 0, 1)


def reinhard(rgb, white=4.0):
    L = luminance(rgb)[..., None]
    Ld = L * (1 + L / (white * white)) / (1 + L)
    return np.clip(rgb[..., :3] * Ld / np.maximum(L, 1e-8), 0, 1)


def pbr_neutral(rgb):
    """Khronos PBR Neutral tone mapper - keeps base colours faithful (good for product shots)."""
    c = rgb[..., :3].copy()
    start, desat = 0.8 - 0.04, 0.15
    x = np.min(c, -1, keepdims=True)
    offset = np.where(x < 0.08, x - 6.25 * x * x, 0.04)
    c = c - offset
    peak = np.max(c, -1, keepdims=True)
    d = 1 - start
    new_peak = 1 - d * d / (peak + d - start)
    scaled = c * new_peak / np.maximum(peak, 1e-8)
    g = 1 - 1 / (desat * (peak - new_peak) + 1)
    out = np.where(peak < start, c, scaled * (1 - g) + new_peak * g)
    return np.clip(out, 0, 1)


TONEMAPPERS = {
    "agx": lambda x: agx(x), "agx_punchy": lambda x: agx(x, "punchy"), "filmic": lambda x: srgb_encode(hable(x)),
    "aces": lambda x: srgb_encode(aces_fitted(x)), "hable": lambda x: srgb_encode(hable(x)),
    "reinhard": lambda x: srgb_encode(reinhard(x)), "neutral": lambda x: srgb_encode(pbr_neutral(x)),
    "standard": lambda x: srgb_encode(np.clip(x[..., :3], 0, 1)), "linear": lambda x: srgb_encode(np.clip(x[..., :3], 0, 1)),
    "srgb": lambda x: srgb_encode(np.clip(x[..., :3], 0, 1)), "none": lambda x: srgb_encode(np.clip(x[..., :3], 0, 1)),
}


# ------------------------------------------------------------------ blur utilities
def _box_blur_axis(img, r, axis):
    if r < 1:
        return img
    n = img.shape[axis]
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r + 1, r)
    p = np.pad(img, pad, mode="edge")
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    hi = np.take(c, np.arange(2 * r + 1, 2 * r + 1 + n), axis=axis)
    lo = np.take(c, np.arange(0, n), axis=axis)
    return ((hi - lo) / (2 * r + 1)).astype(np.float32)


def gaussian_blur(img, sigma):
    """Approximate Gaussian blur (3 box passes), any channel count."""
    if sigma <= 0.3:
        return img
    r = max(1, int(round(np.sqrt(12 * sigma * sigma / 3 + 1) / 2)))
    out = img.astype(np.float32)
    for _ in range(3):
        out = _box_blur_axis(out, r, 0)
        out = _box_blur_axis(out, r, 1)
    return out


def _downsample(img):
    h, w = img.shape[:2]
    h2, w2 = max(1, h // 2), max(1, w // 2)
    im = img[:h2 * 2, :w2 * 2]
    if h < 2 or w < 2:
        return img
    return 0.25 * (im[0::2, 0::2] + im[1::2, 0::2] + im[0::2, 1::2] + im[1::2, 1::2])


def _upsample(img, shape):
    h, w = shape[:2]
    ys = np.clip((np.arange(h) + 0.5) * img.shape[0] / h - 0.5, 0, img.shape[0] - 1)
    xs = np.clip((np.arange(w) + 0.5) * img.shape[1] / w - 0.5, 0, img.shape[1] - 1)
    y0 = np.floor(ys).astype(int)
    x0 = np.floor(xs).astype(int)
    y1 = np.minimum(y0 + 1, img.shape[0] - 1)
    x1 = np.minimum(x0 + 1, img.shape[1] - 1)
    fy = (ys - y0)[:, None, None]
    fx = (xs - x0)[None, :, None]
    a = img[y0][:, x0] * (1 - fx) + img[y0][:, x1] * fx
    b = img[y1][:, x0] * (1 - fx) + img[y1][:, x1] * fx
    return (a * (1 - fy) + b * fy).astype(np.float32)


def bloom(rgb, strength=0.05, threshold=1.0, radius=1.0, levels=6):
    """Energy-conserving multi-scale bloom on linear HDR values."""
    if strength <= 0:
        return rgb
    x = rgb[..., :3]
    L = luminance(x)[..., None]
    knee = 0.5 * threshold
    soft = np.clip(L - threshold + knee, 0, 2 * knee)
    soft = soft * soft / (4 * knee + 1e-6)
    contrib = np.maximum(soft, L - threshold) / np.maximum(L, 1e-6)
    bright = x * contrib
    pyr = []
    cur = bright
    for _ in range(levels):
        cur = _downsample(cur)
        pyr.append(gaussian_blur(cur, max(0.75, 1.0 * radius)))
        if min(cur.shape[:2]) < 8:
            break
    acc = np.zeros_like(x)
    for lvl in pyr:
        acc += _upsample(lvl, x.shape)
    acc /= max(1, len(pyr))
    out = x * (1 - strength) + acc * strength * 4.0
    if rgb.shape[-1] == 4:
        return np.concatenate([out, rgb[..., 3:]], -1)
    return out


def chromatic_aberration(rgb, amount=1.0):
    if amount == 0:
        return rgb
    h, w = rgb.shape[:2]
    out = rgb.copy()
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = (w - 1) / 2, (h - 1) / 2
    dx, dy = xx - cx, yy - cy
    rmax = np.hypot(cx, cy)
    for ch, s in ((0, 1.0), (2, -1.0)):
        k = 1 + s * amount / rmax
        sx = np.clip(cx + dx * k, 0, w - 1)
        sy = np.clip(cy + dy * k, 0, h - 1)
        x0, y0 = np.floor(sx).astype(int), np.floor(sy).astype(int)
        x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
        fx, fy = sx - x0, sy - y0
        c = rgb[..., ch]
        out[..., ch] = (c[y0, x0] * (1 - fx) * (1 - fy) + c[y0, x1] * fx * (1 - fy) + c[y1, x0] * (1 - fx) * fy
                        + c[y1, x1] * fx * fy)
    return out


def vignette(img, amount=0.3, softness=0.5):
    if amount <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    nx = (xx - (w - 1) / 2) / ((w - 1) / 2 + 1e-6)
    ny = (yy - (h - 1) / 2) / ((h - 1) / 2 + 1e-6)
    aspect = w / max(h, 1)
    r = np.sqrt((nx * aspect) ** 2 + ny ** 2) / np.sqrt(aspect ** 2 + 1)
    f = 1 - amount * np.clip((r - (1 - softness) * 0.7) / max(softness, 1e-3), 0, 1) ** 2
    out = img.copy()
    out[..., :3] *= f[..., None]
    return out


def film_grain(img, amount=0.03, size=1.0, seed=0):
    if amount <= 0:
        return img
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    n = rng.standard_normal((h, w, 1)).astype(np.float32)
    if size > 1.01:
        n = gaussian_blur(n, size * 0.5) * size
    lum = luminance(img)[..., None]
    weight = 4 * lum * (1 - lum)  # strongest in the mid-tones, like real film
    out = img.copy()
    out[..., :3] = img[..., :3] + n * amount * np.clip(weight, 0.15, 1)
    return out


# ------------------------------------------------------------------ grading
def white_balance_matrix(kelvin=6500.0, tint=0.0):
    """Von Kries-style gains that make a ``kelvin`` light appear white."""
    from ..materials import kelvin_to_rgb
    ref = kelvin_to_rgb(6500.0)
    src = kelvin_to_rgb(kelvin)
    g = ref / np.maximum(src, 1e-6)
    g = g / (g @ _LUM)
    g = g * np.array([1 + 0.5 * tint, 1 - tint, 1 + 0.5 * tint])
    return g.astype(np.float32)


def auto_exposure(rgb, key=0.18):
    """EV that maps the log-average scene luminance to ``key`` (ignores the brightest 2%)."""
    L = luminance(rgb).ravel()
    L = L[np.isfinite(L)]
    if L.size == 0:
        return 0.0
    hi = np.percentile(L, 98)
    L = L[L <= hi]
    lavg = float(np.exp(np.mean(np.log(L + 1e-4))))
    return float(np.log2(key / max(lavg, 1e-6)))


def read_lut(path):
    """Read a LUT file: ``.cube`` (3D or 1D, Resolve/Adobe), ``.spi3d`` / ``.spi1d`` (Sony
    Imageworks, the OCIO native formats) or ``.3dl`` (Autodesk/Lustre, 10/12-bit integer)."""
    from pathlib import Path
    ext = Path(path).suffix.lower()
    if ext == ".spi3d":
        return read_spi3d(path)
    if ext == ".spi1d":
        return read_spi1d(path)
    if ext == ".3dl":
        return read_3dl(path)
    return read_cube_lut(path)


def read_spi3d(path):
    with open(path) as fh:
        lines = [ln.split() for ln in fh if ln.strip() and not ln.startswith("#")]
    # "SPILUT 1.0" / "3 3" / "N N N" / "i j k r g b"...
    i = 0
    while len(lines[i]) != 3 or not all(t.lstrip("-").isdigit() for t in lines[i]):
        i += 1
    n = [int(t) for t in lines[i]]
    i += 1
    lut = np.zeros((n[2], n[1], n[0], 3), np.float32)  # [b][g][r]
    for row in lines[i:]:
        if len(row) < 6:
            continue
        r, g, b = int(row[0]), int(row[1]), int(row[2])
        lut[b, g, r] = [float(v) for v in row[3:6]]
    if not (n[0] == n[1] == n[2]):
        raise ValueError(f"{path}: non-cubic spi3d LUTs are not supported")
    return lut, np.zeros(3), np.ones(3)


def read_spi1d(path):
    dmin, dmax, comps, vals = 0.0, 1.0, 1, []
    with open(path) as fh:
        body = False
        for ln in fh:
            t = ln.strip()
            if not t or t.startswith("#"):
                continue
            if t.startswith("From"):
                dmin, dmax = (float(v) for v in t.split()[1:3])
            elif t.startswith("Components"):
                comps = int(t.split()[1])
            elif t.startswith("{"):
                body = True
            elif t.startswith("}"):
                body = False
            elif body:
                vals.append([float(v) for v in t.split()])
    tab = np.asarray(vals, np.float32)
    if tab.ndim == 1 or tab.shape[1] == 1:
        tab = np.repeat(tab.reshape(-1, 1), 3, 1)
    elif tab.shape[1] == 2:
        tab = np.concatenate([tab, tab[:, 1:2]], 1)
    return ("1d", tab[:, :3], np.full(3, dmin), np.full(3, dmax))


def read_3dl(path):
    rows, shaper = [], None
    with open(path) as fh:
        for ln in fh:
            t = ln.split()
            if not t or t[0].startswith("#") or not t[0].lstrip("-").isdigit():
                continue
            if shaper is None and len(t) > 3:
                shaper = [int(v) for v in t]
                continue
            if len(t) == 3:
                rows.append([int(v) for v in t])
    data = np.asarray(rows, np.float32)
    n = round(len(data) ** (1 / 3))
    maxv = 4095.0 if data.max() > 1023 else 1023.0
    lut = (data / maxv).reshape(n, n, n, 3)  # 3dl is red-major: [r][g][b]
    return np.ascontiguousarray(lut.transpose(2, 1, 0, 3)), np.zeros(3), np.ones(3)


def read_cube_lut(path):
    size = None
    size1d = None
    data = []
    domain_min, domain_max = np.zeros(3), np.ones(3)
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            up = s.upper()
            if up.startswith("LUT_3D_SIZE"):
                size = int(s.split()[1])
            elif up.startswith("LUT_1D_SIZE"):
                size1d = int(s.split()[1])
            elif up.startswith("DOMAIN_MIN"):
                domain_min = np.array([float(v) for v in s.split()[1:4]])
            elif up.startswith("DOMAIN_MAX"):
                domain_max = np.array([float(v) for v in s.split()[1:4]])
            elif up.startswith(("TITLE", "LUT_1D", "LUT_3D_INPUT")):
                continue
            else:
                parts = s.split()
                if len(parts) == 3:
                    data.append([float(v) for v in parts])
    if size is None and size1d:
        return ("1d", np.asarray(data, np.float32).reshape(size1d, 3), domain_min, domain_max)
    if size is None:
        raise ValueError(f"{path}: no LUT_3D_SIZE / LUT_1D_SIZE found")
    lut = np.asarray(data, np.float32).reshape(size, size, size, 3)  # [b][g][r]
    return lut, domain_min, domain_max


def apply_lut(img, lut_spec):
    if len(lut_spec) == 4 and isinstance(lut_spec[0], str):  # 1D LUT
        _, tab, dmin, dmax = lut_spec
        out = img.copy()
        xs = np.linspace(0, 1, tab.shape[0])
        for c in range(3):
            x = (img[..., c] - dmin[c]) / (dmax[c] - dmin[c])
            out[..., c] = np.interp(x, xs, tab[:, c])
        return out
    lut, dmin, dmax = lut_spec
    n = lut.shape[0]
    x = np.clip((img[..., :3] - dmin) / (dmax - dmin), 0, 1) * (n - 1)
    i0 = np.floor(x).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    f = x - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    r1, g1, b1 = i1[..., 0], i1[..., 1], i1[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]

    def L(b, g, r):
        return lut[b, g, r]

    c00 = L(b0, g0, r0) * (1 - fr) + L(b0, g0, r1) * fr
    c01 = L(b0, g1, r0) * (1 - fr) + L(b0, g1, r1) * fr
    c10 = L(b1, g0, r0) * (1 - fr) + L(b1, g0, r1) * fr
    c11 = L(b1, g1, r0) * (1 - fr) + L(b1, g1, r1) * fr
    c0 = c00 * (1 - fg) + c01 * fg
    c1 = c10 * (1 - fg) + c11 * fg
    out = img.copy()
    out[..., :3] = c0 * (1 - fb) + c1 * fb
    return out


_LUT_CACHE: dict = {}


def _vec3(v, default):
    if v is None:
        return np.asarray(default, np.float32)
    a = np.asarray(v, np.float32).ravel()
    return np.repeat(a[:1], 3) if a.size == 1 else a[:3]


def process(rgba, settings=None, frame=0, base_dir="."):
    """Apply the full display pipeline. Returns float32 (H,W,C) display-referred 0-1 values.

    ``settings`` is a dict of the keys documented in the module docstring. Alpha passes through.
    """
    s = dict(settings or {})
    img = np.array(rgba, np.float32, copy=True)
    has_alpha = img.ndim == 3 and img.shape[-1] == 4
    alpha = img[..., 3:4].copy() if has_alpha else None
    rgb = np.nan_to_num(img[..., :3], nan=0.0, posinf=1e4, neginf=0.0)
    ev = s.get("exposure", 0.0)
    if ev == "auto":
        ev = auto_exposure(rgb) + float(s.get("exposure_compensation", 0.0))
    rgb = rgb * np.float32(2.0 ** float(ev))
    from .. import color as cm
    work = cm.canonical(s.get("working_space") or "lin_srgb")
    display = cm.canonical(s.get("display") or "srgb")
    wb = s.get("white_balance")
    tint = float(s.get("tint", 0.0))
    if wb is not None or tint:
        g = white_balance_matrix(float(wb if wb is not None else 6500.0), tint)
        if work == "lin_srgb":
            rgb = rgb * g
        else:  # gains are defined in Rec.709 primaries; apply them there
            m = cm.matrix("lin_srgb", work) @ np.diag(g) @ cm.matrix(work, "lin_srgb")
            rgb = (rgb @ m.T.astype(np.float32)).astype(np.float32)
    if float(s.get("bloom", 0.0)) > 0:
        rgb = bloom(rgb, float(s["bloom"]), float(s.get("bloom_threshold", 1.0)), float(s.get("bloom_radius", 1.0)))
    if float(s.get("chromatic_aberration", 0.0)):
        rgb = chromatic_aberration(rgb, float(s["chromatic_aberration"]))
    # working space -> display primaries (gamut mapping by soft clip of negatives)
    dm = cm.matrix(work, display)
    if not np.allclose(dm, np.eye(3)):
        rgb = rgb @ dm.T.astype(np.float32)
        rgb = gamut_compress(rgb)
    tm = s.get("tonemap", "agx")
    if tm not in TONEMAPPERS:
        raise ValueError(f"unknown tonemap {tm!r}; available: {sorted(TONEMAPPERS)}")
    out = TONEMAPPERS[tm](rgb).astype(np.float32)
    # ---- look
    c = float(s.get("contrast", 1.0))
    if c != 1.0:
        lin = srgb_decode(out)
        lg = np.log2(np.maximum(lin, 1e-6) / 0.18)
        out = srgb_encode(0.18 * np.power(2.0, lg * c))
    sat = float(s.get("saturation", 1.0))
    if sat != 1.0:
        lum = (out @ _LUM)[..., None]
        out = lum + (out - lum) * sat
    if any(k in s for k in ("lift", "gamma", "gain")):
        lift = _vec3(s.get("lift"), [0, 0, 0])
        gamma = _vec3(s.get("gamma"), [1, 1, 1])
        gain = _vec3(s.get("gain"), [1, 1, 1])
        x = np.clip(out, 0, None)
        out = gain * np.power(np.clip(x + lift * (1 - x), 0, None), 1.0 / np.maximum(gamma, 1e-3))
    lut = s.get("lut")
    if lut:
        from pathlib import Path
        p = Path(lut)
        if not p.is_absolute():
            p = Path(base_dir) / p
        key = str(p)
        if key not in _LUT_CACHE:
            _LUT_CACHE[key] = read_lut(p)
        out = apply_lut(out, _LUT_CACHE[key])
    if float(s.get("vignette", 0.0)) > 0:
        out = vignette(out, float(s["vignette"]), float(s.get("vignette_softness", 0.5)))
    if float(s.get("grain", 0.0)) > 0:
        out = film_grain(out, float(s["grain"]), float(s.get("grain_size", 1.0)), seed=int(frame) * 7919 + 13)
    out = np.clip(out, 0, 1)
    out = np.asarray(cm.display_encode_from_srgb(out, display), np.float32)
    if has_alpha:
        out = np.concatenate([out, np.clip(alpha, 0, 1)], -1)
    return out.astype(np.float32)


def gamut_compress(rgb, threshold=0.8):
    """Pull out-of-gamut (negative) colours back towards the achromatic axis, preserving luminance
    as far as possible (a simplified ACES reference gamut compression)."""
    rgb = np.asarray(rgb, np.float32)
    ach = np.max(rgb, -1, keepdims=True)
    safe = np.where(ach > 1e-8, ach, 1.0)
    d = np.where(ach > 1e-8, (ach - rgb) / safe, 0.0)  # distance from achromatic, 1 = on gamut edge
    lim = 1.2  # distance that is mapped to the gamut boundary
    t = threshold
    s = (lim - t) / np.sqrt(max(((1 - t) / (lim - t)) ** -2.0 - 1.0, 1e-8))
    over = d > t
    dc = np.where(over, t + (d - t) / np.power(1 + np.power(np.maximum(d - t, 0) / s, 2.0), 0.5), d)
    out = ach - dc * safe
    return np.where(ach > 1e-8, out, np.maximum(rgb, 0)).astype(np.float32)


def to_uint8(img, dither=True, seed=0):
    x = np.clip(img, 0, 1) * 255.0
    if dither:
        rng = np.random.default_rng(seed)
        x = x + rng.uniform(-0.5, 0.5, x.shape)
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def to_uint16(img):
    return np.clip(np.round(np.clip(img, 0, 1) * 65535.0), 0, 65535).astype(np.uint16)
