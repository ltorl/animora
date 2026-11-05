"""Image reading / writing with no dependencies beyond numpy (+ Pillow for LDR formats).

Supported formats

=========  =========  ==========================================================
extension  read       write
=========  =========  ==========================================================
``.png``   yes (PIL)  8-bit (PIL) or 16-bit (built-in encoder), RGB/RGBA
``.jpg``   yes (PIL)  yes (PIL)
``.tif``   yes (PIL)  8-bit (PIL)
``.bmp``   yes        yes
``.exr``   yes        yes - built-in OpenEXR scanline codec: NONE/ZIPS/ZIP/RLE
                      read; write NONE or ZIP, HALF or FLOAT channels,
                      arbitrary extra layers (AOVs)
``.hdr``   yes        yes - Radiance RGBE (RLE read & write)
``.pfm``   yes        yes - portable float map
=========  =========  ==========================================================

All float images are linear. :func:`read_image` returns ``float32`` arrays (H,W,C) with LDR
formats converted from sRGB to linear unless ``linear=False``.
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np

HDR_EXTS = {".exr", ".hdr", ".pfm"}
LDR_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".tga", ".webp", ".gif"}


# ----------------------------------------------------------------------------- colour transfer
def srgb_to_linear(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0.0, None)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055).astype(np.float32)


# ----------------------------------------------------------------------------- generic API
def read_image(path, linear=True):
    """Read an image as float32 (H,W,C). LDR files are decoded from sRGB when ``linear``."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".exr":
        layers = read_exr(path)
        return _exr_to_rgb(layers)
    if ext == ".hdr":
        return read_hdr(path)
    if ext == ".pfm":
        return read_pfm(path)
    if ext == ".png":
        a = _read_png16(path)
        if a is not None:
            if linear:
                c = a.shape[2]
                if c in (2, 4):
                    return np.concatenate([srgb_to_linear(a[..., :c - 1]), a[..., c - 1:]], axis=2)
                return srgb_to_linear(a)
            return a
    from PIL import Image
    im = Image.open(path)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, np.float32) / (65535.0 if im.mode.startswith("I;16") else max(1.0, float(np.max(im))))
        a = a[..., None]
        return srgb_to_linear(a) if linear else a
    if im.mode not in ("RGB", "RGBA", "L", "LA"):
        im = im.convert("RGBA" if "A" in im.getbands() or im.mode == "P" else "RGB")
    a = np.asarray(im, np.float32) / 255.0
    if a.ndim == 2:
        a = a[..., None]
    if linear:
        c = a.shape[2]
        if c in (2, 4):
            a = np.concatenate([srgb_to_linear(a[..., :c - 1]), a[..., c - 1:]], axis=2)
        else:
            a = srgb_to_linear(a)
    return a


def write_image(path, img, bit_depth=8, compression="zip", half=True, layers=None, quality=95,
                chromaticities=None):
    """Write an image. ``img`` is float (H,W,C) in [0,1] *display* space for LDR formats
    (already tone-mapped & sRGB encoded) and linear radiance for HDR formats.

    ``layers`` (EXR only): extra ``{"name": array(H,W) or (H,W,C)}`` channels (AOVs).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = img[..., None]
    if ext == ".exr":
        chans = {}
        names = {1: ["Y"], 2: ["Y", "A"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}[img.shape[2]]
        for i, n in enumerate(names):
            chans[n] = img[..., i]
        for lname, arr in (layers or {}).items():
            arr = np.asarray(arr, np.float32)
            if arr.ndim == 2:
                chans[lname] = arr
            else:
                sub = {1: ["V"], 2: ["U", "V"], 3: ["R", "G", "B"], 4: ["R", "G", "B", "A"]}[arr.shape[2]]
                if lname in ("normal", "N", "P", "position", "velocity"):
                    sub = ["X", "Y", "Z", "W"][:arr.shape[2]]
                for i, n in enumerate(sub):
                    chans[f"{lname}.{n}"] = arr[..., i]
        write_exr(path, chans, half=half, compression=compression, chromaticities=chromaticities)
        return path
    if ext == ".hdr":
        write_hdr(path, img[..., :3] if img.shape[2] >= 3 else np.repeat(img[..., :1], 3, 2))
        return path
    if ext == ".pfm":
        write_pfm(path, img)
        return path
    if ext == ".png" and bit_depth == 16:
        write_png16(path, img)
        return path
    from PIL import Image
    a = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    if a.shape[2] == 1:
        a = a[..., 0]
    elif a.shape[2] == 2:
        a = np.concatenate([np.repeat(a[..., :1], 3, 2), a[..., 1:]], 2)
    im = Image.fromarray(a)
    if ext in (".jpg", ".jpeg"):
        if im.mode == "RGBA":
            im = im.convert("RGB")
        im.save(path, quality=quality, subsampling=0)
    else:
        im.save(path)
    return path


# ----------------------------------------------------------------------------- 16-bit PNG
def _png_chunk(tag, data):
    return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)


def write_png16(path, img):
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    ctype = {1: 0, 2: 4, 3: 2, 4: 6}[C]
    a = (np.clip(img, 0, 1) * 65535.0 + 0.5).astype(">u2")
    raw = np.zeros((H, 1 + W * C * 2), np.uint8)
    raw[:, 1:] = a.reshape(H, W * C).view(np.uint8).reshape(H, W * C * 2)
    data = zlib.compress(raw.tobytes(), 6)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_png_chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 16, ctype, 0, 0, 0)))
        f.write(_png_chunk(b"IDAT", data))
        f.write(_png_chunk(b"IEND", b""))


def _read_png16(path):
    """Decode non-interlaced 16-bit PNGs exactly (PIL truncates 16-bit RGB to 8 bits).
    Returns None for anything else so the caller falls back to PIL."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    idat = []
    W = H = depth = ctype = interlace = None
    while pos < len(data):
        (ln,) = struct.unpack(">I", data[pos:pos + 4])
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + ln]
        pos += 12 + ln
        if tag == b"IHDR":
            W, H, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
            if depth != 16 or interlace != 0 or ctype not in (0, 2, 4, 6):
                return None
        elif tag == b"IDAT":
            idat.append(body)
        elif tag == b"IEND":
            break
    C = {0: 1, 2: 3, 4: 2, 6: 4}[ctype]
    raw = np.frombuffer(zlib.decompress(b"".join(idat)), np.uint8).reshape(H, 1 + W * C * 2)
    bpp = C * 2
    out = np.zeros((H, W * C * 2), np.int32)
    prev = np.zeros(W * C * 2, np.int32)
    for y in range(H):
        ft = raw[y, 0]
        line = raw[y, 1:].astype(np.int32)
        if ft == 0:
            cur = line
        elif ft == 2:
            cur = (line + prev) & 255
        else:
            cur = np.zeros_like(line)
            for x in range(len(line)):
                a = cur[x - bpp] if x >= bpp else 0
                b = prev[x]
                c = prev[x - bpp] if x >= bpp else 0
                if ft == 1:
                    pr = a
                elif ft == 3:
                    pr = (a + b) >> 1
                else:
                    p_ = a + b - c
                    pa, pb, pc = abs(p_ - a), abs(p_ - b), abs(p_ - c)
                    pr = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                cur[x] = (line[x] + pr) & 255
        out[y] = cur
        prev = cur
    v = out.astype(np.uint8).reshape(H, W * C, 2)
    vals = (v[..., 0].astype(np.uint32) << 8) | v[..., 1]
    return (vals.reshape(H, W, C) / 65535.0).astype(np.float32)


# ----------------------------------------------------------------------------- PFM
def write_pfm(path, img):
    img = np.asarray(img, np.float32)
    if img.ndim == 2:
        img = img[..., None]
    H, W, C = img.shape
    if C == 1:
        hdr = b"Pf"
        data = img[..., 0]
    else:
        hdr = b"PF"
        data = img[..., :3] if C >= 3 else np.repeat(img[..., :1], 3, 2)
    with open(path, "wb") as f:
        f.write(hdr + b"\n%d %d\n-1.0\n" % (W, H))
        f.write(np.ascontiguousarray(data[::-1]).astype("<f4").tobytes())


def read_pfm(path):
    with open(path, "rb") as f:
        kind = f.readline().strip()
        dims = f.readline().split()
        while len(dims) < 2:
            dims += f.readline().split()
        W, H = int(dims[0]), int(dims[1])
        scale = float(f.readline().strip())
        C = 3 if kind == b"PF" else 1
        dt = "<f4" if scale < 0 else ">f4"
        data = np.frombuffer(f.read(W * H * C * 4), dt).reshape(H, W, C)
    return data[::-1].astype(np.float32).copy()


# ----------------------------------------------------------------------------- Radiance HDR
def write_hdr(path, img):
    img = np.maximum(np.asarray(img, np.float32)[..., :3], 0)
    H, W, _ = img.shape
    m = img.max(axis=2)
    e = np.zeros(m.shape, np.int32)
    mant, e = np.frexp(m)
    scale = np.where(m > 1e-32, mant * 256.0 / np.maximum(m, 1e-32), 0)
    rgbe = np.zeros((H, W, 4), np.uint8)
    rgbe[..., :3] = np.clip(img * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(m > 1e-32, e + 128, 0).astype(np.uint8)
    with open(path, "wb") as f:
        f.write(b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n")
        f.write(b"-Y %d +X %d\n" % (H, W))
        if 8 <= W < 32768:
            for y in range(H):
                f.write(bytes([2, 2, W >> 8, W & 255]))
                for c in range(4):
                    f.write(_rle_encode(rgbe[y, :, c]))
        else:
            f.write(rgbe.tobytes())


def _rle_encode(row):
    out = bytearray()
    n = len(row)
    i = 0
    row = row.tobytes()
    while i < n:
        # run?
        j = i + 1
        while j < n and j - i < 127 and row[j] == row[i]:
            j += 1
        if j - i >= 4:
            out += bytes([128 + (j - i), row[i]])
            i = j
            continue
        # literal: until a run of >=4 starts
        j = i
        while j < n and j - i < 128:
            if j + 3 < n and row[j] == row[j + 1] == row[j + 2] == row[j + 3]:
                break
            j += 1
        out.append(j - i)
        out += row[i:j]
        i = j
    return bytes(out)


def read_hdr(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 0
    # header lines until empty line
    while True:
        e = data.index(b"\n", pos)
        line = data[pos:e]
        pos = e + 1
        if line.strip() == b"":
            break
    e = data.index(b"\n", pos)
    res = data[pos:e].split()
    pos = e + 1
    H, W = int(res[1]), int(res[3])
    flip_y = res[0] == b"+Y"
    out = np.zeros((H, W, 4), np.uint8)
    buf = np.frombuffer(data, np.uint8)
    for y in range(H):
        if W >= 8 and W < 32768 and buf[pos] == 2 and buf[pos + 1] == 2 and (buf[pos + 2] & 0x80) == 0:
            pos += 4
            for c in range(4):
                x = 0
                while x < W:
                    cnt = int(buf[pos])
                    pos += 1
                    if cnt > 128:
                        cnt -= 128
                        out[y, x:x + cnt, c] = buf[pos]
                        pos += 1
                    else:
                        out[y, x:x + cnt, c] = buf[pos:pos + cnt]
                        pos += cnt
                    x += cnt
        else:
            out[y] = buf[pos:pos + W * 4].reshape(W, 4)
            pos += W * 4
    e = out[..., 3].astype(np.int32)
    f = np.where(e > 0, np.ldexp(1.0, e - 136), 0.0).astype(np.float32)
    rgb = out[..., :3].astype(np.float32) * f[..., None]
    if flip_y:
        rgb = rgb[::-1]
    return rgb


# ----------------------------------------------------------------------------- OpenEXR
_EXR_MAGIC = 20000630
_COMP = {"none": 0, "rle": 1, "zips": 2, "zip": 3}
_LINES = {0: 1, 1: 1, 2: 1, 3: 16}


def _attr(name, typ, data):
    return name.encode() + b"\0" + typ.encode() + b"\0" + struct.pack("<i", len(data)) + data


def _predict_encode(raw):
    b = np.frombuffer(raw, np.uint8)
    n = len(b)
    t = np.empty(n, np.uint8)
    half = (n + 1) // 2
    t[:half] = b[0::2]
    t[half:] = b[1::2]
    d = t.astype(np.int16)
    d[1:] = (t[1:].astype(np.int16) - t[:-1].astype(np.int16) + 128 + 256) % 256
    return d.astype(np.uint8).tobytes()


def _predict_decode(raw):
    d = np.frombuffer(raw, np.uint8).astype(np.int64)
    if len(d) == 0:
        return b""
    d[1:] -= 128
    t = (np.cumsum(d) % 256).astype(np.uint8)
    n = len(t)
    half = (n + 1) // 2
    b = np.empty(n, np.uint8)
    b[0::2] = t[:half]
    b[1::2] = t[half:]
    return b.tobytes()


def write_exr(path, channels: dict, half=True, compression="zip", chromaticities=None):
    """Write a scanline OpenEXR file. ``channels`` maps names ("R","G","B","A","depth.V"...) to
    (H,W) float arrays. ``half`` stores HALF (16-bit) except for names containing ``depth``/``Z``
    which are always FLOAT. ``compression`` is ``"none"`` or ``"zip"``."""
    names = sorted(channels)  # EXR requires alphabetical channel order
    H, W = np.asarray(channels[names[0]]).shape[:2]
    comp = _COMP[compression if compression in ("none", "zip", "zips") else "zip"]
    if comp == 1:
        comp = 3
    ptypes = {}
    for n in names:
        is_depth = n in ("Z", "depth") or n.startswith("depth") or n.endswith(".Z") and "normal" not in n
        ptypes[n] = 2 if (not half or is_depth) else 1  # 1 HALF, 2 FLOAT
    chl = b""
    for n in names:
        chl += n.encode() + b"\0" + struct.pack("<iBBBBii", ptypes[n], 0, 0, 0, 0, 1, 1)
    chl += b"\0"
    header = struct.pack("<ii", _EXR_MAGIC, 2)
    header += _attr("channels", "chlist", chl)
    if chromaticities is not None:
        header += _attr("chromaticities", "chromaticities", struct.pack("<8f", *chromaticities))
    header += _attr("compression", "compression", bytes([comp]))
    box = struct.pack("<iiii", 0, 0, W - 1, H - 1)
    header += _attr("dataWindow", "box2i", box)
    header += _attr("displayWindow", "box2i", box)
    header += _attr("lineOrder", "lineOrder", bytes([0]))
    header += _attr("pixelAspectRatio", "float", struct.pack("<f", 1.0))
    header += _attr("screenWindowCenter", "v2f", struct.pack("<ff", 0, 0))
    header += _attr("screenWindowWidth", "float", struct.pack("<f", 1.0))
    header += b"\0"
    lpb = _LINES[comp]
    nblocks = (H + lpb - 1) // lpb
    arrs = {n: np.asarray(channels[n], np.float32) for n in names}
    blocks = []
    for bi in range(nblocks):
        y0 = bi * lpb
        y1 = min(H, y0 + lpb)
        parts = []
        for y in range(y0, y1):
            for n in names:
                row = arrs[n][y]
                parts.append(row.astype("<f2").tobytes() if ptypes[n] == 1 else row.astype("<f4").tobytes())
        raw = b"".join(parts)
        if comp in (2, 3):
            packed = zlib.compress(_predict_encode(raw), 6)
            if len(packed) < len(raw):
                raw = packed
        blocks.append(struct.pack("<ii", y0, len(raw)) + raw)
    table_size = 8 * nblocks
    offset = len(header) + table_size
    offsets = []
    for b in blocks:
        offsets.append(offset)
        offset += len(b)
    with open(path, "wb") as f:
        f.write(header)
        f.write(struct.pack("<%dQ" % nblocks, *offsets))
        for b in blocks:
            f.write(b)


def read_exr(path):
    """Read a scanline OpenEXR (NONE/RLE/ZIPS/ZIP). Returns ``{channel: (H,W) float32}``."""
    with open(path, "rb") as f:
        data = f.read()
    magic, ver = struct.unpack_from("<ii", data, 0)
    if magic != _EXR_MAGIC:
        raise ValueError(f"{path}: not an OpenEXR file")
    if ver & 0x200:
        raise ValueError(f"{path}: tiled EXR files are not supported")
    pos = 8
    attrs = {}
    while data[pos] != 0:
        e = data.index(b"\0", pos)
        name = data[pos:e].decode()
        pos = e + 1
        e = data.index(b"\0", pos)
        typ = data[pos:e].decode()
        pos = e + 1
        (size,) = struct.unpack_from("<i", data, pos)
        pos += 4
        attrs[name] = (typ, data[pos:pos + size])
        pos += size
    pos += 1
    chans = []
    cl = attrs["channels"][1]
    p = 0
    while cl[p] != 0:
        e = cl.index(b"\0", p)
        n = cl[p:e].decode()
        p = e + 1
        ptype, _, _, _, _, xs, ys = struct.unpack_from("<iBBBBii", cl, p)
        p += 16
        chans.append((n, ptype))
    comp = attrs["compression"][1][0]
    x0, y0, x1, y1 = struct.unpack("<iiii", attrs["dataWindow"][1])
    W, H = x1 - x0 + 1, y1 - y0 + 1
    if comp not in _LINES:
        raise ValueError(f"{path}: unsupported EXR compression {comp} (supported: NONE, RLE, ZIPS, ZIP)")
    lpb = _LINES[comp]
    nblocks = (H + lpb - 1) // lpb
    offsets = struct.unpack_from("<%dQ" % nblocks, data, pos)
    out = {n: np.zeros((H, W), np.float32) for n, _ in chans}
    bpp = {0: 4, 1: 2, 2: 4}
    for off in offsets:
        ly, size = struct.unpack_from("<ii", data, off)
        raw = data[off + 8: off + 8 + size]
        nl = min(lpb, y1 - ly + 1)
        expected = sum(W * bpp[t] for _, t in chans) * nl
        if len(raw) < expected:
            if comp in (2, 3):
                raw = _predict_decode(zlib.decompress(raw))
            elif comp == 1:
                raw = _predict_decode(_rle_decode(raw))
        q = 0
        for yy in range(nl):
            for n, t in chans:
                cnt = W * bpp[t]
                seg = raw[q:q + cnt]
                q += cnt
                if t == 1:
                    row = np.frombuffer(seg, "<f2").astype(np.float32)
                elif t == 2:
                    row = np.frombuffer(seg, "<f4")
                else:
                    row = np.frombuffer(seg, "<u4").astype(np.float32)
                out[n][ly - y0 + yy] = row
    return out


def _rle_decode(raw):
    out = bytearray()
    i = 0
    b = raw
    while i < len(b):
        c = struct.unpack_from("<b", b, i)[0]
        i += 1
        if c < 0:
            out += b[i:i - c]
            i += -c
        else:
            out += bytes([b[i]]) * (c + 1)
            i += 1
    return bytes(out)


def _exr_to_rgb(layers):
    keys = set(layers)
    if {"R", "G", "B"} <= keys:
        cs = [layers["R"], layers["G"], layers["B"]] + ([layers["A"]] if "A" in keys else [])
    elif "Y" in keys:
        cs = [layers["Y"]] * 3 + ([layers["A"]] if "A" in keys else [])
    else:
        # first layer group with RGB
        for k in sorted(keys):
            if k.endswith(".R"):
                pre = k[:-2]
                if f"{pre}.G" in keys and f"{pre}.B" in keys:
                    cs = [layers[f"{pre}.R"], layers[f"{pre}.G"], layers[f"{pre}.B"]]
                    break
        else:
            first = layers[sorted(keys)[0]]
            cs = [first] * 3
    return np.stack(cs, axis=2).astype(np.float32)


def image_info(path):
    """Return a dict with width, height, channels and format for any supported image."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".exr":
        layers = read_exr(path)
        a = next(iter(layers.values()))
        return {"width": a.shape[1], "height": a.shape[0], "channels": sorted(layers), "format": "exr"}
    img = read_image(path, linear=False)
    return {"width": img.shape[1], "height": img.shape[0], "channels": img.shape[2], "format": ext[1:]}
