"""ctypes bridge to the native C++ path tracer (``animora/native/pathtracer.cpp``).

The shared library is compiled on first use and cached in ``animora/native/build`` (or in
``$ANIMORA_CACHE`` / ``~/.cache/animora`` when the package directory is not writable). The
structures below mirror the C ABI exactly; :func:`load` verifies the layout with
``an_struct_sizes`` so a mismatch fails loudly instead of corrupting memory.
"""
from __future__ import annotations

import ctypes as C
import hashlib
import os
import platform
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

NATIVE_DIR = Path(__file__).resolve().parent.parent / "native"
SOURCE = NATIVE_DIR / "pathtracer.cpp"

f3 = C.c_float * 3
f2 = C.c_float * 2
f6 = C.c_float * 6
fp = C.POINTER(C.c_float)
ip = C.POINTER(C.c_int32)


class AnTexture(C.Structure):
    _fields_ = [
        ("type", C.c_int32), ("width", C.c_int32), ("height", C.c_int32), ("channels", C.c_int32),
        ("offset", C.c_int32), ("space", C.c_int32), ("wrap", C.c_int32), ("interp", C.c_int32),
        ("color1", f3), ("color2", f3), ("scale", C.c_float), ("params", f6),
        ("uv_scale", f2), ("uv_offset", f2), ("uv_rotation", C.c_float), ("pad", C.c_float),
    ]


class AnMaterial(C.Structure):
    _fields_ = [
        ("base_color", f3), ("metallic", C.c_float),
        ("roughness", C.c_float), ("ior", C.c_float), ("transmission", C.c_float), ("specular", C.c_float),
        ("specular_tint", C.c_float), ("clearcoat", C.c_float), ("clearcoat_roughness", C.c_float), ("sheen", C.c_float),
        ("sheen_tint", C.c_float), ("subsurface", C.c_float),
        ("subsurface_radius", f3), ("subsurface_scale", C.c_float),
        ("emission", f3), ("emission_strength", C.c_float),
        ("alpha", C.c_float), ("normal_strength", C.c_float), ("bump_strength", C.c_float),
        ("transmission_roughness", C.c_float),
        ("attenuation_color", f3), ("attenuation_distance", C.c_float),
        ("tex_base_color", C.c_int32), ("tex_roughness", C.c_int32), ("tex_metallic", C.c_int32), ("tex_normal", C.c_int32),
        ("tex_emission", C.c_int32), ("tex_alpha", C.c_int32), ("tex_bump", C.c_int32), ("tex_transmission", C.c_int32),
        ("thin_walled", C.c_int32), ("emission_one_sided", C.c_int32), ("shadow_catcher", C.c_int32), ("flags", C.c_int32),
    ]


class AnLight(C.Structure):
    _fields_ = [
        ("type", C.c_int32), ("visible", C.c_int32), ("cast_shadows", C.c_int32), ("pad0", C.c_int32),
        ("position", f3), ("radius", C.c_float),
        ("direction", f3), ("angle", C.c_float),
        ("color", f3), ("intensity", C.c_float),
        ("blend", C.c_float), ("pad1", f3),
    ]


class AnCamera(C.Structure):
    _fields_ = [
        ("pos0", f3), ("fwd0", f3), ("up0", f3),
        ("pos1", f3), ("fwd1", f3), ("up1", f3),
        ("fov_y", C.c_float), ("projection", C.c_int32),
        ("ortho_height", C.c_float), ("aperture_radius", C.c_float), ("focus_distance", C.c_float),
        ("blades", C.c_int32),
        ("blade_rotation", C.c_float), ("shift_x", C.c_float), ("shift_y", C.c_float), ("near_clip", C.c_float),
    ]


class AnVolume(C.Structure):
    _fields_ = [
        ("type", C.c_int32), ("nx", C.c_int32), ("ny", C.c_int32), ("nz", C.c_int32),
        ("density_offset", C.c_int32), ("heat_offset", C.c_int32), ("bb_offset", C.c_int32), ("bb_size", C.c_int32),
        ("shape", C.c_int32), ("octaves", C.c_int32),
        ("world_to_local", C.c_float * 12),
        ("albedo", f3), ("density", C.c_float),
        ("emission", f3), ("emission_strength", C.c_float),
        ("anisotropy", C.c_float), ("blackbody_strength", C.c_float), ("majorant", C.c_float),
        ("frequency", C.c_float), ("gain", C.c_float), ("lacunarity", C.c_float), ("falloff", C.c_float),
        ("threshold", C.c_float), ("contrast", C.c_float),
        ("offset", f3), ("heat_from_density", C.c_float),
    ]


class AnScene(C.Structure):
    _fields_ = [
        ("num_tris", C.c_int32), ("has_motion", C.c_int32),
        ("P0", fp), ("P1", fp), ("N0", fp), ("N1", fp), ("UV", fp), ("PREF", fp),
        ("tri_mat", ip), ("tri_obj", ip),
        ("num_materials", C.c_int32), ("num_textures", C.c_int32),
        ("materials", C.POINTER(AnMaterial)), ("textures", C.POINTER(AnTexture)), ("texdata", fp),
        ("num_lights", C.c_int32), ("env_tex", C.c_int32),
        ("lights", C.POINTER(AnLight)),
        ("env_strength", C.c_float), ("env_rotation", C.c_float),
        ("env_visible", C.c_int32), ("pad0", C.c_int32),
        ("env_bg", f3), ("fog_density", C.c_float), ("fog_color", f3), ("fog_anisotropy", C.c_float),
        ("fog_height_min", C.c_float), ("fog_height_max", C.c_float),
        ("camera", AnCamera),
        ("num_volumes", C.c_int32), ("pad1", C.c_int32),
        ("volumes", C.POINTER(AnVolume)), ("voldata", fp),
    ]


class AnSettings(C.Structure):
    _fields_ = [
        ("width", C.c_int32), ("height", C.c_int32), ("spp", C.c_int32), ("min_spp", C.c_int32),
        ("adaptive_threshold", C.c_float),
        ("max_depth", C.c_int32), ("threads", C.c_int32), ("seed", C.c_uint32),
        ("clamp_indirect", C.c_float),
        ("integrator", C.c_int32), ("film_transparent", C.c_int32), ("caustics", C.c_int32),
        ("filter_width", C.c_float), ("ao_distance", C.c_float),
        ("rr_depth", C.c_int32), ("tile_size", C.c_int32),
        ("wire_width", C.c_float),
        ("region_x0", C.c_int32), ("region_y0", C.c_int32), ("region_x1", C.c_int32), ("region_y1", C.c_int32),
        ("exposure_hint", C.c_float),
    ]


PROGRESS = C.CFUNCTYPE(C.c_int, C.c_float)

TEXTURE_TYPES = {"image": 0, "checker": 1, "noise": 2, "marble": 3, "wood": 4, "grid": 5, "gradient": 6,
                 "voronoi": 7, "brick": 8, "stripes": 9, "constant": 10}
INTEGRATORS = {"path": 0, "direct": 1, "ao": 2, "normals": 3, "albedo": 4, "depth": 5, "clay": 6, "uv": 7,
               "wireframe": 8}
PROJECTIONS = {"perspective": 0, "orthographic": 1, "ortho": 1, "equirect": 2, "panorama": 2, "spherical": 2,
               "fisheye": 3}
LIGHT_TYPES = {"point": 0, "spot": 1, "distant": 2, "sun": 2, "directional": 2}


class NativeError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------------
# building
# --------------------------------------------------------------------------------------------
def _lib_suffix():
    if sys.platform == "darwin":
        return ".dylib"
    if sys.platform.startswith("win"):
        return ".dll"
    return ".so"


def _candidate_build_dirs():
    env = os.environ.get("ANIMORA_CACHE")
    if env:
        yield Path(env)
    yield NATIVE_DIR / "build"
    yield Path.home() / ".cache" / "animora"


def find_compiler():
    """Return the C++ compiler command to use (``$ANIMORA_CXX`` / ``$CXX`` / clang++ / g++)."""
    for name in (os.environ.get("ANIMORA_CXX"), os.environ.get("CXX"), "clang++", "g++", "c++"):
        if name and shutil.which(name):
            return shutil.which(name)
    return None


def _flag_sets(cxx):
    """Candidate compiler flag sets, from most to least optimised/specific."""
    base = ["-O3", "-std=c++17", "-shared", "-fPIC", "-fvisibility=hidden"]
    if not sys.platform.startswith("win"):
        base.append("-pthread")
    machine = platform.machine().lower()
    arch = []
    if machine in ("arm64", "aarch64"):
        arch = ["-mcpu=native"]
    elif machine in ("x86_64", "amd64"):
        arch = ["-march=native"]
    sdk_fix = []
    if sys.platform == "darwin":
        # Some Command Line Tools installs ship an incomplete toolchain libc++ header directory
        # that shadows the SDK one. Point the compiler at the SDK's libc++ headers explicitly.
        try:
            sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            sdk = ""
        if sdk and (Path(sdk) / "usr/include/c++/v1").is_dir():
            sdk_fix = ["-nostdinc++", "-isystem", str(Path(sdk) / "usr/include/c++/v1"), "-isysroot", sdk]
    sets = [base + arch, base + arch + sdk_fix, base, base + sdk_fix]
    out, seen = [], set()
    for s in sets:
        k = tuple(s)
        if k not in seen:
            seen.add(k)
            out.append(s)
    return out


def _source_hash():
    h = hashlib.sha1(SOURCE.read_bytes())
    h.update(platform.machine().encode())
    h.update(sys.platform.encode())
    return h.hexdigest()[:12]


def library_path():
    """Path where the compiled library for the current source lives (may not exist yet)."""
    override = os.environ.get("ANIMORA_NATIVE_LIB")
    if override:
        return Path(override)
    name = f"libanimora_{_source_hash()}{_lib_suffix()}"
    for d in _candidate_build_dirs():
        p = d / name
        if p.exists():
            return p
    return next(iter(_candidate_build_dirs())) / name if os.environ.get("ANIMORA_CACHE") else NATIVE_DIR / "build" / name


def build(force=False, verbose=False):
    """Compile the native renderer. Returns the path of the shared library.

    Tries several flag sets (native CPU tuning, macOS SDK libc++ workaround, plain) and keeps
    the first that compiles. Raises :class:`NativeError` with the compiler output on failure.
    """
    if os.environ.get("ANIMORA_NATIVE_LIB"):
        return Path(os.environ["ANIMORA_NATIVE_LIB"])
    name = f"libanimora_{_source_hash()}{_lib_suffix()}"
    if not force:
        for d in _candidate_build_dirs():
            if (d / name).exists():
                return d / name
    cxx = find_compiler()
    if not cxx:
        raise NativeError("no C++ compiler found (install clang++ or g++, or set ANIMORA_CXX)")
    errors = []
    for d in _candidate_build_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            test = d / ".write_test"
            test.write_text("x")
            test.unlink()
        except OSError:
            continue
        out = d / name
        tmp = d / (name + f".tmp{os.getpid()}")
        for flags in _flag_sets(cxx):
            cmd = [cxx, *flags, str(SOURCE), "-o", str(tmp)]
            if verbose:
                print("[animora] " + " ".join(cmd), file=sys.stderr)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and tmp.exists():
                os.replace(tmp, out)
                # remove stale builds of older sources
                for old in d.glob("libanimora_*" + _lib_suffix()):
                    if old != out:
                        try:
                            old.unlink()
                        except OSError:
                            pass
                return out
            errors.append(" ".join(cmd) + "\n" + (r.stderr or r.stdout)[-3000:])
        break
    raise NativeError("failed to compile native renderer:\n" + "\n----\n".join(errors))


_lib = None
_lock = threading.Lock()


def load(auto_build=True):
    """Load (building if needed) the native library and verify the struct layout."""
    global _lib
    with _lock:
        if _lib is not None:
            return _lib
        path = build() if auto_build else library_path()
        if not Path(path).exists():
            raise NativeError(f"native library not found at {path}")
        lib = C.CDLL(str(path))
        lib.an_version.restype = C.c_char_p
        lib.an_struct_sizes.argtypes = [ip]
        lib.an_struct_sizes.restype = C.c_int
        lib.an_render.argtypes = [C.POINTER(AnScene), C.POINTER(AnSettings), fp, fp, fp, fp, fp, ip, PROGRESS]
        lib.an_render.restype = C.c_int
        lib.an_denoise.argtypes = [C.c_int, C.c_int, fp, fp, fp, fp, fp, fp, C.c_int, C.c_float, C.c_float,
                                   C.c_float, C.c_int]
        lib.an_denoise.restype = C.c_int
        lib.an_mesh_sdf.argtypes = [fp, C.c_int, ip, C.c_int, fp, C.c_int, fp, fp, fp, C.c_int]
        lib.an_mesh_sdf.restype = C.c_int
        lib.an_pbf_step.argtypes = [fp, fp, C.c_int, fp, fp, ip, fp, C.c_int]
        lib.an_pbf_step.restype = C.c_int
        lib.an_splat.argtypes = [fp, C.c_int, fp, ip, fp, C.c_float, C.c_int]
        lib.an_splat.restype = C.c_int
        lib.an_self_collide.argtypes = [fp, fp, fp, C.c_int, C.c_float, C.c_float, C.c_int, C.c_int]
        lib.an_self_collide.restype = C.c_int
        u8 = C.POINTER(C.c_uint8)
        lib.an_fluid_advect.argtypes = [fp, fp, fp, ip, C.c_float, fp, fp, C.c_int, C.c_int, u8, C.c_int]
        lib.an_fluid_advect.restype = C.c_int
        lib.an_fluid_vorticity.argtypes = [fp, fp, fp, ip, C.c_float, C.c_int]
        lib.an_fluid_vorticity.restype = C.c_int
        lib.an_fluid_project.argtypes = [fp, fp, fp, fp, ip, u8, fp, C.c_int, C.c_float, C.c_int, C.c_int]
        lib.an_fluid_project.restype = C.c_int
        lib.an_mesh_sdf_vel.argtypes = [fp, C.c_int, ip, C.c_int, fp, fp, C.c_int, C.c_float, fp, fp, C.c_int]
        lib.an_mesh_sdf_vel.restype = C.c_int
        lib.an_fluid_resample.argtypes = [fp, ip, C.c_int, fp, ip, C.c_float, C.c_int, C.c_int]
        lib.an_fluid_resample.restype = C.c_int
        sizes = (C.c_int32 * 7)()
        lib.an_struct_sizes(sizes)
        expected = [C.sizeof(s) for s in (AnTexture, AnMaterial, AnLight, AnCamera, AnScene, AnSettings,
                                                 AnVolume)]
        if list(sizes) != expected:
            raise NativeError(f"struct layout mismatch: native {list(sizes)} vs python {expected}")
        _lib = lib
        return lib


# --------------------------------------------------------------------------------------------
# GPU (Apple Metal) renderer
# --------------------------------------------------------------------------------------------
GPU_SOURCE = NATIVE_DIR / "gpu_metal.mm"
GPU_KERNEL = NATIVE_DIR / "gpu_kernel.msl"
_gpu_lib = None
_gpu_error = None


def _gpu_hash():
    h = hashlib.sha1()
    for f in (GPU_SOURCE, GPU_KERNEL, SOURCE):
        h.update(f.read_bytes())
    h.update(platform.machine().encode())
    return h.hexdigest()[:12]


def gpu_supported():
    """True on platforms with a GPU backend (macOS / Metal)."""
    return sys.platform == "darwin" and GPU_SOURCE.exists() and GPU_KERNEL.exists()


def build_gpu(force=False, verbose=False):
    """Compile the Metal GPU renderer host library (macOS only). Returns its path.

    The Metal kernel itself (``gpu_kernel.msl``) is embedded in the library and compiled by the
    Metal driver on first use (the driver caches the result).
    """
    if not gpu_supported():
        raise NativeError("GPU rendering requires macOS with Metal")
    name = f"libanimoragpu_{_gpu_hash()}{_lib_suffix()}"
    if not force:
        for d in _candidate_build_dirs():
            if (d / name).exists():
                return d / name
    cxx = find_compiler()
    if not cxx:
        raise NativeError("no C++ compiler found")
    errors = []
    for d in _candidate_build_dirs():
        try:
            d.mkdir(parents=True, exist_ok=True)
            (d / ".write_test").write_text("x")
            (d / ".write_test").unlink()
        except OSError:
            continue
        out = d / name
        tmp = d / (name + f".tmp{os.getpid()}")
        for flags in _flag_sets(cxx):
            flags = [f for f in flags if not f.startswith("-mcpu") and not f.startswith("-march")]
            cmd = [cxx, *flags, "-fobjc-arc", "-framework", "Metal", "-framework", "Foundation",
                   str(GPU_SOURCE), "-o", str(tmp)]
            if verbose:
                print("[animora] " + " ".join(cmd), file=sys.stderr)
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode == 0 and tmp.exists():
                os.replace(tmp, out)
                for old in d.glob("libanimoragpu_*" + _lib_suffix()):
                    if old != out:
                        try:
                            old.unlink()
                        except OSError:
                            pass
                return out
            errors.append(" ".join(cmd) + "\n" + (r.stderr or r.stdout)[-3000:])
        break
    raise NativeError("failed to compile GPU renderer:\n" + "\n----\n".join(errors))


def load_gpu(quiet=True):
    """Load the GPU renderer and compile its Metal kernel. Returns the library or ``None``."""
    global _gpu_lib, _gpu_error
    with _lock:
        if _gpu_lib is not None:
            return _gpu_lib
        if _gpu_error is not None:
            return None
        try:
            if not gpu_supported():
                raise NativeError("GPU rendering requires macOS with Metal")
            lib = C.CDLL(str(build_gpu()))
            lib.an_gpu_available.restype = C.c_int
            lib.an_gpu_device_name.restype = C.c_char_p
            lib.an_gpu_last_error.restype = C.c_char_p
            lib.an_gpu_compile.restype = C.c_int
            lib.an_gpu_render.argtypes = [C.POINTER(AnScene), C.POINTER(AnSettings), fp, fp, fp, fp, fp, ip,
                                          PROGRESS]
            lib.an_gpu_render.restype = C.c_int
            if not lib.an_gpu_available():
                raise NativeError("no Metal device")
            rc = lib.an_gpu_compile()
            if rc != 1:
                raise NativeError(f"Metal kernel compile failed: {lib.an_gpu_last_error().decode(errors='replace')}")
            _gpu_lib = lib
            return lib
        except Exception as e:  # noqa: BLE001 - any failure means "no GPU"
            _gpu_error = str(e)
            if not quiet:
                print(f"[animora] GPU renderer unavailable: {e}", file=sys.stderr)
            return None


def gpu_available():
    """True when the Metal GPU renderer can be used on this machine."""
    return load_gpu() is not None


def gpu_info():
    """Dict describing the GPU backend: ``available``, ``device`` name and ``error``."""
    lib = load_gpu()
    if lib is None:
        return {"available": False, "device": None, "error": _gpu_error}
    return {"available": True, "device": lib.an_gpu_device_name().decode(), "error": None}


def available():
    """True when the native renderer can be loaded (compiling it if necessary)."""
    try:
        load()
        return True
    except Exception:
        return False


def version():
    return load().an_version().decode()


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------
def _ptr(a, ctype=C.c_float):
    if a is None:
        return C.cast(None, C.POINTER(ctype))
    return a.ctypes.data_as(C.POINTER(ctype))


def _f32(a):
    return None if a is None else np.ascontiguousarray(a, dtype=np.float32)


def _i32(a):
    return None if a is None else np.ascontiguousarray(a, dtype=np.int32)


def set_vec(field, value):
    for i, v in enumerate(value):
        field[i] = float(v)


class PackedScene:
    """Flat, renderer-ready scene data (all world space, per-triangle-corner arrays).

    Attributes
    ----------
    P0, P1 : (T,3,3) float32 corner positions at shutter open / close (P1 may be None)
    N0, N1 : (T,3,3) shading normals (N1 may be None)
    UV     : (T,3,2) texture coordinates
    PREF   : (T,3,3) rest-space ("orco") positions for procedural textures
    tri_mat, tri_obj : (T,) int32 material index / object id per triangle
    materials : list[AnMaterial];  textures : list[AnTexture];  texdata : float32 array
    lights : list[AnLight];  env_tex : int (index into textures or -1)
    camera : AnCamera;  plus environment & fog scalars
    """

    def __init__(self):
        self.P0 = np.zeros((0, 3, 3), np.float32)
        self.P1 = None
        self.N0 = None
        self.N1 = None
        self.UV = None
        self.PREF = None
        self.tri_mat = np.zeros(0, np.int32)
        self.tri_obj = np.zeros(0, np.int32)
        self.materials = []
        self.textures = []
        self.texdata = np.zeros(1, np.float32)
        self.lights = []
        self.env_tex = -1
        self.env_strength = 1.0
        self.env_rotation = 0.0
        self.env_visible = 1
        self.env_bg = (0.0, 0.0, 0.0)
        self.fog_density = 0.0
        self.fog_color = (1.0, 1.0, 1.0)
        self.fog_anisotropy = 0.0
        self.fog_height_min = 0.0
        self.fog_height_max = 0.0
        self.camera = AnCamera()
        self.volumes = []          # list[AnVolume]
        self.voldata = None        # float32 array referenced by AnVolume offsets

    @property
    def num_tris(self):
        return int(self.P0.shape[0])


def default_material():
    m = AnMaterial()
    set_vec(m.base_color, (0.8, 0.8, 0.8))
    m.roughness = 0.5
    m.ior = 1.5
    m.specular = 0.5
    m.clearcoat_roughness = 0.03
    m.sheen_tint = 0.5
    set_vec(m.subsurface_radius, (1.0, 0.2, 0.1))
    m.subsurface_scale = 0.05
    m.alpha = 1.0
    m.normal_strength = 1.0
    m.bump_strength = 1.0
    m.transmission_roughness = -1.0
    set_vec(m.attenuation_color, (1, 1, 1))
    for k in ("tex_base_color", "tex_roughness", "tex_metallic", "tex_normal", "tex_emission", "tex_alpha",
              "tex_bump", "tex_transmission"):
        setattr(m, k, -1)
    return m


def default_settings(width=640, height=360, spp=64):
    s = AnSettings()
    s.width, s.height, s.spp = width, height, spp
    s.min_spp = 0
    s.adaptive_threshold = 0.01
    s.max_depth = 8
    s.threads = 0
    s.seed = 1
    s.clamp_indirect = 10.0
    s.integrator = 0
    s.film_transparent = 0
    s.caustics = 0
    s.filter_width = 1.5
    s.ao_distance = 1.0
    s.rr_depth = 3
    s.tile_size = 16
    s.wire_width = 1.0
    s.exposure_hint = 1.0
    return s


def render(packed: PackedScene, settings: AnSettings, progress=None, aovs=True, device="cpu"):
    """Render a :class:`PackedScene`. Returns a dict of numpy arrays.

    Keys: ``rgba`` (H,W,4) linear radiance + alpha, and when ``aovs``: ``albedo`` (H,W,3),
    ``normal`` (H,W,3) world-space, ``depth`` (H,W) camera distance, ``variance`` (H,W) luminance
    variance of the mean, ``spp`` (H,W) samples taken. ``cancelled`` is True when the progress
    callback asked to stop. ``device`` is the device that actually rendered (``"cpu"``/``"gpu"``).

    ``device``: ``"cpu"`` (default), ``"gpu"`` (Metal path tracer; falls back to the CPU with a
    warning when unavailable or when the scene contains volumes) or ``"auto"`` (GPU when available,
    silently falling back).
    """
    lib = load()
    W, H = int(settings.width), int(settings.height)
    T = packed.num_tris
    P0 = _f32(packed.P0.reshape(-1, 9)) if T else np.zeros((1, 9), np.float32)
    P1 = _f32(packed.P1.reshape(-1, 9)) if (packed.P1 is not None and T) else None
    N0 = _f32(packed.N0.reshape(-1, 9)) if (packed.N0 is not None and T) else None
    N1 = _f32(packed.N1.reshape(-1, 9)) if (packed.N1 is not None and T) else None
    UV = _f32(packed.UV.reshape(-1, 6)) if (packed.UV is not None and T) else None
    PREF = _f32(packed.PREF.reshape(-1, 9)) if (packed.PREF is not None and T) else None
    tri_mat = _i32(packed.tri_mat) if T else np.zeros(1, np.int32)
    tri_obj = _i32(packed.tri_obj) if T else np.zeros(1, np.int32)
    nm = max(1, len(packed.materials))
    mats = (AnMaterial * nm)(*(packed.materials or [default_material()]))
    nt = len(packed.textures)
    texs = (AnTexture * max(1, nt))(*packed.textures) if nt else (AnTexture * 1)()
    texdata = _f32(packed.texdata if packed.texdata is not None and packed.texdata.size else np.zeros(1))
    nl = len(packed.lights)
    lights = (AnLight * max(1, nl))(*packed.lights) if nl else (AnLight * 1)()

    sc = AnScene()
    sc.num_tris = T
    sc.has_motion = 1 if (P1 is not None or N1 is not None or _camera_moves(packed.camera)) else 0
    sc.P0, sc.P1, sc.N0, sc.N1, sc.UV, sc.PREF = _ptr(P0), _ptr(P1), _ptr(N0), _ptr(N1), _ptr(UV), _ptr(PREF)
    sc.tri_mat, sc.tri_obj = _ptr(tri_mat, C.c_int32), _ptr(tri_obj, C.c_int32)
    sc.num_materials = len(packed.materials) if packed.materials else 1
    sc.num_textures = nt
    sc.materials = C.cast(mats, C.POINTER(AnMaterial))
    sc.textures = C.cast(texs, C.POINTER(AnTexture))
    sc.texdata = _ptr(texdata)
    sc.num_lights = nl
    sc.env_tex = int(packed.env_tex)
    sc.lights = C.cast(lights, C.POINTER(AnLight))
    sc.env_strength = float(packed.env_strength)
    sc.env_rotation = float(packed.env_rotation)
    sc.env_visible = int(packed.env_visible)
    set_vec(sc.env_bg, packed.env_bg)
    sc.fog_density = float(packed.fog_density)
    set_vec(sc.fog_color, packed.fog_color)
    sc.fog_anisotropy = float(packed.fog_anisotropy)
    sc.fog_height_min = float(packed.fog_height_min)
    sc.fog_height_max = float(packed.fog_height_max)
    sc.camera = packed.camera
    vols = getattr(packed, "volumes", None) or []
    voldata = _f32(packed.voldata if getattr(packed, "voldata", None) is not None and packed.voldata.size
                   else np.zeros(1))
    vol_arr = (AnVolume * max(1, len(vols)))(*vols) if vols else (AnVolume * 1)()
    sc.num_volumes = len(vols)
    sc.volumes = C.cast(vol_arr, C.POINTER(AnVolume))
    sc.voldata = _ptr(voldata)

    rgba = np.zeros((H, W, 4), np.float32)
    albedo = np.zeros((H, W, 3), np.float32) if aovs else None
    normal = np.zeros((H, W, 3), np.float32) if aovs else None
    depth = np.zeros((H, W), np.float32) if aovs else None
    variance = np.zeros((H, W), np.float32) if aovs else None
    spp = np.zeros((H, W), np.int32) if aovs else None

    def _cb(f):
        if progress is None:
            return 0
        try:
            return 1 if progress(float(f)) else 0
        except KeyboardInterrupt:
            return 1
        except Exception:
            return 0

    cb = PROGRESS(_cb)
    args = (C.byref(sc), C.byref(settings), _ptr(rgba), _ptr(albedo), _ptr(normal), _ptr(depth),
            _ptr(variance), _ptr(spp, C.c_int32), cb)
    used = "cpu"
    rc = None
    device = (device or "cpu").lower()
    if device in ("gpu", "auto", "metal"):
        glib = load_gpu()
        if glib is not None:
            rc = glib.an_gpu_render(*args)
            if rc >= 0:
                used = "gpu"
            else:
                if device != "auto" or rc == -4:
                    why = glib.an_gpu_last_error().decode(errors="replace")
                    print(f"[animora] GPU render unavailable ({why.splitlines()[0] if why else rc}); "
                          "using the CPU", file=sys.stderr)
                rc = None
        elif device != "auto":
            print(f"[animora] no GPU renderer ({_gpu_error or 'unsupported platform'}); using the CPU",
                  file=sys.stderr)
    if rc is None:
        rc = lib.an_render(*args)
    if rc < 0:
        raise NativeError(f"an_render failed with code {rc}")
    out = {"rgba": rgba, "cancelled": rc == 1, "device": used}
    if aovs:
        out.update(albedo=albedo, normal=normal, depth=depth, variance=variance, spp=spp)
    return out


def _camera_moves(cam):
    a = np.array([*cam.pos0, *cam.fwd0, *cam.up0], np.float32)
    b = np.array([*cam.pos1, *cam.fwd1, *cam.up1], np.float32)
    return bool(np.any(np.abs(a - b) > 1e-7))


def denoise(rgb, albedo, normal, depth, variance=None, iterations=5, sigma_l=4.0, sigma_n=128.0, sigma_z=1.0,
            threads=0):
    """Edge-aware a-trous (SVGF-style) denoise of an (H,W,3) linear image guided by AOVs."""
    lib = load()
    H, W = rgb.shape[:2]
    rgb = _f32(rgb[..., :3])
    albedo = _f32(albedo)
    normal = _f32(normal)
    depth = _f32(depth)
    variance = _f32(variance) if variance is not None else None
    out = np.zeros((H, W, 3), np.float32)
    rc = lib.an_denoise(W, H, _ptr(rgb), _ptr(albedo), _ptr(normal), _ptr(depth), _ptr(variance), _ptr(out),
                        int(iterations), float(sigma_l), float(sigma_n), float(sigma_z), int(threads))
    if rc != 0:
        raise NativeError(f"an_denoise failed with code {rc}")
    return out


def mesh_sdf(verts, tris, points, winding=False, closest=False, threads=0):
    """Signed distance of ``points`` (N,3) to the triangle mesh (verts (V,3), tris (T,3)).

    Negative inside (sign from the generalised winding number). Returns ``dist`` or a tuple
    ``(dist, winding, closest_points)`` with the requested extras."""
    lib = load()
    V = np.ascontiguousarray(verts, np.float32).reshape(-1, 3)
    T = np.ascontiguousarray(tris, np.int32).reshape(-1, 3)
    P = np.ascontiguousarray(points, np.float32).reshape(-1, 3)
    d = np.zeros(len(P), np.float32)
    w = np.zeros(len(P), np.float32) if winding else None
    cp = np.zeros((len(P), 3), np.float32) if closest else None
    as_p = lambda a: None if a is None else a.ctypes.data_as(fp)  # noqa: E731
    r = lib.an_mesh_sdf(as_p(V), len(V), T.ctypes.data_as(ip), len(T), as_p(P), len(P), as_p(d), as_p(w), as_p(cp),
                        int(threads))
    if r:
        raise NativeError("mesh_sdf: empty mesh")
    if not winding and not closest:
        return d
    return (d,) + ((w,) if winding else ()) + ((cp,) if closest else ())


def pbf_step(pos, vel, params, sdf=None, sdf_lo=None, sdf_cell=1.0, threads=0):
    """Advance a Position-Based-Fluids particle set one step in place.

    ``pos``/``vel`` are contiguous float32 (N,3) arrays (modified in place); ``params`` is the
    21-float parameter block documented in pathtracer.cpp; ``sdf`` an optional collider distance
    grid indexed [z, y, x] with origin ``sdf_lo`` and voxel size ``sdf_cell``."""
    lib = load()
    assert pos.dtype == np.float32 and pos.flags.c_contiguous and vel.dtype == np.float32
    prm = np.ascontiguousarray(params, np.float32)
    if sdf is not None:
        sdf = np.ascontiguousarray(sdf, np.float32)
        info = np.array([sdf.shape[2], sdf.shape[1], sdf.shape[0]], np.int32)
        xf = np.array(list(sdf_lo) + [sdf_cell], np.float32)
        return lib.an_pbf_step(_ptr(pos), _ptr(vel), len(pos), _ptr(prm), _ptr(sdf), info.ctypes.data_as(ip),
                               _ptr(xf), int(threads))
    return lib.an_pbf_step(_ptr(pos), _ptr(vel), len(pos), _ptr(prm), None, None, None, int(threads))


def splat(points, dims, lo, cell, radius, threads=0):
    """Splat points (N,3) into a zero grid of shape (nz, ny, nx) with kernel (1-r^2/R^2)^3."""
    lib = load()
    P = np.ascontiguousarray(points, np.float32).reshape(-1, 3)
    nx, ny, nz = (int(d) for d in dims)
    g = np.zeros((nz, ny, nx), np.float32)
    d = np.array([nx, ny, nz], np.int32)
    xf = np.array(list(lo) + [cell], np.float32)
    lib.an_splat(_ptr(P), len(P), _ptr(g), d.ctypes.data_as(ip), _ptr(xf), float(radius), int(threads))
    return g


def self_collide(pos, prev, rest, thickness, rest_ignore, iterations=2, threads=0):
    """Push apart cloth vertices closer than ``thickness`` (in place). Returns the contact count."""
    lib = load()
    assert pos.dtype == np.float32 and pos.flags.c_contiguous
    pv = np.ascontiguousarray(prev, np.float32)
    rs = np.ascontiguousarray(rest, np.float32)
    return lib.an_self_collide(_ptr(pos), _ptr(pv), _ptr(rs), len(pos), float(thickness), float(rest_ignore),
                               int(iterations), int(threads))


# ---------------------------------------------------------------- Eulerian smoke kernels
def _u8p(a):
    return None if a is None else a.ctypes.data_as(C.POINTER(C.c_uint8))


def _grid_dims(a):
    return np.array([a.shape[2], a.shape[1], a.shape[0]], np.int32)


def fluid_advect(u, v, w, fields, dt=1.0, maccormack=True, solid=None, threads=0):
    """Advect a list of [z,y,x] float32 grids through (u, v, w) (grid units/step). Returns a list."""
    lib = load()
    u, v, w = (np.ascontiguousarray(a, np.float32) for a in (u, v, w))
    src = np.ascontiguousarray(np.stack(fields), np.float32)
    dst = np.empty_like(src)
    sol = None if solid is None else np.ascontiguousarray(solid, np.uint8)
    lib.an_fluid_advect(_ptr(u), _ptr(v), _ptr(w), _grid_dims(u).ctypes.data_as(ip), float(dt), _ptr(src),
                        _ptr(dst), len(fields), 1 if maccormack else 0, _u8p(sol), int(threads))
    return list(dst)


def fluid_vorticity(u, v, w, eps, threads=0):
    """Vorticity confinement in place on contiguous float32 grids."""
    lib = load()
    lib.an_fluid_vorticity(_ptr(u), _ptr(v), _ptr(w), _grid_dims(u).ctypes.data_as(ip), float(eps), int(threads))


def fluid_project(u, v, w, p, solid=None, solid_vel=None, iterations=60, omega=1.7, floor=True, threads=0):
    """Pressure projection in place (red-black SOR, Neumann at ``solid`` cells moving with ``solid_vel``)."""
    lib = load()
    sol = None if solid is None else np.ascontiguousarray(solid, np.uint8)
    sv = None if solid_vel is None else np.ascontiguousarray(solid_vel, np.float32)
    lib.an_fluid_project(_ptr(u), _ptr(v), _ptr(w), _ptr(p), _grid_dims(u).ctypes.data_as(ip), _u8p(sol),
                         None if sv is None else _ptr(sv), int(iterations), float(omega), int(bool(floor)),
                         int(threads))


def mesh_sdf_vel(verts, tris, vert_vel, points, threads=0):
    """Signed distance and surface velocity (barycentric at the closest point) of ``points``."""
    lib = load()
    V = np.ascontiguousarray(verts, np.float32).reshape(-1, 3)
    T = np.ascontiguousarray(tris, np.int32).reshape(-1, 3)
    VV = np.ascontiguousarray(vert_vel, np.float32).reshape(-1, 3)
    P = np.ascontiguousarray(points, np.float32).reshape(-1, 3)
    d = np.zeros(len(P), np.float32)
    vel = np.zeros((len(P), 3), np.float32)
    r = lib.an_mesh_sdf_vel(_ptr(V), len(V), T.ctypes.data_as(ip), len(T), _ptr(VV), _ptr(P), len(P), 0.0,
                            _ptr(d), _ptr(vel), int(threads))
    if r:
        raise NativeError("mesh_sdf_vel: empty mesh")
    return d, vel


def fluid_resample(coarse, fine_shape, scale=1.0, threads=0):
    """Trilinearly stretch coarse grids (nf, cz, cy, cx) over ``fine_shape`` (nz, ny, nx)."""
    lib = load()
    src = np.ascontiguousarray(coarse, np.float32)
    nf = src.shape[0]
    out = np.empty((nf,) + tuple(fine_shape), np.float32)
    cd = np.array([src.shape[3], src.shape[2], src.shape[1]], np.int32)
    fd = np.array([fine_shape[2], fine_shape[1], fine_shape[0]], np.int32)
    lib.an_fluid_resample(_ptr(src), cd.ctypes.data_as(ip), nf, _ptr(out), fd.ctypes.data_as(ip), float(scale), 0,
                          int(threads))
    return out
