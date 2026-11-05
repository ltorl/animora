"""Video encoding of rendered image sequences through ``ffmpeg`` (a command-line tool, no UI).

Codecs (``codec`` / ``settings.video.codec``)::

    h264    MP4, libx264, yuv420p - plays everywhere (default)
    h265    MP4, libx265 (hvc1 tag for QuickTime) - ~40% smaller
    prores  MOV, ProRes 422 HQ (prores_ks profile 3) - editing / grading master
    prores4444  MOV, ProRes 4444 with alpha
    webm    WebM, VP9 (yuva420p when alpha) - web
    av1     MP4, libsvtav1 / libaom-av1 when available
    gif     GIF with an optimised palette (palettegen / paletteuse)
    png     MOV with PNG frames (lossless, alpha)
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

CODECS = ("h264", "h265", "prores", "prores4444", "webm", "av1", "gif", "png")
EXT = {"h264": ".mp4", "h265": ".mp4", "prores": ".mov", "prores4444": ".mov", "webm": ".webm", "av1": ".mp4",
       "gif": ".gif", "png": ".mov"}
QUALITY_CRF = {"draft": 28, "low": 26, "medium": 22, "high": 18, "best": 14, "lossless": 0}


class EncodeError(RuntimeError):
    pass


def find_ffmpeg():
    p = os.environ.get("ANIMORA_FFMPEG") or shutil.which("ffmpeg")
    return p


def ffmpeg_encoders():
    ff = find_ffmpeg()
    if not ff:
        return set()
    try:
        out = subprocess.run([ff, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=20).stdout
    except Exception:  # noqa: BLE001
        return set()
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(parts[0]) == 6:
            names.add(parts[1])
    return names


def codec_for_path(path, default="h264"):
    ext = Path(path).suffix.lower()
    return {".mp4": default if default in ("h264", "h265", "av1") else "h264", ".m4v": "h264", ".mov": "prores",
            ".webm": "webm", ".gif": "gif", ".mkv": default}.get(ext, default)


def build_command(pattern, out, fps=24, start_number=1, codec=None, crf=None, quality="high", audio=None,
                  audio_offset=0.0, alpha=False, scale=None, preset="slow", bitrate=None, extra=None, loop=0,
                  ffmpeg=None):
    """Return the ffmpeg argument list that encodes ``pattern`` (ffmpeg %04d form) into ``out``."""
    ff = ffmpeg or find_ffmpeg() or "ffmpeg"
    codec = codec or codec_for_path(out)
    if codec not in CODECS:
        raise EncodeError(f"unknown codec {codec!r}; available: {CODECS}")
    if crf is None:
        crf = QUALITY_CRF.get(quality, 18)
    cmd = [ff, "-y", "-hide_banner", "-loglevel", "error", "-framerate", f"{fps:g}", "-start_number",
           str(int(start_number)), "-i", str(pattern)]
    if audio:
        if audio_offset:
            cmd += ["-itsoffset", f"{audio_offset:g}"]
        cmd += ["-i", str(audio)]
    vf = []
    if scale:
        vf.append(f"scale={scale}:flags=lanczos" if isinstance(scale, str) else f"scale={int(scale)}:-2:flags=lanczos")
    if codec == "gif":
        vf_s = ",".join(vf + ["split[a][b]", "[a]palettegen=stats_mode=diff[p]", "[b][p]paletteuse=dither=sierra2_4a"])
        cmd += ["-filter_complex", vf_s, "-loop", str(int(loop))]
        cmd += [str(out)]
        return cmd
    if codec in ("h264", "h265", "av1"):
        # even dimensions are required for 4:2:0
        vf.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
    if vf:
        cmd += ["-vf", ",".join(vf)]
    if codec == "h264":
        cmd += ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart"]
        if crf == 0:
            cmd[cmd.index("yuv420p")] = "yuv444p"
    elif codec == "h265":
        cmd += ["-c:v", "libx265", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p", "-tag:v", "hvc1",
                "-movflags", "+faststart", "-x265-params", "log-level=error"]
    elif codec == "av1":
        enc = "libsvtav1" if "libsvtav1" in ffmpeg_encoders() else "libaom-av1"
        cmd += ["-c:v", enc, "-crf", str(min(63, crf + 10)), "-pix_fmt", "yuv420p"]
        if enc == "libaom-av1":
            cmd += ["-b:v", "0", "-cpu-used", "4"]
    elif codec == "prores":
        cmd += ["-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le", "-vendor", "apl0"]
    elif codec == "prores4444":
        cmd += ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le" if alpha else "yuv444p10le",
                "-vendor", "apl0"]
    elif codec == "webm":
        cmd += ["-c:v", "libvpx-vp9", "-crf", str(min(63, crf + 12)), "-b:v", "0", "-row-mt", "1",
                "-pix_fmt", "yuva420p" if alpha else "yuv420p"]
    elif codec == "png":
        cmd += ["-c:v", "png", "-pix_fmt", "rgba" if alpha else "rgb24"]
    if bitrate:
        cmd += ["-b:v", str(bitrate)]
    if audio:
        acodec = {"webm": "libopus", "prores": "pcm_s16le", "prores4444": "pcm_s16le", "png": "pcm_s16le"}.get(codec, "aac")
        cmd += ["-c:a", acodec]
        if acodec == "aac":
            cmd += ["-b:a", "192k"]
        cmd += ["-map", "0:v:0", "-map", "1:a:0", "-shortest"]
    if extra:
        cmd += list(extra)
    cmd += [str(out)]
    return cmd


def sequence_frames(pattern):
    """Sorted frame numbers of the files on disk matching ``pattern`` (``####`` / ``%04d``)."""
    import glob
    import re
    pat = str(pattern)
    m = re.search(r"#+|%0?\d*d", pat)
    if not m:
        return []
    head, tail = pat[:m.start()], pat[m.end():]
    rx = re.compile(re.escape(os.path.basename(head)) + r"(-?\d+)" + re.escape(tail) + "$")
    frames = []
    for f in glob.glob(glob.escape(head) + "*" + glob.escape(tail)):
        mm = rx.match(os.path.basename(f))
        if mm:
            frames.append(int(mm.group(1)))
    return sorted(set(frames))


def encode(pattern, out, fps=24, start_number=1, verbose=False, **kw):
    """Encode an image sequence into a video. ``pattern`` uses ``####`` or ``%04d``. Returns ``out``.

    Gaps in the numbering (e.g. a sequence rendered with ``-f 1-96x4``) are handled: the existing
    frames from ``start_number`` on are linked into a temporary contiguous sequence first."""
    import tempfile
    from .render.renderer import ffmpeg_pattern, frame_path
    ff = find_ffmpeg()
    if not ff:
        raise EncodeError("ffmpeg not found on PATH (install it, e.g. `brew install ffmpeg`, or set ANIMORA_FFMPEG)")
    first = frame_path(pattern, start_number)
    if not os.path.exists(first):
        raise EncodeError(f"first frame not found: {first}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    frames = [f for f in sequence_frames(pattern) if f >= start_number]
    tmp = None
    src_pattern, src_start = ffmpeg_pattern(pattern), start_number
    if frames and frames != list(range(frames[0], frames[0] + len(frames))):
        tmp = tempfile.mkdtemp(prefix="animora_enc_")
        ext = Path(first).suffix
        for i, f in enumerate(frames):
            src = os.path.abspath(frame_path(pattern, f))
            dst = os.path.join(tmp, f"f_{i + 1:06d}{ext}")
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copyfile(src, dst)
        src_pattern, src_start = os.path.join(tmp, f"f_%06d{ext}"), 1
    cmd = build_command(src_pattern, out, fps=fps, start_number=src_start, ffmpeg=ff, **kw)
    if verbose:
        print(" ".join(cmd))
    try:
        r = subprocess.run(cmd, capture_output=True, text=True)
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if r.returncode != 0:
        raise EncodeError(f"ffmpeg failed ({r.returncode}):\n{r.stderr.strip()[-2000:]}")
    return str(out)


def encode_from_settings(scene, pattern, frames, verbose=False):
    """Encode using ``settings.video`` (a path string or a dict of :func:`build_command` keys)."""
    v = scene.settings.get("video")
    if not v:
        return None
    if isinstance(v, str):
        v = {"path": v}
    v = dict(v)
    out = scene.resolve_path(v.pop("path", v.pop("file", "render/animation.mp4")))
    if "audio" in v and v["audio"]:
        v["audio"] = scene.resolve_path(v["audio"])
    v.setdefault("alpha", bool(scene.settings.get("film_transparent")))
    fps = float(v.pop("fps", scene.fps)) / float(scene.settings.get("frame_step", 1) or 1)
    start = int(min(frames))
    return encode(pattern, out, fps=fps, start_number=start, verbose=verbose, **v)


def probe(path):
    """Return basic stream info of a video via ffprobe (dict) or None."""
    fp = shutil.which("ffprobe")
    if not fp:
        return None
    r = subprocess.run([fp, "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                        "stream=codec_name,width,height,r_frame_rate,nb_read_frames,pix_fmt", "-of", "json", str(path)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    import json
    s = json.loads(r.stdout).get("streams", [])
    return s[0] if s else None
