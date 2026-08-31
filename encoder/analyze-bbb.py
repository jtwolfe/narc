#!/usr/bin/env python3
"""Flux-aware encode of a Big Buck Bunny clip.

Decomposes inter-frame change into motion / residual / occlusion-like
maps, places keyframes from unexplained energy (not raw flux), and
writes a reconstructed mezzanine from keyframe + warp + residual.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "media"
TMP = Path("/tmp/bbb")
SRC_FULL = TMP / "source.mp4"
CLIP = MEDIA / "source.mp4"
FRAMES = TMP / "frames"
RECON = TMP / "recon"
RESID = TMP / "resid"
STRIP = MEDIA / "strip"
KEYS = MEDIA / "keys"
HEATS = MEDIA / "heat"

# 90s window: forest, butterfly, bunny waking — mixed flux regimes.
START_SEC = 50
DURATION_SEC = 90
ANALYSIS_FPS = 10
W, H = 320, 180
MOTION_W, MOTION_H = 80, 45
SEARCH = 8
MAX_GOP = 40  # 4.0s
MIN_GOP = 6
RESIDUAL_BUDGET = 55.0  # integrated unexplained energy before a new key
RESID_STORE_THRESH = 7.0
CUT_HIST = 0.62
CUT_FLUX_RATIO = 2.4


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def luma(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def hist16(y: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(y, bins=16, range=(0, 255))
    s = h.sum() or 1
    return h.astype(np.float64) / s


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / d)


def best_translation(a: np.ndarray, b: np.ndarray, radius: int = SEARCH) -> tuple[int, int, float]:
    """Integer translation (dx, dy) that warps a toward b. a,b uint8 luma."""
    h, w = a.shape
    aa = a.astype(np.int16)
    bb = b.astype(np.int16)
    best_dx, best_dy, best = 0, 0, 1e18
    inner_y = slice(radius, h - radius)
    inner_x = slice(radius, w - radius)
    ref = bb[inner_y, inner_x]
    for dy in range(-radius, radius + 1):
        y0 = inner_y.start + dy
        y1 = inner_y.stop + dy
        for dx in range(-radius, radius + 1):
            x0 = inner_x.start + dx
            x1 = inner_x.stop + dx
            sad = float(np.abs(aa[y0:y1, x0:x1] - ref).mean())
            if sad < best:
                best = sad
                best_dx, best_dy = dx, dy
    return best_dx, best_dy, best


def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    h, w = img.shape[:2]
    out = np.zeros_like(img)
    src_y0, src_y1 = max(0, -dy), min(h, h - dy)
    src_x0, src_x1 = max(0, -dx), min(w, w - dx)
    if src_y1 <= src_y0 or src_x1 <= src_x0:
        return out
    dst_y0 = src_y0 + dy
    dst_x0 = src_x0 + dx
    out[dst_y0 : dst_y0 + (src_y1 - src_y0), dst_x0 : dst_x0 + (src_x1 - src_x0)] = img[
        src_y0:src_y1, src_x0:src_x1
    ]
    return out


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 82) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=quality, optimize=True)
    return path.stat().st_size


def residual_jpeg(warped: np.ndarray, cur: np.ndarray, path: Path) -> tuple[int, np.ndarray]:
    """Store (cur-warped)+128 as a small jpeg; return bytes and decoded residual."""
    diff = cur.astype(np.int16) - warped.astype(np.int16)
    packed = np.clip(diff + 128, 0, 255).astype(np.uint8)
    small = np.array(Image.fromarray(packed).resize((MOTION_W, MOTION_H), Image.BILINEAR))
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(small).save(path, "JPEG", quality=38, optimize=True)
    decoded_small = np.array(Image.open(path).convert("RGB"))
    decoded = np.array(Image.fromarray(decoded_small).resize((W, H), Image.BILINEAR)).astype(np.int16) - 128
    return path.stat().st_size, decoded


def extract() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    MEDIA.mkdir(parents=True, exist_ok=True)
    for p in FRAMES.glob("*"):
        p.unlink()
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(START_SEC),
            "-i",
            str(SRC_FULL),
            "-t",
            str(DURATION_SEC),
            "-vf",
            "scale=640:360",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-b:a",
            "96k",
            "-movflags",
            "+faststart",
            str(CLIP),
        ]
    )
    run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            str(START_SEC),
            "-i",
            str(SRC_FULL),
            "-t",
            str(DURATION_SEC),
            "-vf",
            f"fps={ANALYSIS_FPS},scale={W}:{H}",
            "-q:v",
            "3",
            str(FRAMES / "%04d.jpg"),
        ]
    )


def load_frames() -> list[np.ndarray]:
    files = sorted(FRAMES.glob("*.jpg"))
    return [np.array(Image.open(p).convert("RGB")) for p in files]


def classify(flux: float, motion: float, residual: float, occ: float, hist: float, luma_jump: float) -> str:
    if hist < CUT_HIST and flux > 12:
        return "cut"
    if abs(luma_jump) > 28 and hist > 0.75 and motion < 1.2:
        return "flash"
    if motion < 0.45 and residual > 9 and occ < 0.08:
        return "grain"
    if motion >= 1.4 and residual < 10:
        return "motion"
    if residual >= 12 or occ >= 0.12:
        return "residual"
    if flux < 4.5 and motion < 0.6:
        return "static"
    return "motion"


def analyze(rgbs: list[np.ndarray]) -> dict:
    n = len(rgbs)
    y_full = [luma(im) for im in rgbs]
    y_small = [
        np.array(Image.fromarray(y.astype(np.uint8)).resize((MOTION_W, MOTION_H), Image.BILINEAR))
        for y in y_full
    ]
    hists = [hist16(y) for y in y_full]

    rows = []
    for i in range(n):
        if i == 0:
            rows.append(
                {
                    "i": 0,
                    "t": 0.0,
                    "flux": 0.0,
                    "motion": 0.0,
                    "residual": 0.0,
                    "occlusion": 0.0,
                    "hist": 1.0,
                    "luma": float(y_full[0].mean()),
                    "lumaJump": 0.0,
                    "dx": 0,
                    "dy": 0,
                    "kind": "keyframe",
                    "key": True,
                    "cut": False,
                    "storedResidual": False,
                }
            )
            continue
        prev, cur = rgbs[i - 1], rgbs[i]
        flux = float(np.abs(cur.astype(np.int16) - prev.astype(np.int16)).mean())
        dx, dy, _ = best_translation(y_small[i - 1], y_small[i])
        motion = float((dx * dx + dy * dy) ** 0.5)
        # scale translation from 80x45 → 320x180
        dx_f, dy_f = dx * (W // MOTION_W), dy * (H // MOTION_H)
        warped = translate(prev, dx_f, dy_f)
        resid_map = np.abs(cur.astype(np.int16) - warped.astype(np.int16)).mean(axis=2)
        residual = float(resid_map.mean())
        occ = float((resid_map > 38).mean())
        # uncovered after warp
        if dx_f or dy_f:
            mask = np.zeros((H, W), dtype=bool)
            if dy_f > 0:
                mask[:dy_f, :] = True
            elif dy_f < 0:
                mask[dy_f:, :] = True
            if dx_f > 0:
                mask[:, :dx_f] = True
            elif dx_f < 0:
                mask[:, dx_f:] = True
            occ = max(occ, float(mask.mean()))
        hc = hist_corr(hists[i - 1], hists[i])
        luma_jump = float(y_full[i].mean() - y_full[i - 1].mean())
        kind = classify(flux, motion, residual, occ, hc, luma_jump)
        cut = kind == "cut"
        rows.append(
            {
                "i": i,
                "t": i / ANALYSIS_FPS,
                "flux": round(flux, 3),
                "motion": round(motion, 3),
                "residual": round(residual, 3),
                "occlusion": round(occ, 4),
                "hist": round(hc, 4),
                "luma": round(float(y_full[i].mean()), 2),
                "lumaJump": round(luma_jump, 2),
                "dx": int(dx_f),
                "dy": int(dy_f),
                "kind": kind,
                "key": False,
                "cut": cut,
                "storedResidual": False,
            }
        )

    # Place keyframes from unexplained energy, not raw flux.
    rows[0]["key"] = True
    last_key = 0
    acc = 0.0
    for i in range(1, n):
        r = rows[i]
        gap = i - last_key
        e = r["residual"] + 40.0 * r["occlusion"]
        if r["kind"] == "motion":
            e *= 0.35  # explained by warp — cheap
        if r["kind"] == "grain":
            e *= 0.25
        if r["kind"] == "flash":
            e *= 0.2
        acc += e
        want = False
        if r["cut"] and gap >= 3:
            want = True
        elif gap >= MAX_GOP:
            want = True
        elif acc >= RESIDUAL_BUDGET and gap >= MIN_GOP:
            want = True
        if want:
            r["key"] = True
            r["kind"] = "cut" if r["cut"] else "keyframe"
            last_key = i
            acc = 0.0

    # Shots
    cuts = [0] + [r["i"] for r in rows if r["cut"] and r["key"]] + [n]
    shots = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        sl = rows[a:b]
        mean_m = float(np.mean([x["motion"] for x in sl])) if sl else 0
        mean_r = float(np.mean([x["residual"] for x in sl])) if sl else 0
        if mean_m < 0.5 and mean_r < 8:
            skind = "locked"
        elif mean_m >= 1.6 and mean_r < 12:
            skind = "tracking"
        else:
            skind = "busy"
        shots.append(
            {
                "i0": a,
                "i1": b,
                "t0": a / ANALYSIS_FPS,
                "t1": b / ANALYSIS_FPS,
                "kind": skind,
                "keys": sum(1 for x in sl if x["key"]),
            }
        )

    return {"frames": rows, "shots": shots}


def encode_model(rgbs: list[np.ndarray], data: dict) -> dict:
    shutil.rmtree(RECON, ignore_errors=True)
    shutil.rmtree(RESID, ignore_errors=True)
    shutil.rmtree(KEYS, ignore_errors=True)
    shutil.rmtree(STRIP, ignore_errors=True)
    shutil.rmtree(HEATS, ignore_errors=True)
    RECON.mkdir(parents=True)
    RESID.mkdir(parents=True)
    KEYS.mkdir(parents=True)
    STRIP.mkdir(parents=True)
    HEATS.mkdir(parents=True)

    frames = data["frames"]
    n = len(rgbs)
    recon = [None] * n
    key_bytes = 0
    resid_bytes = 0
    motion_bytes = 0

    for i, rgb in enumerate(rgbs):
        f = frames[i]
        if f["key"]:
            recon[i] = rgb.copy()
            key_bytes += save_jpg(KEYS / f"{i:04d}.jpg", rgb, quality=84)
            save_jpg(RECON / f"{i:04d}.jpg", rgb, quality=84)
            f["storedResidual"] = False
            motion_bytes += 2
            continue
        prev = recon[i - 1]
        warped = translate(prev, f["dx"], f["dy"])
        motion_bytes += 2
        resid_mean = f["residual"]
        if resid_mean >= RESID_STORE_THRESH:
            b, decoded = residual_jpeg(warped, rgb, RESID / f"{i:04d}.jpg")
            resid_bytes += b
            rec = np.clip(warped.astype(np.int16) + decoded, 0, 255).astype(np.uint8)
            f["storedResidual"] = True
        else:
            rec = warped
            f["storedResidual"] = False
        recon[i] = rec
        save_jpg(RECON / f"{i:04d}.jpg", rec, quality=84)

    # Strip thumbs every 0.5s
    for i in range(0, n, 5):
        im = Image.fromarray(rgbs[i]).resize((160, 90), Image.BILINEAR)
        im.save(STRIP / f"{i:04d}.jpg", "JPEG", quality=70, optimize=True)

    # Heat maps at keys and cuts
    for f in frames:
        if not (f["key"] or f["cut"]):
            continue
        i = f["i"]
        if i == 0:
            continue
        prev, cur = rgbs[i - 1], rgbs[i]
        warped = translate(prev, f["dx"], f["dy"])
        resid = np.abs(cur.astype(np.int16) - warped.astype(np.int16)).mean(axis=2)
        norm = np.clip(resid / 48.0, 0, 1)
        heat = np.zeros((H, W, 3), dtype=np.uint8)
        heat[..., 0] = (40 + 180 * norm).astype(np.uint8)
        heat[..., 1] = (30 + 40 * (1 - norm)).astype(np.uint8)
        heat[..., 2] = (36 + 20 * (1 - norm)).astype(np.uint8)
        blend = (0.45 * cur + 0.55 * heat).astype(np.uint8)
        Image.fromarray(blend).save(HEATS / f"{i:04d}.jpg", "JPEG", quality=78)

    # Scope strip: residual over time
    scope = np.zeros((96, n, 3), dtype=np.uint8)
    max_r = max(x["residual"] for x in frames) or 1
    max_f = max(x["flux"] for x in frames) or 1
    max_m = max(x["motion"] for x in frames) or 1
    for i, f in enumerate(frames):
        rf = f["residual"] / max_r
        ff = f["flux"] / max_f
        mf = f["motion"] / max_m
        h_r = int(rf * 90)
        h_f = int(ff * 90)
        h_m = int(mf * 90)
        scope[96 - h_f : 96, i] = (46, 52, 58)
        scope[96 - h_r : 96, i, 0] = np.maximum(scope[96 - h_r : 96, i, 0], 168)
        scope[96 - h_r : 96, i, 1] = np.maximum(scope[96 - h_r : 96, i, 1] // 2, 48)
        if f["key"]:
            scope[:, i] = (228, 224, 214)
        elif f["cut"]:
            scope[:, i] = (180, 90, 70)
    Image.fromarray(scope).resize((n * 2, 96), Image.NEAREST).save(MEDIA / "scope.png")

    run(
        [
            "ffmpeg",
            "-y",
            "-framerate",
            str(ANALYSIS_FPS),
            "-i",
            str(RECON / "%04d.jpg"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "18",
            "-an",
            "-movflags",
            "+faststart",
            str(MEDIA / "reconstruct.mp4"),
        ]
    )

    source_bytes = CLIP.stat().st_size
    recon_mp4 = (MEDIA / "reconstruct.mp4").stat().st_size
    model_bytes = key_bytes + resid_bytes + motion_bytes
    raw_bytes = n * W * H * 3

    stats = {
        "frames": n,
        "fps": ANALYSIS_FPS,
        "width": W,
        "height": H,
        "duration": n / ANALYSIS_FPS,
        "startSec": START_SEC,
        "shots": len(data["shots"]),
        "keyframes": sum(1 for f in frames if f["key"]),
        "cuts": sum(1 for f in frames if f["cut"]),
        "residualsStored": sum(1 for f in frames if f["storedResidual"]),
        "sourceBytes": source_bytes,
        "modelBytes": model_bytes,
        "keyframeBytes": key_bytes,
        "residualBytes": resid_bytes,
        "motionBytes": motion_bytes,
        "rawBytes": raw_bytes,
        "reconstructMp4Bytes": recon_mp4,
        "meanFlux": round(float(np.mean([f["flux"] for f in frames])), 3),
        "meanResidual": round(float(np.mean([f["residual"] for f in frames])), 3),
        "meanMotion": round(float(np.mean([f["motion"] for f in frames])), 3),
        "ratioVsRaw": round(raw_bytes / max(model_bytes, 1), 2),
        "ratioVsSource": round(source_bytes / max(model_bytes, 1), 2),
    }
    data["stats"] = stats
    data["source"] = {
        "clip": "/media/source.mp4",
        "reconstruct": "/media/reconstruct.mp4",
        "scope": "/media/scope.png",
        "duration": DURATION_SEC,
        "startSec": START_SEC,
        "title": "Big Buck Bunny",
        "credit": "Blender Foundation / peach.blender.org  ·  CC BY 3.0",
        "window": f"{START_SEC}s – {START_SEC + DURATION_SEC}s",
    }
    return data


def main() -> None:
    extract()
    rgbs = load_frames()
    print(f"loaded {len(rgbs)} frames")
    data = analyze(rgbs)
    data = encode_model(rgbs, data)
    out = MEDIA / "analysis.json"
    out.write_text(json.dumps(data))
    print(json.dumps(data["stats"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
