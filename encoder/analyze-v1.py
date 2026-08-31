#!/usr/bin/env python3
"""narc attempt v1 — block MC + half-pel + two refs + J = D + λR.

Closed-loop origin encode of the Big Buck Bunny 90s window.
v0 (global integer translation, 10 fps, frame-mean skip) is frozen;
this script must not overwrite public/media/v0/.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace") if Path("/workspace/public/media").exists() else Path(__file__).resolve().parents[1]
MEDIA = ROOT / "public" / "media" if (ROOT / "public" / "media").exists() else ROOT / "media"
LAB_JSON = ROOT / "src" / "lib" / "analysis-data.json"
TMP = Path("/tmp/bbb")
CLIP = MEDIA / "source.mp4"
FRAMES = TMP / "frames-v1"
RECON = TMP / "recon-v1"
RESID = TMP / "resid-v1"
STRIP = MEDIA / "strip"
KEYS = MEDIA / "keys"
HEATS = MEDIA / "heat"
BLOCKS = MEDIA / "blocks"

ATTEMPT = "v1-block-mc"
START_SEC = 50
DURATION_SEC = 90
ANALYSIS_FPS = 24
W, H = 320, 180
BW, BH = 16, 16
COLS, ROWS = W // BW, H // BH  # 20 × 11, leftover 4 rows
MOTION_W, MOTION_H = 80, 45
SEARCH_GLOBAL = 8
SEARCH_BLOCK = 2
LAMBDA = 0.12
SKIP_MAE = 2.6
INTRA_MAE = 16.0
MAX_GOP = 72  # 3.0s @ 24fps
MIN_GOP = 8
RESIDUAL_BUDGET = 90.0
CUT_HIST = 0.62
KEY_JPEG_Q = 84
RESID_JPEG_Q = 52


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


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    if mse < 1e-8:
        return 99.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 82) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(path, "JPEG", quality=quality, optimize=True)
    return path.stat().st_size


def down_luma(y: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(y.astype(np.uint8)).resize((MOTION_W, MOTION_H), Image.BILINEAR))


def best_translation(a: np.ndarray, b: np.ndarray, radius: int = SEARCH_GLOBAL) -> tuple[int, int, float]:
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


def warp_float(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Shift content by (dx, dy). Uncovered pixels are 0."""
    im = Image.fromarray(img)
    fill = 0 if img.ndim == 2 else (0, 0, 0)
    out = im.transform(
        im.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=fill,
    )
    return np.array(out)


def halfpel_refine(prev_small: np.ndarray, cur_small: np.ndarray, dx: int, dy: int) -> tuple[float, float]:
    """Refine global integer MV to half-pel on the 80×45 grid."""
    best = 1e18
    bx, by = float(dx), float(dy)
    cur = cur_small.astype(np.float32)
    for hy in (-0.5, 0.0, 0.5):
        for hx in (-0.5, 0.0, 0.5):
            warped = warp_float(prev_small, dx + hx, dy + hy).astype(np.float32)
            r = SEARCH_GLOBAL
            mae = float(np.abs(warped[r:-r, r:-r] - cur[r:-r, r:-r]).mean())
            if mae < best:
                best = mae
                bx, by = dx + hx, dy + hy
    return bx, by


def block_mae(err: np.ndarray) -> np.ndarray:
    e = err[: ROWS * BH, : COLS * BW]
    return e.reshape(ROWS, BH, COLS, BW).mean(axis=(1, 3))


def expand_blocks(blocks: np.ndarray, channels: int | None = None) -> np.ndarray:
    up = np.repeat(np.repeat(blocks, BH, axis=0), BW, axis=1)
    if channels is None:
        out = np.zeros((H, W), dtype=blocks.dtype)
        out[: ROWS * BH, : COLS * BW] = up
        if H > ROWS * BH:
            out[ROWS * BH :, : COLS * BW] = np.repeat(blocks[-1], BW)
        return out
    out = np.zeros((H, W, channels), dtype=blocks.dtype)
    out[: ROWS * BH, : COLS * BW] = np.repeat(np.repeat(blocks, BH, axis=0), BW, axis=1)
    return out


def local_search(pred_y: np.ndarray, cur_y: np.ndarray, radius: int = SEARCH_BLOCK) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer block MVs around identity (pred is already globally warped)."""
    pad = radius
    pp = np.pad(pred_y.astype(np.int16), pad, mode="edge")
    cc = cur_y.astype(np.int16)
    best = np.full((ROWS, COLS), 1e18, dtype=np.float64)
    bdx = np.zeros((ROWS, COLS), dtype=np.int16)
    bdy = np.zeros((ROWS, COLS), dtype=np.int16)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            warped = pp[pad + dy : pad + dy + H, pad + dx : pad + dx + W]
            mae = block_mae(np.abs(warped - cc).astype(np.float32))
            better = mae < best
            best = np.where(better, mae, best)
            bdx = np.where(better, dx, bdx)
            bdy = np.where(better, dy, bdy)
    return bdx.astype(np.int16), bdy.astype(np.int16), best


def warp_blocks_rgb(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Per-block integer translate of an already globally warped image."""
    out = img.copy()
    h, w = img.shape[:2]
    for r in range(ROWS):
        y0 = r * BH
        y1 = y0 + BH
        for c in range(COLS):
            x0 = c * BW
            x1 = x0 + BW
            dxi = int(dx[r, c])
            dyi = int(dy[r, c])
            if dxi == 0 and dyi == 0:
                continue
            sy0, sx0 = y0 - dyi, x0 - dxi
            sy1, sx1 = sy0 + BH, sx0 + BW
            if sy0 < 0 or sx0 < 0 or sy1 > h or sx1 > w:
                out[y0:y1, x0:x1] = 0
                continue
            out[y0:y1, x0:x1] = img[sy0:sy1, sx0:sx1]
    if H > ROWS * BH:
        out[ROWS * BH :] = img[ROWS * BH :]
    return out


def uncovered_mask(dx: float, dy: float) -> np.ndarray:
    mask = np.zeros((ROWS, COLS), dtype=bool)
    ix, iy = int(round(dx)), int(round(dy))
    if ix > 0:
        mask[:, : max(1, ix // BW)] = True
    elif ix < 0:
        mask[:, COLS - max(1, (-ix) // BW) :] = True
    if iy > 0:
        mask[: max(1, iy // BH), :] = True
    elif iy < 0:
        mask[ROWS - max(1, (-iy) // BH) :, :] = True
    return mask


def choose_modes(mae: np.ndarray, uncovered: np.ndarray) -> np.ndarray:
    """0 skip, 1 residual, 2 intra. J = D + λR with a tiny rate model."""
    modes = np.zeros(mae.shape, dtype=np.uint8)
    r_resid = 28.0
    r_intra = 72.0
    j_skip = mae
    j_resid = np.minimum(mae * 0.35 + 1.8, mae) + LAMBDA * r_resid
    j_intra = LAMBDA * r_intra + 1.2
    modes = np.where(j_resid < j_skip, 1, modes)
    modes = np.where((j_intra < j_skip) & (j_intra < j_resid) & (mae >= INTRA_MAE), 2, modes)
    modes = np.where(mae <= SKIP_MAE, 0, modes)
    modes = np.where(uncovered & (mae > 8.0), 2, modes)
    return modes


def residual_jpeg(pred: np.ndarray, cur: np.ndarray, skip: np.ndarray, path: Path) -> tuple[int, np.ndarray]:
    diff = cur.astype(np.int16) - pred.astype(np.int16)
    packed = np.clip(diff + 128, 0, 255).astype(np.uint8)
    skip_px = expand_blocks(skip.astype(np.uint8))
    packed[skip_px == 1] = 128
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(packed).save(path, "JPEG", quality=RESID_JPEG_Q, optimize=True)
    decoded = np.array(Image.open(path).convert("RGB")).astype(np.int16) - 128
    decoded[skip_px == 1] = 0
    return path.stat().st_size, decoded


def extract() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    for p in FRAMES.glob("*"):
        p.unlink()
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(CLIP),
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


def predict(ref_rgb: np.ndarray, cur_rgb: np.ndarray, ref_y_small: np.ndarray, cur_y_small: np.ndarray, cur_y: np.ndarray) -> dict:
    dx_i, dy_i, _ = best_translation(ref_y_small, cur_y_small)
    dx, dy = halfpel_refine(ref_y_small, cur_y_small, dx_i, dy_i)
    scale = W / MOTION_W
    dx_f, dy_f = dx * scale, dy * scale
    glob = warp_float(ref_rgb, dx_f, dy_f)
    glob_y = luma(glob)
    ldx, ldy, mae = local_search(glob_y, cur_y)
    pred = warp_blocks_rgb(glob, ldx, ldy)
    unc = uncovered_mask(dx_f, dy_f)
    modes = choose_modes(mae, unc)
    return {
        "dx": dx_f,
        "dy": dy_f,
        "ldx": ldx,
        "ldy": ldy,
        "mae": mae,
        "pred": pred,
        "modes": modes,
        "uncovered": float(unc.mean()),
        "mean_mae": float(mae.mean()),
    }


def classify(flux: float, motion: float, residual: float, occ: float, hist: float, skip_frac: float, intra_frac: float) -> str:
    if hist < CUT_HIST and flux > 12:
        return "cut"
    if intra_frac > 0.18:
        return "residual"
    if skip_frac > 0.88 and residual < 4.0 and motion < 0.8:
        return "static"
    if motion >= 1.2 and residual < 8:
        return "motion"
    if residual >= 8 or occ >= 0.08:
        return "residual"
    if flux < 3.5 and motion < 0.5:
        return "static"
    return "motion"


def encode(rgbs: list[np.ndarray]) -> dict:
    shutil.rmtree(RECON, ignore_errors=True)
    shutil.rmtree(RESID, ignore_errors=True)
    shutil.rmtree(KEYS, ignore_errors=True)
    shutil.rmtree(STRIP, ignore_errors=True)
    shutil.rmtree(HEATS, ignore_errors=True)
    shutil.rmtree(BLOCKS, ignore_errors=True)
    RECON.mkdir(parents=True)
    RESID.mkdir(parents=True)
    KEYS.mkdir(parents=True)
    STRIP.mkdir(parents=True)
    HEATS.mkdir(parents=True)
    BLOCKS.mkdir(parents=True)

    n = len(rgbs)
    y_full = [luma(im) for im in rgbs]
    y_small = [down_luma(y) for y in y_full]
    hists = [hist16(y) for y in y_full]

    recon: list[np.ndarray | None] = [None] * n
    last_key_i = 0
    last_key_small = y_small[0]
    acc = 0.0
    rows = []
    key_bytes = resid_bytes = motion_bytes = intra_bytes = 0
    psnrs: list[float] = []
    skip_total = resid_total = intra_total = 0
    n_blocks = ROWS * COLS

    for i, rgb in enumerate(rgbs):
        if i == 0:
            rec = rgb.copy()
            recon[i] = rec
            key_bytes += save_jpg(KEYS / f"{i:04d}.jpg", rgb, KEY_JPEG_Q)
            save_jpg(RECON / f"{i:04d}.jpg", rec, KEY_JPEG_Q)
            psnrs.append(psnr(rec, rgb))
            rows.append(
                {
                    "i": 0,
                    "t": 0.0,
                    "flux": 0.0,
                    "motion": 0.0,
                    "residual": 0.0,
                    "occlusion": 0.0,
                    "hist": 1.0,
                    "luma": round(float(y_full[0].mean()), 2),
                    "lumaJump": 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "kind": "keyframe",
                    "key": True,
                    "cut": False,
                    "storedResidual": False,
                    "skipBlocks": n_blocks,
                    "residBlocks": 0,
                    "intraBlocks": 0,
                    "ref": 0,
                    "psnr": round(psnrs[-1], 2),
                }
            )
            continue

        flux = float(np.abs(rgb.astype(np.int16) - rgbs[i - 1].astype(np.int16)).mean())
        hc = hist_corr(hists[i - 1], hists[i])
        luma_jump = float(y_full[i].mean() - y_full[i - 1].mean())
        prev = recon[i - 1]
        assert prev is not None
        prev_small = down_luma(luma(prev))

        cand = predict(prev, rgb, prev_small, y_small[i], y_full[i])
        ref_used = 0
        if i - last_key_i >= 2:
            key_rgb = recon[last_key_i]
            assert key_rgb is not None
            alt = predict(key_rgb, rgb, last_key_small, y_small[i], y_full[i])
            if alt["mean_mae"] + 0.4 < cand["mean_mae"]:
                cand = alt
                ref_used = 1

        modes = cand["modes"]
        mae = cand["mae"]
        pred = cand["pred"]
        skip = modes == 0
        resid_m = modes == 1
        intra_m = modes == 2
        n_skip = int(skip.sum())
        n_resid = int(resid_m.sum())
        n_intra = int(intra_m.sum())
        occ = float(cand["uncovered"])
        motion = float((cand["dx"] ** 2 + cand["dy"] ** 2) ** 0.5)
        pred_resid = float(cand["mean_mae"])
        kind = classify(flux, motion, pred_resid, occ, hc, n_skip / n_blocks, n_intra / n_blocks)
        cut = kind == "cut"

        gap = i - last_key_i
        e = pred_resid + 40.0 * occ + 6.0 * (n_intra / n_blocks)
        if kind == "motion":
            e *= 0.4
        if kind == "static":
            e *= 0.2
        acc += e
        want_key = False
        if cut and gap >= 4:
            want_key = True
        elif gap >= MAX_GOP:
            want_key = True
        elif acc >= RESIDUAL_BUDGET and gap >= MIN_GOP:
            want_key = True
        elif n_intra / n_blocks > 0.42 and gap >= MIN_GOP:
            want_key = True

        stored = False
        if want_key:
            rec = rgb.copy()
            key_bytes += save_jpg(KEYS / f"{i:04d}.jpg", rgb, KEY_JPEG_Q)
            kind = "cut" if cut else "keyframe"
            last_key_i = i
            last_key_small = y_small[i]
            acc = 0.0
            n_skip, n_resid, n_intra = n_blocks, 0, 0
            is_key = True
        else:
            rec = pred.copy()
            if n_resid > 0:
                b, decoded = residual_jpeg(pred, rgb, skip | intra_m, RESID / f"{i:04d}.jpg")
                rec = np.clip(pred.astype(np.int16) + decoded, 0, 255).astype(np.uint8)
                resid_bytes += b
                stored = True
            if n_intra > 0:
                intra_px = expand_blocks(intra_m.astype(np.uint8))
                rec[intra_px == 1] = rgb[intra_px == 1]
                intra_bytes += n_intra * 48
            if H > ROWS * BH:
                strip = rgb[ROWS * BH :]
                if float(np.abs(strip.astype(np.int16) - rec[ROWS * BH :].astype(np.int16)).mean()) > SKIP_MAE:
                    rec[ROWS * BH :] = strip
            motion_bytes += 4 + n_blocks * 2
            is_key = False

        residual = float(np.abs(rgb.astype(np.int16) - rec.astype(np.int16)).mean())
        recon[i] = rec
        save_jpg(RECON / f"{i:04d}.jpg", rec, KEY_JPEG_Q)
        psnrs.append(psnr(rec, rgb))
        skip_total += n_skip
        resid_total += n_resid
        intra_total += n_intra

        rows.append(
            {
                "i": i,
                "t": round(i / ANALYSIS_FPS, 4),
                "flux": round(flux, 3),
                "motion": round(motion, 3),
                "residual": round(residual, 3),
                "occlusion": round(occ, 4),
                "hist": round(hc, 4),
                "luma": round(float(y_full[i].mean()), 2),
                "lumaJump": round(luma_jump, 2),
                "dx": round(float(cand["dx"]), 2),
                "dy": round(float(cand["dy"]), 2),
                "kind": kind,
                "key": is_key,
                "cut": cut,
                "storedResidual": stored,
                "skipBlocks": n_skip,
                "residBlocks": n_resid,
                "intraBlocks": n_intra,
                "ref": ref_used,
                "psnr": round(psnrs[-1], 2),
            }
        )

        if i % 120 == 0:
            print(f"  frame {i}/{n} psnr={psnrs[-1]:.1f} key={is_key} skip={n_skip} resid={n_resid} intra={n_intra}")

        if is_key or cut or residual > 10:
            draw_heat(rgb, pred if not is_key else rgb, modes if not is_key else np.zeros((ROWS, COLS), np.uint8), i)

    # shots
    cuts = [0] + [r["i"] for r in rows if r["cut"] and r["key"]] + [n]
    shots = []
    for a, b in zip(cuts, cuts[1:]):
        if b <= a:
            continue
        sl = rows[a:b]
        mean_m = float(np.mean([x["motion"] for x in sl])) if sl else 0
        mean_r = float(np.mean([x["residual"] for x in sl])) if sl else 0
        if mean_m < 0.6 and mean_r < 5:
            skind = "locked"
        elif mean_m >= 1.4 and mean_r < 10:
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

    for i in range(0, n, ANALYSIS_FPS):
        im = Image.fromarray(rgbs[i]).resize((160, 90), Image.BILINEAR)
        im.save(STRIP / f"{i:04d}.jpg", "JPEG", quality=70, optimize=True)

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

    v0_path = MEDIA / "v0" / "stats.json"
    v0 = json.loads(v0_path.read_text())["stats"] if v0_path.exists() else None
    model_bytes = key_bytes + resid_bytes + motion_bytes + intra_bytes
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1

    stats = {
        "attempt": ATTEMPT,
        "frames": n,
        "fps": ANALYSIS_FPS,
        "width": W,
        "height": H,
        "block": [BW, BH],
        "blocksPerFrame": n_blocks,
        "duration": n / ANALYSIS_FPS,
        "startSec": START_SEC,
        "shots": len(shots),
        "keyframes": sum(1 for f in rows if f["key"]),
        "cuts": sum(1 for f in rows if f["cut"]),
        "residualsStored": sum(1 for f in rows if f["storedResidual"]),
        "sourceBytes": CLIP.stat().st_size,
        "modelBytes": model_bytes,
        "keyframeBytes": key_bytes,
        "residualBytes": resid_bytes,
        "motionBytes": motion_bytes,
        "intraBytes": intra_bytes,
        "rawBytes": n * W * H * 3,
        "reconstructMp4Bytes": (MEDIA / "reconstruct.mp4").stat().st_size,
        "meanFlux": round(float(np.mean([f["flux"] for f in rows])), 3),
        "meanResidual": round(float(np.mean([f["residual"] for f in rows])), 3),
        "meanMotion": round(float(np.mean([f["motion"] for f in rows])), 3),
        "meanPsnr": round(float(np.mean(psnrs)), 2),
        "skipBlockFrac": round(skip_total / (n * n_blocks), 4),
        "residBlockFrac": round(resid_total / (n * n_blocks), 4),
        "intraBlockFrac": round(intra_total / (n * n_blocks), 4),
        "ratioVsRaw": round((n * W * H * 3) / max(model_bytes, 1), 2),
        "ratioVsSource": round(CLIP.stat().st_size / max(model_bytes, 1), 2),
        "kinds": kinds,
        "lambda": LAMBDA,
        "skipMae": SKIP_MAE,
        "baseline": (
            {
                "attempt": "v0-global-translation",
                "fps": v0["fps"],
                "frames": v0["frames"],
                "keyframes": v0["keyframes"],
                "residualsStored": v0["residualsStored"],
                "modelBytes": v0["modelBytes"],
                "meanResidual": v0["meanResidual"],
                "reconstructMp4Bytes": v0["reconstructMp4Bytes"],
            }
            if v0
            else None
        ),
    }
    return {
        "attempt": ATTEMPT,
        "frames": rows,
        "shots": shots,
        "stats": stats,
        "source": {
            "clip": "/media/source.mp4",
            "reconstruct": "/media/reconstruct.mp4",
            "reconstructV0": "/media/v0/reconstruct.mp4",
            "scope": "/media/scope.png",
            "duration": DURATION_SEC,
            "startSec": START_SEC,
            "title": "Big Buck Bunny",
            "credit": "Blender Foundation / peach.blender.org  ·  CC BY 3.0",
            "window": f"{START_SEC}s – {START_SEC + DURATION_SEC}s",
        },
    }


def draw_heat(cur: np.ndarray, pred: np.ndarray, modes: np.ndarray, i: int) -> None:
    resid = np.abs(cur.astype(np.int16) - pred.astype(np.int16)).mean(axis=2)
    norm = np.clip(resid / 48.0, 0, 1)
    heat = np.zeros((H, W, 3), dtype=np.uint8)
    heat[..., 0] = (40 + 180 * norm).astype(np.uint8)
    heat[..., 1] = (30 + 40 * (1 - norm)).astype(np.uint8)
    heat[..., 2] = (36 + 20 * (1 - norm)).astype(np.uint8)
    blend = (0.42 * cur + 0.58 * heat).astype(np.uint8)
    # block tint: skip none, residual copper edge, intra bone edge
    vis = blend.copy()
    for r in range(ROWS):
        for c in range(COLS):
            m = int(modes[r, c])
            if m == 0:
                continue
            y0, x0 = r * BH, c * BW
            color = (196, 92, 74) if m == 1 else (228, 224, 214)
            vis[y0 : y0 + 1, x0 : x0 + BW] = color
            vis[y0 + BH - 1 : y0 + BH, x0 : x0 + BW] = color
            vis[y0 : y0 + BH, x0 : x0 + 1] = color
            vis[y0 : y0 + BH, x0 + BW - 1 : x0 + BW] = color
    Image.fromarray(vis).save(HEATS / f"{i:04d}.jpg", "JPEG", quality=78)


def draw_scope(rows: list[dict]) -> None:
    n = len(rows)
    scope = np.zeros((96, n, 3), dtype=np.uint8)
    max_r = max(x["residual"] for x in rows) or 1
    max_f = max(x["flux"] for x in rows) or 1
    for i, f in enumerate(rows):
        h_f = int((f["flux"] / max_f) * 90)
        h_r = int((f["residual"] / max_r) * 90)
        scope[96 - h_f : 96, i] = (46, 52, 58)
        scope[96 - h_r : 96, i, 0] = np.maximum(scope[96 - h_r : 96, i, 0], 168)
        scope[96 - h_r : 96, i, 1] = np.maximum(scope[96 - h_r : 96, i, 1] // 2, 48)
        if f["key"]:
            scope[:, i] = (228, 224, 214)
        elif f["cut"]:
            scope[:, i] = (180, 90, 70)
    Image.fromarray(scope).resize((min(n, 1800), 96), Image.NEAREST).save(MEDIA / "scope.png")


def main() -> None:
    print("extracting 24fps analysis frames")
    extract()
    rgbs = load_frames()
    print(f"loaded {len(rgbs)} frames")
    data = encode(rgbs)
    draw_scope(data["frames"])
    out = MEDIA / "analysis.json"
    out.write_text(json.dumps(data))
    if LAB_JSON.parent.exists():
        LAB_JSON.write_text(json.dumps(data))
    print(json.dumps(data["stats"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
