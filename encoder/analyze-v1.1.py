#!/usr/bin/env python3
"""narc attempt v1.1 — same block-MC stack as v1, with a closed motion loop.

Fixes the inverted MV sign, black OOB tiles, leftover 4px strip, skip-hole
JPEG residual, and mode decisions that never saw the pixels that landed in
recon. v0 and v1 media are frozen under public/media/v0 and public/media/v1.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace") if Path("/workspace/public/media").exists() else Path(__file__).resolve().parents[1]
MEDIA = ROOT / "public" / "media" if (ROOT / "public" / "media").exists() else ROOT / "media"
LAB_JSON = ROOT / "src" / "lib" / "analysis-data.json"
TMP = Path("/tmp/bbb")
CLIP = MEDIA / "source.mp4"
FRAMES = TMP / "frames-v1"
RECON = TMP / "recon-v11"
RESID = TMP / "resid-v11"
STRIP = MEDIA / "thumbs"
KEYS = MEDIA / "anchors"
HEATS = MEDIA / "heatmaps"

ATTEMPT = "v1.1-mc-correct"
START_SEC = 50
DURATION_SEC = 90
ANALYSIS_FPS = 24
W, H_DISP = 320, 180
H = 192  # pad to 12 × 16; crop back to 180 for display / PSNR
BW, BH = 16, 16
COLS, ROWS = W // BW, H // BH  # 20 × 12
MOTION_W, MOTION_H = W // 4, H // 4  # 80 × 48
SEARCH_GLOBAL = 8
SEARCH_COARSE = 2  # on 4× downsampled = ±8 full-res
SEARCH_FINE = 2
LAMBDA = 0.12
SKIP_MAE = 2.6
INTRA_MAE = 16.0
HOLE_FRAC = 0.22
MAX_GOP = 72
MIN_GOP = 8
RESIDUAL_BUDGET = 90.0
CUT_HIST = 0.62
KEY_JPEG_Q = 84
RESID_JPEG_Q = 52
INTRA_JPEG_Q = 78


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def luma(rgb: np.ndarray) -> np.ndarray:
    r, g, b = rgb[..., 0].astype(np.float32), rgb[..., 1].astype(np.float32), rgb[..., 2].astype(np.float32)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def hist16(y: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(y[:H_DISP], bins=16, range=(0, 255))
    s = h.sum() or 1
    return h.astype(np.float64) / s


def hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    d = float(np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
    return float(np.dot(a, b) / d)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    aa = a[:H_DISP].astype(np.float32)
    bb = b[:H_DISP].astype(np.float32)
    mse = float(np.mean((aa - bb) ** 2))
    if mse < 1e-8:
        return 99.0
    return float(10.0 * np.log10(255.0 * 255.0 / mse))


def save_jpg(path: Path, rgb: np.ndarray, quality: int = 82) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[:H_DISP] if rgb.shape[0] > H_DISP else rgb).save(
        path, "JPEG", quality=quality, optimize=True
    )
    return path.stat().st_size


def pad_frame(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] >= H:
        return rgb[:H]
    return np.pad(rgb, ((0, H - rgb.shape[0]), (0, 0), (0, 0)), mode="edge")


def down_luma(y: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(y.astype(np.uint8)).resize((MOTION_W, MOTION_H), Image.BILINEAR))


def best_translation(a: np.ndarray, b: np.ndarray, radius: int = SEARCH_GLOBAL) -> tuple[int, int, float]:
    """Content-displacement search: pred[x] = a[x - dx] should match b[x]."""
    h, w = a.shape
    aa = a.astype(np.int16)
    bb = b.astype(np.int16)
    best_dx, best_dy, best = 0, 0, 1e18
    inner_y = slice(radius, h - radius)
    inner_x = slice(radius, w - radius)
    ref = bb[inner_y, inner_x]
    for dy in range(-radius, radius + 1):
        y0 = inner_y.start - dy
        y1 = inner_y.stop - dy
        for dx in range(-radius, radius + 1):
            x0 = inner_x.start - dx
            x1 = inner_x.stop - dx
            sad = float(np.abs(aa[y0:y1, x0:x1] - ref).mean())
            if sad < best:
                best = sad
                best_dx, best_dy = dx, dy
    return best_dx, best_dy, best


def warp_float(img: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    """pred[x,y] = img[x - dx, y - dy], bilinear, edge-extend. Mask is True where source was in-frame."""
    h, w = img.shape[:2]
    pad = int(np.ceil(max(abs(dx), abs(dy), 1.0))) + 2
    if img.ndim == 2:
        padded = np.pad(img, pad, mode="edge")
    else:
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
    im = Image.fromarray(padded)
    fill = 0 if img.ndim == 2 else (0, 0, 0)
    out = im.transform(
        im.size,
        Image.AFFINE,
        (1, 0, -dx, 0, 1, -dy),
        resample=Image.BILINEAR,
        fillcolor=fill,
    )
    arr = np.array(out)
    if img.ndim == 2:
        cropped = arr[pad : pad + h, pad : pad + w]
    else:
        cropped = arr[pad : pad + h, pad : pad + w]
    yy, xx = np.mgrid[0:h, 0:w]
    sx = xx - dx
    sy = yy - dy
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return cropped, valid


def halfpel_refine(prev_small: np.ndarray, cur_small: np.ndarray, dx: int, dy: int) -> tuple[float, float]:
    best = 1e18
    bx, by = float(dx), float(dy)
    cur = cur_small.astype(np.float32)
    r = SEARCH_GLOBAL
    for hy in (-0.5, 0.0, 0.5):
        for hx in (-0.5, 0.0, 0.5):
            warped, _ = warp_float(prev_small, dx + hx, dy + hy)
            mae = float(np.abs(warped[r:-r, r:-r].astype(np.float32) - cur[r:-r, r:-r]).mean())
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
        return up
    return np.repeat(np.repeat(blocks, BH, axis=0), BW, axis=1)


def local_search(pred_y: np.ndarray, cur_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hierarchical block MVs. Convention: pred_block[x] = ref[x - dx] (content +dx)."""
    pred_s = down_luma(pred_y)
    cur_s = down_luma(cur_y)
    pad = SEARCH_COARSE
    pp = np.pad(pred_s.astype(np.int16), pad, mode="edge")
    cc = cur_s.astype(np.int16)
    best = np.full((ROWS, COLS), 1e18, dtype=np.float64)
    cdx = np.zeros((ROWS, COLS), dtype=np.int16)
    cdy = np.zeros((ROWS, COLS), dtype=np.int16)
    hs, ws = pred_s.shape
    bw, bh = 4, 4

    def coarse_mae(err: np.ndarray) -> np.ndarray:
        e = err[: ROWS * bh, : COLS * bw]
        return e.reshape(ROWS, bh, COLS, bw).mean(axis=(1, 3))

    for dy in range(-SEARCH_COARSE, SEARCH_COARSE + 1):
        for dx in range(-SEARCH_COARSE, SEARCH_COARSE + 1):
            warped = pp[pad - dy : pad - dy + hs, pad - dx : pad - dx + ws]
            mae = coarse_mae(np.abs(warped - cc).astype(np.float32))
            better = mae < best
            best = np.where(better, mae, best)
            cdx = np.where(better, dx, cdx)
            cdy = np.where(better, dy, cdy)
    # scale 4×-down MVs to full-res pixels
    cdx = (cdx * 4).astype(np.int16)
    cdy = (cdy * 4).astype(np.int16)

    radius = SEARCH_FINE
    extra = int(max(np.abs(cdx).max(), np.abs(cdy).max(), 1)) + radius + 2
    pf = np.pad(pred_y.astype(np.int16), extra, mode="edge")
    cf = cur_y.astype(np.int16)
    fine_best = np.full((ROWS, COLS), 1e18, dtype=np.float64)
    fdx = cdx.copy()
    fdy = cdy.copy()
    for r in range(ROWS):
        y0 = r * BH
        for c in range(COLS):
            x0 = c * BW
            bx, by = int(cdx[r, c]), int(cdy[r, c])
            tgt = cf[y0 : y0 + BH, x0 : x0 + BW]
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    sy = extra + y0 - (by + dy)
                    sx = extra + x0 - (bx + dx)
                    src = pf[sy : sy + BH, sx : sx + BW]
                    mae = float(np.abs(src - tgt).mean())
                    if mae < fine_best[r, c]:
                        fine_best[r, c] = mae
                        fdx[r, c] = bx + dx
                        fdy[r, c] = by + dy
    return fdx.astype(np.int16), fdy.astype(np.int16), fine_best


def median_mv(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    px = np.pad(dx.astype(np.int16), 1, mode="edge")
    py = np.pad(dy.astype(np.int16), 1, mode="edge")
    ox = np.empty_like(dx, dtype=np.int16)
    oy = np.empty_like(dy, dtype=np.int16)
    for r in range(ROWS):
        for c in range(COLS):
            ox[r, c] = int(np.median(px[r : r + 3, c : c + 3]))
            oy[r, c] = int(np.median(py[r : r + 3, c : c + 3]))
    return ox, oy


def warp_blocks_rgb(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Per-block translate with edge pad. Never writes 0."""
    extra = int(max(np.abs(dx).max(), np.abs(dy).max(), 1)) + 2
    if img.ndim == 2:
        padded = np.pad(img, extra, mode="edge")
    else:
        padded = np.pad(img, ((extra, extra), (extra, extra), (0, 0)), mode="edge")
    out = img.copy()
    for r in range(ROWS):
        y0 = r * BH
        for c in range(COLS):
            x0 = c * BW
            dxi = int(dx[r, c])
            dyi = int(dy[r, c])
            if dxi == 0 and dyi == 0:
                continue
            sy = extra + y0 - dyi
            sx = extra + x0 - dxi
            out[y0 : y0 + BH, x0 : x0 + BW] = padded[sy : sy + BH, sx : sx + BW]
    return out


def choose_modes(mae: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """0 skip, 1 residual, 2 intra. Decided on the pixels that will be in recon."""
    modes = np.zeros(mae.shape, dtype=np.uint8)
    r_resid = 28.0
    r_intra = 72.0
    j_skip = mae
    j_resid = np.minimum(mae * 0.35 + 1.8, mae) + LAMBDA * r_resid
    j_intra = LAMBDA * r_intra + 1.2
    modes = np.where(j_resid < j_skip, 1, modes)
    modes = np.where((j_intra < j_skip) & (j_intra < j_resid) & (mae >= INTRA_MAE), 2, modes)
    modes = np.where(mae <= SKIP_MAE, 0, modes)
    modes = np.where(hole, 2, modes)
    return modes


def pack_atlas(src: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]]]:
    coords = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    n = len(coords)
    if n == 0:
        return np.zeros((BH, BW, 3), dtype=np.uint8), coords
    cols_a = int(np.ceil(np.sqrt(n)))
    rows_a = int(np.ceil(n / cols_a))
    atlas = np.zeros((rows_a * BH, cols_a * BW, 3), dtype=np.uint8)
    for k, (r, c) in enumerate(coords):
        ar, ac = divmod(k, cols_a)
        atlas[ar * BH : (ar + 1) * BH, ac * BW : (ac + 1) * BW] = src[r * BH : (r + 1) * BH, c * BW : (c + 1) * BW]
    return atlas, coords


def scatter_atlas(atlas: np.ndarray, coords: list[tuple[int, int]], shape: tuple[int, ...], bias: int = 0) -> np.ndarray:
    out = np.zeros(shape, dtype=np.int16)
    if not coords:
        return out
    cols_a = max(1, atlas.shape[1] // BW)
    for k, (r, c) in enumerate(coords):
        ar, ac = divmod(k, cols_a)
        tile = atlas[ar * BH : (ar + 1) * BH, ac * BW : (ac + 1) * BW]
        if tile.shape[0] != BH or tile.shape[1] != BW:
            continue
        out[r * BH : (r + 1) * BH, c * BW : (c + 1) * BW] = tile.astype(np.int16) + bias
    return out


def residual_atlas(pred: np.ndarray, cur: np.ndarray, resid_m: np.ndarray, path: Path) -> tuple[int, np.ndarray]:
    diff = np.clip(cur.astype(np.int16) - pred.astype(np.int16) + 128, 0, 255).astype(np.uint8)
    atlas, coords = pack_atlas(diff, resid_m)
    if not coords:
        return 0, np.zeros(pred.shape, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(path, "JPEG", quality=RESID_JPEG_Q, optimize=True)
    decoded = np.array(Image.open(path).convert("RGB"))
    scattered = scatter_atlas(decoded, coords, pred.shape, bias=-128)
    return path.stat().st_size, scattered


def intra_atlas(cur: np.ndarray, intra_m: np.ndarray, path: Path) -> tuple[int, np.ndarray]:
    atlas, coords = pack_atlas(cur, intra_m)
    if not coords:
        return 0, np.zeros(cur.shape, dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(path, "JPEG", quality=INTRA_JPEG_Q, optimize=True)
    decoded = np.array(Image.open(path).convert("RGB"))
    scattered = scatter_atlas(decoded, coords, cur.shape, bias=0).astype(np.uint8)
    return path.stat().st_size, scattered


def deblock(img: np.ndarray, dx: np.ndarray, dy: np.ndarray, modes: np.ndarray) -> np.ndarray:
    """1-px blend on block edges where MV or mode disagrees. In-loop."""
    out = img.astype(np.float32)
    for r in range(ROWS):
        y0, y1 = r * BH, r * BH + BH
        for c in range(1, COLS):
            if dx[r, c] == dx[r, c - 1] and dy[r, c] == dy[r, c - 1] and modes[r, c] == modes[r, c - 1] == 0:
                continue
            x = c * BW
            left = out[y0:y1, x - 1]
            right = out[y0:y1, x]
            out[y0:y1, x - 1] = (2.0 * left + right) / 3.0
            out[y0:y1, x] = (2.0 * right + left) / 3.0
    for c in range(COLS):
        x0, x1 = c * BW, c * BW + BW
        for r in range(1, ROWS):
            if dx[r, c] == dx[r - 1, c] and dy[r, c] == dy[r - 1, c] and modes[r, c] == modes[r - 1, c] == 0:
                continue
            y = r * BH
            up = out[y - 1, x0:x1]
            down = out[y, x0:x1]
            out[y - 1, x0:x1] = (2.0 * up + down) / 3.0
            out[y, x0:x1] = (2.0 * down + up) / 3.0
    return np.clip(out, 0, 255).astype(np.uint8)


def predict(ref_rgb: np.ndarray, cur_rgb: np.ndarray, ref_y_small: np.ndarray, cur_y_small: np.ndarray, cur_y: np.ndarray) -> dict:
    dx_i, dy_i, _ = best_translation(ref_y_small, cur_y_small)
    dx, dy = halfpel_refine(ref_y_small, cur_y_small, dx_i, dy_i)
    scale = W / MOTION_W
    dx_f, dy_f = dx * scale, dy * scale
    glob, valid = warp_float(ref_rgb, dx_f, dy_f)
    glob_y = luma(glob)
    ldx, ldy, _search_mae = local_search(glob_y, cur_y)
    ldx, ldy = median_mv(ldx, ldy)
    pred = warp_blocks_rgb(glob, ldx, ldy)
    pred_y = luma(pred)
    mae = block_mae(np.abs(pred_y - cur_y).astype(np.float32))
    hole = block_mae((~valid).astype(np.float32)) > HOLE_FRAC
    modes = choose_modes(mae, hole)
    return {
        "dx": dx_f,
        "dy": dy_f,
        "ldx": ldx,
        "ldy": ldy,
        "mae": mae,
        "pred": pred,
        "modes": modes,
        "hole": hole,
        "uncovered": float(hole.mean()),
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


def extract() -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    existing = sorted(FRAMES.glob("*.jpg"))
    if len(existing) == ANALYSIS_FPS * DURATION_SEC:
        print(f"reusing {len(existing)} cached analysis frames")
        return
    for p in FRAMES.glob("*"):
        p.unlink()
    run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(CLIP),
            "-vf",
            f"fps={ANALYSIS_FPS},scale={W}:{H_DISP}",
            "-q:v",
            "3",
            str(FRAMES / "%04d.jpg"),
        ]
    )


def load_frames() -> list[np.ndarray]:
    files = sorted(FRAMES.glob("*.jpg"))
    return [pad_frame(np.array(Image.open(p).convert("RGB"))) for p in files]


def self_test() -> None:
    """Sign, warp, and OOB must agree before we spend an encode."""
    prev = np.zeros((H, W), dtype=np.uint8)
    prev[40:80, 80:160] = 200
    cur = np.zeros_like(prev)
    cur[40:80, 84:164] = 200  # content +4 x
    ps = down_luma(prev)
    cs = down_luma(cur)
    dx, dy, _ = best_translation(ps, cs)
    dx_f, dy_f = dx * (W / MOTION_W), dy * (H / MOTION_H)
    warped, valid = warp_float(prev, dx_f, dy_f)
    mae_w = float(np.abs(warped.astype(np.int16) - cur.astype(np.int16)).mean())
    mae_id = float(np.abs(prev.astype(np.int16) - cur.astype(np.int16)).mean())
    if mae_w >= mae_id * 0.6:
        raise SystemExit(f"self-test FAILED: warp mae {mae_w:.3f} not better than identity {mae_id:.3f} (dx={dx_f})")
    # local: +2 x on already-aligned global
    prev2 = np.zeros((H, W), dtype=np.uint8)
    prev2[40:80, 80:160] = 200
    cur2 = np.zeros_like(prev2)
    cur2[40:80, 82:162] = 200
    ldx, ldy, _ = local_search(prev2, cur2)
    loc = warp_blocks_rgb(prev2, ldx, ldy)
    mae_l = float(np.abs(loc.astype(np.int16) - cur2.astype(np.int16)).mean())
    mae_l_id = float(np.abs(prev2.astype(np.int16) - cur2.astype(np.int16)).mean())
    if mae_l >= mae_l_id:
        raise SystemExit(f"self-test FAILED: local warp mae {mae_l:.3f} >= identity {mae_l_id:.3f}")
    if (loc == 0).sum() > (prev2 == 0).sum():
        raise SystemExit("self-test FAILED: local warp introduced black pixels")
    # OOB local MV must pad, not zero
    big = np.zeros((ROWS, COLS), dtype=np.int16)
    big[0, 0] = 40
    big[0, COLS - 1] = -40
    oob = warp_blocks_rgb(np.full((H, W, 3), 180, dtype=np.uint8), big, np.zeros_like(big))
    if (oob.mean(axis=2) < 12).any():
        raise SystemExit("self-test FAILED: OOB warp wrote black tiles")
    print(f"self-test ok  global dx={dx_f:.1f} warp_mae={mae_w:.3f}  local_mae={mae_l:.3f}")


def encode(rgbs: list[np.ndarray]) -> dict:
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
    black_frames = 0

    for i, rgb in enumerate(rgbs):
        if i == 0:
            rec_path = KEYS / f"{i:04d}.jpg"
            key_bytes += save_jpg(rec_path, rgb, KEY_JPEG_Q)
            rec = pad_frame(np.array(Image.open(rec_path).convert("RGB")))
            recon[i] = rec
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
                    "luma": round(float(y_full[0][:H_DISP].mean()), 2),
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

        flux = float(np.abs(rgb[:H_DISP].astype(np.int16) - rgbs[i - 1][:H_DISP].astype(np.int16)).mean())
        hc = hist_corr(hists[i - 1], hists[i])
        luma_jump = float(y_full[i][:H_DISP].mean() - y_full[i - 1][:H_DISP].mean())
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
            rec_path = KEYS / f"{i:04d}.jpg"
            key_bytes += save_jpg(rec_path, rgb, KEY_JPEG_Q)
            rec = pad_frame(np.array(Image.open(rec_path).convert("RGB")))
            kind = "cut" if cut else "keyframe"
            last_key_i = i
            last_key_small = y_small[i]
            acc = 0.0
            n_skip, n_resid, n_intra = n_blocks, 0, 0
            is_key = True
        else:
            rec = pred.copy()
            if n_resid > 0:
                b, decoded = residual_atlas(pred, rgb, resid_m, RESID / f"{i:04d}.jpg")
                rec = np.clip(rec.astype(np.int16) + decoded, 0, 255).astype(np.uint8)
                resid_bytes += b
                stored = True
            if n_intra > 0:
                b, decoded_i = intra_atlas(rgb, intra_m, RESID / f"{i:04d}-intra.jpg")
                intra_px = expand_blocks(intra_m.astype(np.uint8))
                rec[intra_px == 1] = decoded_i[intra_px == 1]
                intra_bytes += b
            rec = deblock(rec, cand["ldx"], cand["ldy"], modes)
            # skip flags packed + MV for non-skip
            motion_bytes += 4 + (n_blocks + 7) // 8 + (n_resid + n_intra) * 2
            is_key = False

        residual = float(np.abs(rgb[:H_DISP].astype(np.int16) - rec[:H_DISP].astype(np.int16)).mean())
        recon[i] = rec
        save_jpg(RECON / f"{i:04d}.jpg", rec, KEY_JPEG_Q)
        psnrs.append(psnr(rec, rgb))
        skip_total += n_skip
        resid_total += n_resid
        intra_total += n_intra
        vis_rows = H_DISP // BH
        blk_mean = rec[: vis_rows * BH].mean(axis=2).reshape(vis_rows, BH, COLS, BW).mean(axis=(1, 3))
        if int((blk_mean < 12).sum()) >= 3:
            black_frames += 1

        rows.append(
            {
                "i": i,
                "t": round(i / ANALYSIS_FPS, 4),
                "flux": round(flux, 3),
                "motion": round(motion, 3),
                "residual": round(residual, 3),
                "occlusion": round(occ, 4),
                "hist": round(hc, 4),
                "luma": round(float(y_full[i][:H_DISP].mean()), 2),
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
            print(
                f"  frame {i}/{n} psnr={psnrs[-1]:.1f} key={is_key} "
                f"skip={n_skip} resid={n_resid} intra={n_intra} dx={cand['dx']:.1f}"
            )

        if is_key or cut or residual > 10:
            draw_heat(rgb, pred if not is_key else rgb, modes if not is_key else np.zeros((ROWS, COLS), np.uint8), i)

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
        im = Image.fromarray(rgbs[i][:H_DISP]).resize((160, 90), Image.BILINEAR)
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

    def load_stats(path: Path) -> dict | None:
        if not path.exists():
            return None
        blob = json.loads(path.read_text())
        return blob["stats"] if "stats" in blob else blob

    v0 = load_stats(MEDIA / "v0" / "stats.json")
    v1 = load_stats(MEDIA / "v1" / "stats.json")
    model_bytes = key_bytes + resid_bytes + motion_bytes + intra_bytes
    kinds: dict[str, int] = {}
    for r in rows:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    psnr_sorted = sorted(psnrs)

    def slim(s: dict | None, attempt: str) -> dict | None:
        if not s:
            return None
        return {
            "attempt": attempt,
            "fps": s.get("fps"),
            "frames": s.get("frames"),
            "keyframes": s.get("keyframes"),
            "residualsStored": s.get("residualsStored"),
            "modelBytes": s.get("modelBytes"),
            "meanResidual": s.get("meanResidual"),
            "meanPsnr": s.get("meanPsnr"),
            "skipBlockFrac": s.get("skipBlockFrac"),
            "reconstructMp4Bytes": s.get("reconstructMp4Bytes"),
        }

    stats = {
        "attempt": ATTEMPT,
        "frames": n,
        "fps": ANALYSIS_FPS,
        "width": W,
        "height": H_DISP,
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
        "rawBytes": n * W * H_DISP * 3,
        "reconstructMp4Bytes": (MEDIA / "reconstruct.mp4").stat().st_size,
        "meanFlux": round(float(np.mean([f["flux"] for f in rows])), 3),
        "meanResidual": round(float(np.mean([f["residual"] for f in rows])), 3),
        "meanMotion": round(float(np.mean([f["motion"] for f in rows])), 3),
        "meanPsnr": round(float(np.mean(psnrs)), 2),
        "medianPsnr": round(float(psnr_sorted[len(psnr_sorted) // 2]), 2),
        "minPsnr": round(float(psnr_sorted[0]), 2),
        "skipBlockFrac": round(skip_total / (n * n_blocks), 4),
        "residBlockFrac": round(resid_total / (n * n_blocks), 4),
        "intraBlockFrac": round(intra_total / (n * n_blocks), 4),
        "ratioVsRaw": round((n * W * H_DISP * 3) / max(model_bytes, 1), 2),
        "ratioVsSource": round(CLIP.stat().st_size / max(model_bytes, 1), 2),
        "kinds": kinds,
        "lambda": LAMBDA,
        "skipMae": SKIP_MAE,
        "blackTileFrames": black_frames,
        "baseline": slim(v0, "v0-global-translation"),
        "baselineV1": slim(v1, "v1-block-mc"),
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
            "reconstructV1": "/media/v1/reconstruct.mp4",
            "scope": "/media/scope.png",
            "duration": DURATION_SEC,
            "startSec": START_SEC,
            "title": "Big Buck Bunny",
            "credit": "Blender Foundation / peach.blender.org  ·  CC BY 3.0",
            "window": f"{START_SEC}s – {START_SEC + DURATION_SEC}s",
        },
    }


def draw_heat(cur: np.ndarray, pred: np.ndarray, modes: np.ndarray, i: int) -> None:
    resid = np.abs(cur[:H_DISP].astype(np.int16) - pred[:H_DISP].astype(np.int16)).mean(axis=2)
    norm = np.clip(resid / 48.0, 0, 1)
    heat = np.zeros((H_DISP, W, 3), dtype=np.uint8)
    heat[..., 0] = (40 + 180 * norm).astype(np.uint8)
    heat[..., 1] = (30 + 40 * (1 - norm)).astype(np.uint8)
    heat[..., 2] = (36 + 20 * (1 - norm)).astype(np.uint8)
    blend = (0.42 * cur[:H_DISP] + 0.58 * heat).astype(np.uint8)
    vis = blend.copy()
    vis_rows = H_DISP // BH
    for r in range(vis_rows):
        for c in range(COLS):
            m = int(modes[r, c])
            if m == 0:
                continue
            y0, x0 = r * BH, c * BW
            color = (196, 92, 74) if m == 1 else (228, 224, 214)
            vis[y0 : y0 + 1, x0 : x0 + BW] = color
            vis[y0 + BH - 1 : min(y0 + BH, H_DISP), x0 : x0 + BW] = color
            vis[y0 : min(y0 + BH, H_DISP), x0 : x0 + 1] = color
            vis[y0 : min(y0 + BH, H_DISP), x0 + BW - 1 : x0 + BW] = color
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
    self_test()
    if "--self-test" in sys.argv:
        return
    print("extracting 24fps analysis frames")
    extract()
    rgbs = load_frames()
    print(f"loaded {len(rgbs)} frames (padded {H}h, display {H_DISP}h)")
    data = encode(rgbs)
    draw_scope(data["frames"])
    out = MEDIA / "analysis.json"
    out.write_text(json.dumps(data))
    if LAB_JSON.parent.exists():
        LAB_JSON.write_text(json.dumps(data))
    (MEDIA / "stats-v1.1.json").write_text(json.dumps(data["stats"], indent=2))
    print(json.dumps(data["stats"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
