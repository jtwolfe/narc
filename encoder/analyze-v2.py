#!/usr/bin/env python3
"""narc attempt v2 — tiny residual nets on the v1.2 geometric stack.

Same 16×16 affine + sub-pel warp. Residual-mode tiles stay JPEG (it wins on
texture). Skip leftover is RGB the luma skip never paid for — each 16×16
patch that RDO accepts gets its own 3- or 12-parameter field net, int8,
closed-loop dequant. No clip-wide NeRV. No CU tree.
v0 / v1 / v1.1 / v1.2 media are frozen under public/media/v{0,1,1.1,1.2}.
"""

from __future__ import annotations

import json
import math
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
RECON = TMP / "recon-v2"
RESID = TMP / "resid-v2"
STRIP = MEDIA / "thumbs"
KEYS = MEDIA / "anchors"
HEATS = MEDIA / "heatmaps"

ATTEMPT = "v2-residual-nets"
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
NET_LAMBDA = LAMBDA
NET_DC_BYTES = 6.0
NET_LIN_BYTES = 14.0
Affine = tuple[float, float, float, float, float, float]

# Patch field basis: 1, x, y, xy on 16×16, coords in [-0.5, 0.5]
_yy, _xx = np.mgrid[0:BH, 0:BW].astype(np.float32)
_px = (_xx + 0.5) / BW - 0.5
_py = (_yy + 0.5) / BH - 0.5
PATCH_B = np.stack(
    [np.ones_like(_px), _px, _py, _px * _py], axis=-1
).reshape(BH * BW, 4)  # 256 × 4
PATCH_BTB = np.linalg.inv(PATCH_B.T @ PATCH_B + 1e-4 * np.eye(4, dtype=np.float32))


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


def mae_inner(a: np.ndarray, b: np.ndarray, r: int) -> float:
    if r <= 0:
        return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    return float(np.abs(a[r:-r, r:-r].astype(np.float32) - b[r:-r, r:-r].astype(np.float32)).mean())


def affine_identity() -> Affine:
    return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def affine_translation(dx: float, dy: float) -> Affine:
    return (1.0, 0.0, -dx, 0.0, 1.0, -dy)


def affine_from_srt(scale: float, rot_deg: float, tx: float, ty: float, cx: float, cy: float) -> Affine:
    """Sampling affine: pred[x] = ref[A x]. Content scale/rot around (cx,cy), then +t."""
    th = math.radians(rot_deg)
    ct, st = math.cos(th), math.sin(th)
    s = scale if abs(scale) > 1e-6 else 1.0
    a = ct / s
    b = st / s
    d = -st / s
    e = ct / s
    c = cx - a * (cx + tx) - b * (cy + ty)
    f = cy - d * (cx + tx) - e * (cy + ty)
    return (a, b, c, d, e, f)


def affine_scale_full(A: Affine, k: float) -> Affine:
    a, b, c, d, e, f = A
    return (a, b, c * k, d, e, f * k)


def affine_decompose(A: Affine) -> tuple[float, float, float, float]:
    """Return (scale, rot_deg, tx, ty) of the content displacement."""
    a, b, c, d, e, f = A
    det = a * e - b * d
    scale = 1.0 / math.sqrt(abs(det) + 1e-12)
    rot = math.degrees(math.atan2(b, a))
    tx, ty = -c, -f
    return float(scale), float(rot), float(tx), float(ty)


def warp_affine(img: np.ndarray, A: Affine) -> tuple[np.ndarray, np.ndarray]:
    """pred = ref sampled by A. Edge-extend, never write 0. Mask is in-frame source."""
    a, b, c, d, e, f = (float(x) for x in A)
    h, w = img.shape[:2]
    pad = int(
        np.ceil(
            max(
                abs(c),
                abs(f),
                abs(a - 1.0) * w,
                abs(e - 1.0) * h,
                abs(b) * h,
                abs(d) * w,
                1.0,
            )
        )
    ) + 4
    if img.ndim == 2:
        padded = np.pad(img, pad, mode="edge")
        fill = 0
    else:
        padded = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        fill = (0, 0, 0)
    Ap = (
        a,
        b,
        c + pad * (1.0 - a - b),
        d,
        e,
        f + pad * (1.0 - d - e),
    )
    out = Image.fromarray(padded).transform(
        (padded.shape[1], padded.shape[0]),
        Image.AFFINE,
        Ap,
        resample=Image.BILINEAR,
        fillcolor=fill,
    )
    arr = np.array(out)
    cropped = arr[pad : pad + h, pad : pad + w]
    yy, xx = np.mgrid[0:h, 0:w]
    sx = a * xx + b * yy + c
    sy = d * xx + e * yy + f
    valid = (sx >= 0) & (sx <= w - 1) & (sy >= 0) & (sy <= h - 1)
    return cropped, valid


def warp_float(img: np.ndarray, dx: float, dy: float) -> tuple[np.ndarray, np.ndarray]:
    return warp_affine(img, affine_translation(dx, dy))


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
    """Hierarchical integer block MVs. Convention: pred_block[x] = ref[x - dx]."""
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


def _bilinear_tile(padded: np.ndarray, sx0: float, sy0: float, extra: int) -> np.ndarray:
    """16×16 bilinear sample; (sx0, sy0) is the unpadded top-left in padded coords."""
    xs = sx0 + np.arange(BW, dtype=np.float32)
    ys = sy0 + np.arange(BH, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    h, w = padded.shape[:2]
    xx = np.clip(xx, 0, w - 1.001)
    yy = np.clip(yy, 0, h - 1.001)
    x0 = np.floor(xx).astype(np.int32)
    y0 = np.floor(yy).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = xx - x0
    wy = yy - y0
    Ia = padded[y0, x0]
    Ib = padded[y0, x1]
    Ic = padded[y1, x0]
    Id = padded[y1, x1]
    if padded.ndim == 2:
        out = (Ia * (1 - wx) + Ib * wx) * (1 - wy) + (Ic * (1 - wx) + Id * wx) * wy
    else:
        wx = wx[..., None]
        wy = wy[..., None]
        out = (Ia * (1 - wx) + Ib * wx) * (1 - wy) + (Ic * (1 - wx) + Id * wx) * wy
    return out


def local_search_subpel(pred_y: np.ndarray, cur_y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integer hierarchical search, then half-pel + quarter diamond."""
    idx, idy, _ = local_search(pred_y, cur_y)
    fdx = idx.astype(np.float32)
    fdy = idy.astype(np.float32)
    extra = int(max(np.abs(fdx).max(), np.abs(fdy).max(), 1)) + 3
    pf = np.pad(pred_y.astype(np.float32), extra, mode="edge")
    cf = cur_y.astype(np.float32)
    best = np.full((ROWS, COLS), 1e18, dtype=np.float64)
    ox = fdx.copy()
    oy = fdy.copy()
    half = (-0.5, 0.0, 0.5)
    qtr = ((-0.25, 0.0), (0.25, 0.0), (0.0, -0.25), (0.0, 0.25))
    for r in range(ROWS):
        y0 = r * BH
        for c in range(COLS):
            x0 = c * BW
            tgt = cf[y0 : y0 + BH, x0 : x0 + BW]
            bx, by = float(fdx[r, c]), float(fdy[r, c])
            for hy in half:
                for hx in half:
                    tile = _bilinear_tile(pf, extra + x0 - (bx + hx), extra + y0 - (by + hy), extra)
                    mae = float(np.abs(tile - tgt).mean())
                    if mae < best[r, c]:
                        best[r, c] = mae
                        ox[r, c] = bx + hx
                        oy[r, c] = by + hy
            bx, by = float(ox[r, c]), float(oy[r, c])
            for hx, hy in qtr:
                tile = _bilinear_tile(pf, extra + x0 - (bx + hx), extra + y0 - (by + hy), extra)
                mae = float(np.abs(tile - tgt).mean())
                if mae < best[r, c]:
                    best[r, c] = mae
                    ox[r, c] = bx + hx
                    oy[r, c] = by + hy
    return ox, oy, best


def median_mv_adaptive(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Snap isolated spikes only. A zoom / parallax gradient has high MAD and is kept."""
    px = np.pad(dx.astype(np.float32), 1, mode="edge")
    py = np.pad(dy.astype(np.float32), 1, mode="edge")
    ox = dx.astype(np.float32).copy()
    oy = dy.astype(np.float32).copy()
    for r in range(ROWS):
        for c in range(COLS):
            nx = px[r : r + 3, c : c + 3]
            ny = py[r : r + 3, c : c + 3]
            mx = float(np.median(nx))
            my = float(np.median(ny))
            mad = 0.5 * (float(np.mean(np.abs(nx - mx))) + float(np.mean(np.abs(ny - my))))
            dist = abs(float(dx[r, c]) - mx) + abs(float(dy[r, c]) - my)
            if dist > 2.5 and mad < 0.85:
                ox[r, c] = mx
                oy[r, c] = my
    return ox, oy


def bilinear_sample(img: np.ndarray, sx: np.ndarray, sy: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    sx_c = np.clip(sx, 0, w - 1.001)
    sy_c = np.clip(sy, 0, h - 1.001)
    x0 = np.floor(sx_c).astype(np.int32)
    y0 = np.floor(sy_c).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1)
    y1 = np.minimum(y0 + 1, h - 1)
    wx = (sx_c - x0).astype(np.float32)
    wy = (sy_c - y0).astype(np.float32)
    if img.ndim == 2:
        Ia = img[y0, x0].astype(np.float32)
        Ib = img[y0, x1].astype(np.float32)
        Ic = img[y1, x0].astype(np.float32)
        Id = img[y1, x1].astype(np.float32)
        out = (Ia * (1 - wx) + Ib * wx) * (1 - wy) + (Ic * (1 - wx) + Id * wx) * wy
        return np.clip(out, 0, 255).astype(np.uint8)
    wx = wx[..., None]
    wy = wy[..., None]
    Ia = img[y0, x0].astype(np.float32)
    Ib = img[y0, x1].astype(np.float32)
    Ic = img[y1, x0].astype(np.float32)
    Id = img[y1, x1].astype(np.float32)
    out = (Ia * (1 - wx) + Ib * wx) * (1 - wy) + (Ic * (1 - wx) + Id * wx) * wy
    return np.clip(out, 0, 255).astype(np.uint8)


def warp_blocks_rgb(img: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Per-block translate, sub-pel bilinear, edge-extend. Never writes 0."""
    if float(np.abs(dx).max()) < 1e-6 and float(np.abs(dy).max()) < 1e-6:
        return img.copy()
    h, w = img.shape[:2]
    mvx = np.repeat(np.repeat(dx.astype(np.float32), BH, axis=0), BW, axis=1)[:h, :w]
    mvy = np.repeat(np.repeat(dy.astype(np.float32), BH, axis=0), BW, axis=1)[:h, :w]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return bilinear_sample(img, xx - mvx, yy - mvy)


def _solve_affine(pts: np.ndarray) -> Affine | None:
    x, y, rx, ry = pts.T
    n = len(x)
    if n < 3:
        return None
    M = np.zeros((2 * n, 6), dtype=np.float64)
    b = np.zeros(2 * n, dtype=np.float64)
    M[0::2, 0] = x
    M[0::2, 1] = y
    M[0::2, 2] = 1
    M[1::2, 3] = x
    M[1::2, 4] = y
    M[1::2, 5] = 1
    b[0::2] = rx
    b[1::2] = ry
    try:
        coef, *_ = np.linalg.lstsq(M, b, rcond=None)
    except np.linalg.LinAlgError:
        return None
    a, bb, c, d, e, f = (float(v) for v in coef)
    det = a * e - bb * d
    if det < 0.72 or det > 1.38:
        return None
    if abs(a) > 1.35 or abs(e) > 1.35 or abs(bb) > 0.4 or abs(d) > 0.4:
        return None
    return (a, bb, c, d, e, f)


def fit_affine_from_mv(dx: np.ndarray, dy: np.ndarray) -> Affine | None:
    ys = (np.arange(ROWS) + 0.5) * BH
    xs = (np.arange(COLS) + 0.5) * BW
    xx, yy = np.meshgrid(xs, ys)
    rx = xx - dx.astype(np.float64)
    ry = yy - dy.astype(np.float64)
    pts = np.stack([xx.ravel(), yy.ravel(), rx.ravel(), ry.ravel()], axis=1)
    n = pts.shape[0]
    rng = np.random.default_rng(1)
    best_inl = None
    best_c = 0
    for _ in range(24):
        idx = rng.choice(n, size=3, replace=False)
        A = _solve_affine(pts[idx])
        if A is None:
            continue
        a, b, c, d, e, f = A
        prx = a * pts[:, 0] + b * pts[:, 1] + c
        pry = d * pts[:, 0] + e * pts[:, 1] + f
        err = np.hypot(prx - pts[:, 2], pry - pts[:, 3])
        inl = err < 1.35
        cnt = int(inl.sum())
        if cnt > best_c:
            best_c = cnt
            best_inl = inl
    if best_inl is None or best_c < max(12, n // 4):
        return None
    return _solve_affine(pts[best_inl])


def choose_global(ref_small: np.ndarray, cur_small: np.ndarray) -> tuple[Affine, float]:
    """Pick translation vs scale/rot grid vs MV-fitted affine on the 4× luma."""
    dx_i, dy_i, _ = best_translation(ref_small, cur_small)
    dx, dy = halfpel_refine(ref_small, cur_small, dx_i, dy_i)
    A_t = affine_translation(dx, dy)
    w_t, _ = warp_affine(ref_small, A_t)
    r = SEARCH_GLOBAL
    mae_t = mae_inner(w_t, cur_small, r)
    best_A, best_mae = A_t, mae_t
    hs, ws = ref_small.shape
    cx, cy = ws / 2.0, hs / 2.0
    for s in (0.96, 0.98, 1.00, 1.02, 1.04):
        for ang in (-2.0, -1.0, 0.0, 1.0, 2.0):
            if s == 1.0 and ang == 0.0:
                continue
            A = affine_from_srt(s, ang, dx, dy, cx, cy)
            w, _ = warp_affine(ref_small, A)
            mae = mae_inner(w, cur_small, r)
            if mae + 0.03 < best_mae:
                best_mae = mae
                best_A = A
    # cheap 4×4 translational grid → RANSAC affine, in small coords
    gh, gw = 4, 4
    bh, bw = hs // gh, ws // gw
    gdx = np.zeros((gh, gw), dtype=np.float64)
    gdy = np.zeros((gh, gw), dtype=np.float64)
    aa = ref_small.astype(np.int16)
    bb = cur_small.astype(np.int16)
    rad = 3
    for gy in range(gh):
        y0, y1 = gy * bh, (gy + 1) * bh
        for gx in range(gw):
            x0, x1 = gx * bw, (gx + 1) * bw
            tgt = bb[y0 + rad : y1 - rad, x0 + rad : x1 - rad]
            best, bdx, bdy = 1e18, 0, 0
            for dy in range(-rad, rad + 1):
                for dx_ in range(-rad, rad + 1):
                    src = aa[y0 + rad - dy : y1 - rad - dy, x0 + rad - dx_ : x1 - rad - dx_]
                    if src.shape != tgt.shape:
                        continue
                    sad = float(np.abs(src - tgt).mean())
                    if sad < best:
                        best, bdx, bdy = sad, dx_, dy
            gdx[gy, gx] = bdx
            gdy[gy, gx] = bdy
    # reuse fit on this 4×4 by temporarily mapping through fake block grid
    # solve directly on the 16 cells
    ys = (np.arange(gh) + 0.5) * bh
    xs = (np.arange(gw) + 0.5) * bw
    xx, yy = np.meshgrid(xs, ys)
    pts = np.stack([xx.ravel(), yy.ravel(), (xx - gdx).ravel(), (yy - gdy).ravel()], axis=1)
    A_fit = _solve_affine(pts)
    if A_fit is not None:
        w, _ = warp_affine(ref_small, A_fit)
        mae = mae_inner(w, cur_small, r)
        if mae + 0.05 < best_mae:
            best_mae = mae
            best_A = A_fit
    return best_A, best_mae


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


def _tiles_of(diff: np.ndarray) -> np.ndarray:
    """(ROWS*COLS, 256, 3) RGB residual tiles, row-major."""
    t = diff[: ROWS * BH, : COLS * BW]
    t = t.reshape(ROWS, BH, COLS, BW, 3).transpose(0, 2, 1, 3, 4)
    return t.reshape(ROWS * COLS, BH * BW, 3)


def _scatter_tiles(tiles: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    t = tiles.reshape(ROWS, COLS, BH, BW, 3).transpose(0, 2, 1, 3, 4)
    out = np.zeros(shape, dtype=np.float32)
    h, w = ROWS * BH, COLS * BW
    out[:h, :w] = t.reshape(h, w, 3)
    return out


def fit_patch_nets(pred: np.ndarray, cur: np.ndarray, modes: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Many tiny field nets on skip patches (and smooth residual patches).

    kinds: 0 none, 1 DC-only (3 int8 + scale share), 2 linear {1,x,y,xy} (12 int8).
    Returns (delta, kinds, bytes, n_net). Delta is what to add to pred. Closed-loop
    uses dequantized int8 weights, not the float teacher.
    """
    diff = (cur.astype(np.float32) - pred.astype(np.float32))
    tiles = _tiles_of(diff)  # N,256,3
    n = ROWS * COLS
    # closed-form linear: W = (BᵀB)⁻¹ Bᵀ Y  → N,4,3
    bt_y = np.einsum("ni,rnc->ric", PATCH_B, tiles)
    w = np.einsum("ij,rjc->ric", PATCH_BTB, bt_y)
    lin_tiles = np.einsum("ni,ric->rnc", PATCH_B, w)
    dc = tiles.mean(axis=1)  # N,3
    dc_tiles = np.repeat(dc[:, None, :], BH * BW, axis=1)
    mae_skip = np.abs(tiles).mean(axis=(1, 2))
    mae_dc = np.abs(tiles - dc_tiles).mean(axis=(1, 2))
    mae_lin = np.abs(tiles - lin_tiles).mean(axis=(1, 2))

    j_skip = mae_skip
    j_dc = mae_dc + NET_LAMBDA * NET_DC_BYTES
    j_lin = mae_lin + NET_LAMBDA * NET_LIN_BYTES
    kinds = np.zeros(n, dtype=np.uint8)
    skip_i = (modes.reshape(-1) == 0)
    # residual-mode tiles: net only if it beats a JPEG-ish floor (~4 MAE at ~28 B)
    resid_i = (modes.reshape(-1) == 1)
    j_jpeg = np.minimum(mae_skip * 0.22 + 1.6, mae_skip) + NET_LAMBDA * 28.0

    take_dc = skip_i & (j_dc < j_skip) & (j_dc <= j_lin)
    take_lin = skip_i & (j_lin < j_skip) & (j_lin < j_dc)
    # residual tiles: net instead of JPEG when linear is close and cheaper
    take_lin |= resid_i & (j_lin + 0.15 < j_jpeg) & (mae_lin < 6.5)
    take_dc |= resid_i & (~take_lin) & (j_dc + 0.15 < j_jpeg) & (mae_dc < 5.5)
    kinds[take_dc] = 1
    kinds[take_lin] = 2

    # quantize used weights (shared abs-scale per frame, closed loop)
    used = kinds > 0
    delta_tiles = np.zeros_like(tiles)
    nbytes = 0
    n_net = int(used.sum())
    if n_net:
        w_store = w.copy()
        w_store[kinds == 1, 1:, :] = 0.0
        w_store[kinds == 1, 0, :] = dc[kinds == 1]
        sel = w_store[used]
        mx = float(np.max(np.abs(sel))) + 1e-8
        scale = mx / 127.0
        q = np.clip(np.round(sel / scale), -127, 127).astype(np.int8)
        w_hat = np.zeros_like(w_store)
        w_hat[used] = q.astype(np.float32) * scale
        # DC-only reconstruction from dequant
        rec_lin = np.einsum("ni,ric->rnc", PATCH_B, w_hat)
        delta_tiles[used] = rec_lin[used]
        # bytes: 4-byte scale + 240×2-bit kinds packed to 60 B + payload
        n_dc = int((kinds == 1).sum())
        n_lin = int((kinds == 2).sum())
        nbytes = 4 + 60 + n_dc * 3 + n_lin * 12

    delta = _scatter_tiles(delta_tiles, pred.shape)
    # residual-mode blocks taken by the net should not also pay JPEG
    return delta, kinds.reshape(ROWS, COLS), nbytes, n_net


def apply_patch_delta(pred: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.clip(pred.astype(np.float32) + delta, 0, 255).astype(np.uint8)


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
    qdx = np.round(dx.astype(np.float32) * 4)
    qdy = np.round(dy.astype(np.float32) * 4)
    for r in range(ROWS):
        y0, y1 = r * BH, r * BH + BH
        for c in range(1, COLS):
            if qdx[r, c] == qdx[r, c - 1] and qdy[r, c] == qdy[r, c - 1] and modes[r, c] == modes[r, c - 1] == 0:
                continue
            x = c * BW
            left = out[y0:y1, x - 1]
            right = out[y0:y1, x]
            out[y0:y1, x - 1] = (2.0 * left + right) / 3.0
            out[y0:y1, x] = (2.0 * right + left) / 3.0
    for c in range(COLS):
        x0, x1 = c * BW, c * BW + BW
        for r in range(1, ROWS):
            if qdx[r, c] == qdx[r - 1, c] and qdy[r, c] == qdy[r - 1, c] and modes[r, c] == modes[r - 1, c] == 0:
                continue
            y = r * BH
            up = out[y - 1, x0:x1]
            down = out[y, x0:x1]
            out[y - 1, x0:x1] = (2.0 * up + down) / 3.0
            out[y, x0:x1] = (2.0 * down + up) / 3.0
    return np.clip(out, 0, 255).astype(np.uint8)


def predict(ref_rgb: np.ndarray, cur_rgb: np.ndarray, ref_y_small: np.ndarray, cur_y_small: np.ndarray, cur_y: np.ndarray) -> dict:
    A_small, _ = choose_global(ref_y_small, cur_y_small)
    A = affine_scale_full(A_small, W / MOTION_W)
    glob, valid = warp_affine(ref_rgb, A)
    glob_y = luma(glob)
    ldx, ldy, _search_mae = local_search_subpel(glob_y, cur_y)
    ldx, ldy = median_mv_adaptive(ldx, ldy)
    pred = warp_blocks_rgb(glob, ldx, ldy)
    pred_y = luma(pred)
    mae = block_mae(np.abs(pred_y - cur_y).astype(np.float32))
    hole = block_mae((~valid).astype(np.float32)) > HOLE_FRAC
    modes = choose_modes(mae, hole)
    scale, rot, tx, ty = affine_decompose(A)
    return {
        "dx": tx,
        "dy": ty,
        "scale": scale,
        "rot": rot,
        "affine": A,
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
    """Sign, affine, sub-pel, adaptive median, and OOB must agree before an encode."""
    prev = np.zeros((H, W), dtype=np.uint8)
    prev[40:80, 80:160] = 200
    cur = np.zeros_like(prev)
    cur[40:80, 84:164] = 200  # content +4 x
    ps = down_luma(prev)
    cs = down_luma(cur)
    A_s, _ = choose_global(ps, cs)
    A = affine_scale_full(A_s, W / MOTION_W)
    warped, _ = warp_affine(prev, A)
    mae_w = float(np.abs(warped.astype(np.int16) - cur.astype(np.int16)).mean())
    mae_id = float(np.abs(prev.astype(np.int16) - cur.astype(np.int16)).mean())
    scale, rot, tx, ty = affine_decompose(A)
    if mae_w >= mae_id * 0.6:
        raise SystemExit(f"self-test FAILED: warp mae {mae_w:.3f} not better than identity {mae_id:.3f} (tx={tx:.1f})")
    if abs(tx - 4.0) > 1.5:
        raise SystemExit(f"self-test FAILED: expected tx~4, got {tx:.2f}")

    prev2 = np.zeros((H, W), dtype=np.uint8)
    prev2[40:80, 80:160] = 200
    cur2 = np.zeros_like(prev2)
    cur2[40:80, 82:162] = 200
    ldx, ldy, _ = local_search_subpel(prev2, cur2)
    loc = warp_blocks_rgb(prev2, ldx, ldy)
    mae_l = float(np.abs(loc.astype(np.int16) - cur2.astype(np.int16)).mean())
    mae_l_id = float(np.abs(prev2.astype(np.int16) - cur2.astype(np.int16)).mean())
    if mae_l >= mae_l_id:
        raise SystemExit(f"self-test FAILED: local warp mae {mae_l:.3f} >= identity {mae_l_id:.3f}")
    if (loc == 0).sum() > (prev2 == 0).sum():
        raise SystemExit("self-test FAILED: local warp introduced black pixels")

    big = np.zeros((ROWS, COLS), dtype=np.float32)
    big[0, 0] = 40
    big[0, COLS - 1] = -40
    oob = warp_blocks_rgb(np.full((H, W, 3), 180, dtype=np.uint8), big, np.zeros_like(big))
    if (oob.mean(axis=2) < 12).any():
        raise SystemExit("self-test FAILED: OOB warp wrote black tiles")

    # half-pel local
    prevh = np.zeros((H, W), dtype=np.uint8)
    prevh[48:96, 80:160] = 200
    curh = np.zeros_like(prevh)
    curh[48:96, 81:161] = 200  # +1 px; half-pel of a 2-step should land near 1
    ldxh, ldyh, _ = local_search_subpel(prevh, curh)
    loch = warp_blocks_rgb(prevh, ldxh, ldyh)
    mae_h = float(np.abs(loch.astype(np.int16) - curh.astype(np.int16)).mean())
    mae_h_id = float(np.abs(prevh.astype(np.int16) - curh.astype(np.int16)).mean())
    if mae_h >= mae_h_id:
        raise SystemExit(f"self-test FAILED: half-pel local mae {mae_h:.3f} >= identity {mae_h_id:.3f}")

    # affine scale: 4% zoom of a plate should beat pure translation
    plate = np.zeros((H, W), dtype=np.uint8)
    plate[36:156, 60:260] = 180
    plate[60:80, 100:220] = 240
    A_zoom = affine_from_srt(1.04, 0.0, 0.0, 0.0, W / 2.0, H / 2.0)
    zoomed, _ = warp_affine(plate, A_zoom)
    zs = down_luma(plate)
    zc = down_luma(zoomed)
    A_found, mae_aff = choose_global(zs, zc)
    w_t, _ = warp_affine(zs, affine_translation(0, 0))
    mae_trans = mae_inner(w_t, zc, 2)
    sc, _, _, _ = affine_decompose(affine_scale_full(A_found, W / MOTION_W))
    if mae_aff >= mae_trans * 0.92 and abs(sc - 1.04) > 0.03:
        raise SystemExit(
            f"self-test FAILED: affine did not beat translation on 4% zoom "
            f"(aff={mae_aff:.3f} trans={mae_trans:.3f} scale={sc:.3f})"
        )

    # adaptive median keeps a radial field, snaps an isolated spike
    rad = np.zeros((ROWS, COLS), dtype=np.float32)
    for r in range(ROWS):
        for c in range(COLS):
            rad[r, c] = 0.12 * (c - COLS / 2.0)
    rad_spike = rad.copy()
    rad_spike[3, 3] = 8.0
    cleaned, _ = median_mv_adaptive(rad_spike, np.zeros_like(rad_spike))
    if abs(float(cleaned[3, 3])) > 2.0:
        raise SystemExit(f"self-test FAILED: adaptive median did not snap spike ({cleaned[3,3]:.2f})")
    if abs(float(cleaned[6, 16]) - float(rad[6, 16])) > 0.4:
        raise SystemExit("self-test FAILED: adaptive median flattened a radial field")

    # patch field net: DC recovers a chroma offset, linear recovers a gradient
    pred_n = np.full((H, W, 3), 80, dtype=np.uint8)
    cur_n = pred_n.copy()
    cur_n[..., 1] = 110  # +30 green
    modes_n = np.zeros((ROWS, COLS), dtype=np.uint8)
    delta, kinds, _, nnet = fit_patch_nets(pred_n, cur_n, modes_n)
    rec_n = apply_patch_delta(pred_n, delta)
    mae_n = float(np.abs(rec_n.astype(np.int16) - cur_n.astype(np.int16)).mean())
    if mae_n > 2.0 or nnet < 200:
        raise SystemExit(f"self-test FAILED: DC net mae {mae_n:.3f} nnet={nnet}")
    grad = pred_n.copy().astype(np.int16)
    ramp = np.linspace(-12, 12, BW, dtype=np.int16)
    for c in range(COLS):
        grad[:, c * BW : (c + 1) * BW, 0] = np.clip(80 + ramp, 0, 255)
    cur_g = np.clip(grad, 0, 255).astype(np.uint8)
    d2, k2, _, n2 = fit_patch_nets(pred_n, cur_g, modes_n)
    rec_g = apply_patch_delta(pred_n, d2)
    mae_g = float(np.abs(rec_g[:, : COLS * BW].astype(np.int16) - cur_g[:, : COLS * BW].astype(np.int16)).mean())
    mae_g_id = float(np.abs(pred_n[:, : COLS * BW].astype(np.int16) - cur_g[:, : COLS * BW].astype(np.int16)).mean())
    if mae_g >= mae_g_id * 0.55:
        raise SystemExit(f"self-test FAILED: linear net mae {mae_g:.3f} not better than {mae_g_id:.3f}")

    print(
        f"self-test ok  tx={tx:.1f} warp_mae={mae_w:.3f}  local_mae={mae_l:.3f}  "
        f"halfpel_mae={mae_h:.3f}  zoom_scale={sc:.3f} zoom_mae={mae_aff:.3f}  "
        f"net_dc_mae={mae_n:.3f} net_lin_mae={mae_g:.3f}"
    )


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
    key_bytes = resid_bytes = motion_bytes = intra_bytes = net_bytes = 0
    psnrs: list[float] = []
    skip_total = resid_total = intra_total = net_total = 0
    n_blocks = ROWS * COLS
    black_frames = 0
    scales: list[float] = []

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
                    "scale": 1.0,
                    "rot": 0.0,
                    "kind": "keyframe",
                    "key": True,
                    "cut": False,
                    "storedResidual": False,
                    "skipBlocks": n_blocks,
                    "residBlocks": 0,
                    "intraBlocks": 0,
                    "netBlocks": 0,
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
            n_skip, n_resid, n_intra, n_net = n_blocks, 0, 0, 0
            is_key = True
        else:
            net_delta, net_kinds, net_b, n_net = fit_patch_nets(pred, rgb, modes)
            rec = apply_patch_delta(pred, net_delta)
            net_bytes += net_b
            resid_m = (modes == 1) & (net_kinds == 0)
            skip_m = (modes == 0) & (net_kinds == 0)
            n_resid = int(resid_m.sum())
            n_skip = int(skip_m.sum())
            n_net = int((net_kinds > 0).sum())
            if n_resid > 0:
                b, decoded = residual_atlas(rec, rgb, resid_m, RESID / f"{i:04d}.jpg")
                rec = np.clip(rec.astype(np.int16) + decoded, 0, 255).astype(np.uint8)
                resid_bytes += b
                stored = True
            if n_net > 0:
                stored = True
            if n_intra > 0:
                b, decoded_i = intra_atlas(rgb, intra_m, RESID / f"{i:04d}-intra.jpg")
                intra_px = expand_blocks(intra_m.astype(np.uint8))
                rec[intra_px == 1] = decoded_i[intra_px == 1]
                intra_bytes += b
            rec = deblock(rec, cand["ldx"], cand["ldy"], modes)
            # affine (24 B) + skip bitmap + quarter-pel MVs for coded blocks
            motion_bytes += 24 + 4 + (n_blocks + 7) // 8 + (n_resid + n_intra + n_net) * 2
            is_key = False

        residual = float(np.abs(rgb[:H_DISP].astype(np.int16) - rec[:H_DISP].astype(np.int16)).mean())
        recon[i] = rec
        save_jpg(RECON / f"{i:04d}.jpg", rec, KEY_JPEG_Q)
        psnrs.append(psnr(rec, rgb))
        skip_total += n_skip
        resid_total += n_resid
        intra_total += n_intra
        net_total += n_net
        scales.append(float(cand["scale"]))
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
                "scale": round(float(cand["scale"]), 4),
                "rot": round(float(cand["rot"]), 2),
                "kind": kind,
                "key": is_key,
                "cut": cut,
                "storedResidual": stored,
                "skipBlocks": n_skip,
                "residBlocks": n_resid,
                "intraBlocks": n_intra,
                "netBlocks": n_net,
                "ref": ref_used,
                "psnr": round(psnrs[-1], 2),
            }
        )

        if i % 120 == 0:
            print(
                f"  frame {i}/{n} psnr={psnrs[-1]:.1f} key={is_key} "
                f"skip={n_skip} resid={n_resid} net={n_net} intra={n_intra} "
                f"dx={cand['dx']:.1f} s={cand['scale']:.3f}"
            )

        if is_key or cut or residual > 10:
            heat_modes = np.zeros((ROWS, COLS), np.uint8) if is_key else modes.copy()
            if not is_key:
                heat_modes[net_kinds > 0] = 3
            draw_heat(rgb, pred if not is_key else rgb, heat_modes, i)

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
    v11 = load_stats(MEDIA / "v1.1" / "stats.json")
    v12 = load_stats(MEDIA / "v1.2" / "stats.json")
    model_bytes = key_bytes + resid_bytes + motion_bytes + intra_bytes + net_bytes
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
            "residualBytes": s.get("residualBytes"),
            "intraBytes": s.get("intraBytes"),
            "netBytes": s.get("netBytes"),
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
        "netBytes": net_bytes,
        "rawBytes": n * W * H_DISP * 3,
        "reconstructMp4Bytes": (MEDIA / "reconstruct.mp4").stat().st_size,
        "meanFlux": round(float(np.mean([f["flux"] for f in rows])), 3),
        "meanResidual": round(float(np.mean([f["residual"] for f in rows])), 3),
        "meanMotion": round(float(np.mean([f["motion"] for f in rows])), 3),
        "meanScale": round(float(np.mean(scales)) if scales else 1.0, 4),
        "meanAbsRot": round(float(np.mean([abs(f.get("rot") or 0) for f in rows])), 3),
        "meanPsnr": round(float(np.mean(psnrs)), 2),
        "medianPsnr": round(float(psnr_sorted[len(psnr_sorted) // 2]), 2),
        "minPsnr": round(float(psnr_sorted[0]), 2),
        "skipBlockFrac": round(skip_total / (n * n_blocks), 4),
        "residBlockFrac": round(resid_total / (n * n_blocks), 4),
        "intraBlockFrac": round(intra_total / (n * n_blocks), 4),
        "netBlockFrac": round(net_total / (n * n_blocks), 4),
        "ratioVsRaw": round((n * W * H_DISP * 3) / max(model_bytes, 1), 2),
        "ratioVsSource": round(CLIP.stat().st_size / max(model_bytes, 1), 2),
        "kinds": kinds,
        "lambda": LAMBDA,
        "skipMae": SKIP_MAE,
        "blackTileFrames": black_frames,
        "baseline": slim(v0, "v0-global-translation"),
        "baselineV1": slim(v1, "v1-block-mc"),
        "baselineV11": slim(v11, "v1.1-mc-correct"),
        "baselineV12": slim(v12, "v1.2-affine-subpel"),
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
            "reconstructV11": "/media/v1.1/reconstruct.mp4",
            "reconstructV12": "/media/v1.2/reconstruct.mp4",
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
            color = (196, 92, 74) if m == 1 else (138, 154, 170) if m == 3 else (228, 224, 214)
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
    (MEDIA / "stats-v2.json").write_text(json.dumps(data["stats"], indent=2))
    print(json.dumps(data["stats"], indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
