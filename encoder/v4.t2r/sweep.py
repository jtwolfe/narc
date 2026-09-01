#!/usr/bin/env python3
"""v4.t2r: native 640×360, per-shot, three representations.

Does not write lab UI / reconstruct.mp4 / analysis.json.
Reuses v4r math (SVD + int8 + ALS) with local W/H so v4r analyze.py stays frozen.
"""
from __future__ import annotations

import io
import json
import struct
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "attempts" / "v4.t2r"
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"
LOG = OUT / "sweep.log"
TRAIN_CHOICE = OUT / "train_choice.json"
SHOTS_JSON = OUT / "shots.json"

CLIP = (
    ROOT / "public" / "media" / "source.mp4"
    if (ROOT / "public" / "media" / "source.mp4").exists()
    else ROOT / "media" / "source.mp4"
)
FRAMES = Path("/tmp/bbb/frames-640")
FRAMES320 = Path("/tmp/bbb/frames-v1")

FPS = 24
W, H_DISP, H = 640, 360, 384
BW, BH = 16, 16
HOP = 8
CUT_HIST = 0.62
JPEG_Q = 84
K_MAX = 16
K_ROOT = 8
SOURCE_H264_BPS = 530_000  # from ffprobe, video stream


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a") as f:
        f.write(line)


def rss_mb() -> float:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        return 0.0
    return 0.0


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


def ssim_y(a: np.ndarray, b: np.ndarray) -> float:
    ya = luma(a)[:H_DISP]
    yb = luma(b)[:H_DISP]
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    vals = []
    for y in range(0, H_DISP - 7, 8):
        for x in range(0, W - 7, 8):
            p, q = ya[y : y + 8, x : x + 8], yb[y : y + 8, x : x + 8]
            mp, mq = float(p.mean()), float(q.mean())
            sp, sq = float(p.var()), float(q.var())
            cov = float(((p - mp) * (q - mq)).mean())
            vals.append(((2 * mp * mq + c1) * (2 * cov + c2)) / ((mp * mp + mq * mq + c1) * (sp + sq + c2) + 1e-12))
    return float(np.mean(vals)) if vals else 0.0


def pad_frame(rgb: np.ndarray) -> np.ndarray:
    if rgb.shape[0] >= H and rgb.shape[1] >= W:
        return rgb[:H, :W]
    out = np.empty((H, W, 3), dtype=np.uint8)
    h, w = min(rgb.shape[0], H), min(rgb.shape[1], W)
    out[:h, :w] = rgb[:h, :w]
    if h < H:
        out[h:] = out[h - 1]
    if w < W:
        out[:, w:] = out[:, w - 1 : w]
    return out


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> tuple[np.ndarray, bytes]:
    buf = io.BytesIO()
    Image.fromarray(rgb[:H_DISP, :W]).save(buf, "JPEG", quality=quality, optimize=True)
    blob = buf.getvalue()
    rec = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
    return pad_frame(rec), blob


def qint8(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == 0:
        scale = np.max(np.abs(arr), axis=tuple(range(1, arr.ndim)), keepdims=True)
    else:
        scale = np.max(np.abs(arr), axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    codes = np.clip(np.round(arr / scale * 127.0), -127, 127).astype(np.int8)
    dequant = codes.astype(np.float32) * scale / 127.0
    return dequant, codes, scale.astype(np.float32)


def train_factors(Xc: np.ndarray, U: np.ndarray, B: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if steps <= 0 or U.size == 0 or B.size == 0:
        return U, B
    U = U.astype(np.float32, copy=True)
    B = B.astype(np.float32, copy=True)
    for _ in range(max(1, steps)):
        B, *_ = np.linalg.lstsq(U, Xc, rcond=None)
        B = B.astype(np.float32)
        ut, *_ = np.linalg.lstsq(B.T, Xc.T, rcond=None)
        U = ut.T.astype(np.float32)
        if not np.isfinite(U).all() or not np.isfinite(B).all():
            raise SystemExit("train_factors produced non-finite factors")
    return U, B


def fit_patch(Xc: np.ndarray, target_mse: float, k_max: int) -> tuple[int, np.ndarray, np.ndarray]:
    tlen, d = Xc.shape
    empty_u = np.zeros((tlen, 0), np.float32)
    empty_b = np.zeros((0, d), np.float32)
    if tlen == 1:
        return 0, empty_u, empty_b
    energy = float(np.mean(Xc * Xc))
    if energy <= target_mse:
        return 0, empty_u, empty_b
    g = Xc @ Xc.T
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order].astype(np.float64), 0.0)
    evecs = evecs[:, order].astype(np.float32)
    denom = float(tlen * d)
    k = 0
    while k < k_max and k < tlen and (evals[k:].sum() / denom) > target_mse:
        k += 1
    if k == 0:
        return 0, empty_u, empty_b
    u = evecs[:, :k]
    b = u.T @ Xc
    return k, u, b.astype(np.float32)


def quant_train(Xc: np.ndarray, u: np.ndarray, b: np.ndarray, steps: int) -> dict:
    k = int(u.shape[1]) if u.ndim == 2 else 0
    packed: dict = {"k": k}
    if k == 0:
        packed.update({"Uq": u, "Bq": b, "U": np.zeros((0, Xc.shape[0]), np.int8), "B": np.zeros((0, Xc.shape[1]), np.int8),
                       "Us": np.zeros((0,), np.float32), "Bs": np.zeros((0,), np.float32)})
        return packed
    uq, uc, us = qint8(u.T, axis=0)
    uq = uq.T
    bq, bc, bs = qint8(b, axis=0)
    if steps > 0:
        uq, bq = train_factors(Xc, uq, bq, steps=steps)
        uq, uc, us = qint8(uq.T, axis=0)
        uq = uq.T
        bq, bc, bs = qint8(bq, axis=0)
    packed.update(
        {
            "Uq": uq,
            "Bq": bq,
            "U": uc,
            "B": bc,
            "Us": us.reshape(k).astype(np.float32),
            "Bs": bs.reshape(k).astype(np.float32),
        }
    )
    return packed


def seam_mae(src: np.ndarray, rec: np.ndarray, bw: int, bh: int) -> tuple[float, float, float]:
    err = np.abs(src[:H_DISP].astype(np.float32) - rec[:H_DISP].astype(np.float32)).mean(axis=2)
    h, w = err.shape
    seam = np.zeros((h, w), dtype=bool)
    for x in range(bw, w, bw):
        seam[:, max(0, x - 1) : min(w, x + 1)] = True
    for y in range(bh, h, bh):
        seam[max(0, y - 1) : min(h, y + 1), :] = True
    interior = ~seam
    sm = float(err[seam].mean()) if seam.any() else 0.0
    im = float(err[interior].mean()) if interior.any() else 0.0
    ratio = sm / im if im > 1e-6 else 0.0
    return sm, im, ratio


def rgb_to_yuv420_tile(p: np.ndarray) -> np.ndarray:
    """p: T,16,16,3 float → T,384  (Y 16×16 + U 8×8 + V 8×8)."""
    r, g, b = p[..., 0], p[..., 1], p[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    u = cb.reshape(p.shape[0], 8, 2, 8, 2).mean(axis=(2, 4))
    v = cr.reshape(p.shape[0], 8, 2, 8, 2).mean(axis=(2, 4))
    return np.concatenate([y.reshape(p.shape[0], -1), u.reshape(p.shape[0], -1), v.reshape(p.shape[0], -1)], 1)


def yuv420_to_rgb_tile(z: np.ndarray) -> np.ndarray:
    t = z.shape[0]
    y = z[:, :256].reshape(t, 16, 16)
    u = z[:, 256:320].reshape(t, 8, 8)
    v = z[:, 320:384].reshape(t, 8, 8)
    cb = np.repeat(np.repeat(u, 2, 1), 2, 2)
    cr = np.repeat(np.repeat(v, 2, 1), 2, 2)
    r = y + 1.402 * (cr - 128.0)
    g = y - 0.344136 * (cb - 128.0) - 0.714136 * (cr - 128.0)
    b = y + 1.772 * (cb - 128.0)
    return np.stack([r, g, b], -1)


def frame_files() -> list[Path]:
    files = sorted(FRAMES.glob("*.png"))
    if len(files) != FPS * 90:
        raise SystemExit(f"want {FPS*90} pngs in {FRAMES}, got {len(files)}")
    return files


def load_range(files: list[Path], i0: int, i1: int) -> np.ndarray:
    sl = [pad_frame(np.array(Image.open(files[i]).convert("RGB"))) for i in range(i0, i1)]
    return np.stack(sl, 0)


def load320(i0: int, i1: int) -> np.ndarray:
    files = sorted(FRAMES320.glob("*.jpg"))
    return np.stack([np.array(Image.open(files[i]).convert("RGB"))[:180, :320] for i in range(i0, i1)], 0)


def down320(rgb: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(rgb[:H_DISP, :W]).resize((320, 180), Image.BOX))


def label_flux(f: float) -> str:
    if f > 8:
        return "busy"
    if f > 3.5:
        return "tracking"
    return "locked"


def detect_shots(files: list[Path]) -> list[dict]:
    n = len(files)
    cuts = [0]
    prev = luma(np.array(Image.open(files[0]).convert("RGB")))
    prev_h = hist16(prev)
    fluxes = [0.0]
    for i in range(1, n):
        y = luma(np.array(Image.open(files[i]).convert("RGB")))
        h = hist16(y)
        flux = float(np.abs(y.astype(np.int16) - prev.astype(np.int16)).mean())
        fluxes.append(flux)
        hist = hist_corr(h, prev_h)
        if hist < CUT_HIST and flux > 12:
            cuts.append(i)
        prev, prev_h = y, h
    cuts.append(n)
    shots = []
    for k in range(len(cuts) - 1):
        a, b = cuts[k], cuts[k + 1]
        if b <= a:
            continue
        fl = float(np.mean(fluxes[a:b]))
        shots.append(
            {
                "sid": f"S{len(shots):02d}",
                "i0": a,
                "i1": b,
                "t0": a / FPS,
                "t1": b / FPS,
                "t": b - a,
                "flux": round(fl, 3),
                "kind": label_flux(fl),
            }
        )
    return shots or [{"sid": "S00", "i0": 0, "i1": n, "t0": 0, "t1": n / FPS, "t": n, "flux": 0.0, "kind": "locked"}]


def hann2d() -> np.ndarray:
    n = np.arange(BW, dtype=np.float32)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (n + 0.5) / BW)
    return np.outer(w, w).astype(np.float32)


def tile_starts(hop: int) -> tuple[list[int], list[int]]:
    ys = list(range(0, H - BH + 1, hop))
    xs = list(range(0, W - BW + 1, hop))
    if ys[-1] != H - BH:
        ys.append(H - BH)
    if xs[-1] != W - BW:
        xs.append(W - BW)
    return ys, xs


def pack_origin(items: list[dict], n_frames: int, extra: bytes = b"") -> tuple[int, int]:
    body = bytearray()
    body += struct.pack("<HHHIH", W, H_DISP, FPS, n_frames, 1)
    body += extra
    for it in items:
        k = int(it.get("k", 0))
        body += struct.pack("<B", k)
        if k == 0:
            continue
        body += np.asarray(it["Us"], dtype=np.float32).tobytes()
        body += np.asarray(it["Bs"], dtype=np.float32).tobytes()
        body += np.asarray(it["B"], dtype=np.int8).tobytes()
        body += np.asarray(it["U"], dtype=np.int8).tobytes()
    raw = bytes(body)
    z = zlib.compress(raw, 9)
    return len(raw), len(z) + 9


def recon_from_patches(tlen: int, mu_f: np.ndarray, patches: list[dict], ys: list[int], xs: list[int],
                       window: np.ndarray | None, kprime: int | None) -> np.ndarray:
    if window is None:
        rec = np.repeat(mu_f[None, ...], tlen, axis=0)
        for p in patches:
            k = int(p["k"])
            if k == 0:
                continue
            kuse = k if kprime is None else min(k, kprime)
            if kuse <= 0:
                continue
            y0, x0 = p["y0"], p["x0"]
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW].reshape(-1)
            extra = p.get("root", None)
            u = p["Uq"][:, :kuse]
            b = p["Bq"][:kuse]
            recp = (m + (u @ b)).reshape(tlen, BH, BW, 3)
            if extra is not None:
                recp = recp + extra
            rec[:, y0 : y0 + BH, x0 : x0 + BW, :] = recp
        return np.clip(np.round(rec), 0, 255).astype(np.uint8)
    acc = np.zeros((tlen, H, W, 3), np.float32)
    wgt = np.zeros((H, W), np.float32)
    win = window[:, :, None]
    for p in patches:
        y0, x0 = p["y0"], p["x0"]
        m = mu_f[y0 : y0 + BH, x0 : x0 + BW]
        k = int(p["k"])
        recp = np.broadcast_to(m, (tlen, BH, BW, 3)).copy()
        if k > 0:
            kuse = k if kprime is None else min(k, kprime)
            if kuse > 0:
                recp = recp + (p["Uq"][:, :kuse] @ p["Bq"][:kuse]).reshape(tlen, BH, BW, 3)
        acc[:, y0 : y0 + BH, x0 : x0 + BW, :] += recp * win
        wgt[y0 : y0 + BH, x0 : x0 + BW] += window
    acc += np.repeat(mu_f[None], tlen, 0) * 0  # mu already in recp
    w = np.maximum(wgt, 1e-6)[None, :, :, None]
    return np.clip(np.round(acc / w), 0, 255).astype(np.uint8)


def encode_tiles(
    sl: np.ndarray,
    *,
    target_mse: float,
    k_max: int,
    steps: int,
    hop: int,
    yuv: bool,
    root_ub: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    tlen = sl.shape[0]
    accm = np.zeros((H, W, 3), np.float64)
    for i in range(0, tlen, 16):
        accm += sl[i : i + 16].sum(axis=0, dtype=np.float64)
    mu = (accm / tlen).astype(np.float32)
    del accm
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    ys, xs = tile_starts(hop)
    window = hann2d() if hop < BW else None
    root_u = root_b = None
    if root_ub is not None:
        root_u, root_b = root_ub  # U T×Kr, B Kr×H×W×3
    patches: list[dict] = []
    k_hist: Counter[int] = Counter()
    basis_b = coeff_b = 0
    n_sat = 0
    if window is None:
        rec_u = np.empty((tlen, H, W, 3), np.uint8)
        acc = None
        wgt = None
        win = None
    else:
        rec_u = None
        acc = np.zeros((tlen, H, W, 3), np.float32)
        wgt = np.zeros((H, W), np.float32)
        win = window[:, :, None]
    for y0 in ys:
        for x0 in xs:
            p = sl[:, y0 : y0 + BH, x0 : x0 + BW, :].astype(np.float32)
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW]
            if root_u is not None:
                rloc = np.einsum("tk,khwc->thwc", root_u, root_b[:, y0 : y0 + BH, x0 : x0 + BW])
                x_rgb = p - m - rloc
            else:
                rloc = None
                x_rgb = p - m
            if yuv:
                z = rgb_to_yuv420_tile(p)
                mz = rgb_to_yuv420_tile(m[None]).reshape(-1)
                xc = z - mz
            else:
                xc = x_rgb.reshape(tlen, -1)
            k, u, b = fit_patch(xc, target_mse, k_max)
            packed = quant_train(xc, u, b, steps)
            packed["y0"] = y0
            packed["x0"] = x0
            if yuv and k > 0:
                rec_z = mz + packed["Uq"] @ packed["Bq"]
                recp = yuv420_to_rgb_tile(rec_z)
            elif k > 0:
                recp = m + (packed["Uq"] @ packed["Bq"]).reshape(tlen, BH, BW, 3)
            else:
                recp = np.broadcast_to(m, (tlen, BH, BW, 3)).copy()
            if rloc is not None:
                recp = recp + rloc
            k_hist[k] += 1
            if k == k_max:
                n_sat += 1
            if k > 0:
                basis_b += packed["B"].nbytes + packed["Bs"].nbytes
                coeff_b += packed["U"].nbytes + packed["Us"].nbytes
            patches.append(packed)
            recp_u = np.clip(np.round(recp), 0, 255).astype(np.uint8)
            if window is None:
                rec_u[:, y0 : y0 + BH, x0 : x0 + BW, :] = recp_u
            else:
                acc[:, y0 : y0 + BH, x0 : x0 + BW, :] += recp * win
                wgt[y0 : y0 + BH, x0 : x0 + BW] += window
    if window is not None:
        rec_u = np.clip(np.round(acc / np.maximum(wgt, 1e-6)[None, :, :, None]), 0, 255).astype(np.uint8)
    meters = {
        "kHist": {int(k): int(n) for k, n in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "basisBytes": int(basis_b),
        "coeffBytes": int(coeff_b),
        "meanJpegBytes": int(len(mu_blob)),
        "nPatches": len(patches),
        "nSat": int(n_sat),
        "t": tlen,
    }
    return rec_u, mu_blob, patches, meters


def fit_global(sl: np.ndarray, mu_f: np.ndarray, k_fixed: int, target_mse: float) -> tuple[int, np.ndarray, np.ndarray]:
    tlen = sl.shape[0]
    d = H * W * 3
    if tlen == 1:
        return 0, np.zeros((tlen, 0), np.float32), np.zeros((0, H, W, 3), np.float32)
    g = np.zeros((tlen, tlen), np.float64)
    energy = 0.0
    band = 8
    for y0 in range(0, H, band):
        ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
        ch = ch.reshape(tlen, -1)
        energy += float(np.square(ch).sum())
        g += ch.astype(np.float64) @ ch.astype(np.float64).T
    energy /= float(tlen * d)
    if energy <= target_mse:
        return 0, np.zeros((tlen, 0), np.float32), np.zeros((0, H, W, 3), np.float32)
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evecs = evecs[:, order].astype(np.float32)
    k = min(k_fixed, tlen - 1)
    u = evecs[:, :k]
    bvol = np.zeros((k, H, W, 3), np.float32)
    for y0 in range(0, H, band):
        ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
        t, hh, ww, cc = ch.shape
        bvol[:, y0 : y0 + hh] = (u.T @ ch.reshape(t, -1)).reshape(k, hh, ww, cc)
    return k, u, bvol


def train_global(sl: np.ndarray, mu_f: np.ndarray, u: np.ndarray, bvol: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray]:
    if steps <= 0 or u.size == 0:
        return u, bvol
    tlen = sl.shape[0]
    k = u.shape[1]
    u = u.astype(np.float32, copy=True)
    bvol = bvol.astype(np.float32, copy=True)
    band = 8
    for _ in range(max(1, steps)):
        for y0 in range(0, H, band):
            ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
            t, hh, ww, cc = ch.shape
            b, *_ = np.linalg.lstsq(u, ch.reshape(t, -1), rcond=None)
            bvol[:, y0 : y0 + hh] = b.reshape(k, hh, ww, cc)
        kxt = np.zeros((k, tlen), np.float64)
        bbt = np.zeros((k, k), np.float64)
        for y0 in range(0, H, band):
            ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
            bb = bvol[:, y0 : y0 + ch.shape[1]].reshape(k, -1)
            kxt += bb.astype(np.float64) @ ch.reshape(tlen, -1).T.astype(np.float64)
            bbt += bb.astype(np.float64) @ bb.astype(np.float64).T
        u = np.linalg.solve(bbt + 1e-6 * np.eye(k), kxt).T.astype(np.float32)
        if not np.isfinite(u).all() or not np.isfinite(bvol).all():
            raise SystemExit("train_global non-finite")
    return u, bvol


def encode_tree(sl: np.ndarray, target_mse: float, k_max: int, steps: int) -> tuple[np.ndarray, bytes, list[dict], dict, dict]:
    tlen = sl.shape[0]
    accm = np.zeros((H, W, 3), np.float64)
    for i in range(0, tlen, 16):
        accm += sl[i : i + 16].sum(axis=0, dtype=np.float64)
    mu = (accm / tlen).astype(np.float32)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    kr, u, bvol = fit_global(sl, mu_f, K_ROOT, target_mse)
    root_pack = {"k": 0}
    if kr > 0:
        # quantize U (T×K) and B (K×D) without assembling Xc
        uq, uc, us = qint8(u.T, axis=0)
        uq = uq.T
        bflat = bvol.reshape(kr, -1)
        bq, bc, bs = qint8(bflat, axis=0)
        if steps > 0:
            uq, bvol_q = train_global(sl, mu_f, uq, bq.reshape(kr, H, W, 3), steps)
            uq, uc, us = qint8(uq.T, axis=0)
            uq = uq.T
            bq, bc, bs = qint8(bvol_q.reshape(kr, -1), axis=0)
            bvol = bq.reshape(kr, H, W, 3)
        else:
            bvol = bq.reshape(kr, H, W, 3)
        u = uq
        root_pack = {
            "k": kr,
            "U": uc,
            "B": bc,
            "Us": us.reshape(kr).astype(np.float32),
            "Bs": bs.reshape(kr).astype(np.float32),
            "Uq": u,
            "Bq": bvol,
        }
    rec_u, mu_blob2, patches, meters = encode_tiles(
        sl, target_mse=target_mse, k_max=k_max, steps=steps, hop=BW, yuv=False, root_ub=(u, bvol) if kr > 0 else None
    )
    meters["kRoot"] = kr
    meters["rootBytes"] = int(root_pack["B"].nbytes + root_pack["U"].nbytes + root_pack["Us"].nbytes + root_pack["Bs"].nbytes) if kr else 0
    return rec_u, mu_blob, patches, meters, root_pack


def already_done() -> set[str]:
    ids: set[str] = set()
    if not JSONL.exists():
        return ids
    for line in JSONL.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def score(sl: np.ndarray, rec: np.ndarray, i0: int, hop: int) -> dict:
    psnrs, leftovers, ssims, sm, im = [], [], [], [], []
    bridge = []
    has320 = FRAMES320.exists() and len(list(FRAMES320.glob("*.jpg"))) >= i0 + len(sl)
    src320 = load320(i0, i0 + len(sl)) if has320 else None
    bw = BW if hop >= BW else BW  # seam grid always 16 for comparability
    for t in range(len(sl)):
        psnrs.append(psnr(sl[t], rec[t]))
        leftovers.append(float(np.abs(sl[t][:H_DISP].astype(np.int16) - rec[t][:H_DISP].astype(np.int16)).mean()))
        ssims.append(ssim_y(sl[t], rec[t]))
        a, b, _ = seam_mae(sl[t], rec[t], bw, BH)
        sm.append(a)
        im.append(b)
        if src320 is not None:
            bridge.append(psnr(src320[t], down320(rec[t])))
    return {
        "meanPsnr": round(float(np.mean(psnrs)), 3),
        "medianPsnr": round(float(np.median(psnrs)), 3),
        "minPsnr": round(float(np.min(psnrs)), 3),
        "leftover": round(float(np.mean(leftovers)), 3),
        "ssim": round(float(np.mean(ssims)), 4),
        "seamMae": round(float(np.mean(sm)), 3),
        "interiorMae": round(float(np.mean(im)), 3),
        "seamRatio": round(float(np.mean(sm) / max(float(np.mean(im)), 1e-6)), 3),
        "bridgePsnr": round(float(np.mean(bridge)), 3) if bridge else None,
    }


def run_one(cfg: dict, files: list[Path], shots_by_id: dict[str, dict], done: set[str]) -> dict | None:
    cid = cfg["id"]
    if cid in done:
        log(f"skip {cid}")
        return None
    rss0 = rss_mb()
    t0 = time.time()
    ranges = cfg["ranges"]  # list of (i0,i1)
    target_mse = 255.0 * 255.0 / (10 ** (cfg["targetPsnr"] / 10.0))
    rec_all = []
    sl_all = []
    packed_items: list[dict] = []
    extra = b""
    k_hist: Counter[int] = Counter()
    basis_b = coeff_b = jpeg_b = n_sat = n_patch = k_root = root_b = 0
    i0_first = ranges[0][0]
    for (a, b) in ranges:
        sl = load_range(files, a, b)
        sl_all.append(sl)
        arm = cfg["arm"]
        if arm == "C":
            rec, mu_blob, patches, meters, root = encode_tree(sl, target_mse, cfg["kMax"], cfg["trainSteps"])
            k_root = int(meters.get("kRoot", 0))
            root_b += int(meters.get("rootBytes", 0))
            if k_root:
                extra += np.asarray(root["B"]).tobytes() + np.asarray(root["U"]).tobytes()
                packed_items.append(root)
        else:
            rec, mu_blob, patches, meters = encode_tiles(
                sl,
                target_mse=target_mse,
                k_max=cfg["kMax"],
                steps=cfg["trainSteps"],
                hop=cfg["hop"],
                yuv=cfg.get("yuv", False),
                root_ub=None,
            )
        rec_all.append(rec)
        jpeg_b += len(mu_blob)
        extra += mu_blob
        packed_items.extend(patches)
        for k, n in meters["kHist"].items():
            k_hist[int(k)] += n
        basis_b += meters["basisBytes"]
        coeff_b += meters["coeffBytes"]
        n_sat += meters["nSat"]
        n_patch += meters["nPatches"]
    rec = np.concatenate(rec_all, 0)
    sl = np.concatenate(sl_all, 0)
    sc = score(sl, rec, i0_first, cfg["hop"])
    raw_b, z_b = pack_origin(packed_items, len(sl), extra=b"")
    mean_k = float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1))
    macs = mean_k * (n_patch / max(len(ranges), 1)) * BW * BH * 3
    enc_s = time.time() - t0
    rss1 = rss_mb()
    ladder = {}
    if cfg.get("decodeLadder"):
        # rebuild from last shot only if single range
        if len(ranges) == 1 and cfg["arm"] != "C":
            sl1 = sl_all[0]
            mu_f = rec_all[0][0].astype(np.float32) * 0  # replaced
            # jpeg μ from first rec's... better re-jpeg from stored: use rec mean approx
            mu_f = jpeg_roundtrip(np.clip(np.round(sl1.astype(np.float32).mean(0)), 0, 255).astype(np.uint8), JPEG_Q)[0].astype(np.float32)
            ys, xs = tile_starts(cfg["hop"])
            window = hann2d() if cfg["hop"] < BW else None
            for kp in (0, 1, 2, 4, 8, 16):
                rp = recon_from_patches(sl1.shape[0], mu_f, packed_items, ys, xs, window, kp)
                lp = [psnr(sl1[t], rp[t]) for t in range(len(sl1))]
                ladder[str(kp)] = {"meanPsnr": round(float(np.mean(lp)), 3), "minPsnr": round(float(np.min(lp)), 3)}
        elif len(ranges) == 1 and cfg["arm"] == "C":
            # leaf K′ with root always on: use rec at kprime via encode is expensive; skip full, report full only
            ladder["full"] = {"meanPsnr": sc["meanPsnr"], "minPsnr": sc["minPsnr"]}
    kind = cfg.get("kind", "")
    row = {
        "id": cid,
        "phase": cfg["phase"],
        "arm": cfg["arm"],
        "shot": cfg.get("shot"),
        "kind": kind,
        "flux": cfg.get("flux"),
        "i0": ranges[0][0],
        "i1": ranges[-1][1],
        "frames": int(len(sl)),
        "trainSteps": cfg["trainSteps"],
        "kMax": cfg["kMax"],
        "targetPsnr": cfg["targetPsnr"],
        "hop": cfg["hop"],
        "yuv": bool(cfg.get("yuv", False)),
        "merge": cfg.get("merge"),
        **sc,
        "meanK": round(mean_k, 3),
        "nSat": int(n_sat),
        "nPatches": int(n_patch),
        "kHist": {str(k): int(n) for k, n in sorted(k_hist.items())},
        "kRoot": int(k_root),
        "rootBytes": int(root_b),
        "basisBytes": int(basis_b),
        "coeffBytes": int(coeff_b),
        "meanJpegBytes": int(jpeg_b),
        "rawBytes": int(raw_b),
        "originBytes": int(z_b),
        "bytesPerSec": round(z_b / max(len(sl) / FPS, 1e-6), 1),
        "vsSourceH264": round(z_b / max((len(sl) / FPS) * (SOURCE_H264_BPS / 8), 1), 3),
        "macs": round(macs, 1),
        "encodeSec": round(enc_s, 2),
        "rssMb": round(max(rss0, rss1), 1),
        "ladder": ladder,
    }
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log(
        f"{cid:28s} {kind:9s} PSNR {row['meanPsnr']:6.2f}/{row['minPsnr']:5.2f}  "
        f"SSIM {row['ssim']:.3f}  K={row['meanK']:.2f} sat={row['nSat']:4d}  "
        f"orig={row['originBytes']/1e6:.3f}MB  seamR={row['seamRatio']:.2f}  "
        f"{row['encodeSec']:.1f}s  rss={row['rssMb']:.0f}MB"
    )
    return row


def pick_reps(shots: list[dict]) -> dict[str, dict]:
    locked = [s for s in shots if s["kind"] == "locked"]
    track = [s for s in shots if s["kind"] == "tracking"]
    L = min(locked or shots, key=lambda s: (s["flux"], -s["t"]))
    B = max(shots, key=lambda s: (s["flux"], s["t"]))
    cand = [s for s in track if s["sid"] != B["sid"]]
    if not cand:
        cand = [s for s in shots if s["sid"] not in {L["sid"], B["sid"]}]
    T = sorted(cand, key=lambda s: s["flux"])[len(cand) // 2] if cand else L
    return {"L": L, "T": T, "B": B}


def pick_merge_pair(shots: list[dict]) -> tuple[dict, dict]:
    """Busiest shot plus its neighbor (the episode-cut proxy)."""
    B = max(shots, key=lambda s: s["flux"])
    idx = next(i for i, s in enumerate(shots) if s["sid"] == B["sid"])
    if idx + 1 < len(shots):
        return shots[idx], shots[idx + 1]
    return shots[idx - 1], shots[idx]


def make_id(cfg: dict) -> str:
    bits = [
        cfg["phase"],
        cfg.get("shot") or cfg.get("merge") or "x",
        cfg["arm"],
        f"s{cfg['trainSteps']}",
        f"k{cfg['kMax']}",
        f"t{cfg['targetPsnr']}",
        f"h{cfg['hop']}",
        "yuv" if cfg.get("yuv") else "rgb",
    ]
    return "-".join(str(x) for x in bits)


def configs(shots: list[dict], steps: int) -> list[dict]:
    reps = pick_reps(shots)
    m1, m2 = pick_merge_pair(shots)
    out: list[dict] = []

    def add(**kw):
        d = {
            "trainSteps": steps,
            "kMax": K_MAX,
            "targetPsnr": 32.5,
            "hop": BW,
            "yuv": False,
            "decodeLadder": False,
            "merge": None,
            "arm": "A",
            "phase": "A",
        }
        d.update(kw)
        d["id"] = make_id(d)
        out.append(d)

    # Phase T already run separately
    for s in shots:
        add(
            phase="A",
            arm="A",
            shot=s["sid"],
            kind=s["kind"],
            flux=s["flux"],
            ranges=[(s["i0"], s["i1"])],
            decodeLadder=s["sid"] in {reps[k]["sid"] for k in reps},
        )
    for tag, s in reps.items():
        add(phase="B", arm="B", shot=s["sid"], kind=s["kind"], flux=s["flux"], ranges=[(s["i0"], s["i1"])], hop=HOP, decodeLadder=True)
        add(phase="C", arm="C", shot=s["sid"], kind=s["kind"], flux=s["flux"], ranges=[(s["i0"], s["i1"])], decodeLadder=True)
    for tag, s in reps.items():
        for arm, hop in (("A", BW), ("B", HOP), ("C", BW)):
            add(
                phase="Q",
                arm=arm,
                shot=s["sid"],
                kind=s["kind"],
                flux=s["flux"],
                ranges=[(s["i0"], s["i1"])],
                hop=hop,
                targetPsnr=40.0,
            )
    add(
        phase="K",
        arm="A",
        shot=reps["B"]["sid"],
        kind=reps["B"]["kind"],
        flux=reps["B"]["flux"],
        ranges=[(reps["B"]["i0"], reps["B"]["i1"])],
        kMax=32,
    )
    add(
        phase="Y",
        arm="A",
        shot=reps["L"]["sid"],
        kind=reps["L"]["kind"],
        flux=reps["L"]["flux"],
        ranges=[(reps["L"]["i0"], reps["L"]["i1"])],
        yuv=True,
    )
    # M1 naive merge (one μ, one SVD over both)
    add(
        phase="M1",
        arm="A",
        shot=f"{m1['sid']}+{m2['sid']}",
        kind="merge-cut",
        flux=round((m1["flux"] * m1["t"] + m2["flux"] * m2["t"]) / (m1["t"] + m2["t"]), 3),
        ranges=[(m1["i0"], m2["i1"])],
        merge="naive",
    )
    add(
        phase="M1",
        arm="C",
        shot=f"{m1['sid']}+{m2['sid']}",
        kind="merge-cut",
        flux=round((m1["flux"] * m1["t"] + m2["flux"] * m2["t"]) / (m1["t"] + m2["t"]), 3),
        ranges=[(m1["i0"], m2["i1"])],
        merge="naive",
    )
    # M2 shared-B: implemented as two ranges with special arm handled in run_m2
    add(
        phase="M2",
        arm="A",
        shot=f"{m1['sid']}+{m2['sid']}",
        kind="shared-B",
        flux=round((m1["flux"] * m1["t"] + m2["flux"] * m2["t"]) / (m1["t"] + m2["t"]), 3),
        ranges=[(m1["i0"], m1["i1"]), (m2["i0"], m2["i1"])],
        merge="sharedB",
    )
    return out, reps, (m1, m2)


def run_m2(cfg: dict, files: list[Path], steps: int, done: set[str]) -> dict | None:
    """Shared spatial B, per-shot μ and U. Episode-wide bases proxy."""
    cid = cfg["id"]
    if cid in done:
        log(f"skip {cid}")
        return None
    rss0 = rss_mb()
    t0 = time.time()
    (a0, a1), (b0, b1) = cfg["ranges"]
    sl1, sl2 = load_range(files, a0, a1), load_range(files, b0, b1)
    t1, t2 = sl1.shape[0], sl2.shape[0]
    target_mse = 255.0 * 255.0 / (10 ** (cfg["targetPsnr"] / 10.0))

    def shot_mu(sl):
        acc = np.zeros((H, W, 3), np.float64)
        for i in range(0, sl.shape[0], 16):
            acc += sl[i : i + 16].sum(axis=0, dtype=np.float64)
        return (acc / sl.shape[0]).astype(np.float32)

    mu1 = jpeg_roundtrip(np.clip(np.round(shot_mu(sl1)), 0, 255).astype(np.uint8), JPEG_Q)
    mu2 = jpeg_roundtrip(np.clip(np.round(shot_mu(sl2)), 0, 255).astype(np.uint8), JPEG_Q)
    mu1f, blob1 = mu1[0].astype(np.float32), mu1[1]
    mu2f, blob2 = mu2[0].astype(np.float32), mu2[1]
    rec1 = np.repeat(mu1[0][None], t1, 0)
    rec2 = np.repeat(mu2[0][None], t2, 0)
    k_hist: Counter[int] = Counter()
    items = []
    basis_b = coeff_b = n_sat = 0
    ys, xs = tile_starts(BW)
    for y0 in ys:
        for x0 in xs:
            p1 = sl1[:, y0 : y0 + BH, x0 : x0 + BW].astype(np.float32)
            p2 = sl2[:, y0 : y0 + BH, x0 : x0 + BW].astype(np.float32)
            m1 = mu1f[y0 : y0 + BH, x0 : x0 + BW].reshape(-1)
            m2 = mu2f[y0 : y0 + BH, x0 : x0 + BW].reshape(-1)
            xc = np.concatenate([p1.reshape(t1, -1) - m1, p2.reshape(t2, -1) - m2], 0)
            k, u, b = fit_patch(xc, target_mse, cfg["kMax"])
            packed = quant_train(xc, u, b, steps)
            packed["y0"], packed["x0"] = y0, x0
            k_hist[k] += 1
            if k == cfg["kMax"]:
                n_sat += 1
            if k > 0:
                basis_b += packed["B"].nbytes + packed["Bs"].nbytes  # B stored once
                coeff_b += packed["U"].nbytes + packed["Us"].nbytes
                u1, u2 = packed["Uq"][:t1], packed["Uq"][t1:]
                rec1[:, y0 : y0 + BH, x0 : x0 + BW] = np.clip(np.round((m1 + u1 @ packed["Bq"]).reshape(t1, BH, BW, 3)), 0, 255).astype(np.uint8)
                rec2[:, y0 : y0 + BH, x0 : x0 + BW] = np.clip(np.round((m2 + u2 @ packed["Bq"]).reshape(t2, BH, BW, 3)), 0, 255).astype(np.uint8)
            items.append(packed)
    rec = np.concatenate([rec1, rec2], 0)
    sl = np.concatenate([sl1, sl2], 0)
    del rec1, rec2, sl1, sl2
    sc = score(sl, rec, a0, BW)
    raw_b, z_b = pack_origin(items, len(sl))
    mean_k = float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1))
    row = {
        "id": cid,
        "phase": "M2",
        "arm": "A",
        "shot": cfg["shot"],
        "kind": "shared-B",
        "flux": cfg.get("flux"),
        "i0": a0,
        "i1": b1,
        "frames": int(len(sl)),
        "trainSteps": steps,
        "kMax": cfg["kMax"],
        "targetPsnr": cfg["targetPsnr"],
        "hop": BW,
        "yuv": False,
        "merge": "sharedB",
        **sc,
        "meanK": round(mean_k, 3),
        "nSat": int(n_sat),
        "nPatches": len(items),
        "kHist": {str(k): int(n) for k, n in sorted(k_hist.items())},
        "kRoot": 0,
        "rootBytes": 0,
        "basisBytes": int(basis_b),
        "coeffBytes": int(coeff_b),
        "meanJpegBytes": int(len(blob1) + len(blob2)),
        "rawBytes": int(raw_b),
        "originBytes": int(z_b),
        "bytesPerSec": round(z_b / max(len(sl) / FPS, 1e-6), 1),
        "vsSourceH264": round(z_b / max((len(sl) / FPS) * (SOURCE_H264_BPS / 8), 1), 3),
        "macs": round(mean_k * len(items) * BW * BH * 3, 1),
        "encodeSec": round(time.time() - t0, 2),
        "rssMb": round(max(rss0, rss_mb()), 1),
        "ladder": {},
    }
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log(
        f"{cid:28s} shared-B   PSNR {row['meanPsnr']:6.2f}/{row['minPsnr']:5.2f}  "
        f"orig={row['originBytes']/1e6:.3f}MB  {row['encodeSec']:.1f}s"
    )
    return row


def phase_train(files: list[Path], reps: dict[str, dict], done: set[str]) -> int:
    if TRAIN_CHOICE.exists():
        ch = json.loads(TRAIN_CHOICE.read_text())
        log(f"train_choice resume steps={ch['steps']} ({ch['note']})")
        return int(ch["steps"])
    rows = []
    for tag in ("L", "B"):
        s = reps[tag]
        for steps in (0, 2, 8):
            cfg = {
                "phase": "T",
                "arm": "A",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "trainSteps": steps,
                "kMax": K_MAX,
                "targetPsnr": 32.5,
                "hop": BW,
                "yuv": False,
                "decodeLadder": False,
            }
            cfg["id"] = make_id(cfg)
            row = run_one(cfg, files, {}, done)
            if row is None:
                # already recorded
                for line in JSONL.read_text().splitlines():
                    r = json.loads(line)
                    if r["id"] == cfg["id"]:
                        rows.append(r)
                        break
            else:
                rows.append(row)
            done.add(cfg["id"])
    # decide
    def grab(sid, st):
        for r in rows:
            if r["shot"] == sid and r["trainSteps"] == st:
                return r
        return None

    notes = []
    deltas = []
    for tag in ("L", "B"):
        sid = reps[tag]["sid"]
        r0, r2, r8 = grab(sid, 0), grab(sid, 2), grab(sid, 8)
        if not (r0 and r2 and r8):
            continue
        d20 = r2["meanPsnr"] - r0["meanPsnr"]
        d80 = r8["meanPsnr"] - r0["meanPsnr"]
        deltas.append(d80)
        notes.append(
            f"{sid} ({reps[tag]['kind']}): 0={r0['meanPsnr']:.3f} 2={r2['meanPsnr']:.3f} ({d20:+.3f}) "
            f"8={r8['meanPsnr']:.3f} ({d80:+.3f})"
        )
        log("  " + notes[-1])
    max_d = max(deltas) if deltas else 0.0
    if max_d >= 0.15:
        chosen, why = 8, f"ALS +{max_d:.3f} dB at 640 vs 0 steps — t1r 'training is dead' does not hold"
    elif any((grab(reps[t]["sid"], 2)["meanPsnr"] - grab(reps[t]["sid"], 0)["meanPsnr"]) >= 0.05 for t in ("L", "B") if grab(reps[t]["sid"], 2) and grab(reps[t]["sid"], 0)):
        chosen, why = 2, "steps=2 moves ≥0.05 dB; keep shipped ALS"
    else:
        chosen, why = 2, "0≈2≈8 at 640; keep steps=2 for causal vs v4r (not 0)"
    TRAIN_CHOICE.write_text(json.dumps({"steps": chosen, "note": why, "detail": notes}, indent=2))
    log(f"TRAIN CHOICE steps={chosen}  {why}")
    return chosen


def write_markdown(rows: list[dict], shots: list[dict], reps: dict, pair: tuple, steps: int) -> None:
    def fmt(r: dict) -> str:
        br = "—" if r.get("bridgePsnr") is None else f"{r['bridgePsnr']:.2f}"
        return (
            f"| `{r['id']}` | {r.get('shot','')} | {r.get('kind','')} | {r['meanPsnr']:.2f} | {r['minPsnr']:.2f} | "
            f"{r.get('ssim',0):.3f} | {r['leftover']:.2f} | {r['meanK']:.2f} | {r['nSat']} | "
            f"{r['originBytes']/1e3:.0f} | {r['seamRatio']:.2f} | {br} | {r['rssMb']:.0f} | {r['encodeSec']:.1f} |"
        )

    head = (
        "| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Attempt v4.t2r — native 640×360, per-shot, three representations",
        "",
        "Branch: `attempt/v4.t2r` off `attempt/v4.t1r`.  ",
        "Frozen lab: no `src/` / `reconstruct.mp4` rewrite.",
        "",
        "Raster: **640×360** crop, canvas 640×384 (6.7% pad, same ratio as 180→192). "
        "Source file is already 640×360 24 fps. Frames: `/tmp/bbb/frames-640` PNG (H.264 decode, no second JPEG).",
        "",
        f"ALS: measured at 640 before freezing. Chosen **TRAIN_STEPS={steps}**. See `train_choice.json`.",
        "",
        "## Shots (re-detected at 640, cutHist=0.62, flux>12)",
        "",
        "| sid | frames | t | flux | kind |",
        "|---|---|---|---|---|",
    ]
    for s in shots:
        mark = ""
        for tag, r in reps.items():
            if r["sid"] == s["sid"]:
                mark = f" **rep {tag}**"
        lines.append(f"| {s['sid']} | [{s['i0']},{s['i1']}) | {s['t0']:.2f}–{s['t1']:.2f}s | {s['flux']:.2f} | {s['kind']}{mark} |")
    lines += [
        "",
        f"Reps: L={reps['L']['sid']} (lowest flux locked), T={reps['T']['sid']} (median of rest), "
        f"B={reps['B']['sid']} (highest flux — no flux>8 busy shot in this 90s).",
        f"Merge pair: {pair[0]['sid']}+{pair[1]['sid']} (busiest + neighbor).",
        "",
        "Arms: **A** disjoint 16×16 · **B** hop-8 Hann OLA · **C** global K_root=8 then 16×16 leftover.",
        "Phase Q is the same three arms at a **40 dB** knife. M1 = naive concatenated SVD. M2 = shared B, per-shot μ/U.",
        "",
    ]
    phases = [
        ("T", "Phase T — ALS at 640 (do not trust t1r steps=0)"),
        ("A", "Phase A — disjoint 16×16, every shot, 32.5 dB"),
        ("B", "Phase B — 16×16 overlap-add, reps"),
        ("C", "Phase C — global split tree, reps"),
        ("Q", "Phase Q — 40 dB knife, reps × A/B/C"),
        ("K", "Phase K — K_MAX=32 smoke on busiest, arm A"),
        ("Y", "Phase Y — YUV 4:2:0 smoke, arm A locked"),
        ("M1", "Phase M1 — naive merge of two adjacent shots (ignore the cut)"),
        ("M2", "Phase M2 — shared spatial B across two shots (episode-bases proxy)"),
    ]
    by = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for ph, title in phases:
        lines += [f"## {title}", "", head]
        for r in by.get(ph, []):
            lines.append(fmt(r))
        lines.append("")

    lines += ["## Phase E — K′ ladders (32.5 origins)", ""]
    for r in rows:
        if r.get("ladder"):
            lines.append(f"### {r['id']}")
            lines.append("")
            lines.append("| K′ | mean dB | min dB |")
            lines.append("|---|---|---|")
            for kp, v in r["ladder"].items():
                lines.append(f"| {kp} | {v['meanPsnr']:.2f} | {v['minPsnr']:.2f} |")
            lines.append("")

    def grab(phase, shot, arm, **kw):
        for r in rows:
            if r["phase"] != phase or r.get("shot") != shot or r.get("arm") != arm:
                continue
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    lines += ["## Kill sentences", ""]
    # training
    trows = by.get("T", [])
    if trows:
        lines.append(f"- **Training at 640:** chosen steps={steps}. " + (TRAIN_CHOICE.read_text() if TRAIN_CHOICE.exists() else ""))
    for tag, s in reps.items():
        a = grab("A", s["sid"], "A")
        b = grab("B", s["sid"], "B")
        c = grab("C", s["sid"], "C")
        if a:
            lines.append(
                f"- **{s['sid']} raster/A:** native 16×16 {a['meanPsnr']:.2f}/{a['minPsnr']:.2f} dB SSIM {a['ssim']:.3f} "
                f"origin {a['originBytes']/1e3:.0f}KB seamR {a['seamRatio']:.2f} bridge320 {a.get('bridgePsnr')} "
                f"(t1r 320 was ~35 dB on locked 10s)."
            )
        if a and b:
            lines.append(
                f"- **{s['sid']} overlap:** seamR {b['seamRatio']:.2f} vs A {a['seamRatio']:.2f}, "
                f"min {b['minPsnr']:.2f} vs {a['minPsnr']:.2f}, origin {b['originBytes']/1e3:.0f} vs {a['originBytes']/1e3:.0f} KB."
            )
        if a and c:
            lines.append(
                f"- **{s['sid']} tree:** {c['meanPsnr']:.2f}/{c['minPsnr']:.2f} origin {c['originBytes']/1e3:.0f}KB "
                f"kRoot={c.get('kRoot')} vs A {a['meanPsnr']:.2f}/{a['originBytes']/1e3:.0f}KB."
            )
    for tag, s in reps.items():
        q = [r for r in by.get("Q", []) if r.get("shot") == s["sid"]]
        if q:
            bits = ", ".join(f"{r['arm']} {r['meanPsnr']:.2f}dB/{r['originBytes']/1e3:.0f}KB" for r in q)
            lines.append(f"- **{s['sid']} 40 dB knife:** {bits}.")
    k = by.get("K", [])
    if k:
        lines.append(f"- **K_MAX=32:** {k[0]['meanPsnr']:.2f}/{k[0]['minPsnr']:.2f} sat {k[0]['nSat']} origin {k[0]['originBytes']/1e3:.0f}KB.")
    y = by.get("Y", [])
    if y:
        a = grab("A", y[0]["shot"], "A")
        if a:
            lines.append(
                f"- **YUV:** {y[0]['meanPsnr']:.2f} dB origin {y[0]['originBytes']/1e3:.0f}KB vs RGB "
                f"{a['meanPsnr']:.2f} / {a['originBytes']/1e3:.0f}KB."
            )
    m1 = by.get("M1", [])
    m2 = by.get("M2", [])
    if m1:
        for r in m1:
            lines.append(
                f"- **M1 {r['arm']} naive merge {r['shot']}:** {r['meanPsnr']:.2f}/{r['minPsnr']:.2f} "
                f"origin {r['originBytes']/1e3:.0f}KB (one μ, one timeline — the cut is inside the SVD)."
            )
    if m2:
        lines.append(
            f"- **M2 shared B:** {m2[0]['meanPsnr']:.2f}/{m2[0]['minPsnr']:.2f} origin {m2[0]['originBytes']/1e3:.0f}KB. "
            "This is the episode-scale spatial-bases proxy (per-shot μ/U, one B)."
        )
    lines += [
        "",
        "## How to reproduce",
        "",
        "```",
        "python3 encoder/v4.t2r/sweep.py",
        "```",
        "",
        "Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "README.md").write_text(text)
    (ROOT / "attempts" / "v4.t2r.md").write_text(text)
    log(f"wrote {OUT/'README.md'} and attempts/v4.t2r.md")


def rss_smoke(files: list[Path], shots: list[dict]) -> None:
    s = min(shots, key=lambda x: x["t"])
    log(f"RSS smoke C on {s['sid']} T={s['t']} rss0={rss_mb():.0f}MB")
    sl = load_range(files, s["i0"], s["i1"])
    target_mse = 255.0 * 255.0 / (10 ** (32.5 / 10.0))
    rec, *_ = encode_tree(sl, target_mse, K_MAX, 0)
    peak = rss_mb()
    log(f"RSS smoke peak {peak:.0f}MB recon {rec.shape}")
    if peak > 3000:
        raise SystemExit(f"kill: peak RSS {peak:.0f} MB > 3 GB on the shortest shot")
    del sl, rec


def main() -> None:
    if not LOG.exists():
        LOG.write_text("")
    log("=== v4.t2r start ===")
    files = frame_files()
    log(f"{len(files)} frames 640×360 from {FRAMES}")
    if SHOTS_JSON.exists():
        shots = json.loads(SHOTS_JSON.read_text())
        log(f"reusing {len(shots)} shots")
    else:
        log("detecting shots at 640…")
        t0 = time.time()
        shots = detect_shots(files)
        SHOTS_JSON.write_text(json.dumps(shots, indent=2))
        log(f"{len(shots)} shots in {time.time()-t0:.1f}s")
    for s in shots:
        log(f"  {s['sid']} [{s['i0']:4d},{s['i1']:4d}) T={s['t']:3d} flux={s['flux']:.2f} {s['kind']}")
    reps = pick_reps(shots)
    pair = pick_merge_pair(shots)
    log(f"reps L={reps['L']['sid']} T={reps['T']['sid']} B={reps['B']['sid']}  merge {pair[0]['sid']}+{pair[1]['sid']}")
    rss_smoke(files, shots)
    done = already_done()
    steps = phase_train(files, reps, done)
    done = already_done()
    cfgs, reps, pair = configs(shots, steps)
    log(f"{len(cfgs)} configs after train choice steps={steps}")
    for cfg in cfgs:
        if cfg.get("merge") == "sharedB":
            run_m2(cfg, files, steps, done)
        else:
            run_one(cfg, files, {s["sid"]: s for s in shots}, done)
        done = already_done()
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    write_markdown(rows, shots, reps, pair, steps)
    log(f"=== done {len(rows)} rows ===")


if __name__ == "__main__":
    main()
