#!/usr/bin/env python3
"""v4.t3r: native 640×360, per-shot. Tile size, cheap seams, JPEG/int4 B, trees.

Does not write lab UI / reconstruct.mp4 / analysis.json.
H1 = bottom-up exclusive merge. H2 = residual (32→8 and full→16, rate-constrained).
"""
from __future__ import annotations

import io
import json
import math
import struct
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "attempts" / "v4.t3r"
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"
LOG = OUT / "sweep.log"
SHOTS_JSON = OUT / "shots.json"
LEAF_CHOICE = OUT / "leaf_choice.json"

FRAMES = Path("/tmp/bbb/frames-640")
FRAMES320 = Path("/tmp/bbb/frames-v1")

FPS = 24
W, H_DISP, H = 640, 360, 384
JPEG_Q = 84
K_MAX = 16
TRAIN_STEPS = 2
TARGET_PSNR = 32.5
SOURCE_H264_BPS = 530_000


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


def qint4(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if axis == 0:
        scale = np.max(np.abs(arr), axis=tuple(range(1, arr.ndim)), keepdims=True)
    else:
        scale = np.max(np.abs(arr), axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    codes = np.clip(np.round(arr / scale * 7.0), -7, 7).astype(np.int8)
    dequant = codes.astype(np.float32) * scale / 7.0
    return dequant, codes, scale.astype(np.float32)


def pack_int4_blob(codes: np.ndarray) -> bytes:
    flat = np.ascontiguousarray(codes, dtype=np.int8).ravel()
    if flat.size % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    a = flat.view(np.uint8) & np.uint8(0x0F)
    return (a[0::2] | (a[1::2] << 4)).tobytes()


def packed_int4_bytes(codes: np.ndarray) -> int:
    return (codes.size + 1) // 2


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
    if tlen == 1 or d == 0:
        return 0, empty_u, empty_b
    energy = float(np.mean(Xc * Xc))
    if energy <= target_mse:
        return 0, empty_u, empty_b
    denom = float(tlen * d)
    kcap = min(k_max, tlen - 1, d)
    if tlen <= d:
        g = Xc @ Xc.T
        evals, evecs = np.linalg.eigh(g)
        order = np.argsort(evals)[::-1]
        evals = np.maximum(evals[order], 0.0)
        evecs = evecs[:, order].astype(np.float32)
        k = 0
        while k < kcap and (evals[k:].sum() / denom) > target_mse:
            k += 1
        if k == 0:
            return 0, empty_u, empty_b
        u = evecs[:, :k]
        b = u.T @ Xc
        return k, u, b.astype(np.float32)
    g = Xc.T @ Xc
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order].astype(np.float32)
    k = 0
    while k < kcap and (evals[k:].sum() / denom) > target_mse:
        k += 1
    if k == 0:
        return 0, empty_u, empty_b
    v = evecs[:, :k]
    u = Xc @ v
    b = v.T.copy()
    return k, u.astype(np.float32), b.astype(np.float32)


def jpeg_on_b(b: np.ndarray, bw: int, bh: int, quality: int) -> tuple[np.ndarray, bytes, np.ndarray, np.ndarray]:
    """Affine-map each eigenpatch to 0–255, JPEG mosaic, unmap. Returns Bq, blob, mn, span."""
    k, d = b.shape
    mn = b.min(axis=1).astype(np.float32)
    mx = b.max(axis=1).astype(np.float32)
    span = np.maximum(mx - mn, 1e-6)
    if k == 0:
        return b, b"", mn, span
    cols = max(1, int(math.ceil(math.sqrt(k))))
    rows = int(math.ceil(k / cols))
    canvas = np.zeros((rows * bh, cols * bw, 3), np.uint8)
    for i in range(k):
        r, c = divmod(i, cols)
        tile = ((b[i] - mn[i]) / span[i] * 255.0).reshape(bh, bw, 3)
        canvas[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw] = np.clip(np.round(tile), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(canvas).save(buf, "JPEG", quality=quality, optimize=True)
    blob = buf.getvalue()
    rec = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
    out = np.zeros_like(b, dtype=np.float32)
    for i in range(k):
        r, c = divmod(i, cols)
        tile = rec[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw].astype(np.float32)
        out[i] = (tile.reshape(-1) / 255.0) * span[i] + mn[i]
    return out, blob, mn, span


def quant_train(
    Xc: np.ndarray,
    u: np.ndarray,
    b: np.ndarray,
    steps: int,
    *,
    quant_u: str = "int8",
    quant_b: str = "int8",
    jpeg_bq: int | None = None,
    bw: int = 16,
    bh: int = 16,
) -> dict:
    k = int(u.shape[1]) if u.ndim == 2 else 0
    packed: dict = {
        "k": k,
        "quantU": quant_u,
        "quantB": "jpeg" if jpeg_bq is not None else quant_b,
        "jpegB": b"",
        "Bmn": np.zeros((0,), np.float32),
        "Bsp": np.zeros((0,), np.float32),
        "role": "block",
        "add": False,
    }
    tlen, d = Xc.shape
    if k == 0:
        packed.update(
            {
                "Uq": u,
                "Bq": b,
                "U": np.zeros((0, tlen), np.int8),
                "B": np.zeros((0, d), np.int8),
                "Us": np.zeros((0,), np.float32),
                "Bs": np.zeros((0,), np.float32),
            }
        )
        return packed
    qfu = qint4 if quant_u == "int4" else qint8
    qfb = qint4 if quant_b == "int4" else qint8
    uq, uc, us = qfu(u.T, axis=0)
    uq = uq.T
    jpeg_blob = b""
    mn = np.zeros((k,), np.float32)
    sp = np.ones((k,), np.float32)
    if jpeg_bq is not None:
        bq, jpeg_blob, mn, sp = jpeg_on_b(b, bw, bh, jpeg_bq)
        bc = np.zeros((0, d), np.int8)
        bs = sp
    else:
        bq, bc, bs = qfb(b, axis=0)
    if steps > 0:
        uq, bq = train_factors(Xc, uq, bq, steps=steps)
        uq, uc, us = qfu(uq.T, axis=0)
        uq = uq.T
        if jpeg_bq is not None:
            bq, jpeg_blob, mn, sp = jpeg_on_b(bq, bw, bh, jpeg_bq)
            bc = np.zeros((0, d), np.int8)
            bs = sp
        else:
            bq, bc, bs = qfb(bq, axis=0)
    packed.update(
        {
            "Uq": uq,
            "Bq": bq,
            "U": uc,
            "B": bc,
            "Us": us.reshape(k).astype(np.float32),
            "Bs": np.asarray(bs).reshape(-1).astype(np.float32)[:k] if jpeg_bq is None else sp.astype(np.float32),
            "jpegB": jpeg_blob,
            "Bmn": mn.astype(np.float32),
            "Bsp": sp.astype(np.float32),
        }
    )
    return packed


def item_raw_len(it: dict) -> int:
    k = int(it.get("k", 0))
    n = 7  # k + y0 + x0 + sz
    if k == 0:
        return n
    n += np.asarray(it["Us"]).nbytes
    if it.get("quantB") == "jpeg":
        n += 4 + len(it.get("jpegB") or b"") + np.asarray(it.get("Bmn", [])).nbytes + np.asarray(it.get("Bsp", [])).nbytes
    elif it.get("quantB") == "int4":
        n += packed_int4_bytes(np.asarray(it["B"])) + np.asarray(it["Bs"]).nbytes
    else:
        n += np.asarray(it["B"]).nbytes + np.asarray(it["Bs"]).nbytes
    if it.get("quantU") == "int4":
        n += packed_int4_bytes(np.asarray(it["U"]))
    else:
        n += np.asarray(it["U"]).nbytes
    return n


def pack_origin(items: list[dict], n_frames: int, extra: bytes = b"") -> tuple[int, int]:
    body = bytearray()
    body += struct.pack("<HHHIH", W, H_DISP, FPS, n_frames, 1)
    body += extra
    for it in items:
        k = int(it.get("k", 0))
        y0 = int(it.get("y0", 0))
        x0 = int(it.get("x0", 0))
        sz = int(it.get("sz", it.get("bw", 16)))
        qtag = {"int8": 0, "int4": 1, "jpeg": 2}.get(it.get("quantB", "int8"), 0)
        body += struct.pack("<BHHHB", k, y0, x0, sz, qtag)
        if k == 0:
            continue
        body += np.asarray(it["Us"], dtype=np.float32).tobytes()
        if it.get("quantB") == "jpeg":
            body += np.asarray(it.get("Bmn"), dtype=np.float32).tobytes()
            body += np.asarray(it.get("Bsp"), dtype=np.float32).tobytes()
            blob = it.get("jpegB") or b""
            body += struct.pack("<I", len(blob))
            body += blob
        elif it.get("quantB") == "int4":
            body += np.asarray(it["Bs"], dtype=np.float32).tobytes()
            body += pack_int4_blob(np.asarray(it["B"]))
        else:
            body += np.asarray(it["Bs"], dtype=np.float32).tobytes()
            body += np.asarray(it["B"], dtype=np.int8).tobytes()
        if it.get("quantU") == "int4":
            body += pack_int4_blob(np.asarray(it["U"]))
        else:
            body += np.asarray(it["U"], dtype=np.int8).tobytes()
    raw = bytes(body)
    z = zlib.compress(raw, 9)
    return len(raw), len(z) + 9


def seam_mae(src: np.ndarray, rec: np.ndarray, bw: int, bh: int) -> tuple[float, float, float]:
    err = np.abs(src[:H_DISP].astype(np.float32) - rec[:H_DISP].astype(np.float32)).mean(axis=2)
    h, w = err.shape
    if bw <= 0 or bw >= w:
        return 0.0, float(err.mean()), 0.0
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


def frame_files() -> list[Path]:
    files = sorted(FRAMES.glob("*.png"))
    if len(files) != FPS * 90:
        raise SystemExit(f"want {FPS * 90} pngs in {FRAMES}, got {len(files)}")
    return files


def load_range(files: list[Path], i0: int, i1: int) -> np.ndarray:
    sl = [pad_frame(np.array(Image.open(files[i]).convert("RGB"))) for i in range(i0, i1)]
    return np.stack(sl, 0)


def load320(i0: int, i1: int) -> np.ndarray:
    files = sorted(FRAMES320.glob("*.jpg"))
    return np.stack([np.array(Image.open(files[i]).convert("RGB"))[:180, :320] for i in range(i0, i1)], 0)


def down320(rgb: np.ndarray) -> np.ndarray:
    return np.array(Image.fromarray(rgb[:H_DISP, :W]).resize((320, 180), Image.BOX))


def shot_mu(sl: np.ndarray) -> np.ndarray:
    acc = np.zeros((H, W, 3), np.float64)
    for i in range(0, sl.shape[0], 16):
        acc += sl[i : i + 16].sum(axis=0, dtype=np.float64)
    return (acc / sl.shape[0]).astype(np.float32)


def trap_window(bw: int, hop: int) -> np.ndarray:
    """COLA window. hop=bw/2 uses Hann (t2r B). Else a trapezoid with w[-ov:]=1-w[:ov]."""
    ov = max(bw - hop, 0)
    if hop * 2 == bw:
        n = np.arange(bw, dtype=np.float32)
        w = 0.5 - 0.5 * np.cos(2.0 * np.pi * (n + 0.5) / bw)
        return np.outer(w, w).astype(np.float32)
    w = np.ones(bw, np.float32)
    if ov > 0:
        fade = np.linspace(0.0, 1.0, ov, endpoint=False, dtype=np.float32)
        w[:ov] = fade
        w[-ov:] = 1.0 - fade
    return np.outer(w, w).astype(np.float32)



def tile_starts(span: int, size: int, hop: int) -> list[int]:
    xs = list(range(0, span - size + 1, hop))
    if not xs:
        xs = [0]
    if xs[-1] != span - size:
        xs.append(span - size)
    return xs


def apply_patch(rec: np.ndarray, it: dict, mu_f: np.ndarray, kprime: int | None) -> None:
    k = int(it.get("k", 0))
    y0, x0 = int(it["y0"]), int(it["x0"])
    bh, bw = int(it.get("bh", it.get("sz", 16))), int(it.get("bw", it.get("sz", 16)))
    tlen = rec.shape[0]
    kuse = k if kprime is None else min(k, kprime)
    if it.get("add"):
        if kuse <= 0:
            return
        extra = (it["Uq"][:, :kuse] @ it["Bq"][:kuse]).reshape(tlen, bh, bw, 3)
        rec[:, y0 : y0 + bh, x0 : x0 + bw] += extra
        return
    m = mu_f[y0 : y0 + bh, x0 : x0 + bw]
    if kuse <= 0:
        rec[:, y0 : y0 + bh, x0 : x0 + bw] = m
        return
    extra = (it["Uq"][:, :kuse] @ it["Bq"][:kuse]).reshape(tlen, bh, bw, 3)
    rec[:, y0 : y0 + bh, x0 : x0 + bw] = m + extra


def recon_items(tlen: int, mu_f: np.ndarray, items: list[dict], kprime: int | None, peel: str = "flat") -> np.ndarray:
    rec = np.repeat(mu_f[None], tlen, 0)
    for it in items:
        if peel == "h2":
            role = it.get("role", "leaf")
            if kprime is None:
                apply_patch(rec, it, mu_f, None)
            elif kprime < 0:
                if role == "parent":
                    apply_patch(rec, it, mu_f, None)
            else:
                if role == "parent":
                    apply_patch(rec, it, mu_f, None)
                else:
                    apply_patch(rec, it, mu_f, kprime)
        else:
            apply_patch(rec, it, mu_f, kprime)
    return np.clip(np.round(rec), 0, 255).astype(np.uint8)


def blend_seams(rec_u: np.ndarray, bw: int, bh: int, ov: int) -> np.ndarray:
    """Decode-only deblock: mix ov pixels on each side of a disjoint seam, strength decaying from the cut.

    Not a ramp of interiors — that wiped 8 dB on smoke. Same origin as disjoint A.
    """
    if ov <= 0:
        return rec_u
    rec = rec_u.astype(np.float32)
    out = rec.copy()
    _, hh, ww, _ = rec.shape

    def mix_1d(a, b, strength):
        return a * (1.0 - strength) + b * strength

    for x in range(bw, ww, bw):
        for i in range(ov):
            a = 0.10 * (ov - i) / ov
            L, R = x - 1 - i, x + i
            if L < 0 or R >= ww:
                continue
            origL = rec[:, :, L, :]
            origR = rec[:, :, R, :]
            out[:, :, L, :] = mix_1d(origL, origR, a)
            out[:, :, R, :] = mix_1d(origR, origL, a)
    for y in range(bh, hh, bh):
        if y >= H_DISP + ov:
            break
        for i in range(ov):
            a = 0.10 * (ov - i) / ov
            T, B = y - 1 - i, y + i
            if T < 0 or B >= hh:
                continue
            origT = rec[:, T, :, :]
            origB = rec[:, B, :, :]
            out[:, T, :, :] = mix_1d(origT, origB, a)
            out[:, B, :, :] = mix_1d(origB, origT, a)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def encode_block(
    sl: np.ndarray,
    mu_f: np.ndarray,
    y0: int,
    x0: int,
    bh: int,
    bw: int,
    target_mse: float,
    k_max: int,
    steps: int,
    quant_u: str = "int8",
    quant_b: str = "int8",
    jpeg_bq: int | None = None,
) -> dict:
    tlen = sl.shape[0]
    p = sl[:, y0 : y0 + bh, x0 : x0 + bw].astype(np.float32)
    m = mu_f[y0 : y0 + bh, x0 : x0 + bw]
    xc = (p - m).reshape(tlen, -1)
    k, u, b = fit_patch(xc, target_mse, k_max)
    packed = quant_train(xc, u, b, steps, quant_u=quant_u, quant_b=quant_b, jpeg_bq=jpeg_bq, bw=bw, bh=bh)
    packed.update({"y0": y0, "x0": x0, "bw": bw, "bh": bh, "sz": bw, "bytes": item_raw_len(packed)})
    return packed


def paint_exclusive(rec_u: np.ndarray, mu_f: np.ndarray, items: list[dict]) -> np.ndarray:
    tlen = rec_u.shape[0]
    for it in items:
        y0, x0 = it["y0"], it["x0"]
        bh, bw = it["bh"], it["bw"]
        m = mu_f[y0 : y0 + bh, x0 : x0 + bw]
        k = int(it["k"])
        if k == 0:
            rec_u[:, y0 : y0 + bh, x0 : x0 + bw] = np.clip(np.round(m), 0, 255).astype(np.uint8)
            continue
        recp = m + (it["Uq"] @ it["Bq"]).reshape(tlen, bh, bw, 3)
        rec_u[:, y0 : y0 + bh, x0 : x0 + bw] = np.clip(np.round(recp), 0, 255).astype(np.uint8)
    return rec_u


def encode_disjoint(
    sl: np.ndarray,
    *,
    bw: int,
    bh: int,
    target_mse: float,
    k_max: int,
    steps: int,
    quant_u: str = "int8",
    quant_b: str = "int8",
    jpeg_bq: int | None = None,
    hop: int | None = None,
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    tlen = sl.shape[0]
    mu = shot_mu(sl)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    hop = bw if hop is None else hop
    ys = tile_starts(H, bh, hop)
    xs = tile_starts(W, bw, hop)
    window = trap_window(bw, hop) if hop < bw else None
    items: list[dict] = []
    k_hist: Counter[int] = Counter()
    n_sat = basis_b = coeff_b = 0
    n_tiles = len(ys) * len(xs)
    if window is None:
        rec_u = np.repeat(mu_u[None], tlen, 0)
        acc = wgt = None
    else:
        rec_u = None
        acc = np.zeros((tlen, H, W, 3), np.float32)
        wgt = np.zeros((H, W), np.float32)
        win = window[:, :, None]
    done_tiles = 0
    for y0 in ys:
        for x0 in xs:
            it = encode_block(sl, mu_f, y0, x0, bh, bw, target_mse, k_max, steps, quant_u, quant_b, jpeg_bq)
            k = int(it["k"])
            k_hist[k] += 1
            if k == k_max:
                n_sat += 1
            if k > 0:
                basis_b += it["bytes"]
                coeff_b += np.asarray(it["U"]).nbytes + np.asarray(it["Us"]).nbytes
            items.append(it)
            m = mu_f[y0 : y0 + bh, x0 : x0 + bw]
            if k > 0:
                recp = m + (it["Uq"] @ it["Bq"]).reshape(tlen, bh, bw, 3)
            else:
                recp = np.broadcast_to(m, (tlen, bh, bw, 3))
            if window is None:
                rec_u[:, y0 : y0 + bh, x0 : x0 + bw] = np.clip(np.round(recp), 0, 255).astype(np.uint8)
            else:
                acc[:, y0 : y0 + bh, x0 : x0 + bw] += recp * win
                wgt[y0 : y0 + bh, x0 : x0 + bw] += window
            done_tiles += 1
            if n_tiles >= 10000 and done_tiles % 10000 == 0:
                log(f"    tiles {done_tiles}/{n_tiles} rss={rss_mb():.0f}MB")
    if window is not None:
        rec_u = np.clip(np.round(acc / np.maximum(wgt, 1e-6)[None, :, :, None]), 0, 255).astype(np.uint8)
        del acc, wgt
    meters = {
        "kHist": {int(a): int(b) for a, b in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "nSat": int(n_sat),
        "nPatches": len(items),
        "basisBytes": int(basis_b),
        "coeffBytes": int(coeff_b),
        "meanJpegBytes": int(len(mu_blob)),
        "t": tlen,
        "mu_f": mu_f,
    }
    return rec_u, mu_blob, items, meters


def fit_global_adaptive(sl: np.ndarray, mu_f: np.ndarray, k_max: int, target_mse: float) -> tuple[int, np.ndarray, np.ndarray]:
    tlen = sl.shape[0]
    d = H * W * 3
    empty_u = np.zeros((tlen, 0), np.float32)
    empty_b = np.zeros((0, H, W, 3), np.float32)
    if tlen == 1:
        return 0, empty_u, empty_b
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
        return 0, empty_u, empty_b
    evals, evecs = np.linalg.eigh(g)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order], 0.0)
    evecs = evecs[:, order].astype(np.float32)
    denom = float(tlen * d)
    kcap = min(k_max, tlen - 1)
    k = 0
    while k < kcap and (evals[k:].sum() / denom) > target_mse:
        k += 1
    if k == 0:
        return 0, empty_u, empty_b
    u = evecs[:, :k]
    bvol = np.zeros((k, H, W, 3), np.float32)
    for y0 in range(0, H, band):
        ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
        t, hh, ww, cc = ch.shape
        bvol[:, y0 : y0 + hh] = (u.T @ ch.reshape(t, -1)).reshape(k, hh, ww, cc)
    return k, u, bvol


def encode_centered(
    xc_vol: np.ndarray,
    y0: int,
    x0: int,
    bh: int,
    bw: int,
    target_mse: float,
    k_max: int,
    steps: int,
) -> dict:
    tlen = xc_vol.shape[0]
    xc = xc_vol.reshape(tlen, -1)
    k, u, b = fit_patch(xc, target_mse, k_max)
    packed = quant_train(xc, u, b, steps, bw=bw, bh=bh)
    packed.update({"y0": y0, "x0": x0, "bw": bw, "bh": bh, "sz": bw, "bytes": 0})
    packed["bytes"] = item_raw_len(packed)
    return packed


def encode_merge(
    sl: np.ndarray, leaf: int, target_mse: float, k_max: int, steps: int
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    """H1: exclusive bottom-up merge from leaf … 128, then optional full-frame."""
    tlen = sl.shape[0]
    mu = shot_mu(sl)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    ny, nx = H // leaf, W // leaf
    log(f"    H1 leaf={leaf} fitting {ny}x{nx} leaves")
    items: list[dict] = []
    for iy in range(ny):
        for ix in range(nx):
            it = encode_block(sl, mu_f, iy * leaf, ix * leaf, leaf, leaf, target_mse, k_max, steps)
            it["role"] = "block"
            items.append(it)
        if (iy + 1) % 16 == 0:
            log(f"    leaf rows {iy + 1}/{ny} rss={rss_mb():.0f}MB")
    merge_frac: dict[str, float] = {}
    size = leaf
    while size * 2 <= 128 and W % (size * 2) == 0 and H % (size * 2) == 0:
        psz = size * 2
        ncell = (H // psz) * (W // psz)
        merged = 0
        new_items: list[dict] = []
        for y0 in range(0, H, psz):
            for x0 in range(0, W, psz):
                children = [it for it in items if y0 <= it["y0"] < y0 + psz and x0 <= it["x0"] < x0 + psz]
                parent = encode_block(sl, mu_f, y0, x0, psz, psz, target_mse, k_max, steps)
                parent["role"] = "block"
                child_bytes = sum(int(c["bytes"]) for c in children)
                p = sl[:, y0 : y0 + psz, x0 : x0 + psz].astype(np.float32)
                m = mu_f[y0 : y0 + psz, x0 : x0 + psz]
                recp = m + ((parent["Uq"] @ parent["Bq"]).reshape(tlen, psz, psz, 3) if parent["k"] else 0)
                mse = float(np.mean((p - recp) ** 2))
                if mse <= target_mse * 1.05 and parent["bytes"] <= child_bytes:
                    new_items.append(parent)
                    merged += 1
                else:
                    new_items.extend(children)
        merge_frac[str(psz)] = merged / max(ncell, 1)
        log(f"    merge {size}→{psz}: {merged}/{ncell} ({merge_frac[str(psz)]:.2f})")
        items = new_items
        size = psz

    # full-frame via banded Gram (do not materialize T×H×W×3 Xc)
    kr, u, bvol = fit_global_adaptive(sl, mu_f, k_max, target_mse)
    child_bytes = sum(int(c["bytes"]) for c in items)
    full_merged = False
    if kr > 0:
        uq, uc, us = qint8(u.T, axis=0)
        uq = uq.T
        bq, bc, bs = qint8(bvol.reshape(kr, -1), axis=0)
        parent = {
            "k": kr,
            "y0": 0,
            "x0": 0,
            "bw": W,
            "bh": H,
            "sz": W,
            "Uq": uq,
            "Bq": bq,
            "U": uc,
            "B": bc,
            "Us": us.reshape(kr).astype(np.float32),
            "Bs": bs.reshape(kr).astype(np.float32),
            "quantU": "int8",
            "quantB": "int8",
            "jpegB": b"",
            "role": "block",
            "add": False,
        }
        parent["bytes"] = item_raw_len(parent)
        energy = 0.0
        denom = float(tlen * H * W * 3)
        band = 8
        for y0 in range(0, H, band):
            ch = sl[:, y0 : y0 + band].astype(np.float32) - mu_f[y0 : y0 + band]
            rloc = np.einsum("tk,khwc->thwc", uq, bq.reshape(kr, H, W, 3)[:, y0 : y0 + ch.shape[1]])
            energy += float(np.square(ch - rloc).sum())
        mse = energy / denom
        full_merged = bool(mse <= target_mse * 1.05 and parent["bytes"] <= child_bytes)
        log(f"    merge full: {int(full_merged)} k={kr} bytes {parent['bytes']} vs {child_bytes} mse={mse:.1f}")
        if full_merged:
            items = [parent]
    else:
        log("    merge full: skip k=0")
    merge_frac["full"] = 1.0 if full_merged else 0.0

    rec_u = np.repeat(mu_u[None], tlen, 0)
    rec_u = paint_exclusive(rec_u, mu_f, items)
    kh: Counter[int] = Counter(int(it["k"]) for it in items)
    meters = {
        "kHist": {int(a): int(b) for a, b in kh.items()},
        "meanK": float(sum(k * n for k, n in kh.items()) / max(sum(kh.values()), 1)),
        "nSat": int(sum(1 for it in items if it["k"] == k_max)),
        "nPatches": len(items),
        "basisBytes": int(sum(it["bytes"] for it in items)),
        "coeffBytes": 0,
        "meanJpegBytes": int(len(mu_blob)),
        "t": tlen,
        "mu_f": mu_f,
        "mergeFrac": merge_frac,
        "sizes": Counter(int(it["bw"]) for it in items),
    }
    return rec_u, mu_blob, items, meters


def encode_res32(
    sl: np.ndarray, target_mse: float, k_max: int, steps: int, leaf: int = 8, parent: int = 32
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    """H2: residual parent×parent → leaf leftover, per-region pick cheaper of tree vs flat."""
    tlen = sl.shape[0]
    mu = shot_mu(sl)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    rec_u = np.repeat(mu_u[None], tlen, 0)
    items: list[dict] = []
    n_tree = n_flat = 0
    k_hist: Counter[int] = Counter()
    n_sat = 0
    for y0 in range(0, H, parent):
        for x0 in range(0, W, parent):
            ph, pw = min(parent, H - y0), min(parent, W - x0)
            p = sl[:, y0 : y0 + ph, x0 : x0 + pw].astype(np.float32)
            m = mu_f[y0 : y0 + ph, x0 : x0 + pw]
            xc32 = p - m
            flat_items = []
            for dy in range(0, ph, leaf):
                for dx in range(0, pw, leaf):
                    bh, bw = min(leaf, ph - dy), min(leaf, pw - dx)
                    it = encode_centered(xc32[:, dy : dy + bh, dx : dx + bw], y0 + dy, x0 + dx, bh, bw, target_mse, k_max, steps)
                    it["role"] = "leaf"
                    it["add"] = False
                    flat_items.append(it)
            flat_bytes = sum(it["bytes"] for it in flat_items)
            par = encode_centered(xc32, y0, x0, ph, pw, target_mse, k_max, steps)
            par["role"] = "parent"
            par["add"] = True
            pres = (par["Uq"] @ par["Bq"]).reshape(tlen, ph, pw, 3) if par["k"] else np.zeros((tlen, ph, pw, 3), np.float32)
            leftover = xc32 - pres
            tree_leaves = []
            for dy in range(0, ph, leaf):
                for dx in range(0, pw, leaf):
                    bh, bw = min(leaf, ph - dy), min(leaf, pw - dx)
                    it = encode_centered(leftover[:, dy : dy + bh, dx : dx + bw], y0 + dy, x0 + dx, bh, bw, target_mse, k_max, steps)
                    it["role"] = "leaf"
                    it["add"] = True
                    tree_leaves.append(it)
            tree_bytes = par["bytes"] + sum(it["bytes"] for it in tree_leaves)
            tile = rec_u[:, y0 : y0 + ph, x0 : x0 + pw].astype(np.float32)
            if tree_bytes < flat_bytes and par["k"] > 0:
                n_tree += 1
                items.append(par)
                items.extend(tree_leaves)
                tile = m + pres
                for it in tree_leaves:
                    k_hist[it["k"]] += 1
                    if it["k"] == k_max:
                        n_sat += 1
                    if it["k"] == 0:
                        continue
                    yy, xx, bh, bw = it["y0"] - y0, it["x0"] - x0, it["bh"], it["bw"]
                    tile[:, yy : yy + bh, xx : xx + bw] += (it["Uq"] @ it["Bq"]).reshape(tlen, bh, bw, 3)
                k_hist[par["k"]] += 1
            else:
                n_flat += 1
                items.extend(flat_items)
                for it in flat_items:
                    k_hist[it["k"]] += 1
                    if it["k"] == k_max:
                        n_sat += 1
                    yy, xx, bh, bw = it["y0"] - y0, it["x0"] - x0, it["bh"], it["bw"]
                    if it["k"] == 0:
                        tile[:, yy : yy + bh, xx : xx + bw] = m[yy : yy + bh, xx : xx + bw]
                    else:
                        tile[:, yy : yy + bh, xx : xx + bw] = m[yy : yy + bh, xx : xx + bw] + (
                            it["Uq"] @ it["Bq"]
                        ).reshape(tlen, bh, bw, 3)
            rec_u[:, y0 : y0 + ph, x0 : x0 + pw] = np.clip(np.round(tile), 0, 255).astype(np.uint8)
    meters = {
        "kHist": {int(a): int(b) for a, b in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "nSat": int(n_sat),
        "nPatches": len(items),
        "basisBytes": int(sum(it["bytes"] for it in items)),
        "coeffBytes": 0,
        "meanJpegBytes": int(len(mu_blob)),
        "t": tlen,
        "mu_f": mu_f,
        "nTree": n_tree,
        "nFlat": n_flat,
        "splitFrac": n_tree / max(n_tree + n_flat, 1),
    }
    return rec_u, mu_blob, items, meters


def encode_res_full(
    sl: np.ndarray, target_mse: float, k_max: int, steps: int, leaf: int = 16
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    """H2 full-frame adaptive parent + leaf leftover. Collapse if not cheaper than flat leaves."""
    tlen = sl.shape[0]
    rec_flat, mu_blob, flat_leaves, meters_f = encode_disjoint(
        sl, bw=leaf, bh=leaf, target_mse=target_mse, k_max=k_max, steps=steps
    )
    mu_f = meters_f["mu_f"]
    for it in flat_leaves:
        it["role"] = "leaf"
        it["add"] = False
        it["bytes"] = item_raw_len(it)
    flat_bytes = sum(it["bytes"] for it in flat_leaves)
    kr, u, bvol = fit_global_adaptive(sl, mu_f, k_max, target_mse)
    if kr == 0:
        log("    H2-full kRoot=0 collapse")
        meters_f["kRoot"] = 0
        meters_f["collapsed"] = True
        meters_f["rootBytes"] = 0
        meters_f["treeBytes"] = flat_bytes
        meters_f["flatBytes"] = flat_bytes
        return rec_flat, mu_blob, flat_leaves, meters_f
    uq, uc, us = qint8(u.T, axis=0)
    uq = uq.T
    bq, bc, bs = qint8(bvol.reshape(kr, -1), axis=0)
    bvol_q = bq.reshape(kr, H, W, 3)
    root = {
        "k": kr,
        "y0": 0,
        "x0": 0,
        "bw": W,
        "bh": H,
        "sz": W,
        "Uq": uq,
        "Bq": bq,
        "U": uc,
        "B": bc,
        "Us": us.reshape(kr).astype(np.float32),
        "Bs": bs.reshape(kr).astype(np.float32),
        "quantU": "int8",
        "quantB": "int8",
        "jpegB": b"",
        "role": "parent",
        "add": True,
    }
    root["bytes"] = item_raw_len(root)
    leaves: list[dict] = []
    rec_u = np.repeat(np.clip(np.round(mu_f), 0, 255).astype(np.uint8)[None], tlen, 0)
    k_hist: Counter[int] = Counter()
    n_sat = 0
    k_hist[kr] += 1
    for y0 in range(0, H, leaf):
        for x0 in range(0, W, leaf):
            p = sl[:, y0 : y0 + leaf, x0 : x0 + leaf].astype(np.float32)
            m = mu_f[y0 : y0 + leaf, x0 : x0 + leaf]
            rloc = np.einsum("tk,khwc->thwc", uq, bvol_q[:, y0 : y0 + leaf, x0 : x0 + leaf])
            it = encode_centered(p - m - rloc, y0, x0, leaf, leaf, target_mse, k_max, steps)
            it["role"] = "leaf"
            it["add"] = True
            leaves.append(it)
            k_hist[it["k"]] += 1
            if it["k"] == k_max:
                n_sat += 1
            recp = m + rloc
            if it["k"] > 0:
                recp = recp + (it["Uq"] @ it["Bq"]).reshape(tlen, leaf, leaf, 3)
            rec_u[:, y0 : y0 + leaf, x0 : x0 + leaf] = np.clip(np.round(recp), 0, 255).astype(np.uint8)
    tree_bytes = root["bytes"] + sum(it["bytes"] for it in leaves)
    collapsed = tree_bytes >= flat_bytes
    log(f"    H2-full kRoot={kr} tree={tree_bytes} flat={flat_bytes} collapsed={collapsed}")
    if collapsed:
        meters_f["kRoot"] = 0
        meters_f["collapsed"] = True
        meters_f["rootBytes"] = 0
        meters_f["treeBytes"] = tree_bytes
        meters_f["flatBytes"] = flat_bytes
        return rec_flat, mu_blob, flat_leaves, meters_f
    items = [root] + leaves
    meters = {
        "kHist": {int(a): int(b) for a, b in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "nSat": int(n_sat),
        "nPatches": len(items),
        "basisBytes": int(sum(it["bytes"] for it in items)),
        "coeffBytes": 0,
        "meanJpegBytes": int(len(mu_blob)),
        "t": tlen,
        "mu_f": mu_f,
        "kRoot": kr,
        "collapsed": False,
        "rootBytes": int(root["bytes"]),
        "treeBytes": tree_bytes,
        "flatBytes": flat_bytes,
    }
    return rec_u, mu_blob, items, meters


def already_done() -> set[str]:
    ids: set[str] = set()
    if not JSONL.exists():
        return ids
    for line in JSONL.read_text().splitlines():
        if line.strip():
            ids.add(json.loads(line)["id"])
    return ids


def score(sl: np.ndarray, rec: np.ndarray, i0: int, native: int) -> dict:
    psnrs, leftovers, ssims = [], [], []
    bridge = []
    has320 = FRAMES320.exists() and len(list(FRAMES320.glob("*.jpg"))) >= i0 + len(sl)
    src320 = load320(i0, i0 + len(sl)) if has320 else None
    grids = (2, 4, 8, 16, 32)
    seam_acc = {g: [] for g in grids}
    int_acc = {g: [] for g in grids}
    for t in range(len(sl)):
        psnrs.append(psnr(sl[t], rec[t]))
        leftovers.append(float(np.abs(sl[t][:H_DISP].astype(np.int16) - rec[t][:H_DISP].astype(np.int16)).mean()))
        ssims.append(ssim_y(sl[t], rec[t]))
        for g in grids:
            a, b, _ = seam_mae(sl[t], rec[t], g, g)
            seam_acc[g].append(a)
            int_acc[g].append(b)
        if src320 is not None:
            bridge.append(psnr(src320[t], down320(rec[t])))
    def ratio(g):
        sm, im = float(np.mean(seam_acc[g])), float(np.mean(int_acc[g]))
        return round(sm / max(im, 1e-6), 3)
    native = native if native in grids else 16
    return {
        "meanPsnr": round(float(np.mean(psnrs)), 3),
        "medianPsnr": round(float(np.median(psnrs)), 3),
        "minPsnr": round(float(np.min(psnrs)), 3),
        "leftover": round(float(np.mean(leftovers)), 3),
        "ssim": round(float(np.mean(ssims)), 4),
        "seamRatio": ratio(native),
        "seamR2": ratio(2),
        "seamR4": ratio(4),
        "seamR8": ratio(8),
        "seamR16": ratio(16),
        "seamR32": ratio(32),
        "bridgePsnr": round(float(np.mean(bridge)), 3) if bridge else None,
    }


def write_row(row: dict) -> dict:
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log(
        f"{row['id']:42s} {str(row.get('kind','')):9s} PSNR {row['meanPsnr']:6.2f}/{row['minPsnr']:5.2f}  "
        f"K={row['meanK']:.2f} sat={row['nSat']:4d}  orig={row['originBytes']/1e6:.3f}MB  "
        f"seamR={row['seamRatio']:.2f}  {row['encodeSec']:.1f}s  rss={row['rssMb']:.0f}MB"
    )
    return row


def finish_row(cfg, sl, rec, items, mu_blob, meters, t0, rss0, extra_fields=None) -> dict:
    native = int(cfg.get("patch") or cfg.get("leaf") or 16)
    if cfg.get("hop") and cfg["hop"] < native:
        native = 16
    sc = score(sl, rec, cfg["ranges"][0][0], native)
    raw_b, z_b = pack_origin(items, len(sl), extra=mu_blob)
    kh = meters.get("kHist", {})
    mean_k = float(sum(int(k) * int(n) for k, n in kh.items()) / max(sum(int(n) for n in kh.values()), 1))
    n_patch = int(meters.get("nPatches", len(items)))
    bw = int(cfg.get("patch") or 16)
    macs = mean_k * n_patch * bw * bw * 3
    ladder = {}
    mu_f = meters.get("mu_f")
    if cfg.get("decodeLadder") and mu_f is not None and items:
        peel = "h2" if cfg.get("mode") in {"res32", "resfull"} else "flat"
        if peel == "h2":
            keys = [(-1, "p"), (0, "0"), (1, "1"), (2, "2"), (4, "4"), (8, "8"), (16, "16")]
            for kp, name in keys:
                rp = recon_items(sl.shape[0], mu_f, items, None if name == "16" and kp == 16 else kp, peel="h2")
                if name == "0":
                    rp = np.clip(np.round(np.repeat(mu_f[None], sl.shape[0], 0)), 0, 255).astype(np.uint8)
                lp = [psnr(sl[t], rp[t]) for t in range(len(sl))]
                ladder[name] = {"meanPsnr": round(float(np.mean(lp)), 3), "minPsnr": round(float(np.min(lp)), 3)}
        else:
            for kp in (0, 1, 2, 4, 8, 16):
                rp = recon_items(sl.shape[0], mu_f, items, kp, peel="flat")
                lp = [psnr(sl[t], rp[t]) for t in range(len(sl))]
                ladder[str(kp)] = {"meanPsnr": round(float(np.mean(lp)), 3), "minPsnr": round(float(np.min(lp)), 3)}
    row = {
        "id": cfg["id"],
        "phase": cfg["phase"],
        "mode": cfg.get("mode", "disjoint"),
        "arm": cfg.get("arm", cfg["phase"]),
        "shot": cfg.get("shot"),
        "kind": cfg.get("kind"),
        "flux": cfg.get("flux"),
        "i0": cfg["ranges"][0][0],
        "i1": cfg["ranges"][-1][1],
        "frames": int(len(sl)),
        "trainSteps": cfg.get("trainSteps", TRAIN_STEPS),
        "kMax": cfg.get("kMax", K_MAX),
        "targetPsnr": cfg.get("targetPsnr", TARGET_PSNR),
        "patch": int(cfg.get("patch") or native),
        "hop": cfg.get("hop"),
        "blend": cfg.get("blend", 0),
        "quantU": cfg.get("quantU", "int8"),
        "quantB": cfg.get("quantB", "int8"),
        "jpegBq": cfg.get("jpegBq"),
        **sc,
        "meanK": round(mean_k, 3),
        "nSat": int(meters.get("nSat", 0)),
        "nPatches": n_patch,
        "kHist": {str(k): int(n) for k, n in sorted((int(a), int(b)) for a, b in kh.items())},
        "kRoot": int(meters.get("kRoot", 0)),
        "rootBytes": int(meters.get("rootBytes", 0)),
        "basisBytes": int(meters.get("basisBytes", 0)),
        "coeffBytes": int(meters.get("coeffBytes", 0)),
        "meanJpegBytes": int(meters.get("meanJpegBytes", len(mu_blob))),
        "rawBytes": int(raw_b),
        "originBytes": int(z_b),
        "bytesPerSec": round(z_b / max(len(sl) / FPS, 1e-6), 1),
        "vsSourceH264": round(z_b / max((len(sl) / FPS) * (SOURCE_H264_BPS / 8), 1), 3),
        "macs": round(macs, 1),
        "encodeSec": round(time.time() - t0, 2),
        "rssMb": round(max(rss0, rss_mb()), 1),
        "ladder": ladder,
        "mergeFrac": meters.get("mergeFrac"),
        "splitFrac": meters.get("splitFrac"),
        "nTree": meters.get("nTree"),
        "nFlat": meters.get("nFlat"),
        "collapsed": meters.get("collapsed"),
        "sizes": {str(k): int(v) for k, v in dict(meters.get("sizes") or {}).items()},
    }
    if extra_fields:
        row.update(extra_fields)
    return write_row(row)


def make_id(cfg: dict) -> str:
    bits = [cfg["phase"], cfg.get("shot") or "x", cfg.get("mode") or "d"]
    if cfg.get("patch"):
        bits.append(f"p{cfg['patch']}")
    if cfg.get("hop"):
        bits.append(f"h{cfg['hop']}")
    if cfg.get("blend"):
        bits.append(f"b{cfg['blend']}")
    if cfg.get("jpegBq"):
        bits.append(f"jq{cfg['jpegBq']}")
    if cfg.get("quantB") and cfg.get("quantB") != "int8":
        bits.append(f"B{cfg['quantB']}")
    if cfg.get("quantU") and cfg.get("quantU") != "int8":
        bits.append(f"U{cfg['quantU']}")
    bits.append(f"s{cfg.get('trainSteps', TRAIN_STEPS)}")
    return "-".join(str(x) for x in bits)


def run_cfg(cfg: dict, files: list[Path], done: set[str]) -> dict | None:
    cid = cfg["id"]
    if cid in done:
        log(f"skip {cid}")
        return None
    rss0 = rss_mb()
    t0 = time.time()
    a, b = cfg["ranges"][0]
    sl = load_range(files, a, b)
    target_mse = 255.0 * 255.0 / (10 ** (cfg.get("targetPsnr", TARGET_PSNR) / 10.0))
    steps = cfg.get("trainSteps", TRAIN_STEPS)
    k_max = cfg.get("kMax", K_MAX)
    mode = cfg.get("mode", "disjoint")
    if mode in {"disjoint", "blend"}:
        rec, mu_blob, items, meters = encode_disjoint(
            sl,
            bw=cfg["patch"],
            bh=cfg["patch"],
            target_mse=target_mse,
            k_max=k_max,
            steps=steps,
            quant_u=cfg.get("quantU", "int8"),
            quant_b=cfg.get("quantB", "int8"),
            jpeg_bq=cfg.get("jpegBq"),
        )
        if cfg.get("blend"):
            rec = blend_seams(rec, cfg["patch"], cfg["patch"], int(cfg["blend"]))
    elif mode == "overlap":
        rec, mu_blob, items, meters = encode_disjoint(
            sl,
            bw=cfg["patch"],
            bh=cfg["patch"],
            target_mse=target_mse,
            k_max=k_max,
            steps=steps,
            hop=cfg["hop"],
        )
    elif mode == "merge":
        rec, mu_blob, items, meters = encode_merge(sl, cfg["leaf"], target_mse, k_max, steps)
    elif mode == "res32":
        rec, mu_blob, items, meters = encode_res32(sl, target_mse, k_max, steps, leaf=cfg.get("leaf", 8), parent=32)
    elif mode == "resfull":
        rec, mu_blob, items, meters = encode_res_full(sl, target_mse, k_max, steps, leaf=cfg.get("leaf", 16))
    else:
        raise SystemExit(f"unknown mode {mode}")
    row = finish_row(cfg, sl, rec, items, mu_blob, meters, t0, rss0)
    del sl, rec
    return row


def pick_reps(shots: list[dict]) -> dict[str, dict]:
    locked = [s for s in shots if s["kind"] == "locked"]
    track = [s for s in shots if s["kind"] == "tracking"]
    L = min(locked or shots, key=lambda s: (s["flux"], -s["t"]))
    B = max(shots, key=lambda s: (s["flux"], s["t"]))
    cand = [s for s in track if s["sid"] != B["sid"]]
    T = sorted(cand, key=lambda s: s["flux"])[len(cand) // 2] if cand else L
    return {"L": L, "T": T, "B": B}


def load_rows() -> list[dict]:
    if not JSONL.exists():
        return []
    return [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]


def choose_leaf(rows: list[dict], reps: dict) -> int:
    if LEAF_CHOICE.exists():
        ch = json.loads(LEAF_CHOICE.read_text())
        log(f"leaf_choice resume leaf={ch['leaf']} ({ch['note']})")
        return int(ch["leaf"])
    def grab(sid, p):
        for r in rows:
            if r.get("phase") == "P" and r.get("shot") == sid and r.get("patch") == p and r.get("mode") == "disjoint":
                return r
        return None
    s06 = reps["B"]["sid"]
    r4, r8 = grab(s06, 4), grab(s06, 8)
    note = "default 8"
    leaf = 8
    if r4 and r8:
        dpsnr = r4["meanPsnr"] - r8["meanPsnr"]
        ratio = r4["originBytes"] / max(r8["originBytes"], 1)
        if ratio <= 1.15 and dpsnr >= 0.15:
            leaf, note = 4, f"S06 4×4 {r4['meanPsnr']:.2f}dB {r4['originBytes']/1e3:.0f}KB vs 8×8 {r8['meanPsnr']:.2f}/{r8['originBytes']/1e3:.0f} ({dpsnr:+.2f}dB, {ratio:.2f}×)"
        elif dpsnr >= -0.05 and ratio < 1.0:
            leaf, note = 4, f"S06 4×4 cheaper at similar dB ({dpsnr:+.2f}, {ratio:.2f}×)"
        else:
            leaf, note = 8, f"S06 4×4 not worth it ({dpsnr:+.2f}dB, {ratio:.2f}× bytes) → leaf 8"
    LEAF_CHOICE.write_text(json.dumps({"leaf": leaf, "note": note}, indent=2))
    log(f"LEAF CHOICE {leaf}  {note}")
    return leaf


def write_markdown(rows: list[dict], shots: list[dict], reps: dict, leaf: int) -> None:
    def fmt(r: dict) -> str:
        br = "—" if r.get("bridgePsnr") is None else f"{r['bridgePsnr']:.2f}"
        return (
            f"| `{r['id']}` | {r.get('shot','')} | {r.get('kind','')} | {r['meanPsnr']:.2f} | {r['minPsnr']:.2f} | "
            f"{r.get('ssim',0):.3f} | {r['leftover']:.2f} | {r['meanK']:.2f} | {r['nSat']} | "
            f"{r['originBytes']/1e3:.0f} | {r['seamRatio']:.2f} | {r.get('seamR8',0):.2f}/{r.get('seamR16',0):.2f}/{r.get('seamR32',0):.2f} | "
            f"{br} | {r['rssMb']:.0f} | {r['encodeSec']:.1f} |"
        )
    head = (
        "| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines = [
        "# Attempt v4.t3r — tiles, cheap seams, B-size, merge + residual trees",
        "",
        "Branch: `attempt/v4.t3r` off `attempt/v4.t2r`.  ",
        "Frozen lab: no `src/` / `reconstruct.mp4` rewrite.",
        "",
        "Raster: **640×360** crop, canvas 640×384. Same shots as t2r (`shots.json`).",
        f"ALS frozen at **TRAIN_STEPS={TRAIN_STEPS}** except a lossy-B exception if JPEG/int4 drops >0.3 dB.",
        f"H1 leaf chosen after Phase P: **{leaf}×{leaf}**. See `leaf_choice.json`.",
        "",
        "## What this pass is",
        "",
        "- **P** flat disjoint 2 (S09 only) / 4 / 8 / 16 / 32 / 64 on reps L=S09 T=S03 B=S06",
        "- **S** cheap seams: recon-only 2px/4px deblock, stored overlap hop=14 and hop=12, on S09+S06",
        "- **J** JPEG-on-B q=50/70/90, int4-B+int8-U, int4 both, on A-16 S09+S06",
        "- **H1** bottom-up exclusive merge from the P-winner leaf up to 128 then full-frame",
        "- **H2** residual 32→8 (per-32 pick tree vs flat) and full→16 (collapse if not cheaper)",
        "- **E** K′ ladders on A-16, P-winner, S hop-14, H1, H2",
        "",
        "Not C (fixed K_root=8). Not 50% OLA. Not affine. Not all 15 shots.",
        "",
        "## Shots (same as t2r)",
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
    lines += ["", f"Reps L={reps['L']['sid']} T={reps['T']['sid']} B={reps['B']['sid']}.", ""]
    phases = [
        ("P", "Phase P — tile size at native (disjoint)"),
        ("S", "Phase S — cheap seams (deblock vs stored overlap)"),
        ("J", "Phase J — JPEG-on-B / int4 mixed"),
        ("Tj", "Phase Tj — ALS unfreeze on lossy B (only if J dropped >0.3 dB)"),
        ("H1", "Phase H1 — exclusive bottom-up merge"),
        ("H2", "Phase H2 — residual 32→8 and full→16, rate-constrained"),
    ]
    by: dict[str, list] = {}
    for r in rows:
        by.setdefault(r["phase"], []).append(r)
    for ph, title in phases:
        if ph not in by:
            continue
        lines += [f"## {title}", "", head]
        for r in by[ph]:
            lines.append(fmt(r))
        lines.append("")
    lines += ["## Phase E — K′ ladders", ""]
    for r in rows:
        if r.get("ladder"):
            lines.append(f"### {r['id']}")
            lines.append("")
            lines.append("| K′ | mean dB | min dB |")
            lines.append("|---|---|---|")
            for kp, v in r["ladder"].items():
                lines.append(f"| {kp} | {v['meanPsnr']:.2f} | {v['minPsnr']:.2f} |")
            lines.append("")
    def grab(phase, shot, **kw):
        for r in rows:
            if r["phase"] != phase or r.get("shot") != shot:
                continue
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None
    lines += ["## Kill sentences", ""]
    for sid in (reps["L"]["sid"], reps["T"]["sid"], reps["B"]["sid"]):
        bits = []
        for p in (2, 4, 8, 16, 32, 64):
            r = grab("P", sid, patch=p, mode="disjoint")
            if r:
                bits.append(f"{p}={r['meanPsnr']:.2f}dB/{r['originBytes']/1e3:.0f}KB seamR={r['seamRatio']:.2f}")
        if bits:
            lines.append(f"- **{sid} tiles:** " + "; ".join(bits) + ".")
    for sid in (reps["L"]["sid"], reps["B"]["sid"]):
        a = grab("P", sid, patch=16, mode="disjoint")
        b2 = grab("S", sid, blend=2)
        b4 = grab("S", sid, blend=4)
        h14 = grab("S", sid, hop=14)
        h12 = grab("S", sid, hop=12)
        if a and b2:
            lines.append(
                f"- **{sid} deblock 2px:** seamR {b2['seamRatio']:.2f} vs A {a['seamRatio']:.2f} "
                f"origin {b2['originBytes']/1e3:.0f} vs {a['originBytes']/1e3:.0f}KB (should match A)."
            )
        if a and b4:
            lines.append(f"- **{sid} deblock 4px:** seamR {b4['seamRatio']:.2f} min {b4['minPsnr']:.2f} vs A {a['minPsnr']:.2f}.")
        if a and h14:
            lines.append(
                f"- **{sid} hop14:** seamR {h14['seamRatio']:.2f} origin {h14['originBytes']/1e3:.0f}KB "
                f"({h14['originBytes']/max(a['originBytes'],1):.2f}× A)."
            )
        if a and h12:
            lines.append(
                f"- **{sid} hop12:** seamR {h12['seamRatio']:.2f} origin {h12['originBytes']/1e3:.0f}KB "
                f"({h12['originBytes']/max(a['originBytes'],1):.2f}× A)."
            )
    for sid in (reps["L"]["sid"], reps["B"]["sid"]):
        a = grab("P", sid, patch=16, mode="disjoint")
        for r in by.get("J", []):
            if r.get("shot") != sid:
                continue
            tag = r["id"]
            if a:
                d = r["meanPsnr"] - a["meanPsnr"]
                lines.append(
                    f"- **{sid} {r.get('quantB') or 'jpeg'}{r.get('jpegBq') or ''}:** "
                    f"{r['meanPsnr']:.2f} ({d:+.2f} dB) origin {r['originBytes']/1e3:.0f}KB "
                    f"({r['originBytes']/max(a['originBytes'],1):.2f}× A)."
                )
    for r in by.get("Tj", []):
        lines.append(f"- **ALS on lossy B {r['id']}:** {r['meanPsnr']:.2f} dB steps={r['trainSteps']}.")
    for r in by.get("H1", []):
        mf = r.get("mergeFrac") or {}
        sz = r.get("sizes") or {}
        lines.append(
            f"- **H1 {r['shot']} leaf={leaf}:** {r['meanPsnr']:.2f}/{r['minPsnr']:.2f} origin {r['originBytes']/1e3:.0f}KB "
            f"n={r['nPatches']} sizes={sz} mergeFrac={mf} seamR={r['seamRatio']:.2f}."
        )
    for r in by.get("H2", []):
        lines.append(
            f"- **H2 {r['shot']} {r.get('mode')}:** {r['meanPsnr']:.2f}/{r['minPsnr']:.2f} origin {r['originBytes']/1e3:.0f}KB "
            f"splitFrac={r.get('splitFrac')} collapsed={r.get('collapsed')} kRoot={r.get('kRoot')} seamR={r['seamRatio']:.2f}."
        )
    lines += [
        "",
        "## How to reproduce",
        "",
        "```",
        "python3 encoder/v4.t3r/sweep.py",
        "```",
        "",
        "Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.",
        "",
    ]
    text = "\n".join(lines) + "\n"
    (OUT / "README.md").write_text(text)
    (ROOT / "attempts" / "v4.t3r.md").write_text(text)
    log(f"wrote {OUT/'README.md'} and attempts/v4.t3r.md")


def configs_p(reps: dict) -> list[dict]:
    out = []
    for tag, s in reps.items():
        sizes = (4, 8, 16, 32, 64)
        if tag == "L":
            sizes = (2,) + sizes
        for p in sizes:
            d = {
                "phase": "P",
                "mode": "disjoint",
                "arm": "A",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "patch": p,
                "trainSteps": TRAIN_STEPS,
                "kMax": K_MAX,
                "targetPsnr": TARGET_PSNR,
                "decodeLadder": p in (8, 16) and tag in ("L", "B"),
            }
            d["id"] = make_id(d)
            out.append(d)
    return out


def configs_s(reps: dict) -> list[dict]:
    out = []
    for tag in ("L", "B"):
        s = reps[tag]
        for blend in (2, 4):
            d = {
                "phase": "S",
                "mode": "blend",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "patch": 16,
                "blend": blend,
                "trainSteps": TRAIN_STEPS,
                "kMax": K_MAX,
                "targetPsnr": TARGET_PSNR,
                "decodeLadder": False,
            }
            d["id"] = make_id(d)
            out.append(d)
        for hop in (14, 12, 8):
            d = {
                "phase": "S",
                "mode": "overlap",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "patch": 16,
                "hop": hop,
                "trainSteps": TRAIN_STEPS,
                "kMax": K_MAX,
                "targetPsnr": TARGET_PSNR,
                "decodeLadder": hop == 8 and tag == "L",
            }
            d["id"] = make_id(d)
            out.append(d)
    return out


def configs_j(reps: dict) -> list[dict]:
    out = []
    for tag in ("L", "B"):
        s = reps[tag]
        for jq in (50, 70, 90):
            d = {
                "phase": "J",
                "mode": "disjoint",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "patch": 16,
                "jpegBq": jq,
                "quantB": "jpeg",
                "trainSteps": TRAIN_STEPS,
                "kMax": K_MAX,
                "targetPsnr": TARGET_PSNR,
            }
            d["id"] = make_id(d)
            out.append(d)
        d = {
            "phase": "J",
            "mode": "disjoint",
            "shot": s["sid"],
            "kind": s["kind"],
            "flux": s["flux"],
            "ranges": [(s["i0"], s["i1"])],
            "patch": 16,
            "quantB": "int4",
            "quantU": "int8",
            "trainSteps": TRAIN_STEPS,
            "kMax": K_MAX,
            "targetPsnr": TARGET_PSNR,
        }
        d["id"] = make_id(d)
        out.append(d)
        d = {
            "phase": "J",
            "mode": "disjoint",
            "shot": s["sid"],
            "kind": s["kind"],
            "flux": s["flux"],
            "ranges": [(s["i0"], s["i1"])],
            "patch": 16,
            "quantB": "int4",
            "quantU": "int4",
            "trainSteps": TRAIN_STEPS,
            "kMax": K_MAX,
            "targetPsnr": TARGET_PSNR,
        }
        d["id"] = make_id(d)
        out.append(d)
    return out


def configs_tj(rows: list[dict], reps: dict) -> list[dict]:
    """If a J row drops >0.3 dB vs A-16, sweep ALS 0/2/8 on the worst two."""
    out = []
    worst = []
    for tag in ("L", "B"):
        sid = reps[tag]["sid"]
        a = None
        for r in rows:
            if r["phase"] == "P" and r.get("shot") == sid and r.get("patch") == 16 and r.get("mode") == "disjoint":
                a = r
                break
        if not a:
            continue
        for r in rows:
            if r["phase"] != "J" or r.get("shot") != sid:
                continue
            drop = a["meanPsnr"] - r["meanPsnr"]
            if drop > 0.3:
                worst.append((drop, r, reps[tag]))
    worst.sort(reverse=True)
    seen = set()
    for drop, r, s in worst[:2]:
        key = (r.get("jpegBq"), r.get("quantB"), r.get("quantU"), s["sid"])
        if key in seen:
            continue
        seen.add(key)
        for steps in (0, 2, 8):
            d = {
                "phase": "Tj",
                "mode": "disjoint",
                "shot": s["sid"],
                "kind": s["kind"],
                "flux": s["flux"],
                "ranges": [(s["i0"], s["i1"])],
                "patch": 16,
                "jpegBq": r.get("jpegBq"),
                "quantB": r.get("quantB", "int8"),
                "quantU": r.get("quantU", "int8"),
                "trainSteps": steps,
                "kMax": K_MAX,
                "targetPsnr": TARGET_PSNR,
            }
            d["id"] = make_id(d)
            out.append(d)
    return out


def configs_h(reps: dict, leaf: int) -> list[dict]:
    out = []
    for tag, s in reps.items():
        d = {
            "phase": "H1",
            "mode": "merge",
            "shot": s["sid"],
            "kind": s["kind"],
            "flux": s["flux"],
            "ranges": [(s["i0"], s["i1"])],
            "leaf": leaf,
            "patch": leaf,
            "trainSteps": TRAIN_STEPS,
            "kMax": K_MAX,
            "targetPsnr": TARGET_PSNR,
            "decodeLadder": tag == "L",
        }
        d["id"] = make_id(d)
        out.append(d)
        d = {
            "phase": "H2",
            "mode": "res32",
            "shot": s["sid"],
            "kind": s["kind"],
            "flux": s["flux"],
            "ranges": [(s["i0"], s["i1"])],
            "leaf": 8,
            "patch": 8,
            "trainSteps": TRAIN_STEPS,
            "kMax": K_MAX,
            "targetPsnr": TARGET_PSNR,
            "decodeLadder": tag == "L",
        }
        d["id"] = make_id(d)
        out.append(d)
    for tag in ("L", "B"):
        s = reps[tag]
        d = {
            "phase": "H2",
            "mode": "resfull",
            "shot": s["sid"],
            "kind": s["kind"],
            "flux": s["flux"],
            "ranges": [(s["i0"], s["i1"])],
            "leaf": 16,
            "patch": 16,
            "trainSteps": TRAIN_STEPS,
            "kMax": K_MAX,
            "targetPsnr": TARGET_PSNR,
            "decodeLadder": tag == "L",
        }
        d["id"] = make_id(d)
        out.append(d)
    return out


def rss_smoke(files: list[Path], shots: list[dict]) -> None:
    s = min(shots, key=lambda x: x["t"])
    log(f"RSS smoke disjoint 16 on {s['sid']} T={s['t']} rss0={rss_mb():.0f}MB")
    sl = load_range(files, s["i0"], s["i1"])
    target_mse = 255.0 * 255.0 / (10 ** (32.5 / 10.0))
    rec, *_ = encode_disjoint(sl, bw=16, bh=16, target_mse=target_mse, k_max=K_MAX, steps=0)
    peak = rss_mb()
    log(f"RSS smoke peak {peak:.0f}MB recon {rec.shape} psnr0={psnr(sl[0], rec[0]):.2f}")
    if peak > 3500:
        raise SystemExit(f"kill: peak RSS {peak:.0f} MB > 3.5 GB")
    del sl, rec


def main() -> None:
    if not LOG.exists():
        LOG.write_text("")
    log("=== v4.t3r start ===")
    files = frame_files()
    log(f"{len(files)} frames 640×360 from {FRAMES}")
    if not SHOTS_JSON.exists():
        raise SystemExit(f"missing {SHOTS_JSON} (copy from v4.t2r)")
    shots = json.loads(SHOTS_JSON.read_text())
    log(f"{len(shots)} shots")
    for s in shots:
        log(f"  {s['sid']} [{s['i0']:4d},{s['i1']:4d}) T={s['t']:3d} flux={s['flux']:.2f} {s['kind']}")
    reps = pick_reps(shots)
    log(f"reps L={reps['L']['sid']} T={reps['T']['sid']} B={reps['B']['sid']}")
    rss_smoke(files, shots)
    done = already_done()

    def run_all(cfgs):
        nonlocal done
        for cfg in cfgs:
            run_cfg(cfg, files, done)
            done = already_done()

    log(f"Phase P  {len(configs_p(reps))} configs")
    run_all(configs_p(reps))
    leaf = choose_leaf(load_rows(), reps)
    log(f"Phase S  {len(configs_s(reps))} configs")
    run_all(configs_s(reps))
    log(f"Phase J  {len(configs_j(reps))} configs")
    run_all(configs_j(reps))
    tj = configs_tj(load_rows(), reps)
    log(f"Phase Tj {len(tj)} configs")
    run_all(tj)
    hs = configs_h(reps, leaf)
    log(f"Phase H  {len(hs)} configs leaf={leaf}")
    run_all(hs)
    rows = load_rows()
    write_markdown(rows, shots, reps, leaf)
    log(f"=== done {len(rows)} rows ===")


if __name__ == "__main__":
    main()
