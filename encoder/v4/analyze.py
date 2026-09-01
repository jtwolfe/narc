#!/usr/bin/env python3
"""narc attempt v4 — native 640×360, 8×8 temporal SVD, atlas B, sparse leftover.

v4r is frozen under public/media/v4r/. This writes the origin and lab reconstruct.
"""
from __future__ import annotations

import io
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/workspace") if Path("/workspace/public/media").exists() else Path(__file__).resolve().parents[2]
if (ROOT / "public" / "media").exists():
    MEDIA = ROOT / "public" / "media"
    LAB_JSON = ROOT / "src" / "lib" / "analysis-data.json"
else:
    MEDIA = ROOT / "media"
    LAB_JSON = ROOT / "lab" / "analysis-data.json"
TMP = Path("/tmp/bbb")
CLIP = MEDIA / "source.mp4"
FRAMES = TMP / "frames-640"
RECON = TMP / "recon-v4"
STRIP = MEDIA / "thumbs"
KEYS = MEDIA / "anchors"
HEATS = MEDIA / "v4" / "heatmaps"
ORIGIN = MEDIA / "v4"
SHOTS_JSON = ROOT / "attempts" / "v4" / "shots.json"
ORIGIN.mkdir(parents=True, exist_ok=True)

ATTEMPT = "v4"
START_SEC = 50
DURATION_SEC = 90
FPS = 24
W, H_DISP, H = 640, 360, 384
BW, BH = 8, 8
COLS, ROWS = W // BW, H // BH
N_PATCH = COLS * ROWS
MEAN_JPEG_Q = 84
ATLAS_Q = 70
LEFTOVER_Q = 40
LEFTOVER_STRIDE = 8
LEFTOVER_MAE = 3.5
TARGET_PSNR = 32.5
TARGET_MSE = 255.0 * 255.0 / (10 ** (TARGET_PSNR / 10.0))
K_MAX = 16
TRAIN_STEPS = 2
MAGIC = b"NAR4"


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


def jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> tuple[np.ndarray, bytes]:
    blob = jpeg_bytes(rgb[:H_DISP, :W], quality)
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


def shot_mu(sl: np.ndarray) -> np.ndarray:
    acc = np.zeros((H, W, 3), np.float64)
    for i in range(0, sl.shape[0], 16):
        acc += sl[i : i + 16].sum(axis=0, dtype=np.float64)
    return (acc / sl.shape[0]).astype(np.float32)


def make_atlas(bs_list: list[np.ndarray], bw: int, bh: int, quality: int) -> tuple[bytes, list[np.ndarray], np.ndarray, np.ndarray]:
    """One JPEG mosaic of every eigenpatch in the shot. Returns blob, dequant B list, mn, span."""
    ks = [int(b.shape[0]) for b in bs_list]
    total = int(sum(ks))
    d = bw * bh * 3
    if total == 0:
        return b"", [np.zeros((0, d), np.float32) for _ in bs_list], np.zeros((0,), np.float32), np.zeros((0,), np.float32)
    cols = max(1, int(math.ceil(math.sqrt(total))))
    rows = int(math.ceil(total / cols))
    canvas = np.zeros((rows * bh, cols * bw, 3), np.uint8)
    mn = np.zeros((total,), np.float32)
    sp = np.ones((total,), np.float32)
    idx = 0
    for b in bs_list:
        for i in range(b.shape[0]):
            lo = float(b[i].min())
            hi = float(b[i].max())
            span = max(hi - lo, 1e-6)
            mn[idx] = lo
            sp[idx] = span
            tile = ((b[i] - lo) / span * 255.0).reshape(bh, bw, 3)
            r, c = divmod(idx, cols)
            canvas[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw] = np.clip(np.round(tile), 0, 255).astype(np.uint8)
            idx += 1
    blob = jpeg_bytes(canvas, quality)
    rec = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
    out: list[np.ndarray] = []
    idx = 0
    for k in ks:
        row = np.zeros((k, d), np.float32)
        for i in range(k):
            r, c = divmod(idx, cols)
            tile = rec[r * bh : (r + 1) * bh, c * bw : (c + 1) * bw].astype(np.float32).reshape(-1)
            row[i] = (tile / 255.0) * sp[idx] + mn[idx]
            idx += 1
        out.append(row)
    return blob, out, mn, sp


def paint(rec: np.ndarray, mu_u: np.ndarray, items: list[dict], kprime: int | None) -> None:
    """Paint into uint8 rec. mu_u is uint8 shot mean."""
    tlen = rec.shape[0]
    mu_f = mu_u.astype(np.float32)
    for it in items:
        y0, x0 = it["y0"], it["x0"]
        k = int(it["k"])
        kuse = k if kprime is None else min(k, kprime)
        m = mu_f[y0 : y0 + BH, x0 : x0 + BW]
        if kuse <= 0:
            rec[:, y0 : y0 + BH, x0 : x0 + BW] = mu_u[y0 : y0 + BH, x0 : x0 + BW]
            continue
        extra = (it["Uq"][:, :kuse] @ it["Bq"][:kuse]).reshape(tlen, BH, BW, 3)
        rec[:, y0 : y0 + BH, x0 : x0 + BW] = np.clip(np.round(m + extra), 0, 255).astype(np.uint8)


def encode_shot(sl: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    tlen = sl.shape[0]
    mu = shot_mu(sl)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), MEAN_JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    items: list[dict] = []
    bs_list: list[np.ndarray] = []
    k_hist: Counter[int] = Counter()
    n_sat = 0
    for iy in range(ROWS):
        for ix in range(COLS):
            y0, x0 = iy * BH, ix * BW
            p = sl[:, y0 : y0 + BH, x0 : x0 + BW].astype(np.float32).reshape(tlen, -1)
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW].reshape(-1)
            xc = p - m
            k, u, b = fit_patch(xc, TARGET_MSE, K_MAX)
            if k > 0:
                uq, uc, us = qint8(u.T, axis=0)
                uq = uq.T
                bq, bc, bs = qint8(b, axis=0)
                uq, bq = train_factors(xc, uq, bq, TRAIN_STEPS)
                uq, uc, us = qint8(uq.T, axis=0)
                uq = uq.T
            else:
                uq = u
                uc = np.zeros((0, tlen), np.int8)
                us = np.zeros((0,), np.float32)
                bq = b
                bc = np.zeros((0, BW * BH * 3), np.int8)
                bs = np.zeros((0,), np.float32)
            k_hist[k] += 1
            if k == K_MAX:
                n_sat += 1
            items.append(
                {
                    "k": k,
                    "y0": y0,
                    "x0": x0,
                    "Uq": uq,
                    "U": uc,
                    "Us": us.reshape(k).astype(np.float32) if k else us,
                    "Bq": bq,
                    "B": bc,
                    "Bs": bs.reshape(k).astype(np.float32) if k else bs,
                }
            )
            bs_list.append(bq)
    atlas_blob, bq_list, bmn, bsp = make_atlas(bs_list, BW, BH, ATLAS_Q)
    del bs_list
    for it, bq in zip(items, bq_list):
        it["Bq"] = bq
    rec_svd = np.repeat(mu_u[None], tlen, 0)
    paint(rec_svd, mu_u, items, None)
    leftover_keys: list[tuple[int, bytes]] = []
    rec_u = rec_svd  # in-place leftover on the same buffer
    for t in range(0, tlen, LEFTOVER_STRIDE):
        mae = float(np.abs(sl[t][:H_DISP].astype(np.int16) - rec_u[t][:H_DISP].astype(np.int16)).mean())
        if mae < LEFTOVER_MAE:
            continue
        diff = np.clip(sl[t][:H_DISP].astype(np.int16) - rec_u[t][:H_DISP].astype(np.int16) + 128, 0, 255).astype(np.uint8)
        blob = jpeg_bytes(diff, LEFTOVER_Q)
        dec = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
        resid = dec.astype(np.int16) - 128
        rec_u[t, :H_DISP] = np.clip(rec_u[t, :H_DISP].astype(np.int16) + resid, 0, 255).astype(np.uint8)
        leftover_keys.append((t, blob))
    ladder: dict[str, dict] = {}
    # Score K′ on a stride of frames so long shots stay under RAM.
    step = 4 if tlen > 80 else 1
    idxs = list(range(0, tlen, step))
    kps = (0, 1, 2, 4, 8, 16) if tlen <= 210 else (0, 4, 16)
    for kp in kps:
        rp = np.repeat(mu_u[None], tlen, 0)
        paint(rp, mu_u, items, kp)
        lp = [psnr(sl[t], rp[t]) for t in idxs]
        ladder[str(kp)] = {"meanPsnr": round(float(np.mean(lp)), 3), "minPsnr": round(float(np.min(lp)), 3)}
        del rp
    for it in items:
        it.pop("Uq", None)
        it.pop("Bq", None)
        it.pop("B", None)
    meters = {
        "kHist": {int(a): int(b) for a, b in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "nSat": n_sat,
        "nPatches": N_PATCH,
        "meanJpegBytes": len(mu_blob),
        "atlasBytes": len(atlas_blob),
        "leftoverBytes": int(sum(4 + len(b) for _, b in leftover_keys)),
        "nLeftover": len(leftover_keys),
        "t": tlen,
        "mu_f": mu_f,
        "ladder": ladder,
        "k0": int(k_hist.get(0, 0)),
    }
    pack = {
        "items": items,
        "mu_blob": mu_blob,
        "atlas_blob": atlas_blob,
        "bmn": bmn,
        "bsp": bsp,
        "leftover": leftover_keys,
    }
    return rec_u, pack, meters


def pack_origin(shots_blob: list[dict], n_frames: int) -> tuple[bytes, bytes, bytes]:
    body = bytearray()
    body += struct.pack("<HHHIHHBB", W, H_DISP, FPS, n_frames, len(shots_blob), BW, ATLAS_Q, LEFTOVER_Q)
    for sh in shots_blob:
        i0, i1 = sh["i0"], sh["i1"]
        tlen = i1 - i0
        items = sh["items"]
        body += struct.pack("<HH", i0, i1)
        mu = sh["mu_blob"]
        body += struct.pack("<I", len(mu)) + mu
        atlas = sh["atlas_blob"]
        body += struct.pack("<I", len(atlas)) + atlas
        bmn = np.asarray(sh["bmn"], dtype=np.float32)
        bsp = np.asarray(sh["bsp"], dtype=np.float32)
        body += struct.pack("<I", bmn.size)
        body += bmn.tobytes() + bsp.tobytes()
        ks = np.array([int(it["k"]) for it in items], dtype=np.uint8)
        body += ks.tobytes()
        for it in items:
            k = int(it["k"])
            if k == 0:
                continue
            body += np.asarray(it["Us"], dtype=np.float32).tobytes()
            body += np.asarray(it["U"], dtype=np.int8).tobytes()
        left = sh["leftover"]
        body += struct.pack("<H", len(left))
        for t_off, blob in left:
            body += struct.pack("<HI", t_off, len(blob)) + blob
            _ = tlen
    raw = bytes(body)
    z = zlib.compress(raw, 9)
    return MAGIC + struct.pack("<BI", 2, len(raw)) + z, raw, z


def load_shot(files: list[Path], i0: int, i1: int) -> np.ndarray:
    sl = [pad_frame(np.array(Image.open(files[i]).convert("RGB"))) for i in range(i0, i1)]
    return np.stack(sl, 0)


def frame_files() -> list[Path]:
    pngs = sorted(FRAMES.glob("*.png"))
    if len(pngs) == FPS * DURATION_SEC:
        return pngs
    jpgs = sorted(FRAMES.glob("*.jpg"))
    if len(jpgs) == FPS * DURATION_SEC:
        return jpgs
    raise SystemExit(f"want {FPS * DURATION_SEC} frames in {FRAMES}, got png={len(pngs)} jpg={len(jpgs)}")


def load_shots() -> list[dict]:
    if not SHOTS_JSON.exists():
        raise SystemExit(f"missing {SHOTS_JSON}")
    return json.loads(SHOTS_JSON.read_text())


def self_test() -> None:
    still = np.full((8, BH, BW, 3), 80, dtype=np.uint8).astype(np.float32)
    mu = still.reshape(8, -1).mean(0)
    k, _, _ = fit_patch(still.reshape(8, -1) - mu, TARGET_MSE, K_MAX)
    if k != 0:
        raise SystemExit(f"self-test FAILED: still K={k}")
    fade = np.zeros((12, BH, BW, 3), np.float32)
    for t in range(12):
        fade[t] = 40 + 10 * t
    xc = fade.reshape(12, -1) - fade.reshape(12, -1).mean(0)
    kf, uf, bf = fit_patch(xc, TARGET_MSE, K_MAX)
    rec = fade.reshape(12, -1).mean(0) + uf @ bf
    mae = float(np.abs(rec - fade.reshape(12, -1)).mean())
    if kf < 1 or mae > 1.0:
        raise SystemExit(f"self-test FAILED: fade K={kf} mae={mae:.3f}")
    b0 = np.linspace(-0.4, 0.4, BW * BH * 3, dtype=np.float32)[None, :]
    blob, out, _, _ = make_atlas([b0], BW, BH, 90)
    err = float(np.abs(out[0] - b0).mean())
    if not blob or err > 8.0:
        raise SystemExit(f"self-test FAILED: atlas err={err:.3f}")
    print(f"self-test ok  still_k=0  fade_k={kf} fade_mae={mae:.3f}  atlas_mae={err:.3f}")


def draw_heat(src: np.ndarray, rec: np.ndarray, path: Path) -> None:
    resid = np.abs(src[:H_DISP].astype(np.int16) - rec[:H_DISP].astype(np.int16)).mean(axis=2)
    norm = np.clip(resid / 48.0, 0, 1)
    heat = np.zeros((H_DISP, W, 3), dtype=np.uint8)
    heat[..., 0] = (40 + 180 * norm).astype(np.uint8)
    heat[..., 1] = (30 + 40 * (1 - norm)).astype(np.uint8)
    heat[..., 2] = (36 + 20 * (1 - norm)).astype(np.uint8)
    blend = (0.42 * src[:H_DISP] + 0.58 * heat).astype(np.uint8)
    Image.fromarray(blend).save(path, "JPEG", quality=78)


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


def slim_stats(path: Path, attempt: str) -> dict | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    if "fps" not in s and isinstance(s.get("stats"), dict):
        s = s["stats"]
    return {
        "attempt": attempt,
        "fps": s.get("fps"),
        "frames": s.get("frames"),
        "keyframes": s.get("keyframes"),
        "residualsStored": s.get("residualsStored", 0),
        "modelBytes": s.get("modelBytes") or s.get("bitstreamBytes") or 0,
        "meanResidual": s.get("meanResidual", 0),
        "meanPsnr": s.get("meanPsnr"),
        "skipBlockFrac": s.get("skipBlockFrac"),
        "reconstructMp4Bytes": s.get("reconstructMp4Bytes", 0),
        "residualBytes": s.get("residualBytes"),
        "intraBytes": s.get("intraBytes"),
        "netBytes": s.get("netBytes"),
        "bitstreamBytes": s.get("bitstreamBytes"),
        "rawAccountedBytes": s.get("rawAccountedBytes"),
        "gzipControlBytes": s.get("gzipControlBytes"),
    }


def encode(files: list[Path], shots: list[dict]) -> dict:
    shutil.rmtree(RECON, ignore_errors=True)
    RECON.mkdir(parents=True)
    for d in (KEYS, STRIP, HEATS, ORIGIN):
        d.mkdir(parents=True, exist_ok=True)
    n = len(files)
    print(f"encoding {n} frames in {len(shots)} shots  8×8  atlas q={ATLAS_Q}  leftover q={LEFTOVER_Q}/{LEFTOVER_STRIDE}")
    rows: list[dict] = []
    packed: list[dict] = []
    k_hist: Counter[int] = Counter()
    atlas_bytes = 0
    leftover_bytes = 0
    mean_jpeg_bytes = 0
    psnrs: list[float] = []
    ladders: list[tuple[int, dict]] = []
    prev_y: np.ndarray | None = None
    prev_h: np.ndarray | None = None
    t_enc = time.time()
    leftover_frames = 0

    for s_i, s in enumerate(shots):
        i0, i1 = int(s["i0"]), int(s["i1"])
        sl = load_shot(files, i0, i1)
        rec_u, pack, meters = encode_shot(sl)
        packed.append({"i0": i0, "i1": i1, **pack})
        for k, c in meters["kHist"].items():
            k_hist[int(k)] += c
        atlas_bytes += meters["atlasBytes"]
        leftover_bytes += meters["leftoverBytes"]
        mean_jpeg_bytes += meters["meanJpegBytes"]
        leftover_set = {t for t, _ in pack["leftover"]}
        leftover_frames += len(leftover_set)
        ladders.append((i1 - i0, meters["ladder"]))
        k0 = meters["k0"]
        mk = meters["meanK"]
        print(
            f"  {s['sid']} [{i0:4d},{i1:4d}) T={i1 - i0:3d}  "
            f"PSNR~{meters['ladder']['16']['meanPsnr']:.2f}  K={mk:.2f}  "
            f"atlas={meters['atlasBytes']/1e3:.0f}KB  leftover={meters['nLeftover']} "
            f"({meters['leftoverBytes']/1e3:.0f}KB)"
        )
        for t in range(i1 - i0):
            i = i0 + t
            src = sl[t]
            rec = rec_u[t]
            y = luma(src)[:H_DISP]
            h = hist16(y)
            Image.fromarray(rec[:H_DISP]).save(RECON / f"{i:04d}.jpg", "JPEG", quality=92)
            if i % FPS == 0:
                Image.fromarray(rec[:H_DISP]).save(STRIP / f"{i:04d}.jpg", "JPEG", quality=78)
            pv = psnr(src, rec)
            psnrs.append(pv)
            flux = 0.0 if prev_y is None else float(np.abs(y.astype(np.int16) - prev_y.astype(np.int16)).mean())
            residual = float(np.abs(src[:H_DISP].astype(np.int16) - rec[:H_DISP].astype(np.int16)).mean())
            hist = 1.0 if prev_h is None else hist_corr(h, prev_h)
            cut = i == i0 and i0 != 0
            key = i == 0 or cut
            if key:
                Image.fromarray(rec[:H_DISP]).save(KEYS / f"{i:04d}.jpg", "JPEG", quality=84)
            if key or residual > 8 or i % 48 == 0:
                draw_heat(src, rec, HEATS / f"{i:04d}.jpg")
            kind = "static"
            if cut:
                kind = "cut"
            elif key:
                kind = "keyframe"
            elif residual > 10:
                kind = "residual"
            elif flux > 6:
                kind = "motion"
            rows.append(
                {
                    "i": i,
                    "t": round(i / FPS, 4),
                    "flux": round(flux, 3),
                    "motion": 0.0,
                    "residual": round(residual, 3),
                    "occlusion": 0.0,
                    "hist": round(float(hist), 4),
                    "luma": round(float(y.mean()), 2),
                    "lumaJump": round(float(y.mean() - prev_y.mean()), 2) if prev_y is not None else 0.0,
                    "dx": 0.0,
                    "dy": 0.0,
                    "scale": 1.0,
                    "rot": 0.0,
                    "kind": kind,
                    "key": key,
                    "cut": cut,
                    "storedResidual": t in leftover_set,
                    "skipBlocks": k0,
                    "residBlocks": N_PATCH - k0,
                    "intraBlocks": 0,
                    "netBlocks": 0,
                    "rankMean": round(mk, 3),
                    "psnr": round(pv, 2),
                }
            )
            prev_y, prev_h = y, h
        del sl, rec_u
        import gc
        gc.collect()

    print(f"encode {time.time() - t_enc:.1f}s  meanK={sum(k * c for k, c in k_hist.items()) / max(sum(k_hist.values()), 1):.2f}")
    origin, raw, z = pack_origin(packed, n)
    origin_path = ORIGIN / "origin.nar4"
    origin_path.write_bytes(origin)
    print(f"origin raw {len(raw)}  zlib {len(z) + 9}  file {origin_path.stat().st_size}")
    packed.clear()

    kprime: dict[str, dict] = {}
    frames_tot = sum(fr for fr, _ in ladders) or 1
    for kp in ("0", "1", "2", "4", "8", "16"):
        parts = [(fr, lad[kp]) for fr, lad in ladders if kp in lad]
        if not parts:
            continue
        tot = sum(fr for fr, _ in parts) or 1
        wmean = sum(v["meanPsnr"] * fr for fr, v in parts) / tot
        wmin = min(v["minPsnr"] for _, v in parts)
        kprime[kp] = {"meanPsnr": round(wmean, 3), "minPsnr": round(wmin, 3)}

    shot_rows = []
    for s in shots:
        i0, i1 = int(s["i0"]), int(s["i1"])
        shot_rows.append(
            {
                "i0": i0,
                "i1": i1,
                "t0": i0 / FPS,
                "t1": i1 / FPS,
                "kind": s.get("kind", "locked"),
                "keys": 1,
            }
        )

    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(RECON / "%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-an",
            "-movflags", "+faststart", str(MEDIA / "reconstruct.mp4"),
        ]
    )
    shutil.copy2(MEDIA / "reconstruct.mp4", ORIGIN / "reconstruct.mp4")

    psnr_sorted = sorted(psnrs)
    model_bytes = origin_path.stat().st_size
    duration = n / FPS
    v0 = slim_stats(MEDIA / "v0" / "stats.json", "v0-global-translation")
    v1 = slim_stats(MEDIA / "v1" / "stats.json", "v1-block-mc")
    v11 = slim_stats(MEDIA / "v1.1" / "stats.json", "v1.1-mc-correct")
    v12 = slim_stats(MEDIA / "v1.2" / "stats.json", "v1.2-affine-subpel")
    v2 = slim_stats(MEDIA / "v2" / "stats.json", "v2-residual-nets")
    v3 = slim_stats(MEDIA / "v3" / "stats.json", "v3-cu-bitstream")
    v4r = slim_stats(MEDIA / "v4r" / "stats.json", "v4r") or slim_stats(MEDIA / "stats-v4r.json", "v4r")
    stats = {
        "attempt": ATTEMPT,
        "frames": n,
        "fps": FPS,
        "width": W,
        "height": H_DISP,
        "block": [BW, BH],
        "blocksPerFrame": N_PATCH,
        "duration": duration,
        "startSec": START_SEC,
        "shots": len(shots),
        "keyframes": sum(1 for r in rows if r["key"]),
        "cuts": sum(1 for r in rows if r["cut"]),
        "residualsStored": leftover_frames,
        "sourceBytes": CLIP.stat().st_size,
        "modelBytes": model_bytes,
        "keyframeBytes": mean_jpeg_bytes,
        "residualBytes": leftover_bytes,
        "motionBytes": 0,
        "intraBytes": 0,
        "netBytes": 0,
        "bitstreamBytes": model_bytes,
        "rawAccountedBytes": len(raw) + 9,
        "gzipControlBytes": len(z),
        "syntaxBytes": len(raw),
        "syntaxZlibBytes": len(z),
        "basisBytes": atlas_bytes,
        "coeffBytes": 0,
        "meanJpegBytes": mean_jpeg_bytes,
        "atlasBytes": atlas_bytes,
        "leftoverBytes": leftover_bytes,
        "meanRank": round(sum(k * c for k, c in k_hist.items()) / max(sum(k_hist.values()), 1), 3),
        "kHist": {str(k): int(c) for k, c in sorted(k_hist.items())},
        "kMax": K_MAX,
        "targetPsnr": TARGET_PSNR,
        "trainSteps": TRAIN_STEPS,
        "atlasQ": ATLAS_Q,
        "leftoverQ": LEFTOVER_Q,
        "leftoverStride": LEFTOVER_STRIDE,
        "kPrime": kprime,
        "rawBytes": n * W * H_DISP * 3,
        "reconstructMp4Bytes": (MEDIA / "reconstruct.mp4").stat().st_size,
        "meanFlux": round(float(np.mean([f["flux"] for f in rows])), 3),
        "meanResidual": round(float(np.mean([f["residual"] for f in rows])), 3),
        "meanMotion": 0.0,
        "meanPsnr": round(float(np.mean(psnrs)), 2),
        "medianPsnr": round(float(psnr_sorted[len(psnr_sorted) // 2]), 2),
        "minPsnr": round(float(psnr_sorted[0]), 2),
        "skipBlockFrac": round(k_hist.get(0, 0) / max(sum(k_hist.values()), 1), 4),
        "ratioVsRaw": round((n * W * H_DISP * 3) / max(model_bytes, 1), 2),
        "ratioVsSource": round(CLIP.stat().st_size / max(model_bytes, 1), 2),
        "kinds": dict(Counter(r["kind"] for r in rows)),
        "blackTileFrames": int(sum(1 for f in rows if f["residual"] < 1.2 and f["luma"] < 18)),
        "baseline": v0,
        "baselineV1": v1,
        "baselineV11": v11,
        "baselineV12": v12,
        "baselineV2": v2,
        "baselineV3": v3,
        "baselineV4r": v4r,
    }
    return {
        "attempt": ATTEMPT,
        "frames": rows,
        "shots": shot_rows,
        "stats": stats,
        "source": {
            "clip": "/media/source.mp4",
            "clipAnalysis": "/media/source-320.mp4",
            "reconstruct": "/media/reconstruct.mp4",
            "reconstructV4r": "/media/v4r/reconstruct.mp4",
            "reconstructV0": "/media/v0/reconstruct.mp4",
            "reconstructV1": "/media/v1/reconstruct.mp4",
            "reconstructV11": "/media/v1.1/reconstruct.mp4",
            "reconstructV12": "/media/v1.2/reconstruct.mp4",
            "reconstructV2": "/media/v2/reconstruct.mp4",
            "reconstructV3": "/media/v3/reconstruct.mp4",
            "scope": "/media/scope.png",
            "duration": duration,
            "startSec": START_SEC,
            "title": "Big Buck Bunny",
            "credit": "Blender Foundation / peach.blender.org  ·  CC BY 3.0",
            "window": f"{START_SEC}s – {START_SEC + int(duration)}s",
        },
    }


def write_lab(data: dict) -> None:
    draw_scope(data["frames"])
    (MEDIA / "analysis.json").write_text(json.dumps(data))
    if LAB_JSON.parent.exists():
        LAB_JSON.write_text(json.dumps(data))
    (MEDIA / "stats-v4.json").write_text(json.dumps(data["stats"], indent=2))
    (ORIGIN / "stats.json").write_text(json.dumps(data["stats"], indent=2))
    (ORIGIN / "analysis.json").write_text(json.dumps(data))
    s = data["stats"]
    print(json.dumps({k: s[k] for k in (
        "attempt", "frames", "duration", "shots", "meanPsnr", "medianPsnr", "minPsnr",
        "meanResidual", "meanRank", "modelBytes", "atlasBytes", "leftoverBytes",
        "meanJpegBytes", "skipBlockFrac", "kPrime", "ratioVsSource",
    ) if k in s}, indent=2))


def main() -> None:
    self_test()
    if "--self-test" in sys.argv:
        return
    files = frame_files()
    shots = load_shots()
    print(f"{len(files)} frames 640×360  {len(shots)} shots")
    data = encode(files, shots)
    write_lab(data)
    print("wrote", MEDIA / "analysis.json")


if __name__ == "__main__":
    main()
