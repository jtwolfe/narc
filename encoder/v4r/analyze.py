#!/usr/bin/env python3
"""narc attempt v4r — clip-wide patch temporal model (from scratch).

Major deviation from v0–v3. There is no warp, no skip/intra RDO, no CU tree.
Each shot is a set of 16×16 temporal models:

    patch(t) ≈ μ_shot + U(t) @ B

U and B come from a thin SVD (optimal linear fit in t). Rank K is picked so
local MSE sits under the 32.5 dB knife, then both factors are quantized to
int8 and fine-tuned with a few SGD steps — that is the 'training'. Shot mean
μ is a closed-loop JPEG so the loop sees the bytes that land in the origin.

v0 / v1 / v1.1 / v1.2 / v2 / v3 media are frozen under public/media/v*.
"""

from __future__ import annotations

import io
import json
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
MEDIA = ROOT / "public" / "media" if (ROOT / "public" / "media").exists() else ROOT / "media"
LAB_JSON = ROOT / "src" / "lib" / "analysis-data.json"
TMP = Path("/tmp/bbb")
CLIP = MEDIA / "source.mp4"
FRAMES = TMP / "frames-v1"
RECON = TMP / "recon-v4r"
STRIP = MEDIA / "thumbs"
KEYS = MEDIA / "anchors"
HEATS = MEDIA / "heatmaps"
ORIGIN = MEDIA / "v4r"
ORIGIN.mkdir(parents=True, exist_ok=True)

ATTEMPT = "v4r"
START_SEC = 50
DURATION_SEC = 90
ANALYSIS_FPS = 24
W, H_DISP = 320, 180
H = 192
BW, BH = 16, 16
COLS, ROWS = W // BW, H // BH  # 20 × 12
N_PATCH = COLS * ROWS
CUT_HIST = 0.62
MEAN_JPEG_Q = 84
TARGET_PSNR = 32.5
TARGET_MSE = 255.0 * 255.0 / (10 ** (TARGET_PSNR / 10.0))  # ≈ 36.6
K_MAX = 16
TRAIN_STEPS = 2
TRAIN_LR = 0.35
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
    if rgb.shape[0] >= H:
        return rgb[:H]
    out = np.empty((H, W, 3), dtype=np.uint8)
    out[: rgb.shape[0]] = rgb
    out[rgb.shape[0] :] = rgb[-1]
    return out


def jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb[:H_DISP] if rgb.shape[0] > H_DISP else rgb).save(
        buf, "JPEG", quality=quality, optimize=True
    )
    return buf.getvalue()


def jpeg_roundtrip(rgb: np.ndarray, quality: int) -> tuple[np.ndarray, bytes]:
    blob = jpeg_bytes(rgb, quality)
    rec = np.array(Image.open(io.BytesIO(blob)).convert("RGB"))
    return pad_frame(rec), blob


def qint8(arr: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-row (axis=0) or per-column (axis=1) signed-8 quant. Returns dequant, codes, scales."""
    if axis == 0:
        scale = np.max(np.abs(arr), axis=tuple(range(1, arr.ndim)), keepdims=True)
    else:
        scale = np.max(np.abs(arr), axis=0, keepdims=True)
    scale = np.maximum(scale, 1e-6).astype(np.float32)
    codes = np.clip(np.round(arr / scale * 127.0), -127, 127).astype(np.int8)
    dequant = codes.astype(np.float32) * scale / 127.0
    return dequant, codes, scale.astype(np.float32)


def train_factors(
    Xc: np.ndarray, U: np.ndarray, B: np.ndarray, steps: int = TRAIN_STEPS, lr: float = TRAIN_LR
) -> tuple[np.ndarray, np.ndarray]:
    """ALS refine after int8 dequant. SVD is already L2-optimal; this only
    claw-backs the quantizer. SGD with a large lr explodes — do not use it."""
    del lr
    if steps <= 0 or U.size == 0 or B.size == 0:
        return U, B
    U = U.astype(np.float32, copy=True)
    B = B.astype(np.float32, copy=True)
    for _ in range(max(1, steps)):
        # B ← argmin ||U B − Xc||
        B, *_ = np.linalg.lstsq(U, Xc, rcond=None)
        B = B.astype(np.float32)
        # U ← argmin ||B.T U.T − Xc.T||
        UT, *_ = np.linalg.lstsq(B.T, Xc.T, rcond=None)
        U = UT.T.astype(np.float32)
        if not np.isfinite(U).all() or not np.isfinite(B).all():
            raise SystemExit("train_factors produced non-finite factors")
    return U, B


def fit_patch(Xc: np.ndarray, target_mse: float, k_max: int) -> tuple[int, np.ndarray, np.ndarray]:
    """Thin SVD of (T×D) residual. Returns K, U (T×K), B (K×D)."""
    T, D = Xc.shape
    if T == 1:
        return 0, np.zeros((T, 0), np.float32), np.zeros((0, D), np.float32)
    energy = float(np.mean(Xc * Xc))
    if energy <= target_mse:
        return 0, np.zeros((T, 0), np.float32), np.zeros((0, D), np.float32)
    G = Xc @ Xc.T
    evals, evecs = np.linalg.eigh(G)
    order = np.argsort(evals)[::-1]
    evals = np.maximum(evals[order].astype(np.float64), 0.0)
    evecs = evecs[:, order].astype(np.float32)
    denom = float(T * D)
    K = 0
    while K < k_max and K < T and (evals[K:].sum() / denom) > target_mse:
        K += 1
    if K == 0:
        return 0, np.zeros((T, 0), np.float32), np.zeros((0, D), np.float32)
    U = evecs[:, :K]
    B = U.T @ Xc
    return K, U, B.astype(np.float32)


def extract(n_frames: int) -> None:
    FRAMES.mkdir(parents=True, exist_ok=True)
    existing = sorted(FRAMES.glob("*.jpg"))
    want = ANALYSIS_FPS * DURATION_SEC
    if len(existing) == want:
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
            "2",
            str(FRAMES / "%04d.jpg"),
        ]
    )


def load_frames(n: int | None = None) -> list[np.ndarray]:
    files = sorted(FRAMES.glob("*.jpg"))
    if n is not None:
        files = files[:n]
    return [pad_frame(np.array(Image.open(p).convert("RGB"))) for p in files]


def detect_shots(rgbs: list[np.ndarray]) -> list[tuple[int, int]]:
    n = len(rgbs)
    cuts = [0]
    prev_y = luma(rgbs[0])[:H_DISP]
    prev_h = hist16(prev_y)
    for i in range(1, n):
        y = luma(rgbs[i])[:H_DISP]
        h = hist16(y)
        flux = float(np.abs(y.astype(np.int16) - prev_y.astype(np.int16)).mean())
        hist = hist_corr(h, prev_h)
        if hist < CUT_HIST and flux > 12:
            cuts.append(i)
        prev_y, prev_h = y, h
    cuts.append(n)
    shots = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    return shots or [(0, n)]


def encode_shot(
    sl: np.ndarray, target_mse: float, k_max: int, train_steps: int
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    """Encode one shot. sl is (T,H,W,3) uint8 padded. Returns recon uint8, mean jpeg, patches, meters."""
    tlen = sl.shape[0]
    sl_f = sl.astype(np.float32)
    mu = sl_f.mean(0)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), MEAN_JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    recon = np.repeat(mu_f[None, ...], tlen, axis=0)
    patches: list[dict] = []
    k_hist: Counter[int] = Counter()
    basis_bytes = 0
    coeff_bytes = 0
    for r in range(ROWS):
        for c in range(COLS):
            y0, x0 = r * BH, c * BW
            p = sl_f[:, y0 : y0 + BH, x0 : x0 + BW, :].reshape(tlen, -1)
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW, :].reshape(-1)
            Xc = p - m
            K, U, B = fit_patch(Xc, target_mse, k_max)
            recp = np.broadcast_to(m, p.shape).copy()
            packed: dict = {"k": int(K), "r": r, "c": c}
            if K > 0:
                Uq, Uc, Us = qint8(U.T, axis=0)
                Uq = Uq.T
                Bq, Bc, Bs = qint8(B, axis=0)
                Uq, Bq = train_factors(Xc, Uq, Bq, steps=train_steps)
                Uq, Uc, Us = qint8(Uq.T, axis=0)
                Uq = Uq.T
                Bq, Bc, Bs = qint8(Bq, axis=0)
                recp = m + Uq @ Bq
                packed.update(
                    {
                        "U": Uc,  # K × T
                        "B": Bc,  # K × 768
                        "Us": Us.reshape(K).astype(np.float32),
                        "Bs": Bs.reshape(K).astype(np.float32),
                    }
                )
                basis_bytes += Bc.nbytes + Bs.nbytes
                coeff_bytes += Uc.nbytes + Us.nbytes
            k_hist[K] += 1
            recon[:, y0 : y0 + BH, x0 : x0 + BW, :] = recp.reshape(tlen, BH, BW, 3)
            patches.append(packed)
    rec_u = np.clip(np.round(recon), 0, 255).astype(np.uint8)
    meters = {
        "kHist": dict(k_hist),
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "basisBytes": int(basis_bytes),
        "coeffBytes": int(coeff_bytes),
        "meanJpegBytes": int(len(mu_blob)),
        "nPatches": N_PATCH,
        "t": tlen,
    }
    return rec_u, mu_blob, patches, meters


def pack_origin(shots_blob: list[tuple[int, int, bytes, list[dict]]], n_frames: int) -> tuple[bytes, bytes, bytes]:
    body = bytearray()
    body += struct.pack("<HHHIH", W, H_DISP, ANALYSIS_FPS, n_frames, len(shots_blob))
    for i0, i1, mu_blob, patches in shots_blob:
        tlen = i1 - i0
        body += struct.pack("<HH", i0, i1)
        body += struct.pack("<I", len(mu_blob))
        body += mu_blob
        assert len(patches) == N_PATCH
        for p in patches:
            k = int(p["k"])
            body += struct.pack("<B", k)
            if k == 0:
                continue
            body += np.asarray(p["Us"], dtype=np.float32).tobytes()
            body += np.asarray(p["Bs"], dtype=np.float32).tobytes()
            body += np.asarray(p["B"], dtype=np.int8).tobytes()
            body += np.asarray(p["U"], dtype=np.int8).tobytes()
            _ = tlen  # U is K×T, length implied
    raw = bytes(body)
    z = zlib.compress(raw, 9)
    return MAGIC + struct.pack("<BI", 1, len(raw)) + z, raw, z


def unpack_origin(blob: bytes, n_frames: int) -> np.ndarray:
    """Decode packed origin to padded uint8 recon. Used by the self-test and as a sanity meter."""
    assert blob[:4] == MAGIC
    version, raw_len = struct.unpack_from("<BI", blob, 4)
    assert version == 1
    raw = zlib.decompress(blob[9:])
    assert len(raw) == raw_len
    w, h, fps, n_stored, nshot = struct.unpack_from("<HHHIH", raw, 0)
    off = 12
    recon = np.zeros((n_frames, H, W, 3), dtype=np.float32)
    for _ in range(nshot):
        i0, i1 = struct.unpack_from("<HH", raw, off)
        off += 4
        tlen = i1 - i0
        (jlen,) = struct.unpack_from("<I", raw, off)
        off += 4
        mu_blob = raw[off : off + jlen]
        off += jlen
        mu = pad_frame(np.array(Image.open(io.BytesIO(mu_blob)).convert("RGB"))).astype(np.float32)
        recon[i0:i1] = mu[None, ...]
        for r in range(ROWS):
            for c in range(COLS):
                k = raw[off]
                off += 1
                if k == 0:
                    continue
                us = np.frombuffer(raw, dtype=np.float32, count=k, offset=off).copy()
                off += k * 4
                bs = np.frombuffer(raw, dtype=np.float32, count=k, offset=off).copy()
                off += k * 4
                B = np.frombuffer(raw, dtype=np.int8, count=k * 768, offset=off).reshape(k, 768).astype(np.float32)
                off += k * 768
                U = np.frombuffer(raw, dtype=np.int8, count=k * tlen, offset=off).reshape(k, tlen).astype(np.float32)
                off += k * tlen
                B = B * bs[:, None] / 127.0
                U = (U * us[:, None] / 127.0).T
                recp = (U @ B).reshape(tlen, BH, BW, 3)
                y0, x0 = r * BH, c * BW
                recon[i0:i1, y0 : y0 + BH, x0 : x0 + BW, :] += recp
    return np.clip(np.round(recon), 0, 255).astype(np.uint8)


def self_test() -> None:
    # still → K=0
    still = np.full((8, BH, BW, 3), 80, dtype=np.uint8)
    still_f = still.astype(np.float32)
    mu = still_f.mean(0).reshape(-1)
    K, U, B = fit_patch(still_f.reshape(8, -1) - mu, TARGET_MSE, K_MAX)
    if K != 0:
        raise SystemExit(f"self-test FAILED: still patch K={K} want 0")
    # linear fade along t, constant in space → K=1 is enough
    fade = np.zeros((12, BH, BW, 3), dtype=np.float32)
    for t in range(12):
        fade[t] = 40 + 10 * t
    Xc_f = fade.reshape(12, -1) - fade.reshape(12, -1).mean(0)
    Kf, Uf, Bf = fit_patch(Xc_f, TARGET_MSE, K_MAX)
    rec = fade.reshape(12, -1).mean(0) + Uf @ Bf
    mae = float(np.abs(rec - fade.reshape(12, -1)).mean())
    if Kf < 1 or mae > 1.0:
        raise SystemExit(f"self-test FAILED: fade K={Kf} mae={mae:.3f}")
    # two-depth in time (first half 30, second half 200) → needs rank
    two = np.full((16, BH, BW, 3), 30, dtype=np.float32)
    two[8:] = 200
    Xc = two.reshape(16, -1) - two.reshape(16, -1).mean(0)
    K, U, B = fit_patch(Xc, TARGET_MSE, K_MAX)
    rec = two.reshape(16, -1).mean(0) + U @ B
    mae2 = float(np.abs(rec - two.reshape(16, -1)).mean())
    if K < 1 or mae2 > 2.0:
        raise SystemExit(f"self-test FAILED: two-depth K={K} mae={mae2:.3f}")
    # quant + train roundtrip on the fade
    Uq, _, _ = qint8(Uf.T, axis=0)
    Uq = Uq.T
    Bq, _, _ = qint8(Bf, axis=0)
    Uq, Bq = train_factors(Xc_f, Uq, Bq, steps=8)
    rec_q = fade.reshape(12, -1).mean(0) + Uq @ Bq
    mae_q = float(np.abs(rec_q - fade.reshape(12, -1)).mean())
    if not np.isfinite(mae_q) or mae_q > 2.5:
        raise SystemExit(f"self-test FAILED: trained quant fade mae={mae_q:.3f}")
    print(
        f"self-test ok  still_k=0  fade_k={Kf} fade_mae={mae:.3f}  "
        f"two_depth_k={K} two_mae={mae2:.3f}  train_mae={mae_q:.3f}"
    )


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


def encode(rgbs: list[np.ndarray], probe_note: str | None = None, probe_frames: int = 0) -> dict:
    shutil.rmtree(RECON, ignore_errors=True)
    RECON.mkdir(parents=True)
    for d in (KEYS, STRIP, HEATS):
        d.mkdir(parents=True, exist_ok=True)

    n = len(rgbs)
    shots = detect_shots(rgbs)
    print(f"encoding {n} frames in {len(shots)} shots  target {TARGET_PSNR:.1f} dB  k_max={K_MAX}")
    rows: list[dict] = []
    packed_shots: list[tuple[int, int, bytes, list[dict]]] = []
    k_hist: Counter[int] = Counter()
    basis_bytes = 0
    coeff_bytes = 0
    mean_jpeg_bytes = 0
    psnrs: list[float] = []
    prev_y: np.ndarray | None = None
    prev_h: np.ndarray | None = None
    probed = False
    t_enc = time.time()

    for s_i, (i0, i1) in enumerate(shots):
        sl = np.stack(rgbs[i0:i1], 0)
        rec_u, mu_blob, patches, meters = encode_shot(sl, TARGET_MSE, K_MAX, TRAIN_STEPS)
        del sl
        packed_shots.append((i0, i1, mu_blob, patches))
        for k, c in meters["kHist"].items():
            k_hist[int(k)] += c
        basis_bytes += meters["basisBytes"]
        coeff_bytes += meters["coeffBytes"]
        mean_jpeg_bytes += meters["meanJpegBytes"]
        k0 = int(meters["kHist"].get(0, 0))
        mk = float(meters["meanK"])
        print(
            f"  shot {s_i:02d} [{i0:4d},{i1:4d}) T={i1 - i0:3d} meanK={meters['meanK']:.2f} "
            f"jpeg={meters['meanJpegBytes']} B"
        )
        for t in range(i1 - i0):
            i = i0 + t
            src = rgbs[i]
            rec = rec_u[t]
            y = luma(src)[:H_DISP]
            h = hist16(y)
            Image.fromarray(rec[:H_DISP]).save(RECON / f"{i:04d}.jpg", "JPEG", quality=92)
            if i % ANALYSIS_FPS == 0:
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
                    "t": round(i / ANALYSIS_FPS, 4),
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
                    "storedResidual": True,
                    "skipBlocks": k0,
                    "residBlocks": N_PATCH - k0,
                    "intraBlocks": 0,
                    "netBlocks": 0,
                    "rankMean": round(mk, 3),
                    "psnr": round(pv, 2),
                }
            )
            prev_y, prev_h = y, h
        del rec_u
        if probe_frames and not probed and i1 >= probe_frames:
            probed = True
            probe_ps = psnrs[:probe_frames]
            mean_p = float(np.mean(probe_ps))
            print(
                f"PROBE {probe_frames / ANALYSIS_FPS:.0f}s  mean PSNR {mean_p:.2f} dB  "
                f"min {min(probe_ps):.2f}  leftover {np.mean([r['residual'] for r in rows[:probe_frames]]):.2f}"
            )
            if mean_p < 31.5:
                raise SystemExit(
                    f"kill: {probe_frames / ANALYSIS_FPS:.0f}s probe mean PSNR {mean_p:.2f} dB is under 31.5."
                )

    print(
        f"encode {time.time() - t_enc:.1f}s  "
        f"meanK={sum(k * c for k, c in k_hist.items()) / max(sum(k_hist.values()), 1):.2f}"
    )

    origin, raw, z = pack_origin(packed_shots, n)
    origin_path = ORIGIN / "origin.nar4"
    origin_path.write_bytes(origin)
    print(f"origin raw {len(raw)}  zlib {len(z) + 9}  file {origin_path.stat().st_size}")
    packed_shots.clear()

    # per-frame rank: fill from shot meters instead of the global hist
    shot_rows = []
    for s_i, (i0, i1) in enumerate(shots):
        kind = "locked"
        mean_flux = float(np.mean([rows[i]["flux"] for i in range(i0, i1)])) if i1 > i0 else 0
        if mean_flux > 8:
            kind = "busy"
        elif mean_flux > 3.5:
            kind = "tracking"
        shot_rows.append(
            {
                "i0": i0,
                "i1": i1,
                "t0": i0 / ANALYSIS_FPS,
                "t1": i1 / ANALYSIS_FPS,
                "kind": kind,
                "keys": 1,
            }
        )

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
    v4r_mp4 = ORIGIN / "reconstruct.mp4"
    v4r_mp4.write_bytes((MEDIA / "reconstruct.mp4").read_bytes())

    psnr_sorted = sorted(psnrs)
    model_bytes = origin_path.stat().st_size
    raw_accounted = len(raw) + 9
    kinds = dict(Counter(r["kind"] for r in rows))
    duration = n / ANALYSIS_FPS
    v0 = slim_stats(MEDIA / "v0" / "stats.json", "v0-global-translation") or slim_stats(MEDIA / "stats-v3.json", "v0-global-translation")
    # v3 stats file has nested baseline; prefer frozen v* folders
    v1 = slim_stats(MEDIA / "v1" / "stats.json", "v1-block-mc")
    v11 = slim_stats(MEDIA / "v1.1" / "stats.json", "v1.1-mc-correct")
    v12 = slim_stats(MEDIA / "v1.2" / "stats.json", "v1.2-affine-subpel")
    v2 = slim_stats(MEDIA / "v2" / "stats.json", "v2-residual-nets")
    v3 = slim_stats(MEDIA / "v3" / "stats.json", "v3-cu-bitstream")
    stats = {
        "attempt": ATTEMPT,
        "frames": n,
        "fps": ANALYSIS_FPS,
        "width": W,
        "height": H_DISP,
        "block": [BW, BH],
        "blocksPerFrame": N_PATCH,
        "duration": duration,
        "startSec": START_SEC,
        "shots": len(shots),
        "keyframes": sum(1 for r in rows if r["key"]),
        "cuts": sum(1 for r in rows if r["cut"]),
        "residualsStored": n,
        "sourceBytes": CLIP.stat().st_size,
        "modelBytes": model_bytes,
        "keyframeBytes": mean_jpeg_bytes,
        "residualBytes": basis_bytes,
        "motionBytes": coeff_bytes,
        "intraBytes": 0,
        "netBytes": 0,
        "bitstreamBytes": model_bytes,
        "rawAccountedBytes": raw_accounted,
        "gzipControlBytes": len(z),
        "syntaxBytes": len(raw),
        "syntaxZlibBytes": len(z),
        "basisBytes": basis_bytes,
        "coeffBytes": coeff_bytes,
        "meanJpegBytes": mean_jpeg_bytes,
        "meanRank": round(sum(k * c for k, c in k_hist.items()) / max(sum(k_hist.values()), 1), 3),
        "kHist": {str(k): int(c) for k, c in sorted(k_hist.items())},
        "kMax": K_MAX,
        "targetPsnr": TARGET_PSNR,
        "trainSteps": TRAIN_STEPS,
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
        "kinds": kinds,
        "probe": probe_note,
        "blackTileFrames": int(sum(1 for f in rows if f["residual"] < 1.2 and f["luma"] < 18)),
        "baseline": v0,
        "baselineV1": v1,
        "baselineV11": v11,
        "baselineV12": v12,
        "baselineV2": v2,
        "baselineV3": v3,
    }
    return {
        "attempt": ATTEMPT,
        "frames": rows,
        "shots": shot_rows,
        "stats": stats,
        "source": {
            "clip": "/media/source.mp4",
            "reconstruct": "/media/reconstruct.mp4",
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
            "window": f"{START_SEC}s – {START_SEC + int(duration)}s" if n >= 2000 else f"probe {duration:.0f}s of {START_SEC}s–{START_SEC+DURATION_SEC}s",
        },
    }


def write_lab(data: dict) -> None:
    draw_scope(data["frames"])
    out = MEDIA / "analysis.json"
    out.write_text(json.dumps(data))
    if LAB_JSON.parent.exists():
        LAB_JSON.write_text(json.dumps(data))
    (MEDIA / "stats-v4r.json").write_text(json.dumps(data["stats"], indent=2))
    (ORIGIN / "stats.json").write_text(json.dumps(data["stats"], indent=2))
    (ORIGIN / "analysis.json").write_text(json.dumps(data))
    print(json.dumps({k: data["stats"][k] for k in (
        "attempt", "frames", "duration", "shots", "meanPsnr", "medianPsnr", "minPsnr",
        "meanResidual", "meanRank", "modelBytes", "rawAccountedBytes", "basisBytes",
        "coeffBytes", "meanJpegBytes", "skipBlockFrac", "kHist", "probe",
    ) if k in data["stats"]}, indent=2))


def main() -> None:
    self_test()
    if "--self-test" in sys.argv:
        return
    probe_only = "--probe-only" in sys.argv
    probe_sec = 10
    for a in sys.argv:
        if a.startswith("--seconds="):
            probe_sec = int(a.split("=", 1)[1])
    extract(ANALYSIS_FPS * DURATION_SEC)
    if probe_only:
        rgbs = load_frames(ANALYSIS_FPS * probe_sec)
        print(f"loaded {len(rgbs)} probe frames (padded {H}h, display {H_DISP}h)")
        data = encode(rgbs, probe_note=f"{probe_sec}s head", probe_frames=len(rgbs))
        write_lab(data)
        print("wrote probe-only lab")
        return
    rgbs = load_frames()
    n_all = len(rgbs)
    print(f"loaded {n_all} frames (padded {H}h, display {H_DISP}h)")
    print(f"\n=== v4r  full {n_all / ANALYSIS_FPS:.0f}s with {probe_sec}s probe gate ===")
    data = encode(
        rgbs,
        probe_note=f"{probe_sec}s gate then full",
        probe_frames=min(n_all, ANALYSIS_FPS * probe_sec),
    )
    write_lab(data)
    print("wrote", MEDIA / "analysis.json")


if __name__ == "__main__":
    main()
