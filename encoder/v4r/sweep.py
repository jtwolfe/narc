#!/usr/bin/env python3
"""v4.t1r knob sweep. Does not write lab UI / reconstruct.mp4 / analysis.json.

Reads cached 320×180 frames from /tmp/bbb/frames-v1. Records JSONL + a summary
markdown under attempts/v4.t1r/. Reuses encoder/v4r/analyze.py math.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from encoder.v4r.analyze import (  # noqa: E402
    ANALYSIS_FPS,
    FRAMES,
    H,
    H_DISP,
    W,
    fit_patch,
    hist16,
    hist_corr,
    jpeg_roundtrip,
    luma,
    pad_frame,
    psnr,
    qint8,
    train_factors,
)

OUT = Path(__file__).resolve().parents[2] / "attempts" / "v4.t1r"
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "results.jsonl"
LOG = OUT / "sweep.log"

WINDOWS = {
    "W0": (0, 10, "head (source 50–60s)"),
    "W1": (30, 40, "mid (source 80–90s)"),
    "W2": (70, 80, "late (source 120–130s)"),
}


def log(msg: str) -> None:
    line = msg if msg.endswith("\n") else msg + "\n"
    sys.stdout.write(line)
    sys.stdout.flush()
    with LOG.open("a") as f:
        f.write(line)


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
    """Nibble-pack signed codes in -7..7. Size-only; not a decoder."""
    flat = np.ascontiguousarray(codes, dtype=np.int8).ravel()
    n = int(flat.size)
    if n % 2:
        flat = np.concatenate([flat, np.zeros(1, dtype=np.int8)])
    a = flat.view(np.uint8) & np.uint8(0x0F)
    return (a[0::2] | (a[1::2] << 4)).tobytes()


def packed_int4_bytes(codes: np.ndarray) -> int:
    return (codes.size + 1) // 2


def detect_shots(rgbs: list[np.ndarray], cut_hist: float) -> list[tuple[int, int]]:
    n = len(rgbs)
    cuts = [0]
    prev_y = luma(rgbs[0])[:H_DISP]
    prev_h = hist16(prev_y)
    for i in range(1, n):
        y = luma(rgbs[i])[:H_DISP]
        h = hist16(y)
        flux = float(np.abs(y.astype(np.int16) - prev_y.astype(np.int16)).mean())
        hist = hist_corr(h, prev_h)
        if hist < cut_hist and flux > 12:
            cuts.append(i)
        prev_y, prev_h = y, h
    cuts.append(n)
    shots = [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1) if cuts[i + 1] > cuts[i]]
    return shots or [(0, n)]


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


def encode_shot_p(
    sl: np.ndarray,
    *,
    target_mse: float,
    k_max: int,
    train_steps: int,
    bw: int,
    bh: int,
    jpeg_q: int,
    quant: str,
) -> tuple[np.ndarray, bytes, list[dict], dict]:
    tlen, hh, ww, _ = sl.shape
    rows, cols = hh // bh, ww // bw
    sl_f = sl.astype(np.float32)
    mu = sl_f.mean(0)
    mu_u, mu_blob = jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), jpeg_q)
    mu_f = mu_u.astype(np.float32)
    recon = np.repeat(mu_f[None, ...], tlen, axis=0)
    patches: list[dict] = []
    k_hist: Counter[int] = Counter()
    basis_bytes = 0
    coeff_bytes = 0
    qfn = {"int8": qint8, "int4": qint4}.get(quant)
    for r in range(rows):
        for c in range(cols):
            y0, x0 = r * bh, c * bw
            p = sl_f[:, y0 : y0 + bh, x0 : x0 + bw, :].reshape(tlen, -1)
            m = mu_f[y0 : y0 + bh, x0 : x0 + bw, :].reshape(-1)
            Xc = p - m
            K, U, B = fit_patch(Xc, target_mse, k_max)
            recp = np.broadcast_to(m, p.shape).copy()
            packed: dict = {"k": int(K), "r": r, "c": c}
            if K > 0:
                if quant == "float32":
                    recp = m + U @ B
                    packed.update({"U": U.astype(np.float32), "B": B.astype(np.float32)})
                    basis_bytes += packed["B"].nbytes
                    coeff_bytes += packed["U"].nbytes
                else:
                    Uq, Uc, Us = qfn(U.T, axis=0)
                    Uq = Uq.T
                    Bq, Bc, Bs = qfn(B, axis=0)
                    if train_steps > 0:
                        Uq, Bq = train_factors(Xc, Uq, Bq, steps=train_steps)
                        Uq, Uc, Us = qfn(Uq.T, axis=0)
                        Uq = Uq.T
                        Bq, Bc, Bs = qfn(Bq, axis=0)
                    recp = m + Uq @ Bq
                    packed.update(
                        {
                            "U": Uc,
                            "B": Bc,
                            "Us": Us.reshape(K).astype(np.float32),
                            "Bs": Bs.reshape(K).astype(np.float32),
                            "Uq": Uq,
                            "Bq": Bq,
                        }
                    )
                    if quant == "int4":
                        basis_bytes += packed_int4_bytes(Bc) + Bs.nbytes
                        coeff_bytes += packed_int4_bytes(Uc) + Us.nbytes
                    else:
                        basis_bytes += Bc.nbytes + Bs.nbytes
                        coeff_bytes += Uc.nbytes + Us.nbytes
            k_hist[K] += 1
            recon[:, y0 : y0 + bh, x0 : x0 + bw, :] = recp.reshape(tlen, bh, bw, 3)
            patches.append(packed)
    rec_u = np.clip(np.round(recon), 0, 255).astype(np.uint8)
    n_patch = rows * cols
    meters = {
        "kHist": {int(k): int(n) for k, n in k_hist.items()},
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "basisBytes": int(basis_bytes),
        "coeffBytes": int(coeff_bytes),
        "meanJpegBytes": int(len(mu_blob)),
        "nPatches": n_patch,
        "t": tlen,
        "nSat": int(k_hist.get(k_max, 0)),
    }
    return rec_u, mu_blob, patches, meters


def pack_size(shots: list[tuple[bytes, list[dict], int]], n_frames: int, quant: str) -> tuple[int, int]:
    """NAR4-like layout so originBytes is comparable to the shipped encoder.

    Header + per-shot (i0,i1,jpeg) + per-patch (K, scales, codes). int4 nibble-packs
    the codes; float32 stores U,B with no scales.
    """
    body = bytearray()
    body += struct.pack("<HHHIH", W, H_DISP, ANALYSIS_FPS, n_frames, len(shots))
    for mu_blob, patches, tlen in shots:
        body += struct.pack("<HH", 0, tlen)
        body += struct.pack("<I", len(mu_blob))
        body += mu_blob
        for p in patches:
            k = int(p["k"])
            body += struct.pack("<B", k)
            if k == 0:
                continue
            if quant == "float32":
                body += np.asarray(p["B"], dtype=np.float32).tobytes()
                body += np.asarray(p["U"], dtype=np.float32).tobytes()
            elif quant == "int4":
                body += np.asarray(p["Us"], dtype=np.float32).tobytes()
                body += np.asarray(p["Bs"], dtype=np.float32).tobytes()
                body += pack_int4_blob(p["B"])
                body += pack_int4_blob(p["U"])
            else:
                body += np.asarray(p["Us"], dtype=np.float32).tobytes()
                body += np.asarray(p["Bs"], dtype=np.float32).tobytes()
                body += np.asarray(p["B"], dtype=np.int8).tobytes()
                body += np.asarray(p["U"], dtype=np.int8).tobytes()
    raw = bytes(body)
    z = zlib.compress(raw, 9)
    return int(len(raw)), int(len(z) + 9)


def recon_at_kprime(sl: np.ndarray, mu_f: np.ndarray, patches: list[dict], bw: int, bh: int, kprime: int | None) -> np.ndarray:
    tlen = sl.shape[0]
    recon = np.repeat(mu_f[None, ...], tlen, axis=0)
    for p in patches:
        k = int(p["k"])
        if k == 0:
            continue
        kuse = k if kprime is None else min(k, kprime)
        if kuse <= 0:
            continue
        r, c = p["r"], p["c"]
        y0, x0 = r * bh, c * bw
        m = mu_f[y0 : y0 + bh, x0 : x0 + bw, :].reshape(-1)
        if "Uq" in p:
            U = p["Uq"][:, :kuse]
            B = p["Bq"][:kuse]
        else:
            U = p["U"][:, :kuse] if p["U"].ndim == 2 and p["U"].shape[0] == tlen else p["U"][:kuse].T
            B = p["B"][:kuse]
        recp = m + U @ B
        recon[:, y0 : y0 + bh, x0 : x0 + bw, :] = recp.reshape(tlen, bh, bw, 3)
    return np.clip(np.round(recon), 0, 255).astype(np.uint8)


def window_flux(rgbs: list[np.ndarray]) -> float:
    if len(rgbs) < 2:
        return 0.0
    acc = 0.0
    prev = luma(rgbs[0])[:H_DISP]
    for fr in rgbs[1:]:
        y = luma(fr)[:H_DISP]
        acc += float(np.abs(y.astype(np.int16) - prev.astype(np.int16)).mean())
        prev = y
    return acc / (len(rgbs) - 1)


def label_flux(f: float) -> str:
    if f > 8:
        return "busy"
    if f > 3.5:
        return "tracking"
    return "locked"


def already_done() -> set[str]:
    ids: set[str] = set()
    if not JSONL.exists():
        return ids
    for line in JSONL.read_text().splitlines():
        if not line.strip():
            continue
        ids.add(json.loads(line)["id"])
    return ids


def run_one(cfg: dict, frames: list[np.ndarray], done: set[str]) -> dict | None:
    cid = cfg["id"]
    if cid in done:
        log(f"skip {cid} (already recorded)")
        return None
    t0, t1 = cfg["t0"], cfg["t1"]
    i0, i1 = int(t0 * ANALYSIS_FPS), int(t1 * ANALYSIS_FPS)
    sl_list = frames[i0:i1]
    bw, bh = cfg["bw"], cfg["bh"]
    target_psnr = cfg["targetPsnr"]
    target_mse = 255.0 * 255.0 / (10 ** (target_psnr / 10.0))
    n_patch = (H // bh) * (W // bw)
    t_enc = time.time()
    shots = detect_shots(sl_list, cfg["cutHist"])
    rec_all = []
    packed_shots = []
    k_hist: Counter[int] = Counter()
    basis_bytes = coeff_bytes = mean_jpeg = 0
    n_sat = 0
    for s0, s1 in shots:
        sl = np.stack(sl_list[s0:s1], 0)
        rec_u, mu_blob, patches, meters = encode_shot_p(
            sl,
            target_mse=target_mse,
            k_max=cfg["kMax"],
            train_steps=cfg["trainSteps"],
            bw=bw,
            bh=bh,
            jpeg_q=cfg["jpegQ"],
            quant=cfg["quant"],
        )
        rec_all.append(rec_u)
        packed_shots.append((mu_blob, patches, s1 - s0))
        for k, n in meters["kHist"].items():
            k_hist[int(k)] += n
        basis_bytes += meters["basisBytes"]
        coeff_bytes += meters["coeffBytes"]
        mean_jpeg += meters["meanJpegBytes"]
        n_sat += meters["nSat"]
        del sl
    rec = np.concatenate(rec_all, 0)
    enc_s = time.time() - t_enc
    psnrs = []
    leftovers = []
    seam_s = []
    seam_i = []
    for t, src in enumerate(sl_list):
        pv = psnr(src, rec[t])
        psnrs.append(pv)
        leftovers.append(float(np.abs(src[:H_DISP].astype(np.int16) - rec[t][:H_DISP].astype(np.int16)).mean()))
        sm, im, _ = seam_mae(src, rec[t], bw, bh)
        seam_s.append(sm)
        seam_i.append(im)
    raw_b, z_b = pack_size(packed_shots, len(sl_list), cfg["quant"])
    mean_k = float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1))
    macs = mean_k * n_patch * bw * bh * 3
    flux = window_flux(sl_list)

    ladder = {}
    if cfg.get("decodeLadder"):
        for kp in (0, 1, 2, 4, 8, 16):
            rec_p = []
            for (s0, s1), (mu_blob, patches, tlen) in zip(shots, packed_shots):
                sl = np.stack(sl_list[s0:s1], 0)
                mu_u = pad_frame(np.array(Image.open(io.BytesIO(mu_blob)).convert("RGB"))).astype(np.float32)
                rec_p.append(recon_at_kprime(sl, mu_u, patches, bw, bh, kp))
            rec_p = np.concatenate(rec_p, 0)
            lp = [psnr(src, rec_p[t]) for t, src in enumerate(sl_list)]
            k_cap = min(kp, max(k_hist) if k_hist else 0)
            ladder[str(kp)] = {
                "meanPsnr": round(float(np.mean(lp)), 3),
                "minPsnr": round(float(np.min(lp)), 3),
                "macs": round(k_cap * n_patch * bw * bh * 3, 1),
            }

    row = {
        "id": cid,
        "phase": cfg["phase"],
        "window": cfg["window"],
        "t0": t0,
        "t1": t1,
        "sourceSec": [50 + t0, 50 + t1],
        "kind": label_flux(flux),
        "flux": round(flux, 3),
        "trainSteps": cfg["trainSteps"],
        "kMax": cfg["kMax"],
        "targetPsnr": target_psnr,
        "quant": cfg["quant"],
        "bw": bw,
        "bh": bh,
        "cutHist": cfg["cutHist"],
        "jpegQ": cfg["jpegQ"],
        "frames": len(sl_list),
        "shots": len(shots),
        "nPatches": n_patch,
        "meanPsnr": round(float(np.mean(psnrs)), 3),
        "medianPsnr": round(float(np.median(psnrs)), 3),
        "minPsnr": round(float(np.min(psnrs)), 3),
        "leftover": round(float(np.mean(leftovers)), 3),
        "seamMae": round(float(np.mean(seam_s)), 3),
        "interiorMae": round(float(np.mean(seam_i)), 3),
        "seamRatio": round(float(np.mean(seam_s) / max(np.mean(seam_i), 1e-6)), 3),
        "meanK": round(mean_k, 3),
        "nSat": int(n_sat),
        "kHist": {str(k): int(n) for k, n in sorted(k_hist.items())},
        "basisBytes": int(basis_bytes),
        "coeffBytes": int(coeff_bytes),
        "meanJpegBytes": int(mean_jpeg),
        "rawBytes": int(raw_b),
        "originBytes": int(z_b),
        "bytesPerSec": round(z_b / max(cfg["t1"] - cfg["t0"], 1e-6), 1),
        "macs": round(macs, 1),
        "encodeSec": round(enc_s, 2),
        "ladder": ladder,
    }
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    log(
        f"{cid:18s}  {row['kind']:9s}  PSNR {row['meanPsnr']:6.2f}/{row['minPsnr']:5.2f}  "
        f"K={row['meanK']:.2f} sat={row['nSat']:4d}  origin={row['originBytes']/1e6:.3f}MB  "
        f"seamR={row['seamRatio']:.2f}  {row['encodeSec']:.1f}s"
    )
    return row


def configs() -> list[dict]:
    out: list[dict] = []

    def add(phase, window, t0, t1, **kw):
        d = {
            "phase": phase,
            "window": window,
            "t0": t0,
            "t1": t1,
            "trainSteps": 2,
            "kMax": 16,
            "targetPsnr": 32.5,
            "quant": "int8",
            "bw": 16,
            "bh": 16,
            "cutHist": 0.62,
            "jpegQ": 84,
            "decodeLadder": False,
        }
        d.update(kw)
        bits = [
            phase,
            window,
            f"s{d['trainSteps']}",
            f"k{d['kMax']}",
            f"t{d['targetPsnr']}",
            d["quant"],
            f"p{d['bw']}",
            f"c{d['cutHist']}",
        ]
        d["id"] = "-".join(str(x) for x in bits)
        out.append(d)

    for w, (t0, t1, _) in WINDOWS.items():
        add("A", w, t0, t1, decodeLadder=True)
    for steps in (0, 8, 32):
        for w, (t0, t1, _) in WINDOWS.items():
            add("B", w, t0, t1, trainSteps=steps)
    for w, (t0, t1, _) in WINDOWS.items():
        add("B", w, t0, t1, quant="float32", trainSteps=0)
    for kmax in (8, 32):
        for w, (t0, t1, _) in WINDOWS.items():
            add("C", w, t0, t1, kMax=kmax)
    for tgt in (30.0, 35.0):
        for w, (t0, t1, _) in WINDOWS.items():
            add("C", w, t0, t1, targetPsnr=tgt)
    for w, (t0, t1, _) in WINDOWS.items():
        add("C", w, t0, t1, quant="int4")
    for ps in (8, 32):
        for w, (t0, t1, _) in WINDOWS.items():
            add("D", w, t0, t1, bw=ps, bh=ps)
    for ch in (0.40, 0.80):
        for w, (t0, t1, _) in WINDOWS.items():
            add("D", w, t0, t1, cutHist=ch)
    return out


def load_all_frames() -> list[np.ndarray]:
    files = sorted(FRAMES.glob("*.jpg"))
    log(f"loading {len(files)} cached frames from {FRAMES}")
    t0 = time.time()
    rgbs = [pad_frame(np.array(Image.open(p).convert("RGB"))) for p in files]
    log(f"loaded {len(rgbs)} in {time.time() - t0:.1f}s")
    return rgbs


def write_markdown(rows: list[dict], wall_s: float | None = None) -> None:
    phases = {}
    for r in rows:
        phases.setdefault(r["phase"], []).append(r)

    def fmt(r: dict) -> str:
        return (
            f"| `{r['id']}` | {r['window']} | {r['kind']} | {r['meanPsnr']:.2f} | {r['minPsnr']:.2f} | "
            f"{r['leftover']:.2f} | {r['meanK']:.2f} | {r['nSat']} | {r['originBytes']/1e3:.1f} | "
            f"{r['seamRatio']:.2f} | {r['encodeSec']:.1f} |"
        )

    head = (
        "| id | win | kind | mean dB | min dB | leftover | mean K | sat | origin KB | seamR | s |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = [
        "# Attempt v4.t1r — knob sweep on v4r (no UI change)",
        "",
        "Branch: `attempt/v4.t1r`  ",
        "Frozen: `attempt/v4r` encoder math. Lab UI / `public/media/reconstruct.mp4` not rewritten.",
        "",
        "Campaign is one-factor-at-a-time on three 10s windows of the 90s BBB probe.",
        "Harness: `encoder/v4r/sweep.py`. Raw rows: [`results.jsonl`](v4.t1r/results.jsonl).",
        "",
        "## Process",
        "",
        "- Goal: measure training / K / target / quant / patch / cuts **before** adding affine or shared-U.",
        "- Live lab stays on v4r media. This branch does not touch `src/` or rewrite reconstruct.",
        "- Frames: cached `/tmp/bbb/frames-v1` (2160 JPEG @ 320×180 / 24 fps, source 50–140s).",
        "- Shipped point: steps=2, K_MAX=16, target=32.5 dB, int8, 16×16, cutHist=0.62, JPEG q=84.",
        "- Sanity: W0 shipped should land near the v4r 10s gate (**35.16 dB**).",
        "- Resume: re-running `python3 encoder/v4r/sweep.py` skips ids already in `results.jsonl`.",
        "- Phase F (overlap / warp-then-SVD / shared U / JPEG-on-B) is **not run** from this harness.",
    ]
    if wall_s is not None:
        lines.append(f"- Wall time: **{wall_s/60:.1f} min** ({len(rows)} rows).")
    lines += [
        "",
        "## Windows",
        "",
        "| win | analysis | source | flux | kind |",
        "|---|---|---|---|---|",
    ]
    for w, (t0, t1, note) in WINDOWS.items():
        a = next((r for r in rows if r["phase"] == "A" and r["window"] == w), None)
        if a:
            lines.append(f"| {w} | {t0}–{t1}s | {50+t0}–{50+t1}s · {note} | {a['flux']:.2f} | **{a['kind']}** |")
        else:
            lines.append(f"| {w} | {t0}–{t1}s | {50+t0}–{50+t1}s · {note} | — | — |")
    lines += ["", "## Meters", "",
              "- Loss: mean/median/min PSNR, leftover MAE, nSat (K=K_MAX), k-hist",
              "- Size: origin zlib bytes (NAR4-like layout, MAGIC+len header counted as +9), raw, basis/coeff/JPEG",
              "- Seams: MAE on patch-grid lines vs interior; seamR = seam/interior",
              "- Decode: K′ ladder on phase A only; MACs ≈ meanK × patches × bw × bh × 3",
              "",
              "Shipped point: steps=2, K_MAX=16, target=32.5, int8, 16×16, cutHist=0.62, q=84.",
              ""]

    for ph, title in [
        ("A", "Phase A — shipped baselines"),
        ("B", "Phase B — work harder, bytes-free (steps + float ceiling)"),
        ("C", "Phase C — quality ↔ size (K_MAX, target, int4)"),
        ("D", "Phase D — same math, different support (patch, cuts)"),
    ]:
        lines += [f"## {title}", "", head]
        for r in phases.get(ph, []):
            lines.append(fmt(r))
        lines.append("")

    lines += ["## Phase E — tunable decode (K′ ladder on phase A origins)", ""]
    for r in phases.get("A", []):
        lines.append(f"### {r['window']} ({r['kind']}, origin {r['originBytes']/1e3:.1f} KB)")
        lines.append("")
        lines.append("| K′ | mean dB | min dB | MACs |")
        lines.append("|---|---|---|---|")
        for kp, v in r.get("ladder", {}).items():
            lines.append(f"| {kp} | {v['meanPsnr']:.2f} | {v['minPsnr']:.2f} | {v.get('macs', 0):.0f} |")
        lines.append("")

    def grab(phase, window, **kw):
        for r in rows:
            if r["phase"] != phase or r["window"] != window:
                continue
            ok = True
            for k, v in kw.items():
                if r.get(k) != v:
                    ok = False
                    break
            if ok:
                return r
        return None

    lines += ["## Kill sentences (written after the numbers)", ""]
    for w in WINDOWS:
        a = grab("A", w, trainSteps=2, quant="int8")
        b0 = grab("B", w, trainSteps=0, quant="int8")
        b32 = grab("B", w, trainSteps=32, quant="int8")
        fl = grab("B", w, quant="float32")
        if a and b0 and b32:
            dmean = b32["meanPsnr"] - b0["meanPsnr"]
            dmin = b32["minPsnr"] - b0["minPsnr"]
            dseam = b32["seamMae"] - b0["seamMae"]
            extra = (
                f" Float ceiling is {fl['meanPsnr']:.2f} dB ({fl['meanPsnr']-a['meanPsnr']:+.3f} vs shipped)."
                if fl
                else ""
            )
            lines.append(
                f"- **{w} training:** 32 ALS steps − 0 steps = {dmean:+.3f} dB mean, "
                f"{dmin:+.3f} dB min, {dseam:+.3f} seam MAE, origin Δ "
                f"{(b32['originBytes']-b0['originBytes'])/1e3:+.1f} KB. "
                f"Shipped steps=2 is {a['meanPsnr']:.2f} dB.{extra}"
            )
    lines.append("")
    for w in WINDOWS:
        a = grab("A", w)
        k8 = grab("C", w, kMax=8, quant="int8", targetPsnr=32.5)
        k32 = grab("C", w, kMax=32, quant="int8", targetPsnr=32.5)
        if a and k8 and k32:
            lines.append(
                f"- **{w} K_MAX:** 8→{k8['meanPsnr']:.2f}/{k8['minPsnr']:.2f} dB {k8['originBytes']/1e3:.0f}KB; "
                f"16→{a['meanPsnr']:.2f}/{a['minPsnr']:.2f} {a['originBytes']/1e3:.0f}KB; "
                f"32→{k32['meanPsnr']:.2f}/{k32['minPsnr']:.2f} {k32['originBytes']/1e3:.0f}KB "
                f"(sat {k32['nSat']}, seamR {k32['seamRatio']:.2f})."
            )
    lines.append("")
    for w in WINDOWS:
        a = grab("A", w)
        t30 = next((r for r in rows if r["phase"] == "C" and r["window"] == w and r["targetPsnr"] == 30.0), None)
        t35 = next((r for r in rows if r["phase"] == "C" and r["window"] == w and r["targetPsnr"] == 35.0), None)
        i4 = next((r for r in rows if r["phase"] == "C" and r["window"] == w and r["quant"] == "int4"), None)
        if a and t30 and t35 and i4:
            lines.append(
                f"- **{w} target/int4:** 30 dB knife {t30['meanPsnr']:.2f} / {t30['originBytes']/1e3:.0f}KB; "
                f"32.5 {a['meanPsnr']:.2f} / {a['originBytes']/1e3:.0f}KB; "
                f"35 {t35['meanPsnr']:.2f} / {t35['originBytes']/1e3:.0f}KB; "
                f"int4 {i4['meanPsnr']:.2f} / {i4['originBytes']/1e3:.0f}KB "
                f"({i4['meanPsnr']-a['meanPsnr']:+.2f} dB, origin {i4['originBytes']/a['originBytes']:.2f}×)."
            )
    lines.append("")
    for w in WINDOWS:
        a = grab("A", w)
        p8 = next((r for r in rows if r["phase"] == "D" and r["window"] == w and r["bw"] == 8), None)
        p32 = next((r for r in rows if r["phase"] == "D" and r["window"] == w and r["bw"] == 32), None)
        c40 = next((r for r in rows if r["phase"] == "D" and r["window"] == w and r["cutHist"] == 0.40), None)
        c80 = next((r for r in rows if r["phase"] == "D" and r["window"] == w and r["cutHist"] == 0.80), None)
        if a and p8 and p32:
            lines.append(
                f"- **{w} patch:** 8×8 seamR {p8['seamRatio']:.2f} min {p8['minPsnr']:.2f} origin {p8['originBytes']/1e3:.0f}KB; "
                f"16×16 seamR {a['seamRatio']:.2f} min {a['minPsnr']:.2f}; "
                f"32×32 seamR {p32['seamRatio']:.2f} min {p32['minPsnr']:.2f} mean {p32['meanPsnr']:.2f}."
            )
        if a and c40 and c80:
            lines.append(
                f"- **{w} cuts:** 0.40 → {c40['shots']} shots, {c40['meanPsnr']:.2f} dB, {c40['originBytes']/1e3:.0f}KB; "
                f"0.62 → {a['shots']} shots, {a['meanPsnr']:.2f} dB, {a['originBytes']/1e3:.0f}KB; "
                f"0.80 → {c80['shots']} shots, {c80['meanPsnr']:.2f} dB, {c80['originBytes']/1e3:.0f}KB."
            )
    lines += [
        "",
        "## Phase F",
        "",
        "Not run. Affine / shared-U / overlap / warp-then-SVD wait on the kill sentences above.",
        "",
        "## How to reproduce",
        "",
        "```",
        "python3 encoder/v4r/sweep.py",
        "```",
        "",
        "Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines) + "\n")
    (Path(__file__).resolve().parents[2] / "attempts" / "v4.t1r.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'README.md'} and attempts/v4.t1r.md")


def main() -> None:
    if not LOG.exists():
        LOG.write_text("")
    log("=== v4.t1r sweep start ===")
    wall0 = time.time()
    frames = load_all_frames()
    cfgs = configs()
    log(f"{len(cfgs)} configs")
    done = already_done()
    for cfg in cfgs:
        run_one(cfg, frames, done)
        done = already_done()
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    write_markdown(rows, wall_s=time.time() - wall0)
    log(f"=== done {len(rows)} rows ===")


if __name__ == "__main__":
    main()
