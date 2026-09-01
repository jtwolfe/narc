#!/usr/bin/env python3
"""v4.1 blockiness probes. Lab UI / reconstruct.mp4 stay frozen.

Two artifacts: the 8×8 lattice (seamR) and tracking leftover (min dB on S06).
Does not iterate warp-then-SVD (killed in t4r).
"""
from __future__ import annotations

import gc
import importlib.util
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "attempts" / "v4.1"
OUT.mkdir(parents=True, exist_ok=True)
JSONL = OUT / "blockiness.jsonl"
LOG = OUT / "blockiness.log"
SHOTS_JSON = OUT / "shots.json"
if not SHOTS_JSON.exists():
    SHOTS_JSON = ROOT / "attempts" / "v4" / "shots.json"

FRAMES = Path("/tmp/bbb/frames-640")
FPS = 24
W, H_DISP, H = 640, 360, 384
BW, BH = 8, 8
COLS, ROWS = W // BW, H // BH
N_PATCH = COLS * ROWS
K_MAX = 16
TRAIN_STEPS = 2
TARGET_PSNR = 32.5
TARGET_MSE = 255.0 * 255.0 / (10 ** (TARGET_PSNR / 10.0))
ATLAS_Q = 70
MEAN_JPEG_Q = 84


def load_mod(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bake = load_mod("v41bake", HERE / "bake.py")
v4 = load_mod("v4enc", HERE.parent / "v4" / "analyze.py")


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


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    return bake.psnr(a, b)


def seam_mae(src: np.ndarray, rec: np.ndarray, bw: int = BW, bh: int = BH) -> tuple[float, float, float]:
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


def score(src: np.ndarray, rec: np.ndarray) -> dict:
    ps = [psnr(src[t], rec[t]) for t in range(src.shape[0])]
    maes = [
        float(np.abs(src[t, :H_DISP].astype(np.int16) - rec[t, :H_DISP].astype(np.int16)).mean())
        for t in range(src.shape[0])
    ]
    ratios = [seam_mae(src[t], rec[t])[2] for t in range(0, src.shape[0], max(1, src.shape[0] // 8))]
    ordered = sorted(ps)
    return {
        "meanPsnr": round(float(sum(ps) / len(ps)), 3),
        "minPsnr": round(float(ordered[0]), 3),
        "medianPsnr": round(float(ordered[len(ordered) // 2]), 3),
        "meanMae": round(float(sum(maes) / len(maes)), 3),
        "seamR": round(float(sum(ratios) / len(ratios)), 3),
    }


def jpeg_bytes(rgb: np.ndarray, quality: int) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def jpeg_dec(blob: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))


def leftover_frames(src: np.ndarray, rec: np.ndarray, stride: int, mae_cut: float, quality: int) -> tuple[np.ndarray, int, int]:
    out = rec.copy()
    nbytes = 0
    n = 0
    tlen = src.shape[0]
    for t in range(0, tlen, max(1, stride)):
        mae = float(np.abs(src[t, :H_DISP].astype(np.int16) - rec[t, :H_DISP].astype(np.int16)).mean())
        if mae < mae_cut:
            continue
        diff = np.clip(src[t, :H_DISP].astype(np.int16) - rec[t, :H_DISP].astype(np.int16) + 128, 0, 255).astype(np.uint8)
        blob = jpeg_bytes(diff, quality)
        dec = jpeg_dec(blob)
        resid = dec.astype(np.int16) - 128
        out[t, :H_DISP] = np.clip(out[t, :H_DISP].astype(np.int16) + resid, 0, 255).astype(np.uint8)
        nbytes += 4 + len(blob)
        n += 1
    return out, nbytes, n


def leftover_tiles(src: np.ndarray, rec: np.ndarray, mae_cut: float, quality: int) -> tuple[np.ndarray, int, int]:
    """Per-frame mosaic of dirty 8×8 residual tiles only."""
    out = rec.copy()
    nbytes = 0
    nframes = 0
    tlen = src.shape[0]
    d = BH * BW * 3
    for t in range(tlen):
        dirty: list[tuple[int, int, np.ndarray]] = []
        for iy in range(ROWS):
            for ix in range(COLS):
                y0, x0 = iy * BH, ix * BW
                if y0 >= H_DISP:
                    continue
                a = src[t, y0 : y0 + BH, x0 : x0 + BW]
                b = rec[t, y0 : y0 + BH, x0 : x0 + BW]
                mae = float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())
                if mae < mae_cut:
                    continue
                dirty.append((iy, ix, np.clip(a.astype(np.int16) - b.astype(np.int16) + 128, 0, 255).astype(np.uint8)))
        if not dirty:
            continue
        nframes += 1
        cols = max(1, int(math.ceil(math.sqrt(len(dirty)))))
        rows = int(math.ceil(len(dirty) / cols))
        canvas = np.zeros((rows * BH, cols * BW, 3), np.uint8)
        idx = np.zeros((len(dirty), 2), np.uint8)
        for i, (iy, ix, tile) in enumerate(dirty):
            r, c = divmod(i, cols)
            canvas[r * BH : (r + 1) * BH, c * BW : (c + 1) * BW] = tile
            idx[i] = (iy, ix)
        blob = jpeg_bytes(canvas, quality)
        dec = jpeg_dec(blob)
        for i, (iy, ix, _) in enumerate(dirty):
            r, c = divmod(i, cols)
            tile = dec[r * BH : (r + 1) * BH, c * BW : (c + 1) * BW]
            resid = tile.astype(np.int16) - 128
            y0, x0 = iy * BH, ix * BW
            out[t, y0 : y0 + BH, x0 : x0 + BW] = np.clip(
                out[t, y0 : y0 + BH, x0 : x0 + BW].astype(np.int16) + resid, 0, 255
            ).astype(np.uint8)
        nbytes += 4 + len(blob) + 2 + idx.nbytes
        _ = d
    return out, nbytes, nframes


def deblock(rec_u: np.ndarray, ov: int) -> np.ndarray:
    """Per-frame seam mix. Grid is in the model — this only scores the t3r-dead path."""
    if ov <= 0:
        return rec_u
    out = np.empty_like(rec_u)
    for t in range(rec_u.shape[0]):
        rec = rec_u[t].astype(np.float32)
        fr = rec.copy()
        hh, ww, _ = rec.shape

        def mix(a, b, s):
            return a * (1.0 - s) + b * s

        for x in range(BW, ww, BW):
            for i in range(ov):
                a = 0.10 * (ov - i) / ov
                L, R = x - 1 - i, x + i
                if L < 0 or R >= ww:
                    continue
                fr[:, L, :] = mix(rec[:, L, :], rec[:, R, :], a)
                fr[:, R, :] = mix(rec[:, R, :], rec[:, L, :], a)
        for y in range(BH, min(hh, H_DISP + ov), BH):
            for i in range(ov):
                a = 0.10 * (ov - i) / ov
                T, B = y - 1 - i, y + i
                if T < 0 or B >= hh:
                    continue
                fr[T, :, :] = mix(rec[T, :, :], rec[B, :, :], a)
                fr[B, :, :] = mix(rec[B, :, :], rec[T, :, :], a)
        out[t] = np.clip(np.round(fr), 0, 255).astype(np.uint8)
    return out


def trap_window(bw: int, hop: int) -> np.ndarray:
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


def encode_overlap(sl: np.ndarray, hop: int) -> tuple[np.ndarray, dict]:
    tlen = sl.shape[0]
    mu = v4.shot_mu(sl)
    mu_u, mu_blob = v4.jpeg_roundtrip(np.clip(np.round(mu), 0, 255).astype(np.uint8), MEAN_JPEG_Q)
    mu_f = mu_u.astype(np.float32)
    ys = tile_starts(H, BH, hop)
    xs = tile_starts(W, BW, hop)
    window = trap_window(BW, hop)
    tiles: list[dict] = []
    bs_list: list[np.ndarray] = []
    u_bytes = 0
    k_hist: Counter[int] = Counter()
    n_tiles = len(ys) * len(xs)
    log(f"    fit {n_tiles} hop-{hop} tiles T={tlen} rss={rss_mb():.0f}MB")
    for ti, y0 in enumerate(ys):
        if ti and ti % 16 == 0:
            log(f"    row {ti}/{len(ys)} rss={rss_mb():.0f}MB")
        for x0 in xs:
            p = sl[:, y0 : y0 + BH, x0 : x0 + BW].astype(np.float32).reshape(tlen, -1)
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW].reshape(-1)
            xc = p - m
            k, u, b = v4.fit_patch(xc, TARGET_MSE, K_MAX)
            if k > 0:
                uq, uc, us = v4.qint8(u.T, axis=0)
                uq = uq.T
                bq, _, _ = v4.qint8(b, axis=0)
                uq, bq = v4.train_factors(xc, uq, bq, TRAIN_STEPS)
                uq, uc, us = v4.qint8(uq.T, axis=0)
                uq = uq.T
                u_bytes += uc.nbytes + np.asarray(us).nbytes
            else:
                uq = np.zeros((tlen, 0), np.float32)
                bq = np.zeros((0, BW * BH * 3), np.float32)
            k_hist[k] += 1
            tiles.append({"y0": y0, "x0": x0, "k": k, "Uq": uq, "Bq": bq})
            bs_list.append(bq)
    atlas_blob, bq_list, bmn, bsp = v4.make_atlas(bs_list, BW, BH, ATLAS_Q)
    del bs_list
    for it, bq in zip(tiles, bq_list):
        it["Bq"] = bq
    del bq_list
    rec = np.empty((tlen, H, W, 3), np.uint8)
    win = window[:, :, None]
    for t in range(tlen):
        acc = np.zeros((H, W, 3), np.float32)
        wgt = np.zeros((H, W), np.float32)
        for it in tiles:
            y0, x0, k = it["y0"], it["x0"], it["k"]
            m = mu_f[y0 : y0 + BH, x0 : x0 + BW]
            if k <= 0:
                patch = m
            else:
                patch = m + (it["Uq"][t] @ it["Bq"]).reshape(BH, BW, 3)
            acc[y0 : y0 + BH, x0 : x0 + BW] += patch * win
            wgt[y0 : y0 + BH, x0 : x0 + BW] += window
        rec[t] = np.clip(np.round(acc / np.maximum(wgt, 1e-6)[:, :, None]), 0, 255).astype(np.uint8)
    for it in tiles:
        it.pop("Uq", None)
        it.pop("Bq", None)
    raw = len(mu_blob) + len(atlas_blob) + bmn.nbytes + bsp.nbytes + n_tiles + u_bytes
    body = mu_blob + atlas_blob + bmn.tobytes() + bsp.tobytes()
    meters = {
        "nTiles": n_tiles,
        "meanK": float(sum(k * n for k, n in k_hist.items()) / max(sum(k_hist.values()), 1)),
        "muBytes": len(mu_blob),
        "atlasBytes": len(atlas_blob),
        "uBytes": u_bytes,
        "rawBytes": raw,
        "zlibBytes": len(zlib.compress(body, 9)) + u_bytes,
    }
    return rec, meters


EXISTING: set[str] = set()
CROPS = OUT / "crops"


def load_existing() -> None:
    global EXISTING
    if JSONL.exists():
        EXISTING = {json.loads(l)["id"] for l in JSONL.read_text().splitlines() if l.strip()}
    else:
        EXISTING = set()


def emit(row: dict) -> None:
    if row["id"] in EXISTING:
        log(f"  skip {row['id']}")
        return
    with JSONL.open("a") as f:
        f.write(json.dumps(row) + "\n")
    EXISTING.add(row["id"])
    log(
        f"  {row['id']:28s}  {row['shot']}  {row['meanPsnr']:6.2f}/{row['minPsnr']:5.2f}  "
        f"seamR {row['seamR']:.2f}  extra {row['extraBytes']/1e3:.0f}KB  rss={row['rss']:.0f}"
    )


def dump_jpg(name: str, rgb: np.ndarray) -> None:
    CROPS.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb[:H_DISP].astype(np.uint8)).save(CROPS / f"{name}.jpg", quality=88, optimize=True)


def dump_shot_stills(sid: str, src: np.ndarray, rec: np.ndarray, extra: dict[str, np.ndarray], tpick: int) -> None:
    t = min(tpick, src.shape[0] - 1)
    dump_jpg(f"{sid}-t{t}-src", src[t])
    dump_jpg(f"{sid}-t{t}-svd", rec[t])
    for tag, vol in extra.items():
        dump_jpg(f"{sid}-t{t}-{tag}", vol[t] if vol.shape[0] > t else vol[0])
    y0, x0 = (80, 240) if sid == "S06" else (40, 200)
    y0 = min(y0, H_DISP - 160)
    x0 = min(x0, W - 160)
    tiles = [src[t, y0 : y0 + 160, x0 : x0 + 160], rec[t, y0 : y0 + 160, x0 : x0 + 160]]
    for tag in ("left-s1", "deblock2", "overlap"):
        if tag in extra:
            vol = extra[tag]
            fr = vol[t] if vol.shape[0] > t else vol[0]
            tiles.append(fr[y0 : y0 + 160, x0 : x0 + 160])
    dump_jpg(f"{sid}-t{t}-zoom", np.concatenate(tiles, axis=1))


def main() -> None:
    from_overlap = "--from-overlap" in sys.argv
    if not from_overlap:
        JSONL.write_text("")
        LOG.write_text("")
    load_existing()
    shots = json.loads(SHOTS_JSON.read_text())
    files = bake.frame_files()
    origin = Path("/workspace/public/media/v4/origin.nar4")
    if not origin.exists():
        origin = ROOT / "public" / "media" / "v4" / "origin.nar4"
    log(f"unpack {origin} from_overlap={from_overlap} have={len(EXISTING)}")
    model = bake.unpack_origin(origin)
    if from_overlap:
        keep_sid = {"S09", "S06", "S00"}
        for si, sh in enumerate(model["shots"]):
            if shots[si]["sid"] not in keep_sid:
                sh["items"] = []
                sh["leftover"] = []
                sh["mu"] = None
        gc.collect()
        log(f"  dropped unused shots rss={rss_mb():.0f}MB")
    t0 = time.time()
    recs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    focus = {"S00", "S03", "S06", "S09", "S13"}
    leftover_cfgs = [
        ("s8", 8, 3.5, 40),
        ("s4", 4, 3.5, 40),
        ("s2", 2, 3.5, 40),
        ("s1", 1, 3.5, 40),
        ("s1-all", 1, 0.0, 40),
        ("s1-q30", 1, 3.5, 30),
    ]
    if from_overlap:
        log("resume D+O")
        for si, sh in enumerate(model["shots"]):
            sid = shots[si]["sid"]
            if sid not in ("S09", "S06", "S00"):
                continue
            src = bake.load_shot_src(files, sh["i0"], sh["i1"])
            rec = np.repeat(sh["mu"][None], sh["i1"] - sh["i0"], 0)
            bake.paint(rec, sh["mu"], sh["items"], None)
            recs[sid] = (src, rec)
            for it in sh["items"]:
                it.pop("Uq", None)
                it.pop("Bq", None)
            log(f"  loaded {sid} rss={rss_mb():.0f}MB")
        gc.collect()
    else:
        episode_left: dict[str, int] = {k: 0 for k, *_ in leftover_cfgs}
        episode_ps: dict[str, list[tuple[int, float, float]]] = {k: [] for k, *_ in leftover_cfgs}
        for si, sh in enumerate(model["shots"]):
            i0, i1 = sh["i0"], sh["i1"]
            sid = shots[si]["sid"]
            src = bake.load_shot_src(files, i0, i1)
            rec = np.repeat(sh["mu"][None], i1 - i0, 0)
            bake.paint(rec, sh["mu"], sh["items"], None)
            for it in sh["items"]:
                it.pop("Uq", None)
                it.pop("Bq", None)
            sc = score(src, rec)
            emit({
                "id": f"A-{sid}-svd",
                "phase": "A",
                "shot": sid,
                "kind": shots[si]["kind"],
                "extraBytes": 0,
                "nKeys": 0,
                "rss": rss_mb(),
                **sc,
            })
            for name, stride, cut, q in leftover_cfgs:
                out, nbytes, n = leftover_frames(src, rec, stride, cut, q)
                sc2 = score(src, out)
                episode_left[name] += nbytes
                episode_ps[name].append((src.shape[0], sc2["meanPsnr"], sc2["minPsnr"]))
                if sid in focus:
                    emit({
                        "id": f"L-{sid}-{name}",
                        "phase": "L",
                        "shot": sid,
                        "kind": shots[si]["kind"],
                        "extraBytes": nbytes,
                        "nKeys": n,
                        "stride": stride,
                        "maeCut": cut,
                        "q": q,
                        "rss": rss_mb(),
                        **sc2,
                    })
                del out
            if sid in ("S06", "S09", "S00"):
                out, nbytes, n = leftover_tiles(src, rec, 3.5, 40)
                sc2 = score(src, out)
                emit({
                    "id": f"L-{sid}-tiles",
                    "phase": "L",
                    "shot": sid,
                    "kind": shots[si]["kind"],
                    "extraBytes": nbytes,
                    "nKeys": n,
                    "stride": 1,
                    "maeCut": 3.5,
                    "q": 40,
                    "tile": True,
                    "rss": rss_mb(),
                    **sc2,
                })
                del out
            if sid in focus:
                recs[sid] = (src, rec)
            else:
                del src, rec
            gc.collect()
            log(f"  painted {sid} rss={rss_mb():.0f}MB")
        log(f"svd+leftover {time.time() - t0:.1f}s  rss={rss_mb():.0f}MB")

        for name, _, _, _ in leftover_cfgs:
            tot = sum(fr for fr, _, _ in episode_ps[name]) or 1
            wmean = sum(m * fr for fr, m, _ in episode_ps[name]) / tot
            wmin = min(mn for _, _, mn in episode_ps[name])
            emit({
                "id": f"L-EP-{name}",
                "phase": "L",
                "shot": "EP",
                "kind": "episode",
                "extraBytes": episode_left[name],
                "nKeys": 0,
                "rss": rss_mb(),
                "meanPsnr": round(wmean, 3),
                "minPsnr": round(wmin, 3),
                "medianPsnr": 0.0,
                "meanMae": 0.0,
                "seamR": 0.0,
            })

    # D — decode deblock on SVD rec
    for sid in ("S09", "S06", "S00"):
        if sid not in recs:
            continue
        src, rec = recs[sid]
        for ov in (1, 2, 4):
            out = deblock(rec, ov)
            sc = score(src, out)
            emit({
                "id": f"D-{sid}-ov{ov}",
                "phase": "D",
                "shot": sid,
                "kind": next(s["kind"] for s in shots if s["sid"] == sid),
                "extraBytes": 0,
                "nKeys": 0,
                "ov": ov,
                "rss": rss_mb(),
                **sc,
            })
            del out
        gc.collect()

    for sid, tpick in (("S06", 80), ("S09", 100)):
        if sid not in recs:
            continue
        src, rec = recs[sid]
        extra = {"deblock2": deblock(rec, 2)}
        left, _, _ = leftover_frames(src, rec, 1, 3.5, 40)
        extra["left-s1"] = left
        dump_shot_stills(sid, src, rec, extra, tpick)
        del extra, left
        gc.collect()
        log(f"  crops {sid} rss={rss_mb():.0f}MB")

    for sid in list(recs):
        if sid not in ("S09", "S06"):
            del recs[sid]
    gc.collect()
    log(f"overlap prep rss={rss_mb():.0f}MB keep {list(recs)}")

    # O — 8×8 hop-4 Hann overlap (COLA) on locked + tracking reps
    for sid in ("S09", "S06"):
        if sid not in recs:
            continue
        src, rec_a = recs[sid]
        sl = src
        if f"O-{sid}-h4" in EXISTING:
            log(f"overlap {sid} already scored")
            continue
        log(f"overlap {sid} hop-4 tiles~{len(tile_starts(H, BH, 4))*len(tile_starts(W, BW, 4))}")
        rec_o, meters = encode_overlap(sl, hop=4)
        sc = score(sl, rec_o)
        emit({
            "id": f"O-{sid}-h4",
            "phase": "O",
            "shot": sid,
            "kind": next(s["kind"] for s in shots if s["sid"] == sid),
            "extraBytes": meters["zlibBytes"],
            "nKeys": meters["nTiles"],
            "atlasBytes": meters["atlasBytes"],
            "uBytes": meters["uBytes"],
            "zlibBytes": meters["zlibBytes"],
            "rawBytes": meters["rawBytes"],
            "rss": rss_mb(),
            **sc,
        })
        tpick = 80 if sid == "S06" else 100
        extra_o = {"overlap": rec_o}
        if sid == "S06":
            for name, stride, cut, q in (("s8", 8, 3.5, 40), ("s1", 1, 3.5, 40)):
                out, nbytes, n = leftover_frames(sl, rec_o, stride, cut, q)
                sc2 = score(sl, out)
                emit({
                    "id": f"OL-{sid}-{name}",
                    "phase": "OL",
                    "shot": sid,
                    "kind": "tracking",
                    "extraBytes": meters["zlibBytes"] + nbytes,
                    "nKeys": n,
                    "rss": rss_mb(),
                    **sc2,
                })
                if name == "s1":
                    extra_o["left-s1"] = out
                del out
        dump_shot_stills(sid, sl, rec_a, extra_o, tpick)
        del rec_o, extra_o
        gc.collect()

    log(f"done {time.time() - t0:.1f}s  rss={rss_mb():.0f}MB")
    write_reading()


def write_reading() -> None:
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    by = {r["id"]: r for r in rows}

    def g(i):
        return by.get(i)

    lines = [
        "# v4.1 blockiness probes",
        "",
        "Lab frozen. Frozen v4 origin. Two artifacts: 8×8 lattice (seamR) and tracking leftover (S06 min dB).",
        "Warp-then-SVD was not re-run (killed in t4r, −5.7 to −14 dB).",
        "",
        "## Verdict",
        "",
    ]
    a06, a09 = g("A-S06-svd"), g("A-S09-svd")
    l1, l8, l2 = g("L-S06-s1"), g("L-S06-s8"), g("L-S06-s2")
    d = g("D-S09-ov2")
    d06 = g("D-S06-ov2")
    o09, o06 = g("O-S09-h4"), g("O-S06-h4")
    ol = g("OL-S06-s1")
    lep = g("L-EP-s1")
    if l1 and a06:
        lines.append(
            "**Best method for the moving-element blocks: dense leftover JPEG (stride 1, MAE>3.5, q=40).** "
            f"S06 {l1['meanPsnr']:.2f}/{l1['minPsnr']:.2f} vs SVD {a06['meanPsnr']:.2f}/{a06['minPsnr']:.2f} "
            f"({l1['meanPsnr']-a06['meanPsnr']:+.2f} dB mean, min {a06['minPsnr']:.2f}→{l1['minPsnr']:.2f}), "
            f"+{l1['extraBytes']/1e6:.2f} MB on the tracking shot. Episode leftover vs v4's 15.23 MB SVD origin: "
            f"{lep['extraBytes']/1e6:.2f} MB extra / {lep['meanPsnr']:.2f} mean / {lep['minPsnr']:.2f} min." if lep else
            f"+{l1['extraBytes']/1e6:.2f} MB on S06."
        )
    if l2 and l1 and a06:
        lines.append(
            f"Stride must be 1. S06 s2 {l2['meanPsnr']:.2f}/{l2['minPsnr']:.2f} — min barely moves vs SVD {a06['minPsnr']:.2f}. "
            f"s8 {l8['meanPsnr']:.2f}/{l8['minPsnr']:.2f} is a meter. The worst frames are the ones stride skips."
        )
    if d and a09:
        lines.append(
            f"**Deblock is dead.** S09 ov2 {d['meanPsnr']:.2f} seamR {d['seamR']:.2f} vs A {a09['meanPsnr']:.2f}/{a09['seamR']:.2f}. "
            + (f"S06 ov2 {d06['meanPsnr']:.2f}/{d06['seamR']:.2f} vs A {a06['meanPsnr']:.2f}/{a06['seamR']:.2f}. " if d06 and a06 else "")
            + "The grid is in the model, not a 10% seam mix."
        )
    if o09 and a09:
        lines.append(
            f"**Overlap kills the lattice, costs ~4× tiles.** S09 hop-4 {o09['meanPsnr']:.2f} seamR {o09['seamR']:.2f} "
            f"vs A {a09['seamR']:.2f}, zlib {o09['extraBytes']/1e6:.1f} MB, {o09['nKeys']} tiles."
        )
    if o06 and a06:
        lines.append(
            f"S06 hop-4 {o06['meanPsnr']:.2f}/{o06['minPsnr']:.2f} seamR {o06['seamR']:.2f} vs A "
            f"{a06['meanPsnr']:.2f}/{a06['minPsnr']:.2f}/{a06['seamR']:.2f}."
        )
    if ol and l1:
        lines.append(
            f"Overlap+leftover s1 S06 {ol['meanPsnr']:.2f}/{ol['minPsnr']:.2f} vs leftover-only {l1['meanPsnr']:.2f}/{l1['minPsnr']:.2f}. "
            "Ship leftover first; add hop-4 only if the remaining 8×8 lattice on locked shots is still the demo complaint."
        )
    lines += [
        "",
        "Do not ship affine / warp-then-SVD. Do not ship decode deblock. Do not ship stride-8 leftover as a quality stack.",
        "",
        "## Reading",
        "",
    ]
    if a06 and a09:
        lines.append(
            f"SVD-only baseline (leftover off): S09 {a09['meanPsnr']:.2f}/{a09['minPsnr']:.2f} seamR {a09['seamR']:.2f}; "
            f"S06 {a06['meanPsnr']:.2f}/{a06['minPsnr']:.2f} seamR {a06['seamR']:.2f}."
        )
    if l1 and l8 and a06:
        lines.append(
            f"Dense leftover buys the tracking hole. S06 s8 {l8['meanPsnr']:.2f}/{l8['minPsnr']:.2f} "
            f"({l8['meanPsnr']-a06['meanPsnr']:+.2f} dB, {l8['extraBytes']/1e3:.0f}KB) vs "
            f"s1 {l1['meanPsnr']:.2f}/{l1['minPsnr']:.2f} ({l1['meanPsnr']-a06['meanPsnr']:+.2f} dB, {l1['extraBytes']/1e3:.0f}KB)."
        )
    lines += ["", "## Rows", "", "| id | mean | min | seamR | extra |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| `{r['id']}` | {r['meanPsnr']:.2f} | {r['minPsnr']:.2f} | {r['seamR']:.2f} | {r['extraBytes']/1e3:.0f}KB |"
        )
    (OUT / "blockiness.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {OUT / 'blockiness.md'}")


if __name__ == "__main__":
    main()
