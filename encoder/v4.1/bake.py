#!/usr/bin/env python3
"""Decode the frozen v4 NAR4 origin at selectable K′ and bake lab reconstructs.

Leftover JPEG is applied only on the full rung — it was computed against full K.
v4 media under public/media/v4/ is not rewritten.
"""
from __future__ import annotations

import gc
import io
import json
import math
import shutil
import struct
import subprocess
import sys
import time
import zlib
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
FRAMES = TMP / "frames-640"
ORIGIN_V4 = MEDIA / "v4" / "origin.nar4"
OUT = MEDIA / "v4.1"
OUT.mkdir(parents=True, exist_ok=True)

ATTEMPT = "v4.1"
W, H_DISP, H = 640, 360, 384
BW, BH = 8, 8
COLS, ROWS = W // BW, H // BH
N_PATCH = COLS * ROWS
FPS = 24
MAGIC = b"NAR4"
K_RUNGS = (0, 1, 2, 4, 8, 16)


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


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


def jpeg_decode(blob: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(blob)).convert("RGB"))


def atlas_to_B(atlas_blob: bytes, ks: np.ndarray, bmn: np.ndarray, bsp: np.ndarray) -> list[np.ndarray]:
    d = BW * BH * 3
    total = int(bmn.size)
    out: list[np.ndarray] = [np.zeros((int(k), d), np.float32) for k in ks]
    if total == 0 or not atlas_blob:
        return out
    cols = max(1, int(math.ceil(math.sqrt(total))))
    rec = jpeg_decode(atlas_blob)
    idx = 0
    for ti, k in enumerate(ks):
        k = int(k)
        row = np.zeros((k, d), np.float32)
        for i in range(k):
            r, c = divmod(idx, cols)
            tile = rec[r * BH : (r + 1) * BH, c * BW : (c + 1) * BW].astype(np.float32).reshape(-1)
            row[i] = (tile / 255.0) * bsp[idx] + bmn[idx]
            idx += 1
        out[ti] = row
    return out


class Cursor:
    def __init__(self, raw: bytes):
        self.raw = raw
        self.off = 0

    def take(self, fmt: str):
        sz = struct.calcsize(fmt)
        vals = struct.unpack_from(fmt, self.raw, self.off)
        self.off += sz
        return vals

    def take_bytes(self, n: int) -> bytes:
        b = self.raw[self.off : self.off + n]
        if len(b) != n:
            raise SystemExit(f"origin short read want {n} at {self.off}/{len(self.raw)}")
        self.off += n
        return b


def unpack_origin(path: Path) -> dict:
    blob = path.read_bytes()
    if blob[:4] != MAGIC:
        raise SystemExit(f"bad magic {blob[:4]!r}")
    ver = blob[4]
    raw_len = struct.unpack_from("<I", blob, 5)[0]
    raw = zlib.decompress(blob[9:])
    if ver != 2:
        raise SystemExit(f"need NAR4 v2, got {ver}")
    if len(raw) != raw_len:
        raise SystemExit(f"zlib length {len(raw)} != {raw_len}")
    c = Cursor(raw)
    w, h_disp, fps, n_frames, nshot, bw, atlas_q, leftover_q = c.take("<HHHIHHBB")
    if (w, h_disp, bw) != (W, H_DISP, BW):
        raise SystemExit(f"unexpected header {(w, h_disp, fps, bw)}")
    shots = []
    for _ in range(nshot):
        i0, i1 = c.take("<HH")
        tlen = i1 - i0
        nmu = c.take("<I")[0]
        mu_blob = c.take_bytes(nmu)
        nat = c.take("<I")[0]
        atlas_blob = c.take_bytes(nat)
        nbasis = c.take("<I")[0]
        bmn = np.frombuffer(c.take_bytes(nbasis * 4), dtype=np.float32).copy()
        bsp = np.frombuffer(c.take_bytes(nbasis * 4), dtype=np.float32).copy()
        ks = np.frombuffer(c.take_bytes(N_PATCH), dtype=np.uint8).copy()
        b_list = atlas_to_B(atlas_blob, ks, bmn, bsp)
        items = []
        ti = 0
        for iy in range(ROWS):
            for ix in range(COLS):
                k = int(ks[ti])
                y0, x0 = iy * BH, ix * BW
                if k == 0:
                    uq = np.zeros((tlen, 0), np.float32)
                    bq = b_list[ti]
                else:
                    us = np.frombuffer(c.take_bytes(k * 4), dtype=np.float32).copy()
                    u = np.frombuffer(c.take_bytes(k * tlen), dtype=np.int8).reshape(k, tlen)
                    uq = (u.astype(np.float32) * us[:, None] / 127.0).T
                    bq = b_list[ti]
                items.append({"k": k, "y0": y0, "x0": x0, "Uq": uq, "Bq": bq})
                ti += 1
        nleft = c.take("<H")[0]
        leftover: list[tuple[int, bytes]] = []
        for _ in range(nleft):
            t_off, ln = c.take("<HI")
            leftover.append((int(t_off), c.take_bytes(ln)))
        mu = pad_frame(jpeg_decode(mu_blob))
        shots.append({"i0": int(i0), "i1": int(i1), "mu": mu, "items": items, "leftover": leftover})
    if c.off != len(raw):
        print(f"warn: trailing {len(raw) - c.off} bytes")
    return {
        "fps": int(fps),
        "n": int(n_frames),
        "atlasQ": int(atlas_q),
        "leftoverQ": int(leftover_q),
        "shots": shots,
    }


def paint(rec: np.ndarray, mu_u: np.ndarray, items: list[dict], kprime: int | None) -> None:
    tlen = rec.shape[0]
    mu_f = mu_u.astype(np.float32)
    for it in items:
        y0, x0 = it["y0"], it["x0"]
        k = int(it["k"])
        kuse = k if kprime is None else min(k, kprime)
        if kuse <= 0:
            rec[:, y0 : y0 + BH, x0 : x0 + BW] = mu_u[y0 : y0 + BH, x0 : x0 + BW]
            continue
        extra = (it["Uq"][:, :kuse] @ it["Bq"][:kuse]).reshape(tlen, BH, BW, 3)
        rec[:, y0 : y0 + BH, x0 : x0 + BW] = np.clip(np.round(mu_f[y0 : y0 + BH, x0 : x0 + BW] + extra), 0, 255).astype(np.uint8)


def apply_leftover(rec: np.ndarray, leftover: list[tuple[int, bytes]]) -> None:
    for t_off, blob in leftover:
        dec = jpeg_decode(blob)
        resid = dec.astype(np.int16) - 128
        rec[t_off, :H_DISP] = np.clip(rec[t_off, :H_DISP].astype(np.int16) + resid, 0, 255).astype(np.uint8)


def frame_files() -> list[Path]:
    pngs = sorted(FRAMES.glob("*.png"))
    if len(pngs) == FPS * 90:
        return pngs
    jpgs = sorted(FRAMES.glob("*.jpg"))
    if len(jpgs) == FPS * 90:
        return jpgs
    raise SystemExit(f"missing source frames in {FRAMES}")


def load_shot_src(files: list[Path], i0: int, i1: int) -> np.ndarray:
    sl = [pad_frame(np.array(Image.open(files[i]).convert("RGB"))) for i in range(i0, i1)]
    return np.stack(sl, 0)


def ffmpeg_dir(src_dir: Path, dest: Path) -> None:
    run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS), "-i", str(src_dir / "%04d.jpg"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-an",
            "-movflags", "+faststart", str(dest),
        ]
    )


def self_test(model: dict, files: list[Path]) -> None:
    ref_dir = TMP / "recon-v4"
    sh = model["shots"][0]
    rec = np.repeat(sh["mu"][None], sh["i1"] - sh["i0"], 0)
    paint(rec, sh["mu"], sh["items"], None)
    apply_leftover(rec, sh["leftover"])
    src = load_shot_src(files, sh["i0"], sh["i1"])
    d0 = psnr(src[0], rec[0])
    d10 = psnr(src[min(10, src.shape[0] - 1)], rec[min(10, rec.shape[0] - 1)])
    print(f"self-test S00 PSNR f0={d0:.2f} f10={d10:.2f}")
    if ref_dir.exists():
        ref = np.array(Image.open(ref_dir / "0000.jpg").convert("RGB"))
        mae = float(np.abs(ref.astype(np.int16) - rec[0, :H_DISP].astype(np.int16)).mean())
        print(f"self-test vs v4 recon f0 MAE={mae:.3f}")
        if mae > 4.0:
            raise SystemExit(f"decoder mismatch vs v4 reconstruct MAE={mae:.3f}")
    rec0 = np.repeat(sh["mu"][None], sh["i1"] - sh["i0"], 0)
    paint(rec0, sh["mu"], sh["items"], 0)
    if float(np.abs(rec0.astype(np.int16) - sh["mu"][None].astype(np.int16)).mean()) > 0.01:
        raise SystemExit("K′=0 is not the shot mean")
    print("self-test ok")


def bake(model: dict, files: list[Path]) -> dict[str, dict]:
    rungs: list[tuple[str, int | None]] = [(str(k), k) for k in K_RUNGS]
    dirs = {name: TMP / f"recon-v41-{name}" for name, _ in rungs}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    acc: dict[str, list[float]] = {name: [] for name, _ in rungs}
    t0 = time.time()
    for si, sh in enumerate(model["shots"]):
        i0, i1 = sh["i0"], sh["i1"]
        src = load_shot_src(files, i0, i1)
        tlen = i1 - i0
        print(f"  decode S{si:02d} [{i0},{i1}) T={tlen}", flush=True)
        for name, kp in rungs:
            rec = np.repeat(sh["mu"][None], tlen, 0)
            paint(rec, sh["mu"], sh["items"], kp)
            dest = dirs[name]
            for t in range(tlen):
                Image.fromarray(rec[t, :H_DISP]).save(dest / f"{i0 + t:04d}.jpg", "JPEG", quality=92)
                acc[name].append(psnr(src[t], rec[t]))
            del rec
        del src
        gc.collect()
    print(f"decode {time.time() - t0:.1f}s")
    stats: dict[str, dict] = {}
    for name, arr in acc.items():
        ordered = sorted(arr)
        stats[name] = {
            "meanPsnr": round(float(sum(arr) / len(arr)), 3),
            "minPsnr": round(float(ordered[0]), 3),
            "medianPsnr": round(float(ordered[len(ordered) // 2]), 3),
        }
        print(f"  K′={name:4s}  mean {stats[name]['meanPsnr']:.2f}  min {stats[name]['minPsnr']:.2f}")
    v4s = json.loads((MEDIA / "v4" / "stats.json").read_text())
    stats["full"] = {
        "meanPsnr": float(v4s["meanPsnr"]),
        "minPsnr": float(v4s["minPsnr"]),
        "medianPsnr": float(v4s["medianPsnr"]),
    }
    print(f"  K′=full  mean {stats['full']['meanPsnr']:.2f}  min {stats['full']['minPsnr']:.2f}  (frozen v4 reconstruct)")
    t1 = time.time()
    for name, _ in rungs:
        dest = OUT / f"reconstruct-k{name}.mp4"
        print(f"  ffmpeg {name} -> {dest.name}", flush=True)
        ffmpeg_dir(dirs[name], dest)
    shutil.copy2(MEDIA / "v4" / "reconstruct.mp4", OUT / "reconstruct.mp4")
    shutil.copy2(OUT / "reconstruct.mp4", MEDIA / "reconstruct.mp4")
    shutil.copy2(ORIGIN_V4, OUT / "origin.nar4")
    print(f"ffmpeg {time.time() - t1:.1f}s")
    return stats


def slim_stats(path: Path, attempt: str) -> dict | None:
    if not path.exists():
        return None
    s = json.loads(path.read_text())
    keys = (
        "attempt", "fps", "frames", "keyframes", "residualsStored", "modelBytes",
        "meanResidual", "meanPsnr", "skipBlockFrac", "reconstructMp4Bytes",
        "residualBytes", "intraBytes", "netBytes", "bitstreamBytes",
        "rawAccountedBytes", "gzipControlBytes",
    )
    out = {k: s.get(k) for k in keys}
    out["attempt"] = attempt
    return out


def write_lab(kstats: dict) -> None:
    data = json.loads((MEDIA / "analysis.json").read_text()) if (MEDIA / "analysis.json").exists() else json.loads(LAB_JSON.read_text())
    stats = data["stats"]
    if stats.get("attempt") == "v4" and "baselineV4" not in stats:
        stats["baselineV4"] = slim_stats(MEDIA / "v4" / "stats.json", "v4") or slim_stats(MEDIA / "stats-v4.json", "v4")
    stats["attempt"] = ATTEMPT
    kprime = {k: {"meanPsnr": kstats[str(k)]["meanPsnr"], "minPsnr": kstats[str(k)]["minPsnr"]} for k in K_RUNGS}
    kprime["full"] = {"meanPsnr": kstats["full"]["meanPsnr"], "minPsnr": kstats["full"]["minPsnr"]}
    stats["kPrime"] = kprime
    stats["meanPsnr"] = kstats["full"]["meanPsnr"]
    stats["minPsnr"] = kstats["full"]["minPsnr"]
    stats["medianPsnr"] = kstats["full"]["medianPsnr"]
    stats["reconstructMp4Bytes"] = (MEDIA / "reconstruct.mp4").stat().st_size
    src = data.setdefault("source", {})
    src["reconstruct"] = "/media/reconstruct.mp4"
    src["reconstructV4"] = "/media/v4/reconstruct.mp4"
    src["reconstructV4r"] = "/media/v4r/reconstruct.mp4"
    src["reconstructKPrime"] = {str(k): f"/media/v4.1/reconstruct-k{k}.mp4" for k in K_RUNGS}
    src["reconstructKPrime"]["full"] = "/media/v4.1/reconstruct.mp4"
    data["attempt"] = ATTEMPT
    (MEDIA / "analysis.json").write_text(json.dumps(data))
    if LAB_JSON.parent.exists():
        LAB_JSON.write_text(json.dumps(data))
    (MEDIA / "stats-v4.1.json").write_text(json.dumps(stats, indent=2))
    (OUT / "stats.json").write_text(json.dumps(stats, indent=2))
    (OUT / "analysis.json").write_text(json.dumps(data))
    print(json.dumps({"attempt": ATTEMPT, "kPrime": kprime, "origin": ORIGIN_V4.stat().st_size}, indent=2))


def main() -> None:
    if not ORIGIN_V4.exists():
        raise SystemExit(f"missing {ORIGIN_V4}")
    print("unpack", ORIGIN_V4)
    model = unpack_origin(ORIGIN_V4)
    files = frame_files()
    print(f"{model['n']} frames  {len(model['shots'])} shots  atlas q={model['atlasQ']}")
    self_test(model, files)
    if "--self-test" in sys.argv:
        return
    kstats = bake(model, files)
    write_lab(kstats)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
