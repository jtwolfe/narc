# Attempt v4.t1r — knob sweep on v4r (no UI change)

Branch: `attempt/v4.t1r`  
Frozen: `attempt/v4r` encoder math. Lab UI / `public/media/reconstruct.mp4` not rewritten.

Campaign is one-factor-at-a-time on three 10s windows of the 90s BBB probe.
Harness: `encoder/v4r/sweep.py`. Raw rows: [`results.jsonl`](v4.t1r/results.jsonl).

## Process

- Goal: measure training / K / target / quant / patch / cuts **before** adding affine or shared-U.
- Live lab stays on v4r media. This branch does not touch `src/` or rewrite reconstruct.
- Frames: cached `/tmp/bbb/frames-v1` (2160 JPEG @ 320×180 / 24 fps, source 50–140s).
- Shipped point: steps=2, K_MAX=16, target=32.5 dB, int8, 16×16, cutHist=0.62, JPEG q=84.
- Sanity: W0 shipped landed **35.20 / 27.75 dB**, mean K 3.50 — matches the v4r 10s gate (35.16 / 27.75, K 3.5).
- 42 configs (A 3 + B 12 + C 15 + D 12). Phase E is the K′ ladder on the three A origins, not extra encodes.
- Encode time 240 s summed (~4 min wall on 2 cores, including one resume after a watchdog kill). JSONL is append-only; the second `python3 encoder/v4r/sweep.py` skipped 11 ids and finished the rest.
- Phase F (overlap / warp-then-SVD / shared U / JPEG-on-B) is **not run** from this harness.

## Windows

| win | analysis | source | flux | kind |
|---|---|---|---|---|
| W0 | 0–10s | 50–60s · head (source 50–60s) | 2.48 | **locked** |
| W1 | 30–40s | 80–90s · mid (source 80–90s) | 1.78 | **locked** |
| W2 | 70–80s | 120–130s · late (source 120–130s) | 2.82 | **locked** |

Even-spaced 10s slices all labelled locked (flux 1.78–2.82; busy was >8, tracking >3.5). They still contain the hard frames: min PSNR 27.5–29.1, nSat 12–18 at K_MAX=16. We did **not** sample a busy window. Affine / warp conclusions are therefore incomplete; training / K / quant / patch conclusions are not — those knobs do not need motion.

## Meters

- Loss: mean/median/min PSNR, leftover MAE, nSat (K=K_MAX), k-hist
- Size: origin zlib bytes (NAR4-like layout, MAGIC+len header counted as +9), raw, basis/coeff/JPEG
- Seams: MAE on patch-grid lines vs interior; seamR = seam/interior
- Decode: K′ ladder on phase A only; MACs ≈ meanK × patches × bw × bh × 3

Shipped point: steps=2, K_MAX=16, target=32.5, int8, 16×16, cutHist=0.62, q=84.

## Phase A — shipped baselines

| id | win | kind | mean dB | min dB | leftover | mean K | sat | origin KB | seamR | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `A-W0-s2-k16-t32.5-int8-p16-c0.62` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.6 | 1.15 | 4.6 |
| `A-W1-s2-k16-t32.5-int8-p16-c0.62` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 860.7 | 1.15 | 5.7 |
| `A-W2-s2-k16-t32.5-int8-p16-c0.62` | W2 | locked | 34.10 | 29.13 | 3.35 | 3.24 | 18 | 1278.0 | 1.14 | 4.2 |

## Phase B — work harder, bytes-free (steps + float ceiling)

| id | win | kind | mean dB | min dB | leftover | mean K | sat | origin KB | seamR | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `B-W0-s0-k16-t32.5-int8-p16-c0.62` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.9 | 1.15 | 1.3 |
| `B-W1-s0-k16-t32.5-int8-p16-c0.62` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 861.2 | 1.15 | 1.8 |
| `B-W2-s0-k16-t32.5-int8-p16-c0.62` | W2 | locked | 34.10 | 29.13 | 3.35 | 3.24 | 18 | 1278.7 | 1.14 | 1.8 |
| `B-W0-s8-k16-t32.5-int8-p16-c0.62` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.7 | 1.15 | 8.1 |
| `B-W1-s8-k16-t32.5-int8-p16-c0.62` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 860.8 | 1.15 | 7.9 |
| `B-W2-s8-k16-t32.5-int8-p16-c0.62` | W2 | locked | 34.10 | 29.13 | 3.35 | 3.24 | 18 | 1277.9 | 1.14 | 10.4 |
| `B-W0-s32-k16-t32.5-int8-p16-c0.62` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.6 | 1.15 | 28.3 |
| `B-W1-s32-k16-t32.5-int8-p16-c0.62` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 860.8 | 1.15 | 26.4 |
| `B-W2-s32-k16-t32.5-int8-p16-c0.62` | W2 | locked | 34.10 | 29.13 | 3.35 | 3.24 | 18 | 1277.9 | 1.14 | 35.4 |
| `B-W0-s0-k16-t32.5-float32-p16-c0.62` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 5432.4 | 1.15 | 1.3 |
| `B-W1-s0-k16-t32.5-float32-p16-c0.62` | W1 | locked | 35.98 | 27.56 | 2.82 | 2.04 | 14 | 3522.4 | 1.15 | 1.4 |
| `B-W2-s0-k16-t32.5-float32-p16-c0.62` | W2 | locked | 34.11 | 29.14 | 3.34 | 3.24 | 18 | 5238.5 | 1.14 | 1.7 |

## Phase C — quality ↔ size (K_MAX, target, int4)

| id | win | kind | mean dB | min dB | leftover | mean K | sat | origin KB | seamR | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `C-W0-s2-k8-t32.5-int8-p16-c0.62` | W0 | locked | 34.54 | 27.22 | 3.01 | 3.10 | 63 | 1203.3 | 1.15 | 2.9 |
| `C-W1-s2-k8-t32.5-int8-p16-c0.62` | W1 | locked | 34.59 | 27.08 | 3.12 | 1.59 | 50 | 667.1 | 1.15 | 2.8 |
| `C-W2-s2-k8-t32.5-int8-p16-c0.62` | W2 | locked | 33.17 | 28.07 | 3.63 | 2.75 | 58 | 1069.3 | 1.15 | 3.6 |
| `C-W0-s2-k32-t32.5-int8-p16-c0.62` | W0 | locked | 35.34 | 27.82 | 2.81 | 3.64 | 0 | 1423.0 | 1.15 | 3.2 |
| `C-W1-s2-k32-t32.5-int8-p16-c0.62` | W1 | locked | 36.37 | 27.62 | 2.75 | 2.24 | 2 | 946.9 | 1.15 | 3.2 |
| `C-W2-s2-k32-t32.5-int8-p16-c0.62` | W2 | locked | 34.38 | 29.45 | 3.26 | 3.50 | 1 | 1390.1 | 1.14 | 4.0 |
| `C-W0-s2-k16-t30.0-int8-p16-c0.62` | W0 | locked | 33.17 | 25.58 | 3.62 | 2.38 | 6 | 939.3 | 1.13 | 3.0 |
| `C-W1-s2-k16-t30.0-int8-p16-c0.62` | W1 | locked | 33.94 | 25.56 | 3.56 | 1.31 | 6 | 570.5 | 1.15 | 2.5 |
| `C-W2-s2-k16-t30.0-int8-p16-c0.62` | W2 | locked | 32.44 | 26.99 | 4.07 | 2.41 | 9 | 962.3 | 1.13 | 3.8 |
| `C-W0-s2-k16-t35.0-int8-p16-c0.62` | W0 | locked | 37.32 | 30.05 | 2.25 | 4.77 | 21 | 1843.2 | 1.17 | 3.5 |
| `C-W1-s2-k16-t35.0-int8-p16-c0.62` | W1 | locked | 37.47 | 29.39 | 2.37 | 2.83 | 29 | 1156.2 | 1.15 | 4.0 |
| `C-W2-s2-k16-t35.0-int8-p16-c0.62` | W2 | locked | 36.01 | 31.33 | 2.61 | 4.20 | 28 | 1645.5 | 1.16 | 4.5 |
| `C-W0-s2-k16-t32.5-int4-p16-c0.62` | W0 | locked | 33.87 | 27.19 | 3.51 | 3.50 | 12 | 465.6 | 1.12 | 3.2 |
| `C-W1-s2-k16-t32.5-int4-p16-c0.62` | W1 | locked | 34.35 | 26.95 | 3.39 | 2.04 | 14 | 299.8 | 1.11 | 3.1 |
| `C-W2-s2-k16-t32.5-int4-p16-c0.62` | W2 | locked | 31.82 | 28.09 | 4.78 | 3.24 | 18 | 434.3 | 1.08 | 4.3 |

## Phase D — same math, different support (patch, cuts)

| id | win | kind | mean dB | min dB | leftover | mean K | sat | origin KB | seamR | s |
|---|---|---|---|---|---|---|---|---|---|---|
| `D-W0-s2-k16-t32.5-int8-p8-c0.62` | W0 | locked | 35.70 | 27.91 | 2.76 | 2.37 | 8 | 1322.6 | 1.14 | 5.2 |
| `D-W1-s2-k16-t32.5-int8-p8-c0.62` | W1 | locked | 36.52 | 27.82 | 2.76 | 1.37 | 6 | 881.2 | 1.13 | 5.2 |
| `D-W2-s2-k16-t32.5-int8-p8-c0.62` | W2 | locked | 34.75 | 30.12 | 3.22 | 2.28 | 6 | 1317.7 | 1.15 | 6.8 |
| `D-W0-s2-k16-t32.5-int8-p32-c0.62` | W0 | locked | 34.60 | 27.89 | 2.89 | 5.17 | 8 | 1675.7 | 1.19 | 3.3 |
| `D-W1-s2-k16-t32.5-int8-p32-c0.62` | W1 | locked | 34.95 | 27.30 | 2.94 | 2.69 | 11 | 897.3 | 1.16 | 2.7 |
| `D-W2-s2-k16-t32.5-int8-p32-c0.62` | W2 | locked | 33.26 | 28.18 | 3.43 | 4.45 | 11 | 1446.0 | 1.16 | 3.4 |
| `D-W0-s2-k16-t32.5-int8-p16-c0.4` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.6 | 1.15 | 3.4 |
| `D-W1-s2-k16-t32.5-int8-p16-c0.4` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 860.7 | 1.15 | 3.4 |
| `D-W2-s2-k16-t32.5-int8-p16-c0.4` | W2 | locked | 33.83 | 26.24 | 3.45 | 6.39 | 18 | 1318.3 | 1.15 | 4.8 |
| `D-W0-s2-k16-t32.5-int8-p16-c0.8` | W0 | locked | 35.20 | 27.75 | 2.84 | 3.50 | 12 | 1366.6 | 1.15 | 3.4 |
| `D-W1-s2-k16-t32.5-int8-p16-c0.8` | W1 | locked | 35.97 | 27.56 | 2.83 | 2.04 | 14 | 860.7 | 1.15 | 3.4 |
| `D-W2-s2-k16-t32.5-int8-p16-c0.8` | W2 | locked | 35.25 | 31.37 | 2.95 | 2.08 | 18 | 1509.5 | 1.10 | 2.5 |

## Phase E — tunable decode (K′ ladder on phase A origins)

### W0 (locked, origin 1366.6 KB)

| K′ | mean dB | min dB | MACs |
|---|---|---|---|
| 0 | 22.67 | 15.16 | 0 |
| 1 | 27.78 | 21.91 | 184320 |
| 2 | 30.17 | 23.60 | 368640 |
| 4 | 32.78 | 25.50 | 737280 |
| 8 | 34.54 | 27.22 | 1474560 |
| 16 | 35.20 | 27.75 | 2949120 |

### W1 (locked, origin 860.7 KB)

| K′ | mean dB | min dB | MACs |
|---|---|---|---|
| 0 | 24.01 | 16.87 | 0 |
| 1 | 28.02 | 22.05 | 184320 |
| 2 | 29.93 | 24.04 | 368640 |
| 4 | 32.33 | 25.74 | 737280 |
| 8 | 34.59 | 27.09 | 1474560 |
| 16 | 35.97 | 27.56 | 2949120 |

### W2 (locked, origin 1278.0 KB)

| K′ | mean dB | min dB | MACs |
|---|---|---|---|
| 0 | 17.52 | 15.02 | 0 |
| 1 | 24.60 | 20.11 | 184320 |
| 2 | 28.19 | 22.61 | 368640 |
| 4 | 31.20 | 25.66 | 737280 |
| 8 | 33.17 | 28.07 | 1474560 |
| 16 | 34.10 | 29.13 | 2949120 |

K′=0 is JPEG μ only. K′=4 is the ~32 dB decode rung (W2 slightly under). Full K is the encode-time adaptive rank, so K′=16 ≡ shipped.

## Kill sentences (written after the numbers)

- **W0 training:** 32 ALS steps − 0 steps = +0.000 dB mean, +0.001 dB min, +0.000 seam MAE, origin Δ -0.2 KB. Shipped steps=2 is 35.20 dB. Float ceiling is 35.20 dB (+0.007 vs shipped).
- **W1 training:** 32 ALS steps − 0 steps = +0.001 dB mean, -0.001 dB min, +0.000 seam MAE, origin Δ -0.4 KB. Shipped steps=2 is 35.97 dB. Float ceiling is 35.98 dB (+0.009 vs shipped).
- **W2 training:** 32 ALS steps − 0 steps = +0.000 dB mean, +0.000 dB min, -0.001 seam MAE, origin Δ -0.8 KB. Shipped steps=2 is 34.10 dB. Float ceiling is 34.11 dB (+0.012 vs shipped).

- **W0 K_MAX:** 8→34.54/27.22 dB 1203KB; 16→35.20/27.75 1367KB; 32→35.34/27.82 1423KB (sat 0, seamR 1.15).
- **W1 K_MAX:** 8→34.59/27.08 dB 667KB; 16→35.97/27.56 861KB; 32→36.37/27.62 947KB (sat 2, seamR 1.15).
- **W2 K_MAX:** 8→33.17/28.07 dB 1069KB; 16→34.10/29.13 1278KB; 32→34.38/29.45 1390KB (sat 1, seamR 1.14).

- **W0 target/int4:** 30 dB knife 33.17 / 939KB; 32.5 35.20 / 1367KB; 35 37.32 / 1843KB; int4 33.87 / 466KB (-1.32 dB, origin 0.34×).
- **W1 target/int4:** 30 dB knife 33.94 / 571KB; 32.5 35.97 / 861KB; 35 37.47 / 1156KB; int4 34.35 / 300KB (-1.62 dB, origin 0.35×).
- **W2 target/int4:** 30 dB knife 32.44 / 962KB; 32.5 34.10 / 1278KB; 35 36.01 / 1645KB; int4 31.82 / 434KB (-2.28 dB, origin 0.34×).

- **W0 patch:** 8×8 seamR 1.14 min 27.91 origin 1323KB; 16×16 seamR 1.15 min 27.75; 32×32 seamR 1.19 min 27.89 mean 34.60.
- **W0 cuts:** 0.40 → 2 shots, 35.20 dB, 1367KB; 0.62 → 2 shots, 35.20 dB, 1367KB; 0.80 → 2 shots, 35.20 dB, 1367KB.
- **W1 patch:** 8×8 seamR 1.13 min 27.82 origin 881KB; 16×16 seamR 1.15 min 27.56; 32×32 seamR 1.16 min 27.30 mean 34.95.
- **W1 cuts:** 0.40 → 2 shots, 35.97 dB, 861KB; 0.62 → 2 shots, 35.97 dB, 861KB; 0.80 → 2 shots, 35.97 dB, 861KB.
- **W2 patch:** 8×8 seamR 1.15 min 30.12 origin 1318KB; 16×16 seamR 1.14 min 29.13; 32×32 seamR 1.16 min 28.18 mean 33.26.
- **W2 cuts:** 0.40 → 1 shots, 33.83 dB, 1318KB; 0.62 → 2 shots, 34.10 dB, 1278KB; 0.80 → 4 shots, 35.25 dB, 1509KB.

## Reading (for the next v4r branch)

**Training is dead.** 32 ALS steps vs 0 is 0.000 dB on every window. The float32 ceiling is +0.01 dB at 4× origin. SVD is already L2-optimal; int8 is not the hole. `TRAIN_STEPS=2` can become 0 (faster encode, same reconstruct). Do not add SGD, more ALS, or a learned decoder *to claw back quant*. That experiment is done.

**The quality knob that actually moves is the MSE knife, not K_MAX.** Target 30 / 32.5 / 35 ≈ −2 / 0 / +2 dB and 0.69× / 1× / 1.35× origin. K_MAX 8→16 is +0.7–1.4 dB; 16→32 is only +0.14–0.40 dB (sat 12–18 → 0–2) for +4–10% bytes. Min PSNR barely moves with K_MAX. The leftover that sat at K=16 is not a rank-16 problem.

**int4 is the only size win we measured without a new representation.** 0.34× origin, −1.3 to −2.3 dB. W2 falls under the 32 dB knife (31.82). Mixed (int4 B, int8 U) is the obvious follow-up, still one-factor.

**8×8 is a small quality win at similar bytes** (+0.50–0.65 dB, origin 0.97–1.03×). Decode MACs actually drop (meanK falls more than patch count rises). **32×32 is worse** on mean, min, bytes, and seamR. SeamR stays 1.13–1.19 across 8 / 16 / 32 — the 16×16 *look* is not fixed by retilling. Block artifacts are independent tiles, not the tile size.

**Cuts only mattered on W2.** W0/W1 stay at 2 shots from 0.40–0.80. W2: merge to 1 shot loses 0.27 dB and 2.9 min-dB; split to 4 shots gains 1.15 dB / 2.2 min-dB and spends +18% bytes. Under-cutting a changing shot forces one temporal basis to span appearance. That is a shot-segmentation / shared-B problem, not a reason to add affine on locked windows.

**K′ decode is already the tunable path.** K′=4 ≈ 32 dB; K′=8 ≈ 34 dB; K′=0 is the JPEG mean (17–24 dB). Keep the ladder. Do not invent a second decode path until this one is in the lab.

**SeamR is 1.08–1.19 in every row.** Training, K, target, quant, patch, cuts do not invent cross-tile consistency. The 16×16 motion look is the representation. Phase F is now justified for *seams and bytes*, not for mean PSNR:

| candidate | justified by this sweep? |
|---|---|
| more ALS / SGD | **no** — 0.000 dB |
| K_MAX=32 default | weak — +0.2 dB, min unchanged |
| target 35 | only if we accept ~+35% origin |
| int4 or JPEG-on-B / shared B | **yes, size** — bases dominate; float tax is 4× for 0 dB |
| 8×8 default | mild yes — +0.5 dB, similar bytes, fewer MACs; seams stay |
| overlap-add | yes for *seams*, unmeasured visually here |
| affine / warp-then-SVD | **not yet** — all three windows locked; needs a flux>8 10s probe |
| shared U across shots | yes for *bytes* on multi-shot windows; W2 over-cut grew origin because B is per-shot |

## Phase F

Not run. The kill sentences above are the gate. Affine stays parked until a busy window exists. Size work (shared / compressed B) does not need motion.

## How to reproduce

```
python3 encoder/v4r/sweep.py
```

Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.
