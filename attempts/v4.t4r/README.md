# Attempt v4.t4r — 8×8 product stack, JPEG-on-B, warp, leftover ceiling

Branch: `attempt/v4.t4r` off `attempt/v4.t3r`.  
Frozen lab: no `src/` / `reconstruct.mp4` rewrite.

Raster: **640×360** crop, canvas 640×384. Same shots as t2r/t3r.
Default tile **8×8**, TRAIN_STEPS=2, K_MAX=16, target 32.5 dB, JPEG-B q=70.

## Reading (after the numbers)

45 rows. Lab UI frozen. Peak RSS 765 MB (S14).

**Episode 8×8 int8 is the new baseline.** Time-weighted **35.76 dB / min 27.52**, origin **32.5 MB** vs t2r 16×16 35.22 / 34.9 MB. Same raster, +0.54 dB and −7% bytes. S09 37.31 / 1.00 MB and S06 33.80 / 5.92 MB match t3r P exactly. 8×8 is not a probe anymore; it is the unit.

**Per-tile JPEG-on-B is the wrong packing at 8×8.** q=70: episode **34.36 dB / 16.6 MB (2.78× H.264)** — only **0.47–0.53×** A, vs t3r’s 0.20× on 16×16. q=50 vs q=70 on S09 is 444 vs 467 KB. Tiny 8×8 mosaics are JPEG-header bound. Atlas-of-B (one JPEG per shot, not 3840) is the size move this stack still has. Do not ship per-tile JPEG-B at 8×8.

**40 dB is reachable at 8×8 if you spend K.** Locked S09: **43.52 / 39.67** at 2.4 MB (2.4× A). Tracking S06: **40.29 / 37.45**, sat 1045/3840, **14.7 MB (2.5× A)**. t2r 16×16 never hit mean 40 on S06 (36.14). Converter-quality on motion is a tile-size × knife product, not a dead end. JPEG-B at the 40 knife throws the extra K away (S06 35.09).

**Naive translation-then-SVD dies.** wshot is a no-op on locked (|MV|=0, +0.00 dB) and a **−5.7 dB** wreck on S06 (|MV|=9.9 px) — global trans + forward splat vs local motion. wtile is worse (S09 **−8.9 dB**, S06 **−14 dB**): 8×8 search matches neighbors, MVs dance, splat leaves holes. This formulation is not H.264 MC (predict current, residual in current coords). Do not iterate it. Affine/scale is not the next patch on top of this.

**The tracking hole is leftover, and leftover JPEG buys it.** S06 + per-frame leftover JPEG q=40: **36.97 / 35.38 (+3.17 / +3.90 dB vs A)**, origin 7.22 MB (+1.30 MB, leftover blob 1.38 MB). Cheaper than the 40 dB knife (14.7 MB) for a smaller quality step. Residual after 8×8 SVD is the quality path that actually moved min PSNR.

**K′ still works** on 8×8 and on JPEG-B. A S09: K′=4 is 36.76. Q S09: K′=4 is already 40.16.

**What to build next:** (1) **8×8 SVD + leftover residual** (JPEG or a second cheap pass) as the quality stack; (2) **atlas JPEG-on-B** (one mosaic per shot) as the size stack; (3) stop spending on per-tile JPEG-B, exclusive merge, and this warp. Episode 8×8 int8 is 32.5 MB / 35.8 dB; H.264 is 6.8 MB. Atlas + residual is how you close both gaps.


## What this pass is

- **A** 8×8 disjoint int8 on all 15 shots (new baseline / episode total)
- **J** 8×8 JPEG-on-B q=70 on all 15 shots (product size stack); q=50/90 on L+B
- **Q** 8×8 at 40 dB knife, int8 and JPEG-70, on L+B (does finer grid reach converter quality?)
- **W** shot-level translation and per-tile integer translation (±4) on L/T/B
- **Wj** wtile+JPEG-70 on S06 only if wtile beats A by >0.3 dB
- **L** leftover per-frame JPEG q=40 on S06 (residual-byte ceiling)

Not overlap. Not trees. Not ALS sweep. Affine scale/rot not in this pass (translation first).

## Shots (same as t2r)

| sid | frames | t | flux | kind |
|---|---|---|---|---|
| S00 | [0,146) | 0.00–6.08s | 2.39 | locked |
| S01 | [146,256) | 6.08–10.67s | 1.60 | locked |
| S02 | [256,331) | 10.67–13.79s | 2.43 | locked |
| S03 | [331,466) | 13.79–19.42s | 5.40 | tracking **rep T** |
| S04 | [466,511) | 19.42–21.29s | 1.73 | locked |
| S05 | [511,563) | 21.29–23.46s | 1.99 | locked |
| S06 | [563,719) | 23.46–29.96s | 6.15 | tracking **rep B** |
| S07 | [719,954) | 29.96–39.75s | 1.71 | locked |
| S08 | [954,994) | 39.75–41.42s | 2.39 | locked |
| S09 | [994,1194) | 41.42–49.75s | 1.26 | locked **rep L** |
| S10 | [1194,1294) | 49.75–53.92s | 1.98 | locked |
| S11 | [1294,1504) | 53.92–62.67s | 4.14 | tracking |
| S12 | [1504,1631) | 62.67–67.96s | 2.98 | locked |
| S13 | [1631,1701) | 67.96–70.88s | 6.04 | tracking |
| S14 | [1701,2160) | 70.88–90.00s | 1.98 | locked |

Reps L=S09 T=S03 B=S06.

## Phase A — 8×8 disjoint int8, every shot

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-S00-disjoint-p8-s2` | S00 | locked | 34.42 | 27.52 | 0.938 | 3.41 | 3.13 | 0 | 3514 | 1.12 | 8.72 | 32.78 | — | 324 | 42.7 |
| `A-S01-disjoint-p8-s2` | S01 | locked | 38.91 | 34.76 | 0.974 | 1.53 | 0.81 | 0 | 755 | 1.14 | 2.49 | 36.44 | — | 251 | 24.2 |
| `A-S02-disjoint-p8-s2` | S02 | locked | 37.33 | 34.03 | 0.962 | 2.23 | 0.92 | 0 | 911 | 1.16 | 4.40 | 36.27 | — | 218 | 17.2 |
| `A-S03-disjoint-p8-s2` | S03 | tracking | 35.03 | 29.00 | 0.940 | 2.92 | 2.63 | 0 | 2790 | 1.21 | 7.49 | 33.70 | — | 302 | 37.7 |
| `A-S04-disjoint-p8-s2` | S04 | locked | 39.44 | 38.65 | 0.985 | 1.78 | 0.06 | 0 | 86 | 1.05 | 0.69 | 36.64 | — | 148 | 8.9 |
| `A-S05-disjoint-p8-s2` | S05 | locked | 37.22 | 36.55 | 0.980 | 2.40 | 0.21 | 0 | 229 | 1.02 | 1.59 | 34.87 | — | 159 | 11.2 |
| `A-S06-disjoint-p8-s2` | S06 | tracking | 33.80 | 31.48 | 0.869 | 3.81 | 4.80 | 0 | 5924 | 1.20 | 13.76 | 34.50 | — | 354 | 47.2 |
| `A-S07-disjoint-p8-s2` | S07 | locked | 36.79 | 28.14 | 0.945 | 2.66 | 2.03 | 0 | 2437 | 1.11 | 3.76 | 34.44 | — | 441 | 59.9 |
| `A-S08-disjoint-p8-s2` | S08 | locked | 37.64 | 36.40 | 0.981 | 2.10 | 0.42 | 0 | 409 | 1.08 | 3.71 | 33.28 | — | 156 | 8.7 |
| `A-S09-disjoint-p8-s2` | S09 | locked | 37.31 | 32.93 | 0.974 | 1.98 | 0.93 | 0 | 1003 | 1.12 | 1.82 | 34.78 | — | 572 | 61.0 |
| `A-S10-disjoint-p8-s2` | S10 | locked | 35.85 | 30.78 | 0.970 | 2.90 | 1.04 | 0 | 1014 | 1.04 | 3.67 | 33.42 | — | 290 | 25.4 |
| `A-S11-disjoint-p8-s2` | S11 | tracking | 35.59 | 29.89 | 0.958 | 2.19 | 2.15 | 0 | 2425 | 1.20 | 4.18 | 35.34 | — | 400 | 55.7 |
| `A-S12-disjoint-p8-s2` | S12 | locked | 34.94 | 32.58 | 0.951 | 2.83 | 1.75 | 0 | 1762 | 1.12 | 5.03 | 32.93 | — | 286 | 32.2 |
| `A-S13-disjoint-p8-s2` | S13 | tracking | 34.59 | 31.96 | 0.945 | 3.42 | 4.42 | 2 | 4245 | 1.19 | 21.97 | 33.88 | — | 220 | 19.1 |
| `A-S14-disjoint-p8-s2` | S14 | locked | 34.60 | 29.21 | 0.948 | 3.12 | 3.20 | 0 | 4981 | 1.15 | 3.93 | 33.07 | — | 765 | 117.6 |

## Phase J — 8×8 JPEG-on-B q=70, every shot

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `J-S00-disjoint-p8-jq70-Bjpeg-s2` | S00 | locked | 33.09 | 27.25 | 0.935 | 3.97 | 3.13 | 0 | 1793 | 1.16 | 4.45 | 32.87 | — | 328 | 45.3 |
| `J-S01-disjoint-p8-jq70-Bjpeg-s2` | S01 | locked | 38.10 | 34.32 | 0.974 | 1.73 | 0.81 | 0 | 352 | 1.16 | 1.16 | 36.52 | — | 312 | 25.9 |
| `J-S02-disjoint-p8-jq70-Bjpeg-s2` | S02 | locked | 36.81 | 33.65 | 0.962 | 2.36 | 0.92 | 0 | 412 | 1.19 | 1.99 | 36.22 | — | 214 | 18.8 |
| `J-S03-disjoint-p8-jq70-Bjpeg-s2` | S03 | tracking | 33.40 | 28.47 | 0.935 | 3.56 | 2.63 | 0 | 1328 | 1.29 | 3.56 | 33.42 | — | 308 | 41.4 |
| `J-S04-disjoint-p8-jq70-Bjpeg-s2` | S04 | locked | 38.97 | 38.27 | 0.985 | 1.88 | 0.06 | 0 | 60 | 1.04 | 0.48 | 36.85 | — | 174 | 9.1 |
| `J-S05-disjoint-p8-jq70-Bjpeg-s2` | S05 | locked | 36.48 | 35.91 | 0.979 | 2.64 | 0.21 | 0 | 122 | 1.01 | 0.85 | 35.03 | — | 189 | 11.3 |
| `J-S06-disjoint-p8-jq70-Bjpeg-s2` | S06 | tracking | 32.12 | 30.65 | 0.867 | 4.48 | 4.80 | 0 | 3124 | 1.41 | 7.25 | 33.56 | — | 353 | 49.9 |
| `J-S07-disjoint-p8-jq70-Bjpeg-s2` | S07 | locked | 35.24 | 27.71 | 0.943 | 3.03 | 2.03 | 0 | 1303 | 1.14 | 2.01 | 34.65 | — | 452 | 61.4 |
| `J-S08-disjoint-p8-jq70-Bjpeg-s2` | S08 | locked | 35.90 | 35.01 | 0.980 | 2.58 | 0.42 | 0 | 191 | 1.05 | 1.73 | 33.75 | — | 165 | 9.5 |
| `J-S09-disjoint-p8-jq70-Bjpeg-s2` | S09 | locked | 35.84 | 32.35 | 0.972 | 2.34 | 0.93 | 0 | 467 | 1.18 | 0.85 | 34.84 | — | 581 | 55.0 |
| `J-S10-disjoint-p8-jq70-Bjpeg-s2` | S10 | locked | 34.47 | 30.38 | 0.967 | 3.41 | 1.04 | 0 | 465 | 1.05 | 1.68 | 33.60 | — | 300 | 26.4 |
| `J-S11-disjoint-p8-jq70-Bjpeg-s2` | S11 | tracking | 34.83 | 29.21 | 0.957 | 2.42 | 2.15 | 0 | 1228 | 1.28 | 2.12 | 35.09 | — | 403 | 57.6 |
| `J-S12-disjoint-p8-jq70-Bjpeg-s2` | S12 | locked | 33.41 | 31.69 | 0.947 | 3.47 | 1.75 | 0 | 764 | 1.20 | 2.18 | 32.82 | — | 296 | 33.5 |
| `J-S13-disjoint-p8-jq70-Bjpeg-s2` | S13 | tracking | 32.45 | 30.56 | 0.942 | 4.23 | 4.42 | 2 | 1807 | 1.35 | 9.35 | 33.24 | — | 228 | 21.6 |
| `J-S14-disjoint-p8-jq70-Bjpeg-s2` | S14 | locked | 32.93 | 28.58 | 0.945 | 3.82 | 3.20 | 0 | 3172 | 1.24 | 2.50 | 32.91 | — | 760 | 120.8 |

## Phase Jq — JPEG-B q=50/90 on L+B

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Jq-S09-disjoint-p8-jq50-Bjpeg-s2` | S09 | locked | 35.25 | 32.05 | 0.972 | 2.46 | 0.93 | 0 | 444 | 1.18 | 0.80 | 34.61 | — | 390 | 47.8 |
| `Jq-S09-disjoint-p8-jq90-Bjpeg-s2` | S09 | locked | 36.29 | 32.55 | 0.973 | 2.22 | 0.93 | 0 | 531 | 1.19 | 0.96 | 34.92 | — | 389 | 47.1 |
| `Jq-S06-disjoint-p8-jq50-Bjpeg-s2` | S06 | tracking | 31.58 | 30.28 | 0.865 | 4.75 | 4.80 | 0 | 3026 | 1.41 | 7.03 | 32.91 | — | 353 | 49.2 |
| `Jq-S06-disjoint-p8-jq90-Bjpeg-s2` | S06 | tracking | 32.47 | 30.85 | 0.868 | 4.30 | 4.80 | 0 | 3375 | 1.44 | 7.84 | 33.94 | — | 354 | 49.4 |

## Phase Q — 40 dB knife

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Q-S09-disjoint-p8-t40.0-s2` | S09 | locked | 43.52 | 39.67 | 0.990 | 0.97 | 2.37 | 36 | 2401 | 1.14 | 4.35 | 35.54 | — | 530 | 58.1 |
| `Q-S09-disjoint-p8-jq70-Bjpeg-t40.0-s2` | S09 | locked | 38.28 | 36.34 | 0.988 | 1.78 | 2.37 | 36 | 1061 | 1.23 | 1.92 | 35.62 | — | 388 | 52.8 |
| `Q-S06-disjoint-p8-t40.0-s2` | S06 | tracking | 40.29 | 37.45 | 0.969 | 1.84 | 11.84 | 1045 | 14737 | 1.17 | 34.22 | 37.41 | — | 353 | 53.5 |
| `Q-S06-disjoint-p8-jq70-Bjpeg-t40.0-s2` | S06 | tracking | 35.09 | 34.31 | 0.965 | 3.03 | 11.84 | 1045 | 7766 | 1.69 | 18.03 | 35.54 | — | 415 | 57.5 |

## Phase W — translation warp then SVD

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `W-S09-wshot-p8-s2` | S09 | locked | 37.31 | 32.93 | 0.974 | 1.98 | 0.93 | 0 | 1006 | 1.12 | 1.82 | 34.78 | 0.00 | 452 | 47.9 |
| `W-S09-wtile-p8-s2` | S09 | locked | 28.41 | 25.33 | 0.921 | 3.40 | 0.86 | 0 | 1026 | 1.07 | 1.86 | 28.22 | 0.76 | 451 | 215.1 |
| `W-S03-wshot-p8-s2` | S03 | tracking | 35.01 | 26.63 | 0.939 | 2.94 | 2.67 | 0 | 2850 | 1.19 | 7.65 | 33.65 | 0.13 | 377 | 38.6 |
| `W-S03-wtile-p8-s2` | S03 | tracking | 21.44 | 15.11 | 0.728 | 9.33 | 2.79 | 0 | 3173 | 1.07 | 8.51 | 21.52 | 1.94 | 377 | 180.3 |
| `W-S06-wshot-p8-s2` | S06 | tracking | 28.05 | 22.81 | 0.862 | 5.36 | 5.09 | 2 | 6412 | 1.02 | 14.89 | 28.56 | 9.90 | 379 | 48.1 |
| `W-S06-wtile-p8-s2` | S06 | tracking | 19.84 | 15.77 | 0.523 | 13.84 | 4.58 | 0 | 6021 | 1.05 | 13.98 | 20.11 | 2.87 | 379 | 213.7 |

## Phase L — leftover JPEG ceiling on S06

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | vsH264 | bridge320 | |MV| | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `L-S06-disjoint-p8-Lq40-s2` | S06 | tracking | 36.97 | 35.38 | 0.954 | 2.73 | 4.80 | 0 | 7224 | 1.18 | 16.77 | 36.18 | — | 379 | 46.9 |

## Episode totals (all 15 shots)

- **8×8 int8:** time-weighted 35.76 dB, min 27.52, origin 32.49 MB (5.45× H.264 5.96 MB).
- **8×8 JPEG-B q70:** time-weighted 34.36 dB, min 27.25, origin 16.59 MB (2.78× H.264 5.96 MB).

## Phase E — K′ ladders

### A-S09-disjoint-p8-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 31.30 | 26.55 |
| 2 | 34.80 | 30.73 |
| 4 | 36.76 | 32.89 |
| 8 | 37.30 | 32.93 |
| 16 | 37.31 | 32.93 |

### J-S09-disjoint-p8-jq70-Bjpeg-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.99 | 26.50 |
| 2 | 33.97 | 30.43 |
| 4 | 35.46 | 32.32 |
| 8 | 35.83 | 32.35 |
| 16 | 35.84 | 32.35 |

### Q-S09-disjoint-p8-t40.0-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 31.62 | 26.64 |
| 2 | 36.02 | 31.12 |
| 4 | 40.16 | 35.49 |
| 8 | 42.69 | 39.12 |
| 16 | 43.52 | 39.67 |

## Kill sentences

- **A S09 8×8:** 37.31/32.93 origin 1003KB K=0.93.
- **A S06 8×8:** 33.80/31.48 origin 5924KB K=4.80 leftover=3.81.
- **J70 S09:** 35.84 (-1.47 dB) origin 467KB (0.47× A).
- **J70 S06:** 32.12 (-1.68 dB) origin 3124KB (0.53× A).
- **wshot S09:** 37.31/32.93 (+0.00 dB) origin 1006KB K=0.93 |MV|=0.0 leftover=1.98.
- **wtile S09:** 28.41/25.33 (-8.89 dB) origin 1026KB K=0.86 |MV|=0.764 leftover=3.40.
- **wshot S03:** 35.01/26.63 (-0.02 dB) origin 2850KB K=2.67 |MV|=0.133 leftover=2.94.
- **wtile S03:** 21.44/15.11 (-13.59 dB) origin 3173KB K=2.79 |MV|=1.942 leftover=9.33.
- **wshot S06:** 28.05/22.81 (-5.74 dB) origin 6412KB K=5.09 |MV|=9.897 leftover=5.36.
- **wtile S06:** 19.84/15.77 (-13.96 dB) origin 6021KB K=4.58 |MV|=2.871 leftover=13.84.
- **Q S09 t=40.0 B=int8:** 43.52/39.67 sat=36 origin 2401KB.
- **Q S09 t=40.0 B=jpeg:** 38.28/36.34 sat=36 origin 1061KB.
- **Q S06 t=40.0 B=int8:** 40.29/37.45 sat=1045 origin 14737KB.
- **Q S06 t=40.0 B=jpeg:** 35.09/34.31 sat=1045 origin 7766KB.
- **leftover JPEG S06:** 36.97/35.38 origin 7224KB leftoverBytes=1376350.
- **episode A:** 35.76 dB min 27.52 origin 32.49 MB.
- **episode J70:** 34.36 dB min 27.25 origin 16.59 MB (2.78× H.264).

## How to reproduce

```
python3 encoder/v4.t4r/sweep.py
```

Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.

