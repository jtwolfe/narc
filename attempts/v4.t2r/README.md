# Attempt v4.t2r — native 640×360, per-shot, three representations

Branch: `attempt/v4.t2r` off `attempt/v4.t1r`.  
Frozen lab: no `src/` / `reconstruct.mp4` rewrite.

Raster: **640×360** crop, canvas 640×384 (6.7% pad, same ratio as 180→192). Source file is already 640×360 24 fps. Frames: `/tmp/bbb/frames-640` PNG (H.264 decode, no second JPEG).

ALS: measured at 640 before freezing. Chosen **TRAIN_STEPS=2**. See `train_choice.json`.

## Reading (after the numbers)

41 rows. Lab UI / reconstruct.mp4 not rewritten. Peak RSS 1.4 GB on the 19s locked shot and the merge; M2 first try OOM'd on float recon and was rerun uint8 (672 MB).

**TRAIN_STEPS=0 was measured, not assumed.** Locked and tracking at 640: 0 vs 2 vs 8 = **+0.001 dB**. t1r holds. Kept steps=2 for causal vs v4r.

**Native 16×16 (arm A, all 15 shots):** time-weighted **35.22 dB / min 27.07**, SSIM 0.86–0.99, origin **34.9 MB** vs source H.264 **6.8 MB** (~5×). Bridge PSNR to the old 320 raster is ~1–3 dB below native — we are scoring a harder picture, not a weaker model. Tracking shots are the hole: S06 33.35 dB, K=8.66, sat 92, 7.0 MB for 6.5s. Locked S09 36.75 dB, 1.1 MB for 8.3s. No flux>8 busy shot exists in this 90s.

**Overlap (B):** seamR **1.04–1.05** vs A **1.12–1.22**. Min PSNR +1–2 dB. Origin **×4**. The grid is real and OLA moves it. It is not a size move.

**Global tree (C):** K_root=8 is not rate-controlled. On locked S09 it **overshoots** to 40.16 dB at 2.4× A's bytes; min PSNR barely moves (31.88 vs 31.70) and seamR **gets worse** (1.23). On tracking S03/S06 it is +0.12 / +0.28 dB at 1.9× / 1.3× bytes. Shared coarse bases do not amortize this clip at the 32.5 knife. They buy mean dB when the shot is already easy.

**40 dB knife:** locked S09 and tracking S03 **hit 40** on A (42.7 / 40.8). Busiest S06 **does not**: A 36.14, B 37.26, C 37.25, sat 742/960 at K_MAX=16. Converter-quality on motion is not a knife tweak.

**K_MAX=32** on S06: +0.11 dB, sat 92→0, origin +2%. Same t1r story.

**YUV 4:2:0** on S09: **−2.24 dB**, origin **0.35×**. Size yes, "close to source" no.

**Episode proxy S06+S07 (16.3s, a real cut):**
- separate A: time-weighted **35.22 dB**, 9.56 MB
- M1 naive one-μ one-SVD: **34.85 / 28.57**, 6.89 MB (−0.37 dB, −28% bytes)
- M2 shared B, per-shot μ/U: **34.75 / 28.59**, 6.72 MB
Ignoring the cut (or sharing B across it) saves bytes because one eigenpatch spans both shots. Quality drops about a third of a dB on this pair. An episode-scale shared B on a machine with more RAM is the same bet, larger: watch the cut, keep per-shot μ.

**vs the file you'd convert:** 35 dB at native 640×360 is the old 32 dB-gate theme, now on the real raster. It is still **5× the H.264 file** and the min is still 27 dB. Overlap is the seam fix. The tree as specified is not the size fix. JPEG-on-B / shared-B *within* a shot (not just across two) is still the bytes path.


## Shots (re-detected at 640, cutHist=0.62, flux>12)

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

Reps: L=S09 (lowest flux locked), T=S03 (median of rest), B=S06 (highest flux — no flux>8 busy shot in this 90s).
Merge pair: S06+S07 (busiest + neighbor).

Arms: **A** disjoint 16×16 · **B** hop-8 Hann OLA · **C** global K_root=8 then 16×16 leftover.
Phase Q is the same three arms at a **40 dB** knife. M1 = naive concatenated SVD. M2 = shared B, per-shot μ/U.

## Phase T — ALS at 640 (do not trust t1r steps=0)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `T-S09-A-s0-k16-t32.5-h16-rgb` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1144 | 1.12 | 34.49 | 643 | 36.3 |
| `T-S09-A-s2-k16-t32.5-h16-rgb` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1144 | 1.12 | 34.49 | 645 | 38.1 |
| `T-S09-A-s8-k16-t32.5-h16-rgb` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1144 | 1.12 | 34.49 | 645 | 46.4 |
| `T-S06-A-s0-k16-t32.5-h16-rgb` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7016 | 1.17 | 33.92 | 569 | 29.7 |
| `T-S06-A-s2-k16-t32.5-h16-rgb` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7015 | 1.17 | 33.92 | 567 | 39.1 |
| `T-S06-A-s8-k16-t32.5-h16-rgb` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7015 | 1.17 | 33.92 | 568 | 63.8 |

## Phase A — disjoint 16×16, every shot, 32.5 dB

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `A-S00-A-s2-k16-t32.5-h16-rgb` | S00 | locked | 34.02 | 27.07 | 0.943 | 3.46 | 4.48 | 20 | 3456 | 1.12 | 32.47 | 523 | 33.5 |
| `A-S01-A-s2-k16-t32.5-h16-rgb` | S01 | locked | 38.08 | 33.67 | 0.976 | 1.65 | 1.14 | 0 | 774 | 1.14 | 36.12 | 416 | 20.8 |
| `A-S02-A-s2-k16-t32.5-h16-rgb` | S02 | locked | 36.70 | 33.48 | 0.968 | 2.38 | 1.37 | 2 | 1005 | 1.16 | 35.97 | 333 | 14.5 |
| `A-S03-A-s2-k16-t32.5-h16-rgb` | S03 | tracking | 34.48 | 28.00 | 0.945 | 3.07 | 4.07 | 3 | 3100 | 1.22 | 33.14 | 491 | 30.1 |
| `A-S04-A-s2-k16-t32.5-h16-rgb` | S04 | locked | 38.98 | 38.18 | 0.985 | 1.85 | 0.05 | 0 | 34 | 1.03 | 36.79 | 237 | 7.9 |
| `A-S05-A-s2-k16-t32.5-h16-rgb` | S05 | locked | 36.65 | 35.96 | 0.979 | 2.53 | 0.24 | 0 | 161 | 1.02 | 34.90 | 257 | 9.3 |
| `A-S06-A-s2-k16-t32.5-h16-rgb` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7015 | 1.17 | 33.92 | 569 | 38.5 |
| `A-S07-A-s2-k16-t32.5-h16-rgb` | S07 | locked | 36.46 | 27.67 | 0.944 | 2.73 | 3.17 | 21 | 2543 | 1.12 | 34.12 | 778 | 49.9 |
| `A-S08-A-s2-k16-t32.5-h16-rgb` | S08 | locked | 37.24 | 35.54 | 0.981 | 2.10 | 0.51 | 0 | 350 | 1.08 | 33.21 | 216 | 7.5 |
| `A-S09-A-s2-k16-t32.5-h16-rgb` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1144 | 1.12 | 34.49 | 660 | 38.0 |
| `A-S10-A-s2-k16-t32.5-h16-rgb` | S10 | locked | 35.41 | 29.60 | 0.970 | 3.01 | 1.35 | 10 | 986 | 1.04 | 33.34 | 426 | 19.7 |
| `A-S11-A-s2-k16-t32.5-h16-rgb` | S11 | tracking | 34.89 | 29.07 | 0.958 | 2.39 | 3.62 | 10 | 2829 | 1.22 | 34.74 | 688 | 44.9 |
| `A-S12-A-s2-k16-t32.5-h16-rgb` | S12 | locked | 34.34 | 31.60 | 0.953 | 2.93 | 2.76 | 5 | 2100 | 1.13 | 32.58 | 467 | 26.7 |
| `A-S13-A-s2-k16-t32.5-h16-rgb` | S13 | tracking | 34.15 | 31.08 | 0.940 | 3.62 | 7.08 | 94 | 4965 | 1.16 | 33.48 | 321 | 17.0 |
| `A-S14-A-s2-k16-t32.5-h16-rgb` | S14 | locked | 34.05 | 28.38 | 0.949 | 3.23 | 4.94 | 47 | 4442 | 1.15 | 32.64 | 1438 | 112.2 |

## Phase B — 16×16 overlap-add, reps

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `B-S09-B-s2-k16-t32.5-h8-rgb` | S09 | locked | 37.82 | 33.55 | 0.979 | 1.91 | 1.50 | 16 | 4496 | 1.05 | 35.03 | 671 | 56.4 |
| `B-S03-B-s2-k16-t32.5-h8-rgb` | S03 | tracking | 36.11 | 30.34 | 0.963 | 2.63 | 4.12 | 18 | 12174 | 1.05 | 34.22 | 548 | 53.8 |
| `B-S06-B-s2-k16-t32.5-h8-rgb` | S06 | tracking | 34.49 | 31.68 | 0.883 | 3.63 | 8.83 | 387 | 27822 | 1.04 | 34.92 | 637 | 78.8 |

## Phase C — global split tree, reps

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `C-S09-C-s2-k16-t32.5-h16-rgb` | S09 | locked | 40.16 | 31.88 | 0.985 | 0.88 | 0.99 | 0 | 2731 | 1.23 | 34.19 | 782 | 49.8 |
| `C-S03-C-s2-k16-t32.5-h16-rgb` | S03 | tracking | 34.60 | 28.72 | 0.948 | 3.03 | 2.91 | 3 | 6025 | 1.20 | 33.01 | 635 | 36.6 |
| `C-S06-C-s2-k16-t32.5-h16-rgb` | S06 | tracking | 33.63 | 30.89 | 0.879 | 3.91 | 6.40 | 32 | 9222 | 1.15 | 33.87 | 715 | 44.9 |

## Phase Q — 40 dB knife, reps × A/B/C

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Q-S09-A-s2-k16-t40.0-h16-rgb` | S09 | locked | 42.74 | 38.85 | 0.989 | 0.98 | 3.54 | 51 | 2664 | 1.16 | 35.41 | 787 | 41.6 |
| `Q-S09-B-s2-k16-t40.0-h8-rgb` | S09 | locked | 43.88 | 40.82 | 0.992 | 0.91 | 3.63 | 214 | 10545 | 1.10 | 35.57 | 787 | 71.9 |
| `Q-S09-C-s2-k16-t40.0-h16-rgb` | S09 | locked | 45.16 | 39.06 | 0.993 | 0.51 | 2.60 | 34 | 4002 | 1.26 | 35.37 | 828 | 53.1 |
| `Q-S03-A-s2-k16-t40.0-h16-rgb` | S03 | tracking | 40.76 | 36.15 | 0.983 | 1.50 | 8.43 | 68 | 6412 | 1.22 | 35.80 | 576 | 33.0 |
| `Q-S03-B-s2-k16-t40.0-h8-rgb` | S03 | tracking | 42.21 | 38.41 | 0.988 | 1.30 | 8.56 | 265 | 25322 | 1.09 | 36.22 | 576 | 64.9 |
| `Q-S03-C-s2-k16-t40.0-h16-rgb` | S03 | tracking | 41.01 | 36.48 | 0.983 | 1.46 | 6.92 | 41 | 9153 | 1.23 | 35.69 | 668 | 39.5 |
| `Q-S06-A-s2-k16-t40.0-h16-rgb` | S06 | tracking | 36.14 | 33.04 | 0.930 | 2.85 | 14.85 | 742 | 11894 | 1.19 | 35.86 | 641 | 41.9 |
| `Q-S06-B-s2-k16-t40.0-h8-rgb` | S06 | tracking | 37.26 | 33.89 | 0.943 | 2.52 | 14.92 | 2918 | 46614 | 1.07 | 36.61 | 782 | 96.1 |
| `Q-S06-C-s2-k16-t40.0-h16-rgb` | S06 | tracking | 37.25 | 34.14 | 0.945 | 2.53 | 13.92 | 646 | 15226 | 1.17 | 36.26 | 806 | 50.2 |

## Phase K — K_MAX=32 smoke on busiest, arm A

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `K-S06-A-s2-k32-t32.5-h16-rgb` | S06 | tracking | 33.46 | 30.96 | 0.863 | 4.04 | 8.86 | 0 | 7178 | 1.17 | 34.00 | 737 | 38.5 |

## Phase Y — YUV 4:2:0 smoke, arm A locked

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Y-S09-A-s2-k16-t32.5-h16-yuv` | S09 | locked | 34.52 | 29.46 | 0.968 | 2.56 | 0.94 | 0 | 403 | 1.10 | 33.96 | 860 | 38.6 |

## Phase M1 — naive merge of two adjacent shots (ignore the cut)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `M1-S06+S07-A-s2-k16-t32.5-h16-rgb` | S06+S07 | merge-cut | 34.85 | 28.57 | 0.890 | 3.47 | 7.72 | 96 | 6893 | 1.18 | 33.52 | 1402 | 98.6 |
| `M1-S06+S07-C-s2-k16-t32.5-h16-rgb` | S06+S07 | merge-cut | 34.92 | 28.73 | 0.902 | 3.44 | 5.91 | 56 | 9322 | 1.15 | 33.47 | 1420 | 120.2 |

## Phase M2 — shared spatial B across two shots (episode-bases proxy)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `M2-S06+S07-A-s2-k16-t32.5-h16-rgb` | S06+S07 | shared-B | 34.75 | 28.59 | 0.891 | 3.49 | 7.45 | 92 | 6718 | 1.17 | 33.53 | 672 | 95.4 |

## Phase E — K′ ladders (32.5 origins)

### A-S03-A-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 16.25 | 11.18 |
| 1 | 24.04 | 19.44 |
| 2 | 28.19 | 23.15 |
| 4 | 32.05 | 26.91 |
| 8 | 34.18 | 27.99 |
| 16 | 34.48 | 28.00 |

### A-S06-A-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 15.37 | 11.90 |
| 1 | 22.19 | 20.02 |
| 2 | 24.79 | 22.57 |
| 4 | 28.08 | 26.24 |
| 8 | 31.54 | 29.06 |
| 16 | 33.35 | 30.79 |

### A-S09-A-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.36 | 25.55 |
| 2 | 33.05 | 27.82 |
| 4 | 35.28 | 30.36 |
| 8 | 36.43 | 31.57 |
| 16 | 36.75 | 31.70 |

### B-S09-B-s2-k16-t32.5-h8-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 31.25 | 26.21 |
| 2 | 34.34 | 29.62 |
| 4 | 36.46 | 32.35 |
| 8 | 37.52 | 33.44 |
| 16 | 37.82 | 33.55 |

### C-S09-C-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| full | 40.16 | 31.88 |

### B-S03-B-s2-k16-t32.5-h8-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 16.25 | 11.18 |
| 1 | 25.46 | 21.57 |
| 2 | 30.24 | 25.70 |
| 4 | 33.54 | 29.01 |
| 8 | 35.77 | 30.32 |
| 16 | 36.11 | 30.34 |

### C-S03-C-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| full | 34.60 | 28.72 |

### B-S06-B-s2-k16-t32.5-h8-rgb

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 15.37 | 11.90 |
| 1 | 23.89 | 22.25 |
| 2 | 26.23 | 24.58 |
| 4 | 29.30 | 27.16 |
| 8 | 32.68 | 29.85 |
| 16 | 34.49 | 31.68 |

### C-S06-C-s2-k16-t32.5-h16-rgb

| K′ | mean dB | min dB |
|---|---|---|
| full | 33.63 | 30.89 |

## Kill sentences

- **Training:** 0 vs 8 steps = +0.001 dB at 640, locked and tracking. t1r stands. Suspicion noted and answered.
- **Raster:** A on 15 shots weighted 35.22 / min 27.07. Native 16×16 carries the theme. Bridge to 320 is lower because the picture has 4× the pixels, not because the model collapsed.
- **Overlap:** seamR 1.04–1.05 vs 1.12–1.22. It is the grid fix. Origin ×4. Do not ship as the default; window the finest leaves if the block look is the demo problem.
- **Tree:** not a size win at 32.5. Locked overshoot (40 dB / 2.4× bytes, min unchanged). Tracking +0.1–0.3 dB at 1.3–1.9×. SeamR does not fall. K_root=8 needs a rate constraint before this is an architecture.
- **40 dB:** S09 and S03 yes on A; S06 no (36.1 dB, 742 sat). "As close as the source" fails on the busiest shot with this rank cap.
- **K_MAX=32:** +0.11 dB on S06. Dead as a quality knob.
- **YUV:** −2.2 dB / 0.35×. Size-only.
- **Busy window:** none (max flux 6.15). Affine still unmeasured.
- **Episode (2-shot):** sharing B across a cut is −0.37 dB and −28% bytes vs two independent shots. Viable on more RAM; keep per-shot μ (M2 ≈ M1 here because the cut is inside the subspace either way if B is joint).

## How to reproduce

```
python3 encoder/v4.t2r/sweep.py
```

Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.

