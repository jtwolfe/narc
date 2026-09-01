# Attempt v4.t3r — tiles, cheap seams, B-size, merge + residual trees

Branch: `attempt/v4.t3r` off `attempt/v4.t2r`.  
Frozen lab: no `src/` / `reconstruct.mp4` rewrite.

Raster: **640×360** crop, canvas 640×384. Same shots as t2r (`shots.json`).
ALS frozen at **TRAIN_STEPS=2** except a lossy-B exception if JPEG/int4 drops >0.3 dB.
H1 leaf chosen after Phase P: **8×8**. See `leaf_choice.json`.

## Reading (after the numbers)

50 rows. Lab UI / reconstruct.mp4 not rewritten. Peak RSS ~640 MB.

**8×8 is the default tile.** On every rep it beats 16×16 on mean dB *and* origin (S09 37.31/1003KB vs 36.75/1179; S06 33.80/5924 vs 33.35/7038). 4×4 is +0.6–0.8 dB at 1.4–1.5× bytes — not the leaf. 2×2 on S09 is +2.1 dB vs 16 at 2.7× bytes and seamR 1.30; it is a ceiling, not a working unit. 32 and 64 lose on tracking (S06 31.60 / 29.33). SeamR is still ~1.1–1.2 at every size that isn't so big the grid disappears.

**Decode deblock does not replace overlap.** 2px/4px mix of disjoint tiles: origin identical to A, seamR unchanged (1.12), PSNR ±0.1 dB. The grid is in the *model*, not in a 10% seam mix.

**hop-12/14 died on this canvas.** They do not divide 368×624 (H−16, W−16). Combined with a non-COLA window they reconstructed at **24.6 dB** (S09) / 28.7 (S06) — μ-ish. The factors themselves were fine (K′ overwrite ladder ~37 dB). hop-8 *does* tile and with Hann COLA matches t2r B exactly: S09 **37.82/33.55 seamR 1.05 origin 4.54 MB (3.9× A)**; S06 34.49/31.68 seamR 1.04 origin **27.9 MB (4.0×)**. Stored overlap still costs 4× for the seam win. There is no aligned hop between 8 and 16 on 640×384.

**JPEG-on-B is the size lever.** q=50: S09 **35.19 dB / 239 KB (0.20× A, −1.56 dB)**; S06 32.20 / 1.60 MB (0.23×, −1.15 dB). q=90 is 0.27–0.29× at −0.8 dB. int4-B+int8-U: S09 −0.52 dB at 0.41× — better quality than JPEG-50, twice the bytes. int4 both is a tracking tax (S06 −3.08 dB). Bases were a storage format. JPEG mosaic of eigenpatches is the first thing in this family that puts origin in the same order of magnitude as the H.264 file *per shot* (S09 239 KB vs ~8 s × 66 KB/s ≈ 530 KB of source).

**ALS is still dead, even on lossy B.** JPEG-50: 0 = 2 = 8 = 35.19 dB. int4-both on S06: 0→2 is **+0.38 dB** (the first nonzero we have seen — recovering a worse quant), 2→8 = 0. Keep steps=2. Do not train.

**H1 exclusive merge fires on locked, stalls on tracking, never goes full-frame.** S09: 85% of 8s merge to 16, 60% of those to 32, … 5×128s, 0 full. Origin 960 KB vs flat 8×8 1003 KB, **−1.1 dB**. S06: 21% merge-to-16, then almost nothing; 33.62 vs 33.80, 5780 vs 5924 KB. The 32.5 knife lets a parent be cheaper *and worse* than its children. Full-frame SVD is never cheaper. This is not the splat that gets finer — it is a partition that mostly stays at 8.

**H2 residual is flat 8×8.** 32→8 splitFrac 0.04–0.05 (tree almost never cheaper). Full→16 always collapsed (root bytes dominate). Parent-only peel = μ (24.5 dB). No preview layer.

**K′ still works** on 8×8: K′=4 is 36.76 / 31.09 (S09 / S06). Same product feature, smaller tiles.

**What to build next:** default **8×8 + JPEG-on-B** (q~70: ~0.23× origin at about −1 dB). Overlap only if the 4× byte tax is acceptable for seams; deblock is not a substitute. Stop spending on ALS, K_MAX, residual pyramids, and exclusive merge at a quality knife that lets parents get worse. Affine / leftover on tracking is still the quality hole (S06 33.8 dB, 40 dB unreachable). Episode-wide JPEG-B is the size path on more RAM.


## What this pass is

- **P** flat disjoint 2 (S09 only) / 4 / 8 / 16 / 32 / 64 on reps L=S09 T=S03 B=S06
- **S** cheap seams: recon-only 2px/4px deblock, stored overlap hop=14 and hop=12, on S09+S06
- **J** JPEG-on-B q=50/70/90, int4-B+int8-U, int4 both, on A-16 S09+S06
- **H1** bottom-up exclusive merge from the P-winner leaf up to 128 then full-frame
- **H2** residual 32→8 (per-32 pick tree vs flat) and full→16 (collapse if not cheaper)
- **E** K′ ladders on A-16, P-winner, S hop-14, H1, H2

Not C (fixed K_root=8). Not 50% OLA. Not affine. Not all 15 shots.

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

## Phase P — tile size at native (disjoint)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `P-S09-disjoint-p2-s2` | S09 | locked | 38.84 | 36.33 | 0.984 | 1.76 | 0.58 | 0 | 3167 | 1.30 | 1.04/1.03/1.04 | 35.38 | 515 | 48.7 |
| `P-S09-disjoint-p4-s2` | S09 | locked | 37.95 | 34.28 | 0.979 | 1.89 | 0.68 | 0 | 1369 | 1.11 | 1.08/1.07/1.06 | 35.09 | 494 | 44.5 |
| `P-S09-disjoint-p8-s2` | S09 | locked | 37.31 | 32.93 | 0.974 | 1.98 | 0.93 | 0 | 1003 | 1.12 | 1.12/1.10/1.09 | 34.78 | 641 | 54.3 |
| `P-S09-disjoint-p16-s2` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1179 | 1.12 | 1.07/1.12/1.11 | 34.49 | 641 | 50.4 |
| `P-S09-disjoint-p32-s2` | S09 | locked | 36.25 | 30.64 | 0.974 | 2.08 | 2.36 | 7 | 1628 | 1.13 | 1.05/1.07/1.13 | 34.13 | 501 | 42.7 |
| `P-S09-disjoint-p64-s2` | S09 | locked | 35.70 | 30.26 | 0.975 | 2.07 | 3.68 | 5 | 2208 | 1.04 | 1.03/1.04/1.06 | 33.67 | 502 | 44.3 |
| `P-S03-disjoint-p4-s2` | S03 | tracking | 35.83 | 29.95 | 0.956 | 2.70 | 1.92 | 0 | 4083 | 1.19 | 1.11/1.09/1.08 | 34.32 | 414 | 36.1 |
| `P-S03-disjoint-p8-s2` | S03 | tracking | 35.03 | 29.00 | 0.940 | 2.92 | 2.63 | 0 | 2790 | 1.21 | 1.21/1.16/1.15 | 33.70 | 412 | 38.5 |
| `P-S03-disjoint-p16-s2` | S03 | tracking | 34.48 | 28.00 | 0.945 | 3.07 | 4.07 | 3 | 3139 | 1.22 | 1.11/1.22/1.21 | 33.14 | 412 | 32.9 |
| `P-S03-disjoint-p32-s2` | S03 | tracking | 33.77 | 27.50 | 0.947 | 3.27 | 6.67 | 8 | 4494 | 1.24 | 1.05/1.11/1.24 | 32.51 | 412 | 34.5 |
| `P-S03-disjoint-p64-s2` | S03 | tracking | 32.54 | 27.45 | 0.947 | 3.57 | 10.20 | 16 | 6201 | 1.04 | 1.02/1.04/1.10 | 31.44 | 412 | 33.8 |
| `P-S06-disjoint-p4-s2` | S06 | tracking | 34.44 | 32.53 | 0.904 | 3.43 | 3.06 | 0 | 8672 | 1.24 | 1.13/1.10/1.09 | 34.89 | 444 | 42.2 |
| `P-S06-disjoint-p8-s2` | S06 | tracking | 33.80 | 31.48 | 0.869 | 3.81 | 4.80 | 0 | 5924 | 1.20 | 1.20/1.15/1.14 | 34.50 | 553 | 53.2 |
| `P-S06-disjoint-p16-s2` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7038 | 1.17 | 1.09/1.17/1.16 | 33.92 | 553 | 48.1 |
| `P-S06-disjoint-p32-s2` | S06 | tracking | 31.60 | 29.06 | 0.839 | 4.83 | 11.97 | 116 | 8148 | 1.17 | 1.04/1.08/1.17 | 32.03 | 443 | 40.3 |
| `P-S06-disjoint-p64-s2` | S06 | tracking | 29.33 | 27.01 | 0.819 | 5.92 | 13.72 | 42 | 8371 | 1.03 | 1.02/1.03/1.07 | 29.64 | 443 | 40.6 |

## Phase S — cheap seams (deblock vs stored overlap)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `S-S09-blend-p16-b2-s2` | S09 | locked | 36.81 | 31.85 | 0.975 | 2.05 | 1.47 | 5 | 1179 | 1.12 | 1.07/1.12/1.11 | 34.58 | 461 | 44.8 |
| `S-S09-blend-p16-b4-s2` | S09 | locked | 36.73 | 31.85 | 0.974 | 2.08 | 1.47 | 5 | 1179 | 1.10 | 1.05/1.10/1.09 | 34.60 | 460 | 45.5 |
| `S-S09-overlap-p16-h14-s2` | S09 | locked | 24.57 | 24.11 | 0.932 | 3.06 | 1.43 | 3 | 1535 | 0.82 | 0.79/0.82/0.84 | 26.99 | 601 | 54.4 |
| `S-S09-overlap-p16-h12-s2` | S09 | locked | 24.60 | 24.22 | 0.934 | 3.00 | 1.47 | 8 | 2064 | 0.83 | 0.79/0.83/0.84 | 27.00 | 460 | 50.6 |
| `S-S06-blend-p16-b2-s2` | S06 | tracking | 33.48 | 30.90 | 0.865 | 4.03 | 8.66 | 92 | 7038 | 1.12 | 1.06/1.12/1.11 | 34.03 | 423 | 43.1 |
| `S-S06-blend-p16-b4-s2` | S06 | tracking | 33.40 | 30.87 | 0.864 | 4.06 | 8.66 | 92 | 7038 | 1.12 | 1.06/1.12/1.11 | 34.00 | 426 | 44.5 |
| `S-S06-overlap-p16-h14-s2` | S06 | tracking | 28.69 | 26.29 | 0.839 | 4.45 | 8.58 | 137 | 9274 | 0.95 | 0.93/0.95/0.96 | 30.80 | 429 | 48.6 |
| `S-S06-overlap-p16-h12-s2` | S06 | tracking | 28.81 | 26.40 | 0.844 | 4.31 | 8.74 | 177 | 12524 | 0.95 | 0.94/0.95/0.96 | 30.84 | 430 | 53.5 |
| `S-S09-overlap-p16-h8-s2` | S09 | locked | 37.82 | 33.55 | 0.979 | 1.91 | 1.50 | 16 | 4541 | 1.05 | 1.05/1.05/1.05 | 35.03 | 532 | 71.2 |
| `S-S06-overlap-p16-h8-s2` | S06 | tracking | 34.49 | 31.68 | 0.883 | 3.63 | 8.83 | 387 | 27862 | 1.04 | 1.05/1.04/1.03 | 34.92 | 421 | 82.7 |

## Phase J — JPEG-on-B / int4 mixed

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `J-S09-disjoint-p16-jq50-Bjpeg-s2` | S09 | locked | 35.19 | 31.17 | 0.972 | 2.48 | 1.47 | 5 | 239 | 1.19 | 1.11/1.19/1.18 | 34.65 | 467 | 43.7 |
| `J-S09-disjoint-p16-jq70-Bjpeg-s2` | S09 | locked | 35.55 | 31.31 | 0.972 | 2.39 | 1.47 | 5 | 261 | 1.19 | 1.11/1.19/1.18 | 34.63 | 465 | 42.9 |
| `J-S09-disjoint-p16-jq90-Bjpeg-s2` | S09 | locked | 35.93 | 31.44 | 0.973 | 2.28 | 1.47 | 5 | 324 | 1.20 | 1.12/1.20/1.19 | 34.69 | 464 | 42.7 |
| `J-S09-disjoint-p16-Bint4-s2` | S09 | locked | 36.23 | 31.49 | 0.970 | 2.23 | 1.47 | 5 | 481 | 1.10 | 1.06/1.10/1.10 | 34.38 | 464 | 43.3 |
| `J-S09-disjoint-p16-Bint4-Uint4-s2` | S09 | locked | 35.45 | 31.18 | 0.968 | 2.53 | 1.47 | 5 | 395 | 1.08 | 1.05/1.08/1.08 | 33.87 | 463 | 43.0 |
| `J-S06-disjoint-p16-jq50-Bjpeg-s2` | S06 | tracking | 32.20 | 30.31 | 0.857 | 4.53 | 8.66 | 92 | 1602 | 1.37 | 1.20/1.37/1.32 | 33.59 | 427 | 43.0 |
| `J-S06-disjoint-p16-jq70-Bjpeg-s2` | S06 | tracking | 32.35 | 30.38 | 0.859 | 4.46 | 8.66 | 92 | 1713 | 1.38 | 1.20/1.38/1.33 | 33.66 | 427 | 42.9 |
| `J-S06-disjoint-p16-jq90-Bjpeg-s2` | S06 | tracking | 32.51 | 30.45 | 0.860 | 4.38 | 8.66 | 92 | 2028 | 1.39 | 1.21/1.39/1.34 | 33.72 | 427 | 43.6 |
| `J-S06-disjoint-p16-Bint4-s2` | S06 | tracking | 31.96 | 30.19 | 0.840 | 4.83 | 8.66 | 92 | 3109 | 1.12 | 1.06/1.12/1.11 | 33.17 | 427 | 42.0 |
| `J-S06-disjoint-p16-Bint4-Uint4-s2` | S06 | tracking | 30.27 | 28.95 | 0.828 | 5.99 | 8.66 | 92 | 2270 | 1.09 | 1.05/1.09/1.08 | 31.10 | 430 | 41.4 |

## Phase Tj — ALS unfreeze on lossy B (only if J dropped >0.3 dB)

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `Tj-S06-disjoint-p16-Bint4-Uint4-s0` | S06 | tracking | 29.89 | 28.75 | 0.825 | 6.29 | 8.66 | 92 | 2282 | 1.09 | 1.04/1.09/1.08 | 30.65 | 430 | 32.9 |
| `Tj-S06-disjoint-p16-Bint4-Uint4-s2` | S06 | tracking | 30.27 | 28.95 | 0.828 | 5.99 | 8.66 | 92 | 2270 | 1.09 | 1.05/1.09/1.08 | 31.10 | 430 | 41.2 |
| `Tj-S06-disjoint-p16-Bint4-Uint4-s8` | S06 | tracking | 30.27 | 28.95 | 0.828 | 5.99 | 8.66 | 92 | 2271 | 1.09 | 1.05/1.09/1.08 | 31.10 | 430 | 71.8 |
| `Tj-S09-disjoint-p16-jq50-Bjpeg-s0` | S09 | locked | 35.19 | 31.18 | 0.972 | 2.48 | 1.47 | 5 | 240 | 1.19 | 1.11/1.19/1.18 | 34.65 | 466 | 40.3 |
| `Tj-S09-disjoint-p16-jq50-Bjpeg-s2` | S09 | locked | 35.19 | 31.17 | 0.972 | 2.48 | 1.47 | 5 | 239 | 1.19 | 1.11/1.19/1.18 | 34.65 | 465 | 42.4 |
| `Tj-S09-disjoint-p16-jq50-Bjpeg-s8` | S09 | locked | 35.19 | 31.17 | 0.972 | 2.48 | 1.47 | 5 | 239 | 1.19 | 1.11/1.19/1.18 | 34.65 | 464 | 51.3 |

## Phase H1 — exclusive bottom-up merge

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `H1-S09-merge-p8-s2` | S09 | locked | 36.20 | 30.21 | 0.973 | 2.13 | 2.84 | 0 | 960 | 1.08 | 1.08/1.10/1.11 | 34.35 | 590 | 77.2 |
| `H1-S03-merge-p8-s2` | S03 | tracking | 34.60 | 28.10 | 0.940 | 3.05 | 2.93 | 0 | 2701 | 1.17 | 1.17/1.18/1.17 | 33.35 | 358 | 75.9 |
| `H1-S06-merge-p8-s2` | S06 | tracking | 33.62 | 31.28 | 0.868 | 3.90 | 5.38 | 0 | 5780 | 1.17 | 1.17/1.15/1.14 | 34.32 | 426 | 98.4 |

## Phase H2 — residual 32→8 and full→16, rate-constrained

| id | shot | kind | mean dB | min dB | SSIM | leftover | mean K | sat | origin KB | seamR | seam 8/16/32 | bridge320 | RSS | s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `H2-S09-res32-p8-s2` | S09 | locked | 37.33 | 32.92 | 0.974 | 1.96 | 0.89 | 0 | 998 | 1.12 | 1.12/1.10/1.09 | 34.76 | 608 | 60.8 |
| `H2-S03-res32-p8-s2` | S03 | tracking | 35.04 | 28.99 | 0.940 | 2.92 | 2.56 | 0 | 2741 | 1.21 | 1.21/1.16/1.15 | 33.69 | 384 | 48.0 |
| `H2-S06-res32-p8-s2` | S06 | tracking | 33.77 | 31.44 | 0.869 | 3.83 | 4.73 | 0 | 5827 | 1.19 | 1.19/1.15/1.14 | 34.47 | 431 | 69.3 |
| `H2-S09-resfull-p16-s2` | S09 | locked | 36.75 | 31.70 | 0.974 | 2.04 | 1.47 | 5 | 1179 | 1.12 | 1.07/1.12/1.11 | 34.49 | 637 | 59.6 |
| `H2-S06-resfull-p16-s2` | S06 | tracking | 33.35 | 30.79 | 0.862 | 4.08 | 8.66 | 92 | 7038 | 1.17 | 1.09/1.17/1.16 | 33.92 | 434 | 55.4 |

## Phase E — K′ ladders

### P-S09-disjoint-p8-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 31.30 | 26.55 |
| 2 | 34.80 | 30.73 |
| 4 | 36.76 | 32.89 |
| 8 | 37.30 | 32.93 |
| 16 | 37.31 | 32.93 |

### P-S09-disjoint-p16-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.36 | 25.55 |
| 2 | 33.05 | 27.82 |
| 4 | 35.28 | 30.36 |
| 8 | 36.43 | 31.57 |
| 16 | 36.75 | 31.70 |

### P-S06-disjoint-p8-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 15.37 | 11.90 |
| 1 | 24.13 | 22.43 |
| 2 | 27.08 | 25.37 |
| 4 | 31.09 | 28.96 |
| 8 | 33.66 | 31.30 |
| 16 | 33.80 | 31.48 |

### P-S06-disjoint-p16-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 15.37 | 11.90 |
| 1 | 22.19 | 20.02 |
| 2 | 24.79 | 22.57 |
| 4 | 28.08 | 26.24 |
| 8 | 31.54 | 29.06 |
| 16 | 33.35 | 30.79 |

### S-S09-overlap-p16-h14-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.43 | 25.56 |
| 2 | 33.21 | 28.13 |
| 4 | 35.48 | 30.83 |
| 8 | 36.60 | 31.81 |
| 16 | 36.92 | 31.99 |

### H1-S09-merge-p8-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.71 | 26.07 |
| 2 | 33.71 | 29.04 |
| 4 | 35.64 | 30.16 |
| 8 | 36.19 | 30.21 |
| 16 | 36.20 | 30.21 |

### H2-S09-res32-p8-s2

| K′ | mean dB | min dB |
|---|---|---|
| p | 24.58 | 22.13 |
| 0 | 24.53 | 22.11 |
| 1 | 31.39 | 26.57 |
| 2 | 34.81 | 30.74 |
| 4 | 36.78 | 32.88 |
| 8 | 37.32 | 32.92 |
| 16 | 37.33 | 32.92 |

### H2-S09-resfull-p16-s2

| K′ | mean dB | min dB |
|---|---|---|
| p | 24.53 | 22.11 |
| 0 | 24.53 | 22.11 |
| 1 | 30.36 | 25.55 |
| 2 | 33.05 | 27.82 |
| 4 | 35.28 | 30.36 |
| 8 | 36.43 | 31.57 |
| 16 | 36.75 | 31.70 |

### S-S09-overlap-p16-h8-s2

| K′ | mean dB | min dB |
|---|---|---|
| 0 | 24.53 | 22.11 |
| 1 | 30.29 | 25.45 |
| 2 | 33.01 | 27.77 |
| 4 | 35.29 | 30.41 |
| 8 | 36.43 | 31.65 |
| 16 | 36.73 | 31.73 |

## Kill sentences

- **S09 tiles:** 2=38.84dB/3167KB seamR=1.30; 4=37.95dB/1369KB seamR=1.11; 8=37.31dB/1003KB seamR=1.12; 16=36.75dB/1179KB seamR=1.12; 32=36.25dB/1628KB seamR=1.13; 64=35.70dB/2208KB seamR=1.04.
- **S03 tiles:** 4=35.83dB/4083KB seamR=1.19; 8=35.03dB/2790KB seamR=1.21; 16=34.48dB/3139KB seamR=1.22; 32=33.77dB/4494KB seamR=1.24; 64=32.54dB/6201KB seamR=1.04.
- **S06 tiles:** 4=34.44dB/8672KB seamR=1.24; 8=33.80dB/5924KB seamR=1.20; 16=33.35dB/7038KB seamR=1.17; 32=31.60dB/8148KB seamR=1.17; 64=29.33dB/8371KB seamR=1.03.
- **S09 deblock 2px:** seamR 1.12 vs A 1.12 origin 1179 vs 1179KB (should match A).
- **S09 deblock 4px:** seamR 1.10 min 31.85 vs A 31.70.
- **S09 hop14:** seamR 0.82 origin 1535KB (1.30× A).
- **S09 hop12:** seamR 0.83 origin 2064KB (1.75× A).
- **S09 hop8 (aligned Hann):** 37.82/33.55 seamR 1.05 origin 4541KB (3.9× A) — matches t2r B.
- **S06 deblock 2px:** seamR 1.12 vs A 1.17 origin 7038 vs 7038KB (should match A).
- **S06 deblock 4px:** seamR 1.12 min 30.87 vs A 30.79.
- **S06 hop14:** seamR 0.95 origin 9274KB (1.32× A).
- **S06 hop12:** seamR 0.95 origin 12524KB (1.78× A).
- **S06 hop8 (aligned Hann):** 34.49/31.68 seamR 1.04 origin 27862KB (4.0× A) — matches t2r B.
- **S09 jpeg50:** 35.19 (-1.56 dB) origin 239KB (0.20× A).
- **S09 jpeg70:** 35.55 (-1.20 dB) origin 261KB (0.22× A).
- **S09 jpeg90:** 35.93 (-0.82 dB) origin 324KB (0.27× A).
- **S09 int4:** 36.23 (-0.52 dB) origin 481KB (0.41× A).
- **S09 int4:** 35.45 (-1.30 dB) origin 395KB (0.34× A).
- **S06 jpeg50:** 32.20 (-1.15 dB) origin 1602KB (0.23× A).
- **S06 jpeg70:** 32.35 (-1.00 dB) origin 1713KB (0.24× A).
- **S06 jpeg90:** 32.51 (-0.84 dB) origin 2028KB (0.29× A).
- **S06 int4:** 31.96 (-1.39 dB) origin 3109KB (0.44× A).
- **S06 int4:** 30.27 (-3.08 dB) origin 2270KB (0.32× A).
- **ALS on lossy B Tj-S06-disjoint-p16-Bint4-Uint4-s0:** 29.89 dB steps=0.
- **ALS on lossy B Tj-S06-disjoint-p16-Bint4-Uint4-s2:** 30.27 dB steps=2.
- **ALS on lossy B Tj-S06-disjoint-p16-Bint4-Uint4-s8:** 30.27 dB steps=8.
- **ALS on lossy B Tj-S09-disjoint-p16-jq50-Bjpeg-s0:** 35.19 dB steps=0.
- **ALS on lossy B Tj-S09-disjoint-p16-jq50-Bjpeg-s2:** 35.19 dB steps=2.
- **ALS on lossy B Tj-S09-disjoint-p16-jq50-Bjpeg-s8:** 35.19 dB steps=8.
- **H1 S09 leaf=8:** 36.20/30.21 origin 960KB n=789 sizes={'128': 5, '64': 11, '16': 236, '8': 512, '32': 25} mergeFrac={'16': 0.8458333333333333, '32': 0.5958333333333333, '64': 0.48333333333333334, '128': 0.3333333333333333, 'full': 0.0} seamR=1.08.
- **H1 S03 leaf=8:** 34.60/28.10 origin 2701KB n=2577 sizes={'16': 396, '8': 2176, '32': 5} mergeFrac={'16': 0.43125, '32': 0.020833333333333332, '64': 0.0, '128': 0.0, 'full': 0.0} seamR=1.17.
- **H1 S06 leaf=8:** 33.62/31.28 origin 5780KB n=3174 sizes={'8': 3008, '16': 156, '32': 9, '64': 1} mergeFrac={'16': 0.2125, '32': 0.05416666666666667, '64': 0.016666666666666666, '128': 0.0, 'full': 0.0} seamR=1.17.
- **H2 S09 res32:** 37.33/32.92 origin 998KB splitFrac=0.0375 collapsed=None kRoot=0 seamR=1.12.
- **H2 S03 res32:** 35.04/28.99 origin 2741KB splitFrac=0.05 collapsed=None kRoot=0 seamR=1.21.
- **H2 S06 res32:** 33.77/31.44 origin 5827KB splitFrac=0.05416666666666667 collapsed=None kRoot=0 seamR=1.19.
- **H2 S09 resfull:** 36.75/31.70 origin 1179KB splitFrac=None collapsed=True kRoot=0 seamR=1.12.
- **H2 S06 resfull:** 33.35/30.79 origin 7038KB splitFrac=None collapsed=True kRoot=0 seamR=1.17.

## How to reproduce

```
python3 encoder/v4.t3r/sweep.py
```

Does not touch `src/` or `public/media/reconstruct.mp4`. Resumes from `results.jsonl`.

