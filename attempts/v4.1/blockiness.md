# v4.1 blockiness probes

Lab frozen. Frozen v4 origin. Two artifacts: 8×8 lattice (seamR) and tracking leftover (S06 min dB).
Warp-then-SVD was not re-run (killed in t4r, −5.7 to −14 dB).

## Verdict

**Best method for the moving-element blocks: dense leftover JPEG (stride 1, MAE>3.5, q=40).** S06 33.79/33.23 vs SVD 31.46/30.19 (+2.33 dB mean, min 30.19→33.23), +1.41 MB on the tracking shot. Episode leftover vs v4's 15.23 MB SVD origin: 6.45 MB extra / 34.82 mean / 31.34 min.
Stride must be 1. S06 s2 32.62/30.29 — min barely moves vs SVD 30.19. s8 31.75/30.19 is a meter. The worst frames are the ones stride skips.
**Deblock is dead.** S09 ov2 35.31 seamR 1.28 vs A 35.18/1.30. S06 ov2 31.68/1.58 vs A 31.46/1.64. The grid is in the model, not a 10% seam mix.
**Overlap kills the lattice, costs ~4× tiles.** S09 hop-4 37.23 seamR 1.07 vs A 1.30, zlib 3.3 MB, 15105 tiles.
S06 hop-4 34.55/32.48 seamR 1.08 vs A 31.46/30.19/1.64.
Overlap+leftover s1 S06 36.37/35.15 vs leftover-only 33.79/33.23. Ship leftover first; add hop-4 only if the remaining 8×8 lattice on locked shots is still the demo complaint.

Do not ship affine / warp-then-SVD. Do not ship decode deblock. Do not ship stride-8 leftover as a quality stack.

## Reading

SVD-only baseline (leftover off): S09 35.18/31.95 seamR 1.30; S06 31.46/30.19 seamR 1.64.
Dense leftover buys the tracking hole. S06 s8 31.75/30.19 (+0.29 dB, 177KB) vs s1 33.79/33.23 (+2.33 dB, 1412KB).

## Rows

| id | mean | min | seamR | extra |
|---|---|---|---|---|
| `A-S00-svd` | 32.78 | 27.17 | 1.21 | 0KB |
| `L-S00-s8` | 32.97 | 28.25 | 1.18 | 108KB |
| `L-S00-s4` | 33.13 | 28.25 | 1.17 | 192KB |
| `L-S00-s2` | 33.44 | 28.25 | 1.14 | 364KB |
| `L-S00-s1` | 34.09 | 31.82 | 1.14 | 711KB |
| `L-S00-s1-all` | 34.28 | 31.82 | 1.13 | 906KB |
| `L-S00-s1-q30` | 33.88 | 31.10 | 1.15 | 511KB |
| `L-S00-tiles` | 34.11 | 31.56 | 1.16 | 1299KB |
| `A-S01-svd` | 37.54 | 34.09 | 1.28 | 0KB |
| `A-S02-svd` | 36.51 | 33.46 | 1.24 | 0KB |
| `A-S03-svd` | 32.72 | 28.05 | 1.44 | 0KB |
| `L-S03-s8` | 32.89 | 28.05 | 1.30 | 52KB |
| `L-S03-s4` | 33.08 | 28.08 | 1.30 | 107KB |
| `L-S03-s2` | 33.46 | 28.08 | 1.30 | 222KB |
| `L-S03-s1` | 34.18 | 32.77 | 1.30 | 433KB |
| `L-S03-s1-all` | 34.76 | 32.77 | 1.22 | 677KB |
| `L-S03-s1-q30` | 34.00 | 32.33 | 1.32 | 334KB |
| `A-S04-svd` | 38.96 | 38.26 | 1.04 | 0KB |
| `A-S05-svd` | 36.34 | 35.73 | 1.04 | 0KB |
| `A-S06-svd` | 31.46 | 30.19 | 1.64 | 0KB |
| `L-S06-s8` | 31.75 | 30.19 | 1.53 | 177KB |
| `L-S06-s4` | 32.03 | 30.19 | 1.52 | 349KB |
| `L-S06-s2` | 32.62 | 30.29 | 1.45 | 709KB |
| `L-S06-s1` | 33.79 | 33.23 | 1.32 | 1412KB |
| `L-S06-s1-all` | 33.79 | 33.23 | 1.32 | 1412KB |
| `L-S06-s1-q30` | 33.30 | 32.68 | 1.39 | 1048KB |
| `L-S06-tiles` | 33.51 | 32.98 | 1.45 | 2038KB |
| `A-S07-svd` | 34.89 | 27.47 | 1.19 | 0KB |
| `A-S08-svd` | 35.66 | 34.84 | 1.11 | 0KB |
| `A-S09-svd` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-s8` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-s4` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-s2` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-s1` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-s1-all` | 36.90 | 35.88 | 1.15 | 556KB |
| `L-S09-s1-q30` | 35.18 | 31.95 | 1.30 | 0KB |
| `L-S09-tiles` | 36.77 | 35.64 | 1.19 | 614KB |
| `A-S10-svd` | 33.66 | 30.02 | 1.21 | 0KB |
| `A-S11-svd` | 34.38 | 28.28 | 1.43 | 0KB |
| `A-S12-svd` | 32.70 | 31.33 | 1.36 | 0KB |
| `A-S13-svd` | 31.85 | 30.16 | 1.49 | 0KB |
| `L-S13-s8` | 32.06 | 30.16 | 1.23 | 64KB |
| `L-S13-s4` | 32.26 | 30.16 | 1.23 | 127KB |
| `L-S13-s2` | 32.66 | 30.16 | 1.23 | 249KB |
| `L-S13-s1` | 33.45 | 32.27 | 1.23 | 504KB |
| `L-S13-s1-all` | 33.45 | 32.27 | 1.23 | 504KB |
| `L-S13-s1-q30` | 33.06 | 32.01 | 1.27 | 369KB |
| `A-S14-svd` | 32.24 | 28.22 | 1.36 | 0KB |
| `L-EP-s8` | 33.95 | 28.05 | 0.00 | 855KB |
| `L-EP-s4` | 34.07 | 28.08 | 0.00 | 1644KB |
| `L-EP-s2` | 34.32 | 28.08 | 0.00 | 3249KB |
| `L-EP-s1` | 34.82 | 31.34 | 0.00 | 6450KB |
| `L-EP-s1-all` | 35.68 | 31.82 | 0.00 | 10219KB |
| `L-EP-s1-q30` | 34.66 | 31.10 | 0.00 | 4871KB |
| `D-S09-ov1` | 35.32 | 32.16 | 1.28 | 0KB |
| `D-S09-ov2` | 35.31 | 32.17 | 1.28 | 0KB |
| `D-S09-ov4` | 35.22 | 32.08 | 1.27 | 0KB |
| `D-S06-ov1` | 31.70 | 30.41 | 1.57 | 0KB |
| `D-S06-ov2` | 31.68 | 30.39 | 1.58 | 0KB |
| `D-S06-ov4` | 31.58 | 30.31 | 1.58 | 0KB |
| `D-S00-ov1` | 32.92 | 27.32 | 1.18 | 0KB |
| `D-S00-ov2` | 32.89 | 27.33 | 1.18 | 0KB |
| `D-S00-ov4` | 32.80 | 27.33 | 1.18 | 0KB |
| `O-S09-h4` | 37.23 | 34.63 | 1.07 | 3295KB |
| `O-S06-h4` | 34.55 | 32.48 | 1.08 | 13496KB |
| `OL-S06-s8` | 34.77 | 32.48 | 1.09 | 13620KB |
| `OL-S06-s1` | 36.37 | 35.15 | 1.14 | 14512KB |
