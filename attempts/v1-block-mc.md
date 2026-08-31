# Attempt v1 — block MC

Branch: `attempt/v1-block-mc`  
Frozen v0 remains on `attempt/v0-global-translation`.

## What changed (low / medium cost only)

No nets. Same JPEG keys. Same 320×180 analysis raster, now at **24 fps**.

| Lever | v0 | v1 |
|---|---|---|
| Cadence | 10 fps | 24 fps (native clip) |
| Motion | 1 integer translation / frame | half-pel global + 16×16 local ±2 |
| Skip | frame-mean residual ≥ 7 | per-block J = D + λR, skip MAE ≤ 2.6 |
| Residual | optional 80×45 JPEG | full-res JPEG, skip blocks zeroed |
| Uncovered pan | zero fill | intra (copy source block) |
| Refs | previous recon only | previous recon **or** last key |
| Loop | analysis on originals | closed loop on recon |

## Probe numbers

| | v0 | v1 |
|---|---|---|
| Frames | 900 @ 10 fps | 2160 @ 24 fps |
| Keyframes | 66 | 78 |
| Residual frames stored | 166 | 1987 |
| Skip / resid / intra blocks | (none) | 87.6% / 7.9% / 4.4% |
| Origin model | 874 KB | 5.36 MB |
| Mean reconstruct PSNR | — (slideshow) | **27.3 dB** (median 29.1; 16% of frames < 20 dB) |

## What it actually fixed

Locked shots were the v0 slideshow. A butterfly that v0 held as a still is in-frame motion in v1: skip on the forest, residual/intra on the insect. That was the point of killing the frame-mean gate.

## What it did not fix

Tracking / parallax still smears. Local ±2 around a translation cannot express a camera push or a rotating head. Worst frames sit around 10–12 dB (≈ 30s butterfly swarm, ≈ 61s and 68s tracking). Intra paints uncovered *strips*, not disoccluded *surfaces*. No deblock, no sub-pel per block (only global half-pel).

Origin grew ~6× because almost every frame now stores a residual JPEG. Quality-first, as agreed. Most of those bytes are high-frequency leftover, which is exactly the slot a **tiny per-block / per-shot net** would occupy next — not a clip-wide NeRV.

## Next (not this branch)

1. Tiny nets on residual patches / GOPs (many small overfits, not one f(t)).
2. Optional NeRV-style attempt after that, as its own branch, so this raster stays the baseline.
