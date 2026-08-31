# Attempt v2 — tiny residual nets on the v1.2 warp

Branch: `attempt/v2-residual-nets`  
Frozen v1.2: `attempt/v1.2-affine-subpel`  
(v0 / v1 / v1.1 frozen, not rewritten)

Same 16×16 affine + sub-pel stack. No CU tree. No clip-wide NeRV. Many tiny nets on residual patches.

## Kill criterion (written before encode)

v1.2 mean PSNR 31.79 dB, leftover 4.54, residual JPEG 1.26 MB. JPEG q=52 already beats a small coordinate MLP on *residual-mode* tiles (MAE ~4 vs ~7). The quality hole is skip: luma skip at 2.6 leaves RGB leftover ~4–6 on 94% of blocks (chroma DC the warp never paid for).

If mean PSNR does not rise by at least ~1.0 dB (stays under 32.8) **and** leftover does not drop by ~15%, stop. That leftover is not a residual-coder problem; raise analysis resolution next, as agreed.

**Survived.** Mean PSNR 31.79 → **32.76 dB (+0.97)**. Leftover 4.54 → **3.96 (−13%)**. Min PSNR 26.98 → **29.22**. Origin grew 2.60 → 2.79 MB because nets add 260 KB while JPEG residual+intra only drop 100 KB. Neither knife missed by enough to call a fail, and the leftover that moved is exactly skip RGB — the pile this branch was for.

## What it is

| Lever | v1.2 | v2 |
|---|---|---|
| Warp | shot affine + sub-pel | unchanged |
| Skip | luma MAE ≤ 2.6, nothing stored | luma skip, then a **per-patch field net** on RGB leftover if RDO says so |
| Residual tiles | JPEG q=52 atlas | JPEG stays (it wins on textured leftover). Smooth tiles may take the net instead |
| Intra / keys | JPEG atlases | unchanged |
| Net | — | 1-layer field on PE `{1, x, y, xy}` → RGB, int8 weights, closed-loop dequant |

Each 16×16 patch that pays for a net is its own 12-parameter model (or DC-only, 3 params). That is many tiny nets, not one NeRV.

RDO per skip patch: `J = D + λR` with λ = 0.12, R = 6 (DC) or 14 (linear). Quantized weights are what land in recon.

Self-test: `tx=4.0 warp_mae=0.000`, `local_mae=0.008`, `halfpel_mae=0.000`, `zoom_scale=1.040 zoom_mae=1.100`, `net_dc_mae=0.000`, `net_lin_mae=0.208`.

## Probe numbers

| | v0 | v1 | v1.1 | v1.2 | v2 |
|---|---|---|---|---|---|
| Frames | 900 @ 10 fps | 2160 @ 24 fps | 2160 @ 24 fps | 2160 @ 24 fps | 2160 @ 24 fps |
| Keyframes | 66 | 78 | 47 | 42 | **41** |
| Residual frames stored | 166 | 1987 | 1974 | 1911 | **2093** (net counts as stored) |
| Skip / net / JPEG / intra | — | 87.6 / — / 7.9 / 4.4 % | 90.2 / — / 7.7 / 2.0 % | 93.9 / — / 5.0 / 1.1 % | **89.7 / 4.9 / 4.4 / 1.0 %** |
| Residual + intra bytes | — | — | 2.69 MB | 1.94 MB | **1.84 MB** |
| Net bytes | — | — | — | — | **260 KB** |
| Origin model | 874 KB | 5.36 MB | 3.39 MB | 2.60 MB | **2.79 MB** |
| Mean leftover | 6.51 | 5.68 | 3.96 | 4.54 | **3.96** |
| Mean reconstruct PSNR | slideshow | 27.3 dB (min 10.6) | 32.6 dB (min 28.5) | 31.8 dB (min 27.0) | **32.8 dB (min 29.2)** |

Nets fire on 4.9% of blocks, 2062 of 2160 frames. Mean ~12 net tiles per predicted frame. Skip 93.9% → 89.7% + 4.9% net: almost all net tiles were former skip, not stolen JPEG. JPEG residual 1.26 → 1.19 MB.

Quality is back to v1.1 leftover (3.96) and a hair above v1.1 PSNR (32.76 vs 32.57), at 2.79 MB instead of v1.1's 3.39 MB. The v1.2 byte win is mostly kept; the v1.2 PSNR dip from extra skip is paid back with chroma DC/linear fields.

## What it still is not

A rotating head, two-depth parallax inside one 16×16, swarm, occlusion. Those are appearance or a CU tree. A 12-param planar field cannot invent a second depth. Keyframe floor on this raster is still ~39 dB; mean 32.8 is not that. Against the 640×360 source the analysis is a quarter of the pixels — raising 320×180 is the next branch, as agreed. Clip-wide NeRV is later still.
