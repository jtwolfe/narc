# Attempt v3 — CU tree + real bitstream

Branch: `attempt/v3-cu-bitstream`  
Frozen v2: `attempt/v2-residual-nets` (media under `public/media/v2/`)  
(v0 / v1 / v1.1 / v1.2 frozen, not rewritten)

Same 16×16 affine + sub-pel warp. Tiny field nets stay on smooth leftover. No clip-wide NeRV. No 1×1 CUs.

Steal H.264/H.265’s **coding tree**, not its residual payload. Residual leaves are JPEG at 16×16 (texture) or DCT + quant + run-level at 8×8 / 4×4. Syntax is Exp-Golomb, then zlib — not CABAC.

## Kill criterion (written before encode)

v2 origin **2.79 MB**, mean PSNR **32.76 dB**, leftover **3.96**. Almost all leftover that a 12-param planar field can eat is already gone. The remaining hole is two-depth geometry inside a 16×16, plus the fact that v2 never packed a bitstream (JPEG files + raw int8 + 2-byte MVs).

If the **packed bitstream** (zlib(syntax) + JPEG keys + JPEG 16×16 residual + JPEG intra) does not drop ≥ ~20% vs 2.79 MB **and** mean PSNR does not hold within ~0.3 dB of 32.76, stop. That leftover is appearance; raise the 320×180 raster next, as agreed.

Two other meters, not kill:
- **Raw-accounted** — v2-style (JPEG files + raw int8 + 2 B MVs + split flags as 1 bit). Causal vs v2.
- **gzip control** — zlib of v2 side-info only (affine + skip bitmap + MVs + int8 nets; leave JPEG). Isolates “just pack v2 syntax” from “tree + DCT”.

## What it is

| Lever | v2 | v3 |
|---|---|---|
| Warp | shot affine + sub-pel | unchanged |
| Skip leftover | 16×16 DC / linear field net | unchanged (nets do not split) |
| Residual 16×16 | JPEG q=52 atlas | JPEG if unsplit. RDO-split → 8×8 / 4×4 DCT |
| Intra / keys | JPEG atlases | unchanged (16×16 only) |
| Syntax | raw counts | Exp-Golomb + zlib. Split flags, predicted MVs, run-level coeffs |
| Tree | — | 16 → 8 → 4. Never 1×1 |

RDO per CU: `J = D + λR` with λ = 0.12 and **real** R from Exp-Golomb of the committed symbols. Split only when 16×16 residual MAE is high or the four 8×8 quadrants disagree (two-depth / hole). Encode time stays in the same band as v2.

Self-test (before encode): v2 warp + net tests, plus DCT roundtrip, a two-depth 16×16 that must split, bitstream decode of a known MV / run-level residual.

## Probe numbers

| | v1.2 | v2 | v3 |
|---|---|---|---|
| Frames | 2160 @ 24 fps | 2160 @ 24 fps | 2160 @ 24 fps |
| Keyframes | 42 | 41 | |
| Skip / net / JPEG / intra | 93.9 / — / 5.0 / 1.1 % | 89.7 / 4.9 / 4.4 / 1.0 % | |
| CU 16 / 8 / 4 leaves | — | — | |
| Split 16×16 frac | — | — | |
| Residual JPEG + intra | 1.94 MB | 1.84 MB | |
| Net bytes | — | 260 KB | |
| Raw-accounted | 2.60 MB | 2.79 MB | |
| gzip control (v2 side-info) | — | — | |
| Packed bitstream (kill) | — | — | |
| Mean leftover | 4.54 | 3.96 | |
| Mean reconstruct PSNR | 31.8 dB (min 27.0) | 32.8 dB (min 29.2) | |

## What it still is not

A rotating head, swarm, occlusion, clip-wide NeRV. DCT on a 4×4 cannot invent appearance the warp never saw. Keyframe floor on this raster is still ~39 dB. Against the 640×360 source the analysis is a quarter of the pixels — if this kill fails, raising 320×180 is next, not a larger residual net.
