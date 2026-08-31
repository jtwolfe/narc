# Attempt v4r — patch temporal model, from scratch

Branch: `attempt/v4r`  
Frozen v3: `attempt/v3-cu-bitstream` (do not rewrite v0–v3 media)

No warp. No skip/intra RDO. No CU tree. No JPEG residual. Code lives in
`encoder/v4r/` so the v0–v3 scripts stay untouched.

v3 leftover was appearance the 16×16 translation-plus-affine never saw.
A clip-wide NeRV was the planned later branch. v4r is that deviation:
each shot is a set of 16×16 *temporal* models on the source pixels.

    patch(t) ≈ μ_shot + U(t) @ B

## Kill criterion (written before encode)

v3 mean PSNR **32.76 dB**, leftover 3.96, packed origin 3.02 MB on the 90s
probe. v4r does not have to beat bytes. It has to **reach ~32 dB**.

Gate: 10s head of the probe, same 320×180 / 24 fps raster. If mean PSNR
stays under 31.5 dB, stop — that leftover is not a linear temporal-patch
problem.

**Survived.** 10s probe **35.16 dB** (min 27.75). Full 90s **34.68 dB**
(min 27.25, median 34.79). Leftover 3.96 → **3.04**. Origin grew 3.02 →
**12.09 MB** because every shot stores its own spatial bases.

## What we tried on the 10s slice (before committing the architecture)

| Representation | 10s mean PSNR | Notes |
|---|---|---|
| Bilinear 45×80 thumbnail | 26.3 dB | 4× upsample is not 32 dB |
| Bilinear 90×160 | 29.4 dB | Still short |
| Global SVD k=16 | 29.8 dB | 2.8 MB of full-res bases |
| Global SVD k=32 | 33.1 dB | 5.5 MB — quality yes, bytes no |
| Shot-1 (locked) SVD k=4 | 32.5 dB | Locked shots are low-rank |
| Shot-0 (busy) SVD k=16 | 30.3 dB | Cuts + appearance need local rank |
| **16×16 patch SVD k=4** | **30.8 dB** | Almost |
| **16×16 patch SVD k=8** | **34.7 dB** | This is the 32 dB hole |
| Adaptive-K patch SVD + int8 + ALS | **35.2 dB** | Shipped |

A shared conv decoder (HNeRV-style) would amortize the spatial bases.
It also needs a GPU we do not have, and a 10s CPU overfit was not going
to beat closed-form SVD on this raster. SGD on the factors with lr=0.35
exploded to 6.7 dB; ALS after int8 dequant is the training that actually
runs.

## What it is

| Lever | v3 | v4r |
|---|---|---|
| Predictor | shot affine + sub-pel warp | **none** |
| Unit | 16×16 in space, 1 frame in time | 16×16 in space, **the whole shot in time** |
| Appearance | JPEG residual / 12-param spatial net | rank-K temporal SVD |
| Mean | (implicit in the warp) | closed-loop JPEG of the shot mean |
| Training | none (RDO) | 2-step ALS after int8 |
| Syntax | Exp-Golomb + zlib + JPEG sidecars | zlib(int8 U, B, scales) + JPEG μ |

RDO per patch: smallest K such that remaining MSE ≤ 36.6 (32.5 dB), cap
K=16. K=0 stores nothing but the shot mean.

Self-test: `still_k=0`, `fade_k=1 fade_mae=0.000`, `two_depth_k=1`,
`train_mae=0.000`.

## Probe numbers

| | v1.2 | v2 | v3 | v4r |
|---|---|---|---|---|
| Frames | 2160 @ 24 fps | 2160 @ 24 fps | 2160 @ 24 fps | **2160 @ 24 fps** |
| Shots / keys | 14 / 42 | 14 / 41 | 14 / 41 | **15 / 15** (shot starts) |
| K=0 / skip frac | 93.9% skip | 89.7% skip | 89.7% skip | **34.4% K=0** |
| Mean rank | — | — | — | **3.99** |
| Origin | 2.60 MB | 2.79 MB | 3.02 MB packed | **12.09 MB zlib** |
| Raw-accounted | 2.60 MB | 2.79 MB | 2.79 MB | **13.88 MB** |
| Mean leftover | 4.54 | 3.96 | 3.96 | **3.04** |
| Mean reconstruct PSNR | 31.8 dB (min 27.0) | 32.8 dB (min 29.2) | 32.8 dB (min 29.2) | **34.7 dB (min 27.3)** |

10s gate (first two shots): mean **35.16 dB**, leftover 2.84, mean K 3.5
on shot 0 / 1.6 on shot 1. Encoded the rest of the 90s with the same
knife.

226 of 3,600 patch-slots sat at K=16 and still missed the local MSE
target — those are the min-27 dB frames (busy appearance the linear
temporal model cannot span). Raising K further spends bytes on the
same hole a shared spatial decoder would have to invent.

## What it still is not

A rotating head with one 16-dim subspace. A 12 MB origin. A clip-wide
NeRV with a *shared* decoder — the bases here are per-shot, which is why
bytes scaled with shot count, not with a single weight file. zlib barely
moved int8 eigenpatches (13.88 → 12.09 MB).

The 16×16 lattice is visible on motion (especially background pans):
independent tiles, no warp, no shared \(U(t)\). That is a representation
problem, not under-training. v3’s CU tree never split on this clip
(`splitFrac = 0`); splitting v4r without a warp would add seams, not
remove them.

Next honest branch, if this raster stays: compress or share the spatial
bases (JPEG eigen-mosaic, or a small conv decoder trained on them), or
warp-the-tube then SVD. Raising 320×180 is still the apples-to-apples
vs 640×360 source.
