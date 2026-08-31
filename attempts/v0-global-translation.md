# Attempt v0 — global translation

Frozen on `attempt/v0-global-translation` / tag `v0-global-translation`.

## Probe

- Clip: Big Buck Bunny, 50s–140s (forest, butterfly, bunny waking)
- Source window: 640×360 H.264, 90s (`media/source.mp4`)
- Analysis raster: 320×180 @ 10 fps, 900 frames

## Predictor

One integer translation `(dx, dy)` for the **entire frame**. Search is SAD on 80×45 luma, radius ±8, then scaled ×4 to 320×180. Uncovered pixels after the shift are zero.

Inter residual, when stored, is `(current − warped)` packed as an 80×45 JPEG at quality 38. Keys are full-frame JPEG at quality 84. Motion is 2 bytes per frame.

Keyframe rule: accumulate `residual + 40·occlusion` (discounted for explained motion / grain / flash). New key on cut, on budget ≥ 55, or at GOP 40 (~4s). Min GOP 6. Residual JPEG only if frame-mean residual ≥ 7.0.

## What it stored

| | |
|---|---|
| Keyframes | 66 |
| Cuts | 15 |
| Residual JPEGs stored | 166 |
| Motion-only frames | 183 |
| Static / skip frames | 548 |
| Origin model | 874.5 KB (keys 788.5, residual 105, motion 1.8) |
| H.264 source clip | 6.79 MB |
| Raw analysis pixels | 148 MB |

Kinds: `static 548, motion 183, residual 60, keyframe 52, grain 42, cut 15`. Mean flux 5.77, mean residual 6.51, mean motion 0.24.

## Known failure modes (this is the point of v0)

1. **Slideshow on locked shots.** Residual threshold is a *frame mean*. Small in-frame motion is diluted below 7.0, so 548 frames store no picture data at all — warp (often identity) of the last reconstruction.
2. **Small motion dies twice.** Integer search on 80×45 cannot see sub-4px motion at analysis res; the residual that would have saved it is 4× downsampled JPEG.
3. **Pan edges smear.** Global MC leaves a strip of uncovered pixels and cannot express parallax. The residual that should paint the incoming edge is the same 80×45 JPEG as the rest of the frame.

`media/reconstruct.mp4` is the rasterized model. Watch it against `media/source.mp4`.
