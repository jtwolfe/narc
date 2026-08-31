# narc

**not-a-real-codec.** Experiments toward an origin model: encode a file as motion + residual + keys, then rasterize and re-encode at whatever bitrate play asks for.

This is research, not a product codec. Big Buck Bunny (Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) is the probe clip.

## Attempt log

| Branch | What it is |
|---|---|
| `main`, `attempt/v0-global-translation`, tag `v0-global-translation` | One integer translation for the whole frame. JPEG keys. Optional 80×45 residual JPEG. 10 fps. |
| `attempt/v1-block-mc` | 24 fps. Half-pel global + 16×16 skip/residual/intra. Two refs. J = D + λR. **Inverted MV sign; black OOB tiles.** Frozen as the broken baseline. |
| `attempt/v1.1-mc-correct` | Same stack, closed loop: unified MV sign, pad not zero, RDO on real pred, residual/intra atlases, MV median, 1-px deblock. |
| `attempt/v1.2-affine-subpel` | Same 16×16 stack. Shot-level affine instead of a single translation. Local half/quarter-pel corrections. Adaptive median that does not flatten a zoom. |
| `attempt/v2-residual-nets` | Same v1.2 warp. Many tiny per-patch field nets (DC or linear `{1,x,y,xy}` → RGB, int8) on skip leftover. JPEG stays for textured residual. |
| `attempt/v3-cu-bitstream` | Same warp + nets. CU tree 16→8→4. JPEG on unsplit 16×16 residual; DCT + quant + Exp-Golomb on 8×8/4×4. zlib bitstream. Two byte meters. |

Next work goes on a **new branch** off the attempt you want to beat. Do not rewrite v0, v1, v1.1, v1.2, or v2 media.

- v0 notes: [`attempts/v0-global-translation.md`](attempts/v0-global-translation.md)
- v1 notes: [`attempts/v1-block-mc.md`](attempts/v1-block-mc.md)
- v1.1 notes: [`attempts/v1.1-mc-correct.md`](attempts/v1.1-mc-correct.md)
- v1.2 notes: [`attempts/v1.2-affine-subpel.md`](attempts/v1.2-affine-subpel.md)
- v2 notes: [`attempts/v2-residual-nets.md`](attempts/v2-residual-nets.md)
- v3 notes: [`attempts/v3-cu-bitstream.md`](attempts/v3-cu-bitstream.md)

Raising the 320×180 analysis raster (apples-to-apples vs the 640×360 source) is the next branch if v3’s packed bitstream does not drop ~20% at held PSNR. Clip-wide NeRV is later still.

## Layout

```
encoder/analyze-bbb.py    v0 encoder (global translation)
encoder/analyze-v1.py     v1 encoder (block MC, inverted sign)
encoder/analyze-v1.1.py   v1.1 encoder (corrected motion loop)
encoder/analyze-v1.2.py   v1.2 encoder (affine + sub-pel)
encoder/analyze-v2.py     v2 encoder (tiny residual nets)
encoder/analyze-v3.py     v3 encoder (CU tree + bitstream)
media/                    that branch's reconstruct + analysis
lab/                      Fluxfield UI snapshot
attempts/                 per-attempt notes
```

On `attempt/v3-cu-bitstream`, `media/stats-v3.json` holds the probe numbers. The live lab keeps v0, v1, v1.1, v1.2, and v2 reconstructs so you can A/B without checking out a branch.
