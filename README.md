# narc

**not-a-real-codec.** Experiments toward an origin model: encode a file as motion + residual + keys, then rasterize and re-encode at whatever bitrate play asks for.

This is research, not a product codec. Big Buck Bunny (Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) is the probe clip.

## Attempt log

| Branch | What it is |
|---|---|
| `main`, `attempt/v0-global-translation`, tag `v0-global-translation` | One integer translation for the whole frame. JPEG keys. Optional 80×45 residual JPEG. 10 fps. |
| `attempt/v1-block-mc` | 24 fps. Half-pel global + 16×16 skip/residual/intra. Two refs. \(J = D + \lambda R\). **Inverted MV sign; black OOB tiles.** Frozen as the broken baseline. |
| `attempt/v1.1-mc-correct` | Same stack, closed loop: unified MV sign, pad not zero, RDO on real pred, residual/intra atlases, MV median, 1-px deblock. |
| `attempt/v1.2-affine-subpel` | Same 16×16 stack. Shot-level affine instead of a single translation. Local half/quarter-pel corrections. Adaptive median that does not flatten a zoom. |
| `attempt/v2-residual-nets` | Same v1.2 warp. Many tiny per-patch field nets (DC or linear `{1,x,y,xy}` → RGB, int8) on skip leftover. JPEG stays for textured residual. |
| `attempt/v3-cu-bitstream` | Same warp + nets. CU tree 16→8→4. JPEG on unsplit 16×16 residual; DCT + quant + Exp-Golomb on 8×8/4×4. zlib bitstream. Two byte meters. |
| `attempt/v4r` | **From scratch.** No warp. Shot-wise 16×16 temporal SVD, JPEG shot mean, int8 + ALS. Kill is ~32 dB PSNR, not bytes. |
| `attempt/v4.t1r` | Knob sweep on frozen v4r. 3×10s, 42 configs. Training = 0.000 dB. No UI change. |
| `attempt/v4.t2r` | Native 640×360, per-shot. 16×16 / OLA / global+leaves. 2-shot episode proxy. No UI. |
| `attempt/v4.t3r` | Native 640, per-shot. Tile 2–64, cheap seams, JPEG-on-B, exclusive merge + residual trees. No UI. |
| `attempt/v4.t4r` | 8×8 episode baseline, JPEG-B packing, translation warp, leftover JPEG ceiling. No UI. |
| `attempt/v4` | **Product encode.** Native 640×360, 8×8 SVD, atlas JPEG-on-B, sparse leftover. Origin 15.23 MB, 33.95 dB. Lab default. |
| `attempt/v4.1` | Selectable K′ decode of the v4 origin. Lab default. Same 15.23 MB. Blockiness: leftover s1 wins; not shipped yet. |

Next work goes on a **new branch** off the attempt you want to beat. Do not rewrite v0, v1, v1.1, v1.2, v2, v3, v4r, or v4 media.

- v0 notes: [`attempts/v0-global-translation.md`](attempts/v0-global-translation.md)
- v1 notes: [`attempts/v1-block-mc.md`](attempts/v1-block-mc.md)
- v1.1 notes: [`attempts/v1.1-mc-correct.md`](attempts/v1.1-mc-correct.md)
- v1.2 notes: [`attempts/v1.2-affine-subpel.md`](attempts/v1.2-affine-subpel.md)
- v2 notes: [`attempts/v2-residual-nets.md`](attempts/v2-residual-nets.md)
- v3 notes: [`attempts/v3-cu-bitstream.md`](attempts/v3-cu-bitstream.md)
- v4r notes: [`attempts/v4r.md`](attempts/v4r.md)
- v4.t1r notes: [`attempts/v4.t1r.md`](attempts/v4.t1r.md)
- v4.t2r notes: [`attempts/v4.t2r.md`](attempts/v4.t2r.md)
- v4.t3r notes: [`attempts/v4.t3r.md`](attempts/v4.t3r.md)
- v4.t4r notes: [`attempts/v4.t4r.md`](attempts/v4.t4r.md)
- v4 notes: [`attempts/v4.md`](attempts/v4.md)
- v4.1 notes: [`attempts/v4.1.md`](attempts/v4.1.md)

v4.1 is the same origin as v4 with a live K′ peel at decode (μ → 16 → full). Mean 33.95 dB at full, 33.1 dB at K′=4, 20.9 dB at μ. Origin still 15.23 MB. v4 stays frozen. Blockiness campaign on this branch: **dense leftover stride 1** is the moving-element fix (S06 +2.33 dB / min 33.23, episode +6.45 MB); hop-4 overlap kills the 8×8 lattice at ~4× tiles; deblock and affine are dead. Lab not rewritten until that leftover is packed.

## Layout

```
encoder/analyze-bbb.py    v0 encoder (global translation)
encoder/analyze-v1.py     v1 encoder (block MC, inverted sign)
encoder/analyze-v1.1.py   v1.1 encoder (corrected motion loop)
encoder/analyze-v1.2.py   v1.2 encoder (affine + sub-pel)
encoder/analyze-v2.py     v2 encoder (tiny residual nets)
encoder/analyze-v3.py     v3 encoder (CU tree + bitstream)
encoder/v4r/              frozen v4r encoder
encoder/analyze-v4r.py    thin launcher for encoder/v4r
encoder/v4/               v4 encoder (native 8×8 + atlas + leftover)
encoder/analyze-v4.py     thin launcher for encoder/v4
encoder/v4.1/             K′ bake from the v4 origin
encoder/analyze-v4.1.py   thin launcher for encoder/v4.1
media/                    that branch's reconstruct + analysis
lab/                      Fluxfield UI snapshot
attempts/                 per-attempt notes
```

On `attempt/v4.1`, `media/stats-v4.1.json` holds the peel numbers. The live lab keeps v0 through v4 reconstructs so you can A/B without checking out a branch.
