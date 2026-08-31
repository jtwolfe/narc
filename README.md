# narc

**not-a-real-codec.** Experiments toward an origin model: encode a file as motion + residual + keys, then rasterize and re-encode at whatever bitrate play asks for.

This is research, not a product codec. Big Buck Bunny (Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) is the probe clip.

## Attempt log

| Branch | What it is |
|---|---|
| `main`, `attempt/v0-global-translation`, tag `v0-global-translation` | One integer translation for the whole frame. JPEG keys. Optional 80×45 residual JPEG. 10 fps. |
| `attempt/v1-block-mc` | 24 fps. Half-pel global + 16×16 skip/residual/intra. Two refs. J = D + λR. Full-res residual on failing blocks. |

Next work goes on a **new branch** off the attempt you want to beat. Do not rewrite v0 media.

- v0 notes: [`attempts/v0-global-translation.md`](attempts/v0-global-translation.md)
- v1 notes: [`attempts/v1-block-mc.md`](attempts/v1-block-mc.md)

Tiny per-block nets and a clip-wide NeRV are **later branches**, not this one.

## Layout

```
encoder/analyze-bbb.py   v0 encoder (global translation)
encoder/analyze-v1.py    v1 encoder (block MC)
media/                   that branch's reconstruct + analysis
lab/                     Fluxfield UI snapshot
attempts/                per-attempt notes
```

On `attempt/v1-block-mc`, `media/stats-v1.json` holds the probe numbers. The live lab keeps both reconstructs so you can A/B without checking out v0.
