# narc

**not-a-real-codec.** Experiments toward an origin model: encode a file as motion + residual + keys, then rasterize and re-encode at whatever bitrate play asks for.

This is research, not a product codec. Big Buck Bunny (Blender Foundation, [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)) is the probe clip.

## Attempt log

| Branch / tag | What it is |
|---|---|
| `main`, `attempt/v0-global-translation`, tag `v0-global-translation` | One integer translation for the whole frame. JPEG keys. Optional 80×45 residual JPEG. |

Next work goes on a **new branch** off the v0 tag, e.g. `attempt/v1-…`. Do not rewrite v0 media. Checkout the old branch to see that attempt's `media/reconstruct.mp4` and `media/analysis.json`.

Notes for v0: [`attempts/v0-global-translation.md`](attempts/v0-global-translation.md).

## Layout

```
encoder/analyze-bbb.py   the v0 encoder
media/                   that encode's output (source, reconstruct, keys, residuals, analysis)
lab/                     Fluxfield UI as it stood at v0 (not a runnable app by itself)
attempts/                per-attempt notes
```

## Re-running v0

Needs `ffmpeg`, `numpy`, `Pillow`. Source film is not in git at full 720p — `media/source.mp4` is the 90s 640×360 window already cut.

```sh
python3 encoder/analyze-bbb.py
```

That overwrites `media/`. To compare attempts, run the new encoder on a branch and leave v0's media on `attempt/v0-global-translation`.
