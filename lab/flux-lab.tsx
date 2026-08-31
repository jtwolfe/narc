import { useCallback, useEffect, useMemo, useRef, useState, type RefObject } from "react";
import {
  Pause,
  Play,
  SkipBack,
  StepForward,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import type { Analysis, FrameKind, FrameRow, ShotRow } from "@/lib/analysis";
import { frameAtTime } from "@/lib/analysis";
import { cn, formatBytes, formatTime } from "@/lib/utils";

const KIND_LABEL: Record<FrameKind, string> = {
  keyframe: "Keyframe",
  cut: "Cut",
  motion: "Explained motion",
  residual: "Unexplained residual",
  static: "Locked / static",
  flash: "Luma impulse",
  grain: "Grain-like",
};

function kindColor(kind: FrameKind): string {
  switch (kind) {
    case "keyframe":
      return "text-fg";
    case "cut":
      return "text-copper";
    case "motion":
      return "text-steel";
    case "residual":
      return "text-copper";
    case "grain":
      return "text-moss";
    case "flash":
      return "text-accent";
    default:
      return "text-fg-muted";
  }
}

export function FluxLab({ data }: { data: Analysis }) {
  const { frames, shots, stats, source } = data;
  const duration = stats.duration;
  const srcRef = useRef<HTMLVideoElement>(null);
  const recRef = useRef<HTMLVideoElement>(null);
  const [t, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [bitrate, setBitrate] = useState(1200);
  const [heat, setHeat] = useState(false);
  const [decode, setDecode] = useState<"v4r" | "v3" | "v2" | "v12" | "v11" | "v1" | "v0">("v4r");
  const [srcRaster, setSrcRaster] = useState<"full" | "analysis">("full");
  const swappingSrc = useRef(false);
  const tRef = useRef(t);
  tRef.current = t;
  const playingRef = useRef(playing);
  playingRef.current = playing;
  const frame = useMemo(() => frameAtTime(frames, t), [frames, t]);
  const shot = useMemo(
    () => shots.find((s) => t >= s.t0 && t < s.t1) ?? shots[shots.length - 1],
    [shots, t],
  );

  const sync = useCallback((time: number) => {
    const v = Math.max(0, Math.min(duration - 0.04, time));
    setT(v);
    if (srcRef.current && Math.abs(srcRef.current.currentTime - v) > 0.05) {
      srcRef.current.currentTime = v;
    }
    if (recRef.current && Math.abs(recRef.current.currentTime - v) > 0.08) {
      recRef.current.currentTime = v;
    }
  }, [duration]);

  const toggle = useCallback(() => {
    const a = srcRef.current;
    const b = recRef.current;
    if (!a || !b) return;
    if (playing) {
      a.pause();
      b.pause();
      setPlaying(false);
    } else {
      void a.play();
      void b.play();
      setPlaying(true);
    }
  }, [playing]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLButtonElement) return;
      if ((e.target as HTMLElement | null)?.closest?.("[data-raster-toggle]")) return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        sync(tRef.current + 1 / stats.fps);
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        sync(tRef.current - 1 / stats.fps);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [toggle, sync, stats.fps]);

  const nextKey = () => {
    const nxt = frames.find((f) => f.key && f.t > t + 0.05);
    sync(nxt ? nxt.t : 0);
  };
  const prevKey = () => {
    const prev = [...frames].reverse().find((f) => f.key && f.t < t - 0.05);
    sync(prev ? prev.t : 0);
  };

  const toggleSrcRaster = useCallback(() => {
    const hold = srcRef.current?.currentTime ?? tRef.current;
    tRef.current = hold;
    setT(hold);
    swappingSrc.current = true;
    setSrcRaster((r) => (r === "full" ? "analysis" : "full"));
  }, []);

  useEffect(() => {
    const v = srcRef.current;
    if (!v) return;
    const hold = tRef.current;
    const restore = () => {
      if (Math.abs(v.currentTime - hold) > 0.04) v.currentTime = hold;
      if (playingRef.current) void v.play();
      else v.pause();
      swappingSrc.current = false;
    };
    v.addEventListener("loadeddata", restore, { once: true });
    if (v.readyState >= 2) restore();
    return () => v.removeEventListener("loadeddata", restore);
  }, [srcRaster]);

  const srcClip =
    srcRaster === "analysis"
      ? (source.clipAnalysis ?? "/media/source-320.mp4")
      : source.clip;
  const srcSub =
    srcRaster === "analysis"
      ? "Analysis raster · 320×180 · click for 640×360"
      : "POSIX original · 640×360 · click for 320×180";

  const deliveryBytes = Math.round((bitrate * 1000 * duration) / 8);
  const heatSrc =
    frame.key || frame.cut || frame.residual > 10
      ? `/media/heatmaps/${String(frame.i).padStart(4, "0")}.jpg`
      : null;
  const recSrc =
    decode === "v0"
      ? (source.reconstructV0 ?? "/media/v0/reconstruct.mp4")
      : decode === "v1"
        ? (source.reconstructV1 ?? "/media/v1/reconstruct.mp4")
        : decode === "v11"
          ? (source.reconstructV11 ?? "/media/v1.1/reconstruct.mp4")
          : decode === "v12"
            ? (source.reconstructV12 ?? "/media/v1.2/reconstruct.mp4")
            : decode === "v2"
              ? (source.reconstructV2 ?? "/media/v2/reconstruct.mp4")
              : decode === "v3"
                ? (source.reconstructV3 ?? "/media/v3/reconstruct.mp4")
                : source.reconstruct;
  const baseline = stats.baseline;
  const baselineV1 = stats.baselineV1;
  const baselineV11 = stats.baselineV11;
  const isV4r = (stats.attempt ?? "").includes("v4r");
  const isV3 = (stats.attempt ?? "").includes("v3");
  const baselineV12 = stats.baselineV12;
  const baselineV2 = stats.baselineV2;
  const baselineV3 = stats.baselineV3 ?? (!isV4r && isV3
    ? {
        attempt: "v3-cu-bitstream",
        fps: stats.fps,
        frames: stats.frames,
        keyframes: stats.keyframes,
        residualsStored: stats.residualsStored,
        modelBytes: stats.bitstreamBytes ?? stats.modelBytes,
        meanResidual: stats.meanResidual,
        meanPsnr: stats.meanPsnr,
        skipBlockFrac: stats.skipBlockFrac,
        reconstructMp4Bytes: stats.reconstructMp4Bytes,
        residualBytes: stats.residualBytes,
        intraBytes: stats.intraBytes,
        netBytes: stats.netBytes,
        bitstreamBytes: stats.bitstreamBytes,
      }
    : undefined);
  const blocks = stats.blocksPerFrame ?? 220;
  const decodeLabel =
    decode === "v0"
      ? "Model decode · v0"
      : decode === "v1"
        ? "Model decode · v1"
        : decode === "v11"
          ? "Model decode · v1.1"
          : decode === "v12"
            ? "Model decode · v1.2"
            : decode === "v2"
              ? "Model decode · v2"
              : decode === "v3"
                ? "Model decode · v3"
                : "Model decode · v4r";
  const decodeSub =
    decode === "v0"
      ? "Global translation · 10 fps"
      : decode === "v1"
        ? "Inverted-sign block MC · 24 fps"
        : decode === "v11"
          ? "Corrected translation MC · 24 fps"
          : decode === "v12"
            ? "Affine + sub-pel MC · 24 fps"
            : decode === "v2"
              ? "Tiny residual nets · 24 fps"
              : decode === "v3"
                ? "CU tree + bitstream · 24 fps"
                : "Patch temporal model · 24 fps";

  const strip = useMemo(() => {
    const out: number[] = [];
    const step = Math.max(1, Math.round(stats.fps));
    for (let i = 0; i < frames.length; i += step) out.push(i);
    return out;
  }, [frames, stats.fps]);

  return (
    <div className="min-h-dvh bg-bg text-fg">
      <header className="border-b border-border px-4 py-5 sm:px-8">
        <div className="mx-auto flex max-w-6xl flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <p className="font-mono text-xs tracking-[0.22em] text-fg-subtle uppercase">
              Workarr encode lab
            </p>
            <h1 className="font-display mt-1 text-3xl font-medium tracking-tight text-balance sm:text-4xl">
              Fluxfield
            </h1>
            <p className="mt-2 max-w-xl text-sm text-pretty text-fg-muted">
              {source.title} · {source.window}. Attempt{" "}
              <span className="text-fg">{stats.attempt ?? "v4r"}</span>
              : no warp. Each 16×16 is a temporal model on the pixels
              themselves — shot mean JPEG plus a rank-K SVD, int8, ALS
              refine. Toggle back through v3, v2, v1.2, v1.1, broken v1, and v0.
            </p>
          </div>
          <p className="font-mono text-xs leading-relaxed text-fg-subtle sm:text-right">
            {source.credit}
            <br />
            Analysis {stats.width}×{stats.height} @ {stats.fps} fps
            {stats.block ? ` · ${stats.block[0]}×${stats.block[1]} blocks` : ""}
          </p>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 py-6 sm:px-8 sm:py-8">
        <div className="grid gap-3 lg:grid-cols-2">
          <Viewer
            label={srcRaster === "analysis" ? "Source · 320×180" : "Source"}
            sub={srcSub}
            videoRef={srcRef}
            src={srcClip}
            badge={srcRaster === "analysis" ? "320×180" : "640×360"}
            onActivate={toggleSrcRaster}
            onTime={(v) => {
              if (swappingSrc.current) return;
              setT(v);
              if (recRef.current && Math.abs(recRef.current.currentTime - v) > 0.12) {
                recRef.current.currentTime = v;
              }
            }}
            onPlay={() => {
              if (swappingSrc.current) return;
              void recRef.current?.play();
              setPlaying(true);
            }}
            onPause={() => {
              if (swappingSrc.current) return;
              recRef.current?.pause();
              setPlaying(false);
            }}
          />
          <Viewer
            key={decode}
            label={decodeLabel}
            sub={decodeSub}
            videoRef={recRef}
            src={recSrc}
            muted
            overlay={heat && (decode === "v4r" || decode === "v3") && heatSrc ? heatSrc : null}
            onTime={(v) => {
              setT(v);
              if (srcRef.current && Math.abs(srcRef.current.currentTime - v) > 0.12) {
                srcRef.current.currentTime = v;
              }
            }}
          />
        </div>

        <div className="mt-4 rounded-xl bg-bg-elevated p-3 shadow-border sm:p-4">
          <div className="flex flex-wrap items-center gap-2">
            <Button size="icon" variant="outline" onClick={prevKey} aria-label="Previous keyframe">
              <SkipBack className="size-4" />
            </Button>
            <Button size="icon" onClick={toggle} aria-label={playing ? "Pause" : "Play"}>
              {playing ? <Pause className="size-4" /> : <Play className="ml-0.5 size-4" />}
            </Button>
            <Button size="icon" variant="outline" onClick={nextKey} aria-label="Next keyframe">
              <StepForward className="size-4" />
            </Button>
            <span className="font-mono tabular-nums text-sm text-fg">
              {formatTime(t)}
              <span className="text-fg-subtle"> / {formatTime(duration)}</span>
            </span>
            <span className={cn("ml-auto font-mono text-xs", kindColor(frame.kind))}>
              {KIND_LABEL[frame.kind]}
            </span>
            <label className="flex min-h-11 items-center gap-2 px-1 text-xs text-fg-muted">
              <input
                type="checkbox"
                checked={heat}
                onChange={(e) => setHeat(e.target.checked)}
                className="size-4 accent-accent"
              />
              Residual heat
            </label>
            <div className="flex min-h-11 flex-wrap rounded-md bg-bg-subtle p-0.5" role="group" aria-label="Decoder version">
              {(
                [
                  ["v4r", "v4r"],
                  ["v3", "v3"],
                  ["v2", "v2"],
                  ["v12", "v1.2"],
                  ["v11", "v1.1"],
                  ["v1", "v1"],
                  ["v0", "v0"],
                ] as const
              ).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  className={cn(
                    "min-h-10 rounded-sm px-3 font-mono text-xs",
                    decode === id ? "bg-bg-elevated text-fg" : "text-fg-muted",
                  )}
                  onClick={() => setDecode(id)}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>
          <div className="mt-3">
            <Slider
              min={0}
              max={duration}
              step={0.1}
              value={[t]}
              onValueChange={([v]) => {
                setPlaying(false);
                srcRef.current?.pause();
                recRef.current?.pause();
                sync(v ?? 0);
              }}
            />
          </div>
          <FluxScope frames={frames} shots={shots} t={t} duration={duration} onSeek={sync} />
          <Filmstrip frames={frames} indices={strip} t={t} onSeek={sync} />
        </div>

        <div className="mt-6 grid gap-3 md:grid-cols-4">
          <StatCard
            label="v4r origin"
            value={isV4r ? formatBytes(stats.bitstreamBytes ?? stats.modelBytes) : "encoding…"}
            hint="zlib(int8 patch SVD + JPEG shot means)"
          />
          <StatCard
            label="v3 packed (frozen)"
            value={baselineV3 ? formatBytes(baselineV3.bitstreamBytes ?? baselineV3.modelBytes) : "—"}
            hint="CU tree bitstream · 32.8 dB"
          />
          <StatCard
            label="Mean reconstruct PSNR"
            value={isV4r && stats.meanPsnr ? `${stats.meanPsnr.toFixed(1)} dB` : "—"}
            hint={isV4r && stats.minPsnr != null ? `min ${stats.minPsnr.toFixed(1)} · median ${stats.medianPsnr?.toFixed(1)}` : "v4r vs analysis JPEG"}
          />
          <StatCard
            label="H.264 source clip"
            value={formatBytes(stats.sourceBytes)}
            hint="same 90s, 640×360"
          />
        </div>

        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-8">
          <Mini label="Shots" value={String(stats.shots)} />
          <Mini label="Shot-start keys" value={String(stats.keyframes)} />
          <Mini
            label="K=0 patches"
            value={isV4r && stats.skipBlockFrac != null ? `${(stats.skipBlockFrac * 100).toFixed(0)}%` : "—"}
          />
          <Mini
            label="Mean rank"
            value={isV4r && stats.meanRank != null ? stats.meanRank.toFixed(2) : "—"}
          />
          <Mini
            label="Spatial bases"
            value={isV4r && stats.basisBytes != null ? formatBytes(stats.basisBytes) : "—"}
          />
          <Mini
            label="Temporal U"
            value={isV4r && stats.coeffBytes != null ? formatBytes(stats.coeffBytes) : "—"}
          />
          <Mini
            label="Shot-mean JPEG"
            value={isV4r && stats.meanJpegBytes != null ? formatBytes(stats.meanJpegBytes) : "—"}
          />
          <Mini
            label="Raw-accounted"
            value={isV4r && stats.rawAccountedBytes != null ? formatBytes(stats.rawAccountedBytes) : "—"}
          />
        </div>

        {(baseline || baselineV1 || baselineV11 || baselineV12 || baselineV2 || stats.bitstreamBytes != null) ? (
          <section className="mt-3 overflow-x-auto rounded-xl bg-bg-elevated p-4 shadow-border">
            <h2 className="font-display text-lg font-medium">v4r · v3 · v2 · v1.2 · v1.1 · v1 · v0</h2>
            <table className="mt-3 w-full text-left font-mono text-sm">
              <thead className="text-xs text-fg-subtle">
                <tr>
                  <th className="sticky left-0 bg-bg-elevated py-2 pr-2 font-medium"> </th>
                  <th className="py-2 pr-2 font-medium">v4r temporal</th>
                  <th className="py-2 pr-2 font-medium">v3 tree</th>
                  <th className="py-2 pr-2 font-medium">v2 nets</th>
                  <th className="py-2 pr-2 font-medium">v1.2 affine</th>
                  <th className="py-2 pr-2 font-medium">v1.1 correct</th>
                  <th className="py-2 pr-2 font-medium">v1 inverted</th>
                  <th className="py-2 font-medium">v0 global</th>
                </tr>
              </thead>
              <tbody className="text-fg">
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Cadence</td>
                  <td className="py-2 pr-2">{isV4r ? `${stats.fps} fps · ${stats.frames}` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3 ? `${baselineV3.fps} fps · ${baselineV3.frames}` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2 ? `${baselineV2.fps} fps · ${baselineV2.frames}` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12 ? `${baselineV12.fps} fps · ${baselineV12.frames}` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11 ? `${baselineV11.fps} fps · ${baselineV11.frames}` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1 ? `${baselineV1.fps} fps · ${baselineV1.frames}` : "—"}</td>
                  <td className="py-2">{baseline?.fps != null ? `${baseline.fps} fps · ${baseline.frames}` : "—"}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Keyframes</td>
                  <td className="py-2 pr-2">{isV4r ? stats.keyframes : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3 ? baselineV3.keyframes : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2 ? baselineV2.keyframes : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12 ? baselineV12.keyframes : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11 ? baselineV11.keyframes : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1 ? baselineV1.keyframes : "—"}</td>
                  <td className="py-2">{baseline?.keyframes ?? "—"}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Origin bytes</td>
                  <td className="py-2 pr-2">{isV4r ? formatBytes(stats.bitstreamBytes ?? stats.modelBytes) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3 ? formatBytes(baselineV3.bitstreamBytes ?? baselineV3.modelBytes) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2 ? formatBytes(baselineV2.modelBytes) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12 ? formatBytes(baselineV12.modelBytes) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11 ? formatBytes(baselineV11.modelBytes) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1 ? formatBytes(baselineV1.modelBytes) : "—"}</td>
                  <td className="py-2">{baseline?.modelBytes ? formatBytes(baseline.modelBytes) : "—"}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">What the bytes are</td>
                  <td className="py-2 pr-2">{isV4r ? "int8 U,B + JPEG μ" : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3 ? "JPEG + zlib syntax" : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2?.netBytes != null ? formatBytes(baselineV2.netBytes) + " nets" : "—"}</td>
                  <td className="py-2 pr-2">—</td>
                  <td className="py-2 pr-2">—</td>
                  <td className="py-2 pr-2">—</td>
                  <td className="py-2">—</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">K=0 / skip</td>
                  <td className="py-2 pr-2">{isV4r && stats.skipBlockFrac != null ? `${(stats.skipBlockFrac * 100).toFixed(0)}% K=0` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3?.skipBlockFrac != null ? `${(baselineV3.skipBlockFrac * 100).toFixed(0)}% skip` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2?.skipBlockFrac != null ? `${(baselineV2.skipBlockFrac * 100).toFixed(0)}% skip` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12?.skipBlockFrac != null ? `${(baselineV12.skipBlockFrac * 100).toFixed(0)}% skip` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11?.skipBlockFrac != null ? `${(baselineV11.skipBlockFrac * 100).toFixed(0)}% skip` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1?.skipBlockFrac != null ? `${(baselineV1.skipBlockFrac * 100).toFixed(0)}% skip` : "—"}</td>
                  <td className="py-2">—</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Geometry</td>
                  <td className="py-2 pr-2">{isV4r ? "none · temporal SVD" : "—"}</td>
                  <td className="py-2 pr-2">affine + CU 16</td>
                  <td className="py-2 pr-2">16×16 + nets</td>
                  <td className="py-2 pr-2">16×16 affine</td>
                  <td className="py-2 pr-2">16×16 trans</td>
                  <td className="py-2 pr-2">16×16 broken</td>
                  <td className="py-2">frame trans</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Mean leftover</td>
                  <td className="py-2 pr-2">{isV4r ? stats.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3 ? baselineV3.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2 ? baselineV2.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12 ? baselineV12.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11 ? baselineV11.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1 ? baselineV1.meanResidual.toFixed(1) : "—"}</td>
                  <td className="py-2">{baseline?.meanResidual ? baseline.meanResidual.toFixed(1) : "—"}</td>
                </tr>
                <tr className="border-t border-border">
                  <td className="sticky left-0 bg-bg-elevated py-2 pr-2 text-fg-muted">Mean PSNR</td>
                  <td className="py-2 pr-2">{isV4r && stats.meanPsnr != null ? `${stats.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV3?.meanPsnr != null ? `${baselineV3.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV2?.meanPsnr != null ? `${baselineV2.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV12?.meanPsnr != null ? `${baselineV12.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV11?.meanPsnr != null ? `${baselineV11.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2 pr-2">{baselineV1?.meanPsnr != null ? `${baselineV1.meanPsnr.toFixed(1)} dB` : "—"}</td>
                  <td className="py-2">{baseline?.meanPsnr != null ? `${baseline.meanPsnr.toFixed(1)} dB` : "—"}</td>
                </tr>
              </tbody>
            </table>
          </section>
        ) : null}

        <div className="mt-6 grid gap-3 lg:grid-cols-2">
          <section className="rounded-xl bg-bg-elevated p-5 shadow-border">
            <h2 className="font-display text-lg font-medium">This frame</h2>
            <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 font-mono text-sm">
              <Row k="Kind" v={KIND_LABEL[frame.kind]} />
              <Row k="Shot" v={shot ? `${shot.kind} · ${formatTime(shot.t0)}–${formatTime(shot.t1)}` : "—"} />
              <Row k="Flux" v={frame.flux.toFixed(2)} />
              <Row k="Residual" v={frame.residual.toFixed(2)} />
              <Row
                k="Patch rank"
                v={
                  frame.rankMean != null
                    ? `mean K ${frame.rankMean.toFixed(2)} · ${frame.skipBlocks ?? 0} at K=0`
                    : "—"
                }
              />
              <Row k="PSNR" v={frame.psnr != null ? `${frame.psnr.toFixed(1)} dB` : "—"} />
            </dl>
            {frame.skipBlocks != null ? (
              <>
                <div className="mt-4 flex h-2 overflow-hidden rounded-full bg-bg-subtle">
                  <span
                    className="bg-moss"
                    style={{ width: `${(100 * frame.skipBlocks) / blocks}%` }}
                  />
                  <span
                    className="bg-copper"
                    style={{ width: `${(100 * (frame.residBlocks ?? 0)) / blocks}%` }}
                  />
                </div>
                <p className="mt-2 flex flex-wrap gap-x-3 font-mono text-xs text-fg-subtle">
                  <span><span className="text-moss">Moss</span> K=0 (shot mean only)</span>
                  <span><span className="text-copper">Copper</span> rank-K temporal</span>
                </p>
              </>
            ) : null}
            <p className="mt-4 text-sm text-pretty text-fg-muted">
              {frame.kind === "cut"
                ? "Histogram break. New shot, new temporal model. The previous patch bases do not carry over."
                : frame.key
                  ? "Shot start. The origin stores a JPEG of this shot's temporal mean, then rank-K factors on the leftover."
                  : frame.kind === "residual"
                    ? "High leftover after the patch SVD. These tiles sat at K_max and still could not fit a 16×16 that changes appearance inside the shot."
                    : frame.kind === "motion"
                      ? "Flux is high but the temporal model is doing the work — no warp, no MV. Rank K on this shot is what eats the motion."
                      : "Low flux. Most 16×16 tiles are K=0: the shot-mean JPEG already clears 32.5 dB."}
            </p>
          </section>

          <section className="rounded-xl bg-bg-elevated p-5 shadow-border">
            <h2 className="font-display text-lg font-medium">Play encode</h2>
            <p className="mt-2 text-sm text-pretty text-fg-muted">
              Playback rasterizes the model into a cache, then encodes a
              disposable stream at whatever the client asked for. Archive size
              does not change.
            </p>
            <div className="mt-5">
              <div className="flex justify-between font-mono text-xs text-fg-muted">
                <span>Delivery bitrate</span>
                <span className="tabular-nums text-fg">{bitrate} kbps</span>
              </div>
              <Slider
                className="mt-3"
                min={400}
                max={4000}
                step={50}
                value={[bitrate]}
                onValueChange={([v]) => setBitrate(v ?? 1200)}
              />
            </div>
            <dl className="mt-5 grid grid-cols-2 gap-3 font-mono text-sm">
              <Row k="Stream size (est.)" v={formatBytes(deliveryBytes)} />
              <Row k="Cache GOP" v={shot ? `${shot.i0}–${shot.i1}` : "—"} />
            </dl>
            <p className="mt-4 font-mono text-xs text-fg-subtle">
              Model {formatBytes(stats.modelBytes)} stays on disk. NVENC would
              eat the raster, not the origin.
            </p>
          </section>
        </div>

        <section className="mt-6 rounded-xl bg-bg-elevated p-5 shadow-border">
          <h2 className="font-display text-lg font-medium">How this attempt encodes</h2>
          <ol className="mt-4 grid gap-3 text-sm text-fg-muted sm:grid-cols-3">
            <li>
              <span className="block font-medium text-fg">1. No warp</span>
              v0–v3 predicted a frame from a previous recon. v4r does not.
              A histogram cut starts a new shot. Inside the shot, every 16×16
              is its own temporal model on the source pixels.
            </li>
            <li>
              <span className="block font-medium text-fg">2. Rank-K patch SVD</span>
              Shot-mean JPEG, then a thin SVD of the leftover. K grows until
              local MSE sits under 32.5 dB, cap 16. Int8 U and B, then two
              ALS steps — that is the training. Still patches stay at K=0.
            </li>
            <li>
              <span className="block font-medium text-fg">3. Packed origin</span>
              zlib of the factors plus the shot-mean JPEGs. Kill was PSNR
              ≈ 32 dB on a 10s probe, then the full 90s. Bytes are a meter,
              not the knife: spatial bases per shot are expensive.
            </li>
          </ol>
        </section>
      </main>
    </div>
  );
}

function Viewer({
  label,
  sub,
  src,
  videoRef,
  muted,
  overlay,
  badge,
  onActivate,
  onTime,
  onPlay,
  onPause,
}: {
  label: string;
  sub: string;
  src: string;
  videoRef: RefObject<HTMLVideoElement | null>;
  muted?: boolean;
  overlay?: string | null;
  badge?: string;
  onActivate?: () => void;
  onTime: (t: number) => void;
  onPlay?: () => void;
  onPause?: () => void;
}) {
  return (
    <figure className="overflow-hidden rounded-xl bg-bg-elevated shadow-border">
      <div
        className={cn("relative aspect-video bg-bg", onActivate && "cursor-pointer")}
        data-raster-toggle={onActivate ? "" : undefined}
        role={onActivate ? "button" : undefined}
        tabIndex={onActivate ? 0 : undefined}
        aria-label={onActivate ? `${label}. Click to switch raster.` : undefined}
        onClick={onActivate}
        onKeyDown={
          onActivate
            ? (e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  e.stopPropagation();
                  onActivate();
                }
              }
            : undefined
        }
      >
        <video
          ref={videoRef}
          src={src}
          className="pointer-events-none size-full object-contain"
          playsInline
          preload="metadata"
          muted={muted}
          onTimeUpdate={(e) => onTime(e.currentTarget.currentTime)}
          onPlay={onPlay}
          onPause={onPause}
          onEnded={() => onPause?.()}
        />
        {overlay ? (
          <img
            src={overlay}
            alt=""
            className="pointer-events-none absolute inset-0 size-full object-contain opacity-80 mix-blend-screen"
          />
        ) : null}
        {badge ? (
          <span className="pointer-events-none absolute top-2 left-2 rounded-sm bg-bg-elevated/85 px-2 py-1 font-mono text-xs text-fg">
            {badge}
          </span>
        ) : null}
      </div>
      <figcaption className="flex items-baseline justify-between gap-3 px-4 py-3">
        <span className="text-sm font-medium">{label}</span>
        <span className="font-mono text-xs text-fg-subtle">{sub}</span>
      </figcaption>
    </figure>
  );
}

function FluxScope({
  frames,
  shots,
  t,
  duration,
  onSeek,
}: {
  frames: FrameRow[];
  shots: ShotRow[];
  t: number;
  duration: number;
  onSeek: (t: number) => void;
}) {
  const W = 1000;
  const H = 112;
  const { flux, residual, keys } = useMemo(() => {
    const maxF = Math.max(...frames.map((f) => f.flux), 1);
    const maxR = Math.max(...frames.map((f) => f.residual), 1);
    const xOf = (time: number) => (time / duration) * W;
    const fluxPath = frames
      .map((f, i) => `${i === 0 ? "M" : "L"}${xOf(f.t).toFixed(2)} ${(H - (f.flux / maxF) * 86).toFixed(2)}`)
      .join(" ");
    const residualPath = frames
      .map(
        (f, i) =>
          `${i === 0 ? "M" : "L"}${xOf(f.t).toFixed(2)} ${(H - (f.residual / maxR) * 86).toFixed(2)}`,
      )
      .join(" ");
    const keyMarks = frames.filter((f) => f.key).map((f) => ({ x: xOf(f.t), cut: f.cut }));
    return { flux: fluxPath, residual: residualPath, keys: keyMarks };
  }, [frames, duration]);

  const playX = (t / duration) * W;

  return (
    <div className="mt-4">
      <button
        type="button"
        className="relative block w-full overflow-hidden rounded-md"
        onClick={(e) => {
          const r = e.currentTarget.getBoundingClientRect();
          const x = (e.clientX - r.left) / r.width;
          onSeek(x * duration);
        }}
        aria-label="Seek on flux scope"
      >
        <svg
          viewBox={`0 0 ${W} ${H}`}
          className="block h-28 w-full"
          preserveAspectRatio="none"
          role="img"
          aria-hidden="true"
        >
          <rect width={W} height={H} fill="#181a20" />
          {shots.map((s) => (
            <rect
              key={`${s.i0}-${s.i1}`}
              x={(s.t0 / duration) * W}
              y={0}
              width={Math.max(1, ((s.t1 - s.t0) / duration) * W)}
              height={10}
              fill={s.kind === "busy" ? "#2a2422" : s.kind === "tracking" ? "#1c2228" : "#1a1c22"}
            />
          ))}
          <path d={flux} fill="none" stroke="#8a9aaa" strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
          <path d={residual} fill="none" stroke="#c45c4a" strokeWidth="1.6" vectorEffect="non-scaling-stroke" />
          {keys.map((k) => (
            <rect
              key={k.x}
              x={k.x}
              y={10}
              width={1.4}
              height={102}
              fill={k.cut ? "#c45c4a" : "#e7e2d6"}
              opacity={0.85}
            />
          ))}
          <rect x={playX} y={0} width={2} height={H} fill="#d8d2c4" />
        </svg>
      </button>
      <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-fg-subtle">
        <span>
          <span className="text-steel">Steel</span> raw flux
        </span>
        <span>
          <span className="text-copper">Copper</span> leftover after the model
        </span>
        <span>Ticks are shot starts</span>
      </p>
    </div>
  );
}

function Filmstrip({
  frames,
  indices,
  t,
  onSeek,
}: {
  frames: FrameRow[];
  indices: number[];
  t: number;
  onSeek: (t: number) => void;
}) {
  return (
    <div className="mt-3 flex gap-1 overflow-x-auto pb-1">
      {indices.map((i) => {
        const f = frames[i];
        const active = Math.abs(f.t - t) < 0.55;
        return (
          <button
            key={i}
            type="button"
            onClick={() => onSeek(f.t)}
            className={cn(
              "relative h-12 w-20 shrink-0 overflow-hidden rounded-sm",
              active ? "ring-1 ring-accent" : "opacity-70 hover:opacity-100",
            )}
          >
            <img
              src={`/media/thumbs/${String(i).padStart(4, "0")}.jpg`}
              alt=""
              className="size-full object-cover"
            />
            {f.key ? <span className="absolute top-0.5 right-0.5 size-1.5 rounded-full bg-accent" /> : null}
          </button>
        );
      })}
    </div>
  );
}

function StatCard({ label, value, hint }: { label: string; value: string; hint: string }) {
  return (
    <div className="rounded-xl bg-bg-elevated p-4 shadow-border">
      <p className="text-xs tracking-wide text-fg-subtle uppercase">{label}</p>
      <p className="font-display mt-2 text-2xl font-medium tabular-nums tracking-tight">{value}</p>
      <p className="mt-1 text-xs text-fg-muted">{hint}</p>
    </div>
  );
}

function Mini({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg bg-bg-elevated px-3 py-3 shadow-border">
      <p className="text-xs text-fg-subtle">{label}</p>
      <p className="font-mono mt-1 text-lg tabular-nums">{value}</p>
    </div>
  );
}

function Row({ k, v }: { k: string; v: string }) {
  return (
    <div>
      <dt className="text-xs text-fg-subtle">{k}</dt>
      <dd className="mt-0.5 text-fg">{v}</dd>
    </div>
  );
}
