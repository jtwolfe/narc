export type FrameKind =
  | "keyframe"
  | "cut"
  | "motion"
  | "residual"
  | "static"
  | "flash"
  | "grain";

export type ShotKind = "locked" | "tracking" | "busy";

export type FrameRow = {
  i: number;
  t: number;
  flux: number;
  motion: number;
  residual: number;
  occlusion: number;
  hist: number;
  luma: number;
  lumaJump: number;
  dx: number;
  dy: number;
  scale?: number;
  rot?: number;
  kind: FrameKind;
  key: boolean;
  cut: boolean;
  storedResidual: boolean;
  skipBlocks?: number;
  residBlocks?: number;
  intraBlocks?: number;
  netBlocks?: number;
  splitBlocks?: number;
  cu8Blocks?: number;
  cu4Blocks?: number;
  rankMean?: number;
  ref?: number;
  psnr?: number;
};

export type ShotRow = {
  i0: number;
  i1: number;
  t0: number;
  t1: number;
  kind: ShotKind;
  keys: number;
};

export type BaselineStats = {
  attempt: string;
  fps: number;
  frames: number;
  keyframes: number;
  residualsStored: number;
  modelBytes: number;
  meanResidual: number;
  meanPsnr?: number;
  skipBlockFrac?: number;
  reconstructMp4Bytes: number;
  residualBytes?: number;
  intraBytes?: number;
  netBytes?: number;
  bitstreamBytes?: number;
  rawAccountedBytes?: number;
  gzipControlBytes?: number;
};

export type Analysis = {
  attempt?: string;
  frames: FrameRow[];
  shots: ShotRow[];
  stats: {
    frames: number;
    fps: number;
    width: number;
    height: number;
    duration: number;
    startSec: number;
    shots: number;
    keyframes: number;
    cuts: number;
    residualsStored: number;
    sourceBytes: number;
    modelBytes: number;
    keyframeBytes: number;
    residualBytes: number;
    motionBytes: number;
    intraBytes?: number;
    netBytes?: number;
    bitstreamBytes?: number;
    rawAccountedBytes?: number;
    gzipControlBytes?: number;
    syntaxBytes?: number;
    syntaxZlibBytes?: number;
    rawBytes: number;
    reconstructMp4Bytes: number;
    meanFlux: number;
    meanResidual: number;
    meanMotion: number;
    meanScale?: number;
    meanAbsRot?: number;
    meanPsnr?: number;
    medianPsnr?: number;
    minPsnr?: number;
    skipBlockFrac?: number;
    residBlockFrac?: number;
    intraBlockFrac?: number;
    netBlockFrac?: number;
    splitFrac?: number;
    cu16Count?: number;
    cu8Count?: number;
    cu4Count?: number;
    ratioVsRaw: number;
    ratioVsSource: number;
    block?: number[];
    blocksPerFrame?: number;
    meanRank?: number;
    kHist?: Record<string, number>;
    kMax?: number;
    targetPsnr?: number;
    trainSteps?: number;
    basisBytes?: number;
    coeffBytes?: number;
    meanJpegBytes?: number;
    atlasBytes?: number;
    leftoverBytes?: number;
    atlasQ?: number;
    leftoverQ?: number;
    leftoverStride?: number;
    kPrime?: Record<string, { meanPsnr: number; minPsnr: number }>;
    kPrimeLeft?: Record<string, { meanPsnr: number; minPsnr: number }>;
    kinds?: Record<string, number>;
    lambda?: number;
    skipMae?: number;
    blackTileFrames?: number;
    attempt?: string;
    baseline?: BaselineStats;
    baselineV1?: BaselineStats;
    baselineV11?: BaselineStats;
    baselineV12?: BaselineStats;
    baselineV2?: BaselineStats;
    baselineV3?: BaselineStats;
    baselineV4?: BaselineStats;
    baselineV4r?: BaselineStats;
  };
  source: {
    clip: string;
    clipAnalysis?: string;
    reconstruct: string;
    reconstructV4?: string;
    reconstructV4r?: string;
    reconstructV0?: string;
    reconstructV1?: string;
    reconstructV11?: string;
    reconstructV12?: string;
    reconstructV2?: string;
    reconstructV3?: string;
    reconstructKPrime?: Record<string, string>;
    reconstructKPrimeLeft?: Record<string, string>;
    scope: string;
    duration: number;
    startSec: number;
    title: string;
    credit: string;
    window: string;
  };
};

export function frameAtTime(frames: FrameRow[], t: number): FrameRow {
  if (!frames.length) {
    return {
      i: 0,
      t: 0,
      flux: 0,
      motion: 0,
      residual: 0,
      occlusion: 0,
      hist: 1,
      luma: 0,
      lumaJump: 0,
      dx: 0,
      dy: 0,
      kind: "static",
      key: true,
      cut: false,
      storedResidual: false,
    };
  }
  const fps = frames.length > 1 ? 1 / (frames[1].t - frames[0].t || 0.1) : 10;
  const i = Math.min(frames.length - 1, Math.max(0, Math.round(t * fps)));
  return frames[i] ?? frames[0];
}
