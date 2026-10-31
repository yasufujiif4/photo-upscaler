// 顔補正：顔を見つけ（RetinaFace）、512×512 にそろえて切り出し、CodeFormer で補正して元の位置に戻す
import { AiWorker } from './ai-client.js';
import { decodeFaces, estimateSimilarity, invertAffine, composeAffine, FACE_TEMPLATE_512 } from './facemath.js';
import { blurRGBA } from './blur.js';

const DET_URL = new URL('../models/retinaface-mnet025.onnx', import.meta.url).href;
const CF_URLS = ['../models/codeformer.onnx.part0', '../models/codeformer.onnx.part1'].map((u) => new URL(u, import.meta.url).href);
const DET_BYTES = 1733771, CF_BYTES = 188852241;
const FACE = 512;
const DET_MAX = 1024; // 顔を探すときは、長い辺をこの大きさまで縮小する

let facePromise = null;

/**
 * 顔補正のAIを準備する（1回だけ）。
 * CodeFormer は中間データが大きく、iPad の GPU では不安定なので、いつも CPU（WASM）で動かす。
 */
export function loadFaceAi({ onProgress } = {}) {
  if (!facePromise) {
    const ai = new AiWorker();
    const total = DET_BYTES + CF_BYTES;
    facePromise = (async () => {
      await ai.load('det', [DET_URL], { preferWasm: true, totalHint: total, onProgress });
      await ai.load('cf', CF_URLS, { preferWasm: true, totalHint: total, onProgress: (got) => onProgress?.(DET_BYTES + got, total) });
      return ai;
    })();
    facePromise.catch(() => { ai.terminate(); facePromise = null; });
  }
  return facePromise;
}

/** 元の写真の rect の範囲から顔を探す。座標は元の写真のピクセル。 */
export async function findFaces(ai, src, rect) {
  const s = Math.min(1, DET_MAX / Math.max(rect.w, rect.h));
  const w = Math.max(32, Math.round(rect.w * s)), h = Math.max(32, Math.round(rect.h * s));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, rect.x, rect.y, rect.w, rect.h, 0, 0, w, h);
  const rgba = ctx.getImageData(0, 0, w, h).data;
  const n = w * h;
  const chw = new Float32Array(3 * n); // BGR、平均を引く（0〜255）
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    chw[i] = rgba[j + 2] - 104; chw[i + n] = rgba[j + 1] - 117; chw[i + 2 * n] = rgba[j] - 123;
  }
  const out = await ai.run('det', { input: { data: chw, dims: [1, 3, h, w] } });
  const sx = rect.w / w, sy = rect.h / h;
  return decodeFaces(out.loc.data, out.conf.data, out.landms.data, w, h).map((f) => ({
    score: f.score,
    box: [f.box[0] * sx + rect.x, f.box[1] * sy + rect.y, f.box[2] * sx + rect.x, f.box[3] * sy + rect.y],
    pts: f.pts.map(([x, y]) => [x * sx + rect.x, y * sy + rect.y]),
  }));
}

/** 顔を 512×512 にそろえて切り出す。返り値 { M（元の写真→512）, chw（RGB, -1〜1） } */
export function alignFace(src, pts) {
  const M = estimateSimilarity(pts, FACE_TEMPLATE_512);
  const c = document.createElement('canvas');
  c.width = FACE; c.height = FACE;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = 'rgb(132,133,135)';
  ctx.fillRect(0, 0, FACE, FACE);
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.setTransform(...M);
  ctx.drawImage(src, 0, 0);
  const rgba = ctx.getImageData(0, 0, FACE, FACE).data;
  const n = FACE * FACE;
  const chw = new Float32Array(3 * n);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    chw[i] = rgba[j] / 127.5 - 1; chw[i + n] = rgba[j + 1] / 127.5 - 1; chw[i + 2 * n] = rgba[j + 2] / 127.5 - 1;
  }
  return { M, chw };
}

/** CodeFormer で補正する。w：0＝きれいさ優先 〜 1＝本人らしさ優先。返り値 ImageData（512×512） */
export async function restoreFace(ai, chw, w) {
  const out = await ai.run('cf', {
    input: { data: chw.slice(), dims: [1, 3, FACE, FACE] }, // 何度も使うので複製して渡す
    w: { data: new Float32Array([w]), dims: [1] },
  });
  const d = out.output.data;
  const n = FACE * FACE;
  const rgba = new Uint8ClampedArray(n * 4);
  for (let i = 0, j = 0; i < n; i++, j += 4) {
    rgba[j] = (d[i] + 1) * 127.5; rgba[j + 1] = (d[i + n] + 1) * 127.5; rgba[j + 2] = (d[i + 2 * n] + 1) * 127.5; rgba[j + 3] = 255;
  }
  return new ImageData(rgba, FACE, FACE);
}

let maskCanvas = null;
/** 貼り戻すときのぼかした縁（四角の内側を残し、外側に向かってなめらかに薄くする） */
function faceMask() {
  if (maskCanvas) return maskCanvas;
  const S = 128, inset = 14, r = 4;
  const rgba = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const inside = x >= inset && x < S - inset && y >= inset && y < S - inset;
    rgba[(y * S + x) * 4 + 3] = inside ? 255 : 0;
  }
  const small = document.createElement('canvas');
  small.width = S; small.height = S;
  small.getContext('2d').putImageData(new ImageData(blurRGBA(rgba, S, S, r), S, S), 0, 0);
  maskCanvas = document.createElement('canvas');
  maskCanvas.width = FACE; maskCanvas.height = FACE;
  const mctx = maskCanvas.getContext('2d');
  mctx.imageSmoothingEnabled = true;
  mctx.imageSmoothingQuality = 'high';
  mctx.drawImage(small, 0, 0, FACE, FACE);
  return maskCanvas;
}

/**
 * 補正した顔を dst に貼り戻す。
 * M：元の写真→512、toDst：元の写真→dst の変換
 */
export function pasteFace(dst, restored, M, toDst) {
  const fc = document.createElement('canvas');
  fc.width = FACE; fc.height = FACE;
  const fctx = fc.getContext('2d', { willReadFrequently: true });
  fctx.putImageData(restored, 0, 0);
  fctx.globalCompositeOperation = 'destination-in';
  fctx.drawImage(faceMask(), 0, 0);
  dst.save();
  dst.imageSmoothingEnabled = true;
  dst.imageSmoothingQuality = 'high';
  dst.setTransform(...composeAffine(toDst, invertAffine(M)));
  dst.drawImage(fc, 0, 0);
  dst.restore();
}
