// Real-ESRGAN x4plus（ONNX Runtime Web、WebGPU／だめなら WASM）を別スレッドで動かす
import { planTiles, rgbaToChw, chwToRgba, pickTileSize, isBadOutput } from './tiles.js';

export const MODEL_URL = new URL('../models/realesrgan-x4plus.onnx', import.meta.url).href;
const SCALE = 4;

let upscalerPromise = null;
let worker = null;
let workerWasm = false;
// うまくいかなかった設定は覚えておき（端末に保存）、次からは最初からそちらを使う
const LEARN_KEY = 'photo-upscaler-learned-v2';
const learned = (() => {
  try { return { forceWasm: false, tile: 0, ...JSON.parse(localStorage.getItem(LEARN_KEY) || '{}') }; } catch { return { forceWasm: false, tile: 0 }; }
})();
const saveLearned = () => { try { localStorage.setItem(LEARN_KEY, JSON.stringify(learned)); } catch { /* なし */ } };
/** 覚えた設定を忘れる（GPUをもう一度試すとき） */
export function forgetLearned() { learned.forceWasm = false; learned.tile = 0; saveLearned(); }
export const learnedState = () => ({ ...learned });
// WebGPU は最初からこの大きさ以下で始める（iPad で GPU 側が落ちるのを避けるため）
const GPU_TILE_MAX = 96;

/** AIを捨てる（GPUが使えなくなったときに作り直すため） */
export function resetUpscaler() {
  worker?.terminate();
  worker = null;
  upscalerPromise = null;
}

/** AIを準備する（1回だけ）。返り値 { backend, gpuLimit, run(chw, w, h) } */
export function loadUpscaler({ onProgress, preferWasm = false } = {}) {
  if (upscalerPromise && workerWasm !== preferWasm) resetUpscaler();
  if (!upscalerPromise) {
    workerWasm = preferWasm;
    upscalerPromise = new Promise((resolve, reject) => {
      worker = new Worker(new URL('./ai-worker.js', import.meta.url), { type: 'module' });
      const pending = new Map();
      let nextId = 1;
      const run = (data, w, h) => new Promise((res, rej) => {
        const id = nextId++;
        pending.set(id, { res, rej });
        worker.postMessage({ type: 'run', id, data, w, h }, [data.buffer]);
      });
      worker.onmessage = ({ data: m }) => {
        if (m.type === 'progress') onProgress?.(m.got, m.total);
        else if (m.type === 'ready') resolve({ backend: m.backend, gpuLimit: m.gpuLimit, run });
        else if (m.type === 'result') { pending.get(m.id)?.res(m); pending.delete(m.id); }
        else if (m.type === 'error') {
          if (m.id && pending.has(m.id)) { pending.get(m.id).rej(new Error(m.message)); pending.delete(m.id); }
          else reject(new Error(m.message));
        }
      };
      worker.onerror = (e) => reject(new Error(e.message || 'AIの準備に失敗しました'));
      const debug = new URLSearchParams(location.search).get('debug');
      worker.postMessage({ type: 'load', url: MODEL_URL, preferWasm, debug });
    });
    upscalerPromise.catch(() => { upscalerPromise = null; });
  }
  return upscalerPromise;
}

/**
 * input（canvas）を4倍にし、dest（2D context）の destRect に描く。
 * 4倍の大きな画像は作らず、タイルごとに縮小して描くので Safari のキャンバス上限に引っかからない。
 */
export async function upscaleInto(up, { rgba, w, h }, dest, destRect, { tile = 256, pad = 16, onTile, signal } = {}) {
  const tiles = planTiles(w, h, tile, pad);
  const k = destRect.w / w; // AI入力 → 描き先 の倍率
  const tmp = document.createElement('canvas');
  const tctx = tmp.getContext('2d', { willReadFrequently: true });
  dest.save();
  dest.imageSmoothingEnabled = true;
  dest.imageSmoothingQuality = 'high';
  try {
    for (let i = 0; i < tiles.length; i++) {
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      const { core, padded } = tiles[i];
      const y = await up.run(rgbaToChw(rgba, w, padded), padded.w, padded.h);
      if (isBadOutput(y.data)) throw new Error('AIの結果が正しくありません（真っ黒、または計算不能な値）');
      // 境目の継ぎ目を目立たせないよう、中身より1px（入力換算）広く描く
      const ex = { x: Math.max(padded.x, core.x - 1), y: Math.max(padded.y, core.y - 1) };
      ex.w = Math.min(padded.x + padded.w, core.x + core.w + 1) - ex.x;
      ex.h = Math.min(padded.y + padded.h, core.y + core.h + 1) - ex.y;
      const r = { x: (ex.x - padded.x) * SCALE, y: (ex.y - padded.y) * SCALE, w: ex.w * SCALE, h: ex.h * SCALE };
      tmp.width = r.w; tmp.height = r.h;
      tctx.putImageData(new ImageData(chwToRgba(y.data, y.w, y.h, r), r.w, r.h), 0, 0);
      dest.drawImage(tmp, destRect.x + ex.x * k, destRect.y + ex.y * k, ex.w * k, ex.h * k);
      onTile?.(i + 1, tiles.length);
    }
  } finally {
    dest.restore();
  }
}

/** 描いた結果が消えていないか（GPU側が落ちるとキャンバスが透明になることがある） */
function looksBlank(ctx, w, h) {
  const pts = [[0.5, 0.5], [0.25, 0.25], [0.75, 0.75], [0.25, 0.75], [0.75, 0.25]];
  return pts.every(([fx, fy]) => ctx.getImageData(Math.floor(w * fx), Math.floor(h * fy), 1, 1).data[3] === 0);
}

/**
 * 失敗しても自動でやり直す版。
 * mode: 'auto'（GPU→だめならCPU）／'gpu'（GPUのみ）／'cpu'（CPU＝WASMのみ）
 * WebGPU：小さめのタイルで始め、失敗したらタイルを半分にしてやり直す。
 * 自動のときは、それでも失敗したら WASM（遅いが確実）でやり直し、次からも WASM を使う。
 * input は元のキャンバス。GPU側が落ちても困らないよう、最初に画素を読み取っておく。
 * 返り値 { backend, tile }
 */
export async function upscaleRobust(input, dest, destRect, { mode = 'auto', onProgress, onStatus, onTile, signal } = {}) {
  const src = { rgba: input.getContext('2d').getImageData(0, 0, input.width, input.height).data, w: input.width, h: input.height };
  const wantWasm = () => mode === 'cpu' || (mode === 'auto' && learned.forceWasm);
  let up = await loadUpscaler({ onProgress, preferWasm: wantWasm() });
  const startTile = () => (up.backend === 'webgpu' ? Math.min(GPU_TILE_MAX, learned.tile || pickTileSize(up.gpuLimit)) : 256);
  let tile = startTile();
  for (;;) {
    try {
      await upscaleInto(up, src, dest, destRect, { tile, onTile, signal });
      if (looksBlank(dest, destRect.w, destRect.h)) throw new Error('AIの結果が消えてしまいました');
      if (up.backend === 'webgpu') { learned.tile = tile; saveLearned(); }
      return { backend: up.backend, tile };
    } catch (e) {
      if (e.name === 'AbortError' || up.backend !== 'webgpu') throw e;
      console.warn('WebGPU failed', tile, e);
      resetUpscaler(); // GPUが使えなくなっていることがあるので作り直す
      if (tile > 32) {
        tile = Math.max(32, Math.floor(tile / 2));
        learned.tile = tile; saveLearned();
        onStatus?.(`GPUの処理に失敗したため、もっと小さく分けてやり直します（${tile}）…`);
        up = await loadUpscaler({ onProgress, preferWasm: false });
      } else if (mode === 'gpu') {
        throw new Error(`GPUでは処理できませんでした（${e.message}）。「処理：CPU」を選んでください`);
      } else {
        learned.forceWasm = true; saveLearned();
        onStatus?.('GPUでの処理をやめ、CPU（WASM）でやり直します。時間がかかります…');
        up = await loadUpscaler({ onProgress, preferWasm: true });
        tile = 256;
      }
    }
  }
}
