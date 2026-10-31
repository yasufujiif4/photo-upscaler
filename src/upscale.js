// Real-ESRGAN x4plus（ONNX Runtime Web、WebGPU／だめなら WASM）を別スレッドで動かす
import { planTiles, rgbaToChw, chwToRgba, pickTileSize } from './tiles.js';

export const MODEL_URL = new URL('../models/realesrgan-x4plus.onnx', import.meta.url).href;
const SCALE = 4;

let upscalerPromise = null;
let worker = null;
// 一度うまくいかなかった設定は覚えておき、次からは最初からそちらを使う
const learned = { forceWasm: false, tile: 0 };

/** AIを捨てる（GPUが使えなくなったときに作り直すため） */
export function resetUpscaler() {
  worker?.terminate();
  worker = null;
  upscalerPromise = null;
}

/** AIを準備する（1回だけ）。返り値 { backend, gpuLimit, run(chw, w, h) } */
export function loadUpscaler({ onProgress, preferWasm = false } = {}) {
  if (!upscalerPromise) {
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
      worker.postMessage({ type: 'load', url: MODEL_URL, preferWasm: preferWasm || learned.forceWasm, debug });
    });
    upscalerPromise.catch(() => { upscalerPromise = null; });
  }
  return upscalerPromise;
}

/**
 * input（canvas）を4倍にし、dest（2D context）の destRect に描く。
 * 4倍の大きな画像は作らず、タイルごとに縮小して描くので Safari のキャンバス上限に引っかからない。
 */
export async function upscaleInto(up, input, dest, destRect, { tile = 256, pad = 16, onTile, signal } = {}) {
  const w = input.width, h = input.height;
  const rgba = input.getContext('2d').getImageData(0, 0, w, h).data;
  const tiles = planTiles(w, h, tile, pad);
  const k = destRect.w / w; // AI入力 → 描き先 の倍率
  const tmp = document.createElement('canvas');
  const tctx = tmp.getContext('2d');
  dest.save();
  dest.imageSmoothingEnabled = true;
  dest.imageSmoothingQuality = 'high';
  try {
    for (let i = 0; i < tiles.length; i++) {
      if (signal?.aborted) throw new DOMException('中止しました', 'AbortError');
      const { core, padded } = tiles[i];
      const y = await up.run(rgbaToChw(rgba, w, padded), padded.w, padded.h);
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

/**
 * 失敗しても自動でやり直す版。
 * WebGPU：上限に収まるタイルで始め、失敗したらタイルを半分にしてやり直す。
 * それでも失敗したら WASM（遅いが確実）でやり直す。
 * 返り値 { backend, tile }
 */
export async function upscaleRobust(input, dest, destRect, { onProgress, onStatus, onTile, signal } = {}) {
  let up = await loadUpscaler({ onProgress });
  const startTile = () => (up.backend === 'webgpu' ? learned.tile || pickTileSize(up.gpuLimit) : 256);
  let tile = startTile();
  for (;;) {
    try {
      await upscaleInto(up, input, dest, destRect, { tile, onTile, signal });
      if (up.backend === 'webgpu') learned.tile = tile;
      return { backend: up.backend, tile };
    } catch (e) {
      if (e.name === 'AbortError' || up.backend !== 'webgpu') throw e;
      console.warn('WebGPU failed', tile, e);
      resetUpscaler(); // GPUが使えなくなっていることがあるので作り直す
      if (tile > 48) {
        tile = Math.max(48, Math.floor(tile / 2));
        learned.tile = tile;
        onStatus?.(`GPUの処理に失敗したため、もっと小さく分けてやり直します…`);
        up = await loadUpscaler({ onProgress });
        if (up.backend !== 'webgpu') tile = startTile();
      } else {
        learned.forceWasm = true;
        onStatus?.('GPUでの処理をやめ、CPU（WASM）でやり直します。時間がかかります…');
        up = await loadUpscaler({ onProgress });
        tile = 256;
      }
    }
  }
}
