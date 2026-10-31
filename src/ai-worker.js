// AIの計算を画面とは別のスレッドで行う（画面が固まらないように）
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE_NAME = 'photo-upscaler-models-v1';
let session = null;
let debugZero = false; // 試験用：エラーにならずに 0 だけ返る（iPad で起きた状況）
let debugGpuFail = 0; // 試験用：WebGPU の失敗をまねる（この大きさを超えるタイルは失敗）

/** モデルを読み込む（2回目以降は端末に保存したものを使う） */
async function fetchModel(url) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* 使えない環境 */ }
  const hit = cache && (await cache.match(url));
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`AIデータを読み込めませんでした（${res.status}）`);
  const total = Number(res.headers.get('content-length')) || 0;
  const reader = res.body.getReader();
  const chunks = []; let got = 0; let last = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length;
    if (got - last > 1e6) { last = got; postMessage({ type: 'progress', got, total }); }
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* 容量不足など */ }
  return buf;
}

/** WebGPU で1つのデータに使える上限（バイト） */
async function gpuLimitBytes() {
  let lim = null;
  try {
    const dev = await ort.env.webgpu?.device;
    if (dev?.limits) lim = dev.limits;
  } catch { /* 取れない版 */ }
  if (!lim) {
    try { lim = (await navigator.gpu.requestAdapter())?.limits; } catch { /* なし */ }
  }
  if (!lim) return 128 * 1024 * 1024;
  return Math.min(Number(lim.maxStorageBufferBindingSize) || Infinity, Number(lim.maxBufferSize) || Infinity);
}

async function load({ url, preferWasm }) {
  const bytes = await fetchModel(url);
  const order = !preferWasm && 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr;
  for (const ep of order) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      return { backend: ep, gpuLimit: ep === 'webgpu' ? await gpuLimitBytes() : 0 };
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'load') {
      const info = await load(msg);
      // 試験用（?debug=gpufail / gpufailall / gpuzero）：WASM で動かしつつ「WebGPU で大きなタイルが失敗する」ふりをする
      debugGpuFail = msg.preferWasm ? 0 : { gpufail: 100, gpufailall: 1, gpuzero: 1 }[msg.debug] || 0;
      debugZero = msg.debug === 'gpuzero' && !msg.preferWasm;
      if (debugGpuFail) Object.assign(info, { backend: 'webgpu', gpuLimit: 512 * 1024 * 1024 });
      postMessage({ type: 'ready', ...info });
    } else if (msg.type === 'run') {
      if (debugZero) {
        const z = new Float32Array(3 * msg.w * msg.h * 16);
        postMessage({ type: 'result', id: msg.id, data: z, w: msg.w * 4, h: msg.h * 4 }, [z.buffer]);
        return;
      }
      if (debugGpuFail && Math.max(msg.w, msg.h) > debugGpuFail) throw new Error('simulated: map async was not successful');
      const x = new ort.Tensor('float32', msg.data, [1, 3, msg.h, msg.w]);
      const out = await session.run({ input: x });
      const y = out.output;
      const data = await y.getData();
      const copy = data.buffer.byteLength === data.byteLength ? data : data.slice();
      postMessage({ type: 'result', id: msg.id, data: copy, w: y.dims[3], h: y.dims[2] }, [copy.buffer]);
      x.dispose?.(); y.dispose?.();
    }
  } catch (e) {
    postMessage({ type: 'error', id: msg.id, message: e?.message || String(e) });
  }
};
