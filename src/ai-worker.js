// AIの計算を画面とは別のスレッドで行う（画面が固まらないように）。複数のモデルを名前で扱う
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE_NAME = 'photo-upscaler-models-v1';
const sessions = {};
let debugZero = false; // 試験用：エラーにならずに 0 だけ返る（iPad で起きた状況）
let debugGpuFail = 0;  // 試験用：WebGPU の失敗をまねる（この大きさを超えるタイルは失敗）

/** 1ファイルを読み込む（2回目以降は端末に保存したものを使う） */
async function fetchOne(url, onBytes) {
  let cache = null;
  try { cache = await caches.open(CACHE_NAME); } catch { /* 使えない環境 */ }
  const hit = cache && (await cache.match(url));
  if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); onBytes(b.length, true); return b; }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`AIデータを読み込めませんでした（${res.status}）`);
  const reader = res.body.getReader();
  const chunks = []; let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); got += value.length; onBytes(value.length, false);
  }
  const buf = new Uint8Array(got);
  let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(buf, { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* 容量不足など */ }
  return buf;
}

/** 分割されたファイルを読み込んでつなげる（GitHub の1ファイル100MBの上限のため） */
async function fetchModel(urls, totalHint, id) {
  let got = 0, last = 0;
  const parts = [];
  for (const url of urls) {
    parts.push(await fetchOne(url, (n, cached) => {
      got += n;
      if (!cached && got - last > 1e6) { last = got; postMessage({ type: 'progress', id, got, total: totalHint }); }
    }));
  }
  if (parts.length === 1) return parts[0];
  const buf = new Uint8Array(parts.reduce((a, p) => a + p.length, 0));
  let o = 0; for (const p of parts) { buf.set(p, o); o += p.length; }
  return buf;
}

/** WebGPU で1つのデータに使える上限（バイト） */
async function gpuLimitBytes() {
  let lim = null;
  try { const dev = await ort.env.webgpu?.device; if (dev?.limits) lim = dev.limits; } catch { /* 取れない版 */ }
  if (!lim) { try { lim = (await navigator.gpu.requestAdapter())?.limits; } catch { /* なし */ } }
  if (!lim) return 128 * 1024 * 1024;
  return Math.min(Number(lim.maxStorageBufferBindingSize) || Infinity, Number(lim.maxBufferSize) || Infinity);
}

async function load({ id, name, urls, totalHint, preferWasm, debug }) {
  const bytes = await fetchModel(urls, totalHint, id);
  const order = !preferWasm && 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr;
  for (const ep of order) {
    try {
      sessions[name] = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      const info = { backend: ep, gpuLimit: ep === 'webgpu' ? await gpuLimitBytes() : 0 };
      // 試験用（?debug=gpufail / gpufailall / gpuzero）：WASM で動かしつつ「WebGPU で失敗する」ふりをする
      if (name === 'esrgan') {
        debugGpuFail = preferWasm ? 0 : { gpufail: 100, gpufailall: 1, gpuzero: 1 }[debug] || 0;
        debugZero = debug === 'gpuzero' && !preferWasm;
        if (debugGpuFail) Object.assign(info, { backend: 'webgpu', gpuLimit: 512 * 1024 * 1024 });
      }
      return info;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

async function run({ id, name, feeds }) {
  if (name === 'esrgan') {
    const { dims } = feeds.input;
    if (debugZero) {
      const z = new Float32Array(3 * dims[2] * dims[3] * 16);
      return { output: { data: z, dims: [1, 3, dims[2] * 4, dims[3] * 4] } };
    }
    if (debugGpuFail && Math.max(dims[2], dims[3]) > debugGpuFail) throw new Error('simulated: map async was not successful');
  }
  const tensors = {};
  for (const [k, v] of Object.entries(feeds)) tensors[k] = new ort.Tensor(v.type || 'float32', v.data, v.dims);
  const out = await sessions[name].run(tensors);
  const result = {};
  for (const [k, t] of Object.entries(out)) {
    const data = await t.getData();
    result[k] = { data: data.buffer.byteLength === data.byteLength ? data : data.slice(), dims: t.dims };
    t.dispose?.();
  }
  for (const t of Object.values(tensors)) t.dispose?.();
  return result;
}

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'load') {
      postMessage({ type: 'ready', id: msg.id, ...(await load(msg)) });
    } else if (msg.type === 'run') {
      const outputs = await run(msg);
      postMessage({ type: 'result', id: msg.id, outputs }, Object.values(outputs).map((o) => o.data.buffer));
    }
  } catch (e) {
    postMessage({ type: 'error', id: msg.id, message: e?.message || String(e) });
  }
};
