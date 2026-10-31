// AIの計算を画面とは別のスレッドで行う（画面が固まらないように）
import * as ort from '../vendor/ort/ort.webgpu.min.mjs';

ort.env.wasm.wasmPaths = new URL('../vendor/ort/', import.meta.url).href;
ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

const CACHE_NAME = 'photo-upscaler-models-v1';
let session = null;

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

async function load({ url, preferWasm }) {
  const bytes = await fetchModel(url);
  const order = !preferWasm && 'gpu' in navigator ? ['webgpu', 'wasm'] : ['wasm'];
  let lastErr;
  for (const ep of order) {
    try {
      session = await ort.InferenceSession.create(bytes, { executionProviders: [ep], graphOptimizationLevel: 'all' });
      return ep;
    } catch (e) { lastErr = e; }
  }
  throw lastErr;
}

self.onmessage = async ({ data: msg }) => {
  try {
    if (msg.type === 'load') {
      const backend = await load(msg);
      postMessage({ type: 'ready', backend });
    } else if (msg.type === 'run') {
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
