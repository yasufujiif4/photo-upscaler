import {
  PRINT_SIZES, BLEED_MM, mmToPx, outputSize, initialPlacement, clampPlacement, zoomAt,
  visibleSourceRect,
} from './layout.js';
import { renderComposite } from './render.js';
import { setJpegDpi } from './jpeg.js';
import { aiInputScale } from './tiles.js';

const $ = (id) => document.getElementById(id);
const view = $('view');
const vctx = view.getContext('2d');

const state = {
  src: null,        // ImageBitmap
  img: null,        // { w, h }
  name: 'photo',
  sizeKey: '2L',
  landscape: false,
  out: outputSize('2L'),
  placement: null,
  enhanced: null,   // { canvas, rect, key } AI処理後の前景
  backend: '',
  busy: false,
};

// 配置・サイズ・縮小の組み合わせ。これが変わったらAIの結果は使えない
const placementKey = () => JSON.stringify([state.out, state.placement, $('shrink').value]);
const currentEnhanced = () => (state.enhanced && state.enhanced.key === placementKey() ? state.enhanced : null);

// ---------- 表示 ----------
let viewScale = 1; // 書き出しピクセル → 画面ピクセル（CSS px）

function layoutView() {
  const stage = $('stage').getBoundingClientRect();
  const pad = 16;
  viewScale = Math.min((stage.width - pad * 2) / state.out.w, (stage.height - pad * 2) / state.out.h);
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(state.out.w * viewScale), ch = Math.round(state.out.h * viewScale);
  view.style.width = `${cw}px`; view.style.height = `${ch}px`;
  view.width = Math.round(cw * dpr); view.height = Math.round(ch * dpr);
  draw();
}

let raf = 0;
function draw() {
  if (raf) return;
  raf = requestAnimationFrame(() => { raf = 0; drawNow(); });
}

function drawNow() {
  const k = view.width / state.out.w;
  vctx.setTransform(1, 0, 0, 1, 0, 0);
  vctx.fillStyle = '#fff';
  vctx.fillRect(0, 0, view.width, view.height);
  if (!state.src) return;
  vctx.setTransform(k, 0, 0, k, 0, 0);
  const e = currentEnhanced();
  renderComposite(vctx, { src: state.src, fg: e?.canvas, fgRect: e?.rect, img: state.img, out: state.out, placement: state.placement });
  if ($('guide').checked) {
    const b = mmToPx(BLEED_MM);
    vctx.setTransform(k, 0, 0, k, 0, 0);
    vctx.strokeStyle = 'rgba(255, 59, 48, 0.9)';
    vctx.lineWidth = 2 / k;
    vctx.setLineDash([10 / k, 8 / k]);
    vctx.strokeRect(b, b, state.out.w - 2 * b, state.out.h - 2 * b);
    vctx.setLineDash([]);
  }
  updateInfo();
}

function updateInfo() {
  const v = visibleSourceRect(state.placement, state.img, state.out);
  const s = PRINT_SIZES[state.sizeKey];
  const mag = state.placement.scale;
  const effDpi = Math.round(300 / mag);
  $('info').textContent =
    `元画像 ${state.img.w}×${state.img.h} ／ 書き出し ${s.label} ${state.out.w}×${state.out.h} ／ ` +
    `拡大率 ×${mag.toFixed(2)}（実質 ${effDpi}dpi）` + (v ? '' : ' ／ 画像が枠の外です') +
    (state.backend ? ` ／ AI：${state.backend === 'webgpu' ? 'WebGPU' : 'WASM（遅い）'}` : '');
  $('status').textContent = currentEnhanced() ? 'AI処理済み' : state.enhanced ? '位置や設定が変わりました。もう一度「AIで高画質化」を押してください' : '';
  $('enhance').disabled = state.busy || !state.src || !v || !!currentEnhanced();
}

// ---------- 読み込み ----------
async function loadFile(file) {
  const bmp = await createImageBitmap(file, { imageOrientation: 'from-image' });
  state.src?.close?.();
  state.src = bmp;
  state.img = { w: bmp.width, h: bmp.height };
  state.name = (file.name || 'photo').replace(/\.[^.]+$/, '');
  state.placement = initialPlacement(state.img, state.out);
  state.enhanced = null;
  $('empty').hidden = true;
  $('export').disabled = false;
  $('reset').disabled = false;
  draw();
}

// ---------- サイズ・向き ----------
function changeOutput() {
  const prev = state.out;
  state.out = outputSize(state.sizeKey, state.landscape);
  $('orient').textContent = state.landscape ? '横向き' : '縦向き';
  $('orient').setAttribute('aria-pressed', String(state.landscape));
  if (state.src) {
    // 相対位置を保ったまま、全身が入る倍率を基準に置き直す
    const rel = { x: state.placement.cx / prev.w, y: state.placement.cy / prev.h };
    const base = initialPlacement(state.img, prev).scale;
    const ratio = state.placement.scale / base;
    const p = initialPlacement(state.img, state.out);
    state.placement = clampPlacement(
      { scale: p.scale * ratio, cx: rel.x * state.out.w, cy: rel.y * state.out.h }, state.img, state.out);
  }
  layoutView();
}

// ---------- 指での操作 ----------
const pointers = new Map();
let gesture = null;

function toOut(e) {
  const r = view.getBoundingClientRect();
  return { x: (e.clientX - r.left) / viewScale, y: (e.clientY - r.top) / viewScale };
}

function startGesture() {
  const pts = [...pointers.values()];
  if (pts.length === 1) {
    gesture = { kind: 'pan', start: pts[0], p0: { ...state.placement } };
  } else if (pts.length >= 2) {
    const [a, b] = pts;
    gesture = {
      kind: 'pinch', p0: { ...state.placement },
      mid: { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }, dist: Math.hypot(a.x - b.x, a.y - b.y) || 1,
    };
  }
}

$('stage').addEventListener('pointerdown', (e) => {
  if (!state.src || state.busy) return;
  $('stage').setPointerCapture(e.pointerId);
  pointers.set(e.pointerId, toOut(e));
  startGesture();
});
$('stage').addEventListener('pointermove', (e) => {
  if (!pointers.has(e.pointerId) || !gesture) return;
  pointers.set(e.pointerId, toOut(e));
  const pts = [...pointers.values()];
  let p;
  if (gesture.kind === 'pan') {
    p = { ...gesture.p0, cx: gesture.p0.cx + pts[0].x - gesture.start.x, cy: gesture.p0.cy + pts[0].y - gesture.start.y };
  } else {
    const [a, b] = pts;
    const mid = { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
    const f = Math.hypot(a.x - b.x, a.y - b.y) / gesture.dist;
    p = zoomAt(gesture.p0, f, gesture.mid.x, gesture.mid.y);
    p.cx += mid.x - gesture.mid.x; p.cy += mid.y - gesture.mid.y;
  }
  state.placement = clampPlacement(p, state.img, state.out);
  draw();
});
const endPointer = (e) => { pointers.delete(e.pointerId); startGesture(); if (!pointers.size) gesture = null; };
$('stage').addEventListener('pointerup', endPointer);
$('stage').addEventListener('pointercancel', endPointer);
$('stage').addEventListener('wheel', (e) => {
  if (!state.src || state.busy) return;
  e.preventDefault();
  const pt = toOut(e);
  state.placement = clampPlacement(zoomAt(state.placement, Math.exp(-e.deltaY * 0.002), pt.x, pt.y), state.img, state.out);
  draw();
}, { passive: false });
// iPad Safari のページ全体のピンチ拡大を止める
document.addEventListener('gesturestart', (e) => e.preventDefault());

// ---------- AIで高画質化 ----------
let abort = null;

function showBusy(text, value = 0) {
  $('busy').hidden = false;
  $('busyText').textContent = text;
  $('busyBar').value = value;
}

/** 書き出し枠に見えている範囲（少し広め）をAIで4倍にし、書き出しの解像度で前景を作る */
async function enhance() {
  const { loadUpscaler, upscaleInto } = await import('./upscale.js');
  const p = state.placement;
  const v = visibleSourceRect(p, state.img, state.out);
  if (!v) return;
  const m = 8; // 周囲の文脈のため、元画像で8pxほど広げる
  const x0 = Math.max(0, Math.floor(v.x - m)), y0 = Math.max(0, Math.floor(v.y - m));
  const x1 = Math.min(state.img.w, Math.ceil(v.x + v.w + m)), y1 = Math.min(state.img.h, Math.ceil(v.y + v.h + m));
  const rect = { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
  const key = placementKey();

  const s = aiInputScale(Number($('shrink').value), p.scale);
  const input = document.createElement('canvas');
  input.width = Math.max(1, Math.round(rect.w * s));
  input.height = Math.max(1, Math.round(rect.h * s));
  const ictx = input.getContext('2d', { willReadFrequently: true });
  ictx.imageSmoothingEnabled = true;
  ictx.imageSmoothingQuality = 'high';
  ictx.drawImage(state.src, rect.x, rect.y, rect.w, rect.h, 0, 0, input.width, input.height);

  const fg = document.createElement('canvas');
  fg.width = Math.max(1, Math.round(rect.w * p.scale));
  fg.height = Math.max(1, Math.round(rect.h * p.scale));

  showBusy('AIの準備中…');
  const up = await loadUpscaler({
    onProgress: (got, total) => showBusy(`AIデータを読み込み中（初回のみ）… ${(got / 1e6).toFixed(0)}${total ? ` / ${(total / 1e6).toFixed(0)}` : ''} MB`, total ? got / total : 0),
  });
  state.backend = up.backend;
  const t0 = performance.now();
  showBusy('AIで高画質化しています…', 0);
  await upscaleInto(up, input, fg.getContext('2d'), { x: 0, y: 0, w: fg.width, h: fg.height }, {
    signal: abort.signal,
    onTile: (i, n) => {
      const sec = (performance.now() - t0) / 1000;
      const left = Math.round((sec / i) * (n - i));
      showBusy(`AIで高画質化しています… ${i} / ${n}（残り約${left}秒）`, i / n);
    },
  });
  state.enhanced = { canvas: fg, rect, key, seconds: (performance.now() - t0) / 1000 };
}

$('enhance').addEventListener('click', async () => {
  if (state.busy) return;
  state.busy = true; abort = new AbortController();
  $('enhance').disabled = true; $('export').disabled = true;
  try {
    await enhance();
  } catch (err) {
    if (err.name !== 'AbortError') { console.error(err); alert(`AI処理に失敗しました：${err.message}`); }
  } finally {
    state.busy = false; abort = null;
    $('busy').hidden = true; $('export').disabled = false;
    draw();
  }
});
$('cancel').addEventListener('click', () => abort?.abort());
$('shrink').addEventListener('change', draw);

// ---------- 書き出し ----------
async function exportJpeg() {
  const c = document.createElement('canvas');
  c.width = state.out.w; c.height = state.out.h;
  const e = currentEnhanced();
  renderComposite(c.getContext('2d'), { src: state.src, fg: e?.canvas, fgRect: e?.rect, img: state.img, out: state.out, placement: state.placement });
  const blob = await new Promise((res) => c.toBlob(res, 'image/jpeg', 0.95));
  const bytes = setJpegDpi(new Uint8Array(await blob.arrayBuffer()), 300);
  const name = `${state.name}_${state.sizeKey}.jpg`;
  return new File([bytes], name, { type: 'image/jpeg' });
}

$('export').addEventListener('click', async () => {
  $('export').disabled = true;
  try {
    const file = await exportJpeg();
    // iPad では共有シートの「画像を保存」で写真アプリへ保存できる
    if (navigator.canShare?.({ files: [file] })) {
      try { await navigator.share({ files: [file] }); } catch (err) { if (err.name !== 'AbortError') throw err; }
    } else {
      const a = document.createElement('a');
      a.href = URL.createObjectURL(file); a.download = file.name; a.click();
      setTimeout(() => URL.revokeObjectURL(a.href), 10_000);
    }
  } catch (err) {
    alert(`保存できませんでした：${err.message}`);
  } finally {
    $('export').disabled = false;
  }
});

$('file').addEventListener('change', (e) => {
  if (state.busy) { e.target.value = ''; return; } const f = e.target.files?.[0]; if (f) loadFile(f); e.target.value = ''; });
$('size').addEventListener('change', (e) => {
  if (state.busy) { e.target.value = state.sizeKey; return; } state.sizeKey = e.target.value; changeOutput(); });
$('orient').addEventListener('click', () => {
  if (state.busy) return; state.landscape = !state.landscape; changeOutput(); });
$('reset').addEventListener('click', () => {
  if (state.busy) return; state.placement = initialPlacement(state.img, state.out); draw(); });
$('guide').addEventListener('change', draw);
window.addEventListener('resize', layoutView);

// テスト用
window.__app = { state, exportJpeg, loadFile, currentEnhanced };

layoutView();
