// 書き出し画像の描画（ぼかした背景＋前景）
import { coverScale, drawRect } from './layout.js';
import { blurRGBA } from './blur.js';

// 元の写真ごとに、ぼかした背景を覚えておく（指で動かすたびに作り直さない）
const bgCache = new WeakMap();

/**
 * ぼかした背景。小さく縮小してから JS でしっかりぼかし、拡大する
 * （縮小→拡大だけだと四角いマス目が見えるため。ctx.filter は Safari の版によって使えないので使わない）。
 */
export function drawBlurredBackground(ctx, src, img, out) {
  const s = coverScale(img, out);
  const w = img.w * s, h = img.h * s;
  const tw = 96;
  const th = Math.max(1, Math.round((tw * out.h) / out.w));
  const key = `${tw}x${th}|${img.w}x${img.h}|${out.w}x${out.h}`;
  let small = bgCache.get(src);
  if (!small || small.key !== key) {
    const c = document.createElement('canvas');
    c.width = tw; c.height = th;
    const cctx = c.getContext('2d', { willReadFrequently: true });
    cctx.imageSmoothingEnabled = true;
    cctx.imageSmoothingQuality = 'high';
    const k = tw / out.w;
    cctx.drawImage(src, ((out.w - w) / 2) * k, ((out.h - h) / 2) * k, w * k, h * k);
    const d = cctx.getImageData(0, 0, tw, th);
    d.data.set(blurRGBA(d.data, tw, th, 4));
    cctx.putImageData(d, 0, 0);
    small = { key, canvas: c };
    bgCache.set(src, small);
  }
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(small.canvas, 0, 0, out.w, out.h);
  // 前景を目立たせるため、少しだけ明るくする
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.fillRect(0, 0, out.w, out.h);
  ctx.restore();
}

/**
 * 書き出し全体を描く。
 * fg: 前景に使う画像（AI処理後の画像）。省略すると元画像をそのまま使う。
 * fgRect: fg が写している元画像の範囲（元画像の座標）。
 */
export function renderComposite(ctx, { src, fg = null, fgRect = null, img, out, placement }) {
  ctx.clearRect(0, 0, out.w, out.h);
  drawBlurredBackground(ctx, src, img, out);
  const r = drawRect(placement, img);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  if (fg && fgRect) {
    // 元画像を下に敷いてから（AIで処理していない端の部分のため）、AI処理後の画像を重ねる
    ctx.drawImage(src, r.x, r.y, r.w, r.h);
    const s = placement.scale;
    ctx.drawImage(fg, r.x + fgRect.x * s, r.y + fgRect.y * s, fgRect.w * s, fgRect.h * s);
  } else {
    ctx.drawImage(src, r.x, r.y, r.w, r.h);
  }
  ctx.restore();
}
