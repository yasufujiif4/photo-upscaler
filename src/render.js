// 書き出し画像の描画（ぼかした背景＋前景）
import { coverScale, drawRect } from './layout.js';

/** 縮小→拡大でぼかした背景（ctx.filter に頼らず Safari でも同じ見た目にする） */
export function drawBlurredBackground(ctx, src, img, out) {
  const s = coverScale(img, out);
  const w = img.w * s, h = img.h * s;
  const tiny = document.createElement('canvas');
  const tw = 32;
  tiny.width = tw;
  tiny.height = Math.max(1, Math.round((tw * out.h) / out.w));
  const tctx = tiny.getContext('2d');
  tctx.imageSmoothingEnabled = true;
  tctx.imageSmoothingQuality = 'high';
  const k = tiny.width / out.w;
  tctx.drawImage(src, (out.w - w) / 2 * k, (out.h - h) / 2 * k, w * k, h * k);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(tiny, 0, 0, out.w, out.h);
  // 前景を目立たせるため、少しだけ明るくして彩度を落とした印象にする
  ctx.fillStyle = 'rgba(255,255,255,0.12)';
  ctx.fillRect(0, 0, out.w, out.h);
  ctx.restore();
}

/**
 * 書き出し全体を描く。
 * fg: 前景に使う画像（AI処理後の画像を渡せるように img とは別にする）。
 *     fg は元画像と同じ範囲を写した、任意の解像度の画像でよい。
 */
export function renderComposite(ctx, { src, fg = src, img, out, placement }) {
  ctx.clearRect(0, 0, out.w, out.h);
  drawBlurredBackground(ctx, src, img, out);
  const r = drawRect(placement, img);
  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(fg, r.x, r.y, r.w, r.h);
  ctx.restore();
}
