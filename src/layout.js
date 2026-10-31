// 書き出しサイズ・配置の計算（DOMに依存しない純粋関数）

export const DPI = 300;
export const MM_PER_INCH = 25.4;
export const BLEED_MM = 2.5; // フチなし印刷で切れる目安

export const PRINT_SIZES = {
  '2L': { label: '2L判', wMm: 127, hMm: 178 },
  L: { label: 'L判', wMm: 89, hMm: 127 },
  hagaki: { label: 'はがき', wMm: 100, hMm: 148 },
};

export const mmToPx = (mm, dpi = DPI) => Math.round((mm / MM_PER_INCH) * dpi);

/** 書き出すピクセル数（縦向き／横向き） */
export function outputSize(sizeKey, landscape = false) {
  const s = PRINT_SIZES[sizeKey];
  if (!s) throw new Error(`unknown size: ${sizeKey}`);
  const w = mmToPx(s.wMm);
  const h = mmToPx(s.hMm);
  return landscape ? { w: h, h: w } : { w, h };
}

/** 画像全体が枠に収まる倍率（contain） */
export const containScale = (img, out) => Math.min(out.w / img.w, out.h / img.h);
/** 枠を画像で埋め尽くす倍率（cover） */
export const coverScale = (img, out) => Math.max(out.w / img.w, out.h / img.h);

/** 初期配置：全身が入る大きさで中央 */
export function initialPlacement(img, out) {
  return { scale: containScale(img, out), cx: out.w / 2, cy: out.h / 2 };
}

/** 拡大縮小の範囲を制限する */
export function clampPlacement(p, img, out) {
  const min = containScale(img, out) * 0.25;
  const max = coverScale(img, out) * 8;
  const scale = Math.min(max, Math.max(min, p.scale));
  // 画像が枠から完全に外れないよう、中心を枠の範囲内に収める
  const cx = Math.min(out.w, Math.max(0, p.cx));
  const cy = Math.min(out.h, Math.max(0, p.cy));
  return { scale, cx, cy };
}

/** 点 (ax, ay) を中心に倍率 factor で拡大縮小 */
export function zoomAt(p, factor, ax, ay) {
  return {
    scale: p.scale * factor,
    cx: ax + (p.cx - ax) * factor,
    cy: ay + (p.cy - ay) * factor,
  };
}

/** 書き出し上の画像の描画矩形 */
export function drawRect(p, img) {
  const w = img.w * p.scale;
  const h = img.h * p.scale;
  return { x: p.cx - w / 2, y: p.cy - h / 2, w, h };
}

/**
 * 書き出し枠に見えている元画像の範囲（元画像の座標、画像の外は切り詰める）。
 * AI処理の対象範囲を決めるのに使う。
 */
export function visibleSourceRect(p, img, out) {
  const r = drawRect(p, img);
  const x0 = Math.max(0, (0 - r.x) / p.scale);
  const y0 = Math.max(0, (0 - r.y) / p.scale);
  const x1 = Math.min(img.w, (out.w - r.x) / p.scale);
  const y1 = Math.min(img.h, (out.h - r.y) / p.scale);
  if (x1 <= x0 || y1 <= y0) return null;
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}
