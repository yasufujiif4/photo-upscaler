// タイル分割の計算（DOMに依存しない）

/**
 * w×h の画像を tile 四方に分け、各タイルの「中身」と「のりしろ込み」の範囲を返す。
 * のりしろ（pad）はAIに周囲の文脈を見せるためのもので、結果からは切り捨てる。
 */
export function planTiles(w, h, tile = 256, pad = 16) {
  const list = [];
  for (let y = 0; y < h; y += tile) {
    for (let x = 0; x < w; x += tile) {
      const core = { x, y, w: Math.min(tile, w - x), h: Math.min(tile, h - y) };
      const px0 = Math.max(0, x - pad), py0 = Math.max(0, y - pad);
      const px1 = Math.min(w, core.x + core.w + pad), py1 = Math.min(h, core.y + core.h + pad);
      list.push({ core, padded: { x: px0, y: py0, w: px1 - px0, h: py1 - py0 } });
    }
  }
  return list;
}

/**
 * AIに渡す倍率（元画像→AI入力）。
 * - shrink: 元画像を何分の1にしてからAIに渡すか（1=縮小しない, 3=1/3）
 * - placementScale: 元画像→書き出しの倍率
 * AIは4倍にするので、書き出しに必要な大きさの2倍を上限にして無駄な計算を省く。
 */
export function aiInputScale(shrink, placementScale) {
  return Math.min(1, 1 / shrink, placementScale / 2);
}

/** RGBA（ImageData の data）の一部を、CHW の float32（0〜1）に詰める */
export function rgbaToChw(rgba, stride, r) {
  const n = r.w * r.h;
  const out = new Float32Array(3 * n);
  for (let yy = 0; yy < r.h; yy++) {
    let si = ((r.y + yy) * stride + r.x) * 4;
    let di = yy * r.w;
    for (let xx = 0; xx < r.w; xx++, si += 4, di++) {
      out[di] = rgba[si] / 255;
      out[di + n] = rgba[si + 1] / 255;
      out[di + 2 * n] = rgba[si + 2] / 255;
    }
  }
  return out;
}

/** CHW の float32（0〜1）を RGBA に戻す（範囲 r を切り出す。chw の幅は cw） */
export function chwToRgba(chw, cw, ch, r) {
  const n = cw * ch;
  const out = new Uint8ClampedArray(r.w * r.h * 4);
  for (let yy = 0; yy < r.h; yy++) {
    let si = (r.y + yy) * cw + r.x;
    let di = yy * r.w * 4;
    for (let xx = 0; xx < r.w; xx++, si++, di += 4) {
      out[di] = chw[si] * 255;
      out[di + 1] = chw[si + n] * 255;
      out[di + 2] = chw[si + 2 * n] * 255;
      out[di + 3] = 255;
    }
  }
  return out;
}
