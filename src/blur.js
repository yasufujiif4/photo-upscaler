// 小さな画像を滑らかにぼかす（ボックスブラーを3回＝ほぼガウスぼかし）。DOM に依存しない

function boxPass(src, dst, w, h, r, horizontal) {
  const len = horizontal ? w : h, lines = horizontal ? h : w;
  const step = horizontal ? 4 : w * 4, lineStep = horizontal ? w * 4 : 4;
  const norm = 1 / (2 * r + 1);
  for (let l = 0; l < lines; l++) {
    const base = l * lineStep;
    for (let c = 0; c < 4; c++) {
      // 端は端の色が続いているものとして扱う
      let acc = 0;
      for (let k = -r; k <= r; k++) acc += src[base + Math.min(len - 1, Math.max(0, k)) * step + c];
      for (let i = 0; i < len; i++) {
        dst[base + i * step + c] = acc * norm;
        const add = Math.min(len - 1, i + r + 1), sub = Math.max(0, i - r);
        acc += src[base + add * step + c] - src[base + sub * step + c];
      }
    }
  }
}

/** RGBA（Uint8ClampedArray）を半径 r でぼかした新しい配列を返す */
export function blurRGBA(rgba, w, h, r) {
  let a = Float32Array.from(rgba);
  let b = new Float32Array(a.length);
  for (let i = 0; i < 3; i++) {
    boxPass(a, b, w, h, r, true);
    boxPass(b, a, w, h, r, false);
  }
  return Uint8ClampedArray.from(a);
}
