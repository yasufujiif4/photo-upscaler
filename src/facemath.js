// 顔検出（RetinaFace）の後処理と、顔をそろえるための計算（DOMに依存しない）

const MIN_SIZES = [[16, 32], [64, 128], [256, 512]];
const STEPS = [8, 16, 32];
const VAR = [0.1, 0.2];

/** FFHQ の 512×512 の顔での、両目・鼻・口の両端の位置（CodeFormer と同じ） */
export const FACE_TEMPLATE_512 = [
  [192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
  [201.26117, 371.41043], [313.08905, 371.15118],
];

/** 基準の箱（RetinaFace の priors）。[cx, cy, w, h]（0〜1）を並べた配列 */
export function priorBoxes(w, h) {
  const out = [];
  STEPS.forEach((step, k) => {
    const fh = Math.ceil(h / step), fw = Math.ceil(w / step);
    for (let i = 0; i < fh; i++) for (let j = 0; j < fw; j++) {
      for (const ms of MIN_SIZES[k]) out.push((j + 0.5) * step / w, (i + 0.5) * step / h, ms / w, ms / h);
    }
  });
  return new Float32Array(out);
}

/** 重なった箱をまとめる（NMS） */
function iou(a, b) {
  const x0 = Math.max(a[0], b[0]), y0 = Math.max(a[1], b[1]);
  const x1 = Math.min(a[2], b[2]), y1 = Math.min(a[3], b[3]);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
  return ua > 0 ? inter / ua : 0;
}

/**
 * 検出結果を顔の一覧にする。座標は検出に使った画像（w×h）のピクセル。
 * 返り値 [{ score, box: [x0,y0,x1,y1], pts: [[x,y]×5] }]（スコアの高い順）
 */
export function decodeFaces(loc, conf, landms, w, h, { threshold = 0.6, nms = 0.4, minSize = 10 } = {}) {
  const pri = priorBoxes(w, h);
  const n = pri.length / 4;
  const cands = [];
  for (let i = 0; i < n; i++) {
    const score = conf[i * 2 + 1];
    if (score < threshold) continue;
    const [px, py, pw, ph] = [pri[i * 4], pri[i * 4 + 1], pri[i * 4 + 2], pri[i * 4 + 3]];
    const cx = px + loc[i * 4] * VAR[0] * pw, cy = py + loc[i * 4 + 1] * VAR[0] * ph;
    const bw = pw * Math.exp(loc[i * 4 + 2] * VAR[1]), bh = ph * Math.exp(loc[i * 4 + 3] * VAR[1]);
    const box = [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h];
    if (box[2] - box[0] < minSize || box[3] - box[1] < minSize) continue;
    const pts = [];
    for (let k = 0; k < 5; k++) {
      pts.push([(px + landms[i * 10 + 2 * k] * VAR[0] * pw) * w, (py + landms[i * 10 + 2 * k + 1] * VAR[0] * ph) * h]);
    }
    cands.push({ score, box, pts });
  }
  cands.sort((a, b) => b.score - a.score);
  const keep = [];
  for (const c of cands) if (keep.every((k) => iou(k.box, c.box) <= nms)) keep.push(c);
  return keep;
}

/**
 * 回転・拡大縮小・平行移動だけの変換（相似変換）を最小二乗で求める。
 * src の点を dst の点に移す [a, b, c, d, e, f]（x' = a x + c y + e, y' = b x + d y + f。canvas の setTransform と同じ並び）
 */
export function estimateSimilarity(src, dst) {
  const n = src.length;
  let mx = 0, my = 0, ux = 0, uy = 0;
  for (let i = 0; i < n; i++) { mx += src[i][0]; my += src[i][1]; ux += dst[i][0]; uy += dst[i][1]; }
  mx /= n; my /= n; ux /= n; uy /= n;
  let sxx = 0, sxy = 0, ss = 0;
  for (let i = 0; i < n; i++) {
    const x = src[i][0] - mx, y = src[i][1] - my, u = dst[i][0] - ux, v = dst[i][1] - uy;
    sxx += x * u + y * v;   // s cosθ の分子
    sxy += x * v - y * u;   // s sinθ の分子
    ss += x * x + y * y;
  }
  const a = sxx / ss, b = sxy / ss; // a = s cosθ, b = s sinθ
  // x' = a x - b y + tx, y' = b x + a y + ty
  return [a, b, -b, a, ux - (a * mx - b * my), uy - (b * mx + a * my)];
}

/** 変換を当てる */
export const applyAffine = (m, [x, y]) => [m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]];

/** 逆変換 */
export function invertAffine(m) {
  const [a, b, c, d, e, f] = m;
  const det = a * d - b * c;
  return [d / det, -b / det, -c / det, a / det, (c * f - d * e) / det, (b * e - a * f) / det];
}

/** 2つの変換をつなげる（先に m1、次に m2） */
export function composeAffine(m2, m1) {
  return [
    m2[0] * m1[0] + m2[2] * m1[1], m2[1] * m1[0] + m2[3] * m1[1],
    m2[0] * m1[2] + m2[2] * m1[3], m2[1] * m1[2] + m2[3] * m1[3],
    m2[0] * m1[4] + m2[2] * m1[5] + m2[4], m2[1] * m1[4] + m2[3] * m1[5] + m2[5],
  ];
}
