import test from 'node:test';
import assert from 'node:assert/strict';
import {
  priorBoxes, decodeFaces, estimateSimilarity, applyAffine, invertAffine, composeAffine, FACE_TEMPLATE_512,
} from '../src/facemath.js';

const close = (a, b, eps = 1e-6) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);

test('priors の数（Python 版と同じ）', () => {
  // 670×2576 を縮小した 266×1024 で、Python 版は 11456 個（実測）
  assert.equal(priorBoxes(266, 1024).length / 4, 11456);
});

test('相似変換：回転・拡大・移動を正しく求める', () => {
  const th = 0.3, s = 2.5, tx = 40, ty = -12;
  const m0 = [s * Math.cos(th), s * Math.sin(th), -s * Math.sin(th), s * Math.cos(th), tx, ty];
  const src = [[10, 20], [50, 22], [30, 40], [15, 60], [48, 61]];
  const dst = src.map((p) => applyAffine(m0, p));
  const m = estimateSimilarity(src, dst);
  m.forEach((v, i) => close(v, m0[i], 1e-9));
});

test('Python（cv2）版と同じ点からほぼ同じ変換になる', () => {
  // サンプル画像で Python 版が見つけた目・鼻・口
  const pts = [[283, 183], [368, 181], [332, 231], [289, 263], [366, 261]];
  const m = estimateSimilarity(pts, FACE_TEMPLATE_512);
  const p = applyAffine(m, pts[0]);
  assert.ok(Math.hypot(p[0] - FACE_TEMPLATE_512[0][0], p[1] - FACE_TEMPLATE_512[0][1]) < 15);
});

test('逆変換・合成', () => {
  const m = [1.2, 0.3, -0.3, 1.2, 5, 7];
  const id = composeAffine(invertAffine(m), m);
  [1, 0, 0, 1, 0, 0].forEach((v, i) => close(id[i], v));
});

test('検出結果の読み取りと重なりの除去', () => {
  const w = 64, h = 64;
  const n = priorBoxes(w, h).length / 4;
  const loc = new Float32Array(n * 4), conf = new Float32Array(n * 2), landms = new Float32Array(n * 10);
  // 同じ場所の2つの prior（大きさ16と32）。2つ目の大きさを16に直すと同じ箱になり、1つにまとめられる
  conf[1] = 0.9; conf[3] = 0.95;
  loc[1 * 4 + 2] = loc[1 * 4 + 3] = Math.log(0.5) / 0.2;
  const faces = decodeFaces(loc, conf, landms, w, h, { minSize: 1 });
  assert.equal(faces.length, 1);
  close(faces[0].score, 0.95, 1e-6);
});
