import test from 'node:test';
import assert from 'node:assert/strict';
import { planTiles, aiInputScale, rgbaToChw, chwToRgba } from '../src/tiles.js';

test('タイルが隙間なく重ならずに全体を覆う', () => {
  const w = 600, h = 333;
  const covered = new Uint8Array(w * h);
  for (const { core, padded } of planTiles(w, h, 256, 16)) {
    assert.ok(padded.x <= core.x && padded.y <= core.y);
    assert.ok(padded.x + padded.w >= core.x + core.w && padded.y + padded.h >= core.y + core.h);
    assert.ok(padded.x >= 0 && padded.y >= 0 && padded.x + padded.w <= w && padded.y + padded.h <= h);
    for (let y = core.y; y < core.y + core.h; y++) for (let x = core.x; x < core.x + core.w; x++) covered[y * w + x]++;
  }
  assert.ok(covered.every((c) => c === 1));
});

test('AIに渡す倍率', () => {
  assert.equal(aiInputScale(3, 0.82), 1 / 3);   // サンプル画像（全身を2L判）
  assert.equal(aiInputScale(1, 5), 1);          // 縮小しない・大きく拡大
  assert.equal(aiInputScale(3, 0.2), 0.1);      // 大きな写真は必要な分だけ
});

test('RGBA ⇔ CHW の往復', () => {
  const w = 3, h = 2;
  const rgba = new Uint8ClampedArray(w * h * 4).map((_, i) => (i % 4 === 3 ? 255 : (i * 37) % 256));
  const r = { x: 1, y: 0, w: 2, h: 2 };
  const chw = rgbaToChw(rgba, w, r);
  const back = chwToRgba(chw, 2, 2, { x: 0, y: 0, w: 2, h: 2 });
  for (let yy = 0; yy < 2; yy++) for (let xx = 0; xx < 2; xx++) for (let c = 0; c < 4; c++) {
    assert.equal(back[(yy * 2 + xx) * 4 + c], rgba[(yy * w + xx + 1) * 4 + c]);
  }
});
