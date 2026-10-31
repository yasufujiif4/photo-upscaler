import test from 'node:test';
import assert from 'node:assert/strict';
import { blurRGBA } from '../src/blur.js';

test('一色の画像はぼかしても変わらない', () => {
  const w = 10, h = 7;
  const rgba = new Uint8ClampedArray(w * h * 4).fill(0).map((_, i) => [10, 120, 250, 255][i % 4]);
  assert.deepEqual([...blurRGBA(rgba, w, h, 3)], [...rgba]);
});

test('白黒の境目がなめらかになる（段差がなくなる）', () => {
  const w = 40, h = 1;
  const rgba = new Uint8ClampedArray(w * 4);
  for (let x = 0; x < w; x++) { const v = x < 20 ? 0 : 255; rgba.set([v, v, v, 255], x * 4); }
  const out = blurRGBA(rgba, w, h, 3);
  let maxJump = 0;
  for (let x = 1; x < w; x++) maxJump = Math.max(maxJump, Math.abs(out[x * 4] - out[(x - 1) * 4]));
  assert.ok(maxJump < 60, `maxJump=${maxJump}`);
  assert.ok(out[0] < 5 && out[(w - 1) * 4] > 250);
});
