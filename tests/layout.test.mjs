import test from 'node:test';
import assert from 'node:assert/strict';
import {
  outputSize, initialPlacement, drawRect, visibleSourceRect, zoomAt, clampPlacement,
} from '../src/layout.js';

test('書き出しサイズ（300dpi）', () => {
  assert.deepEqual(outputSize('2L'), { w: 1500, h: 2102 });
  assert.deepEqual(outputSize('L'), { w: 1051, h: 1500 });
  assert.deepEqual(outputSize('hagaki'), { w: 1181, h: 1748 });
  assert.deepEqual(outputSize('2L', true), { w: 2102, h: 1500 });
});

test('縦長の画像は初期配置で全身が入り、左右に余白ができる', () => {
  const img = { w: 670, h: 2576 };
  const out = outputSize('2L');
  const p = initialPlacement(img, out);
  const r = drawRect(p, img);
  assert.ok(Math.abs(r.h - out.h) < 1e-6);
  assert.ok(Math.abs(r.y) < 1e-6);
  assert.ok(r.x > 0 && r.x + r.w < out.w);
  assert.deepEqual(visibleSourceRect(p, img, out), { x: 0, y: 0, w: 670, h: 2576 });
});

test('拡大すると見える範囲が縮み、つまんだ点は動かない', () => {
  const img = { w: 670, h: 2576 };
  const out = outputSize('2L');
  const p0 = initialPlacement(img, out);
  const p1 = zoomAt(p0, 2, 750, 300);
  const before = (750 - drawRect(p0, img).x) / p0.scale;
  const after = (750 - drawRect(p1, img).x) / p1.scale;
  assert.ok(Math.abs(before - after) < 1e-9);
  const v = visibleSourceRect(p1, img, out);
  assert.ok(v.h < 2576 / 1.9);
});

test('拡大縮小の範囲を制限する', () => {
  const img = { w: 670, h: 2576 };
  const out = outputSize('2L');
  const p = clampPlacement({ scale: 1e-6, cx: -500, cy: 99999 }, img, out);
  assert.ok(p.scale > 0.1);
  assert.equal(p.cx, 0);
  assert.equal(p.cy, out.h);
});
