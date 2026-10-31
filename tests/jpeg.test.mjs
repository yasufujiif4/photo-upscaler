import test from 'node:test';
import assert from 'node:assert/strict';
import { setJpegDpi, readJpegDpi } from '../src/jpeg.js';

test('JFIF の解像度を書き換える', () => {
  const jfif = new Uint8Array([0xff, 0xd8, 0xff, 0xe0, 0, 16, 0x4a, 0x46, 0x49, 0x46, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0xff, 0xd9]);
  const out = setJpegDpi(jfif, 300);
  assert.deepEqual(readJpegDpi(out), { x: 300, y: 300 });
  assert.equal(out.length, jfif.length);
});

test('APP0 がなければ挿入する', () => {
  const noApp0 = new Uint8Array([0xff, 0xd8, 0xff, 0xe1, 0, 2, 0xff, 0xd9]);
  const out = setJpegDpi(noApp0, 300);
  assert.deepEqual(readJpegDpi(out), { x: 300, y: 300 });
  assert.equal(out.length, noApp0.length + 18);
  assert.deepEqual([...out.subarray(20)], [0xff, 0xe1, 0, 2, 0xff, 0xd9]);
});
