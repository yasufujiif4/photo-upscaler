// JPEG に解像度（dpi）情報を書き込む

/** JFIF APP0 の解像度を dpi に書き換える。APP0 がなければ挿入する。 */
export function setJpegDpi(bytes, dpi) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8[0] !== 0xff || u8[1] !== 0xd8) throw new Error('not a JPEG');
  const isJfif =
    u8[2] === 0xff && u8[3] === 0xe0 &&
    u8[6] === 0x4a && u8[7] === 0x46 && u8[8] === 0x49 && u8[9] === 0x46 && u8[10] === 0x00;
  if (isJfif) {
    const out = u8.slice();
    out[13] = 1; // 単位：dpi
    out[14] = (dpi >> 8) & 0xff; out[15] = dpi & 0xff;
    out[16] = (dpi >> 8) & 0xff; out[17] = dpi & 0xff;
    return out;
  }
  const app0 = new Uint8Array([
    0xff, 0xe0, 0x00, 0x10, 0x4a, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x01,
    (dpi >> 8) & 0xff, dpi & 0xff, (dpi >> 8) & 0xff, dpi & 0xff, 0x00, 0x00,
  ]);
  const out = new Uint8Array(u8.length + app0.length);
  out.set(u8.subarray(0, 2), 0);
  out.set(app0, 2);
  out.set(u8.subarray(2), 2 + app0.length);
  return out;
}

/** 読み取り（テスト・確認用）。単位が dpi でなければ null */
export function readJpegDpi(bytes) {
  const u8 = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
  if (u8[2] !== 0xff || u8[3] !== 0xe0 || u8[13] !== 1) return null;
  return { x: (u8[14] << 8) | u8[15], y: (u8[16] << 8) | u8[17] };
}
