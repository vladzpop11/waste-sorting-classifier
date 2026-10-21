// PNG decoding to 8-bit RGB or RGBA with the same results as Pillow plus preprocess.py's
// to_rgb, so that the demo does not depend on how each browser decodes PNG (colour
// management, premultiplied alpha). PNG is lossless, so exact agreement is possible:
// - 16-bit samples keep their high byte; 1/2/4-bit grey is scaled to 0-255;
// - transparency: alpha channels, palette alpha (tRNS), and a single transparent grey or
//   RGB colour (tRNS), compared with the raw tRNS values as Pillow does;
// - the default image (IDAT) of an animated PNG, as Pillow uses;
// - damaged files are refused where Pillow refuses them: a bad checksum or an invalid name
//   in a chunk before the image data, a chunk too short for Pillow's reader of it, a chunk
//   after the image data cut off by the end of the file, image data that ends early, an
//   unknown filter.
// Memory: the inflated rows are unfiltered in place, so a decode holds the compressed
// data, the raw rows and the 8-bit output.
import { ImageError, pngChunks } from "./image-header.js";
import { inflateInto } from "./inflate.js";

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(bytes) {
  let c = 0xffffffff;
  for (let i = 0; i < bytes.length; i++) c = CRC_TABLE[(c ^ bytes[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

const damaged = () => new ImageError("That file could not be read as an image: it may be damaged.");
const CHANNELS = { 0: 1, 2: 3, 3: 1, 4: 2, 6: 4 };
// The shortest chunk each of Pillow's chunk readers accepts (PngImagePlugin), before or
// after the image data; tRNS depends on the colour type.
const MIN_LENGTH = { IHDR: 13, gAMA: 4, cHRM: 32, sRGB: 1, pHYs: 9, acTL: 8, fcTL: 26 };
const MIN_TRNS = { 0: 2, 2: 6 };
const DEPTHS = { 0: [1, 2, 4, 8, 16], 2: [8, 16], 3: [1, 2, 4, 8], 4: [8, 16], 6: [8, 16] };
// Adam7 passes: x0, y0, dx, dy.
const ADAM7 = [[0, 0, 8, 8], [4, 0, 8, 8], [0, 4, 4, 8], [2, 0, 4, 4], [0, 2, 2, 4], [1, 0, 2, 2], [0, 1, 1, 2]];

// Undo the row filters in place: raw holds rows of [filter type][rowBytes bytes].
function unfilterInPlace(raw, offset, rows, rowBytes, bpp) {
  const stride = rowBytes + 1;
  for (let y = 0; y < rows; y++) {
    const start = offset + y * stride;
    const type = raw[start];
    const cur = start + 1, prev = cur - stride;
    for (let x = 0; x < rowBytes; x++) {
      const a = x >= bpp ? raw[cur + x - bpp] : 0;
      const b = y > 0 ? raw[prev + x] : 0;
      const c = x >= bpp && y > 0 ? raw[prev + x - bpp] : 0;
      let v = raw[cur + x];
      if (type === 1) v += a;
      else if (type === 2) v += b;
      else if (type === 3) v += (a + b) >> 1;
      else if (type === 4) {
        const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
        v += pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
      } else if (type !== 0) throw damaged();
      raw[cur + x] = v & 0xff;
    }
  }
}

// One sample (channel value) of a row, at full precision.
function sample(row, i, depth) {
  if (depth === 8) return row[i];
  if (depth === 16) return (row[2 * i] << 8) | row[2 * i + 1];
  const perByte = 8 / depth;
  return (row[Math.floor(i / perByte)] >> (8 - depth * (1 + (i % perByte)))) & ((1 << depth) - 1);
}

// -> { width, height, channels: 3 | 4, data } (8-bit, not yet flattened)
export async function decodePng(bytes) {
  let ihdr = null, palette = null, trns = null, colour = null, lastCut = false;
  const idat = []; // the first run of IDAT chunks is the image
  for (const c of pngChunks(bytes)) {
    if (c.phase === 1) {
      idat.push(c.data);
      lastCut = c.cut;
      continue;
    }
    if (c.cut || !c.named) throw damaged();
    if (c.phase === 0) {
      const crcAt = c.at + 8 + c.data.length;
      if (crcAt + 4 > bytes.length) throw damaged();
      const stored = new DataView(bytes.buffer, bytes.byteOffset + crcAt, 4).getUint32(0);
      if (crc32(bytes.subarray(c.at + 4, crcAt)) !== stored) throw damaged(); // Pillow checks every chunk before IDAT
    }
    const min = c.type === "tRNS" ? MIN_TRNS[colour] : MIN_LENGTH[c.type];
    if (c.data.length < (min ?? 0)) throw damaged();
    if (c.type === "IHDR") {
      if (c.data[11]) throw damaged(); // unknown filter method
      if (DEPTHS[c.data[9]]?.includes(c.data[8])) colour = c.data[9];
    }
    if (c.phase === 2) continue; // read by Pillow, but not used for the image
    if (c.type === "IHDR") ihdr = c.data; // the last one before IDAT, like Pillow
    else if (c.type === "PLTE") palette = c.data;
    else if (c.type === "tRNS") trns = c.data;
  }
  if (!ihdr || !idat.length) throw damaged();
  const dv = new DataView(ihdr.buffer, ihdr.byteOffset, 13);
  const width = dv.getUint32(0), height = dv.getUint32(4);
  const depth = ihdr[8], type = ihdr[9], interlaced = ihdr[12] === 1;
  if (!DEPTHS[type] || !DEPTHS[type].includes(depth) || ihdr[10] || ihdr[11] || ihdr[12] > 1) throw damaged();
  if (type === 3 && !palette) throw damaged();
  const channels = CHANNELS[type];
  const bitsPerPixel = channels * depth;
  const bpp = Math.max(1, bitsPerPixel >> 3);

  const passes = (interlaced ? ADAM7 : [[0, 0, 1, 1]]).map(([x0, y0, dx, dy]) => {
    const w = Math.max(0, Math.ceil((width - x0) / dx)), h = Math.max(0, Math.ceil((height - y0) / dy));
    return { x0, y0, dx, dy, w, h, rowBytes: Math.ceil((w * bitsPerPixel) / 8) };
  });
  const expected = passes.reduce((n, p) => n + (p.w && p.h ? p.h * (p.rowBytes + 1) : 0), 0);
  const total = idat.reduce((n, d) => n + d.length, 0);
  const compressed = new Uint8Array(total);
  idat.reduce((o, d) => (compressed.set(d, o), o + d.length), 0);
  // Pillow stops reading the image data once every row is there (the zlib checksum is not
  // always checked; README). A later IDAT chunk cut off by the end of the file is then read
  // as an ordinary chunk, and refused.
  const raw = new Uint8Array(expected);
  if (lastCut && idat.length > 1) {
    const before = compressed.subarray(0, total - idat[idat.length - 1].length);
    if (inflateInto(before, raw).filled >= expected) throw damaged();
  }
  if (inflateInto(compressed, raw).filled < expected) throw damaged();

  // Transparency, as Pillow's convert("RGBA") applies it.
  let alphaOfIndex = null, transparent = null;
  if (trns && type === 3) alphaOfIndex = (i) => (i < trns.length ? trns[i] : 255);
  if (trns && type === 0 && depth !== 16 && trns.length >= 2) transparent = [(trns[0] << 8) | trns[1]];
  if (trns && type === 2 && trns.length >= 6) {
    transparent = [0, 2, 4].map((k) => (trns[k] << 8) | trns[k + 1]);
  }
  const hasAlpha = type === 4 || type === 6 || alphaOfIndex !== null || transparent !== null;
  const outChannels = hasAlpha ? 4 : 3;
  const data = new Uint8Array(width * height * outChannels);
  const greyScale = { 1: 255, 2: 85, 4: 17, 8: 1 }[depth];
  const byte = (v) => (depth === 16 ? v >> 8 : v);

  let offset = 0;
  for (const p of passes) {
    if (!p.w || !p.h) continue;
    unfilterInPlace(raw, offset, p.h, p.rowBytes, bpp);
    const passStart = offset;
    offset += p.h * (p.rowBytes + 1);
    for (let y = 0; y < p.h; y++) {
      const rowStart = passStart + y * (p.rowBytes + 1) + 1;
      const row = raw.subarray(rowStart, rowStart + p.rowBytes);
      for (let x = 0; x < p.w; x++) {
        const o = ((p.y0 + y * p.dy) * width + p.x0 + x * p.dx) * outChannels;
        let r, g, b, a = 255;
        if (type === 0) {
          r = g = b = depth === 16 ? sample(row, x, 16) >> 8 : sample(row, x, depth) * greyScale;
          if (transparent && r === transparent[0]) a = 0;
        } else if (type === 2) {
          r = byte(sample(row, 3 * x, depth));
          g = byte(sample(row, 3 * x + 1, depth));
          b = byte(sample(row, 3 * x + 2, depth));
          if (transparent && r === transparent[0] && g === transparent[1] && b === transparent[2]) a = 0;
        } else if (type === 3) {
          const i = sample(row, x, depth);
          const inPalette = 3 * i + 2 < palette.length;
          r = inPalette ? palette[3 * i] : 0;
          g = inPalette ? palette[3 * i + 1] : 0;
          b = inPalette ? palette[3 * i + 2] : 0;
          if (alphaOfIndex) a = alphaOfIndex(i);
        } else if (type === 4) {
          r = g = b = byte(sample(row, 2 * x, depth));
          a = byte(sample(row, 2 * x + 1, depth));
        } else {
          r = byte(sample(row, 4 * x, depth));
          g = byte(sample(row, 4 * x + 1, depth));
          b = byte(sample(row, 4 * x + 2, depth));
          a = byte(sample(row, 4 * x + 3, depth));
        }
        data[o] = r;
        data[o + 1] = g;
        data[o + 2] = b;
        if (hasAlpha) data[o + 3] = a;
      }
    }
  }
  return { width, height, channels: outChannels, data };
}
