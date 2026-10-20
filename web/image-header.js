// Read what is needed before decoding an upload: the real format (from its bytes, not its
// name), the pixel size (to refuse huge or elongated images before decoding), whether the
// file is complete, and the EXIF orientation exactly as Pillow's ImageOps.exif_transpose
// finds it (src/waste_sorting_classifier/preprocess.py):
// - JPEG: every APP1 "Exif" segment, concatenated; XMP from the last XMP segment;
// - PNG: the last eXIf chunk, else a "Raw profile type exif" text chunk; XMP from iTXt;
// - WebP: the first EXIF and XMP chunks, when the VP8X header announces them;
// - in EXIF, any number of "Exif\0\0" prefixes, then a TIFF directory whose last
//   orientation entry wins; XMP's tiff:Orientation only when EXIF has none.
// Every read is bounded by the file and by the block it belongs to: hostile files are
// refused or ignored, never followed outside their data.

const u16 = (b, o, le) => (le ? b[o] | (b[o + 1] << 8) : (b[o] << 8) | b[o + 1]);
const u32 = (b, o, le) =>
  le
    ? (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0
    : ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
const ascii = (b, o, n) => String.fromCharCode(...b.subarray(o, Math.min(o + n, b.length)));
const latin1 = (b) => new TextDecoder("latin1").decode(b);

export class ImageError extends Error {}

export function sniffFormat(b) {
  if (b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "jpeg";
  if (b.length > 8 && b[0] === 0x89 && ascii(b, 1, 7) === "PNG\r\n\x1a\n") return "png";
  if (b.length > 12 && ascii(b, 0, 4) === "RIFF" && ascii(b, 8, 4) === "WEBP") return "webp";
  return null;
}

// --- EXIF and XMP ----------------------------------------------------------------------

// Orientation from a TIFF-structured EXIF block, as Pillow reads it: the first value of a
// numeric entry (byte-typed entries are ignored), stored inline when it fits in 4 bytes and
// at its offset otherwise; the last entry for the tag wins; a directory damaged part-way
// keeps the entries read before the damage. Returns null when there is none.
const SIZES = { 3: 2, 4: 4, 5: 8, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8 }; // numeric TIFF types

function firstValue(b, at, type, le) {
  const dv = new DataView(b.buffer, b.byteOffset + at, SIZES[type]);
  switch (type) {
    case 3: return dv.getUint16(0, le);
    case 4: return dv.getUint32(0, le);
    case 8: return dv.getInt16(0, le);
    case 9: return dv.getInt32(0, le);
    case 11: return dv.getFloat32(0, le);
    case 12: return dv.getFloat64(0, le);
    default: { // rationals: Pillow compares the fraction with the orientation numbers
      const n = type === 5 ? dv.getUint32(0, le) : dv.getInt32(0, le);
      const d = type === 5 ? dv.getUint32(4, le) : dv.getInt32(4, le);
      return d !== 0 ? n / d : NaN;
    }
  }
}

export function exifOrientation(exif) {
  let s = 0;
  while (s + 6 <= exif.length && ascii(exif, s, 6) === "Exif\0\0") s += 6;
  const b = exif.subarray(s);
  if (b.length < 8) return null;
  const order = ascii(b, 0, 2);
  if (order !== "II" && order !== "MM") return null;
  const le = order === "II";
  if (u16(b, 2, le) !== 42 && u16(b, 2, !le) !== 42) return null;
  const ifd = u32(b, 4, le);
  if (ifd + 2 > b.length) return null;
  let found = null;
  const count = u16(b, ifd, le);
  for (let i = 0; i < count; i++) {
    const e = ifd + 2 + i * 12;
    if (e + 12 > b.length) break;
    if (u16(b, e, le) !== 0x0112) continue;
    const type = u16(b, e + 2, le), n = u32(b, e + 4, le);
    if (!SIZES[type] || n === 0) continue;
    const size = SIZES[type] * n;
    const at = size <= 4 ? e + 8 : u32(b, e + 8, le);
    if (at + size > b.length) continue; // Pillow skips a tag whose data is out of range
    found = firstValue(b, at, type, le);
  }
  return found;
}

function xmpOrientation(text) {
  const match = /tiff:Orientation(="|>)([0-9])/.exec(text);
  return match ? Number(match[2]) : null;
}

// What exif_transpose does with the value: 2-8 transform the image, anything else nothing.
const effective = (value) => (Number.isInteger(value) && value >= 2 && value <= 8 ? value : 1);

function orientationOf(exif, xmp) {
  const fromExif = exif ? exifOrientation(exif) : null;
  if (fromExif !== null) return effective(fromExif);
  return effective(xmp !== null ? xmpOrientation(xmp) : null);
}

const concat = (parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

async function inflate(data) {
  const stream = new Blob([data]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// --- Containers ------------------------------------------------------------------------

// JPEG segments up to the first scan. Fill bytes (0xFF runs) and stray bytes between
// segments are skipped like libjpeg does; standalone markers carry no length.
function jpegHeader(b) {
  let width = 0, height = 0, components = 0, scanAt = -1;
  const exif = [];
  let xmp = null;
  let o = 2;
  while (o < b.length) {
    if (b[o] !== 0xff) { o++; continue; }
    while (o < b.length && b[o] === 0xff) o++;
    if (o >= b.length) break;
    const marker = b[o++];
    if (marker === 0x00 || marker === 0x01 || (marker >= 0xd0 && marker <= 0xd8)) continue;
    if (marker === 0xd9 || o + 2 > b.length) break;
    const length = u16(b, o, false);
    if (length < 2 || o + length > b.length) break;
    const data = b.subarray(o + 2, o + length);
    if (marker === 0xda) { scanAt = o + length; break; }
    if (marker === 0xe1 && ascii(data, 0, 6) === "Exif\0\0") exif.push(exif.length ? data.subarray(6) : data);
    else if (marker === 0xe1 && ascii(data, 0, 29) === "http://ns.adobe.com/xap/1.0/\0") xmp = latin1(data.subarray(29));
    const sof = marker >= 0xc0 && marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(marker);
    if (sof && data.length >= 6 && !width) {
      height = u16(data, 1, false);
      width = u16(data, 3, false);
      components = data[5];
    }
    o += length;
  }
  // Pillow refuses a file that ends before the image does: the end-of-image marker must
  // follow the scan (entropy-coded data never contains 0xFF 0xD9; anything after is ignored).
  let complete = false;
  for (let i = scanAt; scanAt >= 0 && i + 1 < b.length; i++) {
    if (b[i] === 0xff && b[i + 1] === 0xd9) { complete = true; break; }
  }
  const orientation = orientationOf(exif.length ? concat(exif) : null, xmp);
  return { width, height, components, complete, orientation };
}

// PNG chunks; a chunk that does not fit in the file ends the walk.
export function* pngChunks(b) {
  for (let o = 8; o + 12 <= b.length; ) {
    const size = u32(b, o, false);
    if (o + 12 + size > b.length) return;
    yield { type: ascii(b, o + 4, 4), at: o, data: b.subarray(o + 8, o + 8 + size) };
    o += 12 + size;
  }
}

async function pngText(chunk) {
  const d = chunk.data;
  const nul = d.indexOf(0);
  if (nul < 0) return null;
  const key = latin1(d.subarray(0, nul));
  if (chunk.type === "tEXt") return { key, text: latin1(d.subarray(nul + 1)) };
  if (chunk.type === "zTXt") return { key, text: latin1(await inflate(d.subarray(nul + 2))) };
  // iTXt: keyword, compression flag, method, language tag, translated keyword, text.
  const compressed = d[nul + 1] === 1;
  const lang = d.indexOf(0, nul + 3);
  const translated = lang < 0 ? -1 : d.indexOf(0, lang + 1);
  if (translated < 0) return null;
  const body = d.subarray(translated + 1);
  return { key, text: new TextDecoder().decode(compressed ? await inflate(body) : body) };
}

async function pngHeader(b) {
  let exif = null, rawProfile = null, xmpText = "", xmpItxt = "", ihdr = false;
  for (const c of pngChunks(b)) {
    if (c.type === "IHDR" && c.data.length >= 8) ihdr = true;
    if (c.type === "eXIf") exif = c.data; // Pillow keeps the last
    if (c.type === "tEXt" || c.type === "zTXt" || c.type === "iTXt") {
      let t = null;
      try { t = await pngText(c); } catch { t = null; } // Pillow skips broken text chunks
      if (t && t.key === "Raw profile type exif") rawProfile = t.text;
      // Pillow looks at XMP from tEXt/zTXt first, then from iTXt.
      if (t && t.key === "XML:com.adobe.xmp") {
        if (c.type === "iTXt") xmpItxt = t.text;
        else xmpText = t.text;
      }
    }
  }
  if (!exif && rawProfile !== null) {
    const hex = rawProfile.split("\n").slice(3).join("").replace(/\s+/g, "");
    if (!/^([0-9a-fA-F]{2})*$/.test(hex)) throw new ImageError("That file could not be read as an image.");
    exif = Uint8Array.from(hex.match(/../g) || [], (h) => parseInt(h, 16));
  }
  return {
    width: ihdr ? u32(b, 16, false) : 0,
    height: ihdr ? u32(b, 20, false) : 0,
    complete: true, // whether all pixel data is there is judged by png.js, like Pillow
    orientation: orientationOf(exif && exif.length ? exif : null, xmpText || xmpItxt || null),
  };
}

function webpHeader(b) {
  let width = 0, height = 0, flags = 0, exif = null, xmp = null, animated = false;
  const declared = u32(b, 4, true) + 8;
  for (let o = 12; o + 8 <= b.length; ) {
    const id = ascii(b, o, 4), size = u32(b, o + 4, true);
    if (o + 8 + size > b.length) break;
    const data = b.subarray(o + 8, o + 8 + size);
    if (id === "VP8X" && size >= 10) {
      flags = data[0];
      width = 1 + (data[4] | (data[5] << 8) | (data[6] << 16));
      height = 1 + (data[7] | (data[8] << 8) | (data[9] << 16));
    } else if (id === "VP8 " && !width && size >= 10) {
      width = u16(data, 6, true) & 0x3fff;
      height = u16(data, 8, true) & 0x3fff;
    } else if (id === "VP8L" && !width && size >= 5) {
      const bits = u32(data, 1, true);
      width = 1 + (bits & 0x3fff);
      height = 1 + ((bits >> 14) & 0x3fff);
    } else if (id === "ANIM" || id === "ANMF") {
      animated = true;
    } else if (id === "EXIF" && exif === null) {
      exif = data;
    } else if (id === "XMP " && xmp === null) {
      xmp = latin1(data);
    }
    o += 8 + size + (size & 1);
  }
  // libwebp's demuxer (used by Pillow) only exposes metadata the VP8X header announces.
  if (!(flags & 0x08)) exif = null;
  if (!(flags & 0x04)) xmp = null;
  return { width, height, animated, complete: declared <= b.length, orientation: orientationOf(exif, xmp) };
}

// { format, width, height, orientation, complete, components (JPEG), animated (WebP) }
export async function readHeader(b) {
  const format = sniffFormat(b);
  if (format === "jpeg") return { format, ...jpegHeader(b) };
  if (format === "png") return { format, ...(await pngHeader(b)) };
  if (format === "webp") return { format, ...webpHeader(b) };
  return { format: null, width: 0, height: 0, orientation: 1, complete: false };
}
