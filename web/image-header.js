// Read what is needed before decoding an upload: the real format (from its bytes, not its
// name), the pixel size (to refuse huge or elongated images before decoding), whether the
// file is complete, and the EXIF orientation exactly as Pillow's ImageOps.exif_transpose
// finds it (src/waste_sorting_classifier/preprocess.py):
// - JPEG: every APP1 "Exif" segment, concatenated; XMP from the last XMP segment;
// - PNG: the last eXIf chunk, else a "Raw profile type exif" text chunk; XMP from iTXt;
// - WebP: the first EXIF and XMP chunks, when the VP8X header (first chunk) announces them;
// - in EXIF, any number of "Exif\0\0" prefixes, then a TIFF directory read like Pillow's
//   ImageFileDirectory_v2; XMP's tiff:Orientation only when EXIF has none.
// Every read is bounded by the file and by the block it belongs to: hostile files are
// refused or ignored, never followed outside their data. Valid files match Pillow exactly;
// for files that break their format's rules the page aims to match Pillow too, but only
// guarantees to stay safe (bounded time and memory).
import { inflateInto } from "./inflate.js";

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

// Orientation from a TIFF-structured EXIF block, read like Pillow's Exif.load and
// ImageFileDirectory_v2.load:
// - the header must be one of Pillow's TIFF prefixes, or Pillow raises (ExifError here);
// - entries of unknown type are skipped; an entry whose data lies outside the block stops
//   the whole directory, keeping what was read before it (Pillow's truncated-read path);
// - the last entry for a tag wins, whatever its type; numeric types give their first
//   value, byte and string types never match an orientation.
// Returns the value (NaN when not numeric) or null when there is none.
export class ExifError extends Error {}
const UNIT = { 1: 1, 2: 1, 3: 2, 4: 4, 5: 8, 6: 1, 7: 1, 8: 2, 9: 4, 10: 8, 11: 4, 12: 8, 13: 4, 16: 8 };

function firstValue(b, at, type, le) {
  const dv = new DataView(b.buffer, b.byteOffset + at, UNIT[type]);
  switch (type) {
    case 3: return dv.getUint16(0, le);
    case 4: case 13: return dv.getUint32(0, le);
    case 6: return dv.getInt8(0);
    case 8: return dv.getInt16(0, le);
    case 9: return dv.getInt32(0, le);
    case 11: return dv.getFloat32(0, le);
    case 12: return dv.getFloat64(0, le);
    case 16: return Number(dv.getBigUint64(0, le));
    case 5: case 10: { // rationals: Pillow compares the fraction with the orientation numbers
      const n = type === 5 ? dv.getUint32(0, le) : dv.getInt32(0, le);
      const d = type === 5 ? dv.getUint32(4, le) : dv.getInt32(4, le);
      return d !== 0 ? n / d : NaN;
    }
    default: return NaN; // BYTE, ASCII, UNDEFINED
  }
}

export function exifOrientation(exif) {
  let s = 0;
  while (s + 6 <= exif.length && ascii(exif, s, 6) === "Exif\0\0") s += 6;
  const b = exif.subarray(s);
  if (b.length === 0) return null;
  const prefix = ascii(b, 0, 4);
  if (!["MM\0*", "II*\0", "MM*\0", "II\0*", "MM\0+", "II+\0"].includes(prefix)) {
    throw new ExifError("not a TIFF header");
  }
  if (b[2] === 43 || b.length < 8) throw new ExifError("truncated TIFF header"); // BigTIFF too
  const le = prefix.startsWith("II");
  const ifd = u32(b, 4, le);
  if (ifd + 2 > b.length) return null;
  let found = null;
  const count = u16(b, ifd, le);
  for (let i = 0; i < count; i++) {
    const e = ifd + 2 + i * 12;
    if (e + 12 > b.length) break;
    const tag = u16(b, e, le), type = u16(b, e + 2, le), n = u32(b, e + 4, le);
    if (!UNIT[type]) continue;
    const size = UNIT[type] * n;
    let at = e + 8;
    if (size > 4) {
      at = u32(b, e + 8, le);
      if (at + size > b.length) break;
    }
    if (size === 0) continue;
    if (tag === 0x0112) found = firstValue(b, at, type, le);
  }
  return found;
}

function xmpOrientation(text) {
  const match = /tiff:Orientation(="|>)([0-9])/.exec(text);
  return match ? Number(match[2]) : null;
}

// What exif_transpose does with the value: 2-8 transform the image, anything else nothing.
const effective = (value) => (Number.isInteger(value) && value >= 2 && value <= 8 ? value : 1);

// onBadExif: "refuse" (what happens in Pillow for PNG, WebP and most JPEGs) or "upright"
// (a JPEG without a JFIF dpi: Pillow reads its EXIF early, swallows the error and then
// ignores XMP too).
function orientationOf(exif, xmp, onBadExif = "refuse") {
  let fromExif = null;
  try {
    fromExif = exif ? exifOrientation(exif) : null;
  } catch (err) {
    if (!(err instanceof ExifError)) throw err;
    if (onBadExif === "upright") return 1;
    throw new ImageError("That file's metadata is damaged, so it could not be read.");
  }
  if (fromExif !== null) return effective(fromExif);
  return effective(xmp !== null ? xmpOrientation(xmp) : null);
}

const concat = (parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

// Pillow's limits on PNG metadata (PngImagePlugin): each compressed chunk may expand to at
// most 1 MiB (MAX_TEXT_CHUNK) and all text together to 64 MiB (MAX_TEXT_MEMORY); beyond
// that the file is refused. Corrupt compressed data is ignored, as Pillow ignores it.
const MAX_TEXT_CHUNK = 1024 * 1024;
const MAX_TEXT_MEMORY = 64 * MAX_TEXT_CHUNK;
const tooMuch = () => new ImageError("That file carries too much metadata to be read safely.");

// -> Uint8Array, or null when the data is corrupt; throws when it expands past the limit.
function inflateLimited(data) {
  const out = new Uint8Array(MAX_TEXT_CHUNK);
  const { filled, overflow, error } = inflateInto(data, out);
  if (overflow) throw tooMuch();
  return error ? null : out.slice(0, filled);
}

// --- Containers ------------------------------------------------------------------------

// JPEG segments up to the first scan. Fill bytes (0xFF runs) and stray bytes between
// segments are skipped like libjpeg does; standalone markers carry no length.
function jpegHeader(b) {
  let width = 0, height = 0, components = 0, scanAt = -1;
  const exif = [];
  let xmp = null, dpi = false;
  const junk = []; // stray bytes before markers, which libjpeg skips with a warning
  let o = 2;
  while (o < b.length) {
    if (b[o] !== 0xff) {
      junk.push(o++);
      continue;
    }
    while (o < b.length && b[o] === 0xff) o++;
    if (o >= b.length) break;
    const marker = b[o++];
    if (marker === 0x01) throw new ImageError("That file could not be read as an image."); // TEM: Pillow refuses
    if (marker === 0x00 || (marker >= 0xd0 && marker <= 0xd8)) continue;
    if (marker === 0xd9 || o + 2 > b.length) break;
    const length = u16(b, o, false);
    if (length < 2 || o + length > b.length) break;
    const data = b.subarray(o + 2, o + length);
    if (marker === 0xda) { scanAt = o + length; break; }
    if (marker === 0xe0 && ascii(data, 0, 5) === "JFIF\0" && data.length >= 12 && (data[7] === 1 || data[7] === 2)) dpi = true;
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
  const orientation = orientationOf(exif.length ? concat(exif) : null, xmp, dpi ? "refuse" : "upright");
  // The WebAssembly decoder treats libjpeg's warning as an error, so it gets the file
  // without the stray bytes, which is what libjpeg (and so Pillow) decodes.
  let decodeBytes = null;
  if (junk.length) {
    decodeBytes = new Uint8Array(b.length - junk.length);
    let from = 0, to = 0;
    for (const at of junk) {
      decodeBytes.set(b.subarray(from, at), to);
      to += at - from;
      from = at + 1;
    }
    decodeBytes.set(b.subarray(from), to);
  }
  return { width, height, components, complete, orientation, decodeBytes };
}

// PNG chunks up to IEND; a chunk that does not fit in the file ends the walk.
export function* pngChunks(b) {
  for (let o = 8; o + 12 <= b.length; ) {
    const size = u32(b, o, false);
    if (o + 12 + size > b.length) return;
    const type = ascii(b, o + 4, 4);
    yield { type, at: o, data: b.subarray(o + 8, o + 8 + size) };
    if (type === "IEND") return; // Pillow stops here
    o += 12 + size;
  }
}

// Characters of valid UTF-8, counted like Python's len() (code points).
function codePoints(bytes) {
  let n = 0;
  for (let i = 0; i < bytes.length; i++) if ((bytes[i] & 0xc0) !== 0x80) n++;
  return n;
}

// Only these text keys are used; other text only counts towards Pillow's memory limit.
const NEEDED = new Set(["Raw profile type exif", "XML:com.adobe.xmp"]);
const textOf = (key, value) => (NEEDED.has(key) ? latin1(value) : "");

// A PNG text chunk as Pillow reads it -> { key, text, xmpBytes, length } or null (ignored).
function pngText(chunk) {
  const d = chunk.data;
  const nul = d.indexOf(0);
  if (chunk.type === "tEXt") {
    const key = latin1(nul < 0 ? d : d.subarray(0, nul));
    const value = nul < 0 ? new Uint8Array() : d.subarray(nul + 1);
    return key ? { key, text: textOf(key, value), length: value.length } : null;
  }
  if (chunk.type === "zTXt") {
    const key = latin1(nul < 0 ? d : d.subarray(0, nul));
    const rest = nul < 0 ? new Uint8Array() : d.subarray(nul + 1);
    if (rest.length && rest[0] !== 0) throw new ImageError("That file could not be read as an image.");
    const value = inflateLimited(rest.subarray(1)) ?? new Uint8Array();
    return key ? { key, text: textOf(key, value), length: value.length } : null;
  }
  // iTXt: keyword, compression flag, method, language tag, translated keyword, text.
  if (nul < 0 || d.length - nul - 1 < 2) return null;
  const flag = d[nul + 1], method = d[nul + 2];
  const lang = d.indexOf(0, nul + 3);
  const translated = lang < 0 ? -1 : d.indexOf(0, lang + 1);
  if (translated < 0) return null;
  let value = d.subarray(translated + 1);
  if (flag !== 0) {
    if (method !== 0) return null;
    value = inflateLimited(value);
    if (value === null) return null;
  }
  const key = latin1(d.subarray(0, nul));
  const xmpBytes = key === "XML:com.adobe.xmp" ? latin1(value) : null; // kept even if not UTF-8
  let text;
  try {
    new TextDecoder("utf-8", { fatal: true }).decode(d.subarray(lang + 1, translated));
    new TextDecoder("utf-8", { fatal: true }).decode(d.subarray(nul + 3, lang));
    text = new TextDecoder("utf-8", { fatal: true }).decode(value);
  } catch {
    return { key: null, xmpBytes };
  }
  return { key, text: NEEDED.has(key) ? text : "", xmpBytes, length: codePoints(value) };
}

// An iCCP colour profile is not used, but Pillow refuses one it cannot safely decompress.
function checkIccp(chunk) {
  const d = chunk.data;
  const nul = d.indexOf(0);
  if (nul < 0 || d[nul + 1] !== 0) throw new ImageError("That file could not be read as an image.");
  inflateLimited(d.subarray(nul + 2));
}

async function pngHeader(b) {
  let exif = null, ihdr = null, seenData = false, memory = 0, xmpRaw = null;
  const info = {}; // Pillow's im_info for the keys that matter: the last chunk wins
  for (const c of pngChunks(b)) {
    if (c.type === "IDAT") seenData = true;
    if (c.type === "IHDR" && !seenData && c.data.length >= 13) ihdr = c.data; // as png.js uses
    if (c.type === "eXIf") exif = c.data; // Pillow keeps the last
    if (c.type === "iCCP") checkIccp(c);
    if (c.type === "tEXt" || c.type === "zTXt" || c.type === "iTXt") {
      const t = pngText(c);
      if (!t) continue;
      if (t.xmpBytes !== null && t.xmpBytes !== undefined) xmpRaw = t.xmpBytes;
      if (t.key === null) continue;
      info[t.key] = t.text;
      memory += t.length;
      if (memory > MAX_TEXT_MEMORY) throw tooMuch();
    }
  }
  const rawProfile = info["Raw profile type exif"];
  if (!exif && rawProfile !== undefined) {
    const hex = rawProfile.split("\n").slice(3).join("").replace(/\s+/g, "");
    if (!/^([0-9a-fA-F]{2})*$/.test(hex)) throw new ImageError("That file could not be read as an image.");
    exif = Uint8Array.from(hex.match(/../g) || [], (h) => parseInt(h, 16));
  }
  // Pillow: XMP stored under its own key (any text chunk) first, then raw iTXt XMP.
  const xmp = info["XML:com.adobe.xmp"] || xmpRaw || null;
  return {
    width: ihdr ? u32(ihdr, 0, false) : 0,
    height: ihdr ? u32(ihdr, 4, false) : 0,
    complete: true, // whether all pixel data is there is judged by png.js, like Pillow
    orientation: orientationOf(exif && exif.length ? exif : null, xmp),
  };
}

// libwebp's WebPDemux acceptance rules (non-incremental), which Pillow's WebP reader uses:
// a file it would reject is refused here too. (Reviewed port of libwebp's demux.c checks.)
function webpDemuxOk(b) {
  const riffSize = u32(b, 4, true);
  if (riffSize < 8 || riffSize > 0xfffffff6) return false;
  const end = Math.min(b.length, riffSize + 8);
  if (riffSize + 8 > b.length) return false; // incomplete
  let o = 12;
  const chunkAt = (p) => (p + 8 <= end ? { id: ascii(b, p, 4), size: u32(b, p + 4, true) } : null);
  const first = chunkAt(o);
  if (!first) return false;
  const fits = (p, size) => size <= 0xfffffff6 && p + 8 + size + (size & 1) <= end;
  // StoreFrame: ALPH? then VP8/VP8L; returns { next, ok, image, alphaAfter }
  const storeFrame = (p) => {
    let alpha = 0, image = 0, alphaAt = -1, imageAt = -1;
    for (;;) {
      const c = chunkAt(p);
      if (!c) return { ok: false };
      if (!fits(p, c.size)) return { ok: false };
      const padded = c.size + (c.size & 1);
      if (c.id === "ALPH" && alpha === 0) { alpha++; alphaAt = p; p += 8 + padded; }
      else if ((c.id === "VP8 " || c.id === "VP8L") && image === 0) {
        if (c.id === "VP8L" && alpha > 0) return { ok: false };
        image++; imageAt = p; p += 8 + padded;
      } else break;
      if (p === end) break;
      if (end - p < 8) return { ok: false };
    }
    if (!image) return { ok: false };
    if (alpha && alphaAt > imageAt) return { ok: false };
    return { ok: true, next: p, alphaAt };
  };
  if (first.id === "VP8 " || first.id === "VP8L") return storeFrame(o).ok;
  if (first.id !== "VP8X") return false;
  if (first.size < 10 || !fits(o, first.size)) return false;
  const flags = b[o + 8];
  if (flags & ~0x3e) return false;
  if (flags & 0x02) return true; // animations are refused by the page anyway
  o += 8 + first.size + (first.size & 1);
  if (end - o < 8) return false;
  let frames = 0;
  while (o < end) {
    const c = chunkAt(o);
    if (!c || !fits(o, c.size)) return false;
    if (c.id === "VP8X" || c.id === "ANMF") return false;
    if (c.id === "ALPH" || c.id === "VP8 " || c.id === "VP8L") {
      if (frames) return false;
      const f = storeFrame(o);
      if (!f.ok) return false;
      frames++;
      o = f.next;
    } else {
      o += 8 + c.size + (c.size & 1);
    }
    if (o === end) break;
    if (end - o < 8) return false;
  }
  return frames === 1;
}

function webpHeader(b) {
  let width = 0, height = 0, flags = 0, exif = null, xmp = null, alpha = false;
  const declared = u32(b, 4, true) + 8;
  for (let o = 12; o + 8 <= b.length; ) {
    const id = ascii(b, o, 4), size = u32(b, o + 4, true);
    if (o + 8 + size > b.length) break;
    const data = b.subarray(o + 8, o + 8 + size);
    if (id === "VP8X" && size >= 10 && o === 12) { // only as the first chunk, as in libwebp
      flags = data[0];
      width = 1 + (data[4] | (data[5] << 8) | (data[6] << 16));
      height = 1 + (data[7] | (data[8] << 8) | (data[9] << 16));
      alpha = Boolean(flags & 0x10);
    } else if (id === "VP8 " && !width && size >= 10) {
      width = u16(data, 6, true) & 0x3fff;
      height = u16(data, 8, true) & 0x3fff;
    } else if (id === "VP8L" && !width && size >= 5) {
      const bits = u32(data, 1, true);
      width = 1 + (bits & 0x3fff);
      height = 1 + ((bits >> 14) & 0x3fff);
      alpha = Boolean((bits >>> 28) & 1); // the alpha_is_used bit of a simple lossless file
    } else if (id === "EXIF" && exif === null) {
      exif = data;
    } else if (id === "XMP " && xmp === null) {
      xmp = latin1(data);
    }
    o += 8 + size + (size & 1);
  }
  // libwebp's demuxer only exposes metadata the VP8X header announces.
  if (!(flags & 0x08)) exif = null;
  if (!(flags & 0x04)) xmp = null;
  const animated = Boolean(flags & 0x02);
  const complete = declared <= b.length;
  if (complete && !animated && !webpDemuxOk(b)) {
    throw new ImageError("That file could not be read as an image: it may be damaged.");
  }
  // alpha: whether Pillow keeps the alpha channel (otherwise the decoded colours are used
  // as they are, without flattening).
  return { width, height, animated, alpha, complete, orientation: orientationOf(exif, xmp) };
}

// { format, width, height, orientation, complete, components (JPEG), animated (WebP) }
export async function readHeader(b) {
  const format = sniffFormat(b);
  if (format === "jpeg") return { format, ...jpegHeader(b) };
  if (format === "png") return { format, ...(await pngHeader(b)) };
  if (format === "webp") return { format, ...webpHeader(b) };
  return { format: null, width: 0, height: 0, orientation: 1, complete: false };
}
