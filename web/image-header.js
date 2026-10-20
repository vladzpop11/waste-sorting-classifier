// Read what is needed before decoding an upload: the real format (not the file name or MIME
// type), the pixel size (to refuse huge or elongated images before the browser decodes them)
// and the EXIF orientation.
//
// Orientation is applied by the page itself, from the EXIF block Pillow's exif_transpose
// reads. Browsers cannot be told to ignore it (Chrome applies it to JPEG and PNG even with
// imageOrientation "none", but not to WebP), so withoutOrientation() sets every orientation
// tag in a copy of the file to 1 before the browser decodes it. (Orientation stored only in
// XMP metadata, which Pillow also reads, is not handled here.)

const u16 = (b, o, le) => (le ? b[o] | (b[o + 1] << 8) : (b[o] << 8) | b[o + 1]);
const u32 = (b, o, le) =>
  le
    ? (b[o] | (b[o + 1] << 8) | (b[o + 2] << 16) | (b[o + 3] << 24)) >>> 0
    : ((b[o] << 24) | (b[o + 1] << 16) | (b[o + 2] << 8) | b[o + 3]) >>> 0;
const ascii = (b, o, n) => String.fromCharCode(...b.subarray(o, o + n));

export function sniffFormat(b) {
  if (b.length > 3 && b[0] === 0xff && b[1] === 0xd8 && b[2] === 0xff) return "jpeg";
  if (b.length > 8 && ascii(b, 1, 3) === "PNG" && b[0] === 0x89) return "png";
  if (b.length > 12 && ascii(b, 0, 4) === "RIFF" && ascii(b, 8, 4) === "WEBP") return "webp";
  return null;
}

// TIFF-structured EXIF: tag 0x0112 in IFD0. Returns its value (1 = normal when absent or
// invalid) and where the value is stored, if anywhere.
function tiffOrientation(b, start) {
  const none = { value: 1, at: -1 };
  if (start + 8 > b.length) return none;
  const le = ascii(b, start, 2) === "II";
  if (!le && ascii(b, start, 2) !== "MM") return none;
  const ifd = start + u32(b, start + 4, le);
  if (ifd + 2 > b.length) return none;
  const count = u16(b, ifd, le);
  for (let i = 0; i < count; i++) {
    const e = ifd + 2 + i * 12;
    if (e + 12 > b.length) break;
    if (u16(b, e, le) === 0x0112) {
      const v = u16(b, e + 8, le);
      return { value: v >= 1 && v <= 8 ? v : 1, at: e + 8, le };
    }
  }
  return none;
}

function* webpChunks(b) {
  for (let o = 12; o + 8 <= b.length; ) {
    const size = u32(b, o + 4, true);
    yield { id: ascii(b, o, 4), data: o + 8, size };
    o += 8 + size + (size & 1);
  }
}

function* pngChunks(b) {
  for (let o = 8; o + 8 <= b.length; ) {
    const size = u32(b, o, false);
    yield { id: ascii(b, o + 4, 4), data: o + 8, size };
    o += 12 + size;
  }
}

function* jpegSegments(b) {
  for (let o = 2; o + 4 <= b.length; ) {
    if (b[o] !== 0xff) return;
    const marker = b[o + 1];
    if (marker === 0xd9 || marker === 0xda) return; // end of image / start of scan
    const size = u16(b, o + 2, false);
    yield { marker, data: o + 4, size: size - 2 };
    o += 2 + size;
  }
}

// Returns { format, width, height, orientation, tags }: tags lists every stored orientation
// value (for withoutOrientation), orientation is the one Pillow uses.
export function readHeader(b) {
  const format = sniffFormat(b);
  let width = 0, height = 0, orientation = 1;
  const tags = [];
  const exif = (start, chunk) => {
    const t = tiffOrientation(b, start);
    if (t.at >= 0) tags.push({ ...t, chunk });
    return t.value;
  };
  if (format === "png") {
    width = u32(b, 16, false);
    height = u32(b, 20, false);
    for (const c of pngChunks(b)) {
      if (c.id === "eXIf") orientation = exif(c.data, c); // Pillow reads it anywhere
    }
  } else if (format === "jpeg") {
    let seenExif = false; // like Pillow, only the first EXIF block counts
    for (const s of jpegSegments(b)) {
      if (s.marker === 0xe1 && ascii(b, s.data, 6) === "Exif\0\0") {
        const value = exif(s.data + 6);
        if (!seenExif) orientation = value;
        seenExif = true;
      }
      const isSof = s.marker >= 0xc0 && s.marker <= 0xcf && ![0xc4, 0xc8, 0xcc].includes(s.marker);
      if (isSof) {
        height = u16(b, s.data + 1, false);
        width = u16(b, s.data + 3, false);
      }
    }
  } else if (format === "webp") {
    for (const c of webpChunks(b)) {
      if (c.id === "VP8X") {
        width = 1 + (b[c.data + 4] | (b[c.data + 5] << 8) | (b[c.data + 6] << 16));
        height = 1 + (b[c.data + 7] | (b[c.data + 8] << 8) | (b[c.data + 9] << 16));
      } else if (c.id === "VP8 " && !width) {
        width = u16(b, c.data + 6, true) & 0x3fff;
        height = u16(b, c.data + 8, true) & 0x3fff;
      } else if (c.id === "VP8L" && !width) {
        const bits = u32(b, c.data + 1, true);
        width = 1 + (bits & 0x3fff);
        height = 1 + ((bits >> 14) & 0x3fff);
      } else if (c.id === "EXIF") {
        const skip = ascii(b, c.data, 4) === "Exif" ? 6 : 0;
        orientation = exif(c.data + skip);
      }
    }
  }
  return { format, width, height, orientation, tags };
}

const CRC_TABLE = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function crc32(b, start, end) {
  let c = 0xffffffff;
  for (let i = start; i < end; i++) c = CRC_TABLE[(c ^ b[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

// A copy of the file with every orientation tag set to 1 (PNG chunk checksums updated), so
// the browser decodes the pixels as stored.
export function withoutOrientation(b, header) {
  if (!header.tags.length) return b;
  const copy = b.slice();
  for (const t of header.tags) {
    copy[t.at] = t.le ? 1 : 0;
    copy[t.at + 1] = t.le ? 0 : 1;
    if (t.chunk) {
      const end = t.chunk.data + t.chunk.size;
      const crc = crc32(copy, t.chunk.data - 4, end); // chunk type + data
      copy.set([crc >>> 24, (crc >>> 16) & 0xff, (crc >>> 8) & 0xff, crc & 0xff], end);
    }
  }
  return copy;
}
