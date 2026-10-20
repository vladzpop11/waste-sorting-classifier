// Uploaded bytes -> upright 8-bit RGB, exactly as preprocess.py's load_rgb produces it, in
// every browser: the page decodes images itself instead of asking the browser, whose
// decoders differ (Safari's JPEG decoder, colour management, premultiplied alpha).
// - JPEG and WebP: libjpeg-turbo 3 and libwebp compiled to WebAssembly, the libraries
//   Pillow uses; PNG: png.js. `codecs` supplies the two WebAssembly decoders.
// - Transparency is flattened onto white with Pillow's arithmetic and the EXIF orientation
//   applied like ImageOps.exif_transpose, in one pass.
// - Refused before decoding: unknown formats, out-of-limit sizes (as the API) and
//   incomplete files. Refused although Python accepts them (a known, documented
//   difference): CMYK JPEGs and animated WebP, which the WebAssembly decoders do not read.
import { ImageError, readHeader } from "./image-header.js";
import { decodePng } from "./png.js";
import { checkSize } from "./preprocess.js";

const damaged = () => new ImageError("That file could not be read as an image: it may be damaged.");

// Source pixel index for output pixel (x, y) under each orientation (2-8), like Pillow's
// transpose methods; w, h are the stored (unrotated) width and height.
const SOURCE = {
  1: (x, y, w) => y * w + x,
  2: (x, y, w) => y * w + (w - 1 - x), // FLIP_LEFT_RIGHT
  3: (x, y, w, h) => (h - 1 - y) * w + (w - 1 - x), // ROTATE_180
  4: (x, y, w, h) => (h - 1 - y) * w + x, // FLIP_TOP_BOTTOM
  5: (x, y, w) => x * w + y, // TRANSPOSE
  6: (x, y, w, h) => (h - 1 - x) * w + y, // ROTATE_270
  7: (x, y, w, h) => (h - 1 - x) * w + (w - 1 - y), // TRANSVERSE
  8: (x, y, w) => x * w + (w - 1 - y), // ROTATE_90
};

// Decoded pixels (1, 3 or 4 channels) -> upright RGB. Alpha is flattened onto white with
// Pillow's alpha_composite arithmetic (checked against Pillow for every colour and alpha).
function toUprightRgb({ width: w, height: h, channels, data }, orientation) {
  const swap = orientation >= 5;
  const W = swap ? h : w, H = swap ? w : h;
  const source = SOURCE[orientation];
  const out = new Uint8Array(W * H * 3);
  for (let y = 0, o = 0; y < H; y++) {
    for (let x = 0; x < W; x++, o += 3) {
      const i = source(x, y, w, h) * channels;
      if (channels === 1) {
        out[o] = out[o + 1] = out[o + 2] = data[i];
      } else if (channels === 3) {
        out[o] = data[i];
        out[o + 1] = data[i + 1];
        out[o + 2] = data[i + 2];
      } else {
        const a = data[i + 3], back = 255 * (255 - a) + 127;
        out[o] = Math.floor((data[i] * a + back) / 255);
        out[o + 1] = Math.floor((data[i + 1] * a + back) / 255);
        out[o + 2] = Math.floor((data[i + 2] * a + back) / 255);
      }
    }
  }
  return { rgb: out, width: W, height: H };
}

// bytes (Uint8Array) -> { rgb, width, height }; throws ImageError (or checkSize's Error)
// with a message for the user.
export async function decodeImage(bytes, codecs) {
  const header = await readHeader(bytes);
  if (!header.format) throw new ImageError("Use a JPEG, PNG or WebP photo.");
  if (!header.complete) {
    throw new ImageError("That file is incomplete: it may have been cut off while downloading.");
  }
  checkSize(header.width, header.height); // before anything is decoded
  if (header.format === "jpeg" && header.components === 4) {
    throw new ImageError("CMYK JPEGs (from print workflows) are not supported here: save the photo as RGB.");
  }
  if (header.format === "webp" && header.animated) {
    throw new ImageError("Animated images are not supported here.");
  }
  let image;
  if (header.format === "png") {
    image = await decodePng(bytes);
  } else {
    try {
      image = await codecs[header.format](header.decodeBytes ?? bytes);
    } catch {
      throw damaged();
    }
    // WebP without an alpha channel in Pillow's eyes: its colours are used as they are.
    if (header.format === "webp" && !header.alpha) image = dropAlpha(image);
  }
  if (image.width !== header.width || image.height !== header.height) throw damaged();
  return toUprightRgb(image, header.orientation);
}

function dropAlpha({ width, height, data }) {
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    rgb[j] = data[i];
    rgb[j + 1] = data[i + 1];
    rgb[j + 2] = data[i + 2];
  }
  return { width, height, channels: 3, data: rgb };
}

// The two WebAssembly decoders from their Emscripten factories (the vendored
// libjpegturbowasm_decode.js and webp_dec.js) and their already verified .wasm bytes.
// Each returns { width, height, channels, data }. A decoder that fails is rebuilt for the
// next file.
export function makeCodecs({ jpegFactory, jpegWasm, webpFactory, webpWasm }) {
  const options = (wasmBinary) => ({ noInitialRun: true, wasmBinary, print: () => {}, printErr: () => {} });
  const wrap = (factory, wasm, call) => {
    let module = null;
    return async (bytes) => {
      module ??= await factory(options(wasm));
      try {
        return call(module, bytes);
      } catch (err) {
        module = null;
        throw err;
      }
    };
  };
  return {
    // libjpeg-turbo: RGB or greyscale output, no EXIF rotation.
    jpeg: wrap(jpegFactory, jpegWasm, (m, bytes) => {
      const decoder = new m.JPEGDecoder();
      try {
        decoder.getEncodedBuffer(bytes.length).set(bytes);
        decoder.decode();
        const info = decoder.getFrameInfo();
        if (info.componentCount !== 1 && info.componentCount !== 3) throw new Error("unsupported");
        return { width: info.width, height: info.height, channels: info.componentCount,
                 data: new Uint8Array(decoder.getDecodedBuffer()) }; // copied out of the WebAssembly heap
      } finally {
        decoder.delete();
      }
    }),
    // libwebp: RGBA output.
    webp: wrap(webpFactory, webpWasm, (m, bytes) => {
      const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      const out = m.decode(buffer);
      if (!out) throw new Error("decoding failed");
      return { width: out.width, height: out.height, channels: 4, data: out.data };
    }),
  };
}
