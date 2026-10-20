// Uploaded bytes -> upright 8-bit RGB, exactly as preprocess.py's load_rgb produces it, in
// every browser: the page decodes images itself instead of asking the browser, whose
// decoders differ (Safari's JPEG decoder, colour management, premultiplied alpha).
// - JPEG and WebP: libjpeg-turbo (via MozJPEG) and libwebp compiled to WebAssembly, the
//   libraries Pillow uses; PNG: png.js. `codecs` supplies the two WebAssembly decoders.
// - Transparency is flattened onto white with Pillow's arithmetic, then the EXIF
//   orientation is applied like ImageOps.exif_transpose.
// - Refused before decoding: unknown formats, out-of-limit sizes (as the API) and
//   incomplete files. Refused although Python accepts them (a known, documented
//   difference): CMYK JPEGs and animated WebP, which the WebAssembly decoders do not read.
import { ImageError, readHeader } from "./image-header.js";
import { decodePng } from "./png.js";
import { checkSize } from "./preprocess.js";

const damaged = () => new ImageError("That file could not be read as an image: it may be damaged.");

// Pillow's alpha_composite onto opaque white, then dropping alpha (checked against Pillow
// for every colour and alpha value).
function flatten({ width, height, channels, data }) {
  if (channels === 3) return data;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; j < rgb.length; i += 4, j += 3) {
    const a = data[i + 3], back = 255 * (255 - a) + 127;
    rgb[j] = Math.floor((data[i] * a + back) / 255);
    rgb[j + 1] = Math.floor((data[i + 1] * a + back) / 255);
    rgb[j + 2] = Math.floor((data[i + 2] * a + back) / 255);
  }
  return rgb;
}

// ImageOps.exif_transpose: orientation 2-8 -> Pillow's transpose methods.
function orient(rgb, w, h, orientation) {
  if (orientation === 1) return { rgb, width: w, height: h };
  const swap = orientation >= 5;
  const W = swap ? h : w, H = swap ? w : h;
  const source = {
    2: (x, y) => [w - 1 - x, y], // FLIP_LEFT_RIGHT
    3: (x, y) => [w - 1 - x, h - 1 - y], // ROTATE_180
    4: (x, y) => [x, h - 1 - y], // FLIP_TOP_BOTTOM
    5: (x, y) => [y, x], // TRANSPOSE
    6: (x, y) => [y, h - 1 - x], // ROTATE_270
    7: (x, y) => [w - 1 - y, h - 1 - x], // TRANSVERSE
    8: (x, y) => [w - 1 - y, x], // ROTATE_90
  }[orientation];
  const out = new Uint8Array(rgb.length);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const [sx, sy] = source(x, y);
      const s = (sy * w + sx) * 3, o = (y * W + x) * 3;
      out[o] = rgb[s];
      out[o + 1] = rgb[s + 1];
      out[o + 2] = rgb[s + 2];
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
    let out;
    try {
      out = await codecs[header.format](bytes);
    } catch {
      throw damaged();
    }
    image = { width: out.width, height: out.height, channels: 4, data: out.data };
  }
  if (image.width !== header.width || image.height !== header.height) throw damaged();
  return orient(flatten(image), image.width, image.height, header.orientation);
}

// The two WebAssembly decoders from their Emscripten factories (vendor/*_dec.js) and
// their already verified .wasm bytes. A decoder that fails is rebuilt for the next file.
export function makeCodecs({ jpegFactory, jpegWasm, webpFactory, webpWasm }) {
  const options = (wasmBinary) => ({ noInitialRun: true, wasmBinary, print: () => {}, printErr: () => {} });
  const wrap = (factory, wasm, call) => {
    let module = null;
    return async (bytes) => {
      module ??= await factory(options(wasm));
      const buffer = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
      try {
        const out = call(module, buffer);
        if (!out) throw new Error("decoding failed");
        return out;
      } catch (err) {
        module = null;
        throw err;
      }
    };
  };
  return {
    jpeg: wrap(jpegFactory, jpegWasm, (m, buffer) => m.decode(buffer, false)), // no auto-rotation
    webp: wrap(webpFactory, webpWasm, (m, buffer) => m.decode(buffer)),
  };
}
