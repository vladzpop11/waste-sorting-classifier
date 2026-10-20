// Image preprocessing identical to the Python pipeline (preprocess.py), so the browser feeds
// the model the same numbers that training, evaluation and the API use:
//   resize the shorter side to 256 with Pillow's bilinear filter (reproduced exactly:
//   two passes, 22-bit fixed-point coefficients, the same rounding), centre-crop 224x224
//   (Python's round-half-to-even), scale to [0, 1], normalise with the ImageNet mean/std.
// Input and intermediate images are RGB Uint8Arrays, row-major, 3 bytes per pixel.
// Verified against the Python code on all 360 validation photos (web/check_preprocess.mjs).

export const RESIZE = 256;
export const CROP = 224;
export const MEAN = [0.485, 0.456, 0.406];
export const STD = [0.229, 0.224, 0.225];
// Same limits as preprocess.py: bigger or more elongated images are refused.
export const MAX_PIXELS = 25_000_000;
export const MAX_ASPECT = 10;

export function checkSize(w, h) {
  if (w < 1 || h < 1) throw new Error("The image is empty.");
  if (w * h > MAX_PIXELS) throw new Error(`That image is too large: ${w}x${h} pixels.`);
  if (Math.max(w, h) / Math.min(w, h) > MAX_ASPECT) {
    throw new Error(`That image is too elongated: ${w}x${h}.`);
  }
}

const PRECISION_BITS = 32 - 8 - 2; // Pillow's Resample.c

function bilinear(x) {
  x = Math.abs(x);
  return x < 1 ? 1 - x : 0;
}

// Pillow precompute_coeffs + normalize_coeffs_8bpc for one axis.
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 1 * filterscale; // bilinear support is 1
  const ksize = Math.ceil(support) * 2 + 1;
  const bounds = new Int32Array(outSize * 2);
  const kk = new Int32Array(outSize * ksize);
  const k = new Float64Array(ksize);
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const ss = 1 / filterscale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    xmax -= xmin;
    let ww = 0;
    for (let x = 0; x < xmax; x++) {
      const w = bilinear((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    for (let x = 0; x < ksize; x++) {
      const v = x < xmax && ww !== 0 ? k[x] / ww : 0;
      kk[xx * ksize + x] = Math.trunc(v < 0 ? -0.5 + v * (1 << PRECISION_BITS) : 0.5 + v * (1 << PRECISION_BITS));
    }
    bounds[xx * 2] = xmin;
    bounds[xx * 2 + 1] = xmax;
  }
  return { bounds, kk, ksize };
}

function clip8(ss) {
  const v = ss >> PRECISION_BITS;
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

function resizeHorizontal(src, w, h, outW) {
  const { bounds, kk, ksize } = coefficients(w, outW);
  const out = new Uint8Array(outW * h * 3);
  const half = 1 << (PRECISION_BITS - 1);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < outW; xx++) {
      const xmin = bounds[xx * 2], xmax = bounds[xx * 2 + 1];
      let s0 = half, s1 = half, s2 = half;
      for (let x = 0; x < xmax; x++) {
        const c = kk[xx * ksize + x], i = (y * w + x + xmin) * 3;
        s0 += src[i] * c; s1 += src[i + 1] * c; s2 += src[i + 2] * c;
      }
      const o = (y * outW + xx) * 3;
      out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
    }
  }
  return out;
}

function resizeVertical(src, w, h, outH) {
  const { bounds, kk, ksize } = coefficients(h, outH);
  const out = new Uint8Array(w * outH * 3);
  const half = 1 << (PRECISION_BITS - 1);
  for (let yy = 0; yy < outH; yy++) {
    const ymin = bounds[yy * 2], ymax = bounds[yy * 2 + 1];
    for (let x = 0; x < w; x++) {
      let s0 = half, s1 = half, s2 = half;
      for (let y = 0; y < ymax; y++) {
        const c = kk[yy * ksize + y], i = ((y + ymin) * w + x) * 3;
        s0 += src[i] * c; s1 += src[i + 1] * c; s2 += src[i + 2] * c;
      }
      const o = (yy * w + x) * 3;
      out[o] = clip8(s0); out[o + 1] = clip8(s1); out[o + 2] = clip8(s2);
    }
  }
  return out;
}

// Pillow resizes horizontally first, then vertically, skipping a pass that changes nothing.
export function resize(src, w, h, outW, outH) {
  let data = src;
  if (outW !== w) data = resizeHorizontal(data, w, h, outW);
  if (outH !== h) data = resizeVertical(data, outW, h, outH);
  return data;
}

export function shorterSideSize(w, h, size = RESIZE) {
  // torchvision: the long side is truncated, int(size * long / short)
  return w <= h ? [size, Math.trunc((size * h) / w)] : [Math.trunc((size * w) / h), size];
}

// Python's round(): halves go to the nearest even integer.
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

export function preprocess(rgb, w, h) {
  checkSize(w, h);
  const [rw, rh] = shorterSideSize(w, h);
  const resized = resize(rgb, w, h, rw, rh);
  const top = roundHalfEven((rh - CROP) / 2);
  const left = roundHalfEven((rw - CROP) / 2);
  const out = new Float32Array(3 * CROP * CROP);
  const plane = CROP * CROP;
  for (let y = 0; y < CROP; y++) {
    for (let x = 0; x < CROP; x++) {
      const i = ((y + top) * rw + x + left) * 3;
      for (let c = 0; c < 3; c++) {
        // float32 arithmetic in the same order as NumPy: (v / 255 - mean) / std
        const v = Math.fround(Math.fround(resized[i + c] / 255) - Math.fround(MEAN[c]));
        out[c * plane + y * CROP + x] = Math.fround(v / Math.fround(STD[c]));
      }
    }
  }
  return out;
}
