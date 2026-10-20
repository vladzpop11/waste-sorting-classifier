// zlib decompression with a hard output limit, in plain JavaScript (fflate), so that the
// result is the same in every browser, including those without DecompressionStream.
import { Unzlib } from "./vendor/fflate.mjs";

const STOP = Symbol("stop");
const SLICE = 4 * 1024; // deflate expands at most ~1032x, so a slice yields at most ~4 MiB

// Inflate `data` into `out` (a preallocated Uint8Array) and stop once it is full.
// -> { filled: bytes written, overflow: the stream had more, error: the stream is corrupt
//    (or ended early) before `out` was full }
export function inflateInto(data, out) {
  let filled = 0, overflow = false, ended = false;
  const z = new Unzlib((chunk, final) => {
    const n = Math.min(chunk.length, out.length - filled);
    out.set(chunk.subarray(0, n), filled);
    filled += n;
    if (n < chunk.length) {
      overflow = true;
      throw STOP;
    }
    if (final) ended = true;
  });
  let error = false;
  try {
    // In slices, so that fflate hands over output as it goes instead of building the
    // whole stream in one buffer first.
    for (let at = 0; at < data.length; at += SLICE) {
      z.push(data.subarray(at, at + SLICE), at + SLICE >= data.length);
    }
    if (!data.length) z.push(data, true);
  } catch (err) {
    if (err !== STOP) error = true;
  }
  return { filled, overflow, error: error || (!ended && !overflow && filled < out.length) };
}
