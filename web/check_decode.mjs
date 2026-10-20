// Run the page's image decoding (decode.js, with the same WebAssembly decoders) in Node and
// compare it with Python's load_rgb, pixel for pixel, on fixtures written by
// `python -m waste_sorting_classifier.web_fixtures DIR`: every validation photo, and every
// unusual or hostile file in DIR/edge (the same pixels, or refused when Python refuses).
// Usage: node web/check_decode.mjs FIXTURES_DIR IMAGES_DIR
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import { join } from "node:path";
import { decodeImage, makeCodecs } from "./decode.js";

globalThis.ImageData ??= class { constructor(data, width, height) { Object.assign(this, { data, width, height }); } };
const vendor = new URL("./vendor/", import.meta.url);
const codecs = makeCodecs({
  jpegFactory: createRequire(import.meta.url)("./vendor/libjpegturbowasm_decode.js"),
  jpegWasm: readFileSync(new URL("libjpegturbowasm_decode.wasm", vendor)),
  webpFactory: (await import(new URL("webp_dec.js", vendor))).default,
  webpWasm: readFileSync(new URL("webp_dec.wasm", vendor)),
});
const [fixtures, imagesDir] = process.argv.slice(2);
const failures = [];

async function attempt(bytes) {
  const start = performance.now();
  try {
    return { ...(await decodeImage(new Uint8Array(bytes), codecs)), ms: performance.now() - start };
  } catch (err) {
    return { error: err.message, ms: performance.now() - start };
  }
}

function differences(got, want, width, height) {
  if (got.width !== width || got.height !== height) return `size ${got.width}x${got.height}, Python ${width}x${height}`;
  let n = 0, max = 0;
  for (let i = 0; i < want.length; i++) {
    const d = Math.abs(got.rgb[i] - want[i]);
    if (d) { n++; max = Math.max(max, d); }
  }
  return n ? `${n} values differ (max ${max})` : null;
}

const { images } = JSON.parse(readFileSync(join(fixtures, "index.json"), "utf8"));
let exact = 0;
for (const im of images) {
  const got = await attempt(readFileSync(join(imagesDir, im.path)));
  const problem = got.error ?? differences(got, readFileSync(join(fixtures, `${im.file}.rgb`)), im.width, im.height);
  if (problem) failures.push(`${im.path}: ${problem}`);
  else exact++;
}
console.log(`validation photos: ${exact}/${images.length} pixel-identical to Python`);

const edges = JSON.parse(readFileSync(join(fixtures, "edge.json"), "utf8"));
const counts = { same: 0, refusedLikePython: 0, knownDifference: 0 };
let slowest = 0;
for (const c of edges) {
  const got = await attempt(readFileSync(join(fixtures, "edge", c.name)));
  slowest = Math.max(slowest, got.ms);
  let problem = null;
  if (c.refused && !c.page_differs) {
    if (!got.error) problem = `Python refuses (${c.refused.slice(0, 60)}), the page decodes it`;
    else counts.refusedLikePython++;
  } else if (c.page_differs) {
    // A documented difference: the page must do exactly what is recorded.
    const { page, reason } = c.page_differs;
    const did = got.error ? "refuses" : c.refused || !c.width ? "decodes" : differences(got,
      readFileSync(join(fixtures, "edge", `${c.name}.rgb`)), c.width, c.height) ? "differs" : "decodes";
    if (did !== page) problem = `documented difference (${reason}): expected the page to ${page.replace(/s$/, "")}, it ${did}`;
    else counts.knownDifference++;
  } else if (got.error) {
    problem = `the page refuses it (${got.error}), Python accepts it`;
  } else {
    problem = differences(got, readFileSync(join(fixtures, "edge", `${c.name}.rgb`)), c.width, c.height);
    if (!problem) counts.same++;
  }
  if (problem) failures.push(`${c.name}: ${problem}`);
}
console.log(`edge cases: ${edges.length}: ${counts.same} pixel-identical, ${counts.refusedLikePython} refused like Python, ` +
  `${counts.knownDifference} documented differences; slowest ${slowest.toFixed(0)} ms`);
if (failures.length) {
  console.error(`FAILED (${failures.length}):\n  ${failures.join("\n  ")}`);
  process.exit(1);
}
