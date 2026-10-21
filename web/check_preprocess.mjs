// Compare the browser pipeline with the Python one on fixtures written by
// `python -m waste_sorting_classifier.web_fixtures DIR`:
//   1. web/preprocess.js on the same decoded pixels -> the Python preprocessing tensor;
//   2. model.onnx run on that tensor by ONNX Runtime Web (the package and WebAssembly engine
//      the page uses, running in Node) -> the Python logits;
//   3. the familiarity score from those features (web/familiarity.js, with the file next
//      to the model) -> the Python score, and the same familiar/unfamiliar decision.
// Usage: node web/check_preprocess.mjs FIXTURES_DIR MODEL_ONNX ONNXRUNTIME_WEB_DIR
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";
import { dirname } from "node:path";
import { mahalanobis } from "./familiarity.js";
import { preprocess } from "./preprocess.js";

const [dir, modelPath, ortDir] = process.argv.slice(2);
const ort = await import(pathToFileURL(join(ortDir, "dist", "ort.node.min.mjs")).href);
ort.env.wasm.numThreads = 1;
const { classes, images } = JSON.parse(readFileSync(join(dir, "index.json"), "utf8"));
const session = await ort.InferenceSession.create(new Uint8Array(readFileSync(modelPath)));

const meta = JSON.parse(readFileSync(join(dirname(modelPath), "model.json"), "utf8")).familiarity;
const values = new Float32Array(readFileSync(join(dirname(modelPath), meta.file)).buffer.slice(0));
// Scores are in the thousands: compare relative to their size.
const tolerance = (want) => 1e-6 * Math.max(1, Math.abs(want));
let maxPixelDiff = 0, exactTensors = 0, maxLogitDiff = 0, samePredictions = 0;
let maxRelativeScoreDiff = 0, sameFamiliarity = 0;
for (const im of images) {
  const rgb = new Uint8Array(readFileSync(join(dir, `${im.file}.rgb`)));
  const expected = new Float32Array(readFileSync(join(dir, `${im.file}.f32`)).buffer.slice(0));
  const x = preprocess(rgb, im.width, im.height);
  let diff = 0;
  for (let i = 0; i < x.length; i++) diff = Math.max(diff, Math.abs(x[i] - expected[i]));
  maxPixelDiff = Math.max(maxPixelDiff, diff);
  if (diff === 0) exactTensors++;
  const out = await session.run({ image: new ort.Tensor("float32", x, [1, 3, 224, 224]) });
  const logits = Array.from(out.logits.data);
  const score = mahalanobis(values, meta.classes, meta.dims, out.features.data);
  maxRelativeScoreDiff = Math.max(maxRelativeScoreDiff, Math.abs(score - im.score) / Math.max(1, Math.abs(im.score)));
  const borderline = Math.abs(im.score - meta.threshold) < 100 * tolerance(im.score);
  if (borderline || (score >= meta.threshold) === (im.score >= meta.threshold)) sameFamiliarity++;
  logits.forEach((v, k) => { maxLogitDiff = Math.max(maxLogitDiff, Math.abs(v - im.logits[k])); });
  const argmax = (a) => a.indexOf(Math.max(...a));
  if (argmax(logits) === argmax(im.logits)) samePredictions++;
}
const n = images.length;
console.log(JSON.stringify({ images: n, exactTensors, maxPixelDiff, samePredictions, maxLogitDiff,
  sameFamiliarity, maxRelativeScoreDiff, classes }));
if (samePredictions !== n || maxPixelDiff > 1e-6 || sameFamiliarity !== n || maxRelativeScoreDiff > 1e-4) process.exit(1);
