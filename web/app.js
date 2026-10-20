// Browser demo: the evaluated ONNX model runs locally with ONNX Runtime Web; nothing is
// sent anywhere. Everything runs from this site (web/vendor), and every binary is checked
// against the SHA-256 in model-info.js (part of the page's own code) before use: the model
// and the three WebAssembly files (ONNX Runtime, the JPEG and WebP decoders). Images are
// decoded by decode.js, not by the browser, so every browser gets Python's pixels.
import * as ort from "./vendor/ort.wasm.min.mjs";
import jpegFactory from "./vendor/mozjpeg_dec.js";
import webpFactory from "./vendor/webp_dec.js";
import { decodeImage, makeCodecs } from "./decode.js";
import { MODEL, WASM } from "./model-info.js";
import { preprocess } from "./preprocess.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation, not set here

const MAX_BYTES = 20 * 1024 * 1024;
const EXAMPLES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

const $ = (id) => document.getElementById(id);
let session, codecs;
let latest = 0; // only the most recent request may show a result

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function fetchVerified(url, expected) {
  const buffer = await (await fetchOk(url)).arrayBuffer();
  if ((await sha256Hex(buffer)) !== expected) {
    throw new Error(`${url} does not match its recorded SHA-256; refusing to run it`);
  }
  return new Uint8Array(buffer);
}

async function load() {
  const [model, ortWasm, jpegWasm, webpWasm] = await Promise.all([
    fetchVerified("model.onnx", MODEL.sha256),
    ...["ort-wasm-simd-threaded.wasm", "mozjpeg_dec.wasm", "webp_dec.wasm"].map((name) =>
      fetchVerified(`vendor/${name}`, WASM[name])),
  ]);
  ort.env.wasm.wasmBinary = ortWasm;
  codecs = makeCodecs({ jpegFactory, jpegWasm, webpFactory, webpWasm });
  session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  $("status").textContent = `Model ready: ${MODEL.arch}, SHA-256 ${MODEL.sha256.slice(0, 16)}… verified.`;
  document.body.dataset.ready = "true";
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function showResult(probabilities) {
  const ranked = MODEL.classes
    .map((name, i) => ({ name, p: probabilities[i] }))
    .sort((a, b) => b.p - a.p);
  const top = ranked[0];
  const result = $("result");
  result.dataset.label = top.name;
  result.dataset.probabilities = JSON.stringify(Object.fromEntries(ranked.map((r) => [r.name, r.p])));
  result.replaceChildren();

  const label = document.createElement("p");
  label.className = "label";
  label.textContent = top.name;
  const conf = document.createElement("p");
  conf.className = "confidence";
  conf.textContent = `${(top.p * 100).toFixed(1)}% confidence`;
  const list = document.createElement("ul");
  list.className = "bars";
  for (const r of ranked) {
    const li = document.createElement("li");
    if (r === top) li.className = "top";
    li.innerHTML = `<span class="name"></span><span class="track"><span class="fill"></span></span><span class="pct"></span>`;
    li.querySelector(".name").textContent = r.name;
    li.querySelector(".fill").style.width = `${(r.p * 100).toFixed(1)}%`;
    li.querySelector(".pct").textContent = `${(r.p * 100).toFixed(1)}%`;
    list.append(li);
  }
  result.append(label, conf, list);
  const flag = MODEL.lowConfidence; // chosen on validation photos by `make errors`
  if (flag && top.p < flag.threshold) {
    const note = document.createElement("p");
    note.className = "note";
    const pct = Math.round(flag.threshold * 100);
    const rate = (x) => `${Math.round(x * 100)}%`;
    note.textContent = flag.accuracyBelow === null
      ? `Less certain than usual: below ${pct}% confidence, check the answer.`
      : `Less certain than usual: on validation photos below ${pct}% confidence, the model was right ${rate(flag.accuracyBelow)} of the time, against ${rate(flag.accuracyAbove)} above it.`;
    result.append(note);
  }
}

function showError(message) {
  const result = $("result");
  delete result.dataset.label;
  result.dataset.error = "true";
  const p = document.createElement("p");
  p.className = "error";
  p.textContent = message;
  result.replaceChildren(p);
}

async function classify(blob, id = ++latest) {
  if (id !== latest) return; // a newer request has started
  if (!session) return showError("The model is still loading; try again in a moment.");
  delete $("result").dataset.error;
  const preview = $("preview");
  try {
    if (blob.size > MAX_BYTES) throw new Error("That file is larger than 20 MB.");
    const { rgb, width, height } = await decodeImage(new Uint8Array(await blob.arrayBuffer()), codecs);
    if (id !== latest) return;
    // The browser shows the photo only once it is known to be readable.
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = URL.createObjectURL(blob);
    preview.hidden = false;
    $("drop-hint").hidden = true;
    const input = new ort.Tensor("float32", preprocess(rgb, width, height), [1, 3, 224, 224]);
    const output = await session.run({ image: input });
    if (id !== latest) return;
    showResult(softmax(Array.from(output.logits.data)));
  } catch (err) {
    if (id !== latest) return;
    preview.hidden = true;
    $("drop-hint").hidden = false;
    showError(err.message || "That file could not be read as an image.");
  }
}

function setUp() {
  const drop = $("drop");
  const file = $("file");
  const open = () => file.click();
  $("pick").addEventListener("click", (e) => {
    e.stopPropagation();
    open();
  });
  drop.addEventListener("click", open);
  file.addEventListener("change", () => {
    if (file.files[0]) classify(file.files[0]);
    file.value = ""; // choosing the same file again must work too
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault();
    drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault();
    drop.classList.remove("over");
    if (e.dataTransfer.files[0]) classify(e.dataTransfer.files[0]);
  });
  for (const name of EXAMPLES) {
    const button = document.createElement("button");
    button.type = "button";
    button.setAttribute("aria-label", `Example photo: ${name}`);
    button.innerHTML = `<img alt=""><span aria-hidden="true"></span>`;
    button.querySelector("img").src = `examples/${name}.jpg`;
    button.querySelector("span").textContent = name;
    button.addEventListener("click", async () => {
      const id = ++latest; // a later upload wins even if this fetch is slow
      try {
        await classify(await (await fetchOk(`examples/${name}.jpg`)).blob(), id);
      } catch (err) {
        if (id === latest) showError(err.message);
      }
    });
    $("examples").append(button);
  }
}

setUp();
load().catch((err) => {
  $("status").textContent = `The model could not be loaded: ${err.message}`;
  $("status").classList.add("error");
});
