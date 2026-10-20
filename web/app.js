// Browser demo: the evaluated ONNX model runs locally with ONNX Runtime Web.
// The model file is checked against the SHA-256 in model.json before it is used.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs";
import { preprocess } from "./preprocess.js";

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation, not set here

const ALLOWED = ["image/jpeg", "image/png", "image/webp"];
const MAX_BYTES = 20 * 1024 * 1024;
const MAX_PIXELS = 40_000_000;
const LOW_CONFIDENCE = 0.63; // most validation errors had confidence below 63%
const EXAMPLES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

const $ = (id) => document.getElementById(id);
let session, meta;

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function loadModel() {
  meta = await (await fetch("model.json")).json();
  const buffer = await (await fetch("model.onnx")).arrayBuffer();
  const actual = await sha256Hex(buffer);
  if (actual !== meta.onnx_sha256) {
    throw new Error(`model.onnx does not match its recorded SHA-256; refusing to run it`);
  }
  session = await ort.InferenceSession.create(new Uint8Array(buffer), {
    executionProviders: ["wasm"],
  });
  $("status").textContent = `Model ready: ${meta.arch}, SHA-256 ${actual.slice(0, 16)}… verified.`;
  document.body.dataset.ready = "true";
}

// Decode to RGB pixels: upright from EXIF, no colour-profile conversion (like Pillow).
async function decode(blob) {
  if (!ALLOWED.includes(blob.type)) throw new Error("Use a JPEG, PNG or WebP photo.");
  if (blob.size > MAX_BYTES) throw new Error("That file is larger than 20 MB.");
  const bitmap = await createImageBitmap(blob, {
    imageOrientation: "from-image",
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS) throw new Error("That image has too many pixels.");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0, j = 0; i < rgba.length; i += 4, j += 3) {
    rgb[j] = rgba[i];
    rgb[j + 1] = rgba[i + 1];
    rgb[j + 2] = rgba[i + 2];
  }
  return { rgb, width, height };
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function showResult(probabilities) {
  const ranked = meta.classes
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
  if (top.p < LOW_CONFIDENCE) {
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "Low confidence: most of the model's mistakes on validation photos were below 63% confidence.";
    result.append(note);
  }
}

function showError(message) {
  const result = $("result");
  delete result.dataset.label;
  const p = document.createElement("p");
  p.className = "error";
  p.textContent = message;
  result.replaceChildren(p);
}

async function classify(blob) {
  if (!session) return showError("The model is still loading; try again in a moment.");
  try {
    const preview = $("preview");
    if (preview.src) URL.revokeObjectURL(preview.src);
    preview.src = URL.createObjectURL(blob);
    preview.hidden = false;
    $("drop-hint").hidden = true;
    const { rgb, width, height } = await decode(blob);
    const input = new ort.Tensor("float32", preprocess(rgb, width, height), [1, 3, 224, 224]);
    const output = await session.run({ image: input });
    showResult(softmax(Array.from(output.logits.data)));
  } catch (err) {
    showError(err.message || "That file could not be read as an image.");
  }
}

function setUp() {
  const drop = $("drop");
  const file = $("file");
  $("pick").addEventListener("click", (e) => {
    e.preventDefault();
    file.click();
  });
  drop.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      file.click();
    }
  });
  file.addEventListener("change", () => file.files[0] && classify(file.files[0]));
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
    button.innerHTML = `<img alt=""><span></span>`;
    button.querySelector("img").src = `examples/${name}.jpg`;
    button.querySelector("img").alt = `Example: ${name}`;
    button.querySelector("span").textContent = name;
    button.addEventListener("click", async () => {
      const blob = await (await fetch(`examples/${name}.jpg`)).blob();
      classify(blob);
    });
    $("examples").append(button);
  }
}

setUp();
loadModel().catch((err) => {
  $("status").textContent = `The model could not be loaded: ${err.message}`;
  $("status").classList.add("error");
});
