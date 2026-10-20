// Browser demo: the evaluated ONNX model runs locally with ONNX Runtime Web, served from
// this site (web/vendor, pinned by SHA-256); nothing is sent anywhere. The model file is
// checked against the SHA-256 in model-info.js (part of the page's own code) before use.
import * as ort from "./vendor/ort.wasm.min.mjs";
import { readHeader, withoutOrientation } from "./image-header.js";
import { MODEL } from "./model-info.js";
import { checkSize, preprocess } from "./preprocess.js";

ort.env.wasm.wasmPaths = new URL("./vendor/", import.meta.url).href;
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation, not set here

const MAX_BYTES = 20 * 1024 * 1024;
const LOW_CONFIDENCE = 0.63; // most validation errors had confidence below 63%
const EXAMPLES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

const $ = (id) => document.getElementById(id);
let session;
let latest = 0; // only the most recent upload may show a result

async function sha256Hex(buffer) {
  const digest = await crypto.subtle.digest("SHA-256", buffer);
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

async function fetchOk(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function loadModel() {
  const buffer = await (await fetchOk("model.onnx")).arrayBuffer();
  if ((await sha256Hex(buffer)) !== MODEL.sha256) {
    throw new Error("model.onnx does not match the evaluated model; refusing to run it");
  }
  session = await ort.InferenceSession.create(new Uint8Array(buffer), {
    executionProviders: ["wasm"],
  });
  $("status").textContent = `Model ready: ${MODEL.arch}, SHA-256 ${MODEL.sha256.slice(0, 16)}… verified.`;
  document.body.dataset.ready = "true";
}

// Draw the bitmap upright for an EXIF orientation (1-8), like Pillow's exif_transpose.
function drawOriented(ctx, bitmap, orientation) {
  const w = bitmap.width, h = bitmap.height;
  const transforms = {
    2: [-1, 0, 0, 1, w, 0],
    3: [-1, 0, 0, -1, w, h],
    4: [1, 0, 0, -1, 0, h],
    5: [0, 1, 1, 0, 0, 0],
    6: [0, 1, -1, 0, h, 0],
    7: [0, -1, -1, 0, h, w],
    8: [0, -1, 1, 0, 0, w],
  };
  if (transforms[orientation]) ctx.transform(...transforms[orientation]);
  ctx.drawImage(bitmap, 0, 0);
}

// Decode to 8-bit RGB like preprocess.py's load_rgb: upright from EXIF, transparency
// flattened onto white, no colour-profile conversion.
async function decode(blob) {
  if (blob.size > MAX_BYTES) throw new Error("That file is larger than 20 MB.");
  const bytes = new Uint8Array(await blob.arrayBuffer());
  const header = readHeader(bytes);
  if (!header.format) throw new Error("Use a JPEG, PNG or WebP photo.");
  checkSize(header.width, header.height); // before the browser decodes anything
  let bitmap;
  try {
    bitmap = await createImageBitmap(new Blob([withoutOrientation(bytes, header)]), {
      colorSpaceConversion: "none", // Pillow ignores colour profiles too
      premultiplyAlpha: "none", // Chrome's premultiplied WebP decoding loses precision
    });
  } catch {
    throw new Error("That file could not be read as an image: it may be damaged.");
  }
  const swap = header.orientation >= 5;
  const width = swap ? bitmap.height : bitmap.width;
  const height = swap ? bitmap.width : bitmap.height;
  checkSize(width, height);
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingEnabled = false;
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  drawOriented(ctx, bitmap, header.orientation);
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
  result.dataset.error = "true";
  const p = document.createElement("p");
  p.className = "error";
  p.textContent = message;
  result.replaceChildren(p);
}

async function classify(blob) {
  const id = ++latest;
  if (!session) return showError("The model is still loading; try again in a moment.");
  delete $("result").dataset.error;
  const preview = $("preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(blob);
  preview.hidden = false;
  $("drop-hint").hidden = true;
  try {
    const { rgb, width, height } = await decode(blob);
    if (id !== latest) return; // a newer upload has started
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
      try {
        classify(await (await fetchOk(`examples/${name}.jpg`)).blob());
      } catch (err) {
        showError(err.message);
      }
    });
    $("examples").append(button);
  }
}

setUp();
loadModel().catch((err) => {
  $("status").textContent = `The model could not be loaded: ${err.message}`;
  $("status").classList.add("error");
});
