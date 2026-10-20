// Drive the real page in headless Chrome and compare it with the Python pipeline
// (fixtures from `python -m waste_sorting_classifier.web_fixtures`):
//   1. every validation photo, uploaded through the file input: same prediction, same
//      probabilities (index.json);
//   2. unusual uploads (edge.json): each EXIF orientation in JPEG, PNG and WebP,
//      transparency, 16-bit, CMYK, ...: the same result as Python, or refused like Python;
//   3. the example buttons, screenshots, and no console errors.
// Exits non-zero on any mismatch.
// Usage: node web/check_page.cjs URL FIXTURES_DIR IMAGES_DIR SCREENSHOT_DIR [CHROME_PATH]
// BROWSER=webkit or BROWSER=firefox runs the same check through Playwright instead of
// Chrome (needs the playwright package and its browsers).
const fs = require("fs");
const path = require("path");
const [url, fixtures, imagesDir, shots] = process.argv.slice(2);
const softmax = (l) => { const m = Math.max(...l); const e = l.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b); return e.map((v) => v / s); };
const argmax = (a) => a.indexOf(Math.max(...a));
const PROBABILITY_TOLERANCE = Number(process.env.TOLERANCE || 1e-4); // decoders may differ slightly
const BROWSER = process.env.BROWSER || "chrome";

// The few calls that differ between Puppeteer (Chrome) and Playwright (WebKit, Firefox).
async function openBrowser() {
  if (BROWSER === "chrome") {
    const puppeteer = require("puppeteer-core");
    const browser = await puppeteer.launch({
      executablePath: process.argv[6] || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
      headless: true,
      // GitHub's Ubuntu runners do not allow Chrome's sandbox.
      args: ["--no-first-run", ...(process.env.CI ? ["--no-sandbox"] : [])],
    });
    const page = await browser.newPage();
    return {
      browser, page,
      version: await browser.version(),
      viewport: (w, h, mobile) => page.setViewport({ width: w, height: h, isMobile: mobile }),
      scheme: (s) => page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: s }]),
      upload: async (file) => (await page.$("#file")).uploadFile(file),
    };
  }
  const playwright = require("playwright");
  const browser = await playwright[BROWSER].launch();
  const page = await browser.newPage();
  return {
    browser, page,
    version: `${BROWSER} ${browser.version()}`,
    viewport: (w, h) => page.setViewportSize({ width: w, height: h }),
    scheme: (s) => page.emulateMedia({ colorScheme: s }),
    upload: (file) => page.setInputFiles("#file", file),
  };
}

(async () => {
  const { browser, page, version, viewport, scheme, upload: setFile } = await openBrowser();
  console.log("browser:", version);
  const problems = [];
  const failures = [];
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) problems.push(`${m.type()}: ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("requestfailed", (r) => problems.push(`request failed: ${r.url()}`));
  await viewport(900, 900);
  const t0 = Date.now();
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("body[data-ready=true]", { timeout: 120000 });
  console.log(`model ready in ${((Date.now() - t0) / 1000).toFixed(1)} s:`, await page.$eval("#status", (e) => e.textContent));

  async function upload(file) {
    await page.$eval("#result", (e) => { delete e.dataset.label; delete e.dataset.error; });
    await setFile(file);
    await page.waitForSelector("#result[data-label], #result[data-error]", { timeout: 30000 });
    return page.$eval("#result", (e) => (e.dataset.error
      ? { error: e.textContent }
      : { label: e.dataset.label, p: JSON.parse(e.dataset.probabilities),
          similarity: Number(e.dataset.similarity), familiar: e.dataset.familiar === "true" }));
  }
  const diff = (got, logits) => Math.max(...softmax(logits).map((q, k) => Math.abs(got.p[classes[k]] - q)));

  // 1. Validation photos.
  const { classes, images } = JSON.parse(fs.readFileSync(path.join(fixtures, "index.json"), "utf8"));
  let same = 0, maxDiff = 0, maxSimilarityDiff = 0;
  const threshold = JSON.parse(fs.readFileSync(path.join(fixtures, "index.json"), "utf8")).threshold;
  // The familiarity decision must match Python's, except within 1e-4 of the threshold.
  const familiarityProblem = (got, want) => {
    if (want === undefined) return null;
    maxSimilarityDiff = Math.max(maxSimilarityDiff, Math.abs(got.similarity - want));
    if (Math.abs(want - threshold) < 1e-4) return null;
    return got.familiar === (want >= threshold) ? null : `familiar ${got.familiar}, Python ${want >= threshold} (similarity ${want.toFixed(4)})`;
  };
  const times = [];
  for (const im of images) {
    const start = Date.now();
    const got = await upload(path.join(imagesDir, im.path));
    times.push(Date.now() - start);
    if (got.label === classes[argmax(im.logits)]) same++;
    else failures.push(`${im.path}: page says ${got.label || got.error}`);
    const fam = got.label ? familiarityProblem(got, im.similarity) : null;
    if (fam) failures.push(`${im.path}: ${fam}`);
    if (got.p) maxDiff = Math.max(maxDiff, diff(got, im.logits));
  }
  times.sort((a, b) => a - b);
  console.log(JSON.stringify({ uploads: images.length, samePredictions: same, maxProbabilityDiff: maxDiff, maxSimilarityDiff, medianMs: times[times.length >> 1] }));
  if (maxDiff > PROBABILITY_TOLERANCE) failures.push(`validation photos: probabilities differ by ${maxDiff}`);

  // 2. Edge cases.
  const edges = JSON.parse(fs.readFileSync(path.join(fixtures, "edge.json"), "utf8"));
  for (const c of edges) {
    const got = await upload(path.join(fixtures, "edge", c.name));
    let line, ok;
    if (c.page_differs) {
      // Documented differences are checked pixel for pixel by check_decode.mjs; here the
      // page only has to answer the same way (refuse, or show a result).
      ok = (c.page_differs.page === "refuses") === Boolean(got.error);
      line = `documented difference (${c.page_differs.reason}) -> page: ${got.error || got.label}`;
    } else if (c.refused) {
      ok = Boolean(got.error);
      line = `refused (Python: ${c.refused.slice(0, 50)}) -> page: ${got.error || got.label}`;
    } else if (got.error) {
      ok = false;
      line = `page refused it: ${got.error}`;
    } else {
      const d = diff(got, c.logits);
      const fam = familiarityProblem(got, c.similarity);
      ok = got.label === classes[argmax(c.logits)] && d <= PROBABILITY_TOLERANCE && !fam;
      line = `${got.label}${got.familiar ? "" : " (flagged unfamiliar)"}, max probability diff ${d.toExponential(1)}${fam ? `; ${fam}` : ""}`;
      if (c.logits_ignoring_orientation) {
        line += ` (ignoring orientation would differ by ${diff(got, c.logits_ignoring_orientation).toExponential(1)})`;
      }
    }
    console.log(`${ok ? "ok  " : "FAIL"} ${c.name.padEnd(24)} ${line}`);
    if (!ok) failures.push(`${c.name}: ${line}`);
  }

  // An unfamiliar photo, for a screenshot of how the page answers it.
  const everyday = edges.find((c) => c.name.startsWith("everyday-") && c.similarity < threshold);
  if (everyday) {
    await upload(path.join(fixtures, "edge", everyday.name));
    await page.screenshot({ path: path.join(shots, "unfamiliar.png"), fullPage: true });
  }

  // 3. Examples, the same file twice, screenshots.
  await page.$eval("#result", (e) => { delete e.dataset.label; });
  await (await page.$$(".examples button"))[1].click();
  await page.waitForSelector("#result[data-label]");
  const example = await page.$eval("#result", (e) => e.dataset.label);
  console.log("example 'glass' ->", example);
  if (example !== "glass") failures.push(`example glass -> ${example}`);
  const twice = path.join(imagesDir, images[0].path);
  await upload(twice);
  const again = await upload(twice);
  if (!again.label) failures.push("the same file chosen twice did not give a result");
  await page.screenshot({ path: path.join(shots, "light.png"), fullPage: true });
  await scheme("dark");
  await page.screenshot({ path: path.join(shots, "dark.png"), fullPage: true });
  await viewport(390, 844, true);
  await scheme("light");
  await page.screenshot({ path: path.join(shots, "mobile.png"), fullPage: true });
  await browser.close();

  console.log("console problems:", problems.length ? problems : "none");
  if (problems.length) failures.push(`${problems.length} console problems`);
  if (failures.length) {
    console.error(`FAILED (${failures.length}):\n  ${failures.join("\n  ")}`);
    process.exit(1);
  }
  console.log("page check passed");
})().catch((e) => { console.error(e); process.exit(1); });
