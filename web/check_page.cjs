// Drive the real page in headless Chrome: upload every validation photo through the file
// input and compare the result with the Python reference (web_fixtures index.json).
// Usage: node web/check_page.cjs URL FIXTURES_DIR IMAGES_DIR SCREENSHOT_DIR [CHROME_PATH]
const puppeteer = require("puppeteer-core");
const fs = require("fs");
const path = require("path");
const [url, fixtures, imagesDir, shots] = process.argv.slice(2);
const softmax = (l) => { const m = Math.max(...l); const e = l.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b); return e.map((v) => v / s); };

(async () => {
  const browser = await puppeteer.launch({
    executablePath: process.argv[6] || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    headless: true, args: ["--no-first-run"],
  });
  const page = await browser.newPage();
  const problems = [];
  page.on("console", (m) => { if (["error", "warning"].includes(m.type())) problems.push(`${m.type()}: ${m.text()}`); });
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  await page.setViewport({ width: 900, height: 900 });
  const t0 = Date.now();
  await page.goto(url, { waitUntil: "load" });
  await page.waitForSelector("body[data-ready=true]", { timeout: 120000 });
  console.log(`model ready in ${((Date.now() - t0) / 1000).toFixed(1)} s:`, await page.$eval("#status", (e) => e.textContent));

  const { classes, images } = JSON.parse(fs.readFileSync(path.join(fixtures, "index.json"), "utf8"));
  const input = await page.$("#file");
  let same = 0, maxDiff = 0, times = [];
  for (const im of images) {
    await page.$eval("#result", (e) => { delete e.dataset.label; });
    const start = Date.now();
    await input.uploadFile(path.join(imagesDir, im.path));
    await page.waitForSelector("#result[data-label]", { timeout: 30000 });
    times.push(Date.now() - start);
    const got = await page.$eval("#result", (e) => ({ label: e.dataset.label, p: JSON.parse(e.dataset.probabilities) }));
    const want = softmax(im.logits);
    const wantLabel = classes[want.indexOf(Math.max(...want))];
    if (got.label === wantLabel) same++;
    classes.forEach((c, k) => { maxDiff = Math.max(maxDiff, Math.abs(got.p[c] - want[k])); });
    await page.$eval("#file", (e) => { e.value = ""; });
  }
  times.sort((a, b) => a - b);
  console.log(JSON.stringify({ uploads: images.length, samePredictions: same, maxProbabilityDiff: maxDiff, medianMs: times[times.length >> 1] }));

  // Example buttons, an unsupported file, and screenshots.
  await page.$eval("#result", (e) => { delete e.dataset.label; });
  await (await page.$$(".examples button"))[1].click();
  await page.waitForSelector("#result[data-label]");
  console.log("example 'glass' ->", await page.$eval("#result", (e) => e.dataset.label));
  await page.screenshot({ path: path.join(shots, "light.png"), fullPage: true });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "dark" }]);
  await page.screenshot({ path: path.join(shots, "dark.png"), fullPage: true });
  const gif = path.join(shots, "x.gif");
  fs.writeFileSync(gif, Buffer.from("R0lGODlhAQABAAAAACw=", "base64"));
  await input.uploadFile(gif);
  await page.waitForSelector("#result .error", { timeout: 10000 });
  console.log("unsupported file ->", await page.$eval("#result .error", (e) => e.textContent));
  await page.setViewport({ width: 390, height: 844, isMobile: true });
  await page.emulateMediaFeatures([{ name: "prefers-color-scheme", value: "light" }]);
  await page.screenshot({ path: path.join(shots, "mobile.png"), fullPage: true });
  console.log("console problems:", problems.length ? problems : "none");
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });
