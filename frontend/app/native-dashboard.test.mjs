import assert from "node:assert/strict";
import { test } from "node:test";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const nativeRoot = join(here, "battery-dashboard");
const nativeHtmlPath = join(nativeRoot, "index.html");
const nativeDataRoot = join(nativeRoot, "data");
const syncManifestPath = join(nativeRoot, "SYNC_MANIFEST.json");

// Synced from the battery_SOH checkout by scripts/sync_battery_dashboard_native.py;
// not tracked in git, so a clean checkout skips instead of failing.
const vendored = existsSync(nativeHtmlPath) && existsSync(nativeDataRoot) && existsSync(syncManifestPath);

test("native battery dashboard is vendored and wired", {
  skip: vendored ? false : "frontend/app/battery-dashboard not synced (run scripts/sync_battery_dashboard_native.py)",
}, () => {

assert.ok(existsSync(nativeHtmlPath), "native battery dashboard index.html should be vendored");
assert.ok(existsSync(nativeDataRoot), "native battery dashboard data directory should be vendored");
assert.ok(existsSync(syncManifestPath), "native battery dashboard sync manifest should be written");

const nativeHtml = readFileSync(nativeHtmlPath, "utf8");
const syncManifest = JSON.parse(readFileSync(syncManifestPath, "utf8"));

assert.equal(syncManifest.include_soh_curves, true);
assert.equal(syncManifest.excluded_detection_assets, true);
assert.ok(syncManifest.copied_files.every((file) => !file.startsWith("ess_detection")));

assert.match(nativeHtml, /URLSearchParams\(location\.search\)/);
assert.match(nativeHtml, /params\.get\("mode"\)/);
assert.match(nativeHtml, /params\.get\("embed"\)/);
assert.match(nativeHtml, /params\.get\("dataset"\)/);
assert.match(nativeHtml, /params\.get\("source"\)/);
assert.match(nativeHtml, /params\.get\("model"\)/);
assert.match(nativeHtml, /params\.get\("featureset"\)/);
assert.match(nativeHtml, /params\.get\("cell"\)/);
assert.match(nativeHtml, /params\.get\("cycle"\)/);
assert.match(nativeHtml, /setMode\(initialMode\)/);
assert.doesNotMatch(nativeHtml, /setMode\("soh"\)/);
assert.match(nativeHtml, /body\.embedded \.modetabs/);
assert.match(nativeHtml, /data-mode="soh"/);
assert.match(nativeHtml, /id="soh"/);
assert.match(nativeHtml, /id="dataset"/);
assert.match(nativeHtml, /id="algorithm"/);
assert.match(nativeHtml, /id="featureset"/);
assert.match(nativeHtml, /id="linkcontext"/);
assert.match(nativeHtml, /function canonicalDataset/);
assert.match(nativeHtml, /function scenarioForDataset/);
assert.match(nativeHtml, /function renderLinkedSelectors/);
assert.match(nativeHtml, /function applyLinkedContext/);
assert.match(nativeHtml, /function emitEvidenceContext/);
assert.match(nativeHtml, /type: "soh-evidence-context"/);
assert.match(nativeHtml, /window\.parent\.postMessage/);
assert.match(nativeHtml, /案例判定以父页面 PredictionRecord 为准/);
const emittedContext = nativeHtml.slice(
  nativeHtml.indexOf("function emitEvidenceContext"),
  nativeHtml.indexOf("function setupCanvas")
);
assert.doesNotMatch(emittedContext, /trueSoh:/);
assert.doesNotMatch(emittedContext, /predSoh:/);
assert.doesNotMatch(emittedContext, /lower:/);
assert.doesNotMatch(emittedContext, /upper:/);
assert.match(nativeHtml, /emitEvidenceContext\("point-selected"/);
assert.match(nativeHtml, /emitEvidenceContext\("cell-changed"/);
assert.match(
  nativeHtml,
  /const text = \[state\.linkedContext\.model, state\.linkedContext\.featureset\]/,
  "native dashboard should use model/featureset hints when URL dataset is absent"
);
assert.match(
  nativeHtml,
  /state\.linkedContext\.dataset = canonicalDataset\(e\.target\.value\);[\s\S]*?const next = scenarioForDataset\(e\.target\.value\);/,
  "dataset selector should update linked context before loading the target scenario"
);
assert.match(nativeHtml, /calce_cs2/);
assert.match(nativeHtml, /rwth_isea_public/);

const dataFiles = new Set(readdirSync(nativeDataRoot));
const manifest = JSON.parse(readFileSync(join(nativeDataRoot, "manifest.json"), "utf8"));
for (const scenario of manifest.scenarios || []) {
  const name = scenario.mode || scenario.id;
  assert.ok(name, "manifest scenario should expose mode or id");
  assert.ok(dataFiles.has(`${name}.json`), `native battery dashboard should include ${name}.json`);
  assert.ok(dataFiles.has(`${name}_curves.json`), `native battery dashboard should include ${name}_curves.json`);
}
const requiredFiles = [
  "manifest.json",
  "overview.json",
  "calce.json",
  "calce_curves.json",
  "hust.json",
  "hust_curves.json",
  "mit.json",
  "mit_curves.json",
  "nasa.json",
  "nasa_curves.json",
  "oxford.json",
  "oxford_curves.json",
  "rwth.json",
  "rwth_curves.json",
  "snl_lfp.json",
  "snl_lfp_curves.json",
  "uconn.json",
  "uconn_curves.json",
  "xjtu.json",
  "xjtu_curves.json",
];

for (const file of requiredFiles) {
  assert.ok(dataFiles.has(file), `native battery dashboard should include ${file}`);
}

assert.ok(!dataFiles.has("ess_detection.json"), "native SOH dashboard should exclude ESS detection data by default");
assert.ok(!dataFiles.has("ess_detection_curves.json"), "native SOH dashboard should exclude ESS detection curves by default");
});
