import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, relative } from "node:path";

const appDir = dirname(fileURLToPath(import.meta.url));
const repoRoot = join(appDir, "..", "..");
const html = readFileSync(join(appDir, "index.html"), "utf8");
const app = readFileSync(join(appDir, "app.js"), "utf8");
const css = readFileSync(join(appDir, "styles.css"), "utf8");
const metadataPath = join(appDir, "battery-dashboard", "README.md");

assert.match(html, /data-side-mode="native"[^>]*>[\s\S]*?SOH 完整看板/);
assert.match(html, /data-mode="native"[^>]*>SOH 完整看板<\/button>/);
assert.match(html, /id="native-dashboard-panel"[^>]*data-mode-panel="native"/);
assert.match(html, /id="native-dashboard-frame"/);
assert.match(html, /id="native-dashboard-unsynced"/);
assert.match(html, /尚未同步原生看板/);
assert.match(html, /id="native-dashboard-source"[^>]*value="D:\\battery_SOH\\dashboard"/);
assert.match(html, /id="sync-native-dashboard"[^>]*>同步原生看板<\/button>/);
assert.match(html, /id="package-portable"[^>]*>生成便携目录包<\/button>/);
assert.match(html, /id="package-standalone"[^>]*>生成单文件包<\/button>/);
assert.match(html, /id="refresh-package-status"[^>]*>刷新交付报告<\/button>/);
assert.match(html, /id="delivery-report"/);
assert.doesNotMatch(html, />[^<]*\?+[^<]*</);

for (const endpoint of [
  "/api/sync/native-dashboard",
  "/api/package/portable",
  "/api/package/standalone",
  "/api/package/status",
]) {
  assert.match(app, new RegExp(endpoint.replaceAll("/", "\\/")));
}
assert.match(app, /function renderNativeDashboard/);
assert.match(app, /function probeNativeDashboard/);
assert.match(app, /function validateNativeDashboardGeneration/);
assert.match(app, /validateLegacyGeneration:\s*true/);
assert.match(app, /resolvePublishedBrowserDirectory\(\s*"\.\/battery-dashboard"/);
assert.match(
  app,
  /resolvePublishedBrowserDirectory\(\s*"\.\/battery-dashboard",[\s\S]*?validateNativeDashboardGeneration/,
);
assert.match(app, /NATIVE_DASHBOARD_ROOT\s*=\s*nativeRoot/);
assert.match(app, /frame\.src\s*=\s*`\$\{NATIVE_DASHBOARD_ROOT\}\/index\.html/);
assert.doesNotMatch(
  app,
  /NATIVE_DASHBOARD_AVAILABLE\s*\|\|\s*nativeOutput\.present/,
);
assert.match(app, /function refreshDeliveryReport/);
assert.match(app, /include_curves:\s*true/);
assert.match(app, /include_detection:\s*false/);

assert.match(css, /\.native-dashboard-layout/);
assert.match(css, /\.native-dashboard-stage/);
assert.match(css, /\.delivery-operations/);
assert.match(css, /\.native-dashboard-frame/);
assert.match(
  css,
  /\.native-dashboard-unsynced\[hidden\]\s*\{[^}]*display\s*:\s*none\s*!important[^}]*\}/,
);

assert.equal(existsSync(metadataPath), true, "native dashboard generation metadata must be committed");
const metadata = readFileSync(metadataPath, "utf8");
assert.match(metadata, /transactional sync/i);
assert.match(metadata, /generated build output/i);
const trackedDashboardFiles = execFileSync(
  "git",
  ["ls-files", "frontend/app/battery-dashboard"],
  { cwd: repoRoot, encoding: "utf8" },
).trim().split(/\r?\n/).filter(Boolean);
assert.equal(
  trackedDashboardFiles.some((path) => path.endsWith(".json")),
  false,
  `generated dashboard JSON must remain untracked: ${trackedDashboardFiles.join(", ")}`,
);

console.log(`native dashboard delivery contract passed for ${relative(repoRoot, appDir)}`);
