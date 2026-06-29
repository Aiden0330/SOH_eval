import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const html = readFileSync(join(here, "index.html"), "utf8");
const app = readFileSync(join(here, "app.js"), "utf8");

assert.match(html, /lang="zh-CN"/);
assert.match(html, /数据集范围/);
assert.match(html, /多 seed 稳定性/);
assert.match(html, /特征 x 算法矩阵/);
assert.match(html, /算法优劣对比/);
assert.match(html, /排行榜/);
assert.match(html, /id="scope-context"/);
assert.match(html, /id="workbench-steps"/);
assert.match(html, /id="run-plan"/);
assert.match(html, /id="mode-tabs"/);
assert.match(html, /id="inference-panel"/);
assert.match(html, /id="model-library"/);
assert.match(html, /id="inference-models"/);
assert.match(html, /id="inference-datasets"/);

assert.match(app, /最佳 MAE/);
assert.match(app, /renderStabilityPanel/);
assert.match(app, /renderInferencePanel/);
assert.match(app, /startInferenceEvaluation/);
assert.match(app, /全部数据集/);
assert.match(app, /判定/);
assert.match(app, /覆盖范围/);
assert.match(app, /预计组合数/);
