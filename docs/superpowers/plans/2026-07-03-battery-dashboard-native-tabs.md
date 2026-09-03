# Battery SOH Native Dashboard Tabs Migration Plan

> Date: 2026-07-03
> Mode: architecture plan first, no implementation in this step
> Workflow: SDD (subagent-driven development) + TDD

## Goal

将 `D:\battery_SOH\dashboard` 前端中的两个主页签整体迁移到 `D:\SOH_eval`，作为当前工作台里的两个独立新页签：

- `SOH 完整看板`：保留 `D:\battery_SOH` 的 SOH 轨迹、3σ 分布、单电池曲线钻取、V/I/T/IC 曲线展示。
- `异常检测看板`：保留 `D:\battery_SOH` 的 ESS fleet heatmap、模块/时间段点击、信号曲线、事件表和 profile 展示。

当前 `SOH_eval` 的全局诊断页只提供摘要和曲线预览，无法完整呈现 `battery_SOH` 的 Canvas 交互。本计划优先追求“完整呈现”和低回归风险。

## Architecture Decision

推荐采用“静态子应用 + iframe 隔离”的迁移方式，而不是把 `D:\battery_SOH\dashboard\index.html` 的脚本直接内联进当前 DOM。

原因：

- `battery_SOH` 页面使用大量全局 id 和选择器，例如 `#scenario`、`#cell`、`.tab`、`#soh`、`#fleetmap`、`#foot`。
- 当前 `SOH_eval` 页面也有自己的页签、状态变量、样式和渲染生命周期。直接内联会产生 DOM id、CSS、事件监听和全局状态冲突。
- iframe 能完整保留原 Canvas 渲染和曲线钻取行为，同时让两个新页签变成独立页面。
- 同源 iframe 可以后续继续增强通信；第一阶段不需要父页面控制 iframe 内部 DOM。

目标形态：

```text
D:\SOH_eval
  frontend\app
    index.html
    app.js
    styles.css
    battery-dashboard\
      index.html
      data\
        manifest.json
        overview.json
        calce.json
        calce_curves.json
        ...
        ess_detection.json
        ess_detection_curves.json
```

父页面新增两个 mode：

```text
battery-soh-full        -> ./battery-dashboard/index.html?mode=soh&embed=1
battery-detection-full  -> ./battery-dashboard/index.html?mode=detection&embed=1
```

`embed=1` 用于隐藏子页面内部的主模式切换区，避免父页面和 iframe 内部出现双层主导航。

## Scope

本轮迁移包含：

- 复制或生成 `frontend/app/battery-dashboard/` 静态子应用。
- 给子应用入口增加 URL 参数：
  - `mode=soh`
  - `mode=detection`
  - `embed=1`
- 在当前前端增加两个独立页签和左侧导航项。
- 父页面只负责显示 iframe、错误提示、加载状态，不接管子页面内部 Canvas 逻辑。
- 引入自动化测试覆盖导航、iframe src、子应用启动参数和数据文件存在性。

本轮不包含：

- 不把 `battery_SOH` 的 Canvas 逻辑重写成 `dashboard-core.mjs`。
- 不把完整 `_curves.json` 混入当前 `data/profiles/battery-soh-diagnostics` 的 profile 数据目录，避免现有 standalone builder 默认产物暴涨。
- 不改变现有 `训练评测`、`推理评测`、`模型资产库`、`推荐决策`、`全局诊断` 页签行为。

## Source Inventory

源前端：

- `D:\battery_SOH\dashboard\index.html`
- `D:\battery_SOH\dashboard\data\manifest.json`
- `D:\battery_SOH\dashboard\data\overview.json`
- `D:\battery_SOH\dashboard\data\*_curves.json`
- `D:\battery_SOH\dashboard\data\ess_detection.json`
- `D:\battery_SOH\dashboard\data\ess_detection_curves.json`

源页面两个主模式：

- `.mtab[data-mode="soh"]`：SOH 评估
- `.mtab[data-mode="detection"]`：异常检测

源页面关键视图：

- `#view-traj`
- `#view-sigma`
- `#view-overview`
- `#view-detection`

源页面关键 Canvas：

- SOH：`#soh`、`#cv`、`#ci`、`#ct`、`#cic`、`#sigmaAll`
- 异常检测：`#fleetmap`、`#detsignal`

当前项目接入点：

- `frontend/app/index.html`
- `frontend/app/app.js`
- `frontend/app/styles.css`
- `frontend/app/chinese-ui.test.mjs`
- `frontend/app/dashboard-core.test.mjs`
- `tests/test_standalone_frontend.py`

当前前端 mode 机制：

- 左侧导航：`.side-mode[data-side-mode="..."]`
- 顶部页签：`.mode-tab[data-mode="..."]`
- 内容面板：`.mode-panel[data-mode-panel="..."]`
- 切换函数：`renderModeTabs()`

## Data Strategy

### Online / Local HTTP App

推荐把完整 `battery_SOH` 数据复制到：

```text
frontend/app/battery-dashboard/data/
```

这样当前静态前端服务 `frontend/app` 时，iframe 内部仍然使用源页面原有的相对路径：

```js
fetch("data/calce.json")
fetch("data/calce_curves.json")
```

优点：

- 不需要第二个端口。
- 不需要新增后端接口。
- 曲线文件按源页面方式完整加载，最大程度保真。
- 可在当前 app 的同源路径下运行，后续如有需要可做父子窗口通信。

### Standalone / Portable Artifact

完整曲线数据体积较大，`battery-dashboard/data` 下 `_curves.json` 总量约 80MB+。因此独立打包需要明确模式：

- 默认 standalone：继续保持当前轻量 profile，不自动内嵌完整曲线。
- 完整曲线 standalone：新增显式参数，例如 `--include-battery-dashboard-native`，将 `battery-dashboard` 目录作为静态资源一起打包或复制。
- 单 HTML 内嵌完整曲线不作为默认方案，因为会导致 HTML 文件过大，浏览器打开和内存占用都不稳定。

## Frontend Layout

新增两个左侧导航项：

```html
<button class="side-mode" data-side-mode="battery-soh-full">SOH 完整看板</button>
<button class="side-mode" data-side-mode="battery-detection-full">异常检测看板</button>
```

新增两个顶部页签：

```html
<button class="mode-tab" data-mode="battery-soh-full">SOH 完整看板</button>
<button class="mode-tab" data-mode="battery-detection-full">异常检测看板</button>
```

新增两个面板：

```html
<section class="mode-panel native-dashboard-panel" data-mode-panel="battery-soh-full" hidden>
  <iframe data-native-dashboard-frame="soh"></iframe>
</section>

<section class="mode-panel native-dashboard-panel" data-mode-panel="battery-detection-full" hidden>
  <iframe data-native-dashboard-frame="detection"></iframe>
</section>
```

父页面布局规则：

- 面板高度对齐现有工作台内容区。
- iframe 使用 `width: 100%`、`height: calc(100vh - shell chrome height)` 或容器 `height: 100%`。
- 外层不再嵌套多余卡片，避免页面过高。
- 父页面只显示一个轻量状态条：
  - 数据源：`battery-dashboard/data`
  - 模式：SOH / 异常检测
  - 加载失败时提示检查数据目录。

## Native Dashboard Patch

对子应用副本 `frontend/app/battery-dashboard/index.html` 做最小补丁：

1. 读取 URL 参数：

```js
const params = new URLSearchParams(location.search);
const initialMode = params.get("mode") === "detection" ? "detection" : "soh";
const embedded = params.get("embed") === "1";
```

2. 隐藏内部主模式页签：

```js
if (embedded) {
  document.body.classList.add("embedded");
}
```

3. 初始化时使用参数：

```js
await setMode(initialMode);
```

4. CSS 增加：

```css
body.embedded .modetabs {
  display: none;
}
```

5. 保留 SOH 内部的 `轨迹视图 / 全体电池 3σ 分布` 子页签。

6. 保留异常检测内部的 fleet heatmap 和事件表，不拆分。

## SDD Agent Plan

采用小任务 + 审核 agent 的方式推进，每个实现 agent 完成后由独立 reviewer 检查风险和测试证据。

### Agent A: Native Dashboard Vendor

职责：

- 创建或更新 `frontend/app/battery-dashboard/`。
- 从 `D:\battery_SOH\dashboard` 复制 `index.html` 和 `data`。
- 只修改副本，不修改源目录。
- 添加 `mode`、`embed` URL 参数。

验收：

- `frontend/app/battery-dashboard/index.html?mode=soh&embed=1` 默认进入 SOH 页面。
- `frontend/app/battery-dashboard/index.html?mode=detection&embed=1` 默认进入异常检测页面。
- `embed=1` 时隐藏内部主模式切换。
- 数据文件完整存在，尤其是 `_curves.json`。

### Agent B: Parent App Shell Integration

职责：

- 修改 `frontend/app/index.html`，新增两个左侧导航项、顶部页签和面板。
- 修改 `frontend/app/app.js`，为 iframe 增加懒加载：
  - 首次打开 SOH 页签时设置 `src="./battery-dashboard/index.html?mode=soh&embed=1"`。
  - 首次打开异常检测页签时设置 `src="./battery-dashboard/index.html?mode=detection&embed=1"`。
- 保持 `renderModeTabs()` 的通用切换机制。

验收：

- 现有五个页签行为不变。
- 点击两个新页签能显示对应 iframe。
- 未打开时不提前加载两个大页面。

### Agent C: Layout And Visual QA

职责：

- 修改 `frontend/app/styles.css`，新增 `.native-dashboard-panel`、`.native-dashboard-frame` 样式。
- 解决高度不一致、页面过高、iframe 边框和滚动体验问题。
- 保证父页面导航和子页面内部内容视觉层级清晰。

验收：

- 1365x768、1440x900、1920x1080 三个视口下，iframe 不溢出父容器。
- 左侧导航不因新增两个项造成明显拥挤。
- 新页签和现有 SaaS 工作台风格一致。

### Agent D: Tests And Acceptance

职责：

- 先写失败测试，再配合实现 agent 让测试通过。
- 覆盖静态 HTML、子应用参数、数据文件和基础 smoke。

验收：

- Node test 通过。
- Python test 通过。
- 浏览器 smoke 通过。
- 记录测试命令和结果。

## TDD Test Plan

### Test 1: Parent Navigation Contract

文件：

- `frontend/app/chinese-ui.test.mjs`

先新增失败断言：

- HTML 中存在 `data-mode="battery-soh-full"`。
- HTML 中存在 `data-mode="battery-detection-full"`。
- HTML 中存在 `data-side-mode="battery-soh-full"`。
- HTML 中存在 `data-side-mode="battery-detection-full"`。
- HTML 中存在 `data-mode-panel="battery-soh-full"`。
- HTML 中存在 `data-mode-panel="battery-detection-full"`。

### Test 2: Lazy Iframe Source Contract

文件：

- `frontend/app/chinese-ui.test.mjs` 或新增 `frontend/app/native-dashboard.test.mjs`

断言：

- SOH iframe src 包含 `battery-dashboard/index.html?mode=soh&embed=1`。
- 异常检测 iframe src 包含 `battery-dashboard/index.html?mode=detection&embed=1`。
- 初始 HTML 不强制同时加载两个 iframe，实际 `src` 可通过 `data-src` 或懒加载函数设置。

### Test 3: Native Dashboard URL Params

文件：

- `frontend/app/native-dashboard.test.mjs`

静态断言：

- `battery-dashboard/index.html` 包含 `URLSearchParams`。
- 包含 `mode` 参数处理。
- 包含 `embed` 参数处理。
- 初始化调用不再写死 `setMode("soh")`。

### Test 4: Native Data Completeness

文件：

- `tests/test_battery_dashboard_native_assets.py`

断言：

- `frontend/app/battery-dashboard/data/manifest.json` 存在。
- `ess_detection.json` 和 `ess_detection_curves.json` 存在。
- manifest 中列出的 scenario 数据存在。
- 每个有曲线的 scenario 对应 `_curves.json` 存在。

### Test 5: Existing Regression Suite

命令：

```powershell
node --test frontend\app\dashboard-core.test.mjs frontend\app\chinese-ui.test.mjs frontend\app\native-dashboard.test.mjs
```

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_api_server.py tests\test_battery_dashboard_import.py tests\test_standalone_frontend.py tests\test_battery_dashboard_native_assets.py -q
```

### Test 6: Browser Smoke

启动：

```powershell
python -m http.server 8010 --directory frontend/app
```

验收：

- 打开 `http://127.0.0.1:8010/?v=native-battery-tabs`
- 点击 `SOH 完整看板`
  - iframe 页面可见。
  - Canvas `#soh` 非空白。
  - 点击轨迹点后 V/I/T/IC 曲线可见。
- 点击 `异常检测看板`
  - fleet heatmap 可见。
  - 点击模块/时间段后信号曲线和事件表可见。

## Risk Register

### Risk: Large Curve Files

完整 `_curves.json` 体积较大，首次点击某些数据集曲线会较慢。

Mitigation:

- 第一阶段保持源页面行为，优先保真。
- 后续可增加按 key 懒加载曲线接口，把 `_curves.json` 切分或后端索引化。

### Risk: Standalone Artifact Size

如果把完整曲线内嵌到单个 HTML，产物会很大。

Mitigation:

- 默认 standalone 不内嵌完整曲线。
- 增加显式完整曲线打包模式。
- 文档中标注体积和运行建议。

### Risk: Parent And Child Double Navigation

父页面已有 mode tabs，子页面也有 `.modetabs`。

Mitigation:

- `embed=1` 隐藏子页面主模式切换。
- 保留 SOH 内部子视图页签，因为它属于业务内容。

### Risk: Cross-Origin If Using Separate Port

如果 iframe 指向 `http://localhost:8350`，父页面无法直接控制子页面。

Mitigation:

- 最终集成使用 `./battery-dashboard/index.html` 同源静态子应用。
- 开发调试时才允许外部端口。

### Risk: Source Drift

后续 `D:\battery_SOH` 更新后，副本可能落后。

Mitigation:

- 增加同步脚本，例如 `scripts/sync_battery_dashboard_native.py`。
- 脚本只复制白名单文件，并保留本项目的 URL 参数补丁。
- 测试数据完整性和入口参数，防止同步后退化。

## Implementation Order

1. 写测试：
   - parent navigation contract
   - native dashboard param contract
   - native data completeness
2. Vendor 子应用：
   - 复制 `D:\battery_SOH\dashboard`
   - patch URL 参数
   - 保留完整曲线数据
3. 接入父页面：
   - 新增左侧导航、顶部页签、两个 panel
   - iframe 懒加载
4. 样式修正：
   - 高度统一
   - iframe 容器滚动
   - 和现有 SaaS 风格一致
5. 浏览器验收：
   - SOH 完整看板
   - 异常检测看板
6. 文档更新：
   - 使用说明
   - 数据同步说明
   - standalone 完整曲线限制说明

## Done Criteria

- 当前工作台出现两个新独立页签：`SOH 完整看板`、`异常检测看板`。
- 两个页签分别打开 `battery_SOH` 原生 SOH 和异常检测页面。
- SOH 页签可完整展示轨迹、3σ 和曲线钻取。
- 异常检测页签可完整展示 fleet heatmap、信号曲线和事件表。
- 现有五个页签无回归。
- 自动化测试通过。
- 浏览器 smoke 验收通过。

