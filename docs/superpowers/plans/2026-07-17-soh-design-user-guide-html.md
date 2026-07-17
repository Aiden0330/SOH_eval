# SOH 设计与工具使用说明书 HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 交付 `docs/soh-evaluation-system-design-and-user-guide.html`，将当前可信测评架构、Battery SOH 领域能力、五页签前端和完整操作流程整合为一个可离线迁移、可搜索、可打印的中文单文件说明书。

**Architecture:** 以一个语义化 HTML 为唯一运行时交付物，CSS、JavaScript 和五张真实前端截图全部内联。用一个 Python 标准库验证器统一检查结构、来源一致性、离线性、中文编码、锚点和内嵌资产；pytest 负责回归契约，真实 Chromium 负责交互、响应式、打印和零网络请求验收。

**Tech Stack:** HTML5、CSS Custom Properties、原生 JavaScript、Python 3.11 标准库、pytest、Chromium/Playwright 浏览器验收。

## Global Constraints

- 最终文件必须是 `docs/soh-evaluation-system-design-and-user-guide.html`，可直接通过 `file://` 打开。
- 最终 HTML 不显示文档版本号、发布日期、更新日期或历史长跑时间线。
- 所有 CSS、JavaScript 和图片必须内联；不得依赖 CDN、在线字体、外部脚本、远程图片或网络 API。
- 正文必须明确区分 `当前已实现`、`已接入领域能力` 和 `目标架构`，不得把规划能力写成已上线能力。
- 内容事实优先级固定为：当前代码/测试/运行契约，Verification Fabric 规格与计划，实际数据和前端/API 产物，`D:\battery_SOH` 设计与研究文档，经核对的旧手册。
- 说明书必须同时服务管理者、研发人员和工具使用者；阅读路径只能改变高亮，不得隐藏正文。
- 必须覆盖十三个主章节、当前五个前端页签、训练与冻结推理分离、资源接入、证据解释、同步打包、排障、安全和速查。
- 必须使用当前候选前端五个页签的真实截图；截图不得包含个人信息、令牌、绝对临时路径或敏感日志。
- 架构图、流程图、状态机和证据链使用 HTML/CSS，不引入外部图形库。
- 桌面采用深色左导航、中央正文、右侧速查栏；中等宽度收回速查栏；移动端折叠目录。
- 1280x720 和 390x844 均不得出现页面级横向溢出、控件重叠或长文本撑破容器。
- JavaScript 关闭时正文、锚点导航、章节内容和图片说明仍必须完整可读。
- 打印样式必须隐藏导航、搜索、复制按钮、阅读路径和返回顶部控件，并避免关键图表被截断。
- 不得嵌入真实 Pickle、可执行插件源码、个人数据或运行密钥。
- 正文和源代码使用 UTF-8；不得出现 U+FFFD 或已知乱码片段。

---

### Task 1: 建立离线文档契约和工作台骨架

**Files:**
- Create: `scripts/validate_design_user_guide.py`
- Create: `tests/test_design_user_guide.py`
- Create: `docs/soh-evaluation-system-design-and-user-guide.html`

**Interfaces:**
- Produces: `validate_document(path: Path, *, source_root: Path = ROOT) -> list[ValidationIssue]`
- Produces: `ValidationIssue(code: str, message: str)`
- Produces: 稳定章节 ID `overview`, `architecture`, `domain`, `evaluation`, `contracts`, `frontend`, `setup`, `onboarding`, `runs`, `evidence`, `packaging`, `troubleshooting`, `reference`
- Consumes: `src/sohbench/api_server.py` 中的当前 API 路由和 `frontend/app/index.html` 中的当前五页签名称

- [ ] **Step 1: 写缺少交付 HTML 时必然失败的契约测试**

```python
# tests/test_design_user_guide.py
from pathlib import Path

from scripts.validate_design_user_guide import REQUIRED_SECTIONS, validate_document


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html"


def test_design_user_guide_satisfies_offline_structure_contract():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert issues == []


def test_design_user_guide_contains_every_required_chapter():
    text = GUIDE.read_text(encoding="utf-8")
    for section_id in REQUIRED_SECTIONS:
        assert f'id="{section_id}"' in text
```

- [ ] **Step 2: 运行测试并确认因 HTML/验证器尚不存在而失败**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: FAIL during import or with `FileNotFoundError` for `docs/soh-evaluation-system-design-and-user-guide.html`.

- [ ] **Step 3: 实现标准库验证器的结构、离线和乱码检查**

```python
# scripts/validate_design_user_guide.py
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_SECTIONS = (
    "overview",
    "architecture",
    "domain",
    "evaluation",
    "contracts",
    "frontend",
    "setup",
    "onboarding",
    "runs",
    "evidence",
    "packaging",
    "troubleshooting",
    "reference",
)
REQUIRED_STATUS_LABELS = ("当前已实现", "已接入领域能力", "目标架构")
REQUIRED_TABS = ("训练测评", "推理测评", "模型资产库", "SOH 完整看板", "全局诊断")
MOJIBAKE_MARKERS = ("\ufffd", "Ã", "Â", "â†", "ç›", "æµ", "è¯", "锟斤拷")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


class GuideParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.anchors: list[str] = []
        self.images: list[dict[str, str]] = []
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []
        self.buttons: list[dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if element_id := values.get("id"):
            self.ids.add(element_id)
        if tag == "a" and (href := values.get("href")):
            self.anchors.append(href)
        if tag == "img":
            self.images.append(values)
        if tag == "link" and values.get("rel") == "stylesheet":
            self.stylesheets.append(values.get("href", ""))
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        if tag == "button":
            self.buttons.append(values)


def _source_routes(source_root: Path) -> set[str]:
    source = (source_root / "src" / "sohbench" / "api_server.py").read_text(encoding="utf-8")
    collections = set(re.findall(r'path\s*==\s*["\'](/api/[^"\']+)', source))
    prefixes = set(re.findall(r'path\.startswith\(\s*["\'](/api/[^"\']+)', source))
    details = {f"{prefix.rstrip('/')}/{{job_id}}" for prefix in prefixes}
    return collections | details


def validate_document(path: Path, *, source_root: Path = ROOT) -> list[ValidationIssue]:
    if not path.is_file():
        return [ValidationIssue("missing-file", f"missing guide: {path}")]
    text = path.read_text(encoding="utf-8")
    parser = GuideParser()
    parser.feed(text)
    issues: list[ValidationIssue] = []
    if '<html lang="zh-CN">' not in text or '<meta charset="utf-8">' not in text.lower():
        issues.append(ValidationIssue("encoding", "lang or UTF-8 metadata is missing"))
    for section_id in REQUIRED_SECTIONS:
        if section_id not in parser.ids:
            issues.append(ValidationIssue("section", f"missing section #{section_id}"))
    for href in parser.anchors:
        if href.startswith("#") and href[1:] not in parser.ids:
            issues.append(ValidationIssue("anchor", f"unresolved anchor {href}"))
        elif href and not href.startswith("#"):
            issues.append(ValidationIssue("external-link", f"non-anchor link {href}"))
    if parser.stylesheets or parser.scripts:
        issues.append(ValidationIssue("external-asset", "external stylesheet or script found"))
    for image in parser.images:
        if not image.get("src", "").startswith("data:image/"):
            issues.append(ValidationIssue("image", f"image is not embedded: {image.get('id', '<unknown>')}"))
        if not image.get("alt", "").strip():
            issues.append(ValidationIssue("image-alt", f"image has no alt text: {image.get('id', '<unknown>')}"))
    for marker in MOJIBAKE_MARKERS:
        if marker in text:
            issues.append(ValidationIssue("mojibake", f"found marker {marker!r}"))
    if re.search(r'(?:src|href)=["\']https?://|url\(\s*["\']?https?://', text, flags=re.IGNORECASE):
        issues.append(ValidationIssue("network", "network-backed asset or active HTTP(S) link found"))
    for label in REQUIRED_STATUS_LABELS + REQUIRED_TABS:
        if label not in text:
            issues.append(ValidationIssue("content", f"missing required label {label}"))
    for route in _source_routes(source_root):
        if route not in text:
            issues.append(ValidationIssue("api-parity", f"missing current API route {route}"))
    return issues


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("validate standalone SOH design and user guide")
    parser.add_argument("path", nargs="?", type=Path, default=ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html")
    args = parser.parse_args(argv)
    issues = validate_document(args.path)
    for issue in issues:
        print(f"{issue.code}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The route extractor converts prefix routes to documented detail forms such as `/api/jobs/{job_id}`, preserving both collection and detail semantics in the route table.

- [ ] **Step 4: 创建无占位词的语义化工作台骨架**

Create one HTML document with:

```html
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SOH 算法测评系统设计与工具使用说明书</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { margin: 0; color: #101828; background: #f3f4f6; font-family: "Microsoft YaHei UI", system-ui, sans-serif; }
    .guide-shell { display: grid; grid-template-columns: 240px minmax(0, 1fr) 260px; min-height: 100vh; }
    .guide-nav { padding: 24px 16px; color: #fff; background: #111827; }
    .guide-nav a { display: block; padding: 8px; color: #dbeafe; text-decoration: none; }
    main { min-width: 0; padding: 32px; background: #fff; }
    .chapter { max-width: 920px; margin: 0 auto 24px; padding: 24px 0; border-bottom: 1px solid #dbe3ee; }
    .quick-rail { padding: 24px 16px; background: #f8fafc; }
    @media (max-width: 720px) {
      .guide-shell { display: block; }
      main { padding: 16px; }
    }
  </style>
</head>
<body>
  <a class="skip-link" href="#main-content">跳到正文</a>
  <div class="guide-shell">
    <aside class="guide-nav" aria-label="说明书目录">
      <p class="product-name">SOH Evaluation Fabric</p>
      <nav>
        <a href="#overview">项目概览</a>
        <a href="#architecture">总体架构与数据流</a>
        <a href="#domain">SOH 数据、特征与算法原理</a>
        <a href="#evaluation">公平测评协议与指标</a>
        <a href="#contracts">核心契约与证据对象</a>
        <a href="#frontend">前端五页签功能说明</a>
        <a href="#setup">工具安装、启动与配置</a>
        <a href="#onboarding">新资源接入</a>
        <a href="#runs">训练测评与冻结推理</a>
        <a href="#evidence">结果、排行与证据解释</a>
        <a href="#packaging">数据同步与离线打包</a>
        <a href="#troubleshooting">常见问题、安全边界与排障</a>
        <a href="#reference">API、命令和目录速查</a>
      </nav>
    </aside>
    <main id="main-content">
      <header class="guide-header">
        <p class="eyebrow">算法测评优先 · 多场景能力演进</p>
        <h1>SOH 算法测评系统设计与工具使用说明书</h1>
        <p>统一说明可信测评架构、Battery SOH 领域能力和本机工作台操作方法。</p>
      </header>
      <section id="overview" class="chapter">
        <h2>项目概览</h2>
        <p>系统以可比较、可追溯、可复核的算法测评为第一目标，并以跨数据集、跨电芯和跨化学体系的多场景算法建设为第二目标。</p>
        <ul>
          <li><strong>当前已实现</strong>：本机资源目录、训练/推理任务、结果发布和证据基础。</li>
          <li><strong>已接入领域能力</strong>：Battery SOH 曲线、特征、算法和完整看板。</li>
          <li><strong>目标架构</strong>：隔离 Worker、前端 V2 和更强的外部插件治理。</li>
        </ul>
      </section>
      <section id="architecture" class="chapter"><h2>总体架构与数据流</h2><p>当前代码提供本机控制面、训练与推理执行、结果发布和证据对象；隔离 Worker 属于目标架构。</p></section>
      <section id="domain" class="chapter"><h2>SOH 数据、特征与算法原理</h2><p>领域能力围绕 SOH 标签、充电曲线、IC/DVA、健康指标和多类回归估计器组织。</p></section>
      <section id="evaluation" class="chapter"><h2>公平测评协议与指标</h2><p>公平比较必须锁定数据快照、协议、随机种子、留出覆盖和统计单位。</p></section>
      <section id="contracts" class="chapter"><h2>核心契约与证据对象</h2><p>RunSpec、DatasetSnapshot、PredictionRecord 和 EvidencePack 共同形成可复核证据链。</p></section>
      <section id="frontend" class="chapter"><h2>前端五页签功能说明</h2><p>训练测评、推理测评、模型资产库、SOH 完整看板和全局诊断承担不同职责。</p></section>
      <section id="setup" class="chapter"><h2>工具安装、启动与配置</h2><p>前端通过静态 HTTP 服务加载，后端 API 默认连接本机 Battery SOH 数据目录。</p></section>
      <section id="onboarding" class="chapter"><h2>新资源接入</h2><p>数据集、算法、特征方案和冻结模型均先验证，成功后再事务性激活。</p></section>
      <section id="runs" class="chapter"><h2>训练测评与冻结推理</h2><p>训练页负责产生模型，推理页负责验证已有模型在新数据上的迁移精度和速度。</p></section>
      <section id="evidence" class="chapter"><h2>结果、排行与证据解释</h2><p>结果文件、质量报告和证据包必须结合覆盖范围、执行结果和验收裁决共同解释。</p></section>
      <section id="packaging" class="chapter"><h2>数据同步与离线打包</h2><p>同步与打包采用不可变 generation 和 last-known-good 发布语义。</p></section>
      <section id="troubleshooting" class="chapter"><h2>常见问题、安全边界与排障</h2><p>排障从预检、任务状态、特征兼容性、前端缓存和结构化失败信息逐层定位。</p></section>
      <section id="reference" class="chapter">
        <h2>API、命令和目录速查</h2>
        <p>本章汇总当前代码可核验的入口、命令、目录和产物。</p>
        <p class="mono">GET /api/health · GET /api/catalog · GET /api/models · GET /api/battery-curves · GET /api/package/status · GET /api/jobs · GET /api/jobs/{job_id} · GET /api/inference-jobs · GET /api/inference-jobs/{job_id}</p>
        <p class="mono">POST /api/upload/dataset · POST /api/upload/algorithm · POST /api/upload/featureset · POST /api/upload/model · POST /api/jobs · POST /api/inference-jobs · POST /api/import/battery-dashboard · POST /api/import/battery-soh-plugins · POST /api/sync/native-dashboard · POST /api/package/portable · POST /api/package/standalone</p>
      </section>
    </main>
    <aside class="quick-rail" aria-label="当前章节速查"><h2>本章速查</h2><div id="quick-content">选择章节查看关键输入、命令、产物和风险。</div></aside>
  </div>
  <script>
    (() => {
      "use strict";
      document.documentElement.classList.add("js");
    })();
  </script>
</body>
</html>
```

- [ ] **Step 5: 运行验证器和契约测试**

Run:

```powershell
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: both commands exit `0`; every required chapter, status label, tab label and current API route is present.

- [ ] **Step 6: 提交工作台骨架**

```powershell
git add scripts/validate_design_user_guide.py tests/test_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
git commit -m "docs: scaffold standalone SOH design guide"
```

---

### Task 2: 写入可信架构、SOH 原理和公平测评设计

**Files:**
- Modify: `docs/soh-evaluation-system-design-and-user-guide.html`
- Modify: `tests/test_design_user_guide.py`

**Interfaces:**
- Consumes: `RunSpec`, `DatasetSnapshot`, `PredictionRecord`, `EvidencePack`, `execution_state`, `outcome`, `verdict`
- Consumes: `D:\battery_SOH\data\processed\*_features.pkl` 发现契约
- Produces: 管理者和研发阅读路径所需的架构、领域、协议、指标与证据正文

- [ ] **Step 1: 添加领域与架构内容完整性测试**

```python
def test_design_user_guide_covers_fabric_and_soh_domain_terms():
    text = GUIDE.read_text(encoding="utf-8")
    expected = {
        "RunSpec",
        "DatasetSnapshot",
        "PredictionRecord",
        "EvidencePack",
        "execution_state",
        "outcome",
        "verdict",
        "LOCO",
        "LODO",
        "LCHO",
        "Frozen Inference Run",
        "MAE",
        "RMSE",
        "PICP",
        "IC/DVA",
        "partial_window",
        "capacity_time",
        "Target-stripped fold view",
    }
    assert expected <= set(term for term in expected if term in text)


def test_design_user_guide_keeps_current_and_target_capabilities_distinct():
    text = GUIDE.read_text(encoding="utf-8")
    assert text.count("当前已实现") >= 5
    assert text.count("已接入领域能力") >= 3
    assert text.count("目标架构") >= 3
    assert "隔离 Worker" in text
    assert "规划能力" in text
```

- [ ] **Step 2: 运行新增测试并确认正文深度不足**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: FAIL because detailed architecture/domain terms and status occurrences are not yet complete.

- [ ] **Step 3: 完成项目定位、五层架构和端到端工作流**

In `#overview` and `#architecture`, add:

- 三类受众阅读路径按钮，分别设置 `data-audience="manager|developer|operator"`。
- 三栏能力状态总览，逐项标注当前、领域接入或目标架构。
- 五层架构图：体验与入口层、控制面、执行面、证据面、Battery SOH Domain Pack。
- 端到端流程图：资源接入、内容摘要、Spec 固化、预检、用户确认、执行、指标计算、证据验证、排行发布、前端展示。
- 当前边界说明：Phase 0 已有内存/本机执行与证据基础；独立 Worker、V2 控制台和更强外部插件治理属于规划能力。
- 一张“为什么可信”的证据链图，显示输入摘要、运行规格、预测记录、指标、EvidencePack 和发布物之间的摘要关联。

Use semantic lists and CSS grid nodes; every diagram must also include a visually hidden ordered-list fallback.

- [ ] **Step 4: 完成 SOH 数据、特征、算法和公平协议正文**

In `#domain` and `#evaluation`, include:

- 数据发现：`D:\battery_SOH\data\processed\*_features.pkl`，并说明 `dataset`, `cell_id`, `cycle_index`, `soh` 的结构性意义。
- 数据质量：缺失列、非有限标签、重复周期、轨迹顺序、样本/电芯覆盖和训练/测试泄漏风险。
- 特征方案：`full`, `voltage`, `ic_dva`, `partial_window`, `capacity_time`。
- 领域原理：固定充电窗口、增量容量 IC、微分电压 DVA、健康指标相关性/单调性/鲁棒性、因果平滑和折内特征计算。
- 算法分组：线性/正则化、树模型、核方法、神经网络、物理残差和 PINN；每组说明归纳偏置、优点、限制和适用样本规模。
- 方法与公开数据集综述：归纳数据驱动、混合物理和不确定度方法，说明公开 LFP/NMC 数据在化学体系、工况、温度、采样和标签口径上的差异；只使用项目研究文档可核验的名称和结论，参考项按作者/项目名称列出而不显示发表年份。
- 项目 gap：明确当前在统一数据卡、跨化学体系覆盖、冻结模型迁移、轨迹/曲线联动和统计证据方面已经覆盖与仍需建设的部分，并继续使用三种状态标签。
- 责任边界：点估计、不确定度、后处理与周期重标定不得混为同一算法能力。
- 协议对比：LOCO、LODO、LCHO，以及 Training Run、Adaptation Run、Frozen Inference Run。
- 公平性前提：同数据快照、同协议、同 seed、同 held-out 覆盖、同特征拟合边界。
- 指标：MAE、RMSE、MAPE、偏差、最大误差、PICP、单调性、越界率、EOL；明确轨迹级、资产级和数据集级统计单位。
- 排行资格：完整覆盖、部分成功、失败组合和不具可比性的处理。

Do not claim a fixed current dataset count; instruct readers to use `/api/catalog` and the frontend catalog for the current discovered count.

- [ ] **Step 5: 完成核心契约、状态机和失败语义**

In `#contracts`, add:

```text
RunSpec -> DatasetSnapshot -> target-stripped fold view
        -> fitted FeatureState -> ModelArtifact
        -> PredictionRecord -> Metrics -> EvidencePack -> Publication
```

Explain:

- RunSpec comparison key and immutable input references.
- DatasetSnapshot content digest and lineage.
- Target-stripped fold view as the leakage boundary.
- PredictionRecord finite-value and sample-identity requirements.
- EvidencePack manifest, digest verification and atomic publication.
- `execution_state` is lifecycle, `outcome` is execution result, `verdict` is acceptance decision.
- Structured failure categories and immutable generations.

Render the state model as `queued -> running -> terminal`, with terminal outcomes `succeeded`, `partially_succeeded`, `failed`, and a separate verdict lane `accepted`, `rejected`, `not_evaluated`.

- [ ] **Step 6: 运行内容、离线和回归测试**

Run:

```powershell
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py tests/fabric/test_specs.py tests/fabric/test_evidence.py tests/fabric/test_outcomes.py -q
```

Expected: all tests PASS; the guide uses current contract names and does not weaken fabric behavior.

- [ ] **Step 7: 提交架构与领域正文**

```powershell
git add docs/soh-evaluation-system-design-and-user-guide.html tests/test_design_user_guide.py
git commit -m "docs: explain SOH verification architecture"
```

---

### Task 3: 写入五页签、资源接入和端到端操作手册

**Files:**
- Modify: `docs/soh-evaluation-system-design-and-user-guide.html`
- Modify: `tests/test_design_user_guide.py`

**Interfaces:**
- Consumes: `src/sohbench/api_server.py::parse_args`
- Consumes: `frontend/app/index.html` 五个 `data-mode-panel`
- Consumes: `scripts/build_frontend_artifacts.py`, `scripts/sync_battery_dashboard_native.py`, `scripts/package_portable_frontend.py`, `scripts/build_standalone_frontend.py`
- Produces: 可直接执行的安装、启动、上传、训练、推理、发布、同步、打包和排障步骤

- [ ] **Step 1: 添加命令、页签和产物覆盖测试**

```python
def test_design_user_guide_documents_current_frontend_modes_and_artifacts():
    text = GUIDE.read_text(encoding="utf-8")
    for mode in ("训练测评", "推理测评", "模型资产库", "SOH 完整看板", "全局诊断"):
        assert f'data-guide-mode="{mode}"' in text
    for artifact in (
        "results.csv",
        "results.parquet",
        "leaderboard.json",
        "dataset_leaderboard.json",
        "qc_report.json",
        "run_config.json",
        "inference_benchmark.json",
        "inference_run_config.json",
        "CURRENT.json",
    ):
        assert artifact in text


def test_design_user_guide_has_verified_start_and_package_commands():
    text = GUIDE.read_text(encoding="utf-8")
    assert "python.exe -m sohbench.api_server" in text
    assert "python.exe -m http.server 8010 --directory frontend/app" in text
    assert "scripts\\package_portable_frontend.py" in text
    assert "scripts\\build_standalone_frontend.py" in text
```

- [ ] **Step 2: 运行新增测试并确认操作内容不完整**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: FAIL on missing `data-guide-mode`, commands or artifact names.

- [ ] **Step 3: 完成安装、启动和健康检查**

Use copyable PowerShell blocks with the exact current commands:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e ".[dev]"
$env:PYTHONPATH = "src"
.venv\Scripts\python.exe -m sohbench.api_server --host 127.0.0.1 --port 8001 --data-dir "D:\battery_SOH\data" --work-dir "work\trusted_plugins" --runs-dir "runs" --frontend-data-dir "frontend\app\data"
```

In a second terminal:

```powershell
.venv\Scripts\python.exe -m http.server 8010 --directory frontend/app
```

Open:

```text
http://127.0.0.1:8010/?api=http://127.0.0.1:8001
```

Because the standalone guide forbids external hyperlinks, render the URL as copyable code rather than an `<a href>`. Add a health-check example for `GET /api/health`, a port-conflict check with `Get-NetTCPConnection`, and a stop procedure that terminates only the captured process IDs.

- [ ] **Step 4: 为五个前端页签写独立任务卡**

For every page use an `<article class="mode-guide" data-guide-mode="...">` and exactly six subsections:

1. 适用场景
2. 必需输入
3. 操作步骤
4. 成功信号
5. 常见失败
6. 关联产物

Required focus:

- 训练测评：源码算法/特征上传、组合矩阵、协议、seed、任务状态、Pickle 发布。
- 推理测评：冻结 Pickle、特征兼容性预检、目标数据、迁移精度、`predict_ms` 和吞吐。
- 模型资产库：模型来源、训练范围、特征契约、摘要、兼容性和推理审计。
- SOH 完整看板：SOH 轨迹、点击周期点、充电电压/容量/时间曲线、IC/DVA 和高亮特征段。
- 全局诊断：数据覆盖、算法 x 特征矩阵、稳定性、QC 和失败组合；不承担单点曲线诊断。

- [ ] **Step 5: 完成四类资源接入与运行流程**

In `#onboarding` and `#runs`, document:

- Dataset Pickle table and required columns; validate before transactional activation.
- Trusted local source algorithm plugin and estimator interface; explicitly state that source plugins execute local code.
- Feature column-selection or fold-local transform; fitted state must not cross held-out boundaries.
- Frozen Pickle model plus manifest/feature contract; explicitly warn that untrusted Pickle deserialization can execute code.
- Upload endpoints, validation, rollback and safe-path behavior.
- Training matrix size calculation and why deep models can remain `running` for a long time.
- Model publication after successful training.
- Inference-only migration benchmark, pairwise compatibility, failed combinations and partial success.
- Job history, polling, terminal state and rerun guidance.

- [ ] **Step 6: 完成结果、同步、打包和排障正文**

In `#evidence`, `#packaging`, `#troubleshooting`, and `#reference`, add:

- Result artifact table for training, inference, QC, run config, model catalog and EvidencePack.
- Correct interpretation of KPI, coverage and “验收通过” counts; explain why zero can be valid when no terminal accepted records exist.
- Dataset selection semantics: selected datasets must drive filtering/reaggregation, not only labels.
- Native dashboard sync and frontend artifact build.
- Portable directory package and standalone single HTML package.
- Immutable generation, `CURRENT.json`, validation-before-publish and last-known-good behavior.
- Troubleshooting decision tables for dataset discovery, unchanged results, long-running training, unresponsive inference, zero accepted runs, Pickle mismatch, stale frontend cache, missing native dashboard data, layout overflow/mojibake and transactional upload rollback.
- Security callouts for path traversal, Windows device aliases, trusted source code and Pickle.
- Complete GET/POST API table derived from the current route set.

Use exact current commands:

```powershell
.venv\Scripts\python.exe scripts\build_frontend_artifacts.py --help
.venv\Scripts\python.exe scripts\sync_battery_dashboard_native.py --help
.venv\Scripts\python.exe scripts\package_portable_frontend.py --output dist\soh-eval-portable
.venv\Scripts\python.exe scripts\build_standalone_frontend.py --output dist\soh-eval-standalone.html
```

The `--help` commands are intentional where required inputs may vary; nearby text must state which concrete arguments the current command reports.

- [ ] **Step 7: 运行操作文档与相关代码回归**

Run:

```powershell
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py tests/test_api_server.py tests/test_standalone_frontend.py tests/test_frontend_publish.py -q
```

Expected: all tests PASS; every live API route, frontend mode and package artifact is documented.

- [ ] **Step 8: 提交完整操作正文**

```powershell
git add docs/soh-evaluation-system-design-and-user-guide.html tests/test_design_user_guide.py
git commit -m "docs: add complete SOH workbench operations guide"
```

---

### Task 4: 捕获并内嵌五个真实前端页签

**Files:**
- Create: `scripts/embed_design_user_guide_screenshots.py`
- Modify: `docs/soh-evaluation-system-design-and-user-guide.html`
- Modify: `tests/test_design_user_guide.py`

**Interfaces:**
- Produces: `embed_screenshots(html_path: Path, captures: dict[str, Path]) -> None`
- Consumes: image markers `data-shot="training|inference|models|native|diagnostics"`
- Produces: five `data:image/webp;base64,...` or `data:image/png;base64,...` image sources with descriptive alt text

- [ ] **Step 1: 添加截图完整性和 authoring helper 测试**

```python
import base64

from scripts.embed_design_user_guide_screenshots import embed_screenshots


def test_embed_screenshots_replaces_each_named_capture(tmp_path):
    html = tmp_path / "guide.html"
    html.write_text('<img data-shot="training" src="data:image/png;base64,AA==" alt="训练页">', encoding="utf-8")
    capture = tmp_path / "training.png"
    capture.write_bytes(b"\x89PNG\r\n\x1a\nfixture")
    embed_screenshots(html, {"training": capture})
    rendered = html.read_text(encoding="utf-8")
    expected = base64.b64encode(capture.read_bytes()).decode("ascii")
    assert f"data:image/png;base64,{expected}" in rendered


def test_design_user_guide_embeds_all_five_frontend_captures():
    import re

    text = GUIDE.read_text(encoding="utf-8")
    for shot in ("training", "inference", "models", "native", "diagnostics"):
        assert f'data-shot="{shot}"' in text
    payloads = re.findall(r'src="data:image/[^;]+;base64,([^"]+)"', text)
    assert len(payloads) >= 5
    assert all(len(base64.b64decode(payload)) >= 10_000 for payload in payloads)
```

- [ ] **Step 2: 运行新增测试并确认 helper/截图缺失**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: FAIL because the helper and five embedded captures do not yet exist.

- [ ] **Step 3: 实现原子截图内嵌 helper**

```python
# scripts/embed_design_user_guide_screenshots.py
from __future__ import annotations

import argparse
import base64
import os
import re
import tempfile
from pathlib import Path


MIME_BY_SUFFIX = {".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}


def _data_uri(path: Path) -> str:
    mime = MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported image type: {path.suffix}")
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def embed_screenshots(html_path: Path, captures: dict[str, Path]) -> None:
    text = html_path.read_text(encoding="utf-8")
    for name, capture in captures.items():
        if not capture.is_file():
            raise FileNotFoundError(capture)
        pattern = re.compile(
            rf'(<img\b(?=[^>]*\bdata-shot="{re.escape(name)}")(?=[^>]*\bsrc=")[^>]*\bsrc=")[^"]*(")',
            flags=re.IGNORECASE,
        )
        text, count = pattern.subn(rf"\g<1>{_data_uri(capture)}\2", text, count=1)
        if count != 1:
            raise ValueError(f"expected one image marker for {name}, found {count}")
    fd, staged_name = tempfile.mkstemp(prefix=f".{html_path.name}.", suffix=".tmp", dir=html_path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staged_name, html_path)
    except BaseException:
        Path(staged_name).unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("embed SOH guide screenshots")
    parser.add_argument("html", type=Path)
    for name in ("training", "inference", "models", "native", "diagnostics"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    embed_screenshots(args.html, {name: getattr(args, name) for name in ("training", "inference", "models", "native", "diagnostics")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

This helper is authoring-only; the finished HTML must not import or call it.

- [ ] **Step 4: 启动当前候选前后端并做截图前预检**

From the isolated worktree, start hidden processes and retain their PIDs:

```powershell
$env:PYTHONPATH = "src"
$api = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m","sohbench.api_server","--host","127.0.0.1","--port","8101","--data-dir","D:\battery_SOH\data","--work-dir","work\trusted_plugins","--runs-dir","runs","--frontend-data-dir","frontend\app\data" -WindowStyle Hidden -PassThru
$web = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList "-m","http.server","8110","--directory","frontend\app" -WindowStyle Hidden -PassThru
Invoke-RestMethod "http://127.0.0.1:8101/api/health"
```

Expected: health response is successful and the frontend opens at the copyable URL:

```text
http://127.0.0.1:8110/?api=http://127.0.0.1:8101
```

- [ ] **Step 5: 用浏览器逐页签捕获代表状态**

Use the in-app browser control skill or Playwright to:

1. Open the frontend URL at 1440x900.
2. Capture `training`, `inference`, `models`, `native`, and `diagnostics`.
3. Keep the product/sidebar as first-viewport context while cropping irrelevant browser chrome.
4. Confirm no password, token, user home path, temporary path, document version/date, timestamped task ID or sensitive run log is visible.
5. Save captures under a temporary authoring directory, not under `docs/`.
6. Prefer WebP quality around 82 when supported; otherwise use optimized PNG.

Each capture must show a meaningful loaded state. If the native dashboard is unavailable, run the documented sync action first; do not replace the missing state with a fabricated image.

- [ ] **Step 6: 在五页签任务卡中添加图注并内嵌截图**

Add one `<figure class="product-shot">` to every mode guide:

```html
<figure class="product-shot">
  <img data-shot="training" src="data:image/png;base64,AA==" alt="训练测评页：资源接入、组合配置、训练执行与模型发布区域">
  <figcaption><strong>训练测评</strong><span>图中展示资源选择、运行计划和训练产物的职责边界。</span></figcaption>
</figure>
```

Use matching descriptions for the other four pages, then run:

```powershell
.venv\Scripts\python.exe scripts\embed_design_user_guide_screenshots.py docs\soh-evaluation-system-design-and-user-guide.html --training "$env:TEMP\soh-guide\training.webp" --inference "$env:TEMP\soh-guide\inference.webp" --models "$env:TEMP\soh-guide\models.webp" --native "$env:TEMP\soh-guide\native.webp" --diagnostics "$env:TEMP\soh-guide\diagnostics.webp"
```

Adjust only the temporary extensions to the files actually captured.

- [ ] **Step 7: 停止本任务启动的服务并验证内嵌资产**

```powershell
Stop-Process -Id $api.Id,$web.Id
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: services stop cleanly; all five images are embedded data URIs with alt text and no external asset remains.

- [ ] **Step 8: 提交真实截图版说明书**

```powershell
git add scripts/embed_design_user_guide_screenshots.py tests/test_design_user_guide.py docs/soh-evaluation-system-design-and-user-guide.html
git commit -m "docs: embed current SOH workbench screenshots"
```

---

### Task 5: 完成交互、响应式和打印体验

**Files:**
- Modify: `docs/soh-evaluation-system-design-and-user-guide.html`
- Modify: `scripts/validate_design_user_guide.py`
- Modify: `tests/test_design_user_guide.py`

**Interfaces:**
- Produces: `setAudience(audience: "all" | "manager" | "developer" | "operator")`
- Produces: `filterGuide(query: string)`
- Produces: `copyCode(button: HTMLButtonElement)`
- Produces: `updateQuickRail(sectionId: string)`
- Produces: progressive enhancement for mobile navigation, scroll-spy and return-to-top

- [ ] **Step 1: 添加交互和响应式静态契约测试**

```python
def test_design_user_guide_has_progressive_interactions_and_print_rules():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        'id="guide-search"',
        'id="search-empty"',
        'data-audience="manager"',
        'data-audience="developer"',
        'data-audience="operator"',
        "function setAudience(",
        "function filterGuide(",
        "function copyCode(",
        "function updateQuickRail(",
        "IntersectionObserver",
        "@media (max-width: 1100px)",
        "@media (max-width: 720px)",
        "@media print",
    ):
        assert token in text


def test_all_icon_only_buttons_have_accessible_names():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert not [issue for issue in issues if issue.code == "button-name"]
```

- [ ] **Step 2: 运行新增测试并确认交互契约尚未满足**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: FAIL on missing controls, functions or media queries.

- [ ] **Step 3: 完成工作台视觉系统和稳定布局**

Implement inline CSS using:

```css
:root {
  --nav: #111827;
  --nav-soft: #1f2937;
  --canvas: #f3f4f6;
  --surface: #ffffff;
  --line: #dbe3ee;
  --text: #101828;
  --muted: #667085;
  --blue: #315ee7;
  --green: #2f7d4a;
  --amber: #a15c00;
  --red: #b42318;
  --radius: 8px;
  --shadow: 0 8px 24px rgba(16, 24, 40, 0.08);
}
*, *::before, *::after { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body { margin: 0; color: var(--text); background: var(--canvas); font-family: "Microsoft YaHei UI", "PingFang SC", "Noto Sans CJK SC", system-ui, sans-serif; }
code, pre, .mono { font-family: Consolas, "Cascadia Mono", monospace; overflow-wrap: anywhere; }
.guide-shell { display: grid; grid-template-columns: 240px minmax(0, 1fr) 260px; height: 100vh; overflow: hidden; }
.guide-nav, .quick-rail { height: 100vh; overflow: auto; }
main { min-width: 0; width: min(100%, 1040px); height: 100vh; margin: 0 auto; padding: 32px; overflow: auto; scroll-behavior: smooth; }
.chapter, .diagram, .product-shot, .table-scroll { min-width: 0; }
.quick-source { display: none; }
.table-scroll { overflow-x: auto; }
img { display: block; max-width: 100%; height: auto; }
@media (max-width: 1100px) {
  .guide-shell { grid-template-columns: 220px minmax(0, 1fr); }
  .quick-rail { display: none; }
  .quick-source { display: block; }
}
@media (max-width: 720px) {
  .guide-shell { display: block; height: auto; overflow: visible; }
  .guide-nav { position: sticky; z-index: 20; height: auto; max-height: 100vh; }
  main { width: 100%; height: auto; padding: 16px; overflow: visible; }
}
@media print {
  .guide-nav, .quick-rail, .search-tools, .audience-switcher, .copy-button, .back-to-top { display: none !important; }
  .guide-shell { display: block; }
  main { width: auto; max-width: none; padding: 0; }
  details:not([open]) > * { display: block !important; }
  .diagram, figure, table, pre { break-inside: avoid; }
  h2, h3 { break-after: avoid; }
  thead { display: table-header-group; }
}
```

Do not use gradients, decorative orbs, nested decorative cards or viewport-scaled typography. Give diagrams, tables, icon buttons and screenshot frames stable dimensions or aspect ratios so dynamic states cannot shift the layout.

- [ ] **Step 4: 完成搜索、阅读路径、折叠、复制和速查**

Use guarded inline JavaScript:

```javascript
(() => {
  "use strict";
  const chapters = [...document.querySelectorAll(".chapter")];
  const navLinks = [...document.querySelectorAll('.guide-nav a[href^="#"]')];
  const search = document.querySelector("#guide-search");
  const clearSearch = document.querySelector("#clear-search");
  const empty = document.querySelector("#search-empty");
  const nav = document.querySelector(".guide-nav");
  const navToggle = document.querySelector("#nav-toggle");
  const backToTop = document.querySelector(".back-to-top");

  function setAudience(audience) {
    document.documentElement.dataset.audience = audience;
    document.querySelectorAll("[data-audience]").forEach((button) => {
      button.setAttribute("aria-pressed", String(button.dataset.audience === audience));
    });
    navLinks.forEach((link) => {
      const chapter = document.querySelector(link.getAttribute("href"));
      link.classList.toggle("on-path", audience !== "all" && chapter?.dataset.paths?.split(" ").includes(audience));
    });
  }

  function filterGuide(query) {
    const needle = query.trim().toLocaleLowerCase("zh-CN");
    let visible = 0;
    chapters.forEach((chapter) => {
      const match = !needle || chapter.textContent.toLocaleLowerCase("zh-CN").includes(needle);
      chapter.hidden = !match;
      visible += Number(match);
    });
    empty.hidden = visible !== 0;
  }

  async function copyCode(button) {
    const code = button.closest(".code-block")?.querySelector("code");
    if (!code) return;
    const value = code.textContent;
    try {
      await navigator.clipboard.writeText(value);
    } catch {
      const range = document.createRange();
      range.selectNodeContents(code);
      const selection = window.getSelection();
      if (!selection) return;
      selection.removeAllRanges();
      selection.addRange(range);
    }
    button.textContent = "已复制";
    window.setTimeout(() => { button.textContent = "复制"; }, 1200);
  }

  function updateQuickRail(sectionId) {
    const chapter = document.getElementById(sectionId);
    const source = chapter?.querySelector(".quick-source");
    const target = document.querySelector("#quick-content");
    if (source && target) target.innerHTML = source.innerHTML;
  }

  search?.addEventListener("input", (event) => filterGuide(event.target.value));
  clearSearch?.addEventListener("click", () => {
    search.value = "";
    filterGuide("");
    search.focus();
  });
  document.querySelectorAll(".copy-button").forEach((button) => button.addEventListener("click", () => copyCode(button)));
  document.querySelectorAll("[data-audience]").forEach((button) => button.addEventListener("click", () => setAudience(button.dataset.audience)));
  navToggle?.addEventListener("click", () => {
    const open = nav.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(open));
  });
  backToTop?.addEventListener("click", () => {
    const scroller = window.matchMedia("(max-width: 720px)").matches ? window : document.querySelector("main");
    scroller.scrollTo({ top: 0, behavior: "smooth" });
  });
  if ("IntersectionObserver" in window) {
    const observer = new IntersectionObserver((entries) => {
      const current = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
      if (!current) return;
      navLinks.forEach((link) => link.classList.toggle("active", link.getAttribute("href") === `#${current.target.id}`));
      updateQuickRail(current.target.id);
    }, { rootMargin: "-15% 0px -70% 0px", threshold: [0, 0.2, 0.6] });
    chapters.forEach((chapter) => observer.observe(chapter));
  }
  setAudience("all");
  updateQuickRail("overview");
})();
```

Add:

- Accessible search label and clear button.
- `role="status"` empty result message.
- 全文搜索、目录定位、章节折叠和代码复制四项文档交互。
- Native `<details open>` subsections so no-JS retains all content.
- Text+icon mobile navigation button with `aria-expanded`.
- Icon-only return-to-top button with `aria-label="返回顶部"` and tooltip.
- `aria-live="polite"` copy status.
- Quick-source content per chapter with inputs, commands, outputs, checks and risks.

- [ ] **Step 5: 扩展验证器检查按钮名称和内联脚本**

Update `GuideParser` to retain button attributes and check `aria-label`. Add:

```python
for button in parser.buttons:
    if button.get("data-icon-only") == "true" and not button.get("aria-label", "").strip():
        issues.append(ValidationIssue("button-name", "icon-only button has no aria-label"))
if "@media print" not in text or "@media (max-width: 720px)" not in text:
    issues.append(ValidationIssue("responsive", "mobile or print rules missing"))
```

Use parser data rather than regex for element-level checks.

- [ ] **Step 6: 运行静态、Python 和 JavaScript 语法验证**

Run:

```powershell
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
@'
from pathlib import Path
import os
import re
text = Path("docs/soh-evaluation-system-design-and-user-guide.html").read_text(encoding="utf-8")
scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.S | re.I)
Path(os.environ["TEMP"], "soh-guide-inline.js").write_text("\n".join(scripts), encoding="utf-8")
'@ | .venv\Scripts\python.exe -
node --check "$env:TEMP\soh-guide-inline.js"
```

Expected: all commands exit `0`.

- [ ] **Step 7: 提交交互和响应式实现**

```powershell
git add docs/soh-evaluation-system-design-and-user-guide.html scripts/validate_design_user_guide.py tests/test_design_user_guide.py
git commit -m "docs: add responsive guide interactions"
```

---

### Task 6: 完成事实核验、浏览器验收和交付冻结

**Files:**
- Modify: `docs/soh-evaluation-system-design-and-user-guide.html`
- Modify: `tests/test_design_user_guide.py`

**Interfaces:**
- Consumes: current branch code, tests, `D:\battery_SOH\data`, current frontend artifacts and all approved source documents
- Produces: final standalone HTML with no unresolved findings

- [ ] **Step 1: 添加最终内容边界和禁用片段测试**

```python
def test_design_user_guide_has_no_unfinished_markers_or_document_metadata():
    import re

    text = GUIDE.read_text(encoding="utf-8")
    forbidden = (
        "TO" + "DO",
        "T" + "BD",
        "待" + "补充",
        "文档版本",
        "发布日期",
        "更新日期",
        "<script src=",
        "<link rel=\"stylesheet\"",
    )
    for fragment in forbidden:
        assert fragment not in text
    assert re.search(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", text) is None
    assert re.search(r"\b20\d{6}T\d{6}\b", text) is None


def test_design_user_guide_contains_source_boundary_and_acceptance_checklist():
    text = GUIDE.read_text(encoding="utf-8")
    assert 'id="source-boundary"' in text
    assert 'id="delivery-checklist"' in text
    assert "当前代码、测试和运行契约" in text
    assert "D:\\battery_SOH" in text
```

- [ ] **Step 2: 运行新增测试并修复所有内容边界问题**

Run:

```powershell
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
```

Expected: first run identifies any missing source-boundary/checklist content; after HTML correction, PASS.

- [ ] **Step 3: 逐项对照当前代码和数据做事实核验**

Verify and correct:

```powershell
rg -n 'path ==|path\.startswith' src\sohbench\api_server.py
rg -n 'data-side-mode|data-mode-panel' frontend\app\index.html
rg -n 'DEFAULT_OUTPUT|CURRENT\.json|generation|last-known-good|publish' scripts src\sohbench
Get-ChildItem 'D:\battery_SOH\data\processed' -Filter '*_features.pkl' | Select-Object -ExpandProperty Name
```

Then check each claim against:

- `docs/superpowers/specs/2026-07-14-soh-verification-fabric-redesign-design.md`
- `docs/superpowers/plans/2026-07-14-soh-verification-fabric-program.md`
- `docs/superpowers/plans/2026-07-14-soh-fabric-contract-evidence-foundation.md`
- `D:\battery_SOH\docs\design\soh-system-design.md`
- `D:\battery_SOH\docs\design\soh-algorithm-design.md`
- `D:\battery_SOH\docs\research\multi_dataset_soh_benchmark_landscape.md`

Do not turn the discovered data filenames into a permanent count claim. Record source names in `#source-boundary`, but keep filesystem paths as copyable text, not active links.

- [ ] **Step 4: 在 `file://` 下执行桌面和移动浏览器验收**

Open the absolute file URL:

```text
file:///D:/SOH_eval/.worktrees/verification-fabric-phase0/docs/soh-evaluation-system-design-and-user-guide.html
```

At 1280x720 and 390x844, use Chromium to assert:

```javascript
({
  viewport: [innerWidth, innerHeight],
  scrollWidth: document.documentElement.scrollWidth,
  clientWidth: document.documentElement.clientWidth,
  overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth,
  images: [...document.images].map((img) => ({ id: img.dataset.shot, complete: img.complete, width: img.naturalWidth })),
  replacementCharacters: document.body.innerText.includes("\uFFFD")
})
```

Expected:

- `overflow` is `false`.
- all five images are complete with nonzero natural width.
- no replacement character.
- no text or control overlaps its parent.
- desktop shows three-column workbench; mobile shows a usable collapsed navigation.

- [ ] **Step 5: 验收交互、无脚本降级、打印和零网络请求**

Using Chromium:

1. Search for `EvidencePack`, verify the matching chapter remains and clear restores all chapters.
2. Select manager/developer/operator paths, verify only nav emphasis changes.
3. Click every internal nav item and verify the target heading is visible.
4. Collapse and reopen a chapter.
5. Copy a PowerShell block, then test the selection fallback with clipboard permission denied.
6. Scroll and verify active nav plus right quick rail update.
7. Disable JavaScript and reload; verify all chapters, anchors and images remain readable.
8. Open print preview; verify navigation/control chrome is absent and diagrams are not clipped.
9. Capture browser network events; expected request count after navigation is exactly `0`.
10. 检查浏览器控制台，确认 console error count is `0`.

- [ ] **Step 6: 运行最终自动化验证**

Run:

```powershell
.venv\Scripts\python.exe scripts\validate_design_user_guide.py
.venv\Scripts\python.exe -m pytest tests/test_design_user_guide.py -q
.venv\Scripts\python.exe -m pytest -q
git diff --check
rg -n 'T[O]DO|T[B]D|待[补]充|文档版本|发布日期|更新日期' docs\soh-evaluation-system-design-and-user-guide.html
@'
from pathlib import Path
text = Path("docs/soh-evaluation-system-design-and-user-guide.html").read_text(encoding="utf-8")
markers = (chr(0xfffd), "\u00c3", "\u00c2", "\u951f\u65a4\u62f7")
assert not [marker for marker in markers if marker in text]
'@ | .venv\Scripts\python.exe -
```

Expected:

- validator exits `0`;
- focused and full pytest suites PASS;
- `git diff --check` exits `0`;
- final `rg` returns no matches and the Unicode marker scan exits `0`.

- [ ] **Step 7: 做最终自审并提交交付物**

Review the final HTML against every item in:

`docs/superpowers/specs/2026-07-17-soh-design-user-guide-html-design.md`

Then commit:

```powershell
git add docs/soh-evaluation-system-design-and-user-guide.html scripts/validate_design_user_guide.py scripts/embed_design_user_guide_screenshots.py tests/test_design_user_guide.py
git commit -m "docs: deliver standalone SOH design and user guide"
```

Expected: worktree is clean; final response links the HTML, reports focused/full test results and summarizes the two viewport plus zero-network browser evidence.
