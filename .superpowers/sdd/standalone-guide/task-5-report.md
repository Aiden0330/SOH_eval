# Task 5 实施报告：离线交互、响应式与打印体验

## 状态

- 结果：完成
- 基线：`179a3f4e2a080db64fcfe93c30061a9a14caaf7a`
- 分支：`codex/verification-fabric-phase0`
- 提交说明：`docs: add responsive guide interactions`
- 修改范围严格限制为用户指定的四个文件。

## RED / GREEN

### 主功能 RED

先扩展 `tests/test_design_user_guide.py`，覆盖搜索、阅读路径、复制回退、
quick rail、scroll-spy root、1100/720 断点、打印恢复、reduced motion、
icon-only button 名称负例、responsive 负例和乱码回归。

```powershell
D:\SOH_eval\.venv\Scripts\python.exe -m pytest `
  tests\test_design_user_guide.py -q
```

结果：`9 failed, 24 passed`。失败来自缺少交互控件/函数、13 章
`data-paths`/`.quick-source`、打印恢复规则，以及 validator 尚未检查
button name 和 responsive rules。

第一次按 brief 原样输入的 `.venv\Scripts\python.exe` 被 PowerShell 当作模块名，
没有进入 pytest，不计为 RED；随后改用已确认存在的绝对解释器路径重跑。

### Scroll-spy 回归 RED

Chrome 初次烟测暴露出基于缓存 `intersectionRatio` 的选章策略不适合高度差异很大的
章节。先增加 root 相对几何选章契约：

```powershell
D:\SOH_eval\.venv\Scripts\python.exe -m pytest `
  tests\test_design_user_guide.py::test_scroll_spy_selects_by_root_relative_geometry_not_cached_ratios -q
```

结果：`1 failed`，原因是 `syncActiveSection` 尚不存在。

后续排查还确认，初次 450ms 浏览器断言发生在平滑滚动途中；最终重测改用关闭平滑
滚动的确定性跳转，不再用固定短等待制造假阴性。

### GREEN

最终 focused tests：

```text
35 passed
```

## 实现

- 新增自包含全文搜索、清除、无结果 `role="status"`；清除恢复 13 章，过滤后点击
  目录锚点会先清除搜索再跳转，打印强制恢复 `.chapter[hidden]`。
- 新增 `all/manager/developer/operator` 阅读路径；只更新按钮
  `aria-pressed` 与目录 `on-path`，正文不隐藏。13 章均有合理 `data-paths`。
- 复制采用 Clipboard API、`document.execCommand("copy")`、代码文本选择三级回退。
  仅选中时通过 `aria-live="polite"` 明确提示“已选中，请按 Ctrl+C”，不虚报成功。
- 每章新增 `.quick-source`，右栏使用 `#quick-content`；1100px 以下隐藏右栏并在章内
  显示速查。
- scroll-spy 在桌面以独立滚动的 `main` 为 root，移动端以 viewport 为 root；
  720px media query 变化时断开并重建 observer。活动章按 root 内标记线和章节几何
  位置决定，目录与 quick rail 共用同一激活函数。
- 移动端无 JS 使用默认展开的原生 `<details>` 目录；JS 模式使用 sticky
  `#nav-toggle`，带文字、熟悉符号和 `aria-expanded`。返回顶部为有
  `aria-label`、`title`、`data-icon-only` 的 icon-only button。
- CSS 使用 `240px minmax(0, 1fr) 260px` 三栏、1100px 两栏、720px 单栏；
  table/pre 只局部滚动或安全换行，无梯度、orb 或 viewport 字号。
- reduced-motion 禁用平滑滚动、transition 和 animation。
- 打印隐藏目录、右栏、搜索、阅读路径、复制和返回顶部；恢复过滤章节与关闭的
  details 正文，避免 figure/table/pre/diagram 截断并重复 thead。
- validator 通过 `HTMLParser` 收集 button 属性和内联 style 文本，检查 icon-only
  `aria-label` 及 1100px、720px、print 规则。

## 浏览器证据

内置浏览器插件初始化时遇到本机路径错误，因此改用已安装的 Chrome headless 和
DevTools Protocol；未安装依赖、未启动网络服务，直接打开最终 `file://` URL。

- 1280x720：`scrollWidth == clientWidth == 1265`，三栏为
  `240px 765px 260px`，`main` 独立滚动，右栏可见。
- 390x844：`scrollWidth == clientWidth == 390`，右栏隐藏，章内 quick source
  可见，nav toggle 从 `aria-expanded=false` 切到 `true`。
- 搜索 `EvidencePack` 后命中 5 章；清除后恢复 13 章。无结果过滤后点击
  `#architecture`，搜索值清空、13 章恢复、hash 正确更新。
- 选择 developer 后隐藏正文数为 0，按钮 `aria-pressed=true`。
- 禁用 Clipboard API 且令 `execCommand` 返回 false 后，按钮显示“已选中”，
  live region 显示“已选中，请按 Ctrl+C”，实际 selection 长度为 96。
- 确定性滚到 `#evidence` 后，目录 active 为 `#evidence`，quick rail 同步显示
  EvidencePack、comparison key、verdict 与 0/0 风险。
- 两个视口五张图片均 `complete=true` 且 natural width 非零；控制台错误 0，
  外部 HTTP(S) 请求 0，正文无 U+FFFD。
- print media 下 nav/rail/search/audience/copy/back-to-top 均为 `display:none`；
  hidden chapter 和关闭 details 的正文为 `display:block`，thead 为
  `table-header-group`。
- reduced-motion 下 html/main 的 `scroll-behavior` 为 `auto`，按钮 transition
  时长为 0。
- 禁用 JavaScript 后仍有 13 个完整可见章节、13 个目录锚点和五张已加载截图；
  390px 无页面级横溢出。

## 截图 Payload 保全

五个既有 payload 的解码字节数与 SHA-256 在修改前后完全一致：

| shot | bytes | SHA-256 |
|---|---:|---|
| training | 445,953 | `29c9390192c6e10afbbefe7693dd76842e62b95f81bf87a3b3e6b9fb17909329` |
| inference | 497,399 | `cdeec39a1195a9bcc900f5addc8652f95fd165d80a9175838eea25e062dc120f` |
| models | 406,876 | `3e2643919718896cf815841fcfe2f8f8437ceb2fa4496293ecffb956b88030f0` |
| native | 366,759 | `4a2e268d75be07b6f8e8fff8c43faecbc632945123f64ebe8fbf498e9d8f5333` |
| diagnostics | 278,429 | `1400e71b599b3c1ce02d8246b09e9e3817248e5351e00b9caec2d7cf1c09e9cb` |

Task 4 的 Native 修正和全部正文事实未被改写。

## 验证

- validator：退出码 0，无问题输出。
- focused pytest：`35 passed`。
- 提取全部 inline JS 后 `node --check`：退出码 0。
- `git diff --check`：退出码 0；仅显示工作树既有的 LF/CRLF 提示。
- 浏览器临时 profile 已在同一 PowerShell 流程中校验位于系统临时目录后删除；
  本次没有留下监听端口或浏览器服务。

## 变更文件

- `docs/soh-evaluation-system-design-and-user-guide.html`
- `scripts/validate_design_user_guide.py`
- `tests/test_design_user_guide.py`
- `.superpowers/sdd/standalone-guide/task-5-report.md`

---

## 独立审查修正

### 修正基线与范围

- 独立审查结论：NOT APPROVED
- 修正基线：`5d658729b47d745418d5dbebdb22c4cc95cad9a8`
- 修正提交说明：`fix: harden guide interaction contracts`
- 仍只修改 Task 5 的四个所有权文件，未新增产品运行时文件或依赖。

### Remediation RED

先新增以下失败契约：

- 真实 Chrome 在搜索 `EvidencePack` 后点击仍可见的 `#evidence`，必须清空搜索、
  恢复 13 章并更新 hash。
- 同一 Chrome 驱动令 Clipboard API reject 且 `execCommand` 返回 false，只能报告
  “已选中”及“已选中，请按 Ctrl+C”。
- 五个 screenshot image 必须有精确固有尺寸，并保持既有 payload SHA-256。
- 没有 `data-icon-only`、只有 `aria-hidden=true` 符号的 button 仍必须报告
  `button-name`。
- 注释中的三个 media token 不能满足 responsive validator。
- 相对 `url("asset.png")` 和相对、绝对、HTTP 三种 `@import` 必须报告
  `external-asset`。

RED 命令：

```powershell
D:\SOH_eval\.venv\Scripts\python.exe -m pytest `
  tests\test_design_user_guide.py -q `
  -k "headless_chrome or embeds_all_five or unmarked_symbol or commented_responsive or relative_css_url or css_imports"
```

RED 结果：`8 failed`。Chrome 实际返回 `queryAfter: "EvidencePack"`；图片缺少
`width`；其余失败分别证明 button text、CSS comment、relative URL 和
`@import` 漏报。

第一次 GREEN 重跑为 `7 passed, 1 failed`：按钮已进入“已选中”，但
`--dump-dom` 的无绘制路径不执行用于更新 live region 的
`requestAnimationFrame`。生产公告改为下一事件循环的 `setTimeout(0)`，保留
清空后重新播报语义，并让后台/无绘制页面可靠更新。随后真实 Chrome 用例通过。

### 交互修正

目录 click 不再检查目标章是否 hidden。只要受保护的
`search && search.value.trim()` 为真，就同步执行：

1. 清空 search value。
2. 调用 `filterGuide("")` 恢复全部 13 章。
3. 让链接默认行为完成锚点跳转。
4. 收起移动目录。

pytest 内的 authoring helper 只使用标准库和临时目录：

- 按常见安装路径和 `PATH` 查找 Chrome/Edge。
- 找不到浏览器时以明确原因 `pytest.skip`。
- 找到后使用 `--headless=new --dump-dom --virtual-time-budget=2000`。
- 向临时 HTML 注入驱动，不修改产品 HTML 或截图 payload。

### 图片固有尺寸

五张 `<img data-shot>` 新增：

| shot | width | height |
|---|---:|---:|
| training | 1440 | 800 |
| inference | 1440 | 800 |
| models | 1440 | 800 |
| native | 1110 | 800 |
| diagnostics | 1440 | 800 |

CSS 继续使用 `.product-shot img { width: 100%; height: auto; }`。测试同时精确检查
属性、解码 payload 和 SHA-256；五个哈希均与修正前一致。

### Validator 加固

- `GuideParser` 为每个 button 收集属性和可见文本。
- `aria-hidden=true` 子树不贡献 accessible text。
- 所有 button 都必须有非空 `aria-label` 或非空可见文本，不再依赖
  `data-icon-only` 标记。
- 合并 inline style 后先移除 `/* ... */`，再用实际 media 规则匹配 1100px、
  720px 和 print。
- 任意 `@import` 均拒绝。
- 任意非 `data:` 的 `url(...)` 均拒绝；相对、绝对与 HTTP(S) CSS 资源统一报告
  `external-asset`。
- 原有 external stylesheet/script、网络、mojibake、图片和 API parity 检查保留。

### Remediation GREEN 与浏览器范围

最终 focused pytest：

```text
42 passed
```

其中 pytest 自动化的真实浏览器范围是：

- 搜索结果中目标章本来可见的目录点击恢复行为。
- Clipboard API 与 `execCommand` 两级失败后的选择和 aria-live 文案。

另以无依赖 Chrome DevTools Protocol 执行一次性 authoring smoke：

- 1280x720：`240px 765px 260px` 三栏、main 独立滚动、无页面横溢出。
- 390x844：右栏隐藏、章内 quick source 可见、sticky toggle 正常、无页面横溢出。
- `#evidence` click 后 query 为空、13 章可见、hash/active 均为 `#evidence`。
- 五图 declared size 与 natural size 分别为 1440x800 或 1110x800，全部加载完成。
- print：nav、rail、search、audience、copy、back-to-top 隐藏；过滤章节和关闭
  details 正文恢复；thead 为 `table-header-group`。
- reduced motion：html/main 为 `scroll-behavior: auto`，transition 为 0。
- JavaScript disabled：13 章可见，原生 mobile fallback 打开，五图加载，390px
  无横溢出。
- console error 为 0，HTTP(S) request 为 0。

print、reduced-motion、JS-disabled 和两视口 computed-style 检查仍是一次性
authoring smoke，不声明为 pytest 持久回归；Task 6 可继续做人工视觉/打印预览验收。
