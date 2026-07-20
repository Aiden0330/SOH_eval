import ast
import base64
from collections import Counter
import hashlib
import html
import json
import re
import shutil
import subprocess
from html.parser import HTMLParser
from pathlib import Path

import pytest

from sohbench.api_server import parse_args
from scripts.validate_design_user_guide import REQUIRED_SECTIONS, validate_document


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html"
TARGET_OUTCOMES = [
    "complete",
    "partial",
    "failed",
    "timed_out",
    "cancelled",
    "invalid",
]
TARGET_VERDICTS = [
    "pass",
    "warn",
    "fail",
    "not_applicable",
    "not_reviewed",
]


def _normalize_visible_text(parts):
    return " ".join(part.strip() for part in parts if part.strip())


class DiagramFallbackParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack = []
        self.role_img_depth = 0
        self.diagram_refs = []
        self.fallback_ids = []
        self.nested_fallbacks = []
        self.adjacency_errors = []
        self.pending_ref = None

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set(values.get("class", "").split())
        is_fallback = tag == "ol" and "diagram-fallback" in classes
        if self.pending_ref is not None:
            fallback_id = values.get("id") if is_fallback else None
            if fallback_id != self.pending_ref:
                self.adjacency_errors.append((self.pending_ref, tag, fallback_id))
            self.pending_ref = None

        is_role_img = values.get("role") == "img"
        if is_role_img:
            self.role_img_depth += 1
            self.diagram_refs.append(values.get("aria-describedby", ""))
        if is_fallback:
            fallback_id = values.get("id", "")
            self.fallback_ids.append(fallback_id)
            if self.role_img_depth:
                self.nested_fallbacks.append(fallback_id)
        self.stack.append((tag, is_role_img, values.get("aria-describedby", "")))

    def handle_endtag(self, tag):
        while self.stack:
            open_tag, is_role_img, described_by = self.stack.pop()
            if is_role_img:
                self.role_img_depth -= 1
                self.pending_ref = described_by
            if open_tag == tag:
                break


class ModeGuideParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.current_mode = None
        self.article_depth = 0
        self.in_heading = False
        self.heading_parts = []
        self.headings = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set(values.get("class", "").split())
        if tag == "article" and "mode-guide" in classes:
            self.current_mode = values.get("data-guide-mode")
            self.article_depth = 1
            self.headings.setdefault(self.current_mode, [])
            return
        if self.article_depth:
            self.article_depth += 1
            if tag == "h3":
                self.in_heading = True
                self.heading_parts = []

    def handle_data(self, data):
        if self.in_heading:
            self.heading_parts.append(data)

    def handle_endtag(self, tag):
        if not self.article_depth:
            return
        if tag == "h3" and self.in_heading:
            self.headings[self.current_mode].append(
                "".join(self.heading_parts).strip()
            )
            self.in_heading = False
            self.heading_parts = []
        self.article_depth -= 1
        if self.article_depth == 0:
            self.current_mode = None


class ApiRouteTableParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.in_route_container = False
        self.in_table = False
        self.cell_index = -1
        self.in_route_code = False
        self.route_parts = []
        self.routes = set()

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set(values.get("class", "").split())
        if tag == "div" and "route-table" in classes:
            self.in_route_container = True
        elif tag == "table" and self.in_route_container:
            self.in_table = True
        elif self.in_table and tag == "tr":
            self.cell_index = -1
        elif self.in_table and tag == "td":
            self.cell_index += 1
        elif self.in_table and tag == "code" and self.cell_index == 0:
            self.in_route_code = True
            self.route_parts = []

    def handle_data(self, data):
        if self.in_route_code:
            self.route_parts.append(data)

    def handle_endtag(self, tag):
        if tag == "code" and self.in_route_code:
            method, route = "".join(self.route_parts).strip().split(maxsplit=1)
            self.routes.add((method, route))
            self.in_route_code = False
            self.route_parts = []
        elif tag == "table" and self.in_table:
            self.in_table = False
        elif tag == "div" and self.in_route_container:
            self.in_route_container = False


class SectionTextParser(HTMLParser):
    def __init__(self, section_id):
        super().__init__(convert_charrefs=True)
        self.section_id = section_id
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if not self.depth and tag == "section" and values.get("id") == self.section_id:
            self.depth = 1
        elif self.depth:
            self.depth += 1

    def handle_data(self, data):
        if self.depth:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.depth:
            self.depth -= 1

    def text(self):
        return " ".join("".join(self.parts).split())


class SectionCodeBlockParser(HTMLParser):
    def __init__(self, section_id):
        super().__init__(convert_charrefs=True)
        self.section_id = section_id
        self.depth = 0
        self.in_pre = False
        self.in_code = False
        self.parts = []
        self.blocks = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if not self.depth and tag == "section" and values.get("id") == self.section_id:
            self.depth = 1
            return
        if not self.depth:
            return
        self.depth += 1
        if tag == "pre":
            self.in_pre = True
        elif tag == "code" and self.in_pre:
            self.in_code = True
            self.parts = []

    def handle_data(self, data):
        if self.in_code:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if not self.depth:
            return
        if tag == "code" and self.in_code:
            self.blocks.append("".join(self.parts).strip())
            self.in_code = False
            self.parts = []
        elif tag == "pre":
            self.in_pre = False
        self.depth -= 1


class StructuredStateLaneParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.current_lane = None
        self.lane_depth = 0
        self.lanes = {}
        self.visible_nodes = {}
        self.current_node_value = None
        self.node_depth = 0
        self.node_parts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        lane = values.get("data-state-lane")
        if lane is not None:
            assert self.current_lane is None, "structured state lanes cannot nest"
            self.current_lane = lane
            self.lane_depth = 1
            self.lanes.setdefault(lane, [])
            self.visible_nodes.setdefault(lane, [])
            return
        if self.current_lane is None:
            return
        self.lane_depth += 1
        if self.current_node_value is not None:
            self.node_depth += 1
            return
        classes = set(values.get("class", "").split())
        if "state-node" in classes:
            self.current_node_value = values.get("data-state-value")
            self.node_depth = 1
            self.node_parts = []
            self.lanes[self.current_lane].append(self.current_node_value)

    def handle_data(self, data):
        if self.current_node_value is not None:
            self.node_parts.append(data)

    def handle_endtag(self, tag):
        if self.current_lane is None:
            return
        if self.current_node_value is not None:
            self.node_depth -= 1
            if self.node_depth == 0:
                self.visible_nodes[self.current_lane].append(
                    (
                        self.current_node_value,
                        _normalize_visible_text(self.node_parts),
                    )
                )
                self.current_node_value = None
                self.node_parts = []
        self.lane_depth -= 1
        if self.lane_depth == 0:
            self.current_lane = None


class StructuredFactParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.datasets = {}
        self.dataset_texts = {}
        self.featuresets = {}
        self.featureset_texts = {}
        self.metric_truth_boundaries = []
        self.current_dataset = None
        self.dataset_depth = 0
        self.dataset_parts = []
        self.current_featureset = None
        self.featureset_depth = 0
        self.featureset_parts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        dataset = values.get("data-dataset-id")
        if dataset is not None:
            self.datasets[dataset] = values.get("data-integration-status")
        if self.current_dataset is not None:
            self.dataset_depth += 1
        elif dataset is not None:
            self.current_dataset = dataset
            self.dataset_depth = 1
            self.dataset_parts = []

        featureset = values.get("data-featureset")
        if featureset is not None:
            self.featuresets[featureset] = {
                key: value
                for key, value in values.items()
                if key.startswith("data-")
            }
        if self.current_featureset is not None:
            self.featureset_depth += 1
        elif featureset is not None:
            self.current_featureset = featureset
            self.featureset_depth = 1
            self.featureset_parts = []

        if "data-metric-truth-current" in values:
            self.metric_truth_boundaries.append(
                (
                    values.get("data-metric-truth-current"),
                    values.get("data-metric-truth-target"),
                )
            )

    def handle_data(self, data):
        if self.current_dataset is not None:
            self.dataset_parts.append(data)
        if self.current_featureset is not None:
            self.featureset_parts.append(data)

    def handle_endtag(self, tag):
        if self.current_dataset is not None:
            self.dataset_depth -= 1
            if self.dataset_depth == 0:
                self.dataset_texts[self.current_dataset] = _normalize_visible_text(
                    self.dataset_parts
                )
                self.current_dataset = None
                self.dataset_parts = []

        if self.current_featureset is not None:
            self.featureset_depth -= 1
            if self.featureset_depth == 0:
                self.featureset_texts[
                    self.current_featureset
                ] = _normalize_visible_text(self.featureset_parts)
                self.current_featureset = None
                self.featureset_parts = []


def _guide_section_text(section_id):
    parser = SectionTextParser(section_id)
    parser.feed(GUIDE.read_text(encoding="utf-8"))
    return parser.text()


def _guide_section_code_blocks(section_id):
    parser = SectionCodeBlockParser(section_id)
    parser.feed(GUIDE.read_text(encoding="utf-8"))
    return parser.blocks


def _guide_without_embedded_images():
    text = GUIDE.read_text(encoding="utf-8")
    return re.sub(
        r'data:image/[^;"]+;base64,[^"]+',
        "data:image/embedded;base64,",
        text,
    )


def _assert_structured_fact_contract(source):
    state_parser = StructuredStateLaneParser()
    state_parser.feed(source)
    assert set(state_parser.lanes) == {"target-outcome", "target-verdict"}
    assert state_parser.lanes["target-outcome"] == TARGET_OUTCOMES
    assert state_parser.lanes["target-verdict"] == TARGET_VERDICTS
    assert state_parser.visible_nodes == {
        "target-outcome": [(value, value) for value in TARGET_OUTCOMES],
        "target-verdict": [(value, value) for value in TARGET_VERDICTS],
    }

    fact_parser = StructuredFactParser()
    fact_parser.feed(source)
    assert fact_parser.datasets == {
        "uconn-isu-ilcc-lfp-gr": "integrated",
        "isu-ilcc-figshare-nmc-gr": "research-only-not-integrated",
    }
    assert fact_parser.dataset_texts == {
        "uconn-isu-ilcc-lfp-gr": "当前已接入的是 UConn-ISU-ILCC LFP/Gr 数据",
        "isu-ilcc-figshare-nmc-gr": (
            "ISU-ILCC Figshare NMC/Gr 只出现在研究资料中，当前未接入"
        ),
    }
    assert fact_parser.featuresets["capacity_time"][
        "data-feature-policy-status"
    ] == "open-gap"
    assert fact_parser.featuresets["capacity_time"][
        "data-current-risk-feature"
    ] == "cycle_fraction"
    capacity_time_text = fact_parser.featureset_texts["capacity_time"]
    assert "当前还包含 cycle_fraction" in capacity_time_text
    assert "cycle_fraction 是待关闭的 FeaturePolicy gap" in capacity_time_text
    assert (
        "在关闭前不能宣称该方案已经全面满足未来信息隔离"
        in capacity_time_text
    )
    assert fact_parser.featuresets["ic_dva"]["data-current-columns"].split() == [
        "ic_peak_ah_per_v",
        "ic_peak_voltage_v",
    ]
    assert fact_parser.metric_truth_boundaries == [
        ("in-process", "exclusive-metric-worker")
    ]


def _replace_once(source, old, new):
    assert source.count(old) == 1
    return source.replace(old, new, 1)


def _replace_element_text(source, marker, replacement):
    pattern = re.compile(
        rf'(<span\b[^>]*{re.escape(marker)}[^>]*>).*?(</span>)',
        flags=re.DOTALL,
    )
    mutated, count = pattern.subn(rf"\1{replacement}\2", source, count=1)
    assert count == 1
    return mutated


def _source_method_routes():
    source = (ROOT / "src" / "sohbench" / "api_server.py").read_text(
        encoding="utf-8"
    )
    tree = ast.parse(source)
    routes = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name not in {
            "do_GET",
            "do_POST",
        }:
            continue
        method = node.name.removeprefix("do_")
        for candidate in ast.walk(node):
            if (
                isinstance(candidate, ast.Compare)
                and isinstance(candidate.left, ast.Name)
                and candidate.left.id == "path"
                and len(candidate.ops) == 1
                and isinstance(candidate.ops[0], ast.Eq)
                and len(candidate.comparators) == 1
                and isinstance(candidate.comparators[0], ast.Constant)
                and isinstance(candidate.comparators[0].value, str)
                and candidate.comparators[0].value.startswith("/api/")
            ):
                routes.add((method, candidate.comparators[0].value))
            if (
                isinstance(candidate, ast.Call)
                and isinstance(candidate.func, ast.Attribute)
                and isinstance(candidate.func.value, ast.Name)
                and candidate.func.value.id == "path"
                and candidate.func.attr == "startswith"
                and len(candidate.args) == 1
                and isinstance(candidate.args[0], ast.Constant)
                and isinstance(candidate.args[0].value, str)
                and candidate.args[0].value.startswith("/api/")
            ):
                routes.add(
                    (
                        method,
                        f"{candidate.args[0].value.rstrip('/')}/{{job_id}}",
                    )
                )
    return routes


def test_design_user_guide_satisfies_offline_structure_contract():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert issues == []


def test_design_user_guide_contains_every_required_chapter():
    text = GUIDE.read_text(encoding="utf-8")
    for section_id in REQUIRED_SECTIONS:
        assert f'id="{section_id}"' in text


def _inline_guide_script():
    text = GUIDE.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", text, flags=re.DOTALL | re.IGNORECASE)
    return "\n".join(scripts)


def _guide_function_source(name):
    script = _inline_guide_script()
    start = script.index(f"function {name}(")
    next_function = re.search(r"\n\s+function \w+\(", script[start + 1 :])
    if next_function is None:
        return script[start:]
    end = start + 1 + next_function.start()
    return script[start:end]


def _find_chrome():
    candidates = (
        shutil.which("chrome"),
        shutil.which("chrome.exe"),
        shutil.which("msedge"),
        shutil.which("msedge.exe"),
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    )
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    return None


def _run_chrome_guide_driver(tmp_path, driver):
    chrome = _find_chrome()
    if chrome is None:
        pytest.skip("Chrome/Edge not found; headless guide behavior smoke skipped")

    driven_guide = tmp_path / "guide-smoke.html"
    source = GUIDE.read_text(encoding="utf-8")
    driven_guide.write_text(
        source.replace("</body>", f"<script>{driver}</script></body>"),
        encoding="utf-8",
    )
    profile = tmp_path / "chrome-profile"
    completed = subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            "--window-size=1280,720",
            "--virtual-time-budget=2000",
            f"--user-data-dir={profile}",
            "--dump-dom",
            driven_guide.as_uri(),
        ],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=45,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    match = re.search(
        r'<pre id="smoke-result">(.*?)</pre>',
        completed.stdout,
        flags=re.DOTALL,
    )
    assert match is not None, completed.stderr or completed.stdout[-1000:]
    return json.loads(html.unescape(match.group(1)))


def test_headless_chrome_clears_nonempty_search_before_visible_anchor_jump(
    tmp_path,
):
    result = _run_chrome_guide_driver(
        tmp_path,
        r"""
        (async () => {
          const search = document.querySelector("#guide-search");
          search.value = "EvidencePack";
          search.dispatchEvent(new Event("input", { bubbles: true }));
          const target = document.querySelector("#evidence");
          const targetWasVisible = !target.hidden;
          document.querySelector('.guide-nav a[href="#evidence"]').click();
          await new Promise((resolve) => window.setTimeout(resolve, 50));

          try {
            Object.defineProperty(navigator, "clipboard", {
              configurable: true,
              value: {
                writeText: () => Promise.reject(new Error("denied")),
              },
            });
          } catch {}
          document.execCommand = () => false;
          document.querySelector(".copy-button").click();
          await new Promise((resolve) => {
            const started = performance.now();
            const checkStatus = () => {
              if (
                document.querySelector("#copy-status").textContent
                || performance.now() - started > 500
              ) {
                resolve();
              } else {
                window.setTimeout(checkStatus, 20);
              }
            };
            checkStatus();
          });

          const payload = {
            targetWasVisible,
            queryAfter: search.value,
            visibleAfter: [...document.querySelectorAll(".chapter")]
              .filter((chapter) => !chapter.hidden).length,
            hashAfter: location.hash,
            copyButton: document.querySelector(".copy-button").textContent,
            copyStatus: document.querySelector("#copy-status").textContent,
            selectionLength: String(getSelection()).length,
          };
          const output = document.createElement("pre");
          output.id = "smoke-result";
          output.textContent = JSON.stringify(payload);
          document.body.replaceChildren(output);
        })();
        """,
    )

    assert result["targetWasVisible"] is True
    assert result["queryAfter"] == ""
    assert result["visibleAfter"] == len(REQUIRED_SECTIONS)
    assert result["hashAfter"] == "#evidence"
    assert result["copyButton"] == "已选中"
    assert result["copyStatus"] == "已选中，请按 Ctrl+C"
    assert result["selectionLength"] > 0


def test_design_user_guide_has_progressive_interactions_and_print_rules():
    text = GUIDE.read_text(encoding="utf-8")
    for token in (
        'id="guide-search"',
        'id="clear-search"',
        'id="search-empty" role="status"',
        'id="copy-status"',
        'id="nav-toggle"',
        'id="quick-content"',
        'data-audience="all"',
        'data-audience="manager"',
        'data-audience="developer"',
        'data-audience="operator"',
        'data-icon-only="true"',
        "function setAudience(",
        "function filterGuide(",
        "function copyCode(",
        "function updateQuickRail(",
        "function rebuildScrollSpy(",
        "IntersectionObserver",
        "@media (max-width: 1100px)",
        "@media (max-width: 720px)",
        "@media (prefers-reduced-motion: reduce)",
        "@media print",
    ):
        assert token in text


def test_every_chapter_has_audience_paths_and_inline_quick_source():
    text = GUIDE.read_text(encoding="utf-8")
    chapter_tags = re.findall(
        r'<section id="[^"]+" class="chapter" data-paths="[^"]+">',
        text,
    )
    assert len(chapter_tags) == len(REQUIRED_SECTIONS)
    assert text.count('class="quick-source"') == len(REQUIRED_SECTIONS)


def test_reading_paths_only_change_navigation_emphasis():
    source = _guide_function_source("setAudience")
    assert 'setAttribute("aria-pressed"' in source
    assert 'classList.toggle("on-path"' in source
    for forbidden in (".hidden", ".style", "display"):
        assert forbidden not in source


def test_search_restores_chapters_before_filtered_anchor_navigation():
    text = GUIDE.read_text(encoding="utf-8")
    source = _guide_function_source("filterGuide")
    assert "chapter.hidden = !match" in source
    assert "filterGuide(\"\")" in text
    assert "search.value.trim()" in text
    assert "chapters.length" in text


def test_copy_fallback_never_claims_success_for_selection_only():
    source = _guide_function_source("copyCode")
    assert "navigator.clipboard" in source
    assert 'document.execCommand("copy")' in source
    assert "已选中，请按 Ctrl+C" in source
    assert "已复制" in source
    assert source.index("已选中，请按 Ctrl+C") != source.index("已复制")


def test_scroll_spy_rebuilds_with_the_correct_scroll_root():
    source = _guide_function_source("rebuildScrollSpy")
    assert "root: mobileQuery.matches ? null : main" in source
    assert "chapterObserver.disconnect()" in source
    assert "mobileQuery.addEventListener" in _inline_guide_script()


def test_scroll_spy_selects_by_root_relative_geometry_not_cached_ratios():
    source = _guide_function_source("syncActiveSection")
    assert "getBoundingClientRect()" in source
    assert "markerLine" in source
    assert "intersectionRatio" not in _inline_guide_script()


def test_print_restores_filtered_content_and_reduced_motion_disables_motion():
    text = GUIDE.read_text(encoding="utf-8")
    print_css = text[text.index("@media print") :]
    assert ".chapter[hidden]" in print_css
    assert "display: block !important" in print_css
    assert re.search(
        r"\.guide-shell\s*\{[^}]*display:\s*block;"
        r"[^}]*height:\s*auto;[^}]*overflow:\s*visible;",
        print_css,
    )
    assert "thead" in print_css
    assert "table-header-group" in print_css

    reduced_css = text[text.index("@media (prefers-reduced-motion: reduce)") :]
    assert "scroll-behavior: auto !important" in reduced_css
    assert "transition: none !important" in reduced_css


def test_all_buttons_have_accessible_names():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert not [issue for issue in issues if issue.code == "button-name"]


def test_validator_rejects_icon_only_button_without_aria_label(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</body>",
            '<button type="button" data-icon-only="true"></button></body>',
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "button-name" for issue in issues)


def test_validator_rejects_unmarked_symbol_only_button_without_name(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</body>",
            '<button type="button"><span aria-hidden="true">↑</span></button>'
            "</body>",
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "button-name" for issue in issues)


def test_validator_rejects_missing_mobile_or_print_rules(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8")
        .replace("@media (max-width: 720px)", "@media screen")
        .replace("@media print", "@media screen"),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "responsive" for issue in issues)


def test_validator_ignores_commented_responsive_tokens(tmp_path):
    guide = tmp_path / "guide.html"
    text = (
        GUIDE.read_text(encoding="utf-8")
        .replace("@media (max-width: 1100px)", "@media screen", 1)
        .replace("@media (max-width: 720px)", "@media screen", 1)
        .replace("@media print", "@media screen", 1)
        .replace(
            "</style>",
            "/* @media (max-width: 1100px) {} "
            "@media (max-width: 720px) {} @media print {} */</style>",
            1,
        )
    )
    guide.write_text(text, encoding="utf-8")

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "responsive" for issue in issues)


def test_validator_rejects_relative_css_url(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</style>",
            '.external { background-image: url("asset.png"); }</style>',
            1,
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "external-asset" for issue in issues)


@pytest.mark.parametrize(
    "rule",
    (
        '@import "theme.css";',
        '@import "/assets/theme.css";',
        '@import url("https://example.invalid/theme.css");',
    ),
)
def test_validator_rejects_css_imports(tmp_path, rule):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</style>",
            f"{rule}</style>",
            1,
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "external-asset" for issue in issues)


def test_validator_rejects_known_mojibake_marker(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace("</body>", "锟斤拷</body>"),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "mojibake" for issue in issues)


def test_embed_screenshots_replaces_each_named_capture(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    html.write_text(
        '<img data-shot="training" src="data:image/png;base64,AA==" alt="训练页">',
        encoding="utf-8",
    )
    capture = tmp_path / "training.png"
    capture.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    embed_screenshots(html, {"training": capture})

    rendered = html.read_text(encoding="utf-8")
    expected = base64.b64encode(capture.read_bytes()).decode("ascii")
    assert f"data:image/png;base64,{expected}" in rendered


def test_embed_screenshots_rejects_missing_marker_without_changing_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    original = '<img data-shot="models" src="data:image/png;base64,AA==" alt="模型页">'
    html.write_text(original, encoding="utf-8")
    capture = tmp_path / "training.png"
    capture.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    with pytest.raises(
        ValueError,
        match="expected one image marker for training, found 0",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_rejects_duplicate_marker_without_changing_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    marker = '<img data-shot="training" src="data:image/png;base64,AA==" alt="训练页">'
    original = marker + marker
    html.write_text(original, encoding="utf-8")
    capture = tmp_path / "training.png"
    capture.write_bytes(b"\x89PNG\r\n\x1a\nfixture")

    with pytest.raises(
        ValueError,
        match="expected one image marker for training, found 2",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_late_marker_failure_leaves_original_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    original = '<img data-shot="training" src="old" alt="训练页">'
    html.write_text(original, encoding="utf-8")
    training = tmp_path / "training.png"
    inference = tmp_path / "inference.png"
    training.write_bytes(b"\x89PNG\r\n\x1a\ntraining")
    inference.write_bytes(b"\x89PNG\r\n\x1a\ninference")

    with pytest.raises(
        ValueError,
        match="expected one image marker for inference, found 0",
    ):
        embed_screenshots(
            html,
            {"training": training, "inference": inference},
        )

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_rejects_unsupported_suffix_without_changing_html(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    original = '<img data-shot="training" src="old" alt="训练页">'
    html.write_text(original, encoding="utf-8")
    capture = tmp_path / "training.bmp"
    capture.write_bytes(b"bitmap fixture")

    with pytest.raises(ValueError, match="unsupported image type: .bmp"):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_uses_webp_mime_type(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = tmp_path / "guide.html"
    html.write_text(
        '<img data-shot="native" src="old" alt="完整看板">',
        encoding="utf-8",
    )
    capture = tmp_path / "native.webp"
    capture.write_bytes(b"RIFFfixtureWEBP")

    embed_screenshots(html, {"native": capture})

    expected = base64.b64encode(capture.read_bytes()).decode("ascii")
    assert (
        f"src=\"data:image/webp;base64,{expected}\""
        in html.read_text(encoding="utf-8")
    )


def test_design_user_guide_embeds_all_five_frontend_captures():
    text = GUIDE.read_text(encoding="utf-8")
    expected = {
        "training": (
            (1440, 800),
            "29c9390192c6e10afbbefe7693dd76842e62b95f81bf87a3b3e6b9fb17909329",
        ),
        "inference": (
            (1440, 800),
            "cdeec39a1195a9bcc900f5addc8652f95fd165d80a9175838eea25e062dc120f",
        ),
        "models": (
            (1440, 800),
            "3e2643919718896cf815841fcfe2f8f8437ceb2fa4496293ecffb956b88030f0",
        ),
        "native": (
            (1110, 800),
            "4a2e268d75be07b6f8e8fff8c43faecbc632945123f64ebe8fbf498e9d8f5333",
        ),
        "diagnostics": (
            (1440, 800),
            "1400e71b599b3c1ce02d8246b09e9e3817248e5351e00b9caec2d7cf1c09e9cb",
        ),
    }
    for shot, (dimensions, expected_hash) in expected.items():
        assert text.count(f'data-shot="{shot}"') == 1
        figures = [
            body
            for body in re.findall(
                r'<figure class="product-shot">(.*?)</figure>',
                text,
                flags=re.DOTALL,
            )
            if f'data-shot="{shot}"' in body
        ]
        assert len(figures) == 1
        structure = re.fullmatch(
            r'\s*<img (?P<attrs>[^>]*)>\s*'
            r'<figcaption>(?P<caption>.*?)</figcaption>\s*',
            figures[0],
            flags=re.DOTALL,
        )
        assert structure is not None
        attrs = dict(
            re.findall(r'([\w-]+)="([^"]*)"', structure.group("attrs"))
        )
        assert attrs["data-shot"] == shot
        assert attrs["alt"].strip()
        assert (int(attrs["width"]), int(attrs["height"])) == dimensions
        assert attrs["src"].startswith("data:image/")
        mime, payload = attrs["src"].split(";base64,", maxsplit=1)
        assert mime in {
            "data:image/png",
            "data:image/webp",
            "data:image/jpeg",
        }
        decoded = base64.b64decode(payload)
        assert len(decoded) >= 10_000
        assert hashlib.sha256(decoded).hexdigest() == expected_hash
        caption = re.sub(r"<[^>]+>", "", structure.group("caption")).strip()
        assert caption


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


def test_design_user_guide_has_frozen_content_boundaries():
    text = _guide_without_embedded_images()
    forbidden = (
        "TO" + "DO",
        "T" + "BD",
        "待" + "补充",
        "文档版本",
        "发布日期",
        "更新日期",
        "<script src=",
        '<link rel="stylesheet"',
    )

    assert not any(marker in text for marker in forbidden)
    assert not re.search(r"\b20\d{2}[-/.]\d{1,2}[-/.]\d{1,2}\b", text)
    assert not re.search(r"\b20\d{6}T\d{6}(?:-[0-9a-f]{8})?\b", text)
    assert not any(
        marker in text
        for marker in ("\ufffd", "\u00c3", "\u00c2", "\u951f\u65a4\u62f7")
    )


def test_design_user_guide_has_source_boundary_and_dynamic_local_snapshot():
    source_boundary = _guide_section_text("source-boundary")
    expected_sources = {
        r"D:\battery_SOH\docs\design\soh-system-design.md",
        r"D:\battery_SOH\docs\design\soh-algorithm-design.md",
        r"D:\battery_SOH\docs\research\multi_dataset_soh_benchmark_landscape.md",
    }
    expected_snapshots = {
        "calce_cs2_features.pkl",
        "hust_features.pkl",
        "nasa_features.pkl",
        "oxford_features.pkl",
        "rwth_isea_public_features.pkl",
        "smoke_eval_demo_features.pkl",
        "snl_lfp_features.pkl",
        "uconn_features.pkl",
    }

    assert "当前代码、测试和运行契约" in source_boundary
    assert "最高优先级" in source_boundary
    assert expected_sources <= {
        item for item in expected_sources if item in source_boundary
    }
    assert expected_snapshots <= {
        item for item in expected_snapshots if item in source_boundary
    }
    assert "本机现场快照" in source_boundary
    assert "动态发现" in source_boundary
    assert not re.search(r"\b8\s*(?:个|套|组|份)\s*数据集", source_boundary)


def test_delivery_checklist_covers_complete_freeze_contract():
    checklist = _guide_section_text("delivery-checklist")
    required = {
        "单文件",
        "内联资产",
        "13 章",
        "五页签",
        "当前实现",
        "领域能力",
        "目标架构",
        "训练",
        "冻结推理",
        "五张截图",
        "alt",
        "搜索",
        "路径",
        "复制",
        "scroll-spy",
        "1280×720",
        "390×844",
        "禁用 JavaScript",
        "打印",
        "网络请求为 0",
        "console error 为 0",
        "敏感信息",
        "安全边界",
        "validator",
        "focused pytest",
        "完整 pytest",
    }
    assert required <= {item for item in required if item in checklist}
    assert "三条受众阅读路径，加一个“全部”切换项" in checklist
    assert "五种阅读路径" not in checklist


def test_architecture_distinguishes_target_flow_from_current_phase0_path():
    architecture = _guide_section_text("architecture")
    target_terms = {
        "目标架构端到端流程",
        "用户确认",
        "Adaptation",
        "Metric Worker",
        "PolicySpec",
        "签署发布",
    }
    current_terms = {
        "当前 Phase 0 路径",
        "同一 Python 进程",
        "剥离候选输入中的目标列",
        "内部计算指标",
        "写入 EvidencePack",
        "目录发布",
    }

    assert target_terms <= {term for term in target_terms if term in architecture}
    assert current_terms <= {term for term in current_terms if term in architecture}


def test_metric_truth_and_publication_claims_are_scoped_by_layer():
    contracts = _guide_section_text("contracts")

    assert "Metric Worker 独占真值只属于目标架构" in contracts
    assert "当前服务会剥离候选输入中的目标列并在内部计算指标" in contracts
    assert "当前发布不以 reviewed verdict 或 PolicySpec 为门控" in contracts
    assert "当前 artifact 完整性" in contracts
    assert "目标资格政策" in contracts


def test_target_state_lanes_use_exact_authoritative_value_sets():
    _assert_structured_fact_contract(GUIDE.read_text(encoding="utf-8"))


def test_target_state_explanations_use_authoritative_vocabulary_only():
    contracts = _guide_section_text("contracts")
    text = _guide_without_embedded_images()

    assert "complete、partial、failed、timed_out、cancelled、invalid" in contracts
    assert "pass、warn、fail、not_applicable、not_reviewed" in contracts
    for forbidden in (
        "partially_succeeded",
        "accepted",
        "rejected",
        "not_evaluated",
    ):
        assert forbidden not in text


def test_domain_claims_match_current_feature_policy_and_dataset_integration():
    domain = _guide_section_text("domain")
    required = {
        "当前内建 ic_dva 只有 IC 峰高和峰位",
        "DVA、面积和形状",
        "方法背景或目标能力",
        "capacity_time",
        "cycle_fraction",
        "FeaturePolicy gap",
        "待关闭",
        "尚未全面阻断未来信息",
        "UConn-ISU-ILCC LFP/Gr",
        "已接入",
        "ISU-ILCC Figshare NMC/Gr",
        "未接入",
    }

    assert required <= {term for term in required if term in domain}


def test_structured_fact_markers_preserve_direction_and_current_target_scope():
    _assert_structured_fact_contract(GUIDE.read_text(encoding="utf-8"))


def test_state_value_attribute_cannot_mask_mutated_visible_text():
    source = GUIDE.read_text(encoding="utf-8")
    mutated = _replace_once(
        source,
        'data-state-value="complete">complete</span>',
        'data-state-value="complete">succeeded</span>',
    )

    with pytest.raises(AssertionError):
        _assert_structured_fact_contract(mutated)


@pytest.mark.parametrize(
    ("marker", "replacement"),
    [
        (
            'data-dataset-id="uconn-isu-ilcc-lfp-gr"',
            "UConn-ISU-ILCC LFP/Gr 当前未接入，只出现在研究资料中",
        ),
        (
            'data-dataset-id="isu-ilcc-figshare-nmc-gr"',
            "当前已接入的是 ISU-ILCC Figshare NMC/Gr 数据",
        ),
    ],
)
def test_dataset_attribute_cannot_mask_reversed_visible_relationship(
    marker, replacement
):
    source = GUIDE.read_text(encoding="utf-8")
    mutated = _replace_element_text(source, marker, replacement)

    with pytest.raises(AssertionError):
        _assert_structured_fact_contract(mutated)


def test_open_gap_attribute_cannot_mask_closed_visible_claim():
    source = GUIDE.read_text(encoding="utf-8")
    mutated = _replace_once(
        source,
        (
            "<code>cycle_fraction</code> 是待关闭的 FeaturePolicy gap；"
            "在关闭前不能宣称该方案已经全面满足未来信息隔离。"
        ),
        (
            "<code>cycle_fraction</code> 的 FeaturePolicy gap 已关闭；"
            "现在可以宣称该方案已经全面满足未来信息隔离。"
        ),
    )

    with pytest.raises(AssertionError):
        _assert_structured_fact_contract(mutated)


def test_design_user_guide_forbids_typical_reversed_fact_claims():
    text = _guide_without_embedded_images()
    reversed_claims = (
        "UConn-ISU-ILCC LFP/Gr 当前未接入",
        "ISU-ILCC Figshare NMC/Gr 当前已接入",
        "capacity_time 的 FeaturePolicy gap 已关闭",
        "capacity_time 不含 cycle_fraction",
        "当前内建 ic_dva 包含 DVA、面积和形状",
        "当前只允许 Metric Worker 连接测试真值",
        "当前 Metric Worker 独占测试真值",
    )

    assert not [claim for claim in reversed_claims if claim in text]


def test_battery_dashboard_is_transactional_sync_target_not_tracked_payload():
    reference = _guide_section_text("reference")
    required = {
        r"frontend\app\battery-dashboard",
        "事务同步目标",
        "生成物不入库",
        "manifest",
        "运行状态",
    }

    assert required <= {term for term in required if term in reference}


def test_design_user_guide_keeps_current_and_target_capabilities_distinct():
    text = GUIDE.read_text(encoding="utf-8")
    assert text.count("当前已实现") >= 5
    assert text.count("已接入领域能力") >= 3
    assert text.count("目标架构") >= 3
    assert "隔离 Worker" in text
    assert "规划能力" in text


def test_diagram_fallbacks_are_adjacent_and_outside_role_img():
    parser = DiagramFallbackParser()
    parser.feed(GUIDE.read_text(encoding="utf-8"))

    assert len(parser.diagram_refs) == 4
    assert all(parser.diagram_refs), "every role=img diagram needs aria-describedby"
    assert parser.fallback_ids == parser.diagram_refs
    assert parser.nested_fallbacks == []
    assert parser.adjacency_errors == []
    assert parser.pending_ref is None


def test_design_user_guide_rejects_multitoken_stylesheet_rel(tmp_path):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8").replace(
            "</head>",
            '<link rel="alternate stylesheet" href="guide.css">\n</head>',
        ),
        encoding="utf-8",
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "external-asset" for issue in issues)


def test_design_user_guide_documents_current_frontend_modes_and_artifacts():
    text = GUIDE.read_text(encoding="utf-8")
    for mode in (
        "训练测评",
        "推理测评",
        "模型资产库",
        "SOH 完整看板",
        "全局诊断",
    ):
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


def test_each_frontend_mode_has_exactly_the_six_required_subsections():
    parser = ModeGuideParser()
    parser.feed(GUIDE.read_text(encoding="utf-8"))

    expected_modes = {
        "训练测评",
        "推理测评",
        "模型资产库",
        "SOH 完整看板",
        "全局诊断",
    }
    expected_headings = [
        "适用场景",
        "必需输入",
        "操作步骤",
        "成功信号",
        "常见失败",
        "关联产物",
    ]
    assert set(parser.headings) == expected_modes
    assert all(headings == expected_headings for headings in parser.headings.values())


def test_design_user_guide_has_verified_start_and_package_commands():
    assert _guide_section_code_blocks("packaging") == [
        r".venv\Scripts\python.exe scripts\build_frontend_artifacts.py --help",
        r".venv\Scripts\python.exe scripts\sync_battery_dashboard_native.py --help",
        (
            r".venv\Scripts\python.exe scripts\package_portable_frontend.py "
            r"--output dist\soh-eval-portable"
        ),
        (
            r".venv\Scripts\python.exe scripts\build_standalone_frontend.py "
            r"--output dist\soh-eval-standalone.html"
        ),
    ]


def test_setup_has_exact_health_and_port_conflict_command_blocks():
    setup_blocks = _guide_section_code_blocks("setup")

    assert setup_blocks[-4:-2] == [
        'Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8001/api/health"',
        (
            "Get-NetTCPConnection -LocalPort 8001,8010 -State Listen "
            "-ErrorAction SilentlyContinue |\n"
            "  Select-Object LocalAddress,LocalPort,OwningProcess"
        ),
    ]


def test_setup_has_exact_captured_pid_start_and_stop_blocks():
    setup_blocks = _guide_section_code_blocks("setup")

    assert setup_blocks[-2:] == [
        (
            '$api = Start-Process -WindowStyle Hidden -FilePath '
            '".venv\\Scripts\\python.exe" -ArgumentList '
            '@("-m","sohbench.api_server","--host","127.0.0.1","--port","8001",'
            '"--data-dir","D:\\battery_SOH\\data","--work-dir",'
            '"work\\trusted_plugins","--runs-dir","runs","--frontend-data-dir",'
            '"frontend\\app\\data") -PassThru\n'
            '$web = Start-Process -WindowStyle Hidden -FilePath '
            '".venv\\Scripts\\python.exe" -ArgumentList '
            '@("-m","http.server","8010","--directory","frontend/app") -PassThru\n'
            "$apiPid = $api.Id\n"
            "$webPid = $web.Id"
        ),
        (
            "# 只停止本次捕获的进程 ID\n"
            "@($apiPid,$webPid) | ForEach-Object {\n"
            "  if (Get-Process -Id $_ -ErrorAction SilentlyContinue) "
            "{ Stop-Process -Id $_ }\n"
            "}"
        ),
    ]


def test_design_user_guide_covers_operational_contracts_and_security_boundaries():
    text = GUIDE.read_text(encoding="utf-8")
    expected = {
        "contentBase64",
        "modelContentBase64",
        "manifestContentBase64",
        "model_manifest.json",
        "fit()",
        "predict()",
        "samples_per_second",
        "Get-NetTCPConnection",
        "Stop-Process",
        "last-known-good",
        "路径穿越",
        "Windows 设备别名",
        "不受信任的 Pickle",
    }
    assert expected <= {term for term in expected if term in text}


def test_api_reference_exactly_matches_source_get_and_post_routes():
    parser = ApiRouteTableParser()
    parser.feed(GUIDE.read_text(encoding="utf-8"))

    assert parser.routes == _source_method_routes()
    assert Counter(method for method, _ in parser.routes) == {
        "GET": 9,
        "POST": 11,
    }


def test_setup_documents_exact_backend_defaults_and_frontend_startup_command():
    defaults = parse_args([])
    assert vars(defaults) == {
        "host": "127.0.0.1",
        "port": 8001,
        "data_dir": r"D:\battery_SOH\data",
        "work_dir": "work/trusted_plugins",
        "runs_dir": "runs",
        "frontend_data_dir": "frontend/app/data",
    }

    setup = _guide_section_text("setup")
    backend = (
        ".venv\\Scripts\\python.exe -m sohbench.api_server "
        f"--host {defaults.host} --port {defaults.port} "
        f'--data-dir "{defaults.data_dir}" '
        f'--work-dir "{defaults.work_dir.replace("/", chr(92))}" '
        f'--runs-dir "{defaults.runs_dir}" '
        f'--frontend-data-dir "{defaults.frontend_data_dir.replace("/", chr(92))}"'
    )
    assert backend in setup
    assert (
        ".venv\\Scripts\\python.exe -m http.server 8010 "
        "--directory frontend/app"
    ) in setup
    assert (
        "http://127.0.0.1:8010/?api=http://127.0.0.1:8001"
    ) in setup


def test_publication_claims_are_scoped_by_artifact_kind_and_success():
    runs = _guide_section_text("runs")
    evidence = _guide_section_text("evidence")
    packaging = _guide_section_text("packaging")

    assert "重跑总是创建新任务和新 run ID" in runs
    assert "目录发布可先物化新的不可变 generation" in runs
    assert "CURRENT.json 更新失败时不会激活该 generation" in runs
    assert "旧 pointer 和 last-known-good 保持可读" in runs

    assert "原始运行目录与前端 publication generation 是两类对象" in evidence
    assert "原始运行目录不会推进 CURRENT.json" in evidence

    assert "generation 和 CURRENT.json 只属于目录发布" in packaging
    assert "generation 可在 CURRENT.json 更新前物化" in packaging
    assert "指针更新失败不会激活它" in packaging
    assert "旧 pointer 和 last-known-good 继续可读" in packaging
    assert "兼容视图后续同步失败不破坏活动 generation" in packaging
    assert "Standalone 不创建 generation，也不读写 CURRENT.json" in packaging
    assert "先验证临时 HTML，再原子替换一个输出文件" in packaging
    assert "失败时保留原有 standalone 文件" in packaging


def test_reference_precisely_describes_raw_run_directories():
    reference = _guide_section_text("reference")
    assert "不可变的原始运行目录，可作为后续合并来源" in reference
