import ast
import base64
from collections import Counter
import hashlib
import html
import json
import re
import shutil
import struct
import subprocess
import zlib
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
    return " ".join("".join(parts).split())


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


class StructuredCurrentClaimParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.claims = {}
        self.target_confirmations = []
        self.current_kind = None
        self.current_key = None
        self.current_attrs = None
        self.depth = 0
        self.parts = []

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        if self.current_kind is not None:
            self.depth += 1
            return

        if "data-current-fact" in values:
            self.current_kind = "claim"
            self.current_key = values["data-current-fact"]
        elif "data-target-user-confirmation" in values:
            self.current_kind = "target-confirmation"
            self.current_key = values["data-target-user-confirmation"]
        else:
            return

        self.current_attrs = values
        self.depth = 1
        self.parts = []

    def handle_data(self, data):
        if self.current_kind is not None:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if self.current_kind is None:
            return
        self.depth -= 1
        if self.depth:
            return

        item = {
            "attributes": self.current_attrs,
            "text": _normalize_visible_text(self.parts),
        }
        if self.current_kind == "claim":
            assert self.current_key not in self.claims
            self.claims[self.current_key] = item
        else:
            self.target_confirmations.append(item)
        self.current_kind = None
        self.current_key = None
        self.current_attrs = None
        self.parts = []


class QuickSourceParser(HTMLParser):
    _VOID_ELEMENTS = {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.chapter_id = None
        self.chapter_depth = 0
        self.in_quick_source = False
        self.quick_depth = 0
        self.in_term = False
        self.term_parts = []
        self.fields = {}

    def handle_starttag(self, tag, attrs):
        values = dict(attrs)
        classes = set(values.get("class", "").split())
        if (
            self.chapter_id is None
            and tag == "section"
            and "chapter" in classes
        ):
            self.chapter_id = values.get("id")
            self.chapter_depth = 1
            self.fields.setdefault(self.chapter_id, [])
            return
        if self.chapter_id is None:
            return

        if tag not in self._VOID_ELEMENTS:
            self.chapter_depth += 1
        if tag == "div" and "quick-source" in classes:
            self.in_quick_source = True
            self.quick_depth = 1
            return
        if self.in_quick_source:
            if tag not in self._VOID_ELEMENTS:
                self.quick_depth += 1
            if tag == "dt":
                self.in_term = True
                self.term_parts = []

    def handle_data(self, data):
        if self.in_term:
            self.term_parts.append(data)

    def handle_endtag(self, tag):
        if self.chapter_id is None:
            return
        if self.in_quick_source:
            if tag == "dt" and self.in_term:
                self.fields[self.chapter_id].append(
                    _normalize_visible_text(self.term_parts)
                )
                self.in_term = False
                self.term_parts = []
            self.quick_depth -= 1
            if self.quick_depth == 0:
                self.in_quick_source = False
        self.chapter_depth -= 1
        if self.chapter_depth == 0:
            self.chapter_id = None


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


def _write_guide(tmp_path, source=None):
    guide = tmp_path / "guide.html"
    guide.write_text(
        GUIDE.read_text(encoding="utf-8") if source is None else source,
        encoding="utf-8",
    )
    return guide


def _replace_shot_src(source, shot, replacement):
    pattern = re.compile(
        rf'(<img\b(?=[^>]*\bdata-shot="{re.escape(shot)}")'
        rf'(?=[^>]*\bsrc=")[^>]*\bsrc=")[^"]*(")',
        flags=re.IGNORECASE,
    )
    mutated, count = pattern.subn(
        lambda match: f"{match.group(1)}{replacement}{match.group(2)}",
        source,
        count=1,
    )
    assert count == 1
    return mutated


def _png_bytes(width=2, height=2, rgb=(32, 96, 160)):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = bytes(rgb) * width
    scanlines = b"".join(b"\x00" + row for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def _png_bytes_with_invalid_zlib_stream(width=2, height=2):
    def chunk(kind, payload):
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", b"not-a-zlib-stream")
        + chunk(b"IEND", b"")
    )


def _non_png_image_bytes(mime):
    if mime == "image/jpeg":
        return (
            b"\xff\xd8\xff\xc0\x00\x11\x08\x00\x01\x00\x01"
            b"\x03\x01\x11\x00\x02\x11\x00\x03\x11\x00\xff\xd9"
        )
    if mime == "image/webp":
        image_data = b"\x2f\x00\x00\x00\x00"
        chunk = (
            b"VP8L"
            + struct.pack("<I", len(image_data))
            + image_data
            + b"\x00"
        )
        return (
            b"RIFF"
            + struct.pack("<I", 4 + len(chunk))
            + b"WEBP"
            + chunk
        )
    raise AssertionError(f"unsupported test MIME: {mime}")


def _approved_capture_set(tmp_path):
    source = GUIDE.read_text(encoding="utf-8")
    captures = {}
    for name in ("training", "inference", "models", "native", "diagnostics"):
        match = re.search(
            rf'<img\b(?=[^>]*data-shot="{name}")'
            rf'(?=[^>]*src="data:image/png;base64,([^"]+)")[^>]*>',
            source,
        )
        assert match is not None
        capture = tmp_path / f"{name}.png"
        capture.write_bytes(base64.b64decode(match.group(1)))
        captures[name] = capture
    return captures


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


def _assert_current_claim_contract(source):
    parser = StructuredCurrentClaimParser()
    parser.feed(source)

    assert parser.claims["task-verdict"]["attributes"][
        "data-current-values"
    ].split() == ["not_reviewed", "fail"]
    assert parser.claims["task-verdict"]["text"] == (
        "当前任务 verdict 字段只使用 not_reviewed / fail。"
    )
    assert parser.claims["leaderboard-verdicts"]["attributes"][
        "data-current-checks"
    ].split() == [
        "generalization",
        "calibration",
        "physical_consistency",
    ]
    assert parser.claims["leaderboard-verdicts"]["text"] == (
        "当前 leaderboard 的 verdicts 是 generalization / calibration / "
        "physical_consistency 三个具名布尔检查，不是任务 verdict 枚举。"
    )

    expected_curve_claims = {
        "overview-ic": "处理后特征表发现、SOH 标签、IC 峰特征和 IC dQ/dV 曲线。",
        "method-ic": "当前内建 ic_dva 只有 IC 峰高和峰位。",
        "domain-status-ic": "已接入多种充电摘要、固定窗口、IC 峰特征、IC dQ/dV 曲线和 SOH 轨迹。",
        "five-tab-ic": "SOH 完整看板当前展示 IC dQ/dV 曲线。",
        "native-purpose-ic": "查看 SOH 轨迹，并从异常周期下钻到充电电压、容量、时间曲线、IC dQ/dV 曲线及高亮特征段。",
        "native-operation-ic": "点击周期点联动电压-容量-时间曲线，查看 IC dQ/dV 曲线与特征高亮区段。",
    }
    assert {
        key: parser.claims[key]["text"] for key in expected_curve_claims
    } == expected_curve_claims
    assert not [
        key
        for key in expected_curve_claims
        if "DVA" in parser.claims[key]["text"]
    ]

    assert [
        (
            item["attributes"].get("data-target-user-confirmation"),
            item["attributes"].get("data-capability-state"),
            item["text"],
        )
        for item in parser.target_confirmations
    ] == [
        (
            "control-visual",
            "target",
            "Registry、RunSpec、comparison key、预检、用户确认（目标）和任务状态。",
        ),
        (
            "control-fallback",
            "target",
            "控制面：Registry、RunSpec、预检、用户确认（目标）和任务状态。",
        ),
        (
            "workflow-visual",
            "target",
            "5 用户确认（目标） 冻结运行矩阵",
        ),
        (
            "workflow-fallback",
            "target",
            "由用户确认（目标）运行矩阵。",
        ),
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


def _run_chrome_guide_driver(
    tmp_path,
    driver,
    *,
    source_transform=None,
    window_size=(1280, 720),
):
    chrome = _find_chrome()
    if chrome is None:
        pytest.skip("Chrome/Edge not found; headless guide behavior smoke skipped")

    driven_guide = tmp_path / "guide-smoke.html"
    source = GUIDE.read_text(encoding="utf-8")
    if source_transform is not None:
        source = source_transform(source)
    driven_guide.write_text(
        source.replace("</body>", f"<script>{driver}</script></body>"),
        encoding="utf-8",
    )
    profile = tmp_path / (
        f"chrome-profile-{window_size[0]}-{len(list(tmp_path.glob('chrome-profile-*')))}"
    )
    completed = subprocess.run(
        [
            str(chrome),
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            "--no-default-browser-check",
            "--allow-file-access-from-files",
            f"--window-size={window_size[0]},{window_size[1]}",
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


def test_headless_chrome_mobile_and_no_enhancement_offline_regression(
    tmp_path,
):
    def install_browser_monitor(source):
        monitor = r"""
        <script>
          window.__guideSmokeErrors = [];
          window.addEventListener("error", (event) => {
            window.__guideSmokeErrors.push(`error:${event.message}`);
          }, true);
          window.addEventListener("unhandledrejection", (event) => {
            window.__guideSmokeErrors.push(
              `unhandledrejection:${String(event.reason)}`
            );
          });
        </script>
        """
        return source.replace("</head>", f"{monitor}</head>", 1)

    driver = r"""
    (async () => {
      await new Promise((resolve) => window.setTimeout(resolve, 100));
      const chapters = [...document.querySelectorAll(".chapter")];
      const anchors = [...document.querySelectorAll(".mobile-toc a[href^='#']")];
      const payload = {
        widthMatches:
          document.documentElement.scrollWidth ===
          document.documentElement.clientWidth,
        visibleChapters: chapters.filter(
          (chapter) => getComputedStyle(chapter).display !== "none"
        ).length,
        validAnchors: anchors.filter((anchor) => {
          const target = document.querySelector(anchor.getAttribute("href"));
          return target && getComputedStyle(target).display !== "none";
        }).length,
        images: [...document.images].map((image) => ({
          complete: image.complete,
          width: image.naturalWidth,
        })),
        httpResources: performance.getEntriesByType("resource")
          .map((entry) => entry.name)
          .filter((name) => /^https?:/i.test(name)),
        runtimeErrors: window.__guideSmokeErrors,
      };
      const output = document.createElement("pre");
      output.id = "smoke-result";
      output.textContent = JSON.stringify(payload);
      document.body.replaceChildren(output);
    })();
    """

    enhanced = _run_chrome_guide_driver(
        tmp_path,
        driver,
        source_transform=install_browser_monitor,
        window_size=(390, 844),
    )

    def remove_enhancement(source):
        stripped, count = re.subn(
            r"\s*<script>\s*\(\(\) => \{.*?</script>",
            "",
            source,
            count=1,
            flags=re.DOTALL,
        )
        assert count == 1
        return install_browser_monitor(stripped)

    no_enhancement = _run_chrome_guide_driver(
        tmp_path,
        driver,
        source_transform=remove_enhancement,
        window_size=(390, 844),
    )

    assert enhanced["widthMatches"] is True
    assert len(enhanced["images"]) == 5
    assert all(
        image["complete"] and image["width"] > 0
        for image in enhanced["images"]
    )
    assert enhanced["httpResources"] == []
    assert enhanced["runtimeErrors"] == []
    assert no_enhancement["widthMatches"] is True
    assert no_enhancement["visibleChapters"] == len(REQUIRED_SECTIONS)
    assert no_enhancement["validAnchors"] == len(REQUIRED_SECTIONS)
    assert len(no_enhancement["images"]) == 5
    assert all(
        image["complete"] and image["width"] > 0
        for image in no_enhancement["images"]
    )
    assert no_enhancement["httpResources"] == []
    assert no_enhancement["runtimeErrors"] == []


def test_headless_chrome_initialization_failure_preserves_mobile_toc(
    tmp_path,
):
    def inject_initialization_failure(source):
        marker = "    enhanceChapterDetails();"
        assert source.count(marker) == 1
        return source.replace(
            marker,
            '    throw new Error("injected initialization failure");\n'
            + marker,
            1,
        )

    result = _run_chrome_guide_driver(
        tmp_path,
        r"""
        (() => {
          const toc = document.querySelector(".mobile-toc");
          const payload = {
            jsActive: document.documentElement.classList.contains("js"),
            tocVisible: getComputedStyle(toc).display !== "none",
            anchors: toc.querySelectorAll('a[href^="#"]').length,
            visibleChapters: [...document.querySelectorAll(".chapter")]
              .filter((chapter) => getComputedStyle(chapter).display !== "none")
              .length,
          };
          const output = document.createElement("pre");
          output.id = "smoke-result";
          output.textContent = JSON.stringify(payload);
          document.body.replaceChildren(output);
        })();
        """,
        source_transform=inject_initialization_failure,
        window_size=(390, 844),
    )

    assert result == {
        "jsActive": False,
        "tocVisible": True,
        "anchors": len(REQUIRED_SECTIONS),
        "visibleChapters": len(REQUIRED_SECTIONS),
    }


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


def test_every_chapter_quick_source_has_exact_required_fields():
    parser = QuickSourceParser()
    parser.feed(GUIDE.read_text(encoding="utf-8"))

    assert set(parser.fields) == set(REQUIRED_SECTIONS)
    assert all(
        fields == ["输入", "命令", "输出", "检查", "风险"]
        for fields in parser.fields.values()
    )


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
    assert re.search(
        r"\.mode-guide\s*\{[^}]*break-inside:\s*auto\s*!important;"
        r"[^}]*page-break-inside:\s*auto\s*!important;",
        print_css,
    )

    reduced_css = text[text.index("@media (prefers-reduced-motion: reduce)") :]
    assert "scroll-behavior: auto !important" in reduced_css
    assert "transition: none !important" in reduced_css


def test_dynamic_chapter_summaries_have_unique_descriptive_accessible_names():
    source = GUIDE.read_text(encoding="utf-8")
    headings = re.findall(
        r'<section id="[^"]+" class="chapter"[^>]*>\s*<h2>([^<]+)</h2>',
        source,
    )
    assert len(headings) == len(REQUIRED_SECTIONS)

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the inline guide contract")
    runner = f"""
const headings = {json.dumps(headings, ensure_ascii=False)};
const observed = [];
const chapters = headings.map((text) => {{
  const heading = {{
    textContent: text,
    nextElementSibling: null,
    nextSibling: null,
  }};
  return {{
    querySelector: () => heading,
    append: () => {{}},
  }};
}});
const document = {{
  createElement(tag) {{
    if (tag === "summary") {{
      return {{
        textContent: "",
        attributes: {{}},
        setAttribute(name, value) {{
          this.attributes[name] = value;
        }},
      }};
    }}
    if (tag === "details") {{
      return {{
        className: "",
        open: false,
        append(summary) {{
          observed.push({{
            visible: summary.textContent,
            accessible: summary.attributes["aria-label"]
              || summary.textContent,
          }});
        }},
      }};
    }}
    return {{ className: "", append() {{}} }};
  }},
}};
{_guide_function_source("enhanceChapterDetails")}
enhanceChapterDetails();
process.stdout.write(JSON.stringify(observed));
"""
    completed = subprocess.run(
        [node, "-e", runner],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    accessible_names = [item["accessible"] for item in observed]

    assert len(observed) == len(REQUIRED_SECTIONS)
    assert len(set(accessible_names)) == len(REQUIRED_SECTIONS)
    assert all(
        f"第 {index} 章" in name and heading in name
        for index, (heading, name) in enumerate(
            zip(headings, accessible_names),
            start=1,
        )
    )
    assert {
        item["visible"] for item in observed
    } == {"展开或折叠本章内容"}


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
    "resource",
    (
        "http://127.0.0.1:9/sidecar.png",
        "./sidecar.png",
    ),
)
def test_validator_rejects_css_image_set_resources(tmp_path, resource):
    rule = f'.mutation {{ background-image: image-set("{resource}" 1x); }}'
    guide = _write_guide(
        tmp_path,
        GUIDE.read_text(encoding="utf-8").replace(
            "</style>",
            f"{rule}</style>",
            1,
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "inline-style" for issue in issues), issues


def test_validator_requires_exactly_one_approved_inline_style_body(tmp_path):
    guide = _write_guide(
        tmp_path,
        GUIDE.read_text(encoding="utf-8").replace(
            "</head>",
            "<style>.mutation { color: inherit; }</style></head>",
            1,
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "inline-style" for issue in issues), issues


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


@pytest.mark.parametrize(
    ("case", "replacement", "expected_code"),
    [
        (
            "malformed-base64",
            "data:image/png;base64,%%%",
            "image-data",
        ),
        (
            "truncated-png",
            "data:image/png;base64,"
            + base64.b64encode(b"\x89PNG\r\n\x1a\n").decode("ascii"),
            "image-data",
        ),
        (
            "mime-magic-mismatch",
            "data:image/jpeg;base64,"
            + base64.b64encode(_png_bytes()).decode("ascii"),
            "image-mime",
        ),
        (
            "zero-payload-width",
            "data:image/png;base64,"
            + base64.b64encode(_png_bytes(width=0, height=1)).decode("ascii"),
            "image-dimensions",
        ),
    ],
)
def test_validator_rejects_malformed_or_misdeclared_embedded_images(
    tmp_path, case, replacement, expected_code
):
    source = _replace_shot_src(
        GUIDE.read_text(encoding="utf-8"),
        "training",
        replacement,
    )
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == expected_code for issue in issues), (
        case,
        issues,
    )


def test_validator_rejects_zero_declared_image_dimension(tmp_path):
    source = GUIDE.read_text(encoding="utf-8")
    marker = re.search(
        r'<img\b(?=[^>]*data-shot="training")[^>]*>',
        source,
    )
    assert marker is not None
    mutated_marker = marker.group(0).replace('width="1440"', 'width="0"', 1)
    assert mutated_marker != marker.group(0)
    source = source.replace(marker.group(0), mutated_marker, 1)
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-dimensions" for issue in issues)


def test_validator_rejects_declared_dimensions_that_do_not_match_payload(
    tmp_path,
):
    source = GUIDE.read_text(encoding="utf-8")
    marker = re.search(
        r'<img\b(?=[^>]*data-shot="training")[^>]*>',
        source,
    )
    assert marker is not None
    mutated_marker = marker.group(0).replace('width="1440"', 'width="1"', 1)
    assert mutated_marker != marker.group(0)
    guide = _write_guide(
        tmp_path,
        source.replace(marker.group(0), mutated_marker, 1),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-dimensions" for issue in issues)


def test_inspect_image_bytes_rejects_crc_correct_png_with_invalid_zlib_stream():
    from scripts.validate_design_user_guide import (
        ImageResourceError,
        inspect_image_bytes,
    )

    with pytest.raises(ImageResourceError, match="compressed pixel data"):
        inspect_image_bytes(
            "image/png",
            _png_bytes_with_invalid_zlib_stream(),
        )


def test_validator_rejects_crc_correct_png_with_invalid_zlib_stream(tmp_path):
    encoded = base64.b64encode(
        _png_bytes_with_invalid_zlib_stream()
    ).decode("ascii")
    source = _replace_shot_src(
        GUIDE.read_text(encoding="utf-8"),
        "training",
        f"data:image/png;base64,{encoded}",
    )
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-data" for issue in issues), issues


def test_validator_rejects_well_formed_but_unapproved_png(tmp_path):
    encoded = base64.b64encode(
        _png_bytes(width=1440, height=800, rgb=(1, 2, 3))
    ).decode("ascii")
    source = _replace_shot_src(
        GUIDE.read_text(encoding="utf-8"),
        "training",
        f"data:image/png;base64,{encoded}",
    )
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-approval" for issue in issues), issues


@pytest.mark.parametrize("mime", ("image/jpeg", "image/webp"))
def test_inspect_image_bytes_support_is_intentionally_png_only(mime):
    from scripts.validate_design_user_guide import (
        ImageResourceError,
        inspect_image_bytes,
    )

    with pytest.raises(ImageResourceError, match="unsupported image MIME"):
        inspect_image_bytes(mime, _non_png_image_bytes(mime))


@pytest.mark.parametrize("mime", ("image/jpeg", "image/webp"))
def test_validator_rejects_non_png_embedded_images(tmp_path, mime):
    encoded = base64.b64encode(_non_png_image_bytes(mime)).decode("ascii")
    source = _replace_shot_src(
        GUIDE.read_text(encoding="utf-8"),
        "training",
        f"data:{mime};base64,{encoded}",
    )
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-mime" for issue in issues), issues


def test_validator_requires_exact_embedded_screenshot_set(tmp_path):
    source = GUIDE.read_text(encoding="utf-8").replace(
        'data-shot="diagnostics"',
        'data-shot="unexpected"',
        1,
    )
    guide = _write_guide(tmp_path, source)

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image-set" for issue in issues)


@pytest.mark.parametrize(
    "injection",
    (
        '<iframe src="frame.html"></iframe>',
        '<object data="payload.bin"></object>',
        '<embed src="payload.bin">',
        '<link rel="icon" href="favicon.ico">',
        "<link rel href>",
        '<script src="guide.js"></script>',
        '<picture><source src="shot.webp"></picture>',
        '<video src="movie.mp4" poster="poster.png"></video>',
        '<audio src="audio.mp3"></audio>',
        '<img src="data:image/png;base64,AA==" srcset="remote.png 2x" '
        'alt="额外资源">',
        '<form action="/submit"><button type="submit">提交</button></form>',
    ),
)
def test_validator_rejects_sidecar_and_active_resource_references(
    tmp_path, injection
):
    guide = _write_guide(
        tmp_path,
        GUIDE.read_text(encoding="utf-8").replace(
            "</body>",
            injection + "</body>",
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "external-resource" for issue in issues), issues


def test_validator_reports_bare_image_resource_attribute_instead_of_crashing(
    tmp_path,
):
    source = GUIDE.read_text(encoding="utf-8")
    marker = re.search(
        r'<img\b(?=[^>]*data-shot="training")[^>]*>',
        source,
    )
    assert marker is not None
    mutated_marker = re.sub(r'src="[^"]+"', "src", marker.group(0), count=1)
    guide = _write_guide(
        tmp_path,
        source.replace(marker.group(0), mutated_marker, 1),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "image" for issue in issues)


def test_validator_rejects_first_duplicate_src_even_when_second_is_embedded(
    tmp_path,
):
    source = GUIDE.read_text(encoding="utf-8")
    marker = re.search(
        r'<img\b(?=[^>]*data-shot="training")[^>]*>',
        source,
    )
    assert marker is not None
    mutated_marker, count = re.subn(
        r'\bsrc="',
        'src="./sidecar.png" src="',
        marker.group(0),
        count=1,
    )
    assert count == 1
    guide = _write_guide(
        tmp_path,
        source.replace(marker.group(0), mutated_marker, 1),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "duplicate-attribute" for issue in issues), issues
    assert any(issue.code == "image" for issue in issues), issues


def test_validator_rejects_duplicate_attributes_case_insensitively(tmp_path):
    guide = _write_guide(
        tmp_path,
        _replace_once(
            GUIDE.read_text(encoding="utf-8"),
            "<body>",
            '<body CLASS="first" class="second">',
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "duplicate-attribute" for issue in issues), issues


def test_validator_rejects_duplicate_nonempty_ids(tmp_path):
    guide = _write_guide(
        tmp_path,
        _replace_once(
            GUIDE.read_text(encoding="utf-8"),
            "<body>",
            '<body id="overview">',
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    duplicate_ids = [
        issue for issue in issues if issue.code == "duplicate-id"
    ]
    assert len(duplicate_ids) == 1, issues
    assert "#overview" in duplicate_ids[0].message


@pytest.mark.parametrize(
    ("old", "new"),
    (
        (
            "<body>",
            '<body style="background-image:url(./sidecar.png)">',
        ),
        (
            "<body>",
            '<body onload="window[\'fetch\'](\'./sidecar.json\')">',
        ),
        (
            '<a class="skip-link" href="#main-content">',
            '<a class="skip-link" ping="./audit" href="#main-content">',
        ),
        (
            "<body>",
            '<body dynsrc="./legacy-sidecar.png">',
        ),
    ),
)
def test_validator_enforces_explicit_safe_attribute_contract(
    tmp_path, old, new
):
    guide = _write_guide(
        tmp_path,
        _replace_once(
            GUIDE.read_text(encoding="utf-8"),
            old,
            new,
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "html-attribute" for issue in issues), issues


def test_validator_rejects_bracket_member_fetch_in_inline_script(tmp_path):
    guide = _write_guide(
        tmp_path,
        _replace_once(
            GUIDE.read_text(encoding="utf-8"),
            "</script>",
            'window["fetch"]("./sidecar.json");\n  </script>',
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "inline-script" for issue in issues), issues


@pytest.mark.parametrize(
    "load_site",
    (
        'fetch("https://example.invalid/data.json")',
        'new XMLHttpRequest()',
        'new WebSocket("wss://example.invalid/socket")',
        'new EventSource("/events")',
        'navigator.sendBeacon("/audit", "x")',
        'import("remote-module.js")',
    ),
)
def test_validator_rejects_active_javascript_load_sites(tmp_path, load_site):
    guide = _write_guide(
        tmp_path,
        GUIDE.read_text(encoding="utf-8").replace(
            "</body>",
            f"<script>{load_site};</script></body>",
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert any(issue.code == "active-load" for issue in issues), issues


def test_validator_does_not_treat_literal_localhost_code_as_active_load(
    tmp_path,
):
    guide = _write_guide(
        tmp_path,
        GUIDE.read_text(encoding="utf-8").replace(
            "</body>",
            '<pre><code>fetch("http://127.0.0.1:8001/api/health")'
            "</code></pre></body>",
        ),
    )

    issues = validate_document(guide, source_root=ROOT)

    assert not [
        issue for issue in issues if issue.code in {"active-load", "network"}
    ]


def test_embed_screenshots_replaces_each_named_capture(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    capture = _approved_capture_set(tmp_path)["training"]

    embed_screenshots(html, {"training": capture})

    rendered = html.read_text(encoding="utf-8")
    expected = base64.b64encode(capture.read_bytes()).decode("ascii")
    assert f"data:image/png;base64,{expected}" in rendered


def test_embed_screenshots_rejects_missing_marker_without_changing_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    original = GUIDE.read_text(encoding="utf-8").replace(
        'data-shot="training"',
        'data-shot="removed-training"',
        1,
    )
    html = _write_guide(tmp_path, original)
    capture = tmp_path / "training.png"
    capture.write_bytes(_png_bytes())

    with pytest.raises(
        ValueError,
        match="expected one image marker for training, found 0",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_rejects_duplicate_marker_without_changing_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    source = GUIDE.read_text(encoding="utf-8")
    marker = re.search(
        r'<img\b(?=[^>]*data-shot="training")[^>]*>',
        source,
    )
    assert marker is not None
    original = source.replace(marker.group(0), marker.group(0) * 2, 1)
    html = _write_guide(tmp_path, original)
    capture = tmp_path / "training.png"
    capture.write_bytes(_png_bytes())

    with pytest.raises(
        ValueError,
        match="expected one image marker for training, found 2",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_late_marker_failure_leaves_original_html(tmp_path):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    original = GUIDE.read_text(encoding="utf-8").replace(
        'data-shot="inference"',
        'data-shot="removed-inference"',
        1,
    )
    html = _write_guide(tmp_path, original)
    training = tmp_path / "training.png"
    inference = tmp_path / "inference.png"
    training.write_bytes(_png_bytes(rgb=(1, 2, 3)))
    inference.write_bytes(_png_bytes(rgb=(4, 5, 6)))

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

    html = _write_guide(tmp_path)
    original = html.read_text(encoding="utf-8")
    capture = tmp_path / "training.bmp"
    capture.write_bytes(b"bitmap fixture")

    with pytest.raises(ValueError, match="unsupported image type: .bmp"):
        embed_screenshots(html, {"training": capture})

    assert html.read_text(encoding="utf-8") == original


def test_embed_screenshots_rejects_truncated_named_image_without_changing_html(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    capture = tmp_path / "training.png"
    capture.write_bytes(b"\x89PNG\r\n\x1a\n")

    with pytest.raises(ValueError, match="invalid image"):
        embed_screenshots(html, {"training": capture})

    assert html.read_bytes() == original


def test_embed_screenshots_rejects_invalid_zlib_stream_without_changing_html(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    capture = tmp_path / "training.png"
    capture.write_bytes(_png_bytes_with_invalid_zlib_stream())

    with pytest.raises(ValueError, match="invalid image"):
        embed_screenshots(html, {"training": capture})

    assert html.read_bytes() == original


def test_embed_screenshots_rejects_unapproved_png_without_changing_html(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    capture = tmp_path / "training.png"
    capture.write_bytes(
        _png_bytes(width=1440, height=800, rgb=(1, 2, 3))
    )

    with pytest.raises(
        ValueError,
        match="candidate guide validation failed: image-approval",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_bytes() == original


@pytest.mark.parametrize(
    "resource",
    (
        "http://127.0.0.1:9/sidecar.png",
        "./sidecar.png",
    ),
)
def test_embed_screenshots_rejects_css_image_set_without_replacing_html(
    tmp_path, resource
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    rule = f'.mutation {{ background-image: image-set("{resource}" 1x); }}'
    source = GUIDE.read_text(encoding="utf-8").replace(
        "</style>",
        f"{rule}</style>",
        1,
    )
    source = _replace_shot_src(
        source,
        "training",
        "data:image/png;base64,AA==",
    )
    html = _write_guide(tmp_path, source)
    original = html.read_bytes()
    capture = _approved_capture_set(tmp_path)["training"]

    with pytest.raises(
        ValueError,
        match="candidate guide validation failed: inline-style",
    ):
        embed_screenshots(html, {"training": capture})

    assert html.read_bytes() == original


@pytest.mark.parametrize(
    ("suffix", "mime"),
    ((".jpg", "image/jpeg"), (".webp", "image/webp")),
)
def test_embed_screenshots_rejects_non_png_formats_without_changing_html(
    tmp_path, suffix, mime
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    capture = tmp_path / f"training{suffix}"
    capture.write_bytes(_non_png_image_bytes(mime))

    with pytest.raises(ValueError, match="unsupported image type"):
        embed_screenshots(html, {"training": capture})

    assert html.read_bytes() == original


def test_embed_screenshots_rejects_duplicate_id_without_replacing_html(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    source = _replace_once(
        GUIDE.read_text(encoding="utf-8"),
        "<body>",
        '<body id="overview">',
    )
    source = _replace_shot_src(
        source,
        "training",
        "data:image/png;base64,AA==",
    )
    html_path = _write_guide(tmp_path, source)
    original = html_path.read_bytes()
    capture = _approved_capture_set(tmp_path)["training"]

    with pytest.raises(
        ValueError,
        match="candidate guide validation failed: duplicate-id",
    ):
        embed_screenshots(html_path, {"training": capture})

    assert html_path.read_bytes() == original


def test_embed_screenshots_candidate_validation_failure_preserves_old_file(
    tmp_path, monkeypatch
):
    import scripts.embed_design_user_guide_screenshots as embed_module
    from scripts.validate_design_user_guide import ValidationIssue

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    captures = _approved_capture_set(tmp_path)
    monkeypatch.setattr(
        embed_module,
        "validate_document",
        lambda path: [
            ValidationIssue("injected-semantic", "candidate rejected")
        ],
    )

    with pytest.raises(ValueError, match="candidate guide validation failed"):
        embed_module.embed_screenshots(html, captures)

    assert html.read_bytes() == original


def test_embed_screenshots_partial_temporary_write_preserves_old_file(
    tmp_path, monkeypatch
):
    import sohbench.frontend_publish as frontend_publish_module
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    captures = _approved_capture_set(tmp_path)
    real_write_text = Path.write_text

    def fail_partial_temporary_write(path, text, *args, **kwargs):
        if path.parent == html.parent and path.name.startswith(
            f".{html.name}.tmp-"
        ):
            path.write_bytes(text[:64].encode("utf-8"))
            raise OSError("injected partial temporary write")
        return real_write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(
        frontend_publish_module.Path,
        "write_text",
        fail_partial_temporary_write,
    )

    with pytest.raises(OSError, match="partial temporary write"):
        embed_screenshots(html, captures)

    assert html.read_bytes() == original


def test_embed_screenshots_replace_failure_preserves_old_file(
    tmp_path, monkeypatch
):
    import sohbench.frontend_publish as frontend_publish_module
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    original = html.read_bytes()
    captures = _approved_capture_set(tmp_path)
    monkeypatch.setattr(
        frontend_publish_module,
        "_atomic_replace",
        lambda source, destination: (_ for _ in ()).throw(
            OSError("injected replace failure")
        ),
    )

    with pytest.raises(OSError, match="replace failure"):
        embed_screenshots(html, captures)

    assert html.read_bytes() == original


def test_embed_screenshots_post_replace_fsync_failure_leaves_new_file_active(
    tmp_path, monkeypatch
):
    import sohbench.frontend_publish as frontend_publish_module
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    source = _replace_shot_src(
        GUIDE.read_text(encoding="utf-8"),
        "training",
        "data:image/png;base64,AA==",
    )
    html = _write_guide(tmp_path, source)
    original_hash = hashlib.sha256(html.read_bytes()).hexdigest()
    captures = _approved_capture_set(tmp_path)
    monkeypatch.setattr(
        frontend_publish_module,
        "_fsync_directory",
        lambda path: (_ for _ in ()).throw(
            OSError("injected post-replace fsync failure")
        ),
    )

    with pytest.raises(OSError, match="post-replace fsync failure"):
        embed_screenshots(html, captures)

    assert hashlib.sha256(html.read_bytes()).hexdigest() != original_hash
    assert validate_document(html, source_root=ROOT) == []


def test_embed_screenshots_successfully_replaces_all_valid_image_hashes(
    tmp_path,
):
    from scripts.embed_design_user_guide_screenshots import embed_screenshots

    html = _write_guide(tmp_path)
    captures = _approved_capture_set(tmp_path)
    expected_hashes = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in captures.items()
    }

    embed_screenshots(html, captures)

    rendered = html.read_text(encoding="utf-8")
    for name, expected_hash in expected_hashes.items():
        match = re.search(
            rf'<img\b(?=[^>]*data-shot="{name}")'
            rf'(?=[^>]*src="data:image/[^;"]+;base64,([^"]+)")[^>]*>',
            rendered,
        )
        assert match is not None
        assert (
            hashlib.sha256(base64.b64decode(match.group(1))).hexdigest()
            == expected_hash
        )
    assert validate_document(html, source_root=ROOT) == []


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
        assert mime == "data:image/png"
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
    assert "controller 在 Round 3 后运行完整 pytest" in checklist
    assert "完整 Python suite 达到 100% 且 exit 0" in checklist
    assert (
        "Round 3 报告中的“未运行全量 Python”只记录此前修复代理 gate"
        in checklist
    )
    assert "HEAD" not in checklist
    assert "9b0f567" not in checklist


def test_architecture_distinguishes_target_flow_from_current_phase0_path():
    architecture = _guide_section_text("architecture")
    target_terms = {
        "目标架构端到端流程",
        "用户确认（目标）",
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
    assert "单文件发布" in contracts
    assert "同目录临时文件" in contracts
    assert "文件 fsync" in contracts
    assert "不创建 generation，也不读写 CURRENT.json" in contracts
    assert "目录发布才物化不可变 generation" in contracts
    assert "CURRENT.json 指向的 last-known-good" in contracts
    assert "当前文件发布使用不可变 generation" not in contracts
    assert "目标资格政策" in contracts


def test_target_state_lanes_use_exact_authoritative_value_sets():
    _assert_structured_fact_contract(GUIDE.read_text(encoding="utf-8"))


def test_current_task_leaderboard_and_curve_facts_are_structurally_separate():
    _assert_current_claim_contract(GUIDE.read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("old", "new"),
    [
        (
            "当前任务 verdict 字段只使用 "
            "<code>not_reviewed</code> / <code>fail</code>。",
            "当前任务 verdict 字段使用 "
            "<code>not_reviewed</code> / <code>pass</code> / "
            "<code>warn</code> / <code>fail</code>。",
        ),
        (
            "SOH 完整看板当前展示 IC <code>dQ/dV</code> 曲线。",
            "SOH 完整看板当前展示 IC/DVA 曲线。",
        ),
        (
            "<strong>5 用户确认（目标）</strong>",
            "<strong>5 用户确认</strong>",
        ),
    ],
)
def test_structured_current_fact_attributes_cannot_mask_reversed_visible_text(
    old, new
):
    source = GUIDE.read_text(encoding="utf-8")
    mutated = _replace_once(source, old, new)

    with pytest.raises(AssertionError):
        _assert_current_claim_contract(mutated)


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
        "IC dQ/dV 曲线",
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
    troubleshooting = _guide_section_text("troubleshooting")

    assert "重跑总是创建新任务和新 run ID" in runs
    assert "目录发布可先物化新的不可变 generation" in runs
    assert "CURRENT.json 的 os.replace 之前失败" in runs
    assert "旧 pointer 和 last-known-good 保持活动" in runs
    assert "替换后的目录 fsync 失败" in runs
    assert "新 pointer 已经活动" in runs
    assert "重试前重新读取 CURRENT.json" in runs

    assert "原始运行目录与前端 publication generation 是两类对象" in evidence
    assert "原始运行目录不会推进 CURRENT.json" in evidence

    assert "generation 和 CURRENT.json 只属于目录发布" in packaging
    assert "generation 可在 CURRENT.json 更新前物化" in packaging
    assert "pointer 的 os.replace 前失败不会激活它" in packaging
    assert "替换后的目录 fsync 失败具有提交状态歧义" in packaging
    assert "重新读取 CURRENT.json" in packaging
    assert "兼容视图后续同步失败不破坏活动 generation" in packaging
    assert "Standalone 不创建 generation，也不读写 CURRENT.json" in packaging
    assert "通用构建器只执行结构性 shell 检查" in packaging
    assert "doctype、standalone 标记和闭合 HTML shell" in packaging
    assert "本说明书的截图嵌入/validator 路径" in packaging
    assert "替换前执行更强的候选语义、资源和图片验证" in packaging
    assert "候选语义、资源和图片验证" in packaging
    assert "os.replace 之前失败时保留原有 standalone 文件" in packaging
    assert "替换后目录 fsync 报错" in packaging
    assert "新文件可能已经活动" in packaging
    assert "校验 standalone 文件摘要" in packaging

    assert "重新读取 CURRENT.json" in troubleshooting
    assert "重新校验 standalone 文件摘要" in troubleshooting


def test_reference_precisely_describes_raw_run_directories():
    reference = _guide_section_text("reference")
    assert "不可变的原始运行目录，可作为后续合并来源" in reference
