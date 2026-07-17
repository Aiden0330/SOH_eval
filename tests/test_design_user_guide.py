from html.parser import HTMLParser
from pathlib import Path

from scripts.validate_design_user_guide import REQUIRED_SECTIONS, validate_document


ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html"


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


def test_design_user_guide_satisfies_offline_structure_contract():
    issues = validate_document(GUIDE, source_root=ROOT)
    assert issues == []


def test_design_user_guide_contains_every_required_chapter():
    text = GUIDE.read_text(encoding="utf-8")
    for section_id in REQUIRED_SECTIONS:
        assert f'id="{section_id}"' in text


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
    text = GUIDE.read_text(encoding="utf-8")
    assert "python.exe -m sohbench.api_server" in text
    assert "python.exe -m http.server 8010 --directory frontend/app" in text
    assert "scripts\\package_portable_frontend.py" in text
    assert "scripts\\build_standalone_frontend.py" in text


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
