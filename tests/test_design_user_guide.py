import ast
import base64
import re
from html.parser import HTMLParser
from pathlib import Path

from sohbench.api_server import parse_args
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


def _guide_section_text(section_id):
    parser = SectionTextParser(section_id)
    parser.feed(GUIDE.read_text(encoding="utf-8"))
    return parser.text()


def _guide_section_code_blocks(section_id):
    parser = SectionCodeBlockParser(section_id)
    parser.feed(GUIDE.read_text(encoding="utf-8"))
    return parser.blocks


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


def test_design_user_guide_embeds_all_five_frontend_captures():
    text = GUIDE.read_text(encoding="utf-8")
    for shot in ("training", "inference", "models", "native", "diagnostics"):
        assert f'data-shot="{shot}"' in text
    payloads = re.findall(r'src="data:image/[^;]+;base64,([^"]+)"', text)
    assert len(payloads) >= 5
    assert all(len(base64.b64decode(payload)) >= 10_000 for payload in payloads)


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
    assert "只有后续目录发布成功才创建不可变 generation 并推进 CURRENT.json" in runs
    assert "目录发布前失败不创建 generation" in runs

    assert "原始运行目录与前端 publication generation 是两类对象" in evidence
    assert "原始运行目录不会推进 CURRENT.json" in evidence

    assert "generation 和 CURRENT.json 只属于目录发布" in packaging
    assert "只有 staging 构建和校验成功后才创建不可变 generation" in packaging
    assert "失败在激活前不创建 generation，也不改变 CURRENT.json" in packaging
    assert "Standalone 不创建 generation，也不读写 CURRENT.json" in packaging
    assert "先验证临时 HTML，再原子替换一个输出文件" in packaging
    assert "失败时保留原有 standalone 文件" in packaging


def test_reference_precisely_describes_raw_run_directories():
    reference = _guide_section_text("reference")
    assert "不可变的原始运行目录，可作为后续合并来源" in reference
