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
MOJIBAKE_MARKERS = ("\ufffd", "Ãƒ", "Ã‚", "Ã¢â€ ", "Ã§â€º", "Ã¦Âµ", "Ã¨Â¯", "锟斤拷")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


class GuideParser(HTMLParser):
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

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.ids: set[str] = set()
        self.anchors: list[str] = []
        self.images: list[dict[str, str]] = []
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []
        self.buttons: list[dict[str, object]] = []
        self.inline_styles: list[str] = []
        self._in_style = False
        self._active_button: dict[str, object] | None = None
        self._button_visibility: list[bool] = []

    @staticmethod
    def _is_aria_hidden(values: dict[str, str | None]) -> bool:
        return str(values.get("aria-hidden") or "").lower() == "true"

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        if element_id := values.get("id"):
            self.ids.add(element_id)
        if tag == "a" and (href := values.get("href")):
            self.anchors.append(href)
        if tag == "img":
            self.images.append(values)
        if tag == "link" and "stylesheet" in values.get("rel", "").lower().split():
            self.stylesheets.append(values.get("href", ""))
        if tag == "script" and values.get("src"):
            self.scripts.append(values["src"])
        if tag == "button":
            button: dict[str, object] = {
                "attributes": values,
                "text_parts": [],
            }
            self.buttons.append(button)
            self._active_button = button
            self._button_visibility = [not self._is_aria_hidden(values)]
        elif self._active_button is not None and tag not in self._VOID_ELEMENTS:
            parent_visible = self._button_visibility[-1]
            self._button_visibility.append(
                parent_visible and not self._is_aria_hidden(values)
            )
        if tag == "style":
            self._in_style = True

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.inline_styles.append(data)
        if (
            self._active_button is not None
            and self._button_visibility
            and self._button_visibility[-1]
        ):
            text_parts = self._active_button["text_parts"]
            assert isinstance(text_parts, list)
            text_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._in_style = False
        if self._active_button is None:
            return
        if tag == "button":
            self._active_button = None
            self._button_visibility = []
        elif len(self._button_visibility) > 1:
            self._button_visibility.pop()


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
    inline_css = "\n".join(parser.inline_styles)
    active_css = re.sub(r"/\*.*?\*/", "", inline_css, flags=re.DOTALL)
    css_import = re.search(r"@import\b", active_css, flags=re.IGNORECASE)
    css_urls = re.findall(
        r"url\(\s*(\"[^\"]*\"|'[^']*'|[^)]*)\s*\)",
        active_css,
        flags=re.IGNORECASE,
    )
    external_css_url = any(
        not value.strip().strip("\"'").lower().startswith("data:")
        for value in css_urls
    )
    if parser.stylesheets or parser.scripts or css_import or external_css_url:
        issues.append(
            ValidationIssue(
                "external-asset",
                "external stylesheet, script, or CSS resource found",
            )
        )
    for image in parser.images:
        if not image.get("src", "").startswith("data:image/"):
            issues.append(ValidationIssue("image", f"image is not embedded: {image.get('id', '<unknown>')}"))
        if not image.get("alt", "").strip():
            issues.append(ValidationIssue("image-alt", f"image has no alt text: {image.get('id', '<unknown>')}"))
    for button in parser.buttons:
        attributes = button["attributes"]
        text_parts = button["text_parts"]
        assert isinstance(attributes, dict)
        assert isinstance(text_parts, list)
        aria_label = str(attributes.get("aria-label") or "").strip()
        visible_text = " ".join("".join(text_parts).split())
        if not aria_label and not visible_text:
            issues.append(
                ValidationIssue("button-name", "button has no accessible name")
            )
    required_media_rules = (
        r"@media\s*\(\s*max-width\s*:\s*1100px\s*\)",
        r"@media\s*\(\s*max-width\s*:\s*720px\s*\)",
        r"@media\s+print\b",
    )
    if any(
        re.search(rule, active_css, flags=re.IGNORECASE) is None
        for rule in required_media_rules
    ):
        issues.append(
            ValidationIssue(
                "responsive",
                "1100px, mobile, or print responsive rules missing",
            )
        )
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
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=ROOT / "docs" / "soh-evaluation-system-design-and-user-guide.html",
    )
    args = parser.parse_args(argv)
    issues = validate_document(args.path)
    for issue in issues:
        print(f"{issue.code}: {issue.message}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
