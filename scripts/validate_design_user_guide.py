from __future__ import annotations

import argparse
import base64
import binascii
import re
import struct
import zlib
from collections import Counter
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
EXPECTED_SCREENSHOTS = ("training", "inference", "models", "native", "diagnostics")
MOJIBAKE_MARKERS = ("\ufffd", "Ãƒ", "Ã‚", "Ã¢â€ ", "Ã§â€º", "Ã¦Âµ", "Ã¨Â¯", "锟斤拷")
SUPPORTED_IMAGE_MIMES = frozenset(
    {"image/png", "image/jpeg", "image/webp"}
)
ALLOWED_DOCUMENT_TAGS = frozenset(
    {
        "a",
        "article",
        "aside",
        "body",
        "br",
        "button",
        "code",
        "dd",
        "details",
        "div",
        "dl",
        "dt",
        "figcaption",
        "figure",
        "h1",
        "h2",
        "h3",
        "head",
        "header",
        "html",
        "img",
        "input",
        "label",
        "li",
        "main",
        "meta",
        "nav",
        "ol",
        "p",
        "pre",
        "script",
        "section",
        "span",
        "strong",
        "style",
        "summary",
        "table",
        "tbody",
        "td",
        "th",
        "thead",
        "title",
        "tr",
        "ul",
    }
)
RESOURCE_ELEMENT_TAGS = frozenset(
    {
        "base",
        "embed",
        "form",
        "iframe",
        "link",
        "object",
        "picture",
        "source",
        "track",
        "audio",
        "video",
    }
)
RESOURCE_ATTRIBUTES = frozenset(
    {
        "action",
        "background",
        "data",
        "formaction",
        "href",
        "manifest",
        "poster",
        "src",
        "srcset",
        "xlink:href",
    }
)
ALLOWED_RESOURCE_ATTRIBUTES = frozenset(
    {
        ("a", "href"),
        ("img", "src"),
    }
)
ACTIVE_LOAD_PATTERNS = (
    r"\bfetch\s*\(",
    r"\bXMLHttpRequest\s*\(",
    r"\bWebSocket\s*\(",
    r"\bEventSource\s*\(",
    r"\bnavigator\s*\.\s*sendBeacon\s*\(",
    r"\bimportScripts\s*\(",
    r"(?<![\w.])import\s*\(",
    r"\.\s*(?:src|href)\s*=",
    r"\bsetAttribute\s*\(\s*[\"'](?:src|href|srcset|poster)[\"']",
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ImageInfo:
    mime: str
    width: int
    height: int
    payload: bytes


class ImageResourceError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


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
        self.inline_scripts: list[str] = []
        self.elements: list[tuple[str, dict[str, str | None]]] = []
        self._in_style = False
        self._in_script = False
        self._active_button: dict[str, object] | None = None
        self._button_visibility: list[bool] = []

    @staticmethod
    def _is_aria_hidden(values: dict[str, str | None]) -> bool:
        return str(values.get("aria-hidden") or "").lower() == "true"

    def handle_starttag(self, tag: str, attrs) -> None:
        values = dict(attrs)
        self.elements.append((tag, values))
        if element_id := values.get("id"):
            self.ids.add(element_id)
        if tag == "a" and (href := values.get("href")):
            self.anchors.append(href)
        if tag == "img":
            self.images.append(values)
        if (
            tag == "link"
            and "stylesheet"
            in str(values.get("rel") or "").lower().split()
        ):
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
        if tag == "script":
            self._in_script = True

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.inline_styles.append(data)
        if self._in_script:
            self.inline_scripts.append(data)
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
        if tag == "script":
            self._in_script = False
        if self._active_button is None:
            return
        if tag == "button":
            self._active_button = None
            self._button_visibility = []
        elif len(self._button_visibility) > 1:
            self._button_visibility.pop()


def _invalid_image(code: str, message: str) -> ImageResourceError:
    return ImageResourceError(code, message)


def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _invalid_image("image-data", "PNG signature is missing")

    offset = 8
    width = height = 0
    seen_header = False
    seen_data = False
    seen_end = False
    while offset < len(payload):
        if len(payload) - offset < 12:
            raise _invalid_image("image-data", "PNG chunk is truncated")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        kind = payload[offset + 4 : offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(payload):
            raise _invalid_image("image-data", "PNG chunk payload is truncated")
        data = payload[offset + 8 : offset + 8 + length]
        expected_crc = struct.unpack(
            ">I", payload[offset + 8 + length : chunk_end]
        )[0]
        actual_crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        if expected_crc != actual_crc:
            raise _invalid_image("image-data", "PNG chunk CRC is invalid")

        if not seen_header:
            if kind != b"IHDR" or length != 13:
                raise _invalid_image("image-data", "PNG IHDR is invalid")
            width, height = struct.unpack(">II", data[:8])
            seen_header = True
        elif kind == b"IHDR":
            raise _invalid_image("image-data", "PNG has duplicate IHDR")
        if kind == b"IDAT":
            seen_data = True
        if kind == b"IEND":
            if length != 0 or chunk_end != len(payload):
                raise _invalid_image("image-data", "PNG IEND is invalid")
            seen_end = True
            offset = chunk_end
            break
        offset = chunk_end

    if not (seen_header and seen_data and seen_end) or offset != len(payload):
        raise _invalid_image("image-data", "PNG is incomplete")
    return width, height


def _jpeg_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(b"\xff\xd8") or not payload.endswith(b"\xff\xd9"):
        raise _invalid_image("image-data", "JPEG boundaries are invalid")

    offset = 2
    dimensions = None
    sof_markers = frozenset(
        {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }
    )
    while offset < len(payload) - 2:
        if payload[offset] != 0xFF:
            raise _invalid_image("image-data", "JPEG marker is malformed")
        while offset < len(payload) and payload[offset] == 0xFF:
            offset += 1
        if offset >= len(payload):
            raise _invalid_image("image-data", "JPEG marker is truncated")
        marker = payload[offset]
        offset += 1
        if marker == 0xDA:
            break
        if marker in {0x01, *range(0xD0, 0xD9)}:
            continue
        if offset + 2 > len(payload):
            raise _invalid_image("image-data", "JPEG segment is truncated")
        segment_length = struct.unpack(">H", payload[offset : offset + 2])[0]
        if segment_length < 2 or offset + segment_length > len(payload):
            raise _invalid_image("image-data", "JPEG segment length is invalid")
        if marker in sof_markers:
            if segment_length < 7:
                raise _invalid_image("image-data", "JPEG SOF is truncated")
            height, width = struct.unpack(
                ">HH", payload[offset + 3 : offset + 7]
            )
            dimensions = (width, height)
        offset += segment_length

    if dimensions is None:
        raise _invalid_image("image-data", "JPEG dimensions are missing")
    return dimensions


def _webp_dimensions(payload: bytes) -> tuple[int, int]:
    if (
        len(payload) < 20
        or payload[:4] != b"RIFF"
        or payload[8:12] != b"WEBP"
    ):
        raise _invalid_image("image-data", "WebP header is invalid")
    if struct.unpack("<I", payload[4:8])[0] != len(payload) - 8:
        raise _invalid_image("image-data", "WebP RIFF length is invalid")

    offset = 12
    dimensions = None
    has_image_data = False
    while offset < len(payload):
        if len(payload) - offset < 8:
            raise _invalid_image("image-data", "WebP chunk is truncated")
        kind = payload[offset : offset + 4]
        length = struct.unpack("<I", payload[offset + 4 : offset + 8])[0]
        data_start = offset + 8
        data_end = data_start + length
        padded_end = data_end + (length % 2)
        if padded_end > len(payload):
            raise _invalid_image("image-data", "WebP chunk payload is truncated")
        data = payload[data_start:data_end]

        if kind == b"VP8X":
            if length < 10:
                raise _invalid_image("image-data", "WebP VP8X is truncated")
            width = int.from_bytes(data[4:7], "little") + 1
            height = int.from_bytes(data[7:10], "little") + 1
            dimensions = (width, height)
        elif kind == b"VP8L":
            if length < 5 or data[0] != 0x2F:
                raise _invalid_image("image-data", "WebP VP8L is invalid")
            bits = int.from_bytes(data[1:5], "little")
            dimensions = (
                (bits & 0x3FFF) + 1,
                ((bits >> 14) & 0x3FFF) + 1,
            )
            has_image_data = True
        elif kind == b"VP8 ":
            if length < 10 or data[3:6] != b"\x9d\x01\x2a":
                raise _invalid_image("image-data", "WebP VP8 frame is invalid")
            dimensions = (
                struct.unpack("<H", data[6:8])[0] & 0x3FFF,
                struct.unpack("<H", data[8:10])[0] & 0x3FFF,
            )
            has_image_data = True
        elif kind == b"ANMF":
            has_image_data = True
        offset = padded_end

    if offset != len(payload) or dimensions is None or not has_image_data:
        raise _invalid_image("image-data", "WebP image data is incomplete")
    return dimensions


def inspect_image_bytes(mime: str, payload: bytes) -> ImageInfo:
    normalized_mime = mime.casefold()
    if normalized_mime not in SUPPORTED_IMAGE_MIMES:
        raise _invalid_image("image-mime", f"unsupported image MIME: {mime}")

    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        actual_mime = "image/png"
        dimensions = _png_dimensions(payload)
    elif payload.startswith(b"\xff\xd8"):
        actual_mime = "image/jpeg"
        dimensions = _jpeg_dimensions(payload)
    elif payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
        actual_mime = "image/webp"
        dimensions = _webp_dimensions(payload)
    else:
        raise _invalid_image("image-data", "unsupported or missing image magic")

    if normalized_mime != actual_mime:
        raise _invalid_image(
            "image-mime",
            f"declared {normalized_mime} but payload is {actual_mime}",
        )
    width, height = dimensions
    if width <= 0 or height <= 0:
        raise _invalid_image(
            "image-dimensions",
            f"invalid image dimensions: {width}x{height}",
        )
    return ImageInfo(normalized_mime, width, height, payload)


def decode_image_data_uri(value: str) -> ImageInfo:
    match = re.fullmatch(
        r"data:(image/(?:png|jpeg|webp));base64,"
        r"([A-Za-z0-9+/]*={0,2})",
        value,
        flags=re.IGNORECASE,
    )
    if match is None or not match.group(2):
        raise _invalid_image("image-data", "malformed image data URI")
    encoded = match.group(2)
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise _invalid_image("image-data", "invalid image base64") from exc
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise _invalid_image("image-data", "non-canonical image base64")
    return inspect_image_bytes(match.group(1), payload)


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

    external_resources = []
    unsupported_tags = []
    for tag, values in parser.elements:
        if tag not in ALLOWED_DOCUMENT_TAGS:
            if tag in RESOURCE_ELEMENT_TAGS:
                external_resources.append(f"<{tag}>")
            else:
                unsupported_tags.append(tag)
        for attribute, value in values.items():
            normalized_attribute = attribute.casefold()
            if (
                normalized_attribute in RESOURCE_ATTRIBUTES
                and (tag, normalized_attribute)
                not in ALLOWED_RESOURCE_ATTRIBUTES
            ):
                external_resources.append(
                    f"<{tag} {attribute}={value!r}>"
                )
        if (
            tag == "meta"
            and str(values.get("http-equiv") or "").casefold() == "refresh"
        ):
            external_resources.append("<meta http-equiv='refresh'>")
    if external_resources:
        issues.append(
            ValidationIssue(
                "external-resource",
                "sidecar or active resource reference found: "
                + ", ".join(external_resources),
            )
        )
    if unsupported_tags:
        issues.append(
            ValidationIssue(
                "html-tag",
                "unsupported HTML tag found: "
                + ", ".join(sorted(set(unsupported_tags))),
            )
        )

    inline_css = "\n".join(parser.inline_styles)
    active_css = re.sub(r"/\*.*?\*/", "", inline_css, flags=re.DOTALL)
    css_import = re.search(r"@import\b", active_css, flags=re.IGNORECASE)
    css_urls = re.findall(
        r"url\(\s*(\"[^\"]*\"|'[^']*'|[^)]*)\s*\)",
        active_css,
        flags=re.IGNORECASE,
    )
    external_css_url = any(
        not value.strip().strip("\"'").lower().startswith("data:image/")
        for value in css_urls
    )
    if parser.stylesheets or parser.scripts or css_import or external_css_url:
        issues.append(
            ValidationIssue(
                "external-asset",
                "external stylesheet, script, or CSS resource found",
            )
        )

    inline_javascript = "\n".join(parser.inline_scripts)
    active_loads = [
        pattern
        for pattern in ACTIVE_LOAD_PATTERNS
        if re.search(pattern, inline_javascript, flags=re.IGNORECASE)
    ]
    if active_loads:
        issues.append(
            ValidationIssue(
                "active-load",
                "active JavaScript resource load site found",
            )
        )

    shot_counts = Counter(
        str(image.get("data-shot") or "") for image in parser.images
    )
    expected_shots = Counter(EXPECTED_SCREENSHOTS)
    if shot_counts != expected_shots:
        issues.append(
            ValidationIssue(
                "image-set",
                "embedded screenshot set must be exactly "
                + ", ".join(EXPECTED_SCREENSHOTS),
            )
        )
    for image in parser.images:
        source = str(image.get("src") or "")
        if not source.startswith("data:image/"):
            issues.append(ValidationIssue("image", f"image is not embedded: {image.get('id', '<unknown>')}"))
        else:
            try:
                decode_image_data_uri(source)
            except ImageResourceError as exc:
                issues.append(
                    ValidationIssue(
                        exc.code,
                        f"{image.get('data-shot', '<unknown>')}: {exc}",
                    )
                )
        try:
            width = int(str(image.get("width") or ""))
            height = int(str(image.get("height") or ""))
        except ValueError:
            width = height = 0
        if width <= 0 or height <= 0:
            issues.append(
                ValidationIssue(
                    "image-dimensions",
                    "image width/height attributes must be positive: "
                    f"{image.get('data-shot', '<unknown>')}",
                )
            )
        if not str(image.get("alt") or "").strip():
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
