from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import re
import struct
import zlib
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
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
SUPPORTED_IMAGE_MIMES = frozenset({"image/png"})
APPROVED_INLINE_SCRIPT_SHA256 = (
    "96540c4320d4e03999c29fdacfe2133257e21c167d290fb3923e908715469b26"
)
APPROVED_INLINE_STYLE_SHA256 = (
    "7bd17690456794cd7b62360858891a3fb0ddbed604af10972b39b0d6c8f6b965"
)
APPROVED_SCREENSHOTS = {
    "training": (
        "image/png",
        1440,
        800,
        "29c9390192c6e10afbbefe7693dd76842e62b95f81bf87a3b3e6b9fb17909329",
    ),
    "inference": (
        "image/png",
        1440,
        800,
        "cdeec39a1195a9bcc900f5addc8652f95fd165d80a9175838eea25e062dc120f",
    ),
    "models": (
        "image/png",
        1440,
        800,
        "3e2643919718896cf815841fcfe2f8f8437ceb2fa4496293ecffb956b88030f0",
    ),
    "native": (
        "image/png",
        1110,
        800,
        "4a2e268d75be07b6f8e8fff8c43faecbc632945123f64ebe8fbf498e9d8f5333",
    ),
    "diagnostics": (
        "image/png",
        1440,
        800,
        "1400e71b599b3c1ce02d8246b09e9e3817248e5351e00b9caec2d7cf1c09e9cb",
    ),
}
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
GLOBAL_ATTRIBUTES = frozenset({"class", "id", "role", "title"})
ALLOWED_ATTRIBUTES_BY_TAG = {
    "a": frozenset({"href"}),
    "article": frozenset({"data-guide-mode"}),
    "aside": frozenset({"aria-label"}),
    "button": frozenset(
        {
            "aria-controls",
            "aria-expanded",
            "aria-label",
            "aria-pressed",
            "data-audience",
            "data-icon-only",
            "type",
        }
    ),
    "dd": frozenset(
        {"data-metric-truth-current", "data-metric-truth-target"}
    ),
    "details": frozenset({"open"}),
    "div": frozenset(
        {
            "aria-atomic",
            "aria-describedby",
            "aria-hidden",
            "aria-label",
            "aria-live",
            "data-capability-state",
            "data-state-lane",
            "data-target-user-confirmation",
        }
    ),
    "html": frozenset({"lang"}),
    "img": frozenset({"alt", "data-shot", "height", "src", "width"}),
    "input": frozenset({"autocomplete", "placeholder", "type"}),
    "label": frozenset({"for"}),
    "li": frozenset(
        {"data-capability-state", "data-target-user-confirmation"}
    ),
    "meta": frozenset({"charset", "content", "name"}),
    "nav": frozenset({"aria-label"}),
    "p": frozenset({"hidden"}),
    "section": frozenset({"aria-labelledby", "data-paths"}),
    "span": frozenset(
        {
            "aria-hidden",
            "data-capability-state",
            "data-current-checks",
            "data-current-fact",
            "data-current-values",
            "data-dataset-id",
            "data-integration-status",
            "data-state-value",
            "data-target-user-confirmation",
        }
    ),
    "tr": frozenset(
        {
            "data-current-columns",
            "data-current-risk-feature",
            "data-feature-policy-status",
            "data-featureset",
        }
    ),
}
ALLOWED_BOOLEAN_ATTRIBUTES = frozenset(
    {
        ("details", "open"),
        ("p", "hidden"),
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


def _first_attribute_values(
    attrs: list[tuple[str, str | None]] | tuple[tuple[str, str | None], ...],
) -> dict[str, str | None]:
    values: dict[str, str | None] = {}
    for name, value in attrs:
        values.setdefault(name.casefold(), value)
    return values


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
        self.images: list[dict[str, str | None]] = []
        self.stylesheets: list[str] = []
        self.scripts: list[str] = []
        self.buttons: list[dict[str, object]] = []
        self.inline_styles: list[str] = []
        self.inline_scripts: list[str] = []
        self.style_bodies: list[str] = []
        self.script_bodies: list[str] = []
        self.elements: list[
            tuple[str, tuple[tuple[str, str | None], ...]]
        ] = []
        self._in_style = False
        self._in_script = False
        self._style_parts: list[str] = []
        self._script_parts: list[str] = []
        self._active_button: dict[str, object] | None = None
        self._button_visibility: list[bool] = []

    @staticmethod
    def _is_aria_hidden(values: dict[str, str | None]) -> bool:
        return str(values.get("aria-hidden") or "").lower() == "true"

    def handle_starttag(self, tag: str, attrs) -> None:
        ordered_attrs = tuple(attrs)
        values = _first_attribute_values(ordered_attrs)
        self.elements.append((tag, ordered_attrs))
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
            self._style_parts = []
        if tag == "script":
            self._in_script = True
            self._script_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_style:
            self.inline_styles.append(data)
            self._style_parts.append(data)
        if self._in_script:
            self.inline_scripts.append(data)
            self._script_parts.append(data)
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
            self.style_bodies.append("".join(self._style_parts))
            self._style_parts = []
        if tag == "script":
            self._in_script = False
            self.script_bodies.append("".join(self._script_parts))
            self._script_parts = []
        if self._active_button is None:
            return
        if tag == "button":
            self._active_button = None
            self._button_visibility = []
        elif len(self._button_visibility) > 1:
            self._button_visibility.pop()


def _invalid_image(code: str, message: str) -> ImageResourceError:
    return ImageResourceError(code, message)


def _validate_png_pixel_stream(
    *,
    width: int,
    height: int,
    bit_depth: int,
    color_type: int,
    compression_method: int,
    filter_method: int,
    interlace_method: int,
    compressed: bytes,
) -> None:
    valid_depths = {
        0: frozenset({1, 2, 4, 8, 16}),
        2: frozenset({8, 16}),
        3: frozenset({1, 2, 4, 8}),
        4: frozenset({8, 16}),
        6: frozenset({8, 16}),
    }
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    if bit_depth not in valid_depths.get(color_type, frozenset()):
        raise _invalid_image("image-data", "PNG color type or bit depth is invalid")
    if compression_method != 0 or filter_method != 0:
        raise _invalid_image("image-data", "PNG compression or filter method is invalid")
    if interlace_method != 0:
        raise _invalid_image("image-data", "interlaced PNG is not approved")

    row_bytes = (width * channels[color_type] * bit_depth + 7) // 8
    expected_size = height * (row_bytes + 1)
    if expected_size <= 0 or expected_size > 64 * 1024 * 1024:
        raise _invalid_image("image-data", "PNG decoded pixel data size is invalid")

    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(compressed, expected_size + 1)
        if decoder.unconsumed_tail:
            raise _invalid_image(
                "image-data",
                "PNG compressed pixel data exceeds declared dimensions",
            )
        decoded += decoder.flush()
    except zlib.error as exc:
        raise _invalid_image(
            "image-data",
            "PNG compressed pixel data is invalid",
        ) from exc

    if (
        not decoder.eof
        or decoder.unused_data
        or len(decoded) != expected_size
    ):
        raise _invalid_image(
            "image-data",
            "PNG compressed pixel data does not match declared dimensions",
        )
    row_offsets = range(0, expected_size, row_bytes + 1)
    if any(decoded[offset] > 4 for offset in row_offsets):
        raise _invalid_image("image-data", "PNG row filter is invalid")


@lru_cache(maxsize=32)
def _png_dimensions(payload: bytes) -> tuple[int, int]:
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise _invalid_image("image-data", "PNG signature is missing")

    offset = 8
    width = height = 0
    bit_depth = color_type = -1
    compression_method = filter_method = interlace_method = -1
    image_data: list[bytes] = []
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
            (
                bit_depth,
                color_type,
                compression_method,
                filter_method,
                interlace_method,
            ) = data[8:13]
            seen_header = True
        elif kind == b"IHDR":
            raise _invalid_image("image-data", "PNG has duplicate IHDR")
        if kind == b"IDAT":
            seen_data = True
            image_data.append(data)
        if kind == b"IEND":
            if length != 0 or chunk_end != len(payload):
                raise _invalid_image("image-data", "PNG IEND is invalid")
            seen_end = True
            offset = chunk_end
            break
        offset = chunk_end

    if not (seen_header and seen_data and seen_end) or offset != len(payload):
        raise _invalid_image("image-data", "PNG is incomplete")
    _validate_png_pixel_stream(
        width=width,
        height=height,
        bit_depth=bit_depth,
        color_type=color_type,
        compression_method=compression_method,
        filter_method=filter_method,
        interlace_method=interlace_method,
        compressed=b"".join(image_data),
    )
    return width, height


def inspect_image_bytes(mime: str, payload: bytes) -> ImageInfo:
    normalized_mime = mime.casefold()
    if normalized_mime not in SUPPORTED_IMAGE_MIMES:
        raise _invalid_image("image-mime", f"unsupported image MIME: {mime}")

    if payload.startswith(b"\x89PNG\r\n\x1a\n"):
        actual_mime = "image/png"
        dimensions = _png_dimensions(payload)
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
        r"data:(image/[A-Za-z0-9.+-]+);base64,"
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

    duplicate_attributes = []
    unsafe_attributes = []
    external_resources = []
    unsupported_tags = []
    for tag, ordered_attrs in parser.elements:
        values = _first_attribute_values(ordered_attrs)
        attribute_counts = Counter(
            name.casefold() for name, _ in ordered_attrs
        )
        duplicate_attributes.extend(
            f"<{tag} {name}>"
            for name, count in attribute_counts.items()
            if count > 1
        )
        allowed_attributes = ALLOWED_ATTRIBUTES_BY_TAG.get(tag, frozenset())
        for attribute, value in ordered_attrs:
            normalized_attribute = attribute.casefold()
            if (
                normalized_attribute.startswith("on")
                or (
                    normalized_attribute not in GLOBAL_ATTRIBUTES
                    and normalized_attribute not in allowed_attributes
                )
                or (
                    value is None
                    and (tag, normalized_attribute)
                    not in ALLOWED_BOOLEAN_ATTRIBUTES
                )
            ):
                unsafe_attributes.append(
                    f"<{tag} {attribute}={value!r}>"
                )
        if tag not in ALLOWED_DOCUMENT_TAGS:
            if tag in RESOURCE_ELEMENT_TAGS:
                external_resources.append(f"<{tag}>")
            else:
                unsupported_tags.append(tag)
        for attribute, value in ordered_attrs:
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
    if duplicate_attributes:
        issues.append(
            ValidationIssue(
                "duplicate-attribute",
                "duplicate HTML attribute found: "
                + ", ".join(duplicate_attributes),
            )
        )
    if unsafe_attributes:
        issues.append(
            ValidationIssue(
                "html-attribute",
                "unsupported or malformed HTML attribute found: "
                + ", ".join(unsafe_attributes),
            )
        )
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

    inline_css = "\n".join(parser.style_bodies)
    style_digest = (
        hashlib.sha256(parser.style_bodies[0].encode("utf-8")).hexdigest()
        if len(parser.style_bodies) == 1
        else None
    )
    if (
        len(parser.style_bodies) != 1
        or style_digest != APPROVED_INLINE_STYLE_SHA256
    ):
        issues.append(
            ValidationIssue(
                "inline-style",
                "inline style must match the approved single-style digest",
            )
        )
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

    inline_javascript = "\n".join(parser.script_bodies)
    script_digest = (
        hashlib.sha256(parser.script_bodies[0].encode("utf-8")).hexdigest()
        if len(parser.script_bodies) == 1
        else None
    )
    if (
        len(parser.script_bodies) != 1
        or script_digest != APPROVED_INLINE_SCRIPT_SHA256
    ):
        issues.append(
            ValidationIssue(
                "inline-script",
                "inline script must match the approved single-script digest",
            )
        )
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
        shot = str(image.get("data-shot") or "")
        source = str(image.get("src") or "")
        image_info = None
        if not source.startswith("data:image/"):
            issues.append(ValidationIssue("image", f"image is not embedded: {image.get('id', '<unknown>')}"))
        else:
            try:
                image_info = decode_image_data_uri(source)
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
        elif image_info is not None and (
            width != image_info.width or height != image_info.height
        ):
            issues.append(
                ValidationIssue(
                    "image-dimensions",
                    "declared image dimensions do not match payload: "
                    f"{shot}",
                )
            )
        if image_info is not None:
            actual_approval = (
                image_info.mime,
                image_info.width,
                image_info.height,
                hashlib.sha256(image_info.payload).hexdigest(),
            )
            if APPROVED_SCREENSHOTS.get(shot) != actual_approval:
                issues.append(
                    ValidationIssue(
                        "image-approval",
                        f"embedded screenshot is not approved: {shot}",
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
