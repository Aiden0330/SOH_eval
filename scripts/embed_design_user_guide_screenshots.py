from __future__ import annotations

import argparse
import base64
import re
from pathlib import Path

from sohbench.frontend_publish import publish_text_file_atomic

try:
    from scripts.validate_design_user_guide import (
        ImageResourceError,
        inspect_image_bytes,
        validate_document,
    )
except ModuleNotFoundError:  # Direct execution from the scripts directory.
    from validate_design_user_guide import (  # type: ignore[no-redef]
        ImageResourceError,
        inspect_image_bytes,
        validate_document,
    )


MIME_BY_SUFFIX = {
    ".png": "image/png",
    ".webp": "image/webp",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}
SHOT_NAMES = ("training", "inference", "models", "native", "diagnostics")


def _data_uri(path: Path) -> str:
    mime = MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported image type: {path.suffix}")
    raw = path.read_bytes()
    try:
        inspect_image_bytes(mime, raw)
    except ImageResourceError as exc:
        raise ValueError(f"invalid image {path}: {exc}") from exc
    payload = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{payload}"


def embed_screenshots(html_path: Path, captures: dict[str, Path]) -> None:
    text = html_path.read_text(encoding="utf-8")
    replacements = []
    for name, capture in captures.items():
        if not capture.is_file():
            raise FileNotFoundError(capture)
        pattern = re.compile(
            rf'(<img\b(?=[^>]*\bdata-shot="{re.escape(name)}")'
            rf'(?=[^>]*\bsrc=")[^>]*\bsrc=")[^"]*(")',
            flags=re.IGNORECASE,
        )
        count = len(pattern.findall(text))
        if count != 1:
            raise ValueError(f"expected one image marker for {name}, found {count}")
        replacements.append((pattern, _data_uri(capture)))

    for pattern, data_uri in replacements:
        text = pattern.sub(rf"\g<1>{data_uri}\2", text, count=1)

    def validate_candidate(candidate: Path) -> None:
        issues = validate_document(candidate)
        if issues:
            details = "; ".join(
                f"{issue.code}: {issue.message}" for issue in issues
            )
            raise ValueError(
                f"candidate guide validation failed: {details}"
            )

    publish_text_file_atomic(
        html_path,
        text,
        validate=validate_candidate,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser("embed SOH guide screenshots")
    parser.add_argument("html", type=Path)
    for name in SHOT_NAMES:
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args(argv)
    embed_screenshots(
        args.html,
        {name: getattr(args, name) for name in SHOT_NAMES},
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
