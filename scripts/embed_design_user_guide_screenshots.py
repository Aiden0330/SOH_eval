from __future__ import annotations

import argparse
import base64
import os
import re
import tempfile
from pathlib import Path


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
    payload = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{payload}"


def embed_screenshots(html_path: Path, captures: dict[str, Path]) -> None:
    text = html_path.read_text(encoding="utf-8")
    for name, capture in captures.items():
        if not capture.is_file():
            raise FileNotFoundError(capture)
        pattern = re.compile(
            rf'(<img\b(?=[^>]*\bdata-shot="{re.escape(name)}")'
            rf'(?=[^>]*\bsrc=")[^>]*\bsrc=")[^"]*(")',
            flags=re.IGNORECASE,
        )
        text, count = pattern.subn(
            rf"\g<1>{_data_uri(capture)}\2",
            text,
            count=1,
        )
        if count != 1:
            raise ValueError(f"expected one image marker for {name}, found {count}")

    fd, staged_name = tempfile.mkstemp(
        prefix=f".{html_path.name}.",
        suffix=".tmp",
        dir=html_path.parent,
    )
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
