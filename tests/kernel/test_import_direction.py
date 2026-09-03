"""Direction rule (ADR-001 §Decision 4): ``sohbench -> sohkernel <- sohagent``.

``sohkernel`` never imports the frozen workbench or the agent platform — neither
through an ``import`` statement nor through a string handed to ``importlib``.
This test walks every kernel source file with ``ast`` and must never be skipped.
"""
from __future__ import annotations

import ast
from pathlib import Path

KERNEL_ROOT = Path(__file__).resolve().parents[2] / "src" / "sohkernel"
FORBIDDEN = {"sohbench", "sohagent"}


def _forbidden_imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN:
                    hits.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] in FORBIDDEN:
                hits.append(f"line {node.lineno}: from {node.module} import ...")
        elif isinstance(node, ast.Call):
            func = node.func
            name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
            if name in {"import_module", "__import__"} and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    if first.value.split(".")[0] in FORBIDDEN:
                        hits.append(f"line {node.lineno}: {name}({first.value!r})")
    return hits


def test_kernel_package_is_present():
    assert (KERNEL_ROOT / "__init__.py").is_file(), "src/sohkernel/__init__.py missing"


def test_kernel_never_imports_workbench_or_platform():
    offenders = {}
    for path in sorted(KERNEL_ROOT.rglob("*.py")):
        hits = _forbidden_imports(path)
        if hits:
            offenders[str(path.relative_to(KERNEL_ROOT.parent))] = hits
    assert not offenders, (
        "sohkernel must not import sohbench or sohagent (ADR-001 direction rule):\n"
        + "\n".join(f"  {k}: {v}" for k, v in offenders.items())
    )


def test_kernel_version_is_declared():
    import sohkernel

    assert sohkernel.__version__ == "0.1.0.dev0"
