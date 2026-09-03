"""sohkernel — the durable SOH evaluation kernel extracted from ``sohbench`` (ADR-001).

Data-first: feature tables in, JSON / Parquet manifests out.  It never depends on
the workbench (``sohbench``), the agent platform (``sohagent``), FastAPI or the
frontend; ``tests/kernel/test_import_direction.py`` enforces the direction rule.

Modules move here one task at a time — see
``docs/superpowers/plans/2026-09-02-kernel-extraction.md`` and the invariants in
``docs/superpowers/specs/2026-09-02-soh-kernel-invariants.md``.
"""

__version__ = "0.1.0"
