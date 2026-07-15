from __future__ import annotations

from enum import Enum


class FailureCategory(str, Enum):
    INPUT_CONTRACT = "INPUT_CONTRACT"
    DATA_QC = "DATA_QC"
    LEAKAGE = "LEAKAGE"
    ALGORITHM = "ALGORITHM"
    RESOURCE = "RESOURCE"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    METRIC = "METRIC"
    POLICY = "POLICY"


_LEAKAGE_PREFIXES = (
    "leakage detected:",
    "feature pipeline references mutable ",
    "feature pipeline is not prefix invariant",
)
_LEAKAGE_MESSAGES = frozenset({"fold index overlap", "fold asset overlap"})


def classify_exception(exc: BaseException) -> FailureCategory:
    """Classify known failures without guessing from broad message fragments."""
    message = str(exc).strip().lower()
    exception_name = type(exc).__name__.lower()

    if isinstance(exc, MemoryError):
        return FailureCategory.RESOURCE
    if exception_name in {"leakageerror", "dataleakageerror"}:
        return FailureCategory.LEAKAGE
    if message in _LEAKAGE_MESSAGES or message.startswith(_LEAKAGE_PREFIXES):
        return FailureCategory.LEAKAGE
    if isinstance(exc, (TypeError, ValueError)):
        return FailureCategory.INPUT_CONTRACT
    return FailureCategory.ALGORITHM
