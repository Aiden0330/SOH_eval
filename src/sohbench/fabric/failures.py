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


class InputContractError(ValueError):
    """A failure while preparing or validating benchmark inputs."""


class MetricEvaluationError(RuntimeError):
    """A failure raised while calculating benchmark metrics."""


_LEAKAGE_PREFIXES = (
    "leakage detected:",
    "feature pipeline references mutable ",
    "feature pipeline is not prefix invariant",
)
_LEAKAGE_MESSAGES = frozenset({"fold index overlap", "fold asset overlap"})


def _exception_chain(exc: BaseException):
    seen = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _is_leakage(exc: BaseException) -> bool:
    message = str(exc).strip().lower()
    exception_name = type(exc).__name__.lower()
    return (
        exception_name in {"leakageerror", "dataleakageerror"}
        or message in _LEAKAGE_MESSAGES
        or message.startswith(_LEAKAGE_PREFIXES)
    )


def classify_exception(exc: BaseException) -> FailureCategory:
    """Classify known failures without guessing from broad message fragments."""
    chain = tuple(_exception_chain(exc))

    if any(isinstance(error, MemoryError) for error in chain):
        return FailureCategory.RESOURCE
    if any(_is_leakage(error) for error in chain):
        return FailureCategory.LEAKAGE
    for error in chain:
        if isinstance(error, InputContractError):
            return FailureCategory.INPUT_CONTRACT
        if isinstance(error, MetricEvaluationError):
            return FailureCategory.METRIC
    return FailureCategory.ALGORITHM
