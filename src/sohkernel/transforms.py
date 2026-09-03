"""Featureset transforms to replay before inference-only evaluation.

Moved from the workbench's ``evaluation_service`` (ADR-001, extraction plan Task 5) so that
``adaptation`` — which needs the same rule to rebuild a frozen model's source feature
statistics — depends only on the kernel.
"""
from __future__ import annotations

from . import registry as R


def inference_transform_featuresets(model_bindings: list[dict]) -> list[str]:
    """Featureset transforms to re-apply before inference-only evaluation.

    Unified models re-apply their own featureset's transform.  BYO models
    self-select features at training time and carry featureset ``"(byo)"``,
    which names no transform — re-apply the training request's featuresets
    when the manifest recorded them (``training_featuresets``); for legacy
    artifacts without the field, fall back to every transform-bearing
    featureset whose candidate columns intersect the model's recorded
    feature columns.
    """
    names: list[str] = []
    for item in model_bindings:
        featureset = str(item.get("featureset") or "")
        recorded = [str(x) for x in (item.get("training_featuresets") or []) if str(x)]
        if recorded:
            names.extend(recorded)
        elif featureset == "(byo)":
            needed = {str(column) for column in item.get("feature_columns") or []}
            for name, registered in sorted(R._FEATURESETS.items()):
                if getattr(registered, "transform", None) is None:
                    continue
                candidates = {str(c) for c in getattr(registered, "candidates", []) or []}
                if needed & candidates:
                    names.append(name)
        elif featureset:
            names.append(featureset)
    return list(dict.fromkeys(names))
