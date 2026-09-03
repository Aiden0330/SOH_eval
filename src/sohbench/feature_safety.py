"""Compatibility shim: the implementation moved to ``sohkernel.feature_safety`` (ADR-001, extraction plan Task 2).

The workbench keeps importing ``sohbench.feature_safety``; every public name and the names listed
below resolve to the kernel objects, so identities (classes, registries) are shared.
"""
from sohkernel.feature_safety import *  # noqa: F401,F403
from sohkernel.feature_safety import (  # noqa: F401
    FeatureSafetyError,
    LABEL_SAFETY_CONTRACT,
    apply_transform_without_label,
    audit_feature_frame,
    is_label_source_feature,
    passed_label_safety,
    suspicious_feature_names,
)
