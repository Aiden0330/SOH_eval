"""Compatibility shim: the implementation moved to ``sohkernel.estimators.torch_base`` (ADR-001, Task 4).

Every module-level name resolves to the kernel object, so registrations, class identities
and pickled artifacts that name ``sohbench.estimators.torch_base`` keep working.
"""
from sohkernel.estimators.torch_base import *  # noqa: F401,F403
from sohkernel.estimators.torch_base import (  # noqa: F401
    copy,
    Sequence,
    np,
    pd,
    SOH_LOW,
    SOH_HIGH,
    set_seed,
    build_windows,
    subsample_idx,
    train_mlp_like,
    clamp,
)
