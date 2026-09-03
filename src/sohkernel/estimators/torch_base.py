"""Shared PyTorch infrastructure for windowed sequence estimators.

Provides:
  - ``set_seed``       : reproducible numpy + torch seeding.
  - ``build_windows``  : (N, W, F) windows aligned to X's row order, grouped
                         per (dataset, cell_id), ordered by cycle_index,
                         left-padded with the cell's first cycle vector.
  - ``subsample_idx``  : cap training size; return all indices if n <= n_max.
  - ``train_mlp_like`` : generic Adam+MSE training loop with early stopping on
                         a tail validation split (works for LSTM/CNN/Transformer).
  - ``clamp``          : np.clip to [SOH_LOW, SOH_HIGH].
"""
from __future__ import annotations

import copy
from typing import Sequence

import numpy as np
import pandas as pd

# Import torch lazily so the module itself can be imported in test discovery
# even without torch (tests gate with pytest.importorskip).
try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TORCH_AVAILABLE = False

from .baselines import SOH_LOW, SOH_HIGH


# ---------------------------------------------------------------------------
# Seed
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Set numpy and torch RNG seeds for reproducibility."""
    np.random.seed(seed)
    if _TORCH_AVAILABLE:
        torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Window builder
# ---------------------------------------------------------------------------

def build_windows(
    X: pd.DataFrame,
    feature_cols: list[str],
    W: int,
) -> np.ndarray:
    """Build (N, W, F) windows aligned to X's original row order.

    For each row (cycle *t* of a cell) the window is the last W cycle-feature
    vectors ``[t-W+1 .. t]`` of **that same cell**, left-padded by repeating
    the cell's earliest available vector when fewer than W prior cycles exist.

    Windows are built per ``(dataset, cell_id)`` group ordered by
    ``cycle_index``.  They never span two cells.

    Parameters
    ----------
    X:
        DataFrame containing at least ``dataset``, ``cell_id``,
        ``cycle_index`` and the columns in ``feature_cols``.
    feature_cols:
        Feature columns to include in the window (F = len(feature_cols)).
    W:
        Window length (number of cycles).

    Returns
    -------
    np.ndarray
        Shape ``(len(X), W, F)``, row order matches X's original row order.
    """
    F = len(feature_cols)
    N = len(X)
    out = np.empty((N, W, F), dtype=np.float32)

    # Work on a reset-index copy so we can map back to original positions.
    # Keep the original integer positions in a column.
    df = X.reset_index(drop=True)
    df["_orig_pos"] = np.arange(N)

    for _, grp in df.groupby(["dataset", "cell_id"], sort=False):
        # Sort within the cell by cycle_index
        grp_sorted = grp.sort_values("cycle_index")
        feats = grp_sorted[feature_cols].to_numpy(dtype=np.float32)  # (C, F)
        orig_positions = grp_sorted["_orig_pos"].to_numpy()           # (C,)
        C = len(feats)

        for local_t in range(C):
            # Window indices in the sorted cell sequence: [local_t-W+1 .. local_t]
            # Negative indices are padded with feats[0].
            window = np.empty((W, F), dtype=np.float32)
            for w_i in range(W):
                src = local_t - (W - 1 - w_i)  # can be negative
                window[w_i] = feats[max(src, 0)]
            out[orig_positions[local_t]] = window

    return out


# ---------------------------------------------------------------------------
# Subsample
# ---------------------------------------------------------------------------

def subsample_idx(n: int, n_max: int, seed: int) -> np.ndarray:
    """Return indices capping training size to *n_max*.

    If ``n <= n_max`` returns ``np.arange(n)``.  Otherwise returns *n_max*
    unique random indices chosen without replacement.
    """
    if n <= n_max:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.choice(n, n_max, replace=False)


# ---------------------------------------------------------------------------
# Generic training loop
# ---------------------------------------------------------------------------

def train_mlp_like(
    model: "nn.Module",
    Xt: "torch.Tensor",
    yt: "torch.Tensor",
    *,
    epochs: int,
    lr: float = 1e-3,
    val_frac: float = 0.15,
    patience: int = 8,
    batch: int = 256,
) -> "nn.Module":
    """Train *model* with Adam + MSE and return the best-val-loss state.

    Validation split is carved from the **tail** of the given arrays (no
    shuffle across the boundary), so temporal ordering is preserved for
    sequence models.

    Parameters
    ----------
    model:
        Any ``nn.Module`` that accepts *Xt* and produces a ``(N,)`` or
        ``(N, 1)`` tensor.
    Xt:
        Input tensor, shape ``(N, W, F)`` or ``(N, F)``.
    yt:
        Target tensor, shape ``(N,)``.
    epochs:
        Maximum training epochs.
    lr:
        Adam learning rate.
    val_frac:
        Fraction of data held out for early stopping (tail split).
    patience:
        Early-stopping patience (epochs without improvement).
    batch:
        Mini-batch size.

    Returns
    -------
    nn.Module
        The model loaded with its best-validation-loss state dict.
    """
    import torch
    import torch.nn as nn

    N = len(yt)
    n_val = max(1, int(N * val_frac))
    n_train = N - n_val

    X_tr, y_tr = Xt[:n_train], yt[:n_train]
    X_val, y_val = Xt[n_train:], yt[n_train:]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    no_improve = 0

    n_tr = len(y_tr)
    for epoch in range(epochs):
        model.train()
        # Mini-batch loop (sequential order; no reshuffle within train split)
        perm = torch.randperm(n_tr)
        for start in range(0, n_tr, batch):
            idx = perm[start: start + batch]
            xb, yb = X_tr[idx], y_tr[idx]
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss_fn(pred, yb).backward()
            optimizer.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val).squeeze(-1)
            val_loss = loss_fn(val_pred, y_val).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Output clamp
# ---------------------------------------------------------------------------

def clamp(arr: np.ndarray) -> np.ndarray:
    """Clip predictions to the physical SOH range ``[SOH_LOW, SOH_HIGH]``."""
    return np.clip(arr, SOH_LOW, SOH_HIGH)
