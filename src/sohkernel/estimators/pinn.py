"""Physics-informed neural network estimator for SOH.

``PINNEstimator`` is the Phase-2 PINN comparison entry. It is a small
PyTorch MLP over per-cycle feature vectors with a training loss:

    data MSE + lambda_mono * monotonicity penalty + lambda_range * range penalty

The monotonicity term penalizes SOH rises larger than ``rebound_tol`` within
each cell trajectory ordered by ``cycle_index``. Predictions are clamped to
the physical SOH range and can optionally be projected to the same monotone
constraint per cell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..contracts import feature_columns
from ..preprocessing import FeatureMedianImputer
from ..registry import register_estimator
from .baselines import SOH_HIGH, SOH_LOW
from .torch_base import clamp, set_seed, subsample_idx

__all__ = ["PINNEstimator"]


def _trajectory_pairs(X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent row-index pairs within each cell, ordered by cycle."""
    prev: list[int] = []
    nxt: list[int] = []
    df = X.reset_index(drop=True)
    for _, grp in df.groupby(["dataset", "cell_id"], sort=False):
        order = grp.sort_values("cycle_index").index.to_numpy()
        if len(order) < 2:
            continue
        prev.extend(order[:-1].tolist())
        nxt.extend(order[1:].tolist())
    return np.asarray(prev, dtype=np.int64), np.asarray(nxt, dtype=np.int64)


@register_estimator
class PINNEstimator:
    """MLP with physics-informed monotonicity and range penalties."""

    name: str = "pinn"
    feature_mode: str = "unified"
    backend: str = "pytorch"
    method_family: str = "physics_regularized_neural"
    pairing_mode: str = "method_package"
    compatible_featuresets = (
        "degradation_kinetics",
        "operating_stress",
        "electrochemical_diagnostics",
        "capacity_time",
        "full",
    )
    preferred_featureset: str = "degradation_kinetics"
    data_granularity: str = "cycle_sequence"
    required_signals = ("cycle_index",)
    physics_constraints = ("trajectory_monotonicity", "soh_range")
    description: str = (
        "逐 Cycle MLP 加单调性和范围约束；属于物理正则网络，"
        "不等同于含电化学控制方程残差的严格 PINN。"
    )

    def __init__(
        self,
        hidden: int = 48,
        epochs: int = 80,
        lr: float = 1e-3,
        seed: int = 0,
        n_max: int = 10_000,
        lambda_mono: float = 10.0,
        lambda_range: float = 1.0,
        rebound_tol: float = 0.005,
        enforce_monotone: bool = True,
    ) -> None:
        self.hidden = hidden
        self.epochs = epochs
        self.lr = lr
        self.seed = seed
        self.n_max = n_max
        self.lambda_mono = lambda_mono
        self.lambda_range = lambda_range
        self.rebound_tol = rebound_tol
        self.enforce_monotone = enforce_monotone

        self._cols: list[str] | None = None
        self._scaler = StandardScaler()
        self._imputer = FeatureMedianImputer()
        self._net = None

    def _build_net(self, n_features: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(n_features, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, 1),
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta=None) -> "PINNEstimator":
        import torch
        import torch.nn.functional as F

        set_seed(self.seed)
        self._cols = feature_columns(X)
        if not self._cols:
            raise ValueError("PINNEstimator requires at least one feature column.")

        self._imputer.fit(X, self._cols)
        idx = subsample_idx(len(X), self.n_max, self.seed)
        X_sub = self._imputer.transform_frame(X.iloc[idx].reset_index(drop=True))
        y_sub = np.asarray(y, dtype=np.float32)[idx]

        X_arr = X_sub[self._cols].to_numpy(dtype=np.float32)
        X_scaled = self._scaler.fit_transform(X_arr).astype(np.float32)

        Xt = torch.from_numpy(X_scaled)
        yt = torch.from_numpy(y_sub)
        prev_idx, next_idx = _trajectory_pairs(X_sub)
        prev_t = torch.from_numpy(prev_idx)
        next_t = torch.from_numpy(next_idx)

        self._net = self._build_net(len(self._cols))
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        for _ in range(self.epochs):
            self._net.train()
            opt.zero_grad()
            pred = self._net(Xt).squeeze(-1)
            loss = F.mse_loss(pred, yt)

            if self.lambda_mono > 0 and len(prev_idx):
                rises = pred[next_t] - pred[prev_t] - self.rebound_tol
                loss = loss + self.lambda_mono * torch.relu(rises).pow(2).mean()

            if self.lambda_range > 0:
                below = torch.relu(torch.tensor(SOH_LOW, dtype=pred.dtype) - pred)
                above = torch.relu(pred - torch.tensor(SOH_HIGH, dtype=pred.dtype))
                loss = loss + self.lambda_range * (below.pow(2).mean() + above.pow(2).mean())

            loss.backward()
            opt.step()

        return self

    def predict(self, X: pd.DataFrame, meta=None) -> np.ndarray:
        import torch

        if self._net is None or self._cols is None:
            raise RuntimeError("Call fit() before predict().")

        X_arr = self._imputer.transform_array(X).astype(np.float32)
        X_scaled = self._scaler.transform(X_arr).astype(np.float32)
        Xt = torch.from_numpy(X_scaled)

        self._net.eval()
        with torch.no_grad():
            pred = self._net(Xt).squeeze(-1).numpy()

        pred = clamp(pred)
        if self.enforce_monotone:
            pred = self._project_monotone(X, pred)
        return clamp(pred)

    def _project_monotone(self, X: pd.DataFrame, pred: np.ndarray) -> np.ndarray:
        """Apply a per-cell cumulative cap allowing only rebound_tol rises."""
        out = np.asarray(pred, dtype=float).copy()
        df = X.reset_index(drop=True)
        for _, grp in df.groupby(["dataset", "cell_id"], sort=False):
            order = grp.sort_values("cycle_index").index.to_numpy()
            for left, right in zip(order[:-1], order[1:]):
                out[right] = min(out[right], out[left] + self.rebound_tol)
        return out
