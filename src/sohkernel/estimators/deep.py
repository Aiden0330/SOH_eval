"""Deep-learning sequence estimators for sohbench.

Implements three windowed-sequence estimators that operate on per-cell
cycle-feature windows (see ``torch_base.build_windows``):

  - ``LSTMEstimator``        (name="lstm")
  - ``CNN1DEstimator``       (name="cnn1d")
  - ``TransformerEstimator`` (name="transformer")

All three share a common base (``_DeepBase``) that handles window
construction, StandardScaler normalisation, training-size subsampling,
the Adam+MSE training loop with early stopping, and output clamping to
the physical SOH range ``[0.0, 1.3]``.

Gate imports: ``import torch`` is deferred to fit/predict so the module
can be imported (and the registry decorated) even without PyTorch;
tests gate with ``pytest.importorskip("torch")``.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ..contracts import feature_columns
from ..preprocessing import FeatureMedianImputer
from ..registry import register_estimator
from .baselines import SOH_LOW, SOH_HIGH
from .torch_base import (
    build_windows,
    clamp,
    set_seed,
    subsample_idx,
    train_mlp_like,
)

__all__ = ["LSTMEstimator", "CNN1DEstimator", "TransformerEstimator"]


# ---------------------------------------------------------------------------
# Shared base
# ---------------------------------------------------------------------------

class _DeepBase:
    """Common fit/predict logic for all windowed DL estimators.

    Subclasses must implement ``_build_net(F: int) -> nn.Module`` and set
    ``name``, ``feature_mode``, ``backend`` as class attributes.
    """

    feature_mode: str = "unified"
    backend: str = "pytorch"
    method_family: str = "sequence_learning"
    pairing_mode: str = "recommended"
    preferred_featureset: str = "full"
    data_granularity: str = "cycle_sequence"

    def __init__(self, W: int = 8, n_max: int = 20_000, seed: int = 0,
                 epochs: int = 30, lr: float = 1e-3) -> None:
        self.W = W
        self.n_max = n_max
        self.seed = seed
        self.epochs = epochs
        self.lr = lr

        self._scaler: StandardScaler = StandardScaler()
        self._imputer = FeatureMedianImputer()
        self._feature_cols: Optional[list[str]] = None
        self._net = None  # nn.Module, set after fit

    def set_seed(self, seed: int) -> None:
        """Evaluation seed for numpy/torch seeding and the training subsample."""
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Subclass hook
    # ------------------------------------------------------------------

    def _build_net(self, F: int):  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: pd.DataFrame, y: np.ndarray, meta=None) -> "_DeepBase":
        import torch

        set_seed(self.seed)

        self._feature_cols = feature_columns(X)
        F = len(self._feature_cols)
        X_work = self._imputer.fit(X, self._feature_cols).transform_frame(X)

        # Build (N, W, F) windows aligned to X's row order
        wins = build_windows(X_work, self._feature_cols, self.W)  # (N, W, F)
        N = len(wins)

        # Fit StandardScaler on the 2D view (N*W, F)
        wins_2d = wins.reshape(-1, F)
        wins_2d_scaled = self._scaler.fit_transform(wins_2d).astype(np.float32)
        wins_scaled = wins_2d_scaled.reshape(N, self.W, F)

        y_arr = np.asarray(y, dtype=np.float32)

        # Subsample training data
        idx = subsample_idx(N, self.n_max, self.seed)
        wins_sub = wins_scaled[idx]
        y_sub = y_arr[idx]

        # Convert to tensors
        Xt = torch.from_numpy(wins_sub)   # (n_sub, W, F)
        yt = torch.from_numpy(y_sub)       # (n_sub,)

        # Build and train the network
        self._net = self._build_net(F)
        train_mlp_like(
            self._net, Xt, yt,
            epochs=self.epochs,
            lr=self.lr,
            val_frac=0.15,
            patience=8,
            batch=256,
        )
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame, meta=None) -> np.ndarray:
        import torch

        if self._net is None or self._feature_cols is None:
            raise RuntimeError("Call fit() before predict().")

        F = len(self._feature_cols)
        X_work = self._imputer.transform_frame(X)
        wins = build_windows(X_work, self._feature_cols, self.W)  # (N, W, F)
        N = len(wins)

        wins_2d = wins.reshape(-1, F)
        wins_2d_scaled = self._scaler.transform(wins_2d).astype(np.float32)
        wins_scaled = wins_2d_scaled.reshape(N, self.W, F)

        Xt = torch.from_numpy(wins_scaled)
        self._net.eval()
        with torch.no_grad():
            preds = self._net(Xt).squeeze(-1).numpy()

        return clamp(preds)


# ---------------------------------------------------------------------------
# LSTM
# ---------------------------------------------------------------------------

@register_estimator
class LSTMEstimator(_DeepBase):
    """LSTM over the cycle window → last hidden → linear → SOH.

    Architecture: ``nn.LSTM(F, hidden=48, num_layers=1, batch_first=True)``
    → last timestep hidden state → ``nn.Linear(48, 1)``.
    """

    name: str = "lstm"

    def __init__(self, W: int = 8, hidden: int = 48, n_max: int = 20_000,
                 seed: int = 0, epochs: int = 30, lr: float = 1e-3) -> None:
        super().__init__(W=W, n_max=n_max, seed=seed, epochs=epochs, lr=lr)
        self.hidden = hidden

    def _build_net(self, F: int):
        import torch.nn as nn

        class _LSTM(nn.Module):
            def __init__(self, in_dim: int, hidden: int) -> None:
                super().__init__()
                self.lstm = nn.LSTM(in_dim, hidden, num_layers=1, batch_first=True)
                self.head = nn.Linear(hidden, 1)

            def forward(self, x):
                # x: (B, W, F)
                _, (hn, _) = self.lstm(x)  # hn: (1, B, hidden)
                return self.head(hn.squeeze(0))  # (B, 1)

        return _LSTM(F, self.hidden)


# ---------------------------------------------------------------------------
# CNN1D
# ---------------------------------------------------------------------------

@register_estimator
class CNN1DEstimator(_DeepBase):
    """1D-conv stack over the cycle window → adaptive avg pool → linear → SOH.

    Treats the window as ``(B, F, W)`` (channels-first); two Conv1d+ReLU
    layers followed by ``AdaptiveAvgPool1d(1)`` → ``Linear``.
    """

    name: str = "cnn1d"

    def __init__(self, W: int = 8, channels: int = 32, n_max: int = 20_000,
                 seed: int = 0, epochs: int = 30, lr: float = 1e-3) -> None:
        super().__init__(W=W, n_max=n_max, seed=seed, epochs=epochs, lr=lr)
        self.channels = channels

    def _build_net(self, F: int):
        import torch.nn as nn

        ch = self.channels

        class _CNN1D(nn.Module):
            def __init__(self, in_dim: int, ch: int) -> None:
                super().__init__()
                self.conv = nn.Sequential(
                    nn.Conv1d(in_dim, ch, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(ch, ch, kernel_size=3, padding=1), nn.ReLU(),
                )
                self.pool = nn.AdaptiveAvgPool1d(1)
                self.head = nn.Linear(ch, 1)

            def forward(self, x):
                # x: (B, W, F) → (B, F, W) for conv
                x = x.permute(0, 2, 1)
                x = self.conv(x)       # (B, ch, W)
                x = self.pool(x)       # (B, ch, 1)
                x = x.squeeze(-1)      # (B, ch)
                return self.head(x)    # (B, 1)

        return _CNN1D(F, ch)


# ---------------------------------------------------------------------------
# Transformer
# ---------------------------------------------------------------------------

@register_estimator
class TransformerEstimator(_DeepBase):
    """Transformer encoder over the cycle window → mean-pool → linear → SOH.

    Architecture:
      - Linear projection from F → d_model=48
      - Sinusoidal positional encoding over W timesteps
      - ``nn.TransformerEncoder`` (1 layer, nhead=4, batch_first=True)
      - Mean-pool over W → ``nn.Linear(48, 1)``
    """

    name: str = "transformer"

    def __init__(self, W: int = 8, d_model: int = 48, nhead: int = 4,
                 n_max: int = 20_000, seed: int = 0, epochs: int = 30,
                 lr: float = 1e-3) -> None:
        super().__init__(W=W, n_max=n_max, seed=seed, epochs=epochs, lr=lr)
        self.d_model = d_model
        self.nhead = nhead

    def _build_net(self, F: int):
        import torch
        import torch.nn as nn

        d_model = self.d_model
        nhead = self.nhead
        W = self.W

        class _Transformer(nn.Module):
            def __init__(self, in_dim: int, d_model: int, nhead: int, W: int) -> None:
                super().__init__()
                self.proj = nn.Linear(in_dim, d_model)
                # Sinusoidal positional encoding — register as buffer (not a param)
                pe = torch.zeros(W, d_model)
                position = torch.arange(0, W, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(
                    torch.arange(0, d_model, 2, dtype=torch.float)
                    * (-math.log(10000.0) / d_model)
                )
                pe[:, 0::2] = torch.sin(position * div_term)
                if d_model % 2 == 0:
                    pe[:, 1::2] = torch.cos(position * div_term)
                else:
                    pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
                self.register_buffer("pe", pe.unsqueeze(0))  # (1, W, d_model)

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
                    batch_first=True, dropout=0.0,
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
                self.head = nn.Linear(d_model, 1)

            def forward(self, x):
                # x: (B, W, F)
                x = self.proj(x) + self.pe  # (B, W, d_model)
                x = self.encoder(x)          # (B, W, d_model)
                x = x.mean(dim=1)            # (B, d_model)
                return self.head(x)          # (B, 1)

        return _Transformer(F, d_model, nhead, W)
