from __future__ import annotations
import numpy as np

def _arr(*xs): return [np.asarray(x, dtype=float) for x in xs]

def rmse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.sqrt(np.mean((yt - yp) ** 2)))

def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.mean(np.abs(yt - yp)))

def max_error(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.max(np.abs(yt - yp)))

def mape(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])))

def mean_bias_error(y_true, y_pred) -> float:
    """Signed mean (pred - true). Positive => systematic over-estimation."""
    yt, yp = _arr(y_true, y_pred); return float(np.mean(yp - yt))

def overestimation_rate(y_true, y_pred) -> float:
    yt, yp = _arr(y_true, y_pred); return float(np.mean(yp > yt))

def alpha_accuracy(y_true, y_pred, alpha: float = 0.15) -> float:
    yt, yp = _arr(y_true, y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask]) <= alpha))

def eol_region_error(y_true, y_pred, eol_threshold: float = 0.8) -> float:
    yt, yp = _arr(y_true, y_pred)
    sel = yt <= eol_threshold
    if not np.any(sel): return float("nan")
    return float(np.sqrt(np.mean((yt[sel] - yp[sel]) ** 2)))

def picp(y_true, lower, upper) -> float:
    yt, lo, hi = _arr(y_true, lower, upper)
    return float(np.mean((yt >= lo) & (yt <= hi)))

def monotonicity_violation_rate(y_pred_ordered, rebound_tol: float = 0.005) -> float:
    yp = np.asarray(y_pred_ordered, dtype=float)
    if len(yp) < 2: return 0.0
    diffs = np.diff(yp)
    return float(np.mean(diffs > rebound_tol))

def per_cell_monotonicity_violation_rate(cells, cycles, y_pred,
                                         rebound_tol: float = 0.005) -> float:
    """Sample-weighted mean of per-cell monotonicity violation rates.

    A global cycle sort interleaves every cell at each cycle index, so on a
    multi-cell fold (LODO / LCHO test sets, whole-dataset inference) the
    cross-cell level jumps would swamp the physical within-cell signal.
    Each cell is ordered by its own cycle index and weighted by its number
    of transitions; cells with fewer than two finite predictions are skipped.
    """
    cells = np.asarray(cells).astype(str)
    cycles = np.asarray(cycles, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    weighted = 0.0
    transitions = 0
    for cell in np.unique(cells):
        rows = np.flatnonzero(cells == cell)
        if len(rows) < 2:
            continue
        ordered = yp[rows[np.argsort(cycles[rows], kind="stable")]]
        finite = ordered[np.isfinite(ordered)]
        if len(finite) < 2:
            continue
        weighted += monotonicity_violation_rate(finite, rebound_tol) * (len(finite) - 1)
        transitions += len(finite) - 1
    return float(weighted / transitions) if transitions else 0.0

def value_range_violation(y_pred, lower: float = 0.0, upper: float = 1.3) -> float:
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean((yp < lower) | (yp >= upper)))
