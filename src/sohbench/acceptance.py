from __future__ import annotations
from .contracts import Verdict

def evaluate_verdicts(metrics: dict, task) -> list[Verdict]:
    a = task.acceptance; out = []

    g_ok = metrics.get("mae", float("inf")) <= a["generalization"]["mae_max"]
    out.append(Verdict("generalization", g_ok,
                       {"mae": metrics.get("mae")},
                       [] if g_ok else [f"mae {metrics.get('mae')} > {a['generalization']['mae_max']}"]))

    c_ok = metrics.get("picp", 0.0) >= a["calibration"]["picp_min"]
    out.append(Verdict("calibration", c_ok, {"picp": metrics.get("picp")},
                       [] if c_ok else [f"picp {metrics.get('picp')} < {a['calibration']['picp_min']}"]))

    p = a["physical_consistency"]
    p_ok = (metrics.get("monotonicity_violation_rate", 1.0) <= p["monotonicity_violation_rate_max"]
            and metrics.get("value_range_violation", 1.0) <= p["value_range_violation_max"])
    out.append(Verdict("physical_consistency", p_ok,
                       {k: metrics.get(k) for k in ("monotonicity_violation_rate","value_range_violation")},
                       [] if p_ok else ["physical-consistency thresholds exceeded"]))
    return out
