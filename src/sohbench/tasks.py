from __future__ import annotations
from dataclasses import dataclass, field
from .registry import register_task

@register_task
@dataclass
class SOHEstimationTask:
    name: str = "soh_estimation"
    target: str = "soh"
    point_metrics: tuple = ("rmse","mae","mape","mean_bias_error","overestimation_rate",
                            "max_error","alpha_accuracy","eol_region_error")
    trajectory_metrics: tuple = ("monotonicity_violation_rate","value_range_violation")
    interval_metrics: tuple = ("picp",)
    acceptance: dict = field(default_factory=lambda: {
        "generalization": {"mae_max": 0.02},
        "calibration": {"picp_min": 0.90, "nominal": 0.90},
        "physical_consistency": {"monotonicity_violation_rate_max": 0.001,
                                 "value_range_violation_max": 0.0},
    })
