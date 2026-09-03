from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable


METHOD_CATEGORIES = {
    "free_combination",
    "recommended_preset",
    "joint_method",
}

PAIRING_TO_CATEGORY = {
    "free": "free_combination",
    "recommended": "recommended_preset",
    "method_package": "joint_method",
    "embedded": "joint_method",
}

PAIRING_TO_POLICY = {
    "free": "cartesian",
    "recommended": "recommended_default",
    "method_package": "joint_package",
    "embedded": "embedded_pipeline",
}


@dataclass(frozen=True)
class MethodContract:
    """Execution and pairing semantics shared by catalog, preflight and runner.

    Estimators keep the existing ``fit/predict/predict_interval`` API.  This
    contract describes *how* that executable should enter an experiment:

    - ``free_combination``: ordinary feature x estimator Cartesian cells.
    - ``recommended_preset``: still executable through the estimator API, but
      the catalog advertises a scientifically preferred feature profile.
    - ``joint_method``: feature extraction, priors and/or the solver form one
      method package; incompatible Cartesian cells must not be scheduled.
    """

    name: str
    method_category: str
    method_family: str
    pairing_mode: str
    combination_policy: str
    execution_contract: str
    feature_mode: str
    backend: str
    compatible_featuresets: tuple[str, ...]
    preferred_featureset: str
    data_granularity: str
    required_signals: tuple[str, ...]
    physics_constraints: tuple[str, ...]
    description: str

    @property
    def is_joint_method(self) -> bool:
        return self.method_category == "joint_method"

    def accepts_featureset(self, featureset: str) -> bool:
        if self.feature_mode == "byo":
            return featureset == "(byo)"
        if not self.compatible_featuresets:
            return True
        return featureset in self.compatible_featuresets

    def feature_iterations(self, requested: Iterable[str]) -> list[str]:
        if self.feature_mode == "byo":
            return ["(byo)"]
        accepted = [str(name) for name in requested if self.accepts_featureset(str(name))]
        if accepted or not self.is_joint_method or not self.preferred_featureset:
            return accepted
        if self.accepts_featureset(self.preferred_featureset):
            return [self.preferred_featureset]
        return accepted

    def catalog_metadata(self) -> dict:
        payload = asdict(self)
        payload.pop("name", None)
        payload["compatible_featuresets"] = list(self.compatible_featuresets)
        payload["required_signals"] = list(self.required_signals)
        payload["physics_constraints"] = list(self.physics_constraints)
        return payload


def method_contract(estimator: type | object) -> MethodContract:
    """Resolve a backward-compatible method contract from an estimator class."""

    feature_mode = str(getattr(estimator, "feature_mode", "unified"))
    default_pairing = "embedded" if feature_mode == "byo" else "free"
    pairing_mode = str(getattr(estimator, "pairing_mode", default_pairing))
    method_category = str(
        getattr(
            estimator,
            "method_category",
            PAIRING_TO_CATEGORY.get(pairing_mode, "free_combination"),
        )
    )
    if method_category not in METHOD_CATEGORIES:
        method_category = "free_combination"
    default_execution_contract = (
        "embedded_pipeline/v1"
        if pairing_mode == "embedded"
        else ("joint_method/v1" if method_category == "joint_method" else "estimator/v1")
    )
    return MethodContract(
        name=str(getattr(estimator, "name", estimator.__class__.__name__)),
        method_category=method_category,
        method_family=str(getattr(estimator, "method_family", "data_driven")),
        pairing_mode=pairing_mode,
        combination_policy=str(
            getattr(
                estimator,
                "combination_policy",
                PAIRING_TO_POLICY.get(pairing_mode, "cartesian"),
            )
        ),
        execution_contract=str(
            getattr(estimator, "execution_contract", default_execution_contract)
        ),
        feature_mode=feature_mode,
        backend=str(getattr(estimator, "backend", "external")),
        compatible_featuresets=tuple(
            str(item)
            for item in (getattr(estimator, "compatible_featuresets", ()) or ())
        ),
        preferred_featureset=str(getattr(estimator, "preferred_featureset", "")),
        data_granularity=str(getattr(estimator, "data_granularity", "cycle_table")),
        required_signals=tuple(
            str(item) for item in (getattr(estimator, "required_signals", ()) or ())
        ),
        physics_constraints=tuple(
            str(item) for item in (getattr(estimator, "physics_constraints", ()) or ())
        ),
        description=str(getattr(estimator, "description", "")),
    )


def incompatible_pairing_reason(contract: MethodContract, featureset: str) -> str:
    return (
        f"{contract.name} uses {contract.combination_policy} and does not accept "
        f"feature set {featureset}"
    )
