from __future__ import annotations

from sohbench.estimators.baselines import RidgeEstimator
from sohbench.estimators.byo import ByoSelectKBestRidge
from sohbench.estimators.mechanism import PLSEstimator, StressResidualEstimator
from sohbench.estimators.physics import PhysicsResidualEstimator
from sohbench.method_contracts import method_contract


def test_free_combination_contract_accepts_the_selected_feature_matrix():
    contract = method_contract(RidgeEstimator)

    assert contract.method_category == "free_combination"
    assert contract.combination_policy == "cartesian"
    assert contract.execution_contract == "estimator/v1"
    assert contract.feature_iterations(["full", "voltage", "ic_dva"]) == [
        "full",
        "voltage",
        "ic_dva",
    ]


def test_recommended_preset_keeps_experiment_freedom_but_exposes_a_default():
    contract = method_contract(PLSEstimator)

    assert contract.method_category == "recommended_preset"
    assert contract.combination_policy == "recommended_default"
    assert contract.preferred_featureset == "electrochemical_diagnostics"
    assert contract.accepts_featureset("full") is True


def test_joint_method_filters_incompatible_cartesian_cells():
    contract = method_contract(StressResidualEstimator)

    assert contract.method_category == "joint_method"
    assert contract.execution_contract == "joint_method/v1"
    assert contract.is_joint_method is True
    assert contract.feature_iterations(
        ["voltage", "operating_stress", "degradation_kinetics", "full"]
    ) == ["operating_stress", "degradation_kinetics", "full"]


def test_joint_method_falls_back_to_its_preferred_package_when_selection_is_incompatible():
    contract = method_contract(PhysicsResidualEstimator)

    assert contract.feature_iterations(["voltage"]) == ["degradation_kinetics"]


def test_embedded_pipeline_is_a_joint_method_with_one_pseudo_featureset():
    contract = method_contract(ByoSelectKBestRidge)

    assert contract.method_category == "joint_method"
    assert contract.combination_policy == "embedded_pipeline"
    assert contract.execution_contract == "embedded_pipeline/v1"
    assert contract.feature_iterations(["full", "voltage"]) == ["(byo)"]
