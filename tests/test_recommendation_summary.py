from scripts.build_recommendation_summary import (
    build_recommendation_summary,
    render_markdown_report,
)


def test_build_recommendation_summary_combines_lodo_and_loco_results():
    lodo = {
        "train_job_id": "train-lodo",
        "inference_job_id": "infer-lodo",
        "datasets": ["nasa", "hust"],
        "featureset": "battery_soh_operational",
        "protocol": "lodo",
        "inference_by_algorithm": [
            {"algorithm": "ridge", "records": 2, "succeeded": 2, "mean_mae": 0.12, "mean_rmse": 0.14, "mean_ms_per_1000": 0.8},
            {"algorithm": "gbdt", "records": 2, "succeeded": 2, "mean_mae": 0.09, "mean_rmse": 0.11, "mean_ms_per_1000": 5.0},
            {"algorithm": "battery_soh_tabular_ensemble", "records": 2, "succeeded": 2, "mean_mae": 0.10, "mean_rmse": 0.105, "mean_ms_per_1000": 30.0},
        ],
        "train_by_algorithm": [
            {"algorithm": "battery_soh_tabular_ensemble", "mae_mean": 0.13, "rmse_mean": 0.16},
            {"algorithm": "gbdt", "mae_mean": 0.14, "rmse_mean": 0.17},
        ],
    }
    loco = {
        "train_job_id": "train-loco",
        "inference_job_id": "infer-loco",
        "datasets": ["nasa", "hust"],
        "featureset": "battery_soh_operational",
        "protocol": "loco",
        "folds": 10,
        "inference_records": 20,
        "inference_by_algorithm": [
            {"algorithm": "battery_soh_tabular_ensemble", "records": 10, "succeeded": 10, "mean_mae": 0.085, "mean_rmse": 0.107, "mean_ms_per_1000": 45.0},
            {"algorithm": "gbdt", "records": 10, "succeeded": 10, "mean_mae": 0.080, "mean_rmse": 0.106, "mean_ms_per_1000": 8.5},
        ],
        "train_by_algorithm": [
            {"algorithm": "gbdt", "mae_mean": 0.074, "rmse_mean": 0.088},
            {"algorithm": "battery_soh_tabular_ensemble", "mae_mean": 0.077, "rmse_mean": 0.089},
        ],
    }

    summary = build_recommendation_summary(lodo, loco)

    assert summary["overall"]["recommended_default"] == "gbdt"
    assert [item["id"] for item in summary["scenarios"]] == [
        "lodo_transfer",
        "loco_cell_generalization",
    ]
    assert summary["scenarios"][0]["accuracy_winner"] == "gbdt"
    assert summary["scenarios"][0]["speed_winner"] == "ridge"
    assert summary["scenarios"][0]["training_winner"] == "battery_soh_tabular_ensemble"
    assert summary["scenarios"][0]["success_rate"] == 1.0
    assert [row["algorithm"] for row in summary["scenarios"][0]["ranking"]] == [
        "gbdt",
        "battery_soh_tabular_ensemble",
        "ridge",
    ]
    assert summary["scenarios"][1]["accuracy_winner"] == "gbdt"
    assert summary["scenarios"][1]["speed_winner"] == "gbdt"
    assert summary["scenarios"][1]["folds"] == 10

    report = render_markdown_report(summary)
    assert "算法推荐报告" in report
    assert "跨数据集迁移能力" in report
    assert "同数据集跨电芯泛化能力" in report
    assert "gbdt" in report
