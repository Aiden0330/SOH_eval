from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LODO = ROOT / "runs" / "20260701T192741" / "algorithm_comparison_summary.json"
DEFAULT_LOCO = ROOT / "runs" / "20260701T203543" / "loco_algorithm_comparison_summary.json"
DEFAULT_FRONTEND = ROOT / "frontend" / "app" / "data" / "recommendation_summary.json"
DEFAULT_EXPORT_JSON = ROOT / "docs" / "exports" / "algorithm-recommendation-summary-20260701.json"
DEFAULT_EXPORT_MD = ROOT / "docs" / "exports" / "algorithm-recommendation-report.md"


def build_recommendation_summary(lodo: dict[str, Any], loco: dict[str, Any]) -> dict[str, Any]:
    scenarios = [
        scenario_from_payload(
            lodo,
            "lodo_transfer",
            "跨数据集迁移能力",
            "模型在未见过数据源上的迁移精度与速度。",
        ),
        scenario_from_payload(
            loco,
            "loco_cell_generalization",
            "同数据集跨电芯泛化能力",
            "模型在同一数据源不同电芯上的泛化精度与速度。",
        ),
    ]
    default_algorithm = choose_default_algorithm(scenarios)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "datasets": sorted(set(lodo.get("datasets", [])) | set(loco.get("datasets", []))),
            "featureset": lodo.get("featureset") or loco.get("featureset") or "",
            "protocols": [scenario["protocol"] for scenario in scenarios],
        },
        "overall": {
            "recommended_default": default_algorithm,
            "why": overall_reasons(scenarios, default_algorithm),
            "caveats": overall_caveats(scenarios, default_algorithm),
        },
        "scenarios": scenarios,
    }


def scenario_from_payload(payload: dict[str, Any], scenario_id: str, title: str, description: str) -> dict[str, Any]:
    inference_rows = sorted(
        [dict(row) for row in payload.get("inference_by_algorithm", [])],
        key=lambda row: numeric(row.get("mean_mae"), float("inf")),
    )
    train_rows = sorted(
        [dict(row) for row in payload.get("train_by_algorithm", [])],
        key=lambda row: numeric(row.get("mae_mean"), float("inf")),
    )
    records = sum(int(row.get("records") or 0) for row in inference_rows)
    succeeded = sum(int(row.get("succeeded") or 0) for row in inference_rows)
    accuracy_winner = inference_rows[0]["algorithm"] if inference_rows else "暂无"
    speed_winner = min(
        inference_rows,
        key=lambda row: numeric(row.get("mean_ms_per_1000"), float("inf")),
        default={},
    ).get("algorithm", "暂无")
    training_winner = train_rows[0]["algorithm"] if train_rows else "暂无"
    return {
        "id": scenario_id,
        "title": title,
        "description": description,
        "protocol": payload.get("protocol", ""),
        "datasets": payload.get("datasets", []),
        "featureset": payload.get("featureset", ""),
        "train_job_id": payload.get("train_job_id", ""),
        "inference_job_id": payload.get("inference_job_id", ""),
        "folds": payload.get("folds"),
        "records": records,
        "succeeded": succeeded,
        "success_rate": succeeded / records if records else None,
        "recommended_algorithm": accuracy_winner,
        "accuracy_winner": accuracy_winner,
        "speed_winner": speed_winner,
        "training_winner": training_winner,
        "ranking": inference_rows,
        "training_ranking": train_rows,
        "recommendation": recommendation_text(title, accuracy_winner, speed_winner, training_winner),
    }


def choose_default_algorithm(scenarios: list[dict[str, Any]]) -> str:
    scores: dict[str, dict[str, float]] = {}
    for scenario in scenarios:
        for rank, row in enumerate(scenario.get("ranking", []), start=1):
            name = str(row.get("algorithm") or "")
            if not name:
                continue
            entry = scores.setdefault(name, {"wins": 0, "rank_sum": 0, "mae_sum": 0, "count": 0})
            entry["wins"] += 1 if name == scenario.get("accuracy_winner") else 0
            entry["rank_sum"] += rank
            entry["mae_sum"] += numeric(row.get("mean_mae"), 1)
            entry["count"] += 1
    if not scores:
        return "暂无"
    return min(
        scores.items(),
        key=lambda item: (-item[1]["wins"], item[1]["rank_sum"] / item[1]["count"], item[1]["mae_sum"] / item[1]["count"], item[0]),
    )[0]


def overall_reasons(scenarios: list[dict[str, Any]], algorithm: str) -> list[str]:
    reasons = []
    for scenario in scenarios:
        if scenario.get("accuracy_winner") == algorithm:
            reasons.append(f"{scenario['title']}中 {algorithm} 的平均 MAE 最低。")
    if not reasons and algorithm != "暂无":
        reasons.append(f"{algorithm} 在可比较场景中综合排名最靠前。")
    return reasons


def overall_caveats(scenarios: list[dict[str, Any]], algorithm: str) -> list[str]:
    caveats = []
    for scenario in scenarios:
        if scenario.get("speed_winner") and scenario.get("speed_winner") != algorithm:
            caveats.append(f"{scenario['title']}的最快推理模型是 {scenario['speed_winner']}，不是默认推荐 {algorithm}。")
        if scenario.get("training_winner") and scenario.get("training_winner") != algorithm:
            caveats.append(f"{scenario['title']}的训练过程均值最佳是 {scenario['training_winner']}，推理推荐仍以迁移结果为准。")
    return caveats


def recommendation_text(title: str, accuracy_winner: str, speed_winner: str, training_winner: str) -> str:
    parts = [f"{title}优先使用 {accuracy_winner}。"]
    if speed_winner != accuracy_winner:
        parts.append(f"若只看推理速度，可对比 {speed_winner}。")
    if training_winner != accuracy_winner:
        parts.append(f"训练过程最优为 {training_winner}，需与推理迁移结果分开解释。")
    return "".join(parts)


def render_markdown_report(summary: dict[str, Any]) -> str:
    lines = [
        "# 算法推荐报告",
        "",
        f"- 默认推荐算法：`{summary['overall']['recommended_default']}`",
        f"- 数据集范围：{', '.join(summary['scope'].get('datasets', []))}",
        f"- 特征集：`{summary['scope'].get('featureset', '')}`",
        "",
        "## 推荐理由",
    ]
    lines.extend(f"- {item}" for item in summary["overall"].get("why", []))
    if summary["overall"].get("caveats"):
        lines.extend(["", "## 注意事项"])
        lines.extend(f"- {item}" for item in summary["overall"]["caveats"])
    for scenario in summary.get("scenarios", []):
        lines.extend([
            "",
            f"## {scenario['title']}",
            "",
            f"- 协议：`{scenario['protocol']}`",
            f"- 成功记录：{scenario['succeeded']}/{scenario['records']}",
            f"- 精度推荐：`{scenario['accuracy_winner']}`",
            f"- 速度最快：`{scenario['speed_winner']}`",
            f"- 训练过程最佳：`{scenario['training_winner']}`",
            "",
            "| 排名 | 算法 | MAE | RMSE | ms/1000 |",
            "| --- | --- | ---: | ---: | ---: |",
        ])
        for index, row in enumerate(scenario.get("ranking", []), start=1):
            lines.append(
                f"| {index} | `{row.get('algorithm', '')}` | {fmt(row.get('mean_mae'))} | "
                f"{fmt(row.get('mean_rmse'))} | {fmt(row.get('mean_ms_per_1000'))} |"
            )
    lines.append("")
    return "\n".join(lines)


def numeric(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def fmt(value: Any) -> str:
    number = numeric(value, float("nan"))
    if number != number:
        return ""
    return f"{number:.6f}".rstrip("0").rstrip(".")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build algorithm recommendation artifacts from LODO/LOCO comparison summaries.")
    parser.add_argument("--lodo", type=Path, default=DEFAULT_LODO)
    parser.add_argument("--loco", type=Path, default=DEFAULT_LOCO)
    parser.add_argument("--frontend-output", type=Path, default=DEFAULT_FRONTEND)
    parser.add_argument("--export-json", type=Path, default=DEFAULT_EXPORT_JSON)
    parser.add_argument("--export-md", type=Path, default=DEFAULT_EXPORT_MD)
    args = parser.parse_args()

    summary = build_recommendation_summary(read_json(args.lodo), read_json(args.loco))
    write_json(args.frontend_output, summary)
    write_json(args.export_json, summary)
    args.export_md.parent.mkdir(parents=True, exist_ok=True)
    args.export_md.write_text(render_markdown_report(summary), encoding="utf-8", newline="\n")
    print(f"Wrote {args.frontend_output}")
    print(f"Wrote {args.export_json}")
    print(f"Wrote {args.export_md}")


if __name__ == "__main__":
    main()
