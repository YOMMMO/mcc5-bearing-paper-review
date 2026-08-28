"""Build the publication-facing consistency audit for the current release."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
TITLE = (
    "Recording-Grouped Bearing Fault Diagnosis under Directional "
    "Operating-Condition Shifts on the MCC5 Electric-Drive Benchmark"
)
VERSION = "3.1.4"


def text(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "evidence")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []

    def add(name: str, condition: bool, detail: str) -> None:
        checks.append(
            {
                "check": name,
                "status": "PASS" if condition else "FAIL",
                "detail": detail,
            }
        )

    manuscript_files = ["paper/main.tex", "paper/main_machines.tex"]
    public_tex = manuscript_files + ["paper/supplementary.tex"]
    for relative in manuscript_files:
        add(f"title_{Path(relative).stem}", TITLE in text(relative), TITLE)

    readme = text("README.md")
    add("neutral_repository_title", "Software Supplement" in readme, "README uses neutral software-supplement identity")
    add("no_public_review_prompt", not (ROOT / "review/GPT_PRO_REVIEW_PROMPT.md").exists(), "no prompt file is distributed")
    add("no_gpt_marketing_in_readme", "GPT Pro" not in readme, "README contains no model-specific review instructions")
    add("release_version_readme", f"review-v{VERSION}" in readme, f"review-v{VERSION}")
    add("release_version_citation", f"version: {VERSION}" in text("CITATION.cff"), VERSION)
    add("release_version_pyproject", f'version = "{VERSION}"' in text("pyproject.toml"), VERSION)

    forbidden = [
        "must be confirmed before submission",
        "must be regenerated",
        "during this review",
        "earlier 0.9020 row",
        "historical 28-scalar setting",
        "E:\\thesis2",
    ]
    violations = [
        f"{relative}:{phrase}"
        for relative in public_tex
        for phrase in forbidden
        if phrase in text(relative)
    ]
    add("no_internal_reminders_in_manuscripts", not violations, "; ".join(violations) or "none")

    for relative in manuscript_files:
        content = text(relative)
        add(
            f"ai_disclosure_{Path(relative).stem}",
            all(
                phrase in content
                for phrase in [
                    "OpenAI ChatGPT Pro",
                    "OpenAI Codex",
                    "did not generate or alter the raw measurements",
                    "takes full responsibility",
                ]
            ),
            "specific tools, scope, verification, and author responsibility are disclosed",
        )

    abstract = text("paper/main.tex").split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
    add("abstract_excludes_posthoc_auxiliary", "auxiliary-26" not in abstract and "0.9162" not in abstract, "post-hoc winner absent")
    add("bootstrap_boundary_wording", "conditional on the fitted model and finite test-recording set" in text("paper/main.tex"), "future-specimen uncertainty is not claimed")

    catalog = pd.read_csv(ROOT / "data/recording_catalog_and_splits.csv")
    add("recording_catalog_count", len(catalog) == 84, f"rows={len(catalog)}")
    split = pd.read_csv(ROOT / "evidence/split.csv")
    source_protocol = split.loc[split["split_name"] == "source_file"].iloc[0]
    source_protocol_windows = int(
        source_protocol["train_window_count"]
        + source_protocol["validation_window_count"]
        + source_protocol["test_window_count"]
    )
    add("window_count", source_protocol_windows == 15036, f"windows={source_protocol_windows}")

    required = [
        "evidence/partition_leakage_controls_summary.csv",
        "evidence/classical_source_recording_model_matrix_metrics.csv",
        "evidence/classical_source_recording_model_matrix_bootstrap.csv",
        "evidence/repeated_source_recording_metrics.csv",
        "evidence/within_date_250707_summary.csv",
        "evidence/training_only_metadata_baselines.csv",
        "evidence/outer_race_date_summary.csv",
    ]
    missing = [relative for relative in required if not (ROOT / relative).exists()]
    add("new_control_artifacts_present", not missing, ", ".join(missing) or "all present")

    matrix = pd.read_csv(ROOT / "evidence/classical_source_recording_model_matrix_metrics.csv")
    combinations = matrix[["split", "model"]].drop_duplicates()
    add("complete_classical_matrix", len(combinations) == 20 and len(matrix) == 40, f"model-protocol pairs={len(combinations)}; rows={len(matrix)}")
    high_speed = matrix[(matrix["split"] == "cross_rpm") & (matrix["aggregation"] == "mean_probability")]
    add("high_speed_complete_range", round(high_speed["macro_f1"].min(), 4) == 0.1585 and round(high_speed["macro_f1"].max(), 4) == 0.7020, f"range={high_speed['macro_f1'].min():.4f}-{high_speed['macro_f1'].max():.4f}")

    repeated = pd.read_csv(ROOT / "evidence/repeated_source_recording_metrics.csv")
    add("repeated_recording_matrix", set(repeated["split_seed"]) == set(range(42, 52)) and set(repeated["model"]) == {"random_forest", "xgboost"}, f"rows={len(repeated)}")

    partitions = pd.read_csv(ROOT / "evidence/partition_leakage_controls_summary.csv").set_index("protocol")
    expected_protocols = {
        "random_overlapping_windows",
        "random_nonoverlapping_windows",
        "recording_grouped_overlapping_windows",
        "recording_grouped_nonoverlapping_windows",
    }
    add("partition_control_protocols", set(partitions.index) == expected_protocols, f"protocols={len(partitions)}")
    random_overlap = partitions.loc["random_overlapping_windows"]
    grouped_overlap = partitions.loc["recording_grouped_overlapping_windows"]
    add("partition_source_overlap_contrast", float(random_overlap["source_recording_overlap_mean"]) == 84.0 and float(grouped_overlap["source_recording_overlap_mean"]) == 0.0, "random=84; grouped=0")
    add("partition_macro_f1_values", round(float(random_overlap["window_macro_f1_mean"]), 4) == 0.9976 and round(float(grouped_overlap["window_macro_f1_mean"]), 4) == 0.9777, "random=0.9976; grouped=0.9777")

    same_date = pd.read_csv(ROOT / "evidence/within_date_250707_summary.csv")
    add("same_date_control", same_date["macro_f1_mean"].eq(1.0).all() and same_date["worst_class_recall_mean"].eq(1.0).all(), f"rows={len(same_date)}")

    metadata = pd.read_csv(ROOT / "evidence/training_only_metadata_baselines.csv")
    date_logit = metadata[(metadata["feature_set"] == "date_only") & (metadata["model"] == "logistic_regression")]
    add("training_only_date_baseline", len(date_logit) == 4 and date_logit["macro_f1"].between(0.70, 0.78).all(), f"range={date_logit['macro_f1'].min():.4f}-{date_logit['macro_f1'].max():.4f}")

    outer = pd.read_csv(ROOT / "evidence/outer_race_date_summary.csv")
    outer_xgb = outer[(outer["feature_set"] == "all_signals_without_order") & (outer["model"] == "xgboost") & (outer["aggregation"] == "mean_probability")].iloc[0]
    add("within_class_date_prediction", round(float(outer_xgb["accuracy_mean"]), 4) == 0.8635 and float(outer_xgb["worst_class_recall_mean"]) == 0.0, f"accuracy={outer_xgb['accuracy_mean']:.4f}; worst={outer_xgb['worst_class_recall_mean']:.4f}")

    for relative in ["paper/main.pdf", "paper/supplementary.pdf", "paper/main_machines.pdf"]:
        path = ROOT / relative
        add(f"pdf_{path.stem}", path.exists() and path.stat().st_size > 10_000, f"bytes={path.stat().st_size if path.exists() else 0}")

    failures = [item for item in checks if item["status"] != "PASS"]
    payload = {
        "schema_version": "1.0",
        "package_version": VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "overall_status": "PASS" if not failures else "FAIL",
        "check_count": len(checks),
        "failure_count": len(failures),
        "checks": checks,
    }
    json_path = args.out_dir / "submission_consistency_audit.json"
    md_path = args.out_dir / "submission_consistency_audit.md"
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Current Release Consistency Audit",
        "",
        f"- Package version: `{VERSION}`",
        f"- Generated: `{payload['generated_at_utc']}`",
        f"- Overall status: **{payload['overall_status']}**",
        f"- Checks: {len(checks)}",
        f"- Failures: {len(failures)}",
        "",
        "| Check | Status | Detail |",
        "| --- | --- | --- |",
    ]
    lines.extend(f"| {item['check']} | {item['status']} | {item['detail']} |" for item in checks)
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["overall_status"], "checks": len(checks), "failures": len(failures)}))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
