"""Integrity checks for the compact MCC5 paper-review release."""

from __future__ import annotations

import json
import hashlib
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class ReleaseIntegrityTests(unittest.TestCase):
    def test_required_public_artifacts_exist(self) -> None:
        required = [
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "review/REVIEW_GUIDE.md",
            "review/CLAIM_EVIDENCE_MAP.md",
            "review/SESSION_CONFOUNDING_AUDIT.md",
            "review/RUN_PROVENANCE_MANIFEST.json",
            "data/recording_catalog_and_splits.csv",
            "data/RAW_DATA_PROVENANCE.md",
            "evidence/raw_summary.csv",
            "evidence/fusion_summary.csv",
            "evidence/all_fusion_input_protocol_results.csv",
            "evidence/severity_stratified_recording_metrics_summary.csv",
            "evidence/per_recording_probability_audit.csv",
            "evidence/raw_file_integrity_audit.csv",
            "evidence/raw_integrity_summary.json",
            "evidence/timestamp_collision_audit.csv",
            "evidence/same_timestamp_pair_comparison.csv",
            "evidence/session_metadata_audit.csv",
            "evidence/acquisition_date_class_counts.csv",
            "evidence/acquisition_date_protocol_role_counts.csv",
            "evidence/session_confounding_summary.json",
            "evidence/classical_source_recording_metrics.csv",
            "evidence/classical_source_recording_bootstrap.csv",
            "evidence/classical_source_recording_predictions.csv",
            "evidence/classical_source_recording_model_matrix_metrics.csv",
            "evidence/repeated_source_recording_metrics.csv",
            "evidence/partition_leakage_controls.csv",
            "evidence/partition_leakage_controls_summary.csv",
            "evidence/training_only_metadata_baselines.csv",
            "evidence/within_date_250707_summary.csv",
            "evidence/outer_race_date_summary.csv",
            "paper/main.tex",
            "paper/supplementary.tex",
        ]
        for relative in required:
            self.assertTrue((ROOT / relative).is_file(), relative)

    def test_frozen_splits_are_recording_grouped(self) -> None:
        catalog = pd.read_csv(ROOT / "data/recording_catalog_and_splits.csv")
        self.assertEqual(len(catalog), 84)
        self.assertEqual(catalog["recording_id"].nunique(), 84)
        self.assertFalse(catalog["publication_label"].isna().any())
        self.assertEqual(
            catalog["publication_label"].value_counts().to_dict(),
            {
                "outer_race": 25,
                "rolling_element": 24,
                "inner_race": 23,
                "healthy": 12,
            },
        )
        for column in [
            "source_file_role",
            "cross_condition_role",
            "cross_load_role",
            "cross_rpm_role",
        ]:
            self.assertFalse(catalog[column].isna().any(), column)
            self.assertEqual(set(catalog[column]), {"train", "val", "test"}, column)

    def test_exploratory_fusion_rows_are_labeled(self) -> None:
        fusion = pd.read_csv(ROOT / "evidence/fusion_summary.csv")
        exploratory = fusion[fusion["setting_label"].isin(["auxiliary_26", "auxiliary_context_28"])]
        self.assertFalse(exploratory.empty)
        self.assertEqual(set(exploratory["evidence_status"]), {"posthoc_exploratory_same_holdout"})
        strict = fusion[fusion["setting_label"] == "full_engineered_254"]
        self.assertFalse(strict.empty)
        self.assertEqual(set(strict["evidence_status"]), {"corrected_formal_train_only_zscore"})

    def test_raw_integrity_audit_excludes_exact_duplicates(self) -> None:
        audit = pd.read_csv(ROOT / "evidence/raw_file_integrity_audit.csv")
        self.assertEqual(len(audit), 84)
        self.assertEqual(audit["recording_id"].nunique(), 84)
        self.assertEqual(audit["sha256"].nunique(), 84)
        self.assertFalse(audit["exact_duplicate_content"].astype(bool).any())
        self.assertEqual(set(audit["data_row_count"]), {1_152_000})
        self.assertEqual(set(audit["column_count"]), {9})

        collisions = pd.read_csv(ROOT / "evidence/timestamp_collision_audit.csv")
        self.assertEqual(len(collisions), 1)
        self.assertEqual(int(collisions.iloc[0]["recording_count"]), 3)
        self.assertTrue(bool(collisions.iloc[0]["all_file_hashes_unique"]))
        comparisons = pd.read_csv(ROOT / "evidence/same_timestamp_pair_comparison.csv")
        self.assertEqual(len(comparisons), 3)
        self.assertFalse(comparisons["sha256_equal"].astype(bool).any())
        self.assertFalse(comparisons["signal_values_equal"].astype(bool).any())
        self.assertTrue(comparisons["time_axis_equal"].astype(bool).all())

    def test_session_audit_is_descriptive(self) -> None:
        summary = json.loads(
            (ROOT / "evidence/session_confounding_summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(int(summary["recording_count"]), 84)
        self.assertIn("Descriptive association only", str(summary["interpretation"]))
        self.assertAlmostEqual(float(summary["date_majority_accuracy_in_sample"]), 61 / 84)
        self.assertGreater(float(summary["cramers_v_date_vs_class"]), 0.7)

    def test_classical_recording_metrics_cover_all_protocols(self) -> None:
        metrics = pd.read_csv(ROOT / "evidence/classical_source_recording_metrics.csv")
        self.assertEqual(
            set(metrics["split"]),
            {"source_file", "cross_condition", "cross_load", "cross_rpm"},
        )
        self.assertEqual(set(metrics["aggregation"]), {"mean_probability", "majority_vote"})
        cross_rpm = metrics[
            (metrics["split"] == "cross_rpm")
            & (metrics["aggregation"] == "mean_probability")
        ].iloc[0]
        self.assertAlmostEqual(float(cross_rpm["macro_f1"]), 0.7019607843137254)
        self.assertAlmostEqual(float(cross_rpm["worst_class_recall"]), 1 / 9)

    def test_classical_recording_matrix_and_repeated_partitions(self) -> None:
        matrix = pd.read_csv(
            ROOT / "evidence/classical_source_recording_model_matrix_metrics.csv"
        )
        self.assertEqual(matrix["split"].nunique(), 4)
        self.assertEqual(matrix["model"].nunique(), 5)
        self.assertEqual(set(matrix["aggregation"]), {"mean_probability", "majority_vote"})
        self.assertEqual(len(matrix), 4 * 5 * 2)

        repeated = pd.read_csv(ROOT / "evidence/repeated_source_recording_metrics.csv")
        self.assertEqual(set(repeated["split_seed"]), set(range(42, 52)))
        self.assertEqual(set(repeated["model"]), {"random_forest", "xgboost"})
        self.assertEqual(set(repeated["aggregation"]), {"mean_probability", "majority_vote"})
        self.assertEqual(set(repeated["source_recording_count"]), {12})

    def test_partition_and_session_proxy_controls(self) -> None:
        partition = pd.read_csv(ROOT / "evidence/partition_leakage_controls.csv")
        self.assertEqual(set(partition["split_seed"]), set(range(42, 52)))
        self.assertEqual(partition["protocol"].nunique(), 4)
        random_rows = partition[~partition["recording_grouped"].astype(bool)]
        grouped_rows = partition[partition["recording_grouped"].astype(bool)]
        self.assertTrue((random_rows["source_recording_overlap"] == 84).all())
        self.assertTrue((grouped_rows["source_recording_overlap"] == 0).all())
        self.assertTrue(grouped_rows["recording_metric_valid"].astype(bool).all())
        self.assertFalse(random_rows["recording_metric_valid"].astype(bool).any())

        metadata = pd.read_csv(ROOT / "evidence/training_only_metadata_baselines.csv")
        self.assertEqual(set(metadata["fit_scope"]), {"training_recordings_only"})
        self.assertEqual(metadata["split"].nunique(), 4)
        date_only = metadata[
            (metadata["feature_set"] == "date_only")
            & (metadata["model"] == "logistic_regression")
        ]
        self.assertEqual(len(date_only), 4)
        self.assertGreater(float(date_only["macro_f1"].min()), 0.65)

        within_date = pd.read_csv(ROOT / "evidence/within_date_250707_summary.csv")
        self.assertEqual(set(within_date["macro_f1_mean"]), {1.0})
        outer_date = pd.read_csv(ROOT / "evidence/outer_race_date_summary.csv")
        xgb = outer_date[
            (outer_date["feature_set"] == "all_signals_without_order")
            & (outer_date["model"] == "xgboost")
            & (outer_date["aggregation"] == "mean_probability")
        ].iloc[0]
        self.assertGreater(float(xgb["accuracy_mean"]), 0.8)

    def test_posthoc_auxiliary_result_is_not_in_the_abstract(self) -> None:
        main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
        abstract = main.split("\\begin{abstract}", 1)[1].split("\\end{abstract}", 1)[0]
        self.assertNotIn("auxiliary-26", abstract)
        self.assertNotIn("0.9162", abstract)

        machines = (ROOT / "paper/main_machines.tex").read_text(encoding="utf-8")
        machines_abstract = machines.split("\\abstract{", 1)[1].split("}\n", 1)[0]
        self.assertNotIn("auxiliary-26", machines_abstract)
        self.assertNotIn("0.9162", machines_abstract)

    def test_public_manuscripts_exclude_internal_instructions(self) -> None:
        forbidden = [
            "must remain future tense",
            "corrected run identifier",
            "centered-only predecessor",
            "Decision C",
            "E:\\thesis2",
            "must be confirmed before submission",
            "must be regenerated",
        ]
        for relative in ["paper/main.tex", "paper/main_machines.tex", "paper/supplementary.tex"]:
            content = (ROOT / relative).read_text(encoding="utf-8")
            for phrase in forbidden:
                self.assertNotIn(phrase, content, f"{relative}: {phrase}")

    def test_public_package_is_neutral_and_has_no_review_prompt(self) -> None:
        self.assertFalse((ROOT / "review/GPT_PRO_REVIEW_PROMPT.md").exists())
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertNotIn("GPT Pro", readme)
        self.assertNotIn("prepared prompt", readme)

    def test_publication_facing_ai_disclosure_is_specific(self) -> None:
        for relative in ["paper/main.tex", "paper/main_machines.tex"]:
            content = (ROOT / relative).read_text(encoding="utf-8")
            self.assertIn("OpenAI ChatGPT Pro", content, relative)
            self.assertIn("OpenAI Codex", content, relative)
            self.assertIn("did not generate or alter the raw measurements", content, relative)
            self.assertIn("takes full responsibility", content, relative)

    def test_run_provenance_sha256_anchors(self) -> None:
        manifest = json.loads(
            (ROOT / "review/RUN_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        for relative, expected in manifest["sha256_anchors"].items():
            payload = (ROOT / relative).read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(), expected, relative)

    def test_current_submission_consistency_audit_passes(self) -> None:
        audit = json.loads(
            (ROOT / "evidence/submission_consistency_audit.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(audit["package_version"], "3.1.4")
        self.assertEqual(audit["overall_status"], "PASS")
        self.assertEqual(int(audit["failure_count"]), 0)
        self.assertEqual(int(audit["check_count"]), 28)

    def test_release_version_alignment(self) -> None:
        self.assertIn("review-v3.1.4", (ROOT / "README.md").read_text(encoding="utf-8"))
        self.assertIn("version: 3.1.4", (ROOT / "CITATION.cff").read_text(encoding="utf-8"))
        self.assertIn('version = "3.1.4"', (ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        provenance = json.loads(
            (ROOT / "review/RUN_PROVENANCE_MANIFEST.json").read_text(encoding="utf-8")
        )
        self.assertEqual(provenance["package_version"], "review-v3.1.4")

    def test_artifact_manifest_hashes(self) -> None:
        manifest = pd.read_csv(ROOT / "review/ARTIFACT_MANIFEST_SHA256.csv")
        self.assertGreater(len(manifest), 180)
        self.assertTrue(manifest["path"].is_unique)
        for row in manifest.itertuples(index=False):
            path = ROOT / str(row.path)
            self.assertTrue(path.is_file(), str(row.path))
            self.assertEqual(
                hashlib.sha256(path.read_bytes()).hexdigest(),
                str(row.sha256),
                str(row.path),
            )


if __name__ == "__main__":
    unittest.main()
