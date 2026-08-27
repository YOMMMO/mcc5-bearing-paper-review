"""Integrity checks for the compact MCC5 paper-review release."""

from __future__ import annotations

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
            "data/recording_catalog_and_splits.csv",
            "evidence/raw_summary.csv",
            "evidence/fusion_summary.csv",
            "evidence/all_fusion_input_protocol_results.csv",
            "evidence/severity_stratified_recording_metrics_summary.csv",
            "evidence/per_recording_probability_audit.csv",
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

    def test_public_manuscripts_exclude_internal_instructions(self) -> None:
        forbidden = [
            "must remain future tense",
            "corrected run identifier",
            "centered-only predecessor",
            "Decision C",
            "GPT Pro",
            "E:\\thesis2",
        ]
        for relative in ["paper/main.tex", "paper/supplementary.tex"]:
            content = (ROOT / relative).read_text(encoding="utf-8")
            for phrase in forbidden:
                self.assertNotIn(phrase, content, f"{relative}: {phrase}")


if __name__ == "__main__":
    unittest.main()
