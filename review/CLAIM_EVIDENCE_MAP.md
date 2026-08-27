# Claim-to-Evidence Map

| Manuscript claim or section | Primary evidence | Supporting audit or code |
|---|---|---|
| 84 recordings, 15,036 windows, four fault-location classes | `paper/main.pdf`, Table 1; `data/recording_catalog_and_splits.csv` | `evidence/split.csv` |
| Zero recording overlap across frozen neural roles | `data/recording_catalog_and_splits.csv` | `evidence/split.csv`; `tests/test_release_integrity.py` |
| Predefined source-recording XGBoost macro-F1 = 0.9964 | `evidence/directional.csv` | `paper/main.pdf`, Table 3 |
| Ten grouped partitions: XGBoost 0.9777 +/- 0.0356 | `evidence/repeated.csv`; `evidence/repeated_by_split.csv` | `evidence/repeated_partition_manifest.csv` |
| Directional cross-mode, cross-load, and leave-one-speed findings | `evidence/directional.csv` | `paper/main.pdf`, Figure 2 and Table 3 |
| Upward 1000/2000 to 3000 rpm transfer is the principal boundary | `evidence/directional.csv` | `evidence/classical_bootstrap.csv`; `evidence/source_summary.csv` |
| Exact 254-feature membership and legacy grouping contamination | `evidence/exact.csv` | `schemas/exact_feature_membership.csv`; `paper/main.pdf`, Figure 4 |
| Corrected raw CNN/TCN/Transformer results | `evidence/raw_summary.csv`; `evidence/raw.csv` | `evidence/normalization.csv`; `src/utils/train_only_normalization.py` |
| Corrected strict engineered and fusion-input results | `evidence/fusion_summary.csv`; `evidence/fusion.csv` | `evidence/all_fusion_input_protocol_results.csv`; `schemas/` |
| Auxiliary-26/28 are post-hoc exploratory | `evidence/fusion_summary.csv` evidence-status field | `paper/supplementary.pdf`; `tests/test_release_integrity.py` |
| RPM/load alone does not explain the higher same-holdout auxiliary association | `evidence/deltas.csv`; `evidence/fusion_summary.csv` | `paper/main.pdf`, Table 5 |
| Recording-level performance and worst-class recall | `evidence/source_summary.csv`; `evidence/source.csv` | `evidence/per_recording_probability_audit.csv` |
| Recording-cluster uncertainty intervals | `evidence/bootstrap.csv`; `evidence/classical_bootstrap.csv` | `paper/supplementary.pdf`, Table S6 |
| Input-length sensitivity | `evidence/length.csv` | `paper/main.pdf`, Table 7 |
| Severity analysis is descriptive only | `evidence/severity_stratified_recording_metrics_summary.csv` | `paper/supplementary.pdf`, Table S5 |
| Manuscript numbers, titles, evidence labels, and release identity are synchronized | `evidence/submission_consistency_audit.md` | `evidence/submission_consistency_audit.json` |

The map identifies the shortest verification path. Reviewers may inspect the per-seed tables for deeper checks.
