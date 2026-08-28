# Claim-to-Evidence Map

| Manuscript claim or section | Primary evidence | Supporting audit or code |
|---|---|---|
| 84 recordings, 15,036 windows, four fault-location classes | `paper/main.pdf`, Table 1; `data/recording_catalog_and_splits.csv` | `evidence/split.csv` |
| Zero recording overlap across frozen neural roles | `data/recording_catalog_and_splits.csv` | `evidence/split.csv`; `tests/test_release_integrity.py` |
| All 84 catalogued raw files resolve and have distinct SHA-256 digests | `evidence/raw_file_integrity_audit.csv`; `evidence/raw_integrity_summary.json` | `src/data/audit_mcc5_provenance.py`; `data/RAW_DATA_PROVENANCE.md` |
| The only repeated acquisition timestamp does not identify byte-identical or pointwise-identical signal files | `evidence/timestamp_collision_audit.csv`; `evidence/same_timestamp_pair_comparison.csv` | `src/data/audit_mcc5_provenance.py` |
| Acquisition date is associated with class composition, indicating residual session-proxy confounding risk | `evidence/acquisition_date_class_counts.csv`; `evidence/session_confounding_summary.json` | `evidence/session_metadata_audit.csv`; `review/SESSION_CONFOUNDING_AUDIT.md` |
| Removing overlap alone does not remove source-recording leakage | `evidence/partition_leakage_controls_summary.csv` | `evidence/partition_leakage_controls.csv`; `src/experiments/exp39_peer_review_controls.py` |
| Same-date discrimination remains possible, but date/session information remains predictive | `evidence/within_date_250707_summary.csv`; `evidence/training_only_metadata_baselines.csv`; `evidence/outer_race_date_summary.csv` | `review/SESSION_CONFOUNDING_AUDIT.md`; `src/experiments/exp39_peer_review_controls.py` |
| Predefined source-recording XGBoost macro-F1 = 0.9964 | `evidence/directional.csv` | `paper/main.pdf`, Table 3 |
| Ten grouped partitions: XGBoost 0.9777 $\pm$ 0.0356 at window level | `evidence/repeated.csv`; `evidence/repeated_by_split.csv` | `evidence/repeated_partition_manifest.csv` |
| Ten grouped partitions at source-recording level | `evidence/repeated_source_recording_metrics.csv`; `evidence/repeated_source_recording_predictions.csv` | `src/experiments/exp38_classical_recording_metrics.py` |
| Directional cross-mode, cross-load, and leave-one-speed findings | `evidence/directional.csv` | `paper/main.pdf`, Figure 2 and Table 3 |
| Upward 1000/2000 to 3000 rpm transfer is the principal boundary | `evidence/directional.csv` | `evidence/classical_bootstrap.csv`; `evidence/source_summary.csv` |
| Exact 254-feature membership and legacy grouping contamination | `evidence/exact.csv` | `schemas/exact_feature_membership.csv`; `paper/main.pdf`, Figure 4 |
| Corrected raw CNN/TCN/Transformer results | `evidence/raw_summary.csv`; `evidence/raw.csv` | `evidence/normalization.csv`; `src/utils/train_only_normalization.py` |
| Corrected strict engineered and fusion-input results | `evidence/fusion_summary.csv`; `evidence/fusion.csv` | `evidence/all_fusion_input_protocol_results.csv`; `schemas/` |
| Auxiliary-26/28 are post-hoc exploratory | `evidence/fusion_summary.csv` evidence-status field | `paper/supplementary.pdf`; `tests/test_release_integrity.py` |
| RPM/load alone does not explain the higher same-holdout auxiliary association | `evidence/deltas.csv`; `evidence/fusion_summary.csv` | `paper/main.pdf`, Table 5 |
| Recording-level neural/fusion performance and worst-class recall | `evidence/source_summary.csv`; `evidence/source.csv` | `evidence/per_recording_probability_audit.csv` |
| Recording-level classical performance for all five models under all four primary protocols | `evidence/classical_source_recording_model_matrix_metrics.csv`; `evidence/classical_source_recording_model_matrix_predictions.csv` | `evidence/classical_source_recording_model_matrix_bootstrap.csv`; `src/experiments/exp38_classical_recording_metrics.py` |
| Recording-cluster uncertainty intervals | `evidence/bootstrap.csv`; `evidence/classical_bootstrap.csv`; `evidence/classical_source_recording_bootstrap.csv` | `paper/supplementary.pdf` |
| Input-length sensitivity | `evidence/length.csv` | `paper/main.pdf`, Table 7 |
| Severity analysis is descriptive only | `evidence/severity_stratified_recording_metrics_summary.csv` | `paper/supplementary.pdf`, Table S5 |
| Manuscript numbers, titles, evidence labels, and release identity are synchronized | `evidence/submission_consistency_audit.md` | `evidence/submission_consistency_audit.json`; `tests/test_release_integrity.py` |

The map identifies the shortest verification path. Reviewers may inspect the per-seed tables for deeper checks.
