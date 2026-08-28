# Evidence Directory

The CSV files in this directory are the compact outputs used to generate manuscript tables, figures, and uncertainty summaries. Start with:

- `directional.csv`
- `repeated.csv`
- `exact.csv`
- `raw_summary.csv`
- `fusion_summary.csv`
- `source_summary.csv`
- `bootstrap.csv`
- `classical_source_recording_metrics.csv`
- `classical_source_recording_bootstrap.csv`
- `classical_source_recording_model_matrix_metrics.csv`
- `classical_source_recording_model_matrix_bootstrap.csv`
- `repeated_source_recording_metrics.csv`
- `partition_leakage_controls_summary.csv`
- `within_date_250707_summary.csv`
- `training_only_metadata_baselines.csv`
- `outer_race_date_summary.csv`
- `raw_file_integrity_audit.csv`
- `timestamp_collision_audit.csv`
- `same_timestamp_pair_comparison.csv`
- `session_metadata_audit.csv`
- `acquisition_date_class_counts.csv`
- `acquisition_date_protocol_role_counts.csv`
- `submission_consistency_audit.md`

`raw_integrity_summary.json` and `session_confounding_summary.json` provide compact machine-readable summaries. The raw audit contains hashes and metadata only; no raw signal samples are redistributed. Detailed prediction files support verification of aggregation and control analyses. See `review/CLAIM_EVIDENCE_MAP.md` for the intended mapping.
