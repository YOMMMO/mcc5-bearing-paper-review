# Submission Consistency Audit

- Formal run: `postfix_neural_fusion_20260710_180646`
- Overall status: **PASS**
- Checks: 50
- Failures: 0

| Check | Status | Detail |
| --- | --- | --- |
| formal_run_id | PASS | run_id=postfix_neural_fusion_20260710_180646; rows=84 |
| protocol_seed_counts | PASS | {'cross_condition': 3, 'cross_load': 3, 'cross_rpm': 5, 'source_file': 3} |
| cross_rpm_seed_range | PASS | expected seeds 42-46 |
| supplement_seed_caption | PASS | S2 distinguishes 3 versus 5 seeds |
| main_table6_seed_counts | PASS | Table 6 exposes the 3-seed raw-CNN and 5-seed focused-fusion aggregation counts |
| main_figure2_display_range | PASS | Figure 2 discloses the truncated display range |
| main_figure3_partition_semantics | PASS | Figure 3 uses source-recording terminology, grayscale-safe line styles, and explicit mean/axis semantics |
| main_figure1_overlap_scope | PASS | Figure 1 states the exact recording-level overlap-control scope |
| main_table5_evidence_caption | PASS | Table 5 caption neutrally covers matched and post-hoc rows and states five seeds |
| main_table7_window_timing | PASS | Table 7 reports inference per test window rather than ambiguously per sample |
| supplement_posthoc_visual_hierarchy | PASS | Supplement separates predefined and same-holdout post-hoc evidence in tables and figures |
| supplement_s3_landscape | PASS | per-recording probability audit is rendered as a landscape full-page figure |
| supplement_s5_readability | PASS | Supplementary Table S5 uses an unscaled small-font table with increased row spacing |
| vector_figure_sources | PASS | publication PDFs reference vector-PDF figure sources while PNG backups remain available |
| main_figures_2_to_4_width | PASS | Figures 2-4 retain near-full text width for final-template legibility |
| severity_boundary | PASS | descriptive-only severity caveat present |
| abstract_auxiliary_wording | PASS | abstract defines the complete auxiliary-26 bundle and avoids causal gains wording |
| recording_aggregation_definition | PASS | Methods reports only the recording aggregation that appears in the results |
| supplement_s3_prediction_definition | PASS | Supplementary Figure S3 defines how the displayed predicted class is assigned |
| matplotlib_type42_fonts | PASS | Matplotlib vector exports request embedded Type 42 fonts |
| no_internal_reminders_in_formal_manuscripts | PASS | formal manuscript, supplement, and Machines source contain no internal reminder or placeholder wording |
| publication_facing_rolling_element_label | PASS | Figure S3 uses rolling-element publication terminology while preserving source-recording identifiers |
| publication_ready_ai_disclosure | PASS | formal manuscripts contain a publication-facing AI-use disclosure without internal instructions |
| model_settings_table | PASS | Supplementary Table S7 contains classical, raw-neural, and fusion settings |
| test_exclusion_statement | PASS | Section 3.4 states the held-out-test exclusion rule |
| repeated_partition_protocol | PASS | seeds 42-51; model seed 42; roles 59/13/12; four test classes |
| related_work_citation_match | PASS | descriptor claim narrowed and motor-current statement directly cited |
| conclusion_posthoc_wording | PASS | same-holdout association wording used in the Conclusion |
| severity_metric_definition | PASS | Supplementary Table S5 defines fault macro-F1 |
| bootstrap_publication_wording | PASS | Supplementary Table S6 uses publication names and recording-level definitions |
| supplement_s3_order | PASS | predefined configurations precede post-hoc configurations |
| inference_timing_conditions | PASS | Table 7 note reports the actual timing conditions |
| main_table_posthoc_marker | PASS | star and footnote present |
| posthoc_evidence_status | PASS | auxiliary-26/28 remain post-hoc exploratory |
| title_main | PASS | exact current title |
| title_supplement | PASS | exact current title |
| title_cover_letter | PASS | exact current title |
| title_title_page | PASS | exact current title |
| title_machines_template | PASS | exact current title |
| no_old_title | PASS | old title absent |
| no_unverified_public_repo_claim | PASS | submission files do not claim the unverified URL |
| main_classical_numbers | PASS | 0.9964, 0.9777, 0.0356, 0.9790, 0.9891, 0.6921 |
| fusion_numbers_csv_to_supplement | PASS | {'vibration_current': (0.7081, 0.0986), 'vibration_current_rpm_load_only': (0.7124, 0.0529), 'vibration_current_auxiliary_only': (0.9162, 0.0245), 'full_engineered_features_254': (0.4001, 0.0813)} |
| reference_order | PASS | first-appearance audit reports PASS |
| main_pdf | PASS | free-format PDF exists |
| supplement_pdf | PASS | supplement PDF exists |
| machines_pdf | PASS | Machines preview exists |
| software_release_hash | PASS | 8FAAF78B53111BD9021EDC67CE9D6DFA65F991D87A800C99F6F342F0C97741FA |
| checklist_release_identity | PASS | submission checklist records software filename and SHA-256 |
| six_file_package | PASS | ['01_PUBLICATION_CLEANUP_COMPLETION.md', '02_main_free_format.pdf', '03_supplementary.pdf', '04_main_machines.pdf', '05_mcc5_recording_grouped_benchmark_v3.1.2.zip', '06_submission_consistency_audit.md'] |
