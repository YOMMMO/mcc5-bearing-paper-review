# Evidence and Reproduction Guide

## Recommended Reading Order

1. `paper/main.pdf`
2. `paper/supplementary.pdf`
3. `review/CLAIM_EVIDENCE_MAP.md`
4. `data/recording_catalog_and_splits.csv`
5. `data/RAW_DATA_PROVENANCE.md`
6. `review/SESSION_CONFOUNDING_AUDIT.md`
7. Relevant files under `evidence/`
8. Code and exact input schemas only where a methodological claim requires implementation verification

## Scientific Identity of the Paper

This is a benchmark and evidence-audit study. Its primary contributions are:

1. recording-grouped evaluation on the MCC5 bearing subset;
2. directional analysis of operating-condition shifts;
3. repeated grouped partitions and recording-level uncertainty;
4. exact feature-membership and fusion-input audits;
5. direct partition controls and session-proxy sensitivity analyses.

It is not presented as a new state-of-the-art domain-generalization network.

## Evidence Hierarchy

### Confirmatory or predefined evidence

- Frozen recording-grouped source-file, cross-condition, cross-load, and cross-RPM protocols.
- Classical model results and repeated grouped partition sensitivity.
- Complete source-recording results for five classical models under all four predefined protocols.
- Corrected raw-neural and strict engineered/fusion results generated with training-only z-score statistics.
- Recording-level aggregation and class-stratified recording bootstrap.
- Matched random-window, nonoverlapping-window, and recording-grouped partition controls.
- Raw-file identity audit showing 84 distinct SHA-256 values and value-level comparison of the only repeated-timestamp group.

### Descriptive sensitivity evidence

- Full one-second raw-input comparison.
- Severity-stratified analysis.
- Per-recording probability visualization.
- Acquisition-date/class association and protocol-role cross-tabs.
- Same-date three-class diagnosis, training-only date/context metadata baselines, and outer-race-only acquisition-date prediction. Taken together, these controls show that date alone cannot explain all discrimination while residual session-proxy confounding remains plausible.

### Post-hoc exploratory evidence

- `auxiliary_26` and `auxiliary_context_28` on the already inspected 3000-rpm holdout.

These auxiliary configurations may generate hypotheses but must not be treated as independently confirmed model superiority.

## Important Boundaries

- The independent unit available in the public metadata is an acquisition recording, not a verifiably unique physical bearing specimen.
- Distinct raw-file hashes exclude exact file duplication but do not establish independent bearings, installations, or acquisition sessions.
- Five acquisition dates are unevenly associated with diagnostic classes; recording grouping cannot remove this residual session-level ambiguity.
- The 15,036 overlapping windows are not 15,036 independent experimental replicates.
- The strongest difficulty is directional upward extrapolation from 1000/2000 rpm to 3000 rpm.
- Current order-related features do not establish universal speed invariance.
- Results are limited to the MCC5 bearing-only subset and one test rig.
- Recording-bootstrap intervals condition on the fitted model and finite listed test recordings; they are not uncertainty intervals for future specimens.

## Verification Questions

1. Are the title, abstract, conclusions, and claimed novelty consistent with the benchmark-level contribution?
2. Are recording-level grouping and held-out-test exclusion described and implemented clearly enough?
3. Are optimization-seed variability, partition variability, and recording-bootstrap uncertainty kept distinct?
4. Does every headline number trace to a supplied evidence table?
5. Are the auxiliary configurations consistently marked post-hoc exploratory?
6. Are mechanical interpretations phrased as supported findings versus hypotheses?
7. Are the limitations sufficient for the absence of physical specimen IDs and external-machine validation?
8. Does the raw/session provenance evidence justify the paper's deliberately narrow recording-level claim boundary?
9. Do the direct partition controls distinguish overlap removal from source-recording separation?
10. Does the complete model matrix prevent selected-model reporting from hiding protocol-specific failures?
