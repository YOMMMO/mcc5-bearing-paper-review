# GitHub Review Release Audit

Date: 2026-08-27

Repository: https://github.com/YOMMMO/mcc5-bearing-paper-review

## Selection Audit

- 84 acquisition recordings represented once each in the compact catalog.
- Four frozen protocol roles included at recording level.
- 24 evidence CSV files included.
- Current manuscript, Supplementary Material, `Machines` preview, and figure sources included.
- Exact input schemas, relevant source code, tests, and environment records included.
- Raw signals, checkpoints, processed windows, duplicate per-window role maps, caches, build products, and historical drafts excluded.

## Validation Performed Before Upload

- Seven unit/integrity tests passed.
- Every included CSV parsed successfully.
- Publication-class counts verified: 24 rolling-element, 23 inner-race, 25 outer-race, and 12 healthy recordings.
- Every protocol contains train, validation, and test roles.
- Auxiliary-26/28 evidence-status labels verified as `posthoc_exploratory_same_holdout`.
- No local absolute workspace or user-profile path found in publication files.
- No GitHub token, common access-key pattern, or private-key header found.
- No raw-data directory, checkpoint file, or file larger than 10 MiB included.
- Formal manuscript sources contain no pre-submission placeholder wording.

## Evidence Boundary

This repository is sufficient for manuscript and compact evidence review. A full numerical rerun additionally requires the separately hosted MCC5 raw dataset. The repository does not establish physical-bearing specimen independence or cross-machine generalization.
