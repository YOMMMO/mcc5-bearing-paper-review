# Software Supplement Release Audit

Date: 2026-08-28

Version: `review-v3.1.5`

Repository: https://github.com/YOMMMO/mcc5-bearing-paper-review

## Selection Audit

- 84 acquisition recordings represented once each in the compact catalog.
- Four frozen protocol roles included at recording level.
- 52 evidence CSV files included.
- Current manuscript, Supplementary Material, `Machines` preview, and figure sources included.
- Exact input schemas, relevant source code, tests, and environment records included.
- Official `Machines` template dependencies included under `paper/Definitions/`.
- Raw signals, checkpoints, processed windows, caches, build products, prompts, and historical drafts excluded.

## Validation Performed Before Release

- Current submission-consistency audit passes 32/32 checks.
- Publication-class counts are 24 rolling-element, 23 inner-race, 25 outer-race, and 12 healthy recordings.
- All four formal protocols have zero source-recording overlap across training, validation, and test roles.
- All 84 catalogued raw files resolved, have the expected 1,152,000-by-9 shape, and have distinct SHA-256 digests.
- The repeated-timestamp trio shares its time vector but is neither byte-identical nor pointwise-identical in the eight measured channels.
- Four comparative random/grouped and overlapping/nonoverlapping partition controls are included.
- Same-date, severity-matched, training-only metadata, and date-severity composite controls are included with their limitations.
- A complete five-model-by-four-protocol recording-level classical matrix and ten repeated grouped-partition results are included.
- Auxiliary-26/28 evidence remains labelled post-hoc exploratory on the same holdout.
- Manuscripts contain a formal AI-use disclosure and no internal pre-submission reminders.
- Public-facing repository text is neutral and contains no model-review prompt.

## Evidence Boundary

The repository supports manuscript, code, and compact-evidence review. A full numerical rerun additionally requires the separately hosted MCC5 raw dataset. The controls do not establish physical-bearing specimen independence, acquisition-session independence, cross-machine generalization, or confirmatory auxiliary-fusion superiority.

Formal-run evidence and follow-up controls are separated in `review/RUN_PROVENANCE_MANIFEST.json`.
