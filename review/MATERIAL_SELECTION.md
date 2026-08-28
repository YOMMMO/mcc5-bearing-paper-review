# Material Selection Rationale

## Included because it can affect scientific review

- Current manuscript, supplement, journal-template preview, and figure sources.
- Recording-level provenance, labels, conditions, and frozen roles.
- Compact raw-file hashes/shapes and acquisition-session metadata audits, without signal redistribution.
- Result summaries, per-seed evidence, bootstrap outputs, and input-definition audits.
- Complete classical probability aggregation and majority-vote results at source-recording level for five models and four protocols.
- Direct partition controls and session-proxy sensitivity experiments.
- Exact feature schemas and preprocessing/model code.
- Tests for train-only normalization, evidence labels, and recording-grouped roles.
- Environment and dependency records.

## Excluded because it is redundant, licensed elsewhere, or irrelevant to review

- Raw MCC5 signals: large, licensed separately, and already publicly hosted by the data owner. Their identity is represented by reproducible relative-path, shape, and SHA-256 evidence.
- Window tensors and feature matrices: derivable from the raw data and code, but unnecessarily large for manuscript review.
- Duplicate per-window split maps: replaced by one compact row per recording.
- Checkpoints: not needed to audit manuscript logic or supplied result tables.
- Build directories, package metadata, bytecode caches, and logs.
- Earlier drafts and internal reviewer conversations.
- Other datasets that the manuscript does not claim as external validation.

This selection keeps the repository small enough for direct inspection while retaining the artifacts needed to evaluate the paper's central claims.
