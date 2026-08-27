# Material Selection Rationale

## Included because it can affect scientific review

- Current manuscript, supplement, journal-template preview, and figure sources.
- Recording-level provenance, labels, conditions, and frozen roles.
- Result summaries, per-seed evidence, bootstrap outputs, and input-definition audits.
- Exact feature schemas and preprocessing/model code.
- Tests for train-only normalization, evidence labels, and recording-grouped roles.
- Environment and dependency records.

## Excluded because it is redundant, licensed elsewhere, or irrelevant to review

- Raw MCC5 signals: large and already publicly hosted by the data owner.
- Window tensors and feature matrices: derivable from the raw data and code, but unnecessarily large for manuscript review.
- Duplicate per-window split maps: replaced by one compact row per recording.
- Checkpoints: not needed to audit manuscript logic or supplied result tables.
- Build directories, package metadata, bytecode caches, and logs.
- Earlier drafts and internal reviewer conversations.
- Other datasets that the manuscript does not claim as external validation.

This selection keeps the repository small enough for direct inspection while retaining every artifact needed to challenge the paper's central claims.
