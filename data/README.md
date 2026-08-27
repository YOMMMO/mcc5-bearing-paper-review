# Data Access and Compact Review Data

## Raw Dataset

Raw MCC5-THU Motor recordings are not redistributed here. Obtain them from:

- Data record: https://doi.org/10.17632/6s3dggj9mw.1
- Formal data article: https://doi.org/10.1016/j.dib.2026.112583

The original dataset license and citation requirements apply.

## Files in This Repository

### `recording_catalog_and_splits.csv`

One row per acquisition recording. It contains:

- recording identifier and original dataset label;
- publication-facing fault-location label;
- severity, operating mode, nominal RPM, and load;
- train/validation/test role for source-file, cross-condition, cross-load, and cross-RPM protocols.

No local filesystem paths or raw signal values are included.

### `recording_level_split_summary.csv`

Counts of recordings by protocol, role, and publication-facing class.

## Why Per-Window Maps Are Omitted

Each recording produces 179 overlapping windows. Repeating one role 179 times does not add review information and can obscure the true independent unit. The compact map preserves the role assignment that matters for leakage assessment. Full preprocessing code can reconstruct window membership from the raw recordings.
