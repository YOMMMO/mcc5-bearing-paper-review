# Data Access and Compact Review Data

## Raw Dataset

Raw MCC5-THU Motor recordings are not redistributed here. Obtain them from:

- Data record: https://doi.org/10.17632/6s3dggj9mw.1
- Formal data article: https://doi.org/10.1016/j.dib.2026.112583

The original dataset license and citation requirements apply.

The repository does not rely on an unverifiable filename-only assertion. A compact provenance audit records each catalogued file's relative dataset path, byte size, data shape, and SHA-256 digest in `evidence/raw_file_integrity_audit.csv`. Reviewers with the official download can regenerate it with:

```powershell
python src/data/audit_mcc5_provenance.py `
  --catalog data/recording_catalog_and_splits.csv `
  --raw-root <path-to-official-MCC5-root> `
  --out evidence `
  --deep-compare-timestamp-groups
```

The published audit resolves all 84 catalogued recordings and finds 84 distinct SHA-256 values. One acquisition timestamp is shared by three filenames; the deep comparison confirms a common time vector but different signal values in every pair. Hash uniqueness excludes byte-identical files only. It does not prove that recordings use different physical bearings, installations, or acquisition sessions.

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

## Session Metadata Boundary

Acquisition dates parsed from public recording identifiers are supplied in `evidence/session_metadata_audit.csv`. Date/class and date/protocol-role cross-tabs are descriptive checks for possible session confounding. The reported in-sample date-majority association is not a trained classifier, a validation result, or evidence that date alone predicts unseen recordings.

## Why Per-Window Maps Are Omitted

Each recording produces 179 overlapping windows. Repeating one role 179 times does not add review information and can obscure the true independent unit. The compact map preserves the role assignment that matters for leakage assessment. Full preprocessing code can reconstruct window membership from the raw recordings.
