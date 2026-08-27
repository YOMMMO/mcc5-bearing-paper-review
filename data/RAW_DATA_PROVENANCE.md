# Raw-Data Provenance and Identity Audit

## Scope

The MCC5-THU Motor recordings are not redistributed in this repository. They remain available from the official [Mendeley Data record](https://doi.org/10.17632/6s3dggj9mw.1), with the formal description in [Data in Brief 65 (2026), 112583](https://doi.org/10.1016/j.dib.2026.112583).

The release contains a compact audit that a reviewer with the official download can reproduce. For every one of the 84 catalogued bearing recordings, it records:

- the de-identified recording ID and original dataset name;
- the relative path below the official dataset root;
- file size, data-row count, and column count;
- SHA-256 digest;
- acquisition timestamp parsed from the public filename.

No signal sample is included in the audit tables.

## Results

- Catalogued recordings: 84.
- Raw files resolved: 84.
- Distinct SHA-256 digests: 84.
- Exact duplicate digest groups: 0.
- Repeated acquisition-timestamp groups: 1.

The repeated timestamp `250708094556` occurs in three differently labelled operating-condition files. All three hashes are different. Pairwise streaming comparison over 1,152,000 rows and nine columns shows an identical time vector but non-identical signal values for every pair. This is consistent with a shared sampling grid, not copied recording content.

## Claim Boundary

The audit excludes byte-identical raw files among the 84 catalogued recordings and resolves the timestamp anomaly raised during external review. It does **not** establish that different files correspond to different physical bearing specimens, installations, or experimental sessions. The manuscript therefore uses the term `recording-grouped` and retains specimen/session independence as a limitation.

## Reproduction

```powershell
python src/data/audit_mcc5_provenance.py `
  --catalog data/recording_catalog_and_splits.csv `
  --raw-root <path-to-official-MCC5-root> `
  --out evidence `
  --deep-compare-timestamp-groups
```

Primary outputs are `evidence/raw_file_integrity_audit.csv`, `evidence/timestamp_collision_audit.csv`, `evidence/same_timestamp_pair_comparison.csv`, and `evidence/raw_integrity_summary.json`.
