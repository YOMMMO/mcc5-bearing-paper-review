# Acquisition-Session Confounding Audit

## Why This Audit Exists

Recording-level grouping prevents overlapping windows from one acquisition file from crossing data roles. It cannot, by itself, prove independence of physical bearings, installations, dates, or sessions. The public recording identifiers expose acquisition timestamps, so this release uses their date component for a compact descriptive audit.

## Observed Structure

The 84 bearing recordings occur on five acquisition dates. Diagnostic classes are unevenly distributed across those dates:

| Acquisition date | Healthy | Inner race | Outer race | Rolling element |
|---|---:|---:|---:|---:|
| 250702 | 12 | 0 | 0 | 0 |
| 250704 | 0 | 11 | 0 | 0 |
| 250707 | 0 | 12 | 11 | 24 |
| 250708 | 0 | 0 | 12 | 0 |
| 250821 | 0 | 0 | 2 | 0 |

Descriptive association measures are:

- in-sample date-majority agreement: 61/84 (0.7262);
- Cramer's V between acquisition date and class: 0.7714;
- normalized mutual information: 0.5989.

The date-majority value is obtained by assigning each date its most common class on the same 84 recordings. It is **not** an out-of-sample classifier, cross-validation score, or causal estimate.

## Interpretation

The association shows that acquisition date can carry class-related session information. It does not prove that the reported models explicitly use the timestamp, because date is not a model input. It does show that recording grouping alone cannot exclude session-level nuisance structure embedded in the measured signals. This is now stated in the manuscript as a limitation and a reason not to claim specimen-independent or cross-session generalization.

Machine-readable details are in:

- `evidence/session_metadata_audit.csv`;
- `evidence/acquisition_date_class_counts.csv`;
- `evidence/acquisition_date_protocol_role_counts.csv`;
- `evidence/session_confounding_summary.json`.
