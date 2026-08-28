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

## Control Experiments

Four complementary controls were added because the cross-tabulation alone cannot determine whether classifiers use fault-related signal structure or date-, severity-, or session-linked nuisance structure.

### Same-date three-class diagnosis

The 47 recordings acquired on 250707 contain inner-race, outer-race, and rolling-element faults. Random forest and XGBoost were evaluated over ten source-grouped splits using only these recordings. Mean-probability and majority-vote source-recording macro-F1, accuracy, and worst-class recall were 1.0000 in every split. This demonstrates within-date fault-class separability, but severity remains class-associated: inner-race recordings are low severity, outer-race recordings are high severity, and rolling-element recordings include both. The task also excludes healthy-versus-fault separation because no healthy recording was acquired on 250707.

### Same-date, severity-matched diagnosis

Two binary tasks additionally hold severity fixed on 250707. The high-severity task compares 12 rolling-element with 11 outer-race recordings; the low-severity task compares 12 rolling-element with 12 inner-race recordings. Across split seeds 42--51, both models and both recording-level aggregation rules achieve 1.0000 macro-F1 and worst-class recall. Each split tests only three high-severity or four low-severity recordings, however, and repeatedly partitions the same finite recording pool. The controls support within-date, severity-matched fault-location separability for these contrasts, not independent specimens or sessions.

### Training-only metadata baselines

Date, nominal RPM, load, and operating-mode metadata were never taken from held-out labels when fitting the baselines. A date-only logistic model achieved macro-F1 values of 0.7560, 0.7723, 0.7500, and 0.7028 on the source-recording, cross-mode, cross-load, and cross-RPM protocols, respectively. The operating-context-only counterpart achieved 0.0000, 0.1818, 0.1818, and 0.1111. These results confirm that date composition carries label information under the frozen protocols; they are sensitivity controls rather than diagnostic models.

### Outer-race date--severity composite prediction

To hold fault location fixed, acquisition date was predicted among the 25 outer-race recordings using ten repeated source-grouped folds. Severity and date are nevertheless coupled: all 12 low-severity recordings occur on 250708, while the 13 high-severity recordings occur on 250707 (11) or 250821 (2). XGBoost with non-order signal descriptors achieves accuracy $0.8635\pm0.0775$ and macro-F1 $0.5995\pm0.0547$. A training-fold severity-majority rule achieves higher accuracy ($0.9199\pm0.0034$) and macro-F1 ($0.6387\pm0.0025$). Both have zero worst-date recall because the rare 250821 date is never recovered. The signal-model score therefore cannot serve as independent evidence of a session signature; it is reported as date--severity composite predictability.

## Interpretation

The combined evidence is deliberately bounded. Same-date and severity-matched tasks support fault-location separability in the available contrasts. Conversely, metadata baselines show that date composition carries label information, and the outer-race control is dominated by the date--severity association. The diagnostic models do not receive timestamps directly, but recording grouping cannot establish specimen or session independence from these data. The manuscript therefore does not claim specimen-independent or cross-session generalization.

Machine-readable details are in:

- `evidence/session_metadata_audit.csv`;
- `evidence/acquisition_date_class_counts.csv`;
- `evidence/acquisition_date_protocol_role_counts.csv`;
- `evidence/session_confounding_summary.json`;
- `evidence/within_date_250707_summary.csv`;
- `evidence/within_date_severity_matched_summary.csv`;
- `evidence/within_date_severity_matched_confusion_matrices.csv`;
- `evidence/training_only_metadata_baselines.csv`;
- `evidence/outer_race_date_summary.csv`;
- `evidence/outer_race_date_severity_baseline_summary.csv`.
