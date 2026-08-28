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

Three complementary controls were added because the cross-tabulation alone cannot determine whether classifiers use fault-related signal structure or date-linked nuisance structure.

### Same-date three-class diagnosis

The 47 recordings acquired on 250707 contain inner-race, outer-race, and rolling-element faults. Random forest and XGBoost were evaluated over ten source-grouped splits using only these recordings. Mean-probability and majority-vote source-recording macro-F1, accuracy, and worst-class recall were 1.0000 in every split. This demonstrates that acquisition date alone cannot explain all three-class discrimination. It does not address healthy-versus-fault separation because no healthy recording was acquired on 250707.

### Training-only metadata baselines

Date, nominal RPM, load, and operating-mode metadata were never taken from held-out labels when fitting the baselines. A date-only logistic model achieved macro-F1 values of 0.7560, 0.7723, 0.7500, and 0.7028 on the source-recording, cross-mode, cross-load, and cross-RPM protocols, respectively. The operating-context-only counterpart achieved 0.0000, 0.1818, 0.1818, and 0.1111. These results confirm that date composition carries label information under the frozen protocols; they are sensitivity controls rather than diagnostic models.

### Within-class date prediction

To hold fault location fixed, acquisition date was predicted among the 25 outer-race recordings using ten repeated source-grouped folds. XGBoost with non-order signal descriptors achieved accuracy $0.8635\pm0.0775$ and macro-F1 $0.5995\pm0.0547$. Worst-date recall was 0 because the 250821 date contains only two recordings. The result indicates measurable date/session information in the signals, but the imbalance prevents it from being interpreted as balanced date classification.

## Interpretation

The combined evidence is deliberately mixed. Perfect same-date three-class diagnosis shows that the diagnostic signal is not reducible to a date lookup. Conversely, the metadata and within-class controls show that date-linked session information remains predictive. The reported models do not receive timestamps directly, but recording grouping alone cannot exclude session-level nuisance structure embedded in measured signals. The manuscript therefore does not claim specimen-independent or cross-session generalization.

Machine-readable details are in:

- `evidence/session_metadata_audit.csv`;
- `evidence/acquisition_date_class_counts.csv`;
- `evidence/acquisition_date_protocol_role_counts.csv`;
- `evidence/session_confounding_summary.json`;
- `evidence/within_date_250707_summary.csv`;
- `evidence/training_only_metadata_baselines.csv`;
- `evidence/outer_race_date_summary.csv`.
