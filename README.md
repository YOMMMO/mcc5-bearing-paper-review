# MCC5 Bearing Paper Review Package

This repository contains the minimal manuscript, evidence, and reproducibility materials needed to review:

**Recording-Grouped Bearing Fault Diagnosis under Directional Operating-Condition Shifts on the MCC5 Electric-Drive Benchmark**

The package is intended for scientific peer review, including review with GPT Pro. It deliberately excludes raw sensor recordings, model checkpoints, caches, build products, duplicate window-level split files, and historical manuscript versions.

## Start Here

1. Read the [main manuscript](paper/main.pdf).
2. Read the [Supplementary Material](paper/supplementary.pdf).
3. Use the [review guide](review/REVIEW_GUIDE.md) to understand the evidence hierarchy.
4. Follow the [claim-to-evidence map](review/CLAIM_EVIDENCE_MAP.md) when checking numerical statements.
5. Paste [this prepared prompt](review/GPT_PRO_REVIEW_PROMPT.md) into GPT Pro and provide the repository URL.

## Included

- Latest free-format manuscript and Supplementary Material in PDF and LaTeX.
- Latest `Machines` template preview and source.
- Publication figures in PDF and PNG.
- Recording-level catalog and frozen train/validation/test roles for all four protocols.
- Raw-file SHA-256/shape audit and deep comparison of repeated acquisition timestamps (without redistributing raw signals).
- Acquisition-date/class cross-tabs documenting residual session-confounding risk.
- Classical source-recording probability aggregation, majority vote, and clustered bootstrap evidence.
- All compact evidence tables used to support the manuscript's numerical claims.
- Exact scalar-input schemas and feature-membership definitions.
- Train-only normalization, model, metric, preprocessing, and experiment code.
- Environment, dependency lock, tests, license, and citation metadata.
- A run-provenance manifest separating formal-run evidence from later review audits.

## Excluded by Design

- Raw MCC5 signals: obtain them from [Mendeley Data, Version 1](https://doi.org/10.17632/6s3dggj9mw.1).
- Large processed windows and duplicate per-window split maps.
- Model checkpoints and training caches.
- Exploratory datasets not used as external validation in the manuscript.
- Internal review history, prompts, archived drafts, and submission reminders.

The formal MCC5 data article is [Chen et al., Data in Brief 65 (2026), 112583](https://doi.org/10.1016/j.dib.2026.112583).

## Evidence Status

- Formal run: `postfix_neural_fusion_20260710_180646`.
- Classical, corrected raw-neural, and strict engineered/fusion results use recording-grouped protocols and training-only preprocessing.
- `auxiliary_26` and `auxiliary_context_28` remain **post-hoc exploratory on the same 3000-rpm holdout**. They are not confirmatory evidence.
- The 26 stored auxiliary fields comprise 24 varying torque/key-phase summaries and two constant modality-presence indicators in this complete-modality subset.
- Recording separation does not establish independent physical bearing specimens.
- All 84 catalogued raw files have distinct SHA-256 hashes. This excludes byte-identical recordings, but it does not establish specimen or acquisition-session independence.
- Acquisition date is associated with class composition and is reported as a descriptive confounding audit, not as diagnostic performance.
- The paper does not claim cross-machine or non-MCC5 generalization.

## Compact Data Representation

`data/recording_catalog_and_splits.csv` contains one row per acquisition recording, rather than one row per overlapping window. It preserves the information needed to audit recording-level role separation while avoiding redundant multi-megabyte window maps.

## Quick Checks

```powershell
python -m compileall -q src tests run_formal_dl_fusion_gpu.py run_postfix_neural_fusion_pipeline.py
python -m unittest discover -s tests -v
```

Full experiment reproduction additionally requires the raw MCC5 dataset. The compact raw-file audit can be rebuilt with `src/data/audit_mcc5_provenance.py`, and classical source-recording metrics with `src/experiments/exp38_classical_recording_metrics.py`. See [data/README.md](data/README.md).

## Versioned Review Snapshot

The tagged [review package release](https://github.com/YOMMMO/mcc5-bearing-paper-review/releases/tag/review-v3.1.3) attaches a versioned snapshot of the compact review package together with the main and supplementary PDFs. Use the repository itself for browsing and the release ZIP when a reviewer needs one versioned download.

## License

Code is released under the MIT License. The MCC5 dataset is not redistributed and retains its original license.
