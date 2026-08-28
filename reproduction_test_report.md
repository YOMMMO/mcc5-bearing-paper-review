# Reproduction Test Report

Date: 2026-08-28

Release: `review-v3.1.5`

## Result

- Python compile check: PASS.
- Normalization and release-integrity tests: PASS, 19/19.
- Frozen recording-group role audit: PASS.
- Current submission-consistency audit: PASS, 32/32.
- Same-date, severity-matched control evidence: PASS for both predefined binary contrasts, with the finite-test-set limitation retained.
- Outer-race date-severity baseline audit: PASS; the training-fold severity rule exceeds the signal-model date accuracy, so no independent session-signature claim is made.
- Runner command-line smoke check: PASS.
- Isolated package build and installation: PASS.
- Installed-package import check: PASS.
- Three publication PDFs compiled: PASS.
- All 43 rendered PDF pages visually inspected: PASS.
- Undefined citation/reference and overfull-box scan: PASS.
- Raw data included: NO.
- Model checkpoints included: NO.

## Verified Runtime

- Python 3.12.13
- NumPy 2.4.6
- pandas 3.0.3
- SciPy 1.18.0
- scikit-learn 1.9.0
- XGBoost 3.3.0
- PyTorch 2.11.0+cu128
- CUDA available: yes
- GPU: NVIDIA GeForce RTX 5080

## Commands

- `python -m compileall -q src tests run_formal_dl_fusion_gpu.py run_postfix_neural_fusion_pipeline.py` -> exit 0.
- `python -m unittest discover -s tests -v` -> 19 tests passed.
- `python src/experiments/exp40_release_consistency_audit.py` -> 32 checks passed.
- `python run_postfix_neural_fusion_pipeline.py --help` -> exit 0.
- `python -m pip install --no-deps --no-build-isolation --target <temporary-install-dir> .` -> wheel built and installed successfully.
- Installed-package import check for split auditing, peer-review controls, the CNN module, and train-only normalization -> PASS.
- `xelatex` two-pass builds for `paper/main.tex` and `paper/supplementary.tex`, and a two-pass `pdflatex` build for `paper/main_machines.tex` -> exit 0.
- Poppler rendering at 110 dpi followed by visual inspection of every page -> no clipping, overlap, missing figure, or broken table continuation.

## Scope

The compact release supports manuscript, code, split, schema, and saved-evidence review. Re-running the complete numerical benchmark requires the original MCC5 dataset, which is not redistributed. Passing these checks does not establish independent physical bearing specimens, acquisition-session independence, or cross-machine generalization.
