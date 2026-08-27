# Reproduction Test Report

- Compile check: PASS
- Normalization and release-integrity unit tests: PASS
- Frozen recording-group split audit: PASS
- Runner command-line smoke check: PASS
- Isolated package installation: PASS
- Installed-package import check: PASS
- Forbidden local/prompt text scan: PASS
- Raw data included: NO
- Model checkpoints included: NO

## Commands

- `python -m compileall -q src tests run_formal_dl_fusion_gpu.py run_postfix_neural_fusion_pipeline.py` -> exit 0
- `python -m unittest discover -s tests -v` -> exit 0
- `python run_postfix_neural_fusion_pipeline.py --help` -> exit 0
- `python -m pip install --no-deps --target <temporary-install-dir> .` -> exit 0
- `python -c import src.data.audit_splits; import src.models.cnn1d; import src.utils.train_only_normalization; print('release imports: PASS')` -> exit 0
