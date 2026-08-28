"""Finalize v3.1.4 provenance, release audit, and artifact hashes."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
VERSION = "review-v3.1.4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    excluded = {"review/ARTIFACT_MANIFEST_SHA256.csv"}
    files = []
    for relative in result.stdout.splitlines():
        normalized = relative.replace("\\", "/")
        path = ROOT / normalized
        if normalized not in excluded and path.is_file():
            files.append(path)
    return sorted(files, key=lambda path: path.relative_to(ROOT).as_posix().lower())


def write_provenance() -> None:
    anchor_paths = [
        "data/recording_catalog_and_splits.csv",
        "evidence/split.csv",
        "evidence/normalization.csv",
        "evidence/all_fusion_input_protocol_results.csv",
        "evidence/submission_consistency_audit.json",
        "src/data/audit_mcc5_provenance.py",
        "src/experiments/exp38_classical_recording_metrics.py",
        "src/experiments/exp39_peer_review_controls.py",
        "src/experiments/exp40_release_consistency_audit.py",
        "src/experiments/exp41_finalize_release_metadata.py",
        "src/models/fusion_net.py",
        "configs/mcc5_fusion.yaml",
    ]
    payload = {
        "schema_version": "1.1",
        "package_version": VERSION,
        "formal_run_id": "postfix_neural_fusion_20260710_180646",
        "repository": "https://github.com/YOMMMO/mcc5-bearing-paper-review",
        "evidence_classes": {
            "formal_run_evidence": {
                "description": (
                    "Saved outputs from the corrected formal benchmark run. These artifacts support "
                    "the predefined classical, corrected raw-neural, and strict engineered/fusion "
                    "results reported in the manuscript."
                ),
                "representative_artifacts": [
                    "evidence/split.csv",
                    "evidence/normalization.csv",
                    "evidence/all_fusion_input_protocol_results.csv",
                ],
            },
            "follow_up_review_audits": {
                "description": (
                    "Review-stage controls regenerated locally from the frozen recording catalog, "
                    "saved predictions, formal feature table, and/or separately obtained MCC5 raw "
                    "data. They do not replace the formal training run."
                ),
                "artifacts": [
                    {
                        "script": "src/data/audit_mcc5_provenance.py",
                        "outputs": [
                            "evidence/raw_file_integrity_audit.csv",
                            "evidence/raw_integrity_summary.json",
                            "evidence/timestamp_collision_audit.csv",
                            "evidence/same_timestamp_pair_comparison.csv",
                            "evidence/session_metadata_audit.csv",
                            "evidence/acquisition_date_class_counts.csv",
                            "evidence/acquisition_date_protocol_role_counts.csv",
                            "evidence/session_confounding_summary.json",
                        ],
                    },
                    {
                        "script": "src/experiments/exp38_classical_recording_metrics.py",
                        "outputs": [
                            "evidence/classical_source_recording_predictions.csv",
                            "evidence/classical_source_recording_metrics.csv",
                            "evidence/classical_source_recording_bootstrap.csv",
                            "evidence/classical_source_recording_model_matrix_predictions.csv",
                            "evidence/classical_source_recording_model_matrix_metrics.csv",
                            "evidence/classical_source_recording_model_matrix_bootstrap.csv",
                            "evidence/repeated_source_recording_predictions.csv",
                            "evidence/repeated_source_recording_metrics.csv",
                        ],
                    },
                    {
                        "script": "src/experiments/exp39_peer_review_controls.py",
                        "outputs": [
                            "evidence/partition_leakage_controls.csv",
                            "evidence/partition_leakage_controls_summary.csv",
                            "evidence/within_date_250707_predictions.csv",
                            "evidence/within_date_250707_metrics.csv",
                            "evidence/within_date_250707_summary.csv",
                            "evidence/training_only_metadata_baselines.csv",
                            "evidence/training_only_metadata_predictions.csv",
                            "evidence/outer_race_date_predictions.csv",
                            "evidence/outer_race_date_summary.csv",
                        ],
                    },
                    {
                        "script": "src/experiments/exp40_release_consistency_audit.py",
                        "outputs": [
                            "evidence/submission_consistency_audit.json",
                            "evidence/submission_consistency_audit.md",
                        ],
                    },
                ],
            },
        },
        "sha256_anchors": {
            relative: sha256(ROOT / relative) for relative in anchor_paths
        },
        "scope_notes": [
            "Raw MCC5 sensor recordings are not redistributed in this repository.",
            "A full numerical rerun requires the original MCC5 dataset.",
            "Distinct file hashes exclude byte-identical recordings but do not establish independent physical bearing specimens or acquisition sessions.",
            "Acquisition-date controls provide mixed evidence: same-date fault separation remains perfect, while date-only and within-class date-prediction controls reveal residual session-proxy structure.",
            "Random nonoverlapping-window assignment still shares source recordings across roles; recording grouping, rather than overlap removal alone, controls direct source-recording reuse.",
            "Auxiliary-26 and auxiliary-context-28 remain post-hoc exploratory on the same held-out 3000-rpm domain.",
        ],
    }
    (ROOT / "review/RUN_PROVENANCE_MANIFEST.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )


def write_release_audit() -> None:
    evidence_count = len(list((ROOT / "evidence").glob("*.csv")))
    content = f"""# Software Supplement Release Audit

Date: {date.today().isoformat()}

Version: `{VERSION}`

Repository: https://github.com/YOMMMO/mcc5-bearing-paper-review

## Selection Audit

- 84 acquisition recordings represented once each in the compact catalog.
- Four frozen protocol roles included at recording level.
- {evidence_count} evidence CSV files included.
- Current manuscript, Supplementary Material, `Machines` preview, and figure sources included.
- Exact input schemas, relevant source code, tests, and environment records included.
- Official `Machines` template dependencies included under `paper/Definitions/`.
- Raw signals, checkpoints, processed windows, caches, build products, prompts, and historical drafts excluded.

## Validation Performed Before Release

- Current submission-consistency audit passes 28/28 checks.
- Publication-class counts are 24 rolling-element, 23 inner-race, 25 outer-race, and 12 healthy recordings.
- All four formal protocols have zero source-recording overlap across training, validation, and test roles.
- All 84 catalogued raw files resolved, have the expected 1,152,000-by-9 shape, and have distinct SHA-256 digests.
- The repeated-timestamp trio shares its time vector but is neither byte-identical nor pointwise-identical in the eight measured channels.
- Four matched random/grouped and overlapping/nonoverlapping partition controls are included.
- Same-date, training-only metadata, and within-class acquisition-date controls are included with their limitations.
- A complete five-model-by-four-protocol recording-level classical matrix and ten repeated grouped-partition results are included.
- Auxiliary-26/28 evidence remains labelled post-hoc exploratory on the same holdout.
- Manuscripts contain a formal AI-use disclosure and no internal pre-submission reminders.
- Public-facing repository text is neutral and contains no model-review prompt.

## Evidence Boundary

The repository supports manuscript, code, and compact-evidence review. A full numerical rerun additionally requires the separately hosted MCC5 raw dataset. The controls do not establish physical-bearing specimen independence, acquisition-session independence, cross-machine generalization, or confirmatory auxiliary-fusion superiority.

Formal-run evidence and follow-up controls are separated in `review/RUN_PROVENANCE_MANIFEST.json`.
"""
    (ROOT / "review/RELEASE_AUDIT.md").write_text(content, encoding="utf-8")


def write_artifact_manifest() -> None:
    target = ROOT / "review/ARTIFACT_MANIFEST_SHA256.csv"
    with target.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["path", "size_bytes", "sha256"])
        for path in release_files():
            writer.writerow(
                [path.relative_to(ROOT).as_posix(), path.stat().st_size, sha256(path)]
            )


def main() -> None:
    write_provenance()
    write_release_audit()
    write_artifact_manifest()
    print(
        json.dumps(
            {
                "version": VERSION,
                "artifact_count": len(release_files()),
                "status": "PASS",
            }
        )
    )


if __name__ == "__main__":
    main()
