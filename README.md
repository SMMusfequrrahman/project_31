# Supplementary code: cross-dataset wearable stress detection (FTS workflow 7.0.0)

This package contains everything needed to reproduce the analysis from the raw
datasets: the analysis code, its tests, the protocol evidence used for task
labelling, and the notebook that runs the steps in order.

The notebook `FTS_analysis.ipynb` contains the outputs of run
`fts_complete_7_d27bf7a0f70a3284` (review archive SHA-256
`d8d9f4e1ac0493d325be7752672d06ff5e3421f4e7877410589d68e874a6561f`).

## Contents

| Path | Purpose |
|---|---|
| `FTS_analysis.ipynb` | Runs the 19 steps in order. Holds no analysis logic of its own: it fixes the settings, builds the environment and calls the scripts below. |
| `requirements.txt` | Exact package versions of the isolated environment (Python 3.11–3.13). |
| `SHA256SUMS.txt` | SHA-256 of every file below. The notebook verifies it before running. |
| `code/prepare_features.py` | Raw-signal reading, 60-second windowing and feature extraction, shared by both datasets. |
| `code/common.py` | Shared loading and validation helpers. |
| `code/audit.py` | Interval-manifest validation and feature-quality summaries. |
| `code/recovery.py` | Raw-signal quality audit, quality flags and the quality-sensitivity analysis. |
| `code/complete_workflow.py` | Workflow steps: label manifest, provenance snapshot, audit, quality policy, reconciliation of both analyses, review archive. |
| `code/models.py` | Nested participant-grouped cross-validation, feature selection (FTS and baselines), threshold development, model locking, target evaluation. |
| `code/statistics_report.py` | Metrics, participant-level bootstrap intervals, paired comparisons, exercise/rest rates, tables and figures. |
| `code/explain.py` | Interventional TreeSHAP on the locked models. |
| `code/smoke_test.py`, `code/run_software_checks.py` | End-to-end integration test on synthetic data and the unit-test runner. |
| `code/reuse_features.py` | Utility for copying verified extraction files from an earlier run. Not called by the notebook. |
| `tests/` | Unit tests (run in notebook Cell 5). |
| `assets/historical_manifest.csv` | Interval manifest of the PhysioNet recordings (tag indices, times, task labels, inclusion). |
| `assets/protocol_evidence/` | Unmodified protocol figures and mapping code from the official PhysioNet notebook, with a README stating their source, licence and a known limitation. |
| `tools/finish_from_review_zip.py` | Standard-library-only checker that recomputes the reported metrics from the predictions stored in the review archive (notebook Cell 18). |
| `tools/same_session_rest_check.py` | Post hoc supplementary check scoring the locked models on post-exercise rest (notebook Cell 19). |

## Data

The datasets are not redistributed. Obtain them from the original providers:

- **WESAD** (Schmidt et al., 2018), source dataset. Expected layout: `WESAD/S2/S2.pkl`, ...
- **Wearable Device Dataset from Induced Stress and Structured Exercise Sessions v1.0.1**
  (Hongn et al., 2025), target dataset: https://physionet.org/content/wearable-device-dataset/1.0.1/
  Expected layout: `PhysioNet_Dataset/STRESS/`, `PhysioNet_Dataset/AEROBIC/`.

## How to run

1. Copy this folder to Google Drive as `MyDrive/FTS_supplement`.
2. Extract the datasets to `MyDrive/WESAD` and `MyDrive/PhysioNet_Dataset`
   (or edit the three paths at the top of notebook Cell 1).
3. Open `FTS_analysis.ipynb` in a CPU Colab runtime and choose *Runtime → Run all*.

Results are written to `MyDrive/FTS_publication_reanalysis/complete_runs/<run id>/`.
The run id is derived from the settings, environment and dataset paths, so a
different Python version or path produces a new, separate run folder.

## Notebook steps

| Cell | Step | Script |
|---|---|---|
| 1 | Settings, paths, run folder | – |
| 2 | Isolated pinned environment | – |
| 3 | Verify and install this package | – |
| 4 | Stage runner (hash-checked execution) | – |
| 5 | Software checks | `run_software_checks.py` |
| 6 | Dataset paths and official task mappings | `complete_workflow.py manifest` |
| 7 | Feature extraction | `prepare_features.py` |
| 8 | Provenance and window audit | `complete_workflow.py snapshot`, `audit` |
| 9 | Uniform quality policy | `complete_workflow.py prepare` |
| 10 | Reference model development (source only) | `models.py develop` |
| 11 | Target evaluation with locked models | `models.py evaluate` |
| 12 | Reference tables and figures | `statistics_report.py` |
| 13 | SHAP on locked models | `explain.py` |
| 14 | Quality-sensitivity analysis | `complete_workflow.py quality` |
| 15 | Reconcile both analyses | `complete_workflow.py summary` |
| 16 | Figures | – |
| 17 | Verify stages and write review archive | `complete_workflow.py archive` |
| 18 | Recompute reported numbers from the archive | `tools/finish_from_review_zip.py` |
| 19 | Supplementary same-session rest check | `tools/same_session_rest_check.py` |

## Integrity

- Check the package: `sha256sum -c SHA256SUMS.txt`.
- The notebook fixes the SHA-256 of `SHA256SUMS.txt`, so the notebook and the
  code cannot drift apart unnoticed.
- Every stage records the SHA-256 of its inputs (data, code, settings,
  environment) and outputs. Completed stages are reused only if all hashes
  match; otherwise execution stops. Recorded files are never overwritten.

## Scope

The analysis is retrospective and target-informed: the evaluation cohort had
been inspected before this workflow was fixed. The signal-quality thresholds are
heuristic. The unscreened reference analysis and the quality-sensitivity
analysis are both produced and are to be reported together.
