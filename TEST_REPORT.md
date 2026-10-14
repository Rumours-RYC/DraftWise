# Preparation Workspace Test Report

Date: 2026-10-14

## Checks completed here

- Python source compile (`python -m compileall -q .`): PASS.
- Helper test suite (`python -m pytest -q`): 14 passed, 1 skipped.
- All Jinja templates parsed successfully: PASS.
- SQLite database initialization, persistence, deletion and failed-insert rollback tests: PASS.
- CSV parsing and XLSX first-sheet parsing tests: PASS.
- Summary/insight generation and histogram, bar, scatter, correlation, and missing-value chart tests: PASS.
- Exported model bundle inspection: both model objects exist; feature schema has 2,996 unique names and exactly matches the feature names recorded by both estimators.
- Model's calculated category importance: bans 56.22199%, picks 41.01648%, role composition 2.76152%, matching the rounded figures reported in the paper.
- Research documentation PDF: generated from the current DOCX, 6 pages, opened and visually checked page-by-page; no visible clipping or overlap found.

## Checks not completed here

- Flask route/integration tests were skipped because Flask is not installed in this preparation workspace and external package installation is unavailable here.
- The serialized estimators warn when loaded under this workspace's scikit-learn 1.8.0 because they were saved with scikit-learn 1.6.1. The application deliberately checks for version 1.6.1 and refuses to start under another version. Install the pinned `requirements.txt` in a clean environment before running the app.
- No comparison against a saved notebook reference feature row/prediction has been completed. The feature builder follows the inspected notebook's exact encoding conventions, but parity should not be claimed until a specific reference row is compared.
- No public GitHub repository or application deployment has been created. This requires the project owner's account authorization and final live testing.

## Required next checks on the target computer

1. Install Python 3.12 and `requirements.txt` into a clean virtual environment.
2. Run `python -m pytest -q`; make sure Flask integration tests run rather than skip.
3. Start the app and test `/healthz`, CSV import, XLSX import, SQLite persistence after restart, charts, deletion, and draft predictions.
4. Compare a known test draft from the notebook against the application's features/predictions.
5. Deploy with persistent SQLite storage and at least 1 GiB RAM; verify the public URL and persistence.
