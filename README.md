# DraftWise Analytics & Dota 2 Draft Prediction

A Flask research prototype with two distinct modules:

1. General CSV/XLSX import, SQLite persistence, data profiling, descriptive insights, and charts.
2. Dota 2 completed-draft predictions using the exported Decision Tree and Random Forest models.

## Important research constraints

- The trained models were built on 29,643 professional Captains Mode matches from the 2023 subset.
- The saved input schema contains exactly 2,996 ordered columns.
- The app uses the original schema order and metadata from the supplied model bundle.
- General uploaded datasets are for analytics only. They are not automatically fed to the Dota 2 classifiers.
- Model outputs are class predictions, not calibrated probabilities; draft-only predictions do not guarantee match outcomes.

## Project files

- `app.py`: Flask routes, upload workflow, and prediction requests.
- `feature_builder.py`: draft validation, role counts, and feature-row construction.
- `analytics.py`: descriptive summary, deterministic insights, and chart creation.
- `upload_utils.py`: independently testable CSV/XLSX parsing and validation.
- `database.py`: SQLite schema and dataset persistence.
- `artifacts/dota2_draft_model_bundle_compressed.joblib`: supplied trained model/metadata bundle.
- `artifacts/feature_columns.json`: saved ordered feature schema.
- `tests/`: automated tests.

## Python compatibility

The saved scikit-learn estimators were trained using scikit-learn 1.6.1. Use the supplied version pin rather than a different scikit-learn release. Python 3.12 is recommended for the packaged deployment setup.

## Run locally (Windows PowerShell)

Open PowerShell in the project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
python app.py
```

Then visit <http://127.0.0.1:5000>.

If `py -3.12` is unavailable, install Python 3.12 and try again. Do not run the app with a scikit-learn version that produces a serialization compatibility warning.

## Environment variables

- `SECRET_KEY`: set a random secret outside source control for hosted use.
- `DATABASE_PATH`: SQLite file location; for a persistent deployment volume, set this to the mounted volume path (for example `/data/dota2_draft_app.sqlite3`).
- `MAX_DATASET_ROWS`: default `100000` maximum rows accepted per import.
- `PORT`: hosting-provided HTTP port; local default is `5000`.

## CSV/XLSX import behavior

- Supports `.csv` and `.xlsx` files up to 16 MiB and 100,000 rows.
- Excel imports use the first worksheet; the selected worksheet is named on the dataset page.
- Imported rows and dataset metadata are stored in SQLite. The app initializes its tables automatically.
- Column names are made unique if an uploaded file contains duplicate headers so values are not overwritten during storage.
- Uploads are treated only as data; macros or embedded code are not executed.

## Deployment note

The application requires persistent storage for SQLite. Before deploying, configure `DATABASE_PATH` to a mounted persistent volume. Do not use an ephemeral filesystem for the live database, because records could disappear after a restart or redeployment. Confirm hosting costs and account requirements before enabling a paid service.

For a container-based host, build with the included `Dockerfile`. The app binds to the host-provided `PORT`. Configure one worker initially to avoid loading duplicate model copies into memory. A measurement in the preparation workspace peaked around 590 MiB while loading the serialized bundle; choose a deployment instance with at least 1 GiB RAM and verify actual runtime memory.

Railway currently documents persistent volumes and a 1 GiB RAM Trial service limit; its free service plan lists 0.5 GB RAM. The 0.5 GB tier is not recommended for this model bundle based on the local memory measurement. Trial duration, credit availability, and billing can change, so check the live plan and cost in your own account before deploying. Attach a persistent volume mounted at `/data` and set `DATABASE_PATH=/data/dota2_draft_app.sqlite3`. Do not enable a paid plan without your approval. Only claim deployment is complete after the public URL is opened and tested.

## Verification checklist

- [ ] Install dependencies with the pinned `scikit-learn==1.6.1`.
- [ ] Run `python -m pytest -q` and save the actual output.
- [ ] Start the app and visit `/healthz`.
- [ ] Import a small CSV and XLSX file, verify the row/column counts, restart the app, and reopen the datasets.
- [ ] Generate a chart and read the data-derived insights.
- [ ] Submit a valid draft and inspect predictions from both models.
- [ ] Compare a known draft feature row/prediction with the notebook before claiming full preprocessing parity.
- [ ] Configure persistent database storage, deploy, and open every required page in the public deployment.
- [ ] Push the repository to GitHub and verify the repository link is accessible.
