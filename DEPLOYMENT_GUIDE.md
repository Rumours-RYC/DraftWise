# Deployment and GitHub checklist

This guide documents the deployment steps; it does not claim that any public repository or deployment already exists.

## 1. Create and push the GitHub repository

From PowerShell in the `Dota2DraftApp` project folder, after reviewing the files:

```powershell
git init
git add .
git status
git commit -m "Initial Dota 2 draft analytics application"
```

Create an empty repository in your own GitHub account, then use the exact remote URL shown by GitHub:

```powershell
git branch -M main
git remote add origin <YOUR_GITHUB_REPOSITORY_URL>
git push -u origin main
```

Replace the placeholder with the real repository URL. Do not paste an invented URL into the submission form. Confirm the repository page opens while signed out or in a private browser window if it is meant to be public.

The compressed model bundle is about 39 MB and is below GitHub's standard 100 MB single-file limit. Never commit `.env`, live SQLite databases, or uploaded user data.

## 2. Deploy on a host with persistent storage

The live application must preserve SQLite data across restarts and redeploys. An ephemeral filesystem is not sufficient.

A possible option is Railway with the included Dockerfile. Before creating the service, confirm current resource limits and pricing in the Railway dashboard. A preparation-workspace measurement peaked around 590 MiB during model loading; use at least 1 GiB RAM and one web worker for the initial demonstration. Do not deploy this model bundle on a 512 MiB instance without an actual successful memory test.

Steps:

1. Create/sign in to your hosting account.
2. Deploy the GitHub repository as a Docker-based service using the included `Dockerfile`.
3. Configure the service with a persistent volume mounted at `/data`.
4. Set these service variables:
   - `DATABASE_PATH=/data/dota2_draft_app.sqlite3`
   - `SECRET_KEY` to a randomly generated long secret
   - `MAX_DATASET_ROWS=100000`
5. Use one web worker at first so the saved model bundle is not loaded multiple times into separate worker processes.
6. Wait for deployment logs to show a successful start. Resolve any dependency/build/startup errors; don't skip the scikit-learn version check.
7. Open `https://YOUR-REAL-HOSTNAME/healthz`. A successful response should include `status: ok`, `feature_count: 2996`, and `models_loaded: true`.
8. In the deployed app, upload a small CSV and an XLSX, create a chart, and run a Dota 2 draft prediction.
9. Restart or redeploy the service, then verify the uploaded dataset still appears. If it disappears, the database is not persisting correctly.
10. Test the real public URL in a private browser window and submit that exact URL.

A paid or trial resource may be required for sufficient memory and persistent storage. Check account-specific billing before enabling paid resources; no plan has been purchased by this project scaffold.

## 3. Final acceptance checklist

- [ ] Public GitHub URL opens and contains source, tests, dependencies, `schema.sql`, README, and documentation PDF.
- [ ] Public application URL opens.
- [ ] `/healthz` returns the expected JSON.
- [ ] CSV import works and data remains after restart.
- [ ] XLSX import works and names the worksheet used.
- [ ] Dataset summary and charts render.
- [ ] Dataset deletion removes its metadata and rows.
- [ ] Draft prediction returns both model classes.
- [ ] A reference feature-row comparison with the notebook has been performed before claiming exact preprocessing parity.
- [ ] No secrets or live uploaded data are in GitHub.
