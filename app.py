from __future__ import annotations

import os
import uuid
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from dotenv import load_dotenv
from flask import Flask, abort, flash, redirect, render_template, request, url_for

from analytics import dataframe_summary, make_chart
from database import delete_dataset, init_db, list_datasets, load_dataframe, save_dataframe
from upload_utils import read_dataset_bytes
from feature_builder import DraftValidationError, build_feature_row, get_hero_choices

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)
ARTIFACT_DIR = BASE_DIR / "artifacts"
BUNDLE_PATH = ARTIFACT_DIR / "dota2_draft_model_bundle_compressed.joblib"
MAX_ROWS = int(os.getenv("MAX_DATASET_ROWS", "100000"))

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "local-prototype-change-this-before-public-use")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MiB upload limit
app.config["DATABASE_PATH"] = os.getenv(
    "DATABASE_PATH", str(BASE_DIR / "instance" / "dota2_draft_app.sqlite3")
)

# The serialized estimators were trained with scikit-learn 1.6.1.
# Fail loudly rather than silently running unverified cross-version artifacts.
EXPECTED_SKLEARN_VERSION = "1.6.1"
if sklearn.__version__ != EXPECTED_SKLEARN_VERSION:
    raise RuntimeError(
        f"This model bundle was trained with scikit-learn {EXPECTED_SKLEARN_VERSION}; "
        f"the active environment has {sklearn.__version__}. Install requirements.txt in a clean environment."
    )

# Models load once during startup, not once per prediction.
if not BUNDLE_PATH.exists():
    raise FileNotFoundError(
        f"Model bundle not found at {BUNDLE_PATH}. Put the supplied bundle in artifacts/."
    )
with warnings.catch_warnings(record=True) as load_warnings:
    warnings.simplefilter("always")
    bundle = joblib.load(BUNDLE_PATH)
    BUNDLE_LOAD_WARNINGS = [str(w.message) for w in load_warnings]

required_keys = {"decision_tree", "random_forest", "feature_columns", "hero_constants", "hero_role_map"}
if not required_keys.issubset(bundle):
    raise ValueError(f"Model bundle is missing keys: {sorted(required_keys - set(bundle))}")
feature_columns = list(bundle["feature_columns"])
if len(feature_columns) != 2996 or len(set(feature_columns)) != 2996:
    raise ValueError("The saved model bundle must contain 2,996 unique feature columns.")
for model_key in ("decision_tree", "random_forest"):
    model = bundle[model_key]
    if int(model.n_features_in_) != 2996:
        raise ValueError(f"{model_key} does not expect 2,996 input features.")
    if list(getattr(model, "feature_names_in_", [])) != feature_columns:
        raise ValueError(f"{model_key} feature order does not match the saved schema.")

hero_options, pick_options_by_slot, ban_options_by_slot = get_hero_choices(bundle)
heroes_df = bundle["hero_constants"].copy()
init_db(app.config["DATABASE_PATH"])

RESEARCH_METRICS = [
    {"metric": "Accuracy", "baseline": "50.48%", "decision_tree": "51.27%", "random_forest": "53.38%"},
    {"metric": "Balanced accuracy", "baseline": "50.00%", "decision_tree": "50.88%", "random_forest": "53.40%"},
    {"metric": "Precision (Radiant-win class)", "baseline": "-", "decision_tree": "50.96%", "random_forest": "54.03%"},
    {"metric": "Recall (Radiant-win class)", "baseline": "-", "decision_tree": "92.32%", "random_forest": "51.32%"},
    {"metric": "F1-score (Radiant-win class)", "baseline": "-", "decision_tree": "65.67%", "random_forest": "52.64%"},
    {"metric": "Macro-F1", "baseline": "-", "decision_tree": "40.88%", "random_forest": "53.37%"},
]


def _read_upload(upload):
    filename = (upload.filename or "").replace("\\", "/").split("/")[-1].strip()
    return read_dataset_bytes(filename, upload.read(), max_rows=MAX_ROWS)


@app.errorhandler(413)
def upload_too_large(_error):
    flash("File is too large. The upload limit is 16 MiB.", "error")
    return redirect(url_for("index")), 303


@app.route("/")
def index():
    return render_template("index.html", datasets=list_datasets(app.config["DATABASE_PATH"]))


@app.route("/upload", methods=["POST"])
def upload_dataset():
    upload = request.files.get("file")
    if upload is None:
        flash("Choose a CSV or Excel file first.", "error")
        return redirect(url_for("index"))
    try:
        df, filename, note = _read_upload(upload)
        dataset_id = str(uuid.uuid4())
        chosen_name = (request.form.get("dataset_name") or "").strip() or Path(filename).stem
        save_dataframe(
            app.config["DATABASE_PATH"], dataset_id, chosen_name, filename,
            Path(filename).suffix.lower().lstrip("."), datetime.now(timezone.utc).isoformat(), df, note,
        )
    except Exception as exc:
        flash(str(exc), "error")
        return redirect(url_for("index"))
    flash(f"Imported {len(df):,} rows and {len(df.columns):,} columns.", "success")
    return redirect(url_for("dataset_detail", dataset_id=dataset_id))


@app.route("/dataset/<dataset_id>")
def dataset_detail(dataset_id: str):
    loaded = load_dataframe(app.config["DATABASE_PATH"], dataset_id)
    if loaded is None:
        abort(404)
    metadata, df = loaded
    summary = dataframe_summary(df)
    preview_columns = [str(c) for c in df.columns]
    preview_rows = [[_display_value(v) for v in row] for row in df.head(12).itertuples(index=False, name=None)]
    auto_charts: list[dict[str, str]] = []
    numeric_columns = list(df.select_dtypes(include="number").columns)
    nonnumeric_columns = list(df.select_dtypes(exclude="number").columns)
    try:
        if numeric_columns:
            auto_charts.append({"title": f"Distribution: {numeric_columns[0]}", "uri": make_chart(df, "histogram", numeric_columns[0])})
    except ValueError:
        pass
    try:
        if nonnumeric_columns:
            auto_charts.append({"title": f"Frequent values: {nonnumeric_columns[0]}", "uri": make_chart(df, "bar", nonnumeric_columns[0])})
    except ValueError:
        pass
    if summary["missing_cells"]:
        auto_charts.append({"title": "Missing values by column", "uri": make_chart(df, "missing")})
    return render_template(
        "dataset.html", metadata=metadata, summary=summary, columns=preview_columns,
        preview_rows=preview_rows, auto_charts=auto_charts, chart_columns=preview_columns,
    )


def _display_value(value):
    if pd.isna(value):
        return ""
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except (ValueError, TypeError):
            pass
    return str(value)


@app.route("/dataset/<dataset_id>/chart", methods=["POST"])
def custom_chart(dataset_id: str):
    loaded = load_dataframe(app.config["DATABASE_PATH"], dataset_id)
    if loaded is None:
        abort(404)
    metadata, df = loaded
    chart_type = request.form.get("chart_type", "histogram")
    x_column = request.form.get("x_column") or None
    y_column = request.form.get("y_column") or None
    try:
        chart_uri = make_chart(df, chart_type, x_column, y_column)
        error = None
    except ValueError as exc:
        chart_uri, error = None, str(exc)
    return render_template("chart.html", metadata=metadata, chart_uri=chart_uri, error=error,
                           chart_type=chart_type, x_column=x_column, y_column=y_column,
                           chart_columns=list(df.columns))


@app.route("/dataset/<dataset_id>/delete", methods=["POST"])
def remove_dataset(dataset_id: str):
    deleted = delete_dataset(app.config["DATABASE_PATH"], dataset_id)
    flash("Dataset deleted." if deleted else "Dataset was not found.", "success" if deleted else "error")
    return redirect(url_for("index"))


@app.route("/draft")
def draft_builder():
    selected = {field: "" for field in [*pick_options_by_slot, *ban_options_by_slot]}
    return render_template("draft.html", hero_options=hero_options,
                           pick_options_by_slot=pick_options_by_slot,
                           ban_options_by_slot=ban_options_by_slot,
                           selected=selected, result=None, errors=[], prediction_warnings=[])


@app.route("/predict", methods=["POST"])
def predict_draft():
    selected = {field: (request.form.get(field, "").strip()) for field in [*pick_options_by_slot, *ban_options_by_slot]}
    payload = {field: value or None for field, value in selected.items()}
    errors: list[str] = []
    prediction_warnings: list[str] = []
    result = None
    try:
        row, prediction_warnings = build_feature_row(
            payload, feature_columns, heroes_df, bundle["hero_role_map"]
        )
        dt_prediction = bool(bundle["decision_tree"].predict(row)[0])
        rf_prediction = bool(bundle["random_forest"].predict(row)[0])
        result = {
            "decision_tree": "Radiant" if dt_prediction else "Dire",
            "random_forest": "Radiant" if rf_prediction else "Dire",
            "feature_count": int(row.shape[1]),
            "role_counts": {col: int(row.at[0, col]) for col in (
                "radiant_disabler_count", "dire_disabler_count", "radiant_initiator_count",
                "dire_initiator_count", "radiant_pusher_count", "dire_pusher_count"
            )},
        }
    except DraftValidationError as exc:
        errors.append(str(exc))
    except Exception:
        app.logger.exception("Draft prediction failed")
        errors.append("Prediction failed. Check that the application artifacts and dependencies are compatible.")
    return render_template("draft.html", hero_options=hero_options,
                           pick_options_by_slot=pick_options_by_slot,
                           ban_options_by_slot=ban_options_by_slot,
                           selected=selected, result=result, errors=errors,
                           prediction_warnings=prediction_warnings)


@app.route("/research")
def research():
    rf = bundle["random_forest"]
    importance_values = dict(zip(feature_columns, rf.feature_importances_))
    pick_importance = sum(v for col, v in importance_values.items() if "_hero_" in col)
    ban_importance = sum(v for col, v in importance_values.items() if "_ban_" in col)
    role_importance = sum(v for col, v in importance_values.items() if col in {
        "radiant_disabler_count", "dire_disabler_count", "radiant_initiator_count",
        "dire_initiator_count", "radiant_pusher_count", "dire_pusher_count"
    })
    total = pick_importance + ban_importance + role_importance
    categories = [
        {"name": "Hero Bans", "value": ban_importance / total * 100},
        {"name": "Hero Picks", "value": pick_importance / total * 100},
        {"name": "Role Composition", "value": role_importance / total * 100},
    ]
    top_features = sorted(importance_values.items(), key=lambda item: item[1], reverse=True)[:15]
    return render_template("research.html", metrics=RESEARCH_METRICS, categories=categories,
                           top_features=top_features, load_warnings=BUNDLE_LOAD_WARNINGS)


@app.route("/healthz")
def healthz():
    return {"status": "ok", "feature_count": len(feature_columns), "models_loaded": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.getenv("PORT", "5000")), debug=False)
