"""Integration tests run when Flask and the pinned ML dependencies are installed."""
from __future__ import annotations

import importlib
import os
import sys
from io import BytesIO
from pathlib import Path

import pytest

pytest.importorskip("flask")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def app_module(tmp_path_factory):
    db_path = tmp_path_factory.mktemp("flask-app") / "test.sqlite3"
    previous = os.environ.get("DATABASE_PATH")
    os.environ["DATABASE_PATH"] = str(db_path)
    try:
        if "app" in sys.modules:
            module = importlib.reload(sys.modules["app"])
        else:
            module = importlib.import_module("app")
        module.app.config.update(TESTING=True)
        yield module
    finally:
        if previous is None:
            os.environ.pop("DATABASE_PATH", None)
        else:
            os.environ["DATABASE_PATH"] = previous


def test_pages_and_health(app_module):
    client = app_module.app.test_client()
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/draft").status_code == 200
    assert client.get("/research").status_code == 200


def test_upload_database_chart_and_delete(app_module):
    client = app_module.app.test_client()
    response = client.post("/upload", data={
        "dataset_name": "Test dataset",
        "file": (BytesIO(b"score,group\n1,A\n2,B\n3,A\n"), "demo.csv"),
    }, content_type="multipart/form-data", follow_redirects=True)
    assert response.status_code == 200
    assert b"Test dataset" in response.data

    datasets = app_module.list_datasets(app_module.app.config["DATABASE_PATH"])
    saved = next(item for item in datasets if item["name"] == "Test dataset")
    dataset_id = saved["dataset_id"]
    detail = client.get(f"/dataset/{dataset_id}")
    assert detail.status_code == 200
    assert b"Data preview" in detail.data

    chart = client.post(f"/dataset/{dataset_id}/chart", data={
        "chart_type": "histogram", "x_column": "score", "y_column": ""
    })
    assert chart.status_code == 200
    assert b"Generated chart" in chart.data

    deleted = client.post(f"/dataset/{dataset_id}/delete", follow_redirects=True)
    assert deleted.status_code == 200
    assert app_module.load_dataframe(app_module.app.config["DATABASE_PATH"], dataset_id) is None


def test_draft_prediction_route(app_module):
    from feature_builder import get_hero_choices

    bundle = app_module.bundle
    _, pick_options, ban_options = get_hero_choices(bundle)
    data = {}
    picks_used = set()
    for side in ("radiant", "dire"):
        for slot in range(1, 6):
            field = f"{side}_hero_{slot}"
            hero = next(h for h in pick_options[field] if h["id"] not in picks_used)
            data[field] = str(hero["id"])
            picks_used.add(hero["id"])
    bans_used = set()
    for side in ("radiant", "dire"):
        for slot in range(1, 8):
            field = f"{side}_ban_{slot}"
            hero = next((h for h in ban_options[field]
                         if h["id"] not in picks_used and h["id"] not in bans_used), None)
            data[field] = str(hero["id"]) if hero else ""
            if hero:
                bans_used.add(hero["id"])
    response = app_module.app.test_client().post("/predict", data=data)
    assert response.status_code == 200
    assert b"Decision Tree" in response.data
    assert b"Random Forest" in response.data
    assert b"2,996" in response.data
