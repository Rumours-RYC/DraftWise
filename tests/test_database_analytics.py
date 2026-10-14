import sqlite3
from pathlib import Path

import pandas as pd

from analytics import dataframe_summary, make_chart
from database import delete_dataset, init_db, list_datasets, load_dataframe, save_dataframe


def test_sqlite_dataset_persistence(tmp_path):
    db_path = tmp_path / "app.sqlite3"
    init_db(db_path)
    df = pd.DataFrame({"value": [1, 2, 3], "group": ["a", "b", "a"], "missing": [None, 2, None]})
    save_dataframe(db_path, "ds1", "sample", "sample.csv", "csv", "2026-10-14T00:00:00+00:00", df)
    loaded = load_dataframe(db_path, "ds1")
    assert loaded is not None
    meta, restored = loaded
    assert meta["row_count"] == 3
    assert meta["column_count"] == 3
    assert restored.shape == df.shape
    assert restored["value"].tolist() == [1, 2, 3]
    assert restored["group"].tolist() == ["a", "b", "a"]
    assert int(restored["missing"].isna().sum()) == 2
    assert len(list_datasets(db_path)) == 1
    assert delete_dataset(db_path, "ds1") is True
    assert load_dataframe(db_path, "ds1") is None


def test_summary_and_insights_use_real_data():
    df = pd.DataFrame({"score": [1.0, 2.0, 3.0], "kind": ["a", "b", "a"], "empty": [None, None, None]})
    summary = dataframe_summary(df)
    assert summary["rows"] == 3
    assert summary["columns"] == 3
    assert summary["missing_cells"] == 3
    assert any("3 rows" in item for item in summary["insights"])
    assert any("missing cells" in item for item in summary["insights"])


def test_charts_are_generated_from_given_data():
    df = pd.DataFrame({"score": [1.0, 2.0, 2.5, 5.0], "group": ["a", "b", "a", "c"], "other": [5, 4, 3, 2]})
    assert make_chart(df, "histogram", "score").startswith("data:image/png;base64,")
    assert make_chart(df, "bar", "group").startswith("data:image/png;base64,")
    assert make_chart(df, "scatter", "score", "other").startswith("data:image/png;base64,")
    assert make_chart(df, "correlation").startswith("data:image/png;base64,")
    assert make_chart(df, "missing").startswith("data:image/png;base64,")


def test_failed_duplicate_dataset_insert_does_not_leave_partial_rows(tmp_path):
    import pytest
    db_path = tmp_path / "rollback.sqlite3"
    init_db(db_path)
    df = pd.DataFrame({"value": [1, 2]})
    save_dataframe(db_path, "same-id", "first", "a.csv", "csv", "2026-10-14T00:00:00+00:00", df)
    with pytest.raises(Exception):
        save_dataframe(db_path, "same-id", "second", "b.csv", "csv", "2026-10-14T00:00:00+00:00", df)
    loaded = load_dataframe(db_path, "same-id")
    assert loaded is not None
    assert loaded[0]["name"] == "first"
    assert len(loaded[1]) == 2
    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM dataset_rows WHERE dataset_id = ?", ("same-id",)).fetchone()[0]
    assert count == 2
