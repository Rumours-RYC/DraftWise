"""SQLite persistence for imported datasets."""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@contextmanager
def db_session(path: str | Path):
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), timeout=20)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def init_db(path: str | Path) -> None:
    schema_path = Path(__file__).resolve().parent / "schema.sql"
    schema_sql = schema_path.read_text(encoding="utf-8")
    with db_session(path) as conn:
        conn.executescript(schema_sql)


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        val = float(value)
        return val if math.isfinite(val) else None
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return value.isoformat()
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    return str(value)


def make_unique_columns(columns: list[Any]) -> list[str]:
    """Preserve every uploaded column while making names unique for JSON storage."""
    used: dict[str, int] = {}
    out: list[str] = []
    for raw in columns:
        base = str(raw).strip() or "unnamed_column"
        count = used.get(base, 0) + 1
        used[base] = count
        candidate = base if count == 1 else f"{base}_{count}"
        while candidate in out:
            count += 1
            used[base] = count
            candidate = f"{base}_{count}"
        out.append(candidate)
    return out


def save_dataframe(
    path: str | Path,
    dataset_id: str,
    name: str,
    original_filename: str,
    file_type: str,
    uploaded_at: str,
    df: pd.DataFrame,
    import_note: str = "",
) -> None:
    columns = [str(c) for c in df.columns]
    dtypes = {str(c): str(df[c].dtype) for c in df.columns}
    records = []
    for row in df.itertuples(index=False, name=None):
        records.append(json.dumps(
            {columns[i]: _json_value(value) for i, value in enumerate(row)},
            ensure_ascii=False, allow_nan=False,
        ))
    with db_session(path) as conn:
        # A single transaction ensures that failed inserts do not leave partial imports.
        conn.execute("""INSERT INTO datasets
            (dataset_id, name, original_filename, file_type, uploaded_at,
             row_count, column_count, columns_json, dtypes_json, import_note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (dataset_id, name, original_filename, file_type, uploaded_at,
             int(len(df)), int(len(df.columns)), json.dumps(columns),
             json.dumps(dtypes), import_note))
        conn.executemany(
            "INSERT INTO dataset_rows (dataset_id, row_index, row_json) VALUES (?, ?, ?)",
            [(dataset_id, i, record) for i, record in enumerate(records)],
        )


def list_datasets(path: str | Path) -> list[dict[str, Any]]:
    with db_session(path) as conn:
        rows = conn.execute("""SELECT dataset_id, name, original_filename, file_type,
            uploaded_at, row_count, column_count, import_note
            FROM datasets ORDER BY uploaded_at DESC""").fetchall()
    return [dict(row) for row in rows]


def get_dataset_metadata(path: str | Path, dataset_id: str) -> dict[str, Any] | None:
    with db_session(path) as conn:
        row = conn.execute("SELECT * FROM datasets WHERE dataset_id = ?", (dataset_id,)).fetchone()
    if row is None:
        return None
    result = dict(row)
    result["columns"] = json.loads(result.pop("columns_json"))
    result["dtypes"] = json.loads(result.pop("dtypes_json"))
    return result


def load_dataframe(path: str | Path, dataset_id: str) -> tuple[dict[str, Any], pd.DataFrame] | None:
    metadata = get_dataset_metadata(path, dataset_id)
    if metadata is None:
        return None
    with db_session(path) as conn:
        rows = conn.execute(
            "SELECT row_json FROM dataset_rows WHERE dataset_id = ? ORDER BY row_index",
            (dataset_id,),
        ).fetchall()
    records = [json.loads(row["row_json"]) for row in rows]
    df = pd.DataFrame.from_records(records, columns=metadata["columns"])
    # Restore common numeric and datetime types for meaningful charts and summaries.
    for col, dtype in metadata["dtypes"].items():
        if col not in df.columns:
            continue
        try:
            if "int" in dtype.lower() or "float" in dtype.lower() or "double" in dtype.lower():
                df[col] = pd.to_numeric(df[col], errors="coerce")
            elif "datetime" in dtype.lower():
                df[col] = pd.to_datetime(df[col], errors="coerce")
            elif dtype.lower() in {"bool", "boolean"}:
                df[col] = df[col].astype("boolean")
        except (TypeError, ValueError):
            # Keep the faithfully stored values if a conversion is not possible.
            pass
    return metadata, df


def delete_dataset(path: str | Path, dataset_id: str) -> bool:
    with db_session(path) as conn:
        cur = conn.execute("DELETE FROM datasets WHERE dataset_id = ?", (dataset_id,))
        return cur.rowcount > 0
