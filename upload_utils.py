"""CSV/XLSX parsing helpers kept independent of Flask for testing."""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pandas as pd

from database import make_unique_columns

ALLOWED_EXTENSIONS = {".csv", ".xlsx"}


def read_dataset_bytes(filename: str, raw: bytes, max_rows: int = 100_000) -> tuple[pd.DataFrame, str, str]:
    safe_filename = (filename or "").replace("\\", "/").split("/")[-1].strip()
    if not safe_filename:
        raise ValueError("Choose a file to upload.")
    extension = Path(safe_filename).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise ValueError("Unsupported file type. Upload a .csv or .xlsx file.")
    if not raw:
        raise ValueError("The uploaded file is empty.")
    try:
        if extension == ".csv":
            df = pd.read_csv(io.BytesIO(raw), low_memory=False)
            note = "CSV file imported."
        else:
            excel = pd.ExcelFile(io.BytesIO(raw), engine="openpyxl")
            if not excel.sheet_names:
                raise ValueError("The Excel workbook contains no worksheets.")
            sheet_name = excel.sheet_names[0]
            df = pd.read_excel(excel, sheet_name=sheet_name)
            note = f"Imported worksheet '{sheet_name}'. This prototype imports the first worksheet only."
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"The file could not be read as a valid {extension[1:].upper()} dataset: {exc}") from exc
    if df.shape[1] == 0:
        raise ValueError("The file has no columns to analyze.")
    if len(df) == 0:
        raise ValueError("The file has no data rows.")
    if len(df) > max_rows:
        raise ValueError(f"The file has {len(df):,} rows. The current limit is {max_rows:,} rows.")
    df = df.copy()
    df.columns = make_unique_columns(list(df.columns))
    return df, safe_filename, note
