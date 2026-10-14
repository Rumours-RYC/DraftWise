from io import BytesIO

import pandas as pd
import pytest

from upload_utils import read_dataset_bytes


def test_csv_upload_parses_rows_and_columns():
    raw = b"score,group\n1,A\n2,B\n3,A\n"
    df, filename, note = read_dataset_bytes("scores.csv", raw)
    assert filename == "scores.csv"
    assert note == "CSV file imported."
    assert df.shape == (3, 2)
    assert df["score"].tolist() == [1, 2, 3]


def test_xlsx_upload_uses_and_reports_first_sheet():
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        pd.DataFrame({"value": [4, 5]}).to_excel(writer, sheet_name="SheetA", index=False)
        pd.DataFrame({"value": [99]}).to_excel(writer, sheet_name="SheetB", index=False)
    df, filename, note = read_dataset_bytes("scores.xlsx", buf.getvalue())
    assert filename == "scores.xlsx"
    assert df["value"].tolist() == [4, 5]
    assert "SheetA" in note
    assert "first worksheet only" in note


def test_unsupported_type_and_empty_file_rejected():
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_dataset_bytes("file.txt", b"hello")
    with pytest.raises(ValueError, match="empty"):
        read_dataset_bytes("empty.csv", b"")


def test_row_limit_enforced():
    with pytest.raises(ValueError, match="current limit is 2 rows"):
        read_dataset_bytes("data.csv", b"x\n1\n2\n3\n", max_rows=2)
