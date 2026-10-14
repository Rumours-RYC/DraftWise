"""Data summaries, deterministic insights and chart generation for uploaded datasets."""
from __future__ import annotations

import base64
import io
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def dataframe_summary(df: pd.DataFrame) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    categorical = df.select_dtypes(exclude=[np.number])
    missing = df.isna().sum().sort_values(ascending=False)
    numeric_summary = numeric.describe().T.reset_index().rename(columns={"index": "column"})
    if len(numeric_summary):
        numeric_summary = numeric_summary.head(12).replace({np.nan: None})
    duplicate_rows = int(df.duplicated().sum()) if len(df) else 0
    column_info = []
    for name in df.columns:
        column_info.append({
            "name": str(name),
            "dtype": str(df[name].dtype),
            "missing": int(df[name].isna().sum()),
            "missing_pct": round(float(df[name].isna().mean() * 100), 2) if len(df) else 0,
            "unique": int(df[name].nunique(dropna=True)),
        })

    insights = [f"The dataset contains {len(df):,} rows and {len(df.columns):,} columns."]
    if len(df):
        miss_count = int(df.isna().sum().sum())
        if miss_count:
            insights.append(f"There are {miss_count:,} missing cells across {int((df.isna().sum() > 0).sum())} columns.")
        else:
            insights.append("No missing cells were detected in the imported data.")
    if duplicate_rows:
        insights.append(f"There are {duplicate_rows:,} completely duplicated rows. Inspect them before deciding whether to remove them.")
    else:
        insights.append("No completely duplicated rows were detected.")
    if len(numeric.columns):
        insights.append(f"{len(numeric.columns)} columns were recognized as numeric and can be summarized or plotted with distributions.")
        if len(numeric.columns) >= 2 and len(df) >= 3:
            corr = numeric.corr(numeric_only=True).abs()
            upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
            pairs = upper.stack().sort_values(ascending=False)
            pairs = pairs[(pairs < 0.999999) & pairs.notna()]
            if len(pairs):
                (col_a, col_b), strength = pairs.index[0], float(pairs.iloc[0])
                insights.append(f"The strongest non-perfect absolute numeric correlation is between '{col_a}' and '{col_b}' (|r|={strength:.2f}). Correlation does not establish causation.")
    if len(categorical.columns):
        insights.append(f"{len(categorical.columns)} columns were recognized as non-numeric/categorical or other types.")
        for name in categorical.columns[:3]:
            counts = df[name].dropna().astype(str).value_counts()
            if len(counts):
                insights.append(f"Most frequent value in '{name}': {counts.index[0]} ({int(counts.iloc[0]):,} records).")
    return {
        "rows": len(df), "columns": len(df.columns), "numeric_columns": len(numeric.columns),
        "other_columns": len(categorical.columns), "missing_cells": int(df.isna().sum().sum()),
        "duplicate_rows": duplicate_rows, "column_info": column_info,
        "numeric_summary": numeric_summary.to_dict(orient="records") if len(numeric_summary) else [],
        "insights": insights,
        "missing_by_column": [(str(k), int(v)) for k, v in missing.items() if int(v) > 0][:15],
    }


def chart_to_data_uri(fig) -> str:
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=125, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def make_chart(df: pd.DataFrame, chart_type: str, x_column: str | None = None, y_column: str | None = None) -> str:
    """Create a PNG data URI from the actual data; raises ValueError on bad requests."""
    if chart_type == "missing":
        vals = df.isna().sum().sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.bar(vals.index.astype(str), vals.values)
        ax.set_title("Missing values by column (top 20)")
        ax.set_ylabel("Missing cells")
        ax.tick_params(axis="x", rotation=65)
        return chart_to_data_uri(fig)
    if chart_type == "correlation":
        numeric = df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            raise ValueError("Correlation chart requires at least two numeric columns.")
        numeric = numeric.iloc[:, :15]
        corr = numeric.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 6))
        image = ax.imshow(corr.to_numpy(), aspect="auto", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns.astype(str), rotation=80, ha="right")
        ax.set_yticks(range(len(corr.index)), corr.index.astype(str))
        ax.set_title("Numeric correlation (first 15 numeric columns)")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        return chart_to_data_uri(fig)
    if x_column not in df.columns:
        raise ValueError("Choose a valid X column.")
    if chart_type == "histogram":
        values = pd.to_numeric(df[x_column], errors="coerce").dropna()
        if not len(values):
            raise ValueError("Histogram requires numeric values in the selected column.")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.hist(values, bins=min(30, max(5, int(np.sqrt(len(values))))) )
        ax.set_title(f"Distribution of {x_column}")
        ax.set_xlabel(str(x_column))
        ax.set_ylabel("Count")
        return chart_to_data_uri(fig)
    if chart_type == "bar":
        counts = df[x_column].fillna("(missing)").astype(str).value_counts().head(20)
        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(counts.index.astype(str), counts.values)
        ax.set_title(f"Top values in {x_column} (top 20)")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=65)
        return chart_to_data_uri(fig)
    if chart_type == "scatter":
        if y_column not in df.columns:
            raise ValueError("Choose a valid Y column.")
        xv = pd.to_numeric(df[x_column], errors="coerce")
        yv = pd.to_numeric(df[y_column], errors="coerce")
        valid = xv.notna() & yv.notna()
        if int(valid.sum()) < 2:
            raise ValueError("Scatter plot requires at least two rows with numeric X and Y values.")
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.scatter(xv[valid], yv[valid], alpha=0.65)
        ax.set_title(f"{y_column} vs. {x_column}")
        ax.set_xlabel(str(x_column))
        ax.set_ylabel(str(y_column))
        return chart_to_data_uri(fig)
    raise ValueError("Unsupported chart type.")
