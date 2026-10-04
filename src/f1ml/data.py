"""Load the gold ML feature views, from SQL Server or from a local CSV export.

    python -m f1ml.data export          # SQL -> data/*.csv (run once, then train offline)
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import settings

VIEWS = {
    "driver_race": "gold.ml_driver_race_features",
    "pit": "gold.ml_pit_features",
}


def _csv_path(view_key: str):
    return settings.data_dir / f"ml_{view_key}_features.csv"


def load_features(view_key: str, source: str = "sql") -> pd.DataFrame:
    """Return one feature view as a DataFrame, sorted by date."""
    if view_key not in VIEWS:
        raise ValueError(f"unknown view {view_key!r}; choose from {list(VIEWS)}")

    if source == "sql":
        from .db import read_sql

        df = read_sql(f"SELECT * FROM {VIEWS[view_key]}")
    elif source == "csv":
        path = _csv_path(view_key)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found. Run `python -m f1ml.data export` first, "
                "or use --source sql."
            )
        df = pd.read_csv(path)
    else:
        raise ValueError("source must be 'sql' or 'csv'")

    df["race_date"] = pd.to_datetime(df["race_date"])
    # pyodbc returns DECIMAL columns as Python Decimal objects; make them floats
    text_cols = {"era", "race_date", "target_stop_class"}
    for col in df.columns.difference(list(text_cols)):
        if df[col].dtype == object:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.sort_values(["race_date", "race_id"]).reset_index(drop=True)


def export_features() -> None:
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    for key in VIEWS:
        df = load_features(key, source="sql")
        path = _csv_path(key)
        df.to_csv(path, index=False)
        print(f"{VIEWS[key]} -> {path}  ({len(df):,} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature view utilities")
    parser.add_argument("command", choices=["export"])
    args = parser.parse_args()
    if args.command == "export":
        export_features()


if __name__ == "__main__":
    main()
