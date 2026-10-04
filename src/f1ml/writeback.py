"""Write model runs and predictions back to the warehouse (ml.model_runs, ml.predictions)."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def _clean(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    return v


def _json(obj) -> str:
    """JSON without NaN/Infinity (SQL Server ISJSON rejects them)."""
    def fix(v):
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            return [fix(x) for x in v]
        if isinstance(v, (float, np.floating)):
            return None if not np.isfinite(v) else float(v)
        if isinstance(v, np.integer):
            return int(v)
        return v
    return json.dumps(fix(obj), allow_nan=False)


def write_run(conn, *, model_name: str, model_version: str, train_from: int, train_to: int,
              test_from: int | None, test_to: int | None, metrics: dict, features: list[str]) -> int:
    from sqlalchemy import text

    run_id = conn.execute(
        text("""
            INSERT INTO ml.model_runs
                (model_name, model_version, train_from_season, train_to_season,
                 test_from_season, test_to_season, metrics_json, feature_list_json)
            OUTPUT INSERTED.run_id
            VALUES (:name, :version, :tf, :tt, :sf, :st, :metrics, :features)
        """),
        {
            "name": model_name, "version": model_version,
            "tf": train_from, "tt": train_to, "sf": test_from, "st": test_to,
            "metrics": _json(metrics),
            "features": _json(features),
        },
    ).scalar_one()
    return int(run_id)


def write_predictions(conn, run_id: int, preds: pd.DataFrame) -> int:
    """preds columns: race_id, driver_id, predicted_value, predicted_class, probability."""
    from sqlalchemy import text

    if preds.empty:
        return 0
    rows = [
        {
            "run_id": run_id,
            "race_id": int(r.race_id),
            "driver_id": int(r.driver_id),
            "value": _clean(r.predicted_value),
            "cls": _clean(r.predicted_class),
            "prob": _clean(r.probability),
        }
        for r in preds.itertuples(index=False)
    ]
    conn.execute(
        text("""
            INSERT INTO ml.predictions (run_id, race_id, driver_id, predicted_value, predicted_class, probability)
            VALUES (:run_id, :race_id, :driver_id, :value, :cls, :prob)
        """),
        rows,
    )
    return len(rows)
