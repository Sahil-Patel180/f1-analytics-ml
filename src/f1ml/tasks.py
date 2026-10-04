"""The six prediction tasks: target, features, model, baseline and how each is scored.

Leakage rule: features only come from the lists below, which hold pre-race
information as built by the gold.ml_* views. ids and target_* columns can never
be features; tests/test_leakage.py enforces it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from . import metrics as M

ID_COLUMNS = {
    "result_id", "race_id", "driver_id", "constructor_id", "circuit_id",
    "season_year", "race_date",
}

DRIVER_FEATURES = [
    "round", "has_sprint", "era_code", "driver_age_years",
    # known before the start
    "grid_position", "is_pit_lane_start", "quali_position",
    "quali_gap_to_fastest_ms", "reached_q3",
    # driver form (previous races only)
    "driver_career_starts_before", "driver_career_wins_before",
    "driver_avg_finish_last3", "driver_avg_finish_last5",
    "driver_points_last5", "driver_podiums_last5", "driver_dnf_rate_last10",
    "driver_avg_grid_last5", "driver_circuit_starts_before",
    "driver_circuit_avg_finish_before",
    # championship after the previous round
    "champ_position_before", "champ_points_before",
    # constructor form (previous races only)
    "constructor_points_last5", "constructor_best_finish_avg_last5",
    "constructor_dnf_rate_last10",
]

PIT_FEATURES = [
    "round", "has_sprint", "grid_position", "race_laps",
    "circuit_avg_stops_prev_edition", "constructor_avg_stop_ms_last5",
]


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    """Encodings that need no target information."""
    df = df.copy()
    if "era" in df.columns:
        # era labels start with their first year, so sorting them is chronological
        order = {e: i for i, e in enumerate(sorted(df["era"].dropna().unique()))}
        df["era_code"] = df["era"].map(order)
    return df


# ---------------------------------------------------------------- baselines
# Each takes (test_df, train_df) and returns (prediction, score_or_proba).

def _baseline_podium(test, train):
    grid = test["grid_position"].fillna(30)
    proba = (grid <= 3).astype(float)          # "the top 3 on the grid finish on the podium"
    return proba, -grid                         # -grid ranks drivers for AUC


def _baseline_finish(test, train):
    pred = test["grid_position"].fillna(30)    # finish where you started
    return pred, pred


def _baseline_dnf(test, train):
    rate = test["constructor_dnf_rate_last10"].fillna(train["target_dnf"].mean())
    return rate, rate                           # the team's recent retirement rate


def _baseline_delta(test, train):
    zero = pd.Series(0.0, index=test.index)    # "nobody gains or loses places"
    return zero, zero


def _baseline_pit_class(test, train):
    mode = train["target_stop_class"].mode().iloc[0]
    prev = test["circuit_avg_stops_prev_edition"].round()
    pred = np.select([prev <= 1, prev == 2, prev >= 3], ["0-1", "2", "3+"], default=mode)
    return pd.Series(pred, index=test.index), None   # last year's typical strategy here


def _baseline_pit_ms(test, train):
    pred = test["constructor_avg_stop_ms_last5"].fillna(train["target_avg_stop_ms"].median())
    return pred, pred                           # the team's recent average stop


# ---------------------------------------------------------------- task spec
@dataclass
class Task:
    name: str
    view: str                       # driver_race | pit
    target: str
    kind: str                       # binary | regression | multiclass | ranking
    features: list[str]
    baseline: Callable
    tune_metric: str                # metric used to pick hyper-parameters on validation
    higher_is_better: bool
    description: str
    row_filter: Callable[[pd.DataFrame], pd.Series] = field(default=lambda d: d.index == d.index)

    # ---- model
    def make_model(self, learning_rate: float = 0.05, max_leaf_nodes: int = 15):
        common = dict(
            learning_rate=learning_rate,
            max_leaf_nodes=max_leaf_nodes,
            max_iter=400,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=25,
            random_state=42,
        )
        if self.kind in ("binary", "multiclass"):
            return HistGradientBoostingClassifier(**common)
        return HistGradientBoostingRegressor(loss="absolute_error" if self.kind == "regression" else "squared_error", **common)

    # ---- prediction in a common shape
    def predict(self, model, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Returns (value, probability). value = class / number; probability for classifiers."""
        if self.kind == "binary":
            p = model.predict_proba(X)[:, 1]
            return p, p
        if self.kind == "multiclass":
            proba = model.predict_proba(X)
            return model.classes_[proba.argmax(axis=1)], proba.max(axis=1)
        return model.predict(X), None

    # ---- scoring
    def score(self, df: pd.DataFrame, value, prob_or_score) -> dict:
        y = df[self.target].to_numpy()
        if self.kind == "binary":
            out = M.binary_metrics(y, prob_or_score, value)
            if self.name == "podium":
                out["top3_hit_rate"] = M.top_k_hit_rate(df["race_id"], y, prob_or_score, k=3)
            return out
        if self.kind == "ranking":
            return M.ranking_metrics(df["race_id"], y, value)
        if self.kind == "multiclass":
            return M.multiclass_metrics(y, value)
        return M.regression_metrics(y, value, sign=(self.name == "positions_gained"))


TASKS: dict[str, Task] = {
    t.name: t
    for t in [
        Task(
            name="podium", view="driver_race", target="target_podium", kind="binary",
            features=DRIVER_FEATURES, baseline=_baseline_podium,
            tune_metric="pr_auc", higher_is_better=True,
            description="Probability that a starter finishes in the top 3.",
        ),
        Task(
            name="finish_position", view="driver_race", target="target_finish_position",
            kind="ranking", features=DRIVER_FEATURES, baseline=_baseline_finish,
            tune_metric="spearman_per_race", higher_is_better=True,
            description="Finishing order of the field (position_order, retirements at the back).",
        ),
        Task(
            name="dnf", view="driver_race", target="target_dnf", kind="binary",
            features=DRIVER_FEATURES, baseline=_baseline_dnf,
            tune_metric="pr_auc", higher_is_better=True,
            description="Probability that a starter does not finish (not classified, not disqualified).",
        ),
        Task(
            name="positions_gained", view="driver_race", target="target_positions_gained",
            kind="regression", features=DRIVER_FEATURES, baseline=_baseline_delta,
            tune_metric="mae", higher_is_better=False,
            description="Grid position minus finishing position, for classified starters with a grid slot.",
            row_filter=lambda d: d["target_positions_gained"].notna(),
        ),
        Task(
            name="pit_stop_class", view="pit", target="target_stop_class", kind="multiclass",
            features=PIT_FEATURES, baseline=_baseline_pit_class,
            tune_metric="macro_f1", higher_is_better=True,
            description="Number of pit stops (0-1, 2, 3+) for classified finishers, 2010 onward.",
            row_filter=lambda d: d["target_stop_class"].notna(),
        ),
        Task(
            name="pit_stop_ms", view="pit", target="target_avg_stop_ms", kind="regression",
            features=PIT_FEATURES, baseline=_baseline_pit_ms,
            tune_metric="mae", higher_is_better=False,
            description="Average pit-lane time per stop in ms (stops over 120 s excluded), 2010 onward.",
            row_filter=lambda d: d["target_avg_stop_ms"].notna(),
        ),
    ]
}
