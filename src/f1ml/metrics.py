"""Evaluation metrics per task type. Every function returns a flat dict of floats."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)


def binary_metrics(y_true, score, proba) -> dict:
    """score ranks rows (for AUC); proba is a calibrated probability (for Brier / log loss)."""
    y_true = np.asarray(y_true)
    proba = np.clip(np.asarray(proba, dtype=float), 1e-6, 1 - 1e-6)
    return {
        "roc_auc": roc_auc_score(y_true, score),
        "pr_auc": average_precision_score(y_true, score),
        "brier": brier_score_loss(y_true, proba),
        "log_loss": log_loss(y_true, proba, labels=[0, 1]),
        "positive_rate": float(y_true.mean()),
    }


def top_k_hit_rate(race_ids, y_true, score, k: int = 3) -> float:
    """Per race: share of the k highest-scored drivers who really were in the top k."""
    df = pd.DataFrame({"race": race_ids, "y": y_true, "s": score})
    hits = []
    for _, g in df.groupby("race"):
        picked = g.nlargest(k, "s")
        hits.append(picked["y"].sum() / min(k, len(g)))
    return float(np.mean(hits))


def ranking_metrics(race_ids, y_true, pred) -> dict:
    """Finish order: Spearman correlation of predicted vs actual order, averaged over races."""
    df = pd.DataFrame({"race": race_ids, "y": y_true, "p": pred})
    rhos = []
    for _, g in df.groupby("race"):
        if len(g) > 2 and g["p"].nunique() > 1:
            rhos.append(spearmanr(g["p"], g["y"]).correlation)
    pred_rank = df.groupby("race")["p"].rank(method="first")
    return {
        "spearman_per_race": float(np.nanmean(rhos)),
        "mae_places": mean_absolute_error(df["y"], pred_rank),
        "winner_hit_rate": float(
            df.assign(r=pred_rank).query("r == 1")["y"].eq(1).mean()
        ),
    }


def regression_metrics(y_true, pred, sign: bool = False) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    pred = np.asarray(pred, dtype=float)
    out = {
        "mae": mean_absolute_error(y_true, pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, pred))),
    }
    if sign:
        moved = y_true != 0
        out["sign_accuracy"] = float((np.sign(pred[moved]) == np.sign(y_true[moved])).mean())
    return out


def multiclass_metrics(y_true, pred) -> dict:
    return {
        "macro_f1": f1_score(y_true, pred, average="macro"),
        "accuracy": accuracy_score(y_true, pred),
    }
