"""Train, evaluate and (optionally) write back every model.

    python -m f1ml.train                         # all tasks, data from SQL Server
    python -m f1ml.train --source csv            # from data/*.csv (see f1ml.data export)
    python -m f1ml.train --tasks podium dnf      # a subset
    python -m f1ml.train --write-back            # also fill ml.model_runs / ml.predictions

For each task:
  1. pick hyper-parameters on the validation season (small grid)
  2. refit on train + validation, score the test seasons, compare with the baseline
  3. refit on every completed season and predict the current season's races
  4. save the model, metrics.json and a model card; write back if asked
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .config import settings
from .data import load_features
from .splits import SeasonSplit
from .tasks import ID_COLUMNS, TASKS, Task, add_derived

GRID = list(product([0.03, 0.08], [15, 31]))       # (learning_rate, max_leaf_nodes)


def assert_no_leakage(task: Task) -> None:
    bad = [f for f in task.features if f.startswith("target_") or f in ID_COLUMNS]
    if bad:
        raise ValueError(f"{task.name}: forbidden feature columns {bad}")


def _better(task: Task, a: float, b: float) -> bool:
    return a > b if task.higher_is_better else a < b


def tune(task: Task, train: pd.DataFrame, valid: pd.DataFrame) -> dict:
    best, best_params = None, None
    for lr, leaves in GRID:
        model = task.make_model(learning_rate=lr, max_leaf_nodes=leaves)
        model.fit(train[task.features], train[task.target])
        value, prob = task.predict(model, valid[task.features])
        m = task.score(valid, value, prob)[task.tune_metric]
        if best is None or _better(task, m, best):
            best, best_params = m, {"learning_rate": lr, "max_leaf_nodes": leaves}
    return best_params


def top_features(task: Task, model, test: pd.DataFrame, n: int = 10) -> list[dict]:
    scoring = {
        "binary": "average_precision",
        "multiclass": "f1_macro",
        "regression": "neg_mean_absolute_error",
        "ranking": "neg_mean_absolute_error",
    }[task.kind]
    sample = test.sample(min(len(test), 3000), random_state=0)
    imp = permutation_importance(model, sample[task.features], sample[task.target],
                                 scoring=scoring, n_repeats=5, random_state=0)
    order = np.argsort(imp.importances_mean)[::-1][:n]
    return [{"feature": task.features[i], "importance": round(float(imp.importances_mean[i]), 5)}
            for i in order]


def prediction_frame(task: Task, df: pd.DataFrame, value, prob) -> pd.DataFrame:
    out = pd.DataFrame({"race_id": df["race_id"].to_numpy(), "driver_id": df["driver_id"].to_numpy()})
    if task.kind == "multiclass":
        out["predicted_value"] = None
        out["predicted_class"] = value
    else:
        out["predicted_value"] = np.round(np.asarray(value, dtype=float), 4)
        out["predicted_class"] = None
    out["probability"] = None if prob is None else np.round(np.asarray(prob, dtype=float), 5)
    return out


def run_task(task: Task, df: pd.DataFrame, split: SeasonSplit) -> dict:
    assert_no_leakage(task)
    df = df[task.row_filter(df)]
    parts = split.parts(df)
    for name, part in parts.items():
        if part.empty:
            raise ValueError(f"{task.name}: no rows in {name} seasons ({split})")

    params = tune(task, parts["train"], parts["valid"])

    # evaluation model: train + valid -> test
    eval_model = task.make_model(**params).fit(parts["train_valid"][task.features],
                                               parts["train_valid"][task.target])
    test = parts["test"]
    value, prob = task.predict(eval_model, test[task.features])
    model_metrics = task.score(test, value, prob)
    b_value, b_score = task.baseline(test, parts["train_valid"])
    baseline_metrics = task.score(test, b_value, b_score)
    importance = top_features(task, eval_model, test)

    # production model: every completed season up to the last full one
    last_full = int(df["season_year"].max()) - 1
    prod_train = df[(df["season_year"] >= split.train_from) & (df["season_year"] <= last_full)]
    prod_model = task.make_model(**params).fit(prod_train[task.features], prod_train[task.target])
    current = df[df["season_year"] > last_full]
    cur_value, cur_prob = (task.predict(prod_model, current[task.features])
                           if len(current) else ([], None))

    settings.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": prod_model, "features": task.features, "params": params},
                settings.model_dir / f"{task.name}.joblib")

    beats = all(
        _better(task, model_metrics[k], baseline_metrics[k]) or model_metrics[k] == baseline_metrics[k]
        for k in [task.tune_metric]
    )
    return {
        "task": task.name,
        "description": task.description,
        "target": task.target,
        "kind": task.kind,
        "params": params,
        "seasons": {
            "train": [split.train_from, split.train_to], "valid": split.valid,
            "test": [split.test_from, split.test_to], "production_train_to": last_full,
        },
        "rows": {k: int(len(v)) for k, v in parts.items()} | {"current": int(len(current))},
        "test_metrics": {k: round(float(v), 4) for k, v in model_metrics.items()},
        "baseline_metrics": {k: round(float(v), 4) for k, v in baseline_metrics.items()},
        "beats_baseline_on": task.tune_metric if beats else None,
        "top_features": importance,
        "_test_predictions": prediction_frame(task, test, value, prob),
        "_current_predictions": prediction_frame(task, current, cur_value, cur_prob) if len(current) else pd.DataFrame(),
    }


def model_card(r: dict) -> str:
    rows = "\n".join(
        f"| {k} | {r['test_metrics'][k]} | {r['baseline_metrics'].get(k, '')} |"
        for k in r["test_metrics"]
    )
    feats = "\n".join(f"{i+1}. `{f['feature']}` ({f['importance']})" for i, f in enumerate(r["top_features"]))
    s = r["seasons"]
    verdict = (f"Beats the baseline on **{r['beats_baseline_on']}**."
               if r["beats_baseline_on"] else "**Does not beat the baseline** on the tuning metric.")
    return f"""# Model card: {r['task']}

{r['description']}

- Target: `{r['target']}` ({r['kind']})
- Model: scikit-learn HistGradientBoosting, params {r['params']}
- Train {s['train'][0]}-{s['train'][1]}, validate {s['valid']}, test {s['test'][0]}-{s['test'][1]}
- Production model retrained on {s['train'][0]}-{s['production_train_to']}
- Rows: {r['rows']}

## Test-season results

{verdict}

| Metric | Model | Baseline |
| --- | --- | --- |
{rows}

## Most important features (permutation importance on the test seasons)

{feats}
"""


def main() -> None:
    p = argparse.ArgumentParser(description="Train and evaluate the F1 models")
    p.add_argument("--source", choices=["sql", "csv"], default="sql")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--train-from", type=int, default=2010)
    p.add_argument("--train-to", type=int, default=2021)
    p.add_argument("--valid", type=int, default=2022)
    p.add_argument("--test-from", type=int, default=2023)
    p.add_argument("--test-to", type=int, default=2024)
    p.add_argument("--write-back", action="store_true", help="write runs + predictions to ml.* tables")
    args = p.parse_args()

    split = SeasonSplit(args.train_from, args.train_to, args.valid, args.test_from, args.test_to)
    views = {TASKS[t].view for t in args.tasks}
    data = {v: add_derived(load_features(v, args.source)) for v in views}

    results = []
    for name in args.tasks:
        task = TASKS[name]
        print(f"\n=== {name} ===")
        r = run_task(task, data[task.view], split)
        results.append(r)
        print(f"  params   {r['params']}")
        for k, v in r["test_metrics"].items():
            print(f"  {k:<18} model {v:>9}   baseline {r['baseline_metrics'].get(k, '')}")
        print(f"  beats baseline on {task.tune_metric}: {'yes' if r['beats_baseline_on'] else 'NO'}")

    settings.report_dir.mkdir(parents=True, exist_ok=True)
    (settings.report_dir / "model_cards").mkdir(exist_ok=True)
    public = [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "source": args.source, "results": public}
    (settings.report_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for r in results:
        (settings.report_dir / "model_cards" / f"{r['task']}.md").write_text(model_card(r), encoding="utf-8")
    print(f"\nReports written to {settings.report_dir}")

    if args.write_back:
        from .db import get_engine
        from .writeback import write_predictions, write_run

        with get_engine().begin() as conn:
            for r in results:
                feats = TASKS[r["task"]].features
                s = r["seasons"]
                eval_id = write_run(conn, model_name=r["task"], model_version="eval",
                                    train_from=s["train"][0], train_to=s["valid"],
                                    test_from=s["test"][0], test_to=s["test"][1],
                                    metrics={"model": r["test_metrics"], "baseline": r["baseline_metrics"],
                                             "params": r["params"], "top_features": r["top_features"]},
                                    features=feats)
                n1 = write_predictions(conn, eval_id, r["_test_predictions"])
                n2 = 0
                if not r["_current_predictions"].empty:
                    prod_id = write_run(conn, model_name=r["task"], model_version="prod",
                                        train_from=s["train"][0], train_to=s["production_train_to"],
                                        test_from=None, test_to=None,
                                        metrics={"params": r["params"]}, features=feats)
                    n2 = write_predictions(conn, prod_id, r["_current_predictions"])
                print(f"  {r['task']}: wrote {n1} test + {n2} current-season predictions")


if __name__ == "__main__":
    main()
