"""Walk-forward backtest: for each season, train on every earlier season (2010+)
and score that season. Shows whether accuracy holds across regulation changes.

    python -m f1ml.backtest --source csv --first 2016 --last 2024
Writes reports/backtest.csv (task, season, metric, model, baseline).
"""
from __future__ import annotations

import argparse

import pandas as pd

from .config import settings
from .data import load_features
from .splits import walk_forward_seasons
from .tasks import TASKS, add_derived


def backtest(task_names, source: str, first: int, last: int) -> pd.DataFrame:
    rows = []
    cache = {}
    for name in task_names:
        task = TASKS[name]
        if task.view not in cache:
            cache[task.view] = add_derived(load_features(task.view, source))
        df = cache[task.view]
        df = df[task.row_filter(df)]
        for season, train, test in walk_forward_seasons(df, first, last):
            model = task.make_model().fit(train[task.features], train[task.target])
            value, prob = task.predict(model, test[task.features])
            m = task.score(test, value, prob)
            bv, bs = task.baseline(test, train)
            b = task.score(test, bv, bs)
            for metric in m:
                rows.append({"task": name, "season": season, "metric": metric,
                             "model": round(float(m[metric]), 4), "baseline": round(float(b[metric]), 4)})
            print(f"{name:<17} {season}  {task.tune_metric}: model {m[task.tune_metric]:.4f}  "
                  f"baseline {b[task.tune_metric]:.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Walk-forward backtest")
    p.add_argument("--source", choices=["sql", "csv"], default="sql")
    p.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    p.add_argument("--first", type=int, default=2016)
    p.add_argument("--last", type=int, default=2024)
    args = p.parse_args()

    out = backtest(args.tasks, args.source, args.first, args.last)
    settings.report_dir.mkdir(parents=True, exist_ok=True)
    path = settings.report_dir / "backtest.csv"
    out.to_csv(path, index=False)
    print(f"\n{path}  ({len(out)} rows)")


if __name__ == "__main__":
    main()
