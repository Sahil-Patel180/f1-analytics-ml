# F1 Analytics ML

Machine-learning models trained on the **[F1 Data Warehouse](https://github.com/Sahil-Patel180/SQL_Projects/tree/main/F1%20Data%20Warehouse)** (SQL Server, bronze → silver → gold). The models read the warehouse's leakage-safe feature views, are tested on later seasons than they learned from, are compared against simple baselines, and write their predictions back into the warehouse for Power BI.

```mermaid
flowchart LR
    G["F1_DB2<br/>gold.ml_driver_race_features<br/>gold.ml_pit_features"] -->|read| T["f1ml.train<br/>tune · test · baseline"]
    T --> R["reports/<br/>metrics.json · model cards"]
    T --> M["models/*.joblib"]
    T -->|--write-back| W["F1_DB2<br/>ml.model_runs · ml.predictions"]
    W --> PBI["Power BI<br/>predictions vs actual"]
    G -->|facts| L["f1ml.llm.narrate<br/>Ollama · qwen2.5-coder:3b"]
    W -->|facts| L
    L -->|verified text| N["F1_DB2<br/>ml.narratives"]
    N --> PBI
```

## Models

| Task | Predicts | Type | Baseline it must beat |
| --- | --- | --- | --- |
| `podium` | Probability a starter finishes top 3 | binary classifier | "Top 3 on the grid finish on the podium" |
| `finish_position` | Finishing order of the field | regressor, ranked per race | "Finish where you started" |
| `dnf` | Probability a starter does not finish | binary classifier | The team's DNF rate over its last 10 races |
| `positions_gained` | Grid minus finish, classified starters | regressor | "Nobody gains or loses places" |
| `pit_stop_class` | 0-1, 2 or 3+ pit stops (2010+) | multiclass classifier | Last year's average at the same circuit |
| `pit_stop_ms` | Average pit-lane time per stop (2010+) | regressor | The team's average over its last 5 races |

All models are scikit-learn `HistGradientBoosting`, which handles missing values natively (e.g. no qualifying time, a driver's first race at a circuit).

## Results (test seasons 2023–2024, from `F1_DB2`)

| Model | Key metric | Model | Baseline | Beats baseline |
| --- | --- | --- | --- | --- |
| `podium` | pr_auc | 0.706 | 0.664 | yes |
| `finish_position` | spearman_per_race | 0.700 | 0.666 | yes |
| `dnf` | pr_auc | 0.119 | 0.097 | yes |
| `positions_gained` | mae | 2.250 | 2.811 | yes |
| `pit_stop_class` | macro_f1 | 0.431 | 0.455 | **no** |
| `pit_stop_ms` | mae | 2,469 | 3,549 | yes |

`pit_stop_class` loses to its baseline on the test seasons (it wins 6 of 9 seasons in the 2016–2024 backtest), so the simple rule remains the better predictor until race-condition features exist. Full numbers: [`reports/metrics.json`](reports/metrics.json), [`reports/backtest.csv`](reports/backtest.csv) and one card per model in [`reports/model_cards/`](reports/model_cards).

## How it avoids fooling itself

- **No leakage.** Features come only from the `gold.ml_*` views, where every rolling value ends at the driver's *previous* race. The warehouse's gold quality checks recompute one feature from scratch to prove it, and `tests/test_leakage.py` blocks ids and `target_*` columns from ever being features.
- **Split by season, never at random.** Train 2010–2021, choose hyper-parameters on 2022, report on 2023–2024 once. A random split would put drivers of the same race on both sides.
- **Always against a baseline.** Every metric is shown next to a simple rule a fan would use. A model that does not beat its baseline is reported as such, not hidden.
- **Walk-forward backtest.** `f1ml.backtest` retrains for every season from 2016 and scores it, which shows whether accuracy survives regulation changes (2017, 2022).

## Run it (Windows, PowerShell)

Prerequisites: the F1 Data Warehouse loaded in SQL Server (database `F1_DB2`), Python 3.10+, and **ODBC Driver 17 or 18 for SQL Server**.

```powershell
git clone https://github.com/Sahil-Patel180/f1-analytics-ml.git
cd f1-analytics-ml
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"

copy .env.example .env      # set SQL_SERVER, e.g. SAHIL\SQLEXPRESS
pytest -q                   # unit tests, no database needed
```

Then:

```powershell
python -m f1ml.data export                     # optional: cache the feature views to data/*.csv
python -m f1ml.train                           # train + evaluate all 6 models (reads SQL Server)
python -m f1ml.train --source csv              # same, from the CSV cache
python -m f1ml.backtest --first 2016 --last 2024
python -m f1ml.train --write-back              # also fill ml.model_runs and ml.predictions
```

Outputs:

| Path | Contents |
| --- | --- |
| `reports/metrics.json` | Test-season metrics, baseline metrics, chosen parameters, top features, per task |
| `reports/model_cards/<task>.md` | One readable card per model |
| `reports/backtest.csv` | Season-by-season model vs baseline |
| `models/<task>.joblib` | Production model trained on every completed season |

Check the write-back in SSMS:

```sql
USE F1_DB2;
SELECT model_name, model_version, train_from_season, train_to_season,
       test_from_season, test_to_season, created_at
FROM ml.model_runs ORDER BY run_id DESC;

-- 2024 podium probabilities vs what happened
SELECT r.race_label, d.full_name, p.probability, f.finish_position
FROM ml.predictions p
JOIN ml.model_runs m        ON m.run_id = p.run_id AND m.model_name = 'podium' AND m.model_version = 'eval'
JOIN gold.dim_race r        ON r.race_id = p.race_id
JOIN gold.dim_driver d      ON d.driver_id = p.driver_id
JOIN gold.fact_results f    ON f.race_id = p.race_id AND f.driver_id = p.driver_id AND f.session_type = 'Race'
WHERE r.season_year = 2024
ORDER BY r.round, p.probability DESC;
```

Each `--write-back` adds two runs per model: `eval` (trained to 2022, predictions for the test seasons) and `prod` (trained on every completed season, predictions for the current season's completed races).

## Local LLM narratives

A small local model (`qwen2.5-coder:3b` via [Ollama](https://ollama.com)) writes short race recaps, driver-season summaries, prediction explanations and model reports. It is never trusted with facts:

1. **SQL computes every fact.** A JSON *fact pack* is built from the gold views (podium, pole, fastest lap, biggest gain, retirements, ...), already rounded and labelled.
2. **The model only writes prose** from that JSON, under a strict system prompt.
3. **A verifier checks the text.** Every number (including `P3`, `3rd`, `ninth`) must appear in the fact pack, and every known driver surname or team name must be one the facts mention.
4. **One retry at temperature 0**, then a plain template sentence built straight from the facts. Nothing unverified is stored.
5. **Unchanged facts are skipped.** A SHA-256 hash of facts + prompt version + model is stored with each narrative, so re-runs only process new races.

Setup:

```powershell
# install Ollama for Windows from https://ollama.com, then:
ollama pull qwen2.5-coder:3b
```

Run:

```powershell
python -m f1ml.llm.narrate race_recap --season 2024 --dry-run   # print only
python -m f1ml.llm.narrate race_recap --season 2024             # write to ml.narratives
python -m f1ml.llm.narrate driver_season --season 2024 --top 10
python -m f1ml.llm.narrate prediction --season 2024 --top 20    # needs train --write-back first
python -m f1ml.llm.narrate model_report
```

`qwen2.5-coder:3b` is tuned for code, so its prose can read stiffly; `qwen2.5:3b` is the general model at the same size. Compare them with `--model qwen2.5:3b --dry-run` and keep the one whose text you prefer: the verifier keeps both honest.

Read them in SSMS:

```sql
SELECT n.narrative_type, r.race_label, n.llm_model, n.is_verified, n.narrative_text
FROM F1_DB2.ml.narratives n
LEFT JOIN F1_DB2.gold.dim_race r ON r.race_id = n.race_id
ORDER BY n.narrative_id DESC;
```

## Project layout

```
f1-analytics-ml/
├── src/f1ml/
│   ├── config.py      settings from .env
│   ├── db.py          SQL Server connection (SQLAlchemy + pyodbc)
│   ├── data.py        load / export the gold.ml_* views
│   ├── tasks.py       the 6 tasks: target, features, model, baseline, scoring
│   ├── splits.py      season splits and walk-forward
│   ├── metrics.py     AUC, PR-AUC, Brier, top-3 hit rate, per-race Spearman, MAE, F1
│   ├── train.py       tune → test → production fit → reports → write-back
│   ├── backtest.py    walk-forward evaluation
│   ├── writeback.py   ml.model_runs / ml.predictions inserts
│   └── llm/
│       ├── client.py  Ollama chat client
│       ├── facts.py   fact packs built from gold rows
│       ├── prompts.py system prompts + template fallback
│       ├── verify.py  number / name verifier
│       └── narrate.py CLI: facts -> LLM -> verify -> ml.narratives
├── tests/             leakage, split, metric and LLM-verifier tests (run in CI)
└── reports/           generated metrics and model cards
```

## Limits

- Predictions exist only for races already in the warehouse. A race with no result row yet has no grid or qualifying in the feature view, so upcoming races are not scored.
- Pit-stop strategy depends heavily on safety cars and weather, which are not features yet (planned: FastF1 / Meteostat enrichment for 2018+).
- Global feature importance is permutation-based; per-prediction explanations (SHAP) are not included.

## Related

- Warehouse: [SQL_Projects / F1 Data Warehouse](https://github.com/Sahil-Patel180/SQL_Projects/tree/main/F1%20Data%20Warehouse)
- Next: Power BI report on the gold views, `ml.predictions` and `ml.narratives` (Power_BI_Projects).
