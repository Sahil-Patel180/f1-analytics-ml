"""Generate narratives with a local LLM and store them in ml.narratives.

    python -m f1ml.llm.narrate race_recap --season 2024
    python -m f1ml.llm.narrate race_recap --race-id 1121 --dry-run
    python -m f1ml.llm.narrate driver_season --season 2024 --top 10
    python -m f1ml.llm.narrate prediction --season 2024 --top 20
    python -m f1ml.llm.narrate model_report

Per narrative: build facts (SQL) -> ask the model -> verify every number and
name -> one retry at temperature 0 -> otherwise a plain template sentence.
Unchanged facts (same hash) are skipped, so re-runs only touch new races.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import pandas as pd

from ..db import get_engine, read_sql
from . import facts as F
from .client import OllamaClient
from .prompts import PROMPT_VERSION, SYSTEM_PROMPTS, template_fallback
from .verify import fact_hash, verify


@dataclass
class Subject:
    kind: str
    facts: dict
    race_id: int | None = None
    driver_id: int | None = None
    season_year: int | None = None
    run_id: int | None = None
    label: str = ""


# ---------------------------------------------------------------- generation core
def generate(client, subject: Subject, roster: set[str]) -> tuple[str, str, bool, str]:
    """Returns (text, model_used, verified, note). Pure apart from the client call."""
    system = SYSTEM_PROMPTS[subject.kind]
    user = json.dumps(subject.facts, ensure_ascii=False, indent=1)
    note = ""
    for temperature in (0.2, 0.0):
        text = client.chat(system, user, temperature=temperature)
        v = verify(text, subject.facts, roster)
        if v.ok:
            return text, client.model, True, note
        note = v.reason()
    return template_fallback(subject.kind, subject.facts), "template-fallback", True, f"LLM rejected: {note}"


# ---------------------------------------------------------------- subjects from SQL
def roster_names() -> set[str]:
    d = read_sql("SELECT DISTINCT surname FROM gold.dim_driver")
    c = read_sql("SELECT DISTINCT constructor_name FROM gold.dim_constructor")
    names = {s for s in d["surname"].dropna() if len(s) >= 4} | set(c["constructor_name"].dropna())
    return names


def race_subjects(season: int | None, race_id: int | None, limit: int | None) -> list[Subject]:
    where = "race_id = :rid" if race_id else "season_year = :season"
    params = {"rid": race_id} if race_id else {"season": season}
    rows = read_sql(f"SELECT * FROM gold.vw_f1_master WHERE {where}", params)
    out = []
    for rid, g in rows.groupby("race_id", sort=False):
        if not (g["session_type"] == "Race").any():
            continue
        out.append(Subject("race_recap", F.race_recap_facts(g), race_id=int(rid),
                           season_year=int(g["season_year"].iloc[0]),
                           label=f"{g['season_year'].iloc[0]} {g['race_name'].iloc[0]}"))
    out.sort(key=lambda s: s.facts["round"])
    return out[:limit] if limit else out


def driver_season_subjects(season: int, top: int | None) -> list[Subject]:
    agg = read_sql("""
        SELECT a.*, d.full_name, d.nationality
        FROM gold.agg_driver_season a
        JOIN gold.dim_driver d ON d.driver_id = a.driver_id
        WHERE a.season_year = :season AND a.race_starts > 0
        ORDER BY COALESCE(a.final_position, 999)""", {"season": season})
    rows = read_sql("SELECT * FROM gold.vw_f1_master WHERE season_year = :season AND session_type = 'Race'",
                    {"season": season})
    if top:
        agg = agg.head(top)
    out = []
    for _, a in agg.iterrows():
        mine = rows[rows["driver_id"] == a["driver_id"]]
        team = rows[rows["constructor_id"].isin(mine["constructor_id"].unique())]
        out.append(Subject("driver_season", F.driver_season_facts(a, mine, team),
                           driver_id=int(a["driver_id"]), season_year=season,
                           label=f"{season} {a['full_name']}"))
    return out


def prediction_subjects(season: int, top: int | None) -> list[Subject]:
    preds = read_sql("""
        WITH latest AS (
            SELECT model_version, MAX(run_id) AS run_id
            FROM ml.model_runs WHERE model_name = 'podium'
            GROUP BY model_version
        )
        SELECT p.run_id, p.race_id, p.driver_id, p.probability,
               r.race_label, r.season_year,
               v.driver_name, v.constructor_name, v.grid_position, v.quali_position,
               v.finish_position, v.position_text, m.metrics_json
        FROM ml.predictions p
        JOIN latest l           ON l.run_id = p.run_id
        JOIN ml.model_runs m    ON m.run_id = p.run_id
        JOIN gold.dim_race r    ON r.race_id = p.race_id
        JOIN gold.vw_f1_master v ON v.race_id = p.race_id AND v.driver_id = p.driver_id
                                AND v.session_type = 'Race'
        WHERE r.season_year = :season""", {"season": season})
    if preds.empty:
        raise SystemExit(f"No podium predictions for {season}. Run `python -m f1ml.train --write-back` first.")
    # the most interesting cases: highest predicted chances
    preds = preds.sort_values("probability", ascending=False)
    if top:
        preds = preds.head(top)
    eval_metrics = read_sql("""
        SELECT TOP (1) metrics_json FROM ml.model_runs
        WHERE model_name = 'podium' AND model_version = 'eval' ORDER BY run_id DESC""")
    top_feats = json.loads(eval_metrics["metrics_json"].iloc[0]).get("top_features", []) if len(eval_metrics) else []
    return [Subject("prediction", F.prediction_facts(r, top_feats), race_id=int(r["race_id"]),
                    driver_id=int(r["driver_id"]), season_year=season, run_id=int(r["run_id"]),
                    label=f"{r['race_label']} {r['driver_name']}")
            for _, r in preds.iterrows()]


def model_report_subjects() -> list[Subject]:
    runs = read_sql("""
        SELECT r.* FROM ml.model_runs r
        JOIN (SELECT model_name, MAX(run_id) AS run_id FROM ml.model_runs
              WHERE model_version = 'eval' GROUP BY model_name) l ON l.run_id = r.run_id""")
    if runs.empty:
        raise SystemExit("No model runs found. Run `python -m f1ml.train --write-back` first.")
    return [Subject("model_report", F.model_report_facts(r), run_id=int(r["run_id"]), label=r["model_name"])
            for _, r in runs.iterrows()]


# ---------------------------------------------------------------- storage
def existing_hashes(kind: str) -> set[str]:
    df = read_sql("SELECT fact_pack_hash FROM ml.narratives WHERE narrative_type = :k", {"k": kind})
    return set(df["fact_pack_hash"])


def save(conn, s: Subject, text: str, model: str, verified: bool, h: str) -> None:
    from sqlalchemy import text as sql

    conn.execute(sql("""
        INSERT INTO ml.narratives
            (narrative_type, race_id, driver_id, season_year, run_id, narrative_text,
             llm_model, prompt_version, fact_pack_hash, is_verified)
        VALUES (:t, :race, :drv, :season, :run, :txt, :model, :pv, :h, :ok)"""),
        {"t": s.kind, "race": s.race_id, "drv": s.driver_id, "season": s.season_year,
         "run": s.run_id, "txt": text, "model": model, "pv": PROMPT_VERSION, "h": h, "ok": int(verified)})


# ---------------------------------------------------------------- CLI
def main() -> None:
    p = argparse.ArgumentParser(description="Generate verified LLM narratives")
    p.add_argument("kind", choices=list(SYSTEM_PROMPTS))
    p.add_argument("--season", type=int)
    p.add_argument("--race-id", type=int)
    p.add_argument("--top", type=int, help="limit the number of subjects")
    p.add_argument("--model", help="Ollama model (default from .env)")
    p.add_argument("--dry-run", action="store_true", help="print, do not write to ml.narratives")
    p.add_argument("--force", action="store_true", help="regenerate even if facts are unchanged")
    args = p.parse_args()

    if args.kind in ("race_recap", "driver_season", "prediction") and not (args.season or args.race_id):
        p.error("--season is required for this narrative type")

    client = OllamaClient(model=args.model)
    client.check()

    if args.kind == "race_recap":
        subjects = race_subjects(args.season, args.race_id, args.top)
    elif args.kind == "driver_season":
        subjects = driver_season_subjects(args.season, args.top)
    elif args.kind == "prediction":
        subjects = prediction_subjects(args.season, args.top)
    else:
        subjects = model_report_subjects()

    roster = roster_names()
    done = set() if (args.force or args.dry_run) else existing_hashes(args.kind)
    engine = None if args.dry_run else get_engine()
    stats = {"written": 0, "skipped": 0, "fallback": 0}

    for s in subjects:
        h = fact_hash(s.facts, PROMPT_VERSION, client.model)
        if h in done:
            stats["skipped"] += 1
            continue
        text, model_used, verified, note = generate(client, s, roster)
        if model_used == "template-fallback":
            stats["fallback"] += 1
        print(f"\n--- {s.label}  [{model_used}]{'  ' + note if note else ''}\n{text}")
        if engine is not None:
            with engine.begin() as conn:
                save(conn, s, text, model_used, verified, h)
            stats["written"] += 1

    print(f"\n{len(subjects)} subjects: {stats}")


if __name__ == "__main__":
    main()
