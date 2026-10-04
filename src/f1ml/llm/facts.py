"""Fact packs: small, pre-computed JSON objects the LLM is allowed to talk about.

SQL fetches rows; the pure functions below turn rows into facts, so they can be
unit-tested without a database. Every value is already rounded and labelled
with its unit, so the model never has to calculate anything.
"""
from __future__ import annotations

import json

import pandas as pd


def fmt_ms(ms) -> str | None:
    """86572 -> '1:26.572'; 3903456 -> '1:05:03.456'."""
    if ms is None or pd.isna(ms):
        return None
    ms = int(ms)
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s = rem / 1000
    return f"{h}:{m:02d}:{s:06.3f}" if h else f"{m}:{s:06.3f}"


def _int(v):
    return None if v is None or pd.isna(v) else int(v)


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None and v != [] and v != {}}


# ---------------------------------------------------------------- race recap
def race_recap_facts(rows: pd.DataFrame) -> dict:
    """rows: gold.vw_f1_master for one race (Race + Sprint sessions)."""
    race = rows[rows["session_type"] == "Race"].sort_values("position_order")
    if race.empty:
        raise ValueError("no race-session rows")
    first = race.iloc[0]

    def driver(r):
        return {"driver": r["driver_name"], "team": r["constructor_name"]}

    podium = [
        {"position": int(r["finish_position"]), **driver(r), "grid": _int(r["grid_position"])}
        for _, r in race[race["finish_position"].between(1, 3)].iterrows()
    ]
    pole = race[race["is_pole"] == 1]
    fl = race[race["fastest_lap_rank"] == 1]
    gainers = race[race["positions_gained"].notna()].sort_values("positions_gained", ascending=False)
    started = race[race["is_started"] == 1]
    dnfs = started[started["is_dnf"] == 1].sort_values("grid_position")
    sprint = rows[(rows["session_type"] == "Sprint") & (rows["finish_position"] == 1)]

    facts = {
        "race": first["race_name"],
        "season": int(first["season_year"]),
        "round": int(first["round"]),
        "circuit": first["circuit_name"],
        "country": first["country"],
        "date": str(pd.Timestamp(first["race_date"]).date()),
        "starters": int(len(started)),
        "podium": podium,
        "pole_sitter": driver(pole.iloc[0]) if len(pole) else None,
        "winner_started_from_grid": _int(race[race["finish_position"] == 1]["grid_position"].iloc[0])
        if (race["finish_position"] == 1).any() else None,
        "fastest_lap": ({**driver(fl.iloc[0]), "time": fmt_ms(fl.iloc[0]["fastest_lap_ms"])}
                        if len(fl) else None),
        "biggest_gain": ({**driver(gainers.iloc[0]),
                          "from_grid": _int(gainers.iloc[0]["grid_position"]),
                          "finished": _int(gainers.iloc[0]["finish_position"]),
                          "places_gained": _int(gainers.iloc[0]["positions_gained"])}
                         if len(gainers) and gainers.iloc[0]["positions_gained"] > 0 else None),
        "retirements": int(len(dnfs)),
        "notable_retirements": [
            {**driver(r), "grid": _int(r["grid_position"]), "reason": r["status"]}
            for _, r in dnfs.head(3).iterrows()
        ],
        "sprint_winner": driver(sprint.iloc[0]) if len(sprint) else None,
    }
    return _drop_none(facts)


# ---------------------------------------------------------------- driver season
def driver_season_facts(agg: pd.Series, driver_rows: pd.DataFrame, team_rows: pd.DataFrame) -> dict:
    """agg: one gold.agg_driver_season row joined to dim_driver (full_name, nationality).
    driver_rows: the driver's Race rows that season (vw_f1_master).
    team_rows: Race rows of the driver's team(s) that season, all drivers."""
    teams = list(dict.fromkeys(driver_rows.sort_values("round")["constructor_name"]))
    mates = (team_rows[team_rows["driver_id"] != agg["driver_id"]]
             .groupby("driver_name")["points"].sum().sort_values(ascending=False))
    best = driver_rows[driver_rows["finish_position"].notna()].sort_values("finish_position").head(1)
    facts = {
        "driver": agg["full_name"],
        "nationality": agg.get("nationality"),
        "season": int(agg["season_year"]),
        "teams": teams,
        "races_started": _int(agg["race_starts"]),
        "wins": _int(agg["wins"]),
        "podiums": _int(agg["podiums"]),
        "poles": _int(agg["poles"]),
        "retirements": _int(agg["dnfs"]),
        "points": float(agg["points_scored"]),
        "championship_position": _int(agg["final_position"]),
        "season_complete": bool(agg.get("is_season_complete", 1)),
        "champion": bool(agg["is_champion"]),
        "best_result": ({"position": _int(best.iloc[0]["finish_position"]),
                         "race": best.iloc[0]["race_name"]} if len(best) else None),
        "teammates": [{"driver": n, "points": float(p)} for n, p in mates.head(2).items()],
    }
    if facts["points"].is_integer():
        facts["points"] = int(facts["points"])
    for t in facts["teammates"]:
        if float(t["points"]).is_integer():
            t["points"] = int(t["points"])
    return _drop_none(facts)


# ---------------------------------------------------------------- prediction explainer
def prediction_facts(row: pd.Series, top_features: list[dict]) -> dict:
    """row: one ml.predictions row (podium model) joined to race/driver/result columns."""
    feature_names = {
        "grid_position": "grid position", "quali_position": "qualifying position",
        "driver_avg_finish_last3": "average finish over the last 3 races",
        "driver_avg_finish_last5": "average finish over the last 5 races",
        "constructor_points_last5": "team points over the last 5 races",
        "driver_points_last5": "driver points over the last 5 races",
        "quali_gap_to_fastest_ms": "gap to the fastest qualifying time",
        "champ_position_before": "championship position before the race",
        "champ_points_before": "championship points before the race",
        "driver_podiums_last5": "podiums in the last 5 races",
    }
    facts = {
        "driver": row["driver_name"],
        "team": row["constructor_name"],
        "race": row["race_label"],
        "predicted_podium_chance": f"{round(float(row['probability']) * 100)}%",
        "grid": _int(row.get("grid_position")),
        "qualifying_position": _int(row.get("quali_position")),
        "actual_finish": _int(row.get("finish_position")),
        "actual_result": row.get("position_text"),
        "model_relies_most_on": [feature_names.get(f["feature"], f["feature"].replace("_", " "))
                                 for f in top_features[:3]],
    }
    return _drop_none(facts)


# ---------------------------------------------------------------- model report
def model_report_facts(run: pd.Series) -> dict:
    """run: one ml.model_runs row (eval version) with metrics_json."""
    m = json.loads(run["metrics_json"])
    model, base = m.get("model", {}), m.get("baseline", {})
    keep = [k for k in model if k in base and k != "positive_rate"]
    return _drop_none({
        "model": run["model_name"],
        "trained_on_seasons": f"{int(run['train_from_season'])}-{int(run['train_to_season'])}",
        "tested_on_seasons": f"{int(run['test_from_season'])}-{int(run['test_to_season'])}",
        "metrics": [{"metric": k, "model": round(model[k], 3), "baseline": round(base[k], 3)} for k in keep],
        "most_important_features": [f["feature"] for f in m.get("top_features", [])[:3]],
    })
