"""System prompts per narrative type. Bump PROMPT_VERSION when any prompt changes:
it is part of the fact-pack hash, so changed prompts regenerate their narratives."""

PROMPT_VERSION = "v2"

_RULES = """Rules you must follow:
- Use ONLY the facts in the JSON the user sends. Do not add any number, date, name, cause, quote or opinion that is not in it.
- Copy numbers exactly as written in the JSON (keep '1:27.452', '64%'). Never calculate new numbers.
- Write positions as digits with a P prefix when helpful (P1, P3), never as words.
- If a fact is missing, leave it out. Never guess.
- Skip any statistic whose value is 0. Never write '0 wins', '0 podiums' or 'finished 0 poles'.
- Plain English, past tense, no headings, no bullet points, no emojis.
- Output only the paragraph."""

SYSTEM_PROMPTS = {
    "race_recap": f"""You write short Formula 1 race recaps for a fan dashboard.
Write one paragraph of 70-110 words: who won and from where on the grid, the podium,
pole, fastest lap, the biggest gain and notable retirements if given.
{_RULES}""",
    "driver_season": f"""You write short Formula 1 season summaries for one driver.
Write one paragraph of 60-100 words: team, results (wins, podiums, poles, points),
championship position, best result and how they compared with their teammate.
If season_complete is false, say the season is still in progress.
{_RULES}""",
    "prediction": f"""You explain a machine-learning prediction to a fan in one paragraph of 40-70 words:
the predicted podium chance, what the model relies on most, and what actually happened.
Do not claim the model is certain. Do not explain how machine learning works.
{_RULES}""",
    "model_report": f"""You summarise a machine-learning model evaluation for a portfolio README.
Write one paragraph of 50-90 words: what the model predicts, the seasons used,
whether it beat the baseline on each metric (higher is better for auc, f1, spearman,
accuracy and hit rates; lower is better for mae, rmse, brier and log_loss), and its most important features.
{_RULES}""",
}


def template_fallback(kind: str, facts: dict) -> str:
    """Plain, always-correct text used when the LLM fails verification twice."""
    if kind == "race_recap":
        p = facts.get("podium", [])
        win = p[0] if p else None
        s = f"{facts['season']} {facts['race']} (round {facts['round']}). "
        if win:
            s += f"{win['driver']} ({win['team']}) won"
            if facts.get("winner_started_from_grid"):
                s += f" from P{facts['winner_started_from_grid']} on the grid"
            s += ". "
        if len(p) >= 3:
            s += f"{p[1]['driver']} was P2 and {p[2]['driver']} P3. "
        if facts.get("pole_sitter"):
            s += f"Pole: {facts['pole_sitter']['driver']}. "
        if facts.get("fastest_lap"):
            s += f"Fastest lap: {facts['fastest_lap']['driver']} ({facts['fastest_lap']['time']}). "
        s += f"Retirements: {facts.get('retirements', 0)}."
        return s
    if kind == "driver_season":
        s = (f"{facts['driver']} drove for {', '.join(facts.get('teams', []))} in {facts['season']}: "
             f"{facts.get('wins', 0)} wins, {facts.get('podiums', 0)} podiums, "
             f"{facts.get('points', 0)} points")
        if facts.get("championship_position"):
            s += f", P{facts['championship_position']} in the championship"
        return s + "."
    if kind == "prediction":
        return (f"The model gave {facts['driver']} a {facts['predicted_podium_chance']} podium chance "
                f"at the {facts['race']}. Actual result: {facts.get('actual_result', 'n/a')}.")
    if kind == "model_report":
        parts = [f"{m['metric']} {m['model']} vs baseline {m['baseline']}" for m in facts.get("metrics", [])]
        return f"{facts['model']} (trained {facts['trained_on_seasons']}, tested {facts['tested_on_seasons']}): " + "; ".join(parts) + "."
    raise ValueError(kind)