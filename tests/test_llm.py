import pandas as pd

from f1ml.llm import facts as F
from f1ml.llm.narrate import Subject, generate
from f1ml.llm.prompts import template_fallback
from f1ml.llm.verify import verify

RACE = {
    "race": "Monaco Grand Prix", "season": 2024, "round": 8, "circuit": "Circuit de Monaco",
    "podium": [
        {"position": 1, "driver": "Charles Leclerc", "team": "Ferrari", "grid": 1},
        {"position": 2, "driver": "Oscar Piastri", "team": "McLaren", "grid": 2},
        {"position": 3, "driver": "Carlos Sainz", "team": "Ferrari", "grid": 3},
    ],
    "pole_sitter": {"driver": "Charles Leclerc", "team": "Ferrari"},
    "winner_started_from_grid": 1,
    "fastest_lap": {"driver": "Lewis Hamilton", "team": "Mercedes", "time": "1:14.165"},
    "retirements": 4,
    "starters": 20,
}
ROSTER = {"Leclerc", "Piastri", "Sainz", "Hamilton", "Verstappen", "Ferrari", "McLaren", "Mercedes", "Red Bull"}


# ---------------------------------------------------------------- verifier
def test_verify_accepts_faithful_text():
    text = ("Charles Leclerc won the 2024 Monaco Grand Prix from pole for Ferrari, ahead of "
            "Oscar Piastri in P2 and Carlos Sainz in P3. Lewis Hamilton set the fastest lap, "
            "1:14.165, and 4 of the 20 starters retired in round 8.")
    v = verify(text, RACE, ROSTER)
    assert v.ok, v.reason()


def test_verify_rejects_invented_number():
    v = verify("Leclerc won by 7.2 seconds.", RACE, ROSTER)
    assert not v.ok and "7.2" in v.unknown_numbers


def test_verify_rejects_invented_position_with_p_prefix():
    v = verify("Hamilton finished P7.", RACE, ROSTER)
    assert not v.ok and "7" in v.unknown_numbers


def test_verify_rejects_number_words():
    v = verify("Leclerc won his ninth race.", RACE, ROSTER)
    assert not v.ok and "9" in v.unknown_numbers


def test_verify_rejects_name_not_in_facts():
    v = verify("Verstappen could not challenge Leclerc.", RACE, ROSTER)
    assert not v.ok and v.unknown_names == ["Verstappen"]


def test_verify_allows_f1_and_ordinals_from_facts():
    v = verify("In round 8 of the F1 season, Sainz took 3rd.", RACE, ROSTER)
    assert v.ok, v.reason()


# ---------------------------------------------------------------- generation loop
class FakeClient:
    model = "fake-llm"

    def __init__(self, replies):
        self.replies = list(replies)
        self.temperatures = []

    def chat(self, system, user, temperature=0.2, max_tokens=260):
        self.temperatures.append(temperature)
        return self.replies.pop(0)


def test_generate_retries_then_accepts():
    client = FakeClient(["Leclerc won by 9 seconds.", "Charles Leclerc won from P1 for Ferrari."])
    text, model, ok, _ = generate(client, Subject("race_recap", RACE), ROSTER)
    assert ok and model == "fake-llm" and text.startswith("Charles")
    assert client.temperatures == [0.2, 0.0]


def test_generate_falls_back_to_template():
    client = FakeClient(["Verstappen won.", "Leclerc won by 9 seconds."])
    text, model, ok, note = generate(client, Subject("race_recap", RACE), ROSTER)
    assert model == "template-fallback" and "LLM rejected" in note
    assert verify(text, RACE, ROSTER).ok          # the fallback itself is always faithful


def test_every_template_passes_its_own_verifier():
    samples = {
        "race_recap": RACE,
        "driver_season": {"driver": "Lando Norris", "season": 2024, "teams": ["McLaren"], "wins": 4,
                          "podiums": 13, "points": 374, "championship_position": 2},
        "prediction": {"driver": "Lando Norris", "race": "2024 Dutch Grand Prix",
                       "predicted_podium_chance": "71%", "actual_result": "1"},
        "model_report": {"model": "podium", "trained_on_seasons": "2010-2022", "tested_on_seasons": "2023-2024",
                         "metrics": [{"metric": "pr_auc", "model": 0.706, "baseline": 0.664}]},
    }
    for kind, facts in samples.items():
        assert verify(template_fallback(kind, facts), facts).ok, kind


# ---------------------------------------------------------------- fact packs
def test_fmt_ms():
    assert F.fmt_ms(86572) == "1:26.572"
    assert F.fmt_ms(3903456) == "1:05:03.456"
    assert F.fmt_ms(None) is None


def test_race_recap_facts_from_rows():
    base = dict(race_id=1, race_name="Test GP", season_year=2024, round=3, circuit_name="Test Ring",
                country="Nowhere", race_date="2024-04-07", session_type="Race", status="Finished",
                fastest_lap_ms=None)
    rows = pd.DataFrame([
        {**base, "driver_name": "A One", "constructor_name": "Team X", "position_order": 1, "finish_position": 1,
         "grid_position": 2, "is_pole": 0, "fastest_lap_rank": 2, "positions_gained": 1, "is_started": 1, "is_dnf": 0},
        {**base, "driver_name": "B Two", "constructor_name": "Team Y", "position_order": 2, "finish_position": 2,
         "grid_position": 1, "is_pole": 1, "fastest_lap_rank": 1, "fastest_lap_ms": 81234, "positions_gained": -1,
         "is_started": 1, "is_dnf": 0},
        {**base, "driver_name": "C Three", "constructor_name": "Team X", "position_order": 3, "finish_position": 3,
         "grid_position": 9, "is_pole": 0, "fastest_lap_rank": 3, "positions_gained": 6, "is_started": 1, "is_dnf": 0},
        {**base, "driver_name": "D Four", "constructor_name": "Team Y", "position_order": 4, "finish_position": None,
         "grid_position": 3, "is_pole": 0, "fastest_lap_rank": None, "positions_gained": None, "is_started": 1,
         "is_dnf": 1, "status": "Engine"},
    ])
    f = F.race_recap_facts(rows)
    assert [p["driver"] for p in f["podium"]] == ["A One", "B Two", "C Three"]
    assert f["winner_started_from_grid"] == 2
    assert f["pole_sitter"]["driver"] == "B Two"
    assert f["fastest_lap"] == {"driver": "B Two", "team": "Team Y", "time": "1:21.234"}
    assert f["biggest_gain"]["driver"] == "C Three" and f["biggest_gain"]["places_gained"] == 6
    assert f["retirements"] == 1 and f["notable_retirements"][0]["reason"] == "Engine"
    assert "sprint_winner" not in f
