import numpy as np
import pandas as pd
import pytest

from f1ml import metrics as M
from f1ml.splits import SeasonSplit, walk_forward_seasons


def _frame(seasons):
    return pd.DataFrame({"season_year": seasons, "x": range(len(seasons))})


def test_split_parts_are_disjoint_and_ordered():
    df = _frame(list(range(2008, 2027)))
    parts = SeasonSplit().parts(df)
    assert parts["train"]["season_year"].between(2010, 2021).all()
    assert set(parts["valid"]["season_year"]) == {2022}
    assert parts["test"]["season_year"].between(2023, 2024).all()
    assert parts["train"]["season_year"].max() < parts["test"]["season_year"].min()
    assert len(parts["train_valid"]) == len(parts["train"]) + len(parts["valid"])


def test_split_rejects_overlap():
    with pytest.raises(ValueError):
        SeasonSplit(train_to=2023, valid=2022).check()


def test_walk_forward_trains_only_on_the_past():
    df = _frame(list(range(2010, 2025)))
    for season, train, test in walk_forward_seasons(df, 2015, 2024):
        assert train["season_year"].max() < season
        assert set(test["season_year"]) == {season}


def test_top_k_hit_rate():
    race = [1, 1, 1, 1, 2, 2, 2, 2]
    y = [1, 1, 1, 0, 1, 1, 1, 0]
    perfect = [4, 3, 2, 1, 4, 3, 2, 1]
    assert M.top_k_hit_rate(race, y, perfect, k=3) == 1.0
    worst = [1, 2, 3, 4, 1, 2, 3, 4]
    assert M.top_k_hit_rate(race, y, worst, k=3) == pytest.approx(2 / 3)


def test_ranking_perfect_order():
    race = [1] * 5 + [2] * 5
    y = [1, 2, 3, 4, 5] * 2
    m = M.ranking_metrics(race, y, y)
    assert m["spearman_per_race"] == pytest.approx(1.0)
    assert m["mae_places"] == 0
    assert m["winner_hit_rate"] == 1.0


def test_sign_accuracy_ignores_no_change():
    m = M.regression_metrics([2, -1, 0], [1.5, -0.2, 3.0], sign=True)
    assert m["sign_accuracy"] == 1.0
    assert m["mae"] == pytest.approx(np.mean([0.5, 0.8, 3.0]))
