"""Season-based splits. Never random: a random split puts drivers of the same
race on both sides and lets later seasons teach the model about earlier ones."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SeasonSplit:
    train_from: int = 2010      # one points system (25 for a win) from 2010
    train_to: int = 2021
    valid: int = 2022           # hyper-parameter choice only
    test_from: int = 2023       # reported once, never used for choices
    test_to: int = 2024

    def check(self) -> None:
        if not (self.train_from <= self.train_to < self.valid < self.test_from <= self.test_to):
            raise ValueError(f"seasons must be ordered train < valid < test: {self}")

    def parts(self, df: pd.DataFrame) -> dict[str, pd.DataFrame]:
        self.check()
        s = df["season_year"]
        return {
            "train": df[(s >= self.train_from) & (s <= self.train_to)],
            "valid": df[s == self.valid],
            "train_valid": df[(s >= self.train_from) & (s <= self.valid)],
            "test": df[(s >= self.test_from) & (s <= self.test_to)],
        }


def walk_forward_seasons(df: pd.DataFrame, first_test: int, last_test: int, train_from: int = 2010):
    """Yield (season, train_df, test_df): train on every season before `season`, test on it."""
    for season in range(first_test, last_test + 1):
        train = df[(df["season_year"] >= train_from) & (df["season_year"] < season)]
        test = df[df["season_year"] == season]
        if len(train) and len(test):
            yield season, train, test
