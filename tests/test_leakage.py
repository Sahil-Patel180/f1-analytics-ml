import pandas as pd
import pytest

from f1ml.tasks import ID_COLUMNS, TASKS, Task
from f1ml.train import assert_no_leakage


@pytest.mark.parametrize("name", list(TASKS))
def test_features_never_ids_or_targets(name):
    task = TASKS[name]
    for f in task.features:
        assert not f.startswith("target_"), f"{name}: target column {f} used as a feature"
        assert f not in ID_COLUMNS, f"{name}: id column {f} used as a feature"
    assert task.target.startswith("target_")


def test_assert_no_leakage_rejects_target_feature():
    base = TASKS["podium"]
    bad = Task(**{**base.__dict__, "features": base.features + ["target_podium"]})
    with pytest.raises(ValueError):
        assert_no_leakage(bad)


def test_assert_no_leakage_rejects_id_feature():
    base = TASKS["dnf"]
    bad = Task(**{**base.__dict__, "features": base.features + ["race_id"]})
    with pytest.raises(ValueError):
        assert_no_leakage(bad)


def test_baselines_use_no_same_race_targets():
    """Baselines may read the test frame's pre-race columns, but never its target."""
    test = pd.DataFrame({
        "grid_position": [1, 2, 5], "constructor_dnf_rate_last10": [0.1, None, 0.3],
        "circuit_avg_stops_prev_edition": [1.2, 2.0, None],
        "constructor_avg_stop_ms_last5": [23000, None, 25000],
        "race_id": [1, 1, 1],
    })
    train = pd.DataFrame({"target_dnf": [0, 1], "target_stop_class": ["2", "2"],
                          "target_avg_stop_ms": [24000, 26000]})
    for name in TASKS:
        value, _ = TASKS[name].baseline(test, train)   # would KeyError if it touched test targets
        assert len(value) == len(test)
