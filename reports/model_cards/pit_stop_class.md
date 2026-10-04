# Model card: pit_stop_class

Number of pit stops (0-1, 2, 3+) for classified finishers, 2010 onward.

- Target: `target_stop_class` (multiclass)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.03, 'max_leaf_nodes': 31}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 4204, 'valid': 376, 'train_valid': 4580, 'test': 817, 'current': 264}

## Test-season results

**Does not beat the baseline** on the tuning metric.

| Metric | Model | Baseline |
| --- | --- | --- |
| macro_f1 | 0.4313 | 0.4549 |
| accuracy | 0.4614 | 0.4786 |

## Most important features (permutation importance on the test seasons)

1. `circuit_avg_stops_prev_edition` (0.08037)
2. `race_laps` (0.04638)
3. `round` (0.04454)
4. `has_sprint` (0.00983)
5. `grid_position` (0.00649)
6. `constructor_avg_stop_ms_last5` (-0.0018)
