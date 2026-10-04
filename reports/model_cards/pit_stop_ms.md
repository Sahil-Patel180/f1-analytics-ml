# Model card: pit_stop_ms

Average pit-lane time per stop in ms (stops over 120 s excluded), 2010 onward.

- Target: `target_avg_stop_ms` (regression)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.08, 'max_leaf_nodes': 31}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 4666, 'valid': 409, 'train_valid': 5075, 'test': 851, 'current': 288}

## Test-season results

Beats the baseline on **mae**.

| Metric | Model | Baseline |
| --- | --- | --- |
| mae | 2468.8717 | 3549.2688 |
| rmse | 4865.3163 | 5479.2613 |

## Most important features (permutation importance on the test seasons)

1. `race_laps` (977.25521)
2. `round` (75.3917)
3. `constructor_avg_stop_ms_last5` (60.67097)
4. `grid_position` (24.85277)
5. `circuit_avg_stops_prev_edition` (4.31665)
6. `has_sprint` (3.03287)
