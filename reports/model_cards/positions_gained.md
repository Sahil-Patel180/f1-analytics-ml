# Model card: positions_gained

Grid position minus finishing position, for classified starters with a grid slot.

- Target: `target_positions_gained` (regression)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.03, 'max_leaf_nodes': 15}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 4177, 'valid': 366, 'train_valid': 4543, 'test': 794, 'current': 248}

## Test-season results

Beats the baseline on **mae**.

| Metric | Model | Baseline |
| --- | --- | --- |
| mae | 2.25 | 2.8111 |
| rmse | 3.1676 | 3.9432 |
| sign_accuracy | 0.6961 | 0.0 |

## Most important features (permutation importance on the test seasons)

1. `grid_position` (2.33617)
2. `constructor_points_last5` (0.19273)
3. `champ_points_before` (0.10573)
4. `quali_gap_to_fastest_ms` (0.07652)
5. `champ_position_before` (0.06067)
6. `driver_podiums_last5` (0.03456)
7. `driver_points_last5` (0.02696)
8. `driver_age_years` (0.02409)
9. `constructor_best_finish_avg_last5` (0.01985)
10. `round` (0.01481)
