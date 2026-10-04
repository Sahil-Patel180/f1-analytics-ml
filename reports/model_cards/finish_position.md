# Model card: finish_position

Finishing order of the field (position_order, retirements at the back).

- Target: `target_finish_position` (ranking)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.03, 'max_leaf_nodes': 31}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 5042, 'valid': 439, 'train_valid': 5481, 'test': 912, 'current': 322}

## Test-season results

Beats the baseline on **spearman_per_race**.

| Metric | Model | Baseline |
| --- | --- | --- |
| spearman_per_race | 0.6999 | 0.6659 |
| mae_places | 3.1513 | 3.261 |
| winner_hit_rate | 0.5652 | 0.5435 |

## Most important features (permutation importance on the test seasons)

1. `grid_position` (0.56032)
2. `quali_position` (0.33245)
3. `driver_avg_grid_last5` (0.13937)
4. `driver_career_wins_before` (0.1315)
5. `driver_avg_finish_last5` (0.07879)
6. `driver_points_last5` (0.07139)
7. `constructor_points_last5` (0.06804)
8. `quali_gap_to_fastest_ms` (0.05905)
9. `champ_position_before` (0.05467)
10. `constructor_best_finish_avg_last5` (0.03827)
