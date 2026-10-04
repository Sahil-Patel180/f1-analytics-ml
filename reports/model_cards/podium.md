# Model card: podium

Probability that a starter finishes in the top 3.

- Target: `target_podium` (binary)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.08, 'max_leaf_nodes': 15}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 5042, 'valid': 439, 'train_valid': 5481, 'test': 912, 'current': 322}

## Test-season results

Beats the baseline on **pr_auc**.

| Metric | Model | Baseline |
| --- | --- | --- |
| roc_auc | 0.9247 | 0.9107 |
| pr_auc | 0.7058 | 0.6639 |
| brier | 0.0716 | 0.1042 |
| log_loss | 0.2337 | 1.4391 |
| positive_rate | 0.1513 | 0.1513 |
| top3_hit_rate | 0.6449 | 0.6594 |

## Most important features (permutation importance on the test seasons)

1. `grid_position` (0.14896)
2. `constructor_points_last5` (0.03823)
3. `quali_position` (0.03411)
4. `driver_avg_finish_last3` (0.0248)
5. `driver_avg_grid_last5` (0.01267)
6. `quali_gap_to_fastest_ms` (0.00886)
7. `champ_position_before` (0.00804)
8. `constructor_best_finish_avg_last5` (0.00765)
9. `driver_career_starts_before` (0.00315)
10. `driver_age_years` (0.00197)
