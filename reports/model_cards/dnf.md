# Model card: dnf

Probability that a starter does not finish (not classified, not disqualified).

- Target: `target_dnf` (binary)
- Model: scikit-learn HistGradientBoosting, params {'learning_rate': 0.08, 'max_leaf_nodes': 31}
- Train 2010-2021, validate 2022, test 2023-2024
- Production model retrained on 2010-2025
- Rows: {'train': 5042, 'valid': 439, 'train_valid': 5481, 'test': 912, 'current': 322}

## Test-season results

Beats the baseline on **pr_auc**.

| Metric | Model | Baseline |
| --- | --- | --- |
| roc_auc | 0.5538 | 0.4951 |
| pr_auc | 0.1188 | 0.0973 |
| brier | 0.0943 | 0.0952 |
| log_loss | 0.3374 | 0.4697 |
| positive_rate | 0.0998 | 0.0998 |

## Most important features (permutation importance on the test seasons)

1. `driver_avg_grid_last5` (0.00794)
2. `quali_gap_to_fastest_ms` (0.00436)
3. `grid_position` (0.00418)
4. `driver_avg_finish_last3` (0.00373)
5. `constructor_dnf_rate_last10` (0.00373)
6. `quali_position` (0.00372)
7. `driver_points_last5` (0.00265)
8. `driver_career_starts_before` (0.0021)
9. `driver_circuit_avg_finish_before` (0.00121)
10. `driver_career_wins_before` (0.00111)
