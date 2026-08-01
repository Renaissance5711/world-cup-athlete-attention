# Goalkeeper role-allocation analysis data dictionary

## `outputs/goalkeeper/goalkeeper_on_target_shots.csv`

One row per on-target shot that ended as a goal or goalkeeper save.

- `sb_match_id`, `match_id`: StatsBomb and Fjelstul match identifiers.
- `shot_id`, `event_index`, `period`, `minute`, `second`: shot-event identifiers and timing.
- `shooter_player_id`, `goalkeeper_player_id`: mapped Fjelstul player identifiers.
- `sb_shooter_id`, `sb_goalkeeper_id`: StatsBomb player identifiers.
- `shot_xg`: StatsBomb expected-goals value.
- `is_save`, `is_goal`: mutually exclusive realised-outcome indicators.
- `shot_outcome`, `goalkeeper_outcome`: source event labels.
- `play_pattern`, `body_part`, `technique`: shot context.
- `under_pressure`, `open_goal`, `one_on_one`, `deflected`: binary context fields.
- `start_x`, `start_y`, `shot_distance`, `shot_angle`: shot-origin geometry.
- `end_x`, `end_y`, `end_z`: shot-end coordinates when supplied.
- `shooter_baseline_log_views`, `goalkeeper_baseline_log_views`: October mean daily `ln(1 + pageviews)`.
- `shooter_immediate_attention_log_lift`, `goalkeeper_immediate_attention_log_lift`: 0–1-day proportional attention outcomes.
- `shooter_additional_pageviews`, `goalkeeper_additional_pageviews`: winsorised additional 0–1-day pageviews.
- `predicted_save_probability`: match-held-out predicted probability of a save.
- `crossfit_fold`: match-grouped cross-fitting fold.
- `performance_category`: model-based descriptive class.

## Performance categories

Using a 0.50 held-out predicted-save threshold:

- `heroic_save`: saved shot with predicted save probability below 0.50.
- `routine_save`: saved shot with predicted save probability at least 0.50.
- `understandable_goal`: goal with predicted save probability below 0.50.
- `unexpected_goal`: goal with predicted save probability at least 0.50.

These labels describe model-implied expected difficulty. They do not assign definitive fault, credit, blame or public responsibility.

## `outputs/goalkeeper/shooter_goalkeeper_bilateral_pairs.csv`

One row per shooter-goalkeeper-match pair, aggregated across that pair's on-target shots.

- `on_target_shots`, `goals`, `saves`: pair-level event counts.
- `mean_expected_save_probability`: average held-out expected-save probability.
- `heroic_save_count`, `routine_save_count`, `understandable_goal_count`, `unexpected_goal_count`: class counts.
- `heroic_save_intensity`, `unexpected_goal_intensity`, `net_shooter_surprise`: retained legacy diagnostic quantities used to reconstruct the prior R14 specification; they are not the v1.4.0 confirmatory estimand.
- `focal_category`: descriptive pair class used in legacy all-pair category summaries.
- `attention_log_lift_difference`: shooter minus goalkeeper immediate proportional attention.
- `additional_pageview_difference`: shooter minus goalkeeper winsorised additional pageviews.
- `baseline_log_visibility_difference`: shooter minus goalkeeper baseline visibility.
- `team_win_difference`: shooter-team win indicator minus goalkeeper-team win indicator.

## Expected-save diagnostics

`expected_save_diagnostics.json` records observations, matches, save rate, held-out Brier score, intercept-only Brier benchmark, log loss, ROC AUC, fold count, leakage count and shot-category totals.

## Primary v1.4.0 outputs

### `goalkeeper_bilateral_sample_structure.csv`

Reports total shots and pairs; single-shot, multi-shot and mixed-outcome pair counts; and the number of goals and saves in the primary single-shot sample.

### `goalkeeper_outcome_gradient_results.csv`

Primary 303-pair single-shot regression results. Each outcome contains three inferential rows:

- `goal_vs_save_at_mean_expected_save`: realised goal relative to save at the centered expected-save value.
- `expected_save_gradient_among_saves`: expected-save-probability slope among saved shots.
- `expected_save_gradient_among_goals`: linear-combination slope among goals.

The file includes coefficients, clustered standard errors, p-values, confidence intervals, sample size, match count and the centering value. Models include baseline visibility difference and match fixed effects; standard errors are clustered by match.

### `goalkeeper_outcome_balance_sensitivity.csv`

All-386-pair sensitivity results using pair goal share, centered mean expected-save probability, their interaction, baseline visibility difference, shot count and match fixed effects. Standard errors are clustered by match.

### `goalkeeper_single_shot_category_means.csv`

Unadjusted category counts and distribution summaries for the 303 single-shot pairs.

### `goalkeeper_single_shot_category_contrasts.csv`

Adjusted direct contrasts within saves (`routine_save - heroic_save`) and within goals (`unexpected_goal - understandable_goal`) for the single-shot sample.

## Legacy/descriptive pair outputs

- `goalkeeper_bilateral_model_results.csv`: four-category all-pair regressions retained for transparent reconstruction of the R14 category specification.
- `goalkeeper_bilateral_contrasts.csv`: corresponding direct all-pair category contrasts.
- `goalkeeper_category_means.csv`: unadjusted all-pair category summaries.

The prior `goalkeeper_continuous_responsibility_results.csv` is intentionally excluded from the v1.4.0 release because that specification does not separate the realised outcome from within-outcome expected-difficulty gradients.

## Headline-audit files

- `goalkeeper_news_headlines_retained.csv`: strictly retained post-match indexed headlines.
- `goalkeeper_news_match_summary.csv`: headline counts and framing by goalkeeper-match.
- `goalkeeper_news_validation_panel.csv`: performance and framing merged at goalkeeper-match level; the historical filename is retained for code compatibility.
- `goalkeeper_news_validation_results.csv`: Fisher exact comparisons for any praise and any blame; the historical filename is retained for code compatibility.
- `goalkeeper_news_query_manifest.csv`: query coverage and filtering audit.

Framing values are `praise`, `blame` or `neutral`, assigned with conservative goalkeeper-specific dictionaries. The analysis is exploratory because search coverage is sparse, ranked and conditional on a headline being retained.
