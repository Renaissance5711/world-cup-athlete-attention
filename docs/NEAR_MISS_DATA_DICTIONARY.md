# Shot-opportunity panel data dictionary

The main processed file is `outputs/near_miss/player_match_shot_panel.csv`.

- `player_id`, `match_id`: Fjelstul stable identifiers.
- `sb_match_id`, `sb_player_id`: StatsBomb identifiers.
- `minutes_played`: participation duration derived from lineup positions and substitutions.
- `shots`, `non_penalty_shots`: total and non-penalty attempts.
- `total_xg`, `npxg`: total and non-penalty expected goals.
- `max_shot_xg`, `max_npxg`: maximum xG among all/non-penalty attempts.
- `penalty_attempts`: number of penalty shots.
- `shots_on_target`: attempts coded on target.
- `open_play_shots`, `under_pressure_shots`: shot-context counts.
- `first_shot_score_diff`: player's team score minus opponent score at the first shot.
- `shot_pressure_share`, `open_play_share`: context counts divided by shot count.
- `fjelstul_non_penalty_scorer`: primary treatment indicator, consistent with the manuscript goal source.
- `statsbomb_non_penalty_scorer`: sensitivity treatment indicator.
- `overlap_weight`: propensity-based overlap weight in `near_miss_primary_sample.csv`.
- `propensity_score`: estimated probability of scoring conditional on observed shot opportunity.

Outcome variables reuse the pageview definitions documented in `base_archive/docs/DATA_DICTIONARY.md`.
