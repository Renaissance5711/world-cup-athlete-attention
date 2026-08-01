import numpy as np
import pandas as pd

from ijsms.expected_save import classify_save_outcome, cross_fit_expected_save


def _synthetic_shots():
    rows = []
    rng = np.random.default_rng(7)
    for match in range(12):
        for shot in range(12):
            xg = 0.05 + 0.9 * rng.random()
            save_probability = 1 - xg
            actual_save = int(rng.random() < save_probability)
            rows.append({
                "sb_match_id": match,
                "shot_xg": xg,
                "shot_distance": 5 + 25 * rng.random(),
                "shot_angle": 0.1 + 1.2 * rng.random(),
                "goalkeeper_distance_to_goal_center": rng.random() * 5,
                "end_z": rng.random() * 2.5,
                "under_pressure": int(rng.random() < 0.4),
                "one_on_one": int(rng.random() < 0.2),
                "open_goal": 0,
                "first_time": int(rng.random() < 0.3),
                "deflected": 0,
                "minute": shot * 7,
                "score_diff_before_shot": 0,
                "shot_type": "Open Play",
                "body_part": "Right Foot",
                "technique": "Normal",
                "play_pattern": "Regular Play",
                "actual_save": actual_save,
                "actual_goal": 1 - actual_save,
            })
    return pd.DataFrame(rows)


def test_cross_fit_expected_save_produces_out_of_match_probabilities_and_beats_intercept():
    predicted, diagnostics = cross_fit_expected_save(_synthetic_shots(), n_splits=6)
    assert predicted["expected_save_probability"].between(0, 1).all()
    assert predicted["prediction_fold"].notna().all()
    assert diagnostics["group_leakage_count"] == 0
    assert diagnostics["brier_score"] < diagnostics["intercept_brier_score"]


def test_classify_save_outcome_uses_actual_result_and_half_probability_threshold():
    assert classify_save_outcome(1, 0.20) == "heroic_save"
    assert classify_save_outcome(1, 0.80) == "routine_save"
    assert classify_save_outcome(0, 0.20) == "understandable_goal"
    assert classify_save_outcome(0, 0.80) == "unexpected_goal"
