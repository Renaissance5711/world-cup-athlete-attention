import pandas as pd
import pytest

from ijsms.rank_reversal import validate_rank_input


def test_validate_rank_input_rejects_nonpositive_baseline():
    frame = pd.DataFrame({
        "player_id": ["A"],
        "player_name": ["A"],
        "observed_proportional_log_lift": [1.0],
        "observed_additional_pageviews": [100.0],
        "baseline_views": [0.0],
        "proportional_rank": [1],
        "additive_rank": [1],
    })
    with pytest.raises(ValueError, match="baseline_views must be positive"):
        validate_rank_input(frame)


def test_build_pairwise_decomposition_distinguishes_observed_and_identity_ordering():
    from ijsms.rank_reversal import build_pairwise_decomposition

    frame = pd.DataFrame({
        "player_id": ["A", "B", "C"],
        "player_name": ["Alpha", "Beta", "Gamma"],
        "observed_proportional_log_lift": [1.0, 2.0, 1.5],
        "observed_additional_pageviews": [300.0, 200.0, 200.0],
        "baseline_views": [300.0, 20.0, 100.0],
        "proportional_rank": [3, 1, 2],
        "additive_rank": [1, 2, 2],
    })
    pairs = build_pairwise_decomposition(frame)
    ab = pairs.query("player_i_id == 'A' and player_j_id == 'B'").iloc[0]
    assert ab["observed_order"] == "reversed"
    assert ab["identity_order"] == "reversed"
    assert ab["identity_matches_observed"]
    assert ab["dominance_margin"] > 0
    bc = pairs.query("player_i_id == 'B' and player_j_id == 'C'").iloc[0]
    assert bc["observed_order"] == "tied_additive"
    assert pd.isna(bc["dominance_margin"])


def synthetic_rank_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": ["A", "B", "C"],
        "player_name": ["Alpha", "Beta", "Gamma"],
        "observed_proportional_log_lift": [1.0, 2.0, 1.5],
        "observed_additional_pageviews": [300.0, 200.0, 200.0],
        "baseline_views": [300.0, 20.0, 100.0],
        "proportional_rank": [3, 1, 2],
        "additive_rank": [1, 2, 2],
    })


def test_build_athlete_displacements_uses_absolute_and_signed_rank_changes():
    from ijsms.rank_reversal import build_athlete_displacements

    result = build_athlete_displacements(synthetic_rank_frame()).set_index("player_id")
    assert result.loc["A", "rank_displacement"] == -2
    assert result.loc["A", "absolute_rank_displacement"] == 2


def test_summarize_decomposition_excludes_ties_from_reversal_rate():
    from ijsms.rank_reversal import (
        build_athlete_displacements,
        build_pairwise_decomposition,
        summarize_decomposition,
    )

    frame = synthetic_rank_frame()
    pairs = build_pairwise_decomposition(frame)
    summary = summarize_decomposition(pairs, build_athlete_displacements(frame))
    assert summary["total_pairs"] == 3
    assert summary["comparable_pairs"] == 2
    assert summary["tied_pairs"] == 1
    assert summary["observed_reversal_n"] == 2
    assert summary["observed_concordant_n"] == 0
    assert summary["observed_reversal_rate"] == 1.0


def synthetic_scoring_appearances() -> pd.DataFrame:
    return pd.DataFrame({
        "player_id": ["A", "A", "B", "B", "C", "C"],
        "match_id": ["M1", "M2", "M1", "M2", "M2", "M3"],
        "immediate_attention_log_lift": [1.0, 3.0, 2.0, 1.0, 1.5, 2.5],
        "winsorized_additional_pageviews": [100.0, 500.0, 300.0, 200.0, 150.0, 450.0],
        "baseline_views": [50.0, 50.0, 100.0, 100.0, 75.0, 75.0],
    })


def test_bootstrap_reversal_statistics_is_reproducible_and_bounded():
    from ijsms.rank_reversal import bootstrap_reversal_statistics

    scoring = synthetic_scoring_appearances()
    first = bootstrap_reversal_statistics(scoring, draws=20, seed=11, batch_size=5)
    second = bootstrap_reversal_statistics(scoring, draws=20, seed=11, batch_size=5)
    pd.testing.assert_frame_equal(first, second)
    assert first["observed_reversal_rate"].between(0, 1).all()
    assert first["identity_agreement_rate"].between(0, 1).all()
    assert (first["unique_scorers"] == 3).all()


def test_weighted_player_means_apply_the_same_match_weights_to_all_athletes():
    import numpy as np
    from ijsms.rank_reversal import _prepare_scoring_matrices, _weighted_player_means

    scoring = pd.DataFrame({
        "player_id": ["A", "A", "B", "B"],
        "match_id": ["M1", "M2", "M1", "M2"],
        "immediate_attention_log_lift": [1.0, 3.0, 2.0, 4.0],
        "winsorized_additional_pageviews": [100.0, 300.0, 200.0, 400.0],
        "baseline_views": [50.0, 50.0, 100.0, 100.0],
    })
    _, _, mask, prop, additive, _ = _prepare_scoring_matrices(scoring)
    weights = np.array([[0.8, 0.2], [0.2, 0.8]])
    prop_values, additive_values = _weighted_player_means(weights, mask, prop, additive)
    np.testing.assert_allclose(prop_values, [[1.4, 2.4], [2.6, 3.6]])
    np.testing.assert_allclose(additive_values, [[140.0, 240.0], [260.0, 360.0]])
