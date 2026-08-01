from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "player_id",
    "player_name",
    "observed_proportional_log_lift",
    "observed_additional_pageviews",
    "baseline_views",
    "proportional_rank",
    "additive_rank",
}


def validate_rank_input(frame: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if frame["player_id"].duplicated().any():
        raise ValueError("player_id must be unique")
    if frame[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("rank input contains missing values")
    if (frame["baseline_views"] <= 0).any():
        raise ValueError("baseline_views must be positive")
    if (frame["observed_additional_pageviews"] <= 0).any():
        raise ValueError("observed_additional_pageviews must be positive")


def build_pairwise_decomposition(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify every unordered athlete pair under observed and reconstructed scores."""
    validate_rank_input(frame)
    ordered = frame.sort_values("player_id", kind="stable").reset_index(drop=True)
    n = len(ordered)
    left, right = np.triu_indices(n, 1)

    player_id = ordered["player_id"].astype(str).to_numpy()
    player_name = ordered["player_name"].astype(str).to_numpy()
    prop = ordered["observed_proportional_log_lift"].to_numpy(dtype=float)
    additive = ordered["observed_additional_pageviews"].to_numpy(dtype=float)
    baseline = ordered["baseline_views"].to_numpy(dtype=float)
    reconstructed = np.log1p(additive / baseline)

    prop_diff = prop[left] - prop[right]
    additive_diff = additive[left] - additive[right]
    baseline_diff = baseline[left] - baseline[right]
    reconstructed_diff = reconstructed[left] - reconstructed[right]

    prop_sign = np.sign(prop_diff)
    additive_sign = np.sign(additive_diff)
    reconstructed_sign = np.sign(reconstructed_diff)

    observed_order = np.select(
        [additive_sign == 0, prop_sign == 0, additive_sign != prop_sign],
        ["tied_additive", "tied_proportional", "reversed"],
        default="concordant",
    )
    identity_order = np.select(
        [additive_sign == 0, reconstructed_sign == 0, additive_sign != reconstructed_sign],
        ["tied_additive", "tied_reconstructed", "reversed"],
        default="concordant",
    )

    non_tied_additive = additive_sign != 0
    larger_i = additive_diff > 0
    additive_large = np.where(larger_i, additive[left], additive[right])
    additive_small = np.where(larger_i, additive[right], additive[left])
    baseline_large = np.where(larger_i, baseline[left], baseline[right])
    baseline_small = np.where(larger_i, baseline[right], baseline[left])
    increment_log_gap = np.where(
        non_tied_additive, np.log(additive_large / additive_small), np.nan
    )
    baseline_log_gap = np.where(
        non_tied_additive, np.log(baseline_large / baseline_small), np.nan
    )
    dominance_margin = baseline_log_gap - increment_log_gap

    return pd.DataFrame(
        {
            "player_i_id": player_id[left],
            "player_j_id": player_id[right],
            "player_i_name": player_name[left],
            "player_j_name": player_name[right],
            "proportional_difference": prop_diff,
            "additive_difference": additive_diff,
            "baseline_difference": baseline_diff,
            "observed_order": observed_order,
            "reconstructed_proportional_i": reconstructed[left],
            "reconstructed_proportional_j": reconstructed[right],
            "identity_order": identity_order,
            "identity_matches_observed": identity_order == observed_order,
            "increment_log_gap": increment_log_gap,
            "baseline_log_gap": baseline_log_gap,
            "dominance_margin": dominance_margin,
        }
    )


def build_athlete_displacements(frame: pd.DataFrame) -> pd.DataFrame:
    """Return athlete-level signed and absolute displacement between rankings."""
    validate_rank_input(frame)
    result = frame.copy()
    result["rank_displacement"] = (
        result["additive_rank"].astype(int) - result["proportional_rank"].astype(int)
    )
    result["absolute_rank_displacement"] = result["rank_displacement"].abs()
    return result.sort_values(
        ["absolute_rank_displacement", "player_id"], ascending=[False, True], kind="stable"
    ).reset_index(drop=True)


def _safe_rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def summarize_decomposition(
    pairs: pd.DataFrame, athletes: pd.DataFrame
) -> dict[str, Any]:
    """Summarize pair classifications and athlete-level rank displacement."""
    required_pair_columns = {
        "observed_order",
        "identity_order",
        "identity_matches_observed",
        "dominance_margin",
    }
    missing = sorted(required_pair_columns.difference(pairs.columns))
    if missing:
        raise ValueError(f"pair decomposition missing columns: {missing}")
    if "absolute_rank_displacement" not in athletes.columns:
        raise ValueError("athlete displacement table is missing absolute_rank_displacement")

    observed_tied = pairs["observed_order"].str.startswith("tied")
    observed_comparable = ~observed_tied
    identity_tied = pairs["identity_order"].str.startswith("tied")
    identity_comparable = ~identity_tied
    jointly_comparable = observed_comparable & identity_comparable

    observed_reversed = pairs["observed_order"].eq("reversed")
    observed_concordant = pairs["observed_order"].eq("concordant")
    identity_reversed = pairs["identity_order"].eq("reversed")
    identity_concordant = pairs["identity_order"].eq("concordant")

    comparable_pairs = int(observed_comparable.sum())
    identity_comparable_pairs = int(identity_comparable.sum())
    agreement_n = int(pairs.loc[jointly_comparable, "identity_matches_observed"].sum())
    agreement_denominator = int(jointly_comparable.sum())

    reversal_margins = pairs.loc[identity_reversed, "dominance_margin"].dropna()
    abs_displacement = athletes["absolute_rank_displacement"].astype(float)

    return {
        "total_pairs": int(len(pairs)),
        "comparable_pairs": comparable_pairs,
        "tied_pairs": int(observed_tied.sum()),
        "observed_reversal_n": int(observed_reversed.sum()),
        "observed_concordant_n": int(observed_concordant.sum()),
        "observed_reversal_rate": _safe_rate(int(observed_reversed.sum()), comparable_pairs),
        "identity_comparable_pairs": identity_comparable_pairs,
        "identity_tied_pairs": int(identity_tied.sum()),
        "identity_reversal_n": int(identity_reversed.sum()),
        "identity_concordant_n": int(identity_concordant.sum()),
        "identity_reversal_rate": _safe_rate(
            int(identity_reversed.sum()), identity_comparable_pairs
        ),
        "identity_observed_agreement_n": agreement_n,
        "identity_observed_agreement_denominator": agreement_denominator,
        "identity_observed_agreement_rate": _safe_rate(agreement_n, agreement_denominator),
        "aggregation_residual_n": int(agreement_denominator - agreement_n),
        "aggregation_residual_rate": _safe_rate(
            agreement_denominator - agreement_n, agreement_denominator
        ),
        "identity_reversal_dominance_margin_median": (
            float(reversal_margins.median()) if len(reversal_margins) else float("nan")
        ),
        "identity_reversal_dominance_margin_q1": (
            float(reversal_margins.quantile(0.25)) if len(reversal_margins) else float("nan")
        ),
        "identity_reversal_dominance_margin_q3": (
            float(reversal_margins.quantile(0.75)) if len(reversal_margins) else float("nan")
        ),
        "median_absolute_rank_displacement": float(abs_displacement.median()),
        "mean_absolute_rank_displacement": float(abs_displacement.mean()),
        "maximum_absolute_rank_displacement": int(abs_displacement.max()),
        "athletes_moving_at_least_10_ranks": int((abs_displacement >= 10).sum()),
        "athletes_moving_at_least_25_ranks": int((abs_displacement >= 25).sum()),
        "athletes_moving_at_least_50_ranks": int((abs_displacement >= 50).sum()),
    }

SCORING_REQUIRED_COLUMNS = {
    "player_id",
    "match_id",
    "immediate_attention_log_lift",
    "winsorized_additional_pageviews",
    "baseline_views",
}


def _prepare_scoring_matrices(
    scoring: pd.DataFrame,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    missing = sorted(SCORING_REQUIRED_COLUMNS.difference(scoring.columns))
    if missing:
        raise ValueError(f"scoring input missing required columns: {missing}")
    required = list(SCORING_REQUIRED_COLUMNS)
    if scoring[required].isna().any().any():
        raise ValueError("scoring input contains missing values")
    if (scoring["baseline_views"] <= 0).any():
        raise ValueError("baseline_views must be positive")
    if (scoring["winsorized_additional_pageviews"] <= 0).any():
        raise ValueError("winsorized_additional_pageviews must be positive")
    if scoring.duplicated(["player_id", "match_id"]).any():
        raise ValueError("scoring input must contain one row per player-match")

    baseline_counts = scoring.groupby("player_id")["baseline_views"].nunique(dropna=False)
    if (baseline_counts > 1).any():
        raise ValueError("baseline_views must be constant within player_id")

    players = sorted(scoring["player_id"].astype(str).unique().tolist())
    matches = sorted(scoring["match_id"].astype(str).unique().tolist())
    player_index = {player: index for index, player in enumerate(players)}
    match_index = {match: index for index, match in enumerate(matches)}

    p_count = len(players)
    m_count = len(matches)
    mask = np.zeros((p_count, m_count), dtype=float)
    proportional = np.zeros((p_count, m_count), dtype=float)
    additive = np.zeros((p_count, m_count), dtype=float)
    baseline = np.empty(p_count, dtype=float)

    baseline_by_player = (
        scoring.assign(player_id=scoring["player_id"].astype(str))
        .groupby("player_id", sort=False)["baseline_views"]
        .first()
    )
    for player, index in player_index.items():
        baseline[index] = float(baseline_by_player.loc[player])

    for row in scoring.itertuples(index=False):
        player = str(row.player_id)
        match = str(row.match_id)
        p_idx = player_index[player]
        m_idx = match_index[match]
        mask[p_idx, m_idx] = 1.0
        proportional[p_idx, m_idx] = float(row.immediate_attention_log_lift)
        additive[p_idx, m_idx] = float(row.winsorized_additional_pageviews)

    return players, matches, mask, proportional, additive, baseline


def _weighted_player_means(
    weights: np.ndarray,
    mask: np.ndarray,
    proportional: np.ndarray,
    additive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    denominator = weights @ mask.T
    if (denominator <= 0).any():
        raise ValueError("every player must have positive bootstrap weight")
    proportional_values = (weights @ proportional.T) / denominator
    additive_values = (weights @ additive.T) / denominator
    return proportional_values, additive_values


def bootstrap_reversal_statistics(
    scoring: pd.DataFrame,
    draws: int,
    seed: int,
    batch_size: int = 250,
) -> pd.DataFrame:
    """Recompute reversal statistics under shared match-level Dirichlet weights."""
    if draws <= 0:
        raise ValueError("draws must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    players, matches, mask, proportional, additive, baseline = _prepare_scoring_matrices(
        scoring
    )
    p_count = len(players)
    total_pairs = p_count * (p_count - 1) // 2
    left, right = np.triu_indices(p_count, 1)

    rng = np.random.default_rng(seed)
    all_weights = rng.dirichlet(np.ones(len(matches)), size=draws)
    records: list[pd.DataFrame] = []

    for start in range(0, draws, batch_size):
        stop = min(start + batch_size, draws)
        weights = all_weights[start:stop]
        prop_values, add_values = _weighted_player_means(
            weights, mask, proportional, additive
        )
        reconstructed = np.log1p(add_values / baseline[None, :])

        prop_sign = np.sign(prop_values[:, left] - prop_values[:, right])
        add_sign = np.sign(add_values[:, left] - add_values[:, right])
        reconstructed_sign = np.sign(
            reconstructed[:, left] - reconstructed[:, right]
        )

        observed_comparable = (add_sign != 0) & (prop_sign != 0)
        observed_reversed = observed_comparable & (add_sign != prop_sign)
        identity_comparable = (add_sign != 0) & (reconstructed_sign != 0)
        identity_reversed = identity_comparable & (add_sign != reconstructed_sign)
        jointly_comparable = observed_comparable & identity_comparable
        identity_agreement = jointly_comparable & (
            observed_reversed == identity_reversed
        )

        comparable_n = observed_comparable.sum(axis=1).astype(int)
        tied_n = total_pairs - comparable_n
        observed_reversal_n = observed_reversed.sum(axis=1).astype(int)
        identity_comparable_n = identity_comparable.sum(axis=1).astype(int)
        identity_reversal_n = identity_reversed.sum(axis=1).astype(int)
        agreement_denominator = jointly_comparable.sum(axis=1).astype(int)
        identity_agreement_n = identity_agreement.sum(axis=1).astype(int)

        def divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
            return np.divide(
                numerator,
                denominator,
                out=np.full(numerator.shape, np.nan, dtype=float),
                where=denominator != 0,
            )

        records.append(
            pd.DataFrame(
                {
                    "draw": np.arange(start, stop, dtype=int),
                    "unique_scorers": p_count,
                    "total_pairs": total_pairs,
                    "comparable_pairs": comparable_n,
                    "tied_pairs": tied_n,
                    "observed_reversal_n": observed_reversal_n,
                    "observed_reversal_rate": divide(
                        observed_reversal_n, comparable_n
                    ),
                    "identity_reversal_n": identity_reversal_n,
                    "identity_reversal_rate": divide(
                        identity_reversal_n, identity_comparable_n
                    ),
                    "identity_agreement_n": identity_agreement_n,
                    "identity_agreement_rate": divide(
                        identity_agreement_n, agreement_denominator
                    ),
                }
            )
        )

    return pd.concat(records, ignore_index=True)
