from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups


CONTINUOUS_COVARIATES = [
    "npxg",
    "non_penalty_shots",
    "max_npxg",
    "minutes_played",
    "baseline_log_views",
    "first_shot_score_diff",
    "shot_pressure_share",
    "open_play_share",
]
CATEGORICAL_COVARIATES = ["position_group", "knockout_stage"]
OUTCOMES = {
    "winsorized_additional_pageviews": ("0-1 days", "additive"),
    "immediate_attention_log_lift": ("0-1 days", "proportional"),
    "persistence_2_7_excess_views": ("2-7 days", "additive"),
    "persistence_2_7_log_lift": ("2-7 days", "proportional"),
    "persistence_8_30_excess_views": ("8-30 days", "additive"),
    "persistence_8_30_log_lift": ("8-30 days", "proportional"),
}
POSITION_GROUPS = {
    "GK": "GK",
    "CB": "DEF",
    "LB": "DEF",
    "RB": "DEF",
    "LWB": "DEF",
    "RWB": "DEF",
    "DM": "MID",
    "CM": "MID",
    "AM": "MID",
    "LM": "MID",
    "RM": "MID",
    "LW": "ATT",
    "RW": "ATT",
    "CF": "ATT",
    "ST": "ATT",
}


def prepare_near_miss_sample(
    panel: pd.DataFrame,
    *,
    treatment_column: str = "fjelstul_non_penalty_scorer",
    exclude_penalty_attempts: bool = True,
) -> pd.DataFrame:
    """Restrict the design to non-penalty shooters on common opportunity support."""

    sample = panel.loc[panel["non_penalty_shots"].gt(0)].copy()
    if exclude_penalty_attempts:
        sample = sample.loc[sample["penalty_attempts"].eq(0)].copy()
    sample["position_group"] = (
        sample["position_code"].map(POSITION_GROUPS).fillna("OTHER")
    )
    denominator = sample["non_penalty_shots"].clip(lower=1)
    sample["shot_pressure_share"] = sample["under_pressure_shots"] / denominator
    sample["open_play_share"] = sample["open_play_shots"] / denominator
    sample["treatment"] = sample[treatment_column].astype(int)
    return sample.reset_index(drop=True)


def _effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squared = float(np.square(weights).sum())
    return total * total / squared if squared > 0 else 0.0


def _balance_matrix(
    frame: pd.DataFrame,
    continuous_covariates: Iterable[str],
    categorical_covariates: Iterable[str],
) -> pd.DataFrame:
    continuous = frame[list(continuous_covariates)].astype(float).reset_index(drop=True)
    categorical = pd.get_dummies(
        frame[list(categorical_covariates)].astype(str),
        prefix=list(categorical_covariates),
        dtype=float,
    ).reset_index(drop=True)
    return pd.concat([continuous, categorical], axis=1)


def _smd_rows(
    matrix: pd.DataFrame,
    treatment: np.ndarray,
    weights: np.ndarray,
    suffix: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    treated = treatment == 1
    for column in matrix.columns:
        values = matrix[column].to_numpy(dtype=float)
        treated_mean = float(np.average(values[treated], weights=weights[treated]))
        control_mean = float(np.average(values[~treated], weights=weights[~treated]))
        treated_var = float(
            np.average(np.square(values[treated] - treated_mean), weights=weights[treated])
        )
        control_var = float(
            np.average(np.square(values[~treated] - control_mean), weights=weights[~treated])
        )
        denominator = np.sqrt((treated_var + control_var) / 2.0)
        smd = (treated_mean - control_mean) / denominator if denominator > 0 else 0.0
        rows.append(
            {
                "covariate": str(column),
                f"treated_mean_{suffix}": treated_mean,
                f"control_mean_{suffix}": control_mean,
                f"smd_{suffix}": float(smd),
            }
        )
    return pd.DataFrame(rows)


def fit_overlap_weights(
    sample: pd.DataFrame,
    *,
    continuous_covariates: Iterable[str] = CONTINUOUS_COVARIATES,
    categorical_covariates: Iterable[str] = CATEGORICAL_COVARIATES,
    regularization_c: float = 1.0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Estimate bounded overlap weights from a logistic propensity model."""

    continuous_covariates = list(continuous_covariates)
    categorical_covariates = list(categorical_covariates)
    frame = sample.copy()
    if frame["treatment"].nunique() != 2:
        raise ValueError("Overlap weighting requires treated and control observations")

    transformer = ColumnTransformer(
        [
            ("continuous", StandardScaler(), continuous_covariates),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_covariates,
            ),
        ]
    )
    model = Pipeline(
        [
            ("transform", transformer),
            (
                "logit",
                LogisticRegression(
                    C=regularization_c,
                    max_iter=5000,
                    solver="lbfgs",
                    random_state=20260730,
                ),
            ),
        ]
    )
    model.fit(frame, frame["treatment"])
    propensity = np.clip(model.predict_proba(frame)[:, 1], 1e-6, 1 - 1e-6)
    treatment = frame["treatment"].to_numpy(dtype=int)
    overlap_weight = np.where(treatment == 1, 1.0 - propensity, propensity)
    frame["propensity_score"] = propensity
    frame["overlap_weight"] = overlap_weight

    matrix = _balance_matrix(frame, continuous_covariates, categorical_covariates)
    raw = _smd_rows(matrix, treatment, np.ones(len(frame)), "unweighted")
    weighted = _smd_rows(matrix, treatment, overlap_weight, "weighted")
    balance = raw.merge(weighted, on="covariate", validate="one_to_one")

    treated = treatment == 1
    diagnostics: dict[str, Any] = {
        "observations": int(len(frame)),
        "treated_observations": int(treated.sum()),
        "control_observations": int((~treated).sum()),
        "treated_effective_sample_size": _effective_sample_size(
            overlap_weight[treated]
        ),
        "control_effective_sample_size": _effective_sample_size(
            overlap_weight[~treated]
        ),
        "max_absolute_smd_unweighted": float(balance["smd_unweighted"].abs().max()),
        "max_absolute_smd_weighted": float(balance["smd_weighted"].abs().max()),
        "minimum_propensity": float(propensity.min()),
        "maximum_propensity": float(propensity.max()),
        "regularization_c": float(regularization_c),
    }
    return frame, balance, diagnostics


def _cluster_codes(values: pd.Series) -> np.ndarray:
    return pd.factorize(values.astype(str), sort=True)[0].astype(np.int64)


def _holm_adjust(values: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=values.index, dtype=float)
    valid = values.dropna().sort_values()
    running = 0.0
    count = len(valid)
    for rank, (index, value) in enumerate(valid.items()):
        adjusted = min(1.0, (count - rank) * float(value))
        running = max(running, adjusted)
        result.loc[index] = running
    return result


def fit_near_miss_models(
    weighted_sample: pd.DataFrame,
    *,
    outcomes: dict[str, tuple[str, str]] = OUTCOMES,
) -> pd.DataFrame:
    """Fit overlap-weighted, match-FE models with two-way clustered inference."""

    data = weighted_sample.copy()
    formula_covariates = (
        "npxg + non_penalty_shots + max_npxg + minutes_played + "
        "first_shot_score_diff + shot_pressure_share + open_play_share + "
        "C(position_group) + C(knockout_stage) + C(match_id)"
    )
    rows: list[dict[str, Any]] = []
    player_groups = _cluster_codes(data["player_id"])
    match_groups = _cluster_codes(data["match_id"])

    for outcome, (window, scale) in outcomes.items():
        formula = (
            f"{outcome} ~ treatment * baseline_visibility_z + "
            f"{formula_covariates}"
        )
        result = smf.wls(
            formula,
            data=data,
            weights=data["overlap_weight"],
        ).fit()
        covariance, _, _ = cov_cluster_2groups(
            result,
            player_groups,
            match_groups,
        )
        names = result.model.exog_names
        for raw_term, label in [
            ("treatment", "treatment"),
            ("treatment:baseline_visibility_z", "treatment_x_baseline"),
        ]:
            index = names.index(raw_term)
            coefficient = float(result.params.iloc[index])
            standard_error = float(np.sqrt(max(float(covariance[index, index]), 0.0)))
            z_value = coefficient / standard_error if standard_error > 0 else np.nan
            p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            rows.append(
                {
                    "outcome": outcome,
                    "window": window,
                    "scale": scale,
                    "term": label,
                    "coefficient": coefficient,
                    "std_error": standard_error,
                    "p_value": p_value,
                    "ci_low": coefficient - 1.96 * standard_error,
                    "ci_high": coefficient + 1.96 * standard_error,
                    "observations": int(len(data)),
                    "treated_observations": int(data["treatment"].sum()),
                    "matches": int(data["match_id"].nunique()),
                    "players": int(data["player_id"].nunique()),
                }
            )

    results = pd.DataFrame(rows)
    results["p_holm_all_windows"] = np.nan
    for term in results["term"].unique():
        mask = results["term"].eq(term)
        results.loc[mask, "p_holm_all_windows"] = _holm_adjust(
            results.loc[mask, "p_value"]
        )

    results["p_holm_primary"] = np.nan
    immediate = results["window"].eq("0-1 days")
    for term in results["term"].unique():
        mask = immediate & results["term"].eq(term)
        results.loc[mask, "p_holm_primary"] = _holm_adjust(
            results.loc[mask, "p_value"]
        )
    return results
