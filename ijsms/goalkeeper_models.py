from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import fisher_exact, norm

CATEGORY_ORDER = ["heroic_save", "routine_save", "understandable_goal", "unexpected_goal"]


def _model_components(frame: pd.DataFrame, outcome: str):
    data = frame.dropna(subset=[outcome, "focal_category", "match_id"]).copy()
    categories = pd.Categorical(data["focal_category"], categories=CATEGORY_ORDER)
    dummies = pd.get_dummies(categories, drop_first=True, dtype=float)
    dummies.columns = CATEGORY_ORDER[1:]
    controls = pd.DataFrame(index=data.index)
    for column in [
        "baseline_log_visibility_difference",
        "team_win_difference",
        "on_target_shots",
        "knockout_stage",
    ]:
        source = data[column] if column in data else 0.0
        controls[column] = pd.to_numeric(source, errors="coerce").fillna(0.0)
    design = pd.concat([dummies.set_axis(data.index), controls], axis=1)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(pd.to_numeric(data[outcome]), design).fit(
        cov_type="cluster", cov_kwds={"groups": data["match_id"]}
    )
    return data, design, model


def _fit(frame: pd.DataFrame, outcome: str, scale: str) -> pd.DataFrame:
    data, design, model = _model_components(frame, outcome)
    rows = []
    for term in design.columns:
        rows.append(
            {
                "scale": scale,
                "outcome": outcome,
                "term": term,
                "estimate": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(model.conf_int().loc[term, 0]),
                "ci_high": float(model.conf_int().loc[term, 1]),
                "n": int(model.nobs),
                "clusters": int(data["match_id"].nunique()),
                "cluster": "match_id",
                "reference_category": "heroic_save",
            }
        )
    return pd.DataFrame(rows)


def fit_bilateral_attention_models(pairs: pd.DataFrame) -> pd.DataFrame:
    proportional = _fit(pairs, "attention_log_lift_difference", "proportional")
    additive = _fit(pairs, "additional_pageview_difference", "additive")
    return pd.concat([proportional, additive], ignore_index=True)


def fit_bilateral_category_contrasts(pairs: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("proportional", "attention_log_lift_difference"),
        ("additive", "additional_pageview_difference"),
    ]
    contrasts = [
        ("routine_save - heroic_save", {"routine_save": 1.0}),
        ("understandable_goal - heroic_save", {"understandable_goal": 1.0}),
        ("unexpected_goal - heroic_save", {"unexpected_goal": 1.0}),
        (
            "unexpected_goal - understandable_goal",
            {"unexpected_goal": 1.0, "understandable_goal": -1.0},
        ),
    ]
    rows: list[dict] = []
    for scale, outcome in specs:
        data, design, model = _model_components(pairs, outcome)
        params = model.params
        covariance = model.cov_params()
        for name, weights in contrasts:
            vector = pd.Series(0.0, index=params.index)
            for term, weight in weights.items():
                vector[term] = weight
            estimate = float(vector @ params)
            variance = float(vector @ covariance @ vector)
            std_error = float(np.sqrt(max(variance, 0.0)))
            z_value = estimate / std_error if std_error > 0 else np.nan
            p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
            rows.append(
                {
                    "scale": scale,
                    "outcome": outcome,
                    "contrast": name,
                    "estimate": estimate,
                    "std_error": std_error,
                    "p_value": p_value,
                    "ci_low": estimate - 1.96 * std_error,
                    "ci_high": estimate + 1.96 * std_error,
                    "n": int(model.nobs),
                    "clusters": int(data["match_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def summarize_category_means(pairs: pd.DataFrame) -> pd.DataFrame:
    outcomes = ["attention_log_lift_difference", "additional_pageview_difference"]
    rows = []
    for category in CATEGORY_ORDER:
        group = pairs.loc[pairs["focal_category"].eq(category)]
        row = {
            "focal_category": category,
            "pairs": int(len(group)),
            "matches": int(group["match_id"].nunique()),
        }
        for outcome in outcomes:
            values = pd.to_numeric(group[outcome], errors="coerce").dropna()
            row[f"mean_{outcome}"] = float(values.mean()) if len(values) else np.nan
            row[f"median_{outcome}"] = float(values.median()) if len(values) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)



def fit_continuous_responsibility_models(pairs: pd.DataFrame) -> pd.DataFrame:
    """Estimate how performance surprise reallocates attention between roles."""

    rows: list[dict] = []
    for scale, outcome in [
        ("proportional", "attention_log_lift_difference"),
        ("additive", "additional_pageview_difference"),
    ]:
        required = [outcome, "net_shooter_surprise", "match_id"]
        data = pairs.dropna(subset=required).copy()
        design = pd.DataFrame(index=data.index)
        for column in [
            "net_shooter_surprise",
            "mean_expected_save_probability",
            "on_target_shots",
            "baseline_log_visibility_difference",
            "team_win_difference",
            "knockout_stage",
        ]:
            source = data[column] if column in data else 0.0
            design[column] = pd.to_numeric(source, errors="coerce").fillna(0.0)
        design = sm.add_constant(design, has_constant="add")
        model = sm.OLS(pd.to_numeric(data[outcome]), design).fit(
            cov_type="cluster", cov_kwds={"groups": data["match_id"]}
        )
        term = "net_shooter_surprise"
        rows.append(
            {
                "scale": scale,
                "outcome": outcome,
                "term": term,
                "estimate": float(model.params[term]),
                "std_error": float(model.bse[term]),
                "p_value": float(model.pvalues[term]),
                "ci_low": float(model.conf_int().loc[term, 0]),
                "ci_high": float(model.conf_int().loc[term, 1]),
                "n": int(model.nobs),
                "clusters": int(data["match_id"].nunique()),
                "cluster": "match_id",
            }
        )
    return pd.DataFrame(rows)

def fit_news_framing_validation(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact validation tests for sparse directional headline framing.

    The analysis is restricted to goalkeeper-match observations with at least
    one retained headline when headline counts are available. It tests whether
    notable saves are associated with any praise framing and whether unexpected
    goals are associated with any blame framing.
    """

    data = frame.copy()
    if "headline_count" in data:
        data = data.loc[pd.to_numeric(data["headline_count"], errors="coerce").fillna(0).gt(0)]
    specs = [
        ("praise_any", "notable_save_any"),
        ("blame_any", "unexpected_goal_any"),
    ]
    rows: list[dict] = []
    for outcome, exposure in specs:
        y = pd.to_numeric(data[outcome], errors="coerce").fillna(0).gt(0).astype(int)
        x = pd.to_numeric(data[exposure], errors="coerce").fillna(0).gt(0).astype(int)
        table = pd.crosstab(x, y).reindex(index=[0, 1], columns=[0, 1], fill_value=0)
        odds_ratio, p_value = fisher_exact(table.to_numpy())
        exposed = y.loc[x.eq(1)]
        unexposed = y.loc[x.eq(0)]
        exposed_rate = float(exposed.mean()) if len(exposed) else np.nan
        unexposed_rate = float(unexposed.mean()) if len(unexposed) else np.nan
        rows.append(
            {
                "outcome": outcome,
                "exposure": exposure,
                "covered_goalkeeper_matches": int(len(data)),
                "exposed_n": int(len(exposed)),
                "unexposed_n": int(len(unexposed)),
                "exposed_rate": exposed_rate,
                "unexposed_rate": unexposed_rate,
                "rate_difference": exposed_rate - unexposed_rate,
                "odds_ratio": float(odds_ratio),
                "fisher_p_value": float(p_value),
                "table_00": int(table.loc[0, 0]),
                "table_01": int(table.loc[0, 1]),
                "table_10": int(table.loc[1, 0]),
                "table_11": int(table.loc[1, 1]),
            }
        )
    return pd.DataFrame(rows)


def _linear_combination_row(
    *,
    model,
    vector: pd.Series,
    scale: str,
    outcome: str,
    term: str,
    data: pd.DataFrame,
    sample: str,
    centered_expected_save_at: float,
    shot_count_control: int,
) -> dict:
    estimate = float(vector @ model.params)
    variance = float(vector @ model.cov_params() @ vector)
    std_error = float(np.sqrt(max(variance, 0.0)))
    z_value = estimate / std_error if std_error > 0 else np.nan
    p_value = float(2 * norm.sf(abs(z_value))) if np.isfinite(z_value) else np.nan
    return {
        "scale": scale,
        "outcome": outcome,
        "term": term,
        "estimate": estimate,
        "std_error": std_error,
        "p_value": p_value,
        "ci_low": estimate - 1.96 * std_error,
        "ci_high": estimate + 1.96 * std_error,
        "n": int(model.nobs),
        "clusters": int(data["match_id"].nunique()),
        "cluster": "match_id",
        "sample": sample,
        "match_fixed_effects": 1,
        "baseline_visibility_control": 1,
        "expected_save_center": float(centered_expected_save_at),
        "shot_count_control": int(shot_count_control),
    }


def _fit_outcome_gradient_spec(
    frame: pd.DataFrame,
    *,
    exposure_column: str,
    term_names: tuple[str, str, str],
    sample: str,
    extra_controls: tuple[str, ...] = (),
) -> pd.DataFrame:
    rows: list[dict] = []
    for scale, outcome in [
        ("proportional", "attention_log_lift_difference"),
        ("additive", "additional_pageview_difference"),
    ]:
        required = [
            outcome,
            exposure_column,
            "mean_expected_save_probability",
            "baseline_log_visibility_difference",
            "match_id",
            *extra_controls,
        ]
        data = frame.dropna(subset=required).copy()
        expected_save_center = float(data["mean_expected_save_probability"].mean())
        data["expected_save_centered"] = (
            pd.to_numeric(data["mean_expected_save_probability"], errors="coerce")
            - expected_save_center
        )
        data["outcome_x_expected_save"] = (
            pd.to_numeric(data[exposure_column], errors="coerce")
            * data["expected_save_centered"]
        )
        design = pd.DataFrame(
            {
                exposure_column: pd.to_numeric(data[exposure_column], errors="coerce"),
                "expected_save_centered": data["expected_save_centered"],
                "outcome_x_expected_save": data["outcome_x_expected_save"],
                "baseline_log_visibility_difference": pd.to_numeric(
                    data["baseline_log_visibility_difference"], errors="coerce"
                ),
            },
            index=data.index,
        )
        for control in extra_controls:
            design[control] = pd.to_numeric(data[control], errors="coerce")
        match_dummies = pd.get_dummies(
            data["match_id"].astype(str), prefix="match", drop_first=True, dtype=float
        ).set_axis(data.index)
        design = pd.concat([design, match_dummies], axis=1)
        design = sm.add_constant(design, has_constant="add")
        model = sm.OLS(pd.to_numeric(data[outcome]), design).fit(
            cov_type="cluster", cov_kwds={"groups": data["match_id"]}
        )

        direct = pd.Series(0.0, index=model.params.index)
        direct[exposure_column] = 1.0
        rows.append(
            _linear_combination_row(
                model=model,
                vector=direct,
                scale=scale,
                outcome=outcome,
                term=term_names[0],
                data=data,
                sample=sample,
                centered_expected_save_at=expected_save_center,
                shot_count_control=int("on_target_shots" in extra_controls),
            )
        )

        reference_gradient = pd.Series(0.0, index=model.params.index)
        reference_gradient["expected_save_centered"] = 1.0
        rows.append(
            _linear_combination_row(
                model=model,
                vector=reference_gradient,
                scale=scale,
                outcome=outcome,
                term=term_names[1],
                data=data,
                sample=sample,
                centered_expected_save_at=expected_save_center,
                shot_count_control=int("on_target_shots" in extra_controls),
            )
        )

        exposed_gradient = reference_gradient.copy()
        exposed_gradient["outcome_x_expected_save"] = 1.0
        rows.append(
            _linear_combination_row(
                model=model,
                vector=exposed_gradient,
                scale=scale,
                outcome=outcome,
                term=term_names[2],
                data=data,
                sample=sample,
                centered_expected_save_at=expected_save_center,
                shot_count_control=int("on_target_shots" in extra_controls),
            )
        )
    return pd.DataFrame(rows)


def fit_single_shot_outcome_gradient_models(pairs: pd.DataFrame) -> pd.DataFrame:
    """Separate the realised goal-versus-save association from surprise gradients.

    The primary sample contains shooter-goalkeeper-match pairs with exactly one
    on-target shot. Save is the reference outcome. Expected-save probability is
    centered at the primary-sample mean, and match fixed effects absorb common
    match-level attention shocks.
    """

    data = pairs.loc[pd.to_numeric(pairs["on_target_shots"], errors="coerce").eq(1)].copy()
    data["realised_goal"] = pd.to_numeric(data["goals"], errors="coerce").fillna(0).astype(float)
    return _fit_outcome_gradient_spec(
        data,
        exposure_column="realised_goal",
        term_names=(
            "goal_vs_save_at_mean_expected_save",
            "expected_save_gradient_among_saves",
            "expected_save_gradient_among_goals",
        ),
        sample="single_shot_pairs_primary",
    )


def fit_all_pair_outcome_balance_sensitivity(pairs: pd.DataFrame) -> pd.DataFrame:
    """Sensitivity model using the within-pair share of on-target shots scored."""

    data = pairs.copy()
    shots = pd.to_numeric(data["on_target_shots"], errors="coerce")
    data["goal_share"] = pd.to_numeric(data["goals"], errors="coerce") / shots
    return _fit_outcome_gradient_spec(
        data,
        exposure_column="goal_share",
        term_names=(
            "goal_share_at_mean_expected_save",
            "expected_save_gradient_at_zero_goal_share",
            "expected_save_gradient_at_full_goal_share",
        ),
        sample="all_pairs_goal_share_sensitivity",
        extra_controls=("on_target_shots",),
    )
