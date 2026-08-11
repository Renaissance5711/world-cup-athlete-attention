"""COMPOT moderation extensions for the validated temporal realization ranking model."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from stage2_pilot_models import (
    _aggregate_metrics,
    _apply_scaling,
    _derive_relationship_inputs,
    _drop_redundant_columns,
    _fit_conditional_model,
    _fit_scaling,
    _project_metrics,
    _require,
)


def prepare_compot_features(
    candidate_long: pd.DataFrame,
    train_end_year: int = 2018,
) -> pd.DataFrame:
    """Scale project-level COMPOT using training projects only and map it to candidate rows."""
    _require(candidate_long, {"work_id", "publication_year", "compot"})
    data = candidate_long.copy()
    data["publication_year"] = pd.to_numeric(
        data["publication_year"], errors="raise"
    ).astype(int)
    data["compot"] = pd.to_numeric(data["compot"], errors="raise").astype(float)
    if data["compot"].isna().any():
        raise ValueError("COMPOT is missing for one or more candidate rows")
    consistency = (
        data.groupby("work_id", observed=True)[["publication_year", "compot"]]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if consistency.any():
        examples = consistency[consistency].index.astype(str).tolist()[:5]
        raise ValueError(f"Within-project COMPOT/year disagreement for work_ids={examples}")
    projects = data[["work_id", "publication_year", "compot"]].drop_duplicates("work_id")
    train_projects = projects[projects["publication_year"].le(train_end_year)]
    test_projects = projects[projects["publication_year"].gt(train_end_year)]
    if train_projects.empty or test_projects.empty:
        raise ValueError("Temporal COMPOT scaling requires training and post-split projects")
    mean = float(train_projects["compot"].mean())
    sd = float(train_projects["compot"].std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    project_z = projects.set_index("work_id")["compot"].sub(mean).div(sd)
    data["compot_z"] = data["work_id"].map(project_z).astype(float)
    return data


def _safe_odds_ratio(coefficient: float) -> float:
    return math.exp(float(np.clip(coefficient, -700.0, 700.0)))


def fit_compot_temporal_ranking_models(
    candidate_long: pd.DataFrame,
    train_end_year: int = 2018,
    *,
    bootstrap_reps: int = 500,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit B0/B1/B2 temporal ranking models with project-level COMPOT interactions."""
    required = {
        "work_id", "publication_year", "company_id", "selected", "compot",
        "cognitive_fit_cosine", "cognitive_fit_publication_count",
        "prior_subfield_publication_count", "author_prior_partner",
        "university_prior_partner", "strong_university_candidate",
    }
    _require(candidate_long, required)
    compot_prepared = prepare_compot_features(candidate_long, train_end_year=train_end_year)
    data = _derive_relationship_inputs(candidate_long)
    data["compot_z"] = compot_prepared["compot_z"].to_numpy(dtype=float)
    data["publication_year"] = pd.to_numeric(
        data["publication_year"], errors="raise"
    ).astype(int)
    data["selected"] = pd.to_numeric(data["selected"], errors="raise").astype(int)
    for column in [
        "author_prior_partner", "university_prior_partner", "strong_university_candidate"
    ]:
        data[column] = pd.to_numeric(data[column], errors="raise").fillna(0).astype(int)

    data["log_subfield_count"] = np.log1p(
        pd.to_numeric(data["prior_subfield_publication_count"], errors="coerce").clip(lower=0)
    )
    data["cognitive_evidence_volume"] = np.log1p(
        pd.to_numeric(data["cognitive_fit_publication_count"], errors="coerce").clip(lower=0)
    )
    data["log_author_relationship_strength"] = np.log1p(
        pd.to_numeric(data["author_relationship_strength"], errors="coerce").clip(lower=0)
    )

    train_mask = data["publication_year"].le(train_end_year)
    test_mask = data["publication_year"].gt(train_end_year)
    if not train_mask.any() or not test_mask.any():
        raise ValueError("Temporal split requires both training and post-split projects")

    scaling_map = {
        "cognitive_fit_cosine": "cognitive_fit_z",
        "log_subfield_count": "log_subfield_count_z",
        "cognitive_evidence_volume": "cognitive_evidence_volume_z",
        "log_author_relationship_strength": "log_author_relationship_strength_z",
        "author_recency_years": "author_recency_z",
    }
    train_source = data[train_mask]
    for source, target in scaling_map.items():
        scaling = _fit_scaling(train_source, source)
        _apply_scaling(data, source, target, scaling)

    data["fit_x_author"] = data["cognitive_fit_z"] * data["author_prior_partner"]
    data["fit_x_strong_university"] = (
        data["cognitive_fit_z"] * data["strong_university_candidate"]
    )
    data["author_x_compot"] = data["author_prior_partner"] * data["compot_z"]
    data["university_x_compot"] = data["university_prior_partner"] * data["compot_z"]
    data["strong_university_x_compot"] = (
        data["strong_university_candidate"] * data["compot_z"]
    )
    data["fit_x_compot"] = data["cognitive_fit_z"] * data["compot_z"]

    technical_terms = [
        "cognitive_fit_z", "log_subfield_count_z", "cognitive_evidence_volume_z"
    ]
    relational_terms = [
        "author_prior_partner", "university_prior_partner",
        "strong_university_candidate", "log_author_relationship_strength_z",
        "author_recency_z",
    ]
    baseline_terms = technical_terms + relational_terms + [
        "fit_x_author", "fit_x_strong_university"
    ]
    relationship_compot_terms = baseline_terms + [
        "author_x_compot", "university_x_compot", "strong_university_x_compot"
    ]
    fit_compot_terms = relationship_compot_terms + ["fit_x_compot"]
    model_terms = {
        "B0_combined": baseline_terms,
        "B1_relationship_compot": relationship_compot_terms,
        "B2_fit_compot": fit_compot_terms,
    }

    common_columns = sorted(set(fit_compot_terms))
    finite = np.isfinite(data[common_columns].astype(float)).all(axis=1)
    data = data[finite].copy()

    group_counts = data.groupby("work_id", observed=True)["selected"].agg(["sum", "count"])
    eligible_groups = group_counts.index[
        (group_counts["sum"] > 0) & (group_counts["sum"] < group_counts["count"])
    ]
    data = data[data["work_id"].isin(eligible_groups)].copy()
    train = data[data["publication_year"].le(train_end_year)].copy()
    test = data[data["publication_year"].gt(train_end_year)].copy()
    if train.empty or test.empty:
        raise ValueError("No eligible projects remain in the temporal train or test split")

    coefficient_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for model_name, requested_terms in model_terms.items():
        terms, dropped = _drop_redundant_columns(train, requested_terms)
        params, bse, method = _fit_conditional_model(train, terms)
        for term in requested_terms:
            if term in terms:
                coefficient = float(params[term])
                standard_error = float(bse[term]) if np.isfinite(bse[term]) else np.nan
                coefficient_rows.append({
                    "model": model_name,
                    "term": term,
                    "included": True,
                    "coefficient": coefficient,
                    "standard_error": standard_error,
                    "odds_ratio": _safe_odds_ratio(coefficient),
                    "fit_method": method,
                })
            else:
                coefficient_rows.append({
                    "model": model_name,
                    "term": term,
                    "included": False,
                    "coefficient": np.nan,
                    "standard_error": np.nan,
                    "odds_ratio": np.nan,
                    "fit_method": method,
                })

        score_column = f"_{model_name}_score"
        test[score_column] = (
            test[terms].astype(float).to_numpy() @ params[terms].to_numpy()
        )
        project_rows: list[dict[str, object]] = []
        for work_id, group in test.groupby("work_id", observed=True, sort=True):
            metrics = _project_metrics(group, score_column)
            project_rows.append({
                "model": model_name,
                "work_id": work_id,
                "publication_year": int(group["publication_year"].iloc[0]),
                "is_aggregate": False,
                **metrics,
            })
        project_frame = pd.DataFrame(project_rows)
        metric_rows.extend(project_rows)
        aggregate = _aggregate_metrics(project_frame, bootstrap_reps, seed)
        metric_rows.append({
            "model": model_name,
            "work_id": "__AGGREGATE__",
            "publication_year": np.nan,
            "is_aggregate": True,
            **aggregate,
        })

    return pd.DataFrame(coefficient_rows), pd.DataFrame(metric_rows)
