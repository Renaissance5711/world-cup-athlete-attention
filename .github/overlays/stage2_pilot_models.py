"""Temporal ranking models for the TEM realization pilot."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from statsmodels.discrete.conditional_models import ConditionalLogit
from statsmodels.tools.sm_exceptions import HessianInversionWarning


@dataclass
class _Scaling:
    mean: float
    sd: float


def _require(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"Candidate panel is missing required columns: {missing}")


def _derive_relationship_inputs(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "author_relationship_strength" not in data.columns:
        if "author_prior_work_count" in data.columns:
            data["author_relationship_strength"] = data["author_prior_work_count"]
        elif "author_prior_partner" in data.columns:
            data["author_relationship_strength"] = data["author_prior_partner"]
        else:
            raise ValueError("Author relationship strength input is required")

    if "author_recency_years" not in data.columns:
        if {"author_last_evidence_date", "publication_date"}.issubset(data.columns):
            publication = pd.to_datetime(data["publication_date"], errors="coerce")
            evidence = pd.to_datetime(data["author_last_evidence_date"], errors="coerce")
            data["author_recency_years"] = (publication - evidence).dt.days / 365.25
            data.loc[data["author_prior_partner"].eq(0), "author_recency_years"] = 5.0
        else:
            raise ValueError(
                "Author recency input is required: provide author_recency_years "
                "or publication_date plus author_last_evidence_date"
            )

    data["author_recency_years"] = pd.to_numeric(
        data["author_recency_years"], errors="coerce"
    )
    data.loc[
        data["author_prior_partner"].eq(0)
        & data["author_recency_years"].isna(),
        "author_recency_years",
    ] = 5.0
    return data


def _fit_scaling(train: pd.DataFrame, source: str) -> _Scaling:
    values = pd.to_numeric(train[source], errors="coerce").astype(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        sd = 1.0
    return _Scaling(mean=mean, sd=sd)


def _apply_scaling(frame: pd.DataFrame, source: str, target: str, scaling: _Scaling) -> None:
    frame[target] = (pd.to_numeric(frame[source], errors="coerce") - scaling.mean) / scaling.sd


def _drop_redundant_columns(frame: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    kept: list[str] = []
    dropped: list[str] = []
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all() or np.nanstd(values) == 0:
            dropped.append(column)
            continue
        duplicate = False
        for existing in kept:
            other = pd.to_numeric(frame[existing], errors="coerce").to_numpy(dtype=float)
            if np.allclose(values, other, equal_nan=True) or np.allclose(values, -other, equal_nan=True):
                duplicate = True
                break
        if duplicate:
            dropped.append(column)
        else:
            kept.append(column)
    if not kept:
        raise ValueError("No nonredundant model terms remain")
    return kept, dropped


def _fit_conditional_model(train: pd.DataFrame, terms: list[str]):
    model = ConditionalLogit(
        train["selected"].astype(int),
        train[terms].astype(float),
        groups=train["work_id"],
    )
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", HessianInversionWarning)
            result = model.fit(method="bfgs", maxiter=500, disp=False)
        if any(isinstance(item.message, HessianInversionWarning) for item in caught):
            raise RuntimeError("Conditional-logit Hessian inversion failed")
        params = pd.Series(result.params, index=terms, dtype=float)
        bse_values = getattr(result, "bse", pd.Series(np.nan, index=terms))
        bse = pd.Series(np.asarray(bse_values, dtype=float), index=terms)
        if not np.isfinite(params).all():
            raise RuntimeError("Conditional-logit parameters are non-finite")
        method = "conditional_logit_bfgs"
    except Exception:
        result = model.fit_regularized(method="elastic_net", alpha=1e-6, L1_wt=0.0, refit=False)
        params = pd.Series(np.asarray(result.params, dtype=float), index=terms)
        bse = pd.Series(np.nan, index=terms)
        method = "conditional_logit_l2_fallback"
    return params, bse, method


def _project_metrics(group: pd.DataFrame, score_column: str) -> dict[str, float]:
    ranked = group.sort_values([score_column, "company_id"], ascending=[False, True]).reset_index(drop=True)
    ranked["rank"] = np.arange(1, len(ranked) + 1)
    selected_ranks = ranked.loc[ranked["selected"].eq(1), "rank"].astype(int).tolist()
    if not selected_ranks:
        raise ValueError(f"Project {group['work_id'].iloc[0]} has no selected firm")
    selected_ranks.sort()
    best_rank = selected_ranks[0]
    precision_terms = [
        index / rank for index, rank in enumerate(selected_ranks, start=1)
    ]
    selected_count = len(selected_ranks)
    return {
        "selected_best_rank": float(best_rank),
        "reciprocal_rank": 1.0 / best_rank,
        "recall_at_5": sum(rank <= 5 for rank in selected_ranks) / selected_count,
        "recall_at_10": sum(rank <= 10 for rank in selected_ranks) / selected_count,
        "average_precision": float(np.mean(precision_terms)),
    }


def _aggregate_metrics(project_rows: pd.DataFrame, bootstrap_reps: int, seed: int) -> dict[str, float]:
    metric_names = [
        "selected_best_rank", "reciprocal_rank", "recall_at_5",
        "recall_at_10", "average_precision",
    ]
    result: dict[str, float] = {}
    for metric in metric_names:
        result[metric] = float(project_rows[metric].mean())
        result[f"mean_{metric}"] = result[metric]
    if bootstrap_reps > 0:
        rng = np.random.default_rng(seed)
        values = project_rows[metric_names].to_numpy(dtype=float)
        boot = np.empty((bootstrap_reps, len(metric_names)), dtype=float)
        for rep in range(bootstrap_reps):
            indices = rng.integers(0, len(values), size=len(values))
            boot[rep] = values[indices].mean(axis=0)
        for index, metric in enumerate(metric_names):
            result[f"{metric}_ci_low"] = float(np.quantile(boot[:, index], 0.025))
            result[f"{metric}_ci_high"] = float(np.quantile(boot[:, index], 0.975))
    return result


def fit_temporal_ranking_models(
    candidate_long: pd.DataFrame,
    train_end_year: int = 2018,
    *,
    bootstrap_reps: int = 500,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit technical, relational, and combined models and rank later projects."""
    required = {
        "work_id", "publication_year", "company_id", "selected",
        "cognitive_fit_cosine", "cognitive_fit_publication_count",
        "prior_subfield_publication_count", "author_prior_partner",
        "university_prior_partner", "strong_university_candidate",
    }
    _require(candidate_long, required)
    data = _derive_relationship_inputs(candidate_long)
    data["publication_year"] = pd.to_numeric(data["publication_year"], errors="raise").astype(int)
    data["selected"] = pd.to_numeric(data["selected"], errors="raise").astype(int)
    for column in ["author_prior_partner", "university_prior_partner", "strong_university_candidate"]:
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
    data["fit_x_strong_university"] = data["cognitive_fit_z"] * data["strong_university_candidate"]

    technical_terms = [
        "cognitive_fit_z", "log_subfield_count_z", "cognitive_evidence_volume_z"
    ]
    relational_terms = [
        "author_prior_partner", "university_prior_partner",
        "strong_university_candidate", "log_author_relationship_strength_z",
        "author_recency_z",
    ]
    combined_terms = technical_terms + relational_terms + [
        "fit_x_author", "fit_x_strong_university"
    ]
    model_terms = {
        "technical": technical_terms,
        "relational": relational_terms,
        "combined": combined_terms,
    }

    common_columns = sorted(set(combined_terms))
    finite = np.isfinite(data[common_columns].astype(float)).all(axis=1)
    data = data[finite].copy()

    group_counts = data.groupby("work_id", observed=True)["selected"].agg(["sum", "count"])
    eligible_groups = group_counts.index[(group_counts["sum"] > 0) & (group_counts["sum"] < group_counts["count"])]
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
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "term": term,
                        "included": True,
                        "coefficient": coefficient,
                        "standard_error": standard_error,
                        "odds_ratio": math.exp(coefficient),
                        "fit_method": method,
                    }
                )
            else:
                coefficient_rows.append(
                    {
                        "model": model_name,
                        "term": term,
                        "included": False,
                        "coefficient": np.nan,
                        "standard_error": np.nan,
                        "odds_ratio": np.nan,
                        "fit_method": method,
                    }
                )

        score_column = f"_{model_name}_score"
        test[score_column] = test[terms].astype(float).to_numpy() @ params[terms].to_numpy()
        project_rows: list[dict[str, object]] = []
        for work_id, group in test.groupby("work_id", observed=True, sort=True):
            metrics = _project_metrics(group, score_column)
            project_rows.append(
                {
                    "model": model_name,
                    "work_id": work_id,
                    "publication_year": int(group["publication_year"].iloc[0]),
                    "is_aggregate": False,
                    **metrics,
                }
            )
        project_frame = pd.DataFrame(project_rows)
        metric_rows.extend(project_rows)
        aggregate = _aggregate_metrics(project_frame, bootstrap_reps, seed)
        metric_rows.append(
            {
                "model": model_name,
                "work_id": "__AGGREGATE__",
                "publication_year": np.nan,
                "is_aggregate": True,
                **aggregate,
            }
        )

    return pd.DataFrame(coefficient_rows), pd.DataFrame(metric_rows)
