"""Realization-gap metrics and machine-readable pilot gate decisions."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _measurable_fit(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["cognitive_fit_publication_count"], errors="coerce").gt(0)
        & pd.to_numeric(frame["cognitive_fit_cosine"], errors="coerce").notna()
    )


def build_project_realization_metrics(
    candidate_long: pd.DataFrame,
    conflict_threshold: float = 0.05,
) -> pd.DataFrame:
    """Summarize natural recall, cognitive coverage, and relation–fit conflicts."""
    required = {
        "work_id", "company_id", "selected", "natural_candidate",
        "forced_selected_candidate", "author_prior_partner",
        "strong_university_candidate", "cognitive_fit_cosine",
        "cognitive_fit_publication_count",
    }
    _require_columns(candidate_long, required, "candidate_long")
    if conflict_threshold < 0:
        raise ValueError("conflict_threshold must be nonnegative")
    if candidate_long.groupby(["work_id", "company_id"]).size().gt(1).any():
        raise ValueError("candidate_long contains duplicate project-company rows")

    thresholds = sorted(set([0.02, float(conflict_threshold), 0.10]))
    rows: list[dict[str, Any]] = []

    for work_id, group in candidate_long.groupby("work_id", observed=True, sort=True):
        group = group.copy()
        group["selected"] = pd.to_numeric(group["selected"], errors="raise").astype(int)
        group["natural_candidate"] = pd.to_numeric(group["natural_candidate"], errors="raise").astype(int)
        group["author_prior_partner"] = pd.to_numeric(group["author_prior_partner"], errors="raise").fillna(0).astype(int)
        group["strong_university_candidate"] = pd.to_numeric(group["strong_university_candidate"], errors="raise").fillna(0).astype(int)
        measurable = _measurable_fit(group)
        group["fit_measurable"] = measurable
        group["fit"] = pd.to_numeric(group["cognitive_fit_cosine"], errors="coerce")

        selected = group["selected"].eq(1)
        natural = group["natural_candidate"].eq(1)
        natural_selected_count = int((selected & natural).sum())
        actual_selected_count = int(selected.sum())
        selected_fit_covered_count = int((selected & measurable).sum())

        measurable_group = group[measurable].copy()
        if measurable_group.empty:
            max_natural_fit = np.nan
            mean_selected_fit = np.nan
            fit_shortfall = np.nan
            selected_fit_percentile = np.nan
            top_decile_fit_not_selected = np.nan
            conflicts = {threshold: np.nan for threshold in thresholds}
        else:
            measurable_group["fit_percentile"] = measurable_group["fit"].rank(
                method="average", pct=True, ascending=True
            )
            natural_fit = measurable_group[measurable_group["natural_candidate"].eq(1)]["fit"]
            selected_fit = measurable_group[measurable_group["selected"].eq(1)]["fit"]
            max_natural_fit = float(natural_fit.max()) if not natural_fit.empty else np.nan
            mean_selected_fit = float(selected_fit.mean()) if not selected_fit.empty else np.nan
            fit_shortfall = (
                max_natural_fit - mean_selected_fit
                if np.isfinite(max_natural_fit) and np.isfinite(mean_selected_fit)
                else np.nan
            )
            selected_percentiles = measurable_group.loc[
                measurable_group["selected"].eq(1), "fit_percentile"
            ]
            selected_fit_percentile = (
                float(selected_percentiles.mean()) if not selected_percentiles.empty else np.nan
            )
            top_decile = measurable_group["fit_percentile"].ge(0.90)
            top_decile_fit_not_selected = (
                int(not (top_decile & measurable_group["selected"].eq(1)).any())
                if top_decile.any()
                else np.nan
            )

            embedded_selected = measurable_group[
                measurable_group["selected"].eq(1)
                & (
                    measurable_group["author_prior_partner"].eq(1)
                    | measurable_group["strong_university_candidate"].eq(1)
                )
            ]
            unembedded_alternatives = measurable_group[
                measurable_group["selected"].eq(0)
                & measurable_group["natural_candidate"].eq(1)
                & measurable_group["author_prior_partner"].eq(0)
                & measurable_group["strong_university_candidate"].eq(0)
            ]
            if embedded_selected.empty or unembedded_alternatives.empty:
                conflicts = {threshold: 0 for threshold in thresholds}
            else:
                selected_reference = float(embedded_selected["fit"].max())
                outside_best = float(unembedded_alternatives["fit"].max())
                conflicts = {
                    threshold: int(outside_best - selected_reference >= threshold - 1e-12)
                    for threshold in thresholds
                }

        row: dict[str, Any] = {
            "work_id": work_id,
            "natural_selected_count": natural_selected_count,
            "actual_selected_count": actual_selected_count,
            "natural_project_recall": int(natural_selected_count > 0),
            "selected_fit_covered_count": selected_fit_covered_count,
            "selected_fit_coverage": (
                selected_fit_covered_count / actual_selected_count
                if actual_selected_count > 0
                else np.nan
            ),
            "measurable_candidate_count": int(measurable.sum()),
            "max_natural_candidate_fit": max_natural_fit,
            "mean_selected_fit": mean_selected_fit,
            "fit_shortfall": fit_shortfall,
            "selected_fit_percentile": selected_fit_percentile,
            "top_decile_fit_not_selected": top_decile_fit_not_selected,
        }
        for threshold, value in conflicts.items():
            suffix = str(threshold).replace("0.", "0_")
            row[f"relation_fit_conflict_{suffix}"] = value
        main_suffix = str(float(conflict_threshold)).replace("0.", "0_")
        row["relation_fit_conflict"] = row[f"relation_fit_conflict_{main_suffix}"]
        rows.append(row)

    return pd.DataFrame(rows)


def _aggregate_ranking_row(ranking_comparison: pd.DataFrame, model: str) -> pd.Series:
    subset = ranking_comparison[ranking_comparison["model"].astype(str).eq(model)]
    if subset.empty:
        raise ValueError(f"Ranking comparison has no {model!r} model row")
    if "is_aggregate" in subset.columns and subset["is_aggregate"].astype(bool).any():
        subset = subset[subset["is_aggregate"].astype(bool)]
    if len(subset) == 1:
        return subset.iloc[0]
    numeric = subset.select_dtypes(include=[np.number]).mean()
    combined = subset.iloc[0].copy()
    for column, value in numeric.items():
        combined[column] = value
    return combined


def _metric(row: pd.Series, names: list[str], default: float = np.nan) -> float:
    for name in names:
        if name in row.index and pd.notna(row[name]):
            return float(row[name])
    return float(default)


def evaluate_pilot_gates(
    candidate_long: pd.DataFrame,
    ranking_comparison: pd.DataFrame,
    project_metrics: pd.DataFrame,
) -> dict[str, object]:
    """Evaluate the four prespecified empirical viability gates."""
    _require_columns(
        candidate_long,
        {"work_id", "selected", "natural_candidate", "cognitive_fit_publication_count"},
        "candidate_long",
    )
    _require_columns(project_metrics, {"work_id", "relation_fit_conflict"}, "project_metrics")
    _require_columns(ranking_comparison, {"model"}, "ranking_comparison")

    selected = pd.to_numeric(candidate_long["selected"], errors="raise").eq(1)
    natural = pd.to_numeric(candidate_long["natural_candidate"], errors="raise").eq(1)
    selected_total = int(selected.sum())
    project_natural = (
        candidate_long.assign(_natural_selected=(selected & natural).astype(int))
        .groupby("work_id", observed=True)["_natural_selected"]
        .max()
    )
    natural_project_recall = float(project_natural.mean()) if len(project_natural) else np.nan
    natural_firm_instance_recall = (
        float((selected & natural).sum() / selected_total) if selected_total else np.nan
    )
    fit_covered = pd.to_numeric(
        candidate_long["cognitive_fit_publication_count"], errors="coerce"
    ).gt(0)
    if "cognitive_fit_cosine" in candidate_long.columns:
        fit_covered &= pd.to_numeric(
            candidate_long["cognitive_fit_cosine"], errors="coerce"
        ).notna()
    if "project_text_missing" in candidate_long.columns:
        fit_covered &= pd.to_numeric(
            candidate_long["project_text_missing"], errors="coerce"
        ).fillna(1).eq(0)
    selected_firm_cognitive_coverage = (
        float((selected & fit_covered).sum() / selected_total) if selected_total else np.nan
    )
    all_candidate_cognitive_coverage = float(fit_covered.mean()) if len(candidate_long) else np.nan
    conflict_values = pd.to_numeric(project_metrics["relation_fit_conflict"], errors="coerce").dropna()
    conflict_share = float(conflict_values.mean()) if len(conflict_values) else np.nan

    technical = _aggregate_ranking_row(ranking_comparison, "technical")
    combined = _aggregate_ranking_row(ranking_comparison, "combined")
    technical_mrr = _metric(technical, ["mean_reciprocal_rank", "reciprocal_rank"])
    combined_mrr = _metric(combined, ["mean_reciprocal_rank", "reciprocal_rank"])
    technical_recall10 = _metric(technical, ["mean_recall_at_10", "recall_at_10"])
    combined_recall10 = _metric(combined, ["mean_recall_at_10", "recall_at_10"])
    mrr_improvement = combined_mrr - technical_mrr
    recall10_improvement = combined_recall10 - technical_recall10

    gates = {
        "natural_recall_80": bool(np.isfinite(natural_project_recall) and natural_project_recall >= 0.80),
        "selected_fit_coverage_80": bool(
            np.isfinite(selected_firm_cognitive_coverage)
            and selected_firm_cognitive_coverage >= 0.80
        ),
        "conflict_share_10": bool(np.isfinite(conflict_share) and conflict_share >= 0.10),
        "combined_rank_improves": bool(
            (np.isfinite(mrr_improvement) and mrr_improvement >= 0.02)
            or (np.isfinite(recall10_improvement) and recall10_improvement >= 0.05)
        ),
    }
    passed = int(sum(gates.values()))
    if passed >= 3:
        recommendation = "GO_FULL_STAGE2"
    elif passed == 2:
        recommendation = "REVISE_AND_RERUN_PILOT"
    else:
        recommendation = "STOP_MAIN_TOPIC"

    return {
        "natural_project_recall": natural_project_recall,
        "natural_firm_instance_recall": natural_firm_instance_recall,
        "selected_firm_cognitive_coverage": selected_firm_cognitive_coverage,
        "all_candidate_cognitive_coverage": all_candidate_cognitive_coverage,
        "conflict_project_share_0_05": conflict_share,
        "technical_mrr": technical_mrr,
        "combined_mrr": combined_mrr,
        "combined_mrr_improvement": mrr_improvement,
        "technical_recall_at_10": technical_recall10,
        "combined_recall_at_10": combined_recall10,
        "combined_recall_at_10_improvement": recall10_improvement,
        "gates": gates,
        "passed_gate_count": passed,
        "recommendation": recommendation,
    }
