"""COMPOT heterogeneity summaries for realization-gap and relation-fit conflicts."""
from __future__ import annotations

import numpy as np
import pandas as pd

from stage2_pilot_metrics import build_project_realization_metrics


def _quartiles(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if numeric.isna().any():
        raise ValueError("COMPOT contains missing values")
    if len(numeric) < 4:
        raise ValueError("At least four projects are required for COMPOT quartiles")
    return pd.qcut(
        numeric.rank(method="first"), 4, labels=[1, 2, 3, 4]
    ).astype(int)


def _threshold_suffix(value: float) -> str:
    return f"{value:.2f}".replace(".", "_")


def summarize_compot_conflict(
    candidate_long: pd.DataFrame,
    thresholds: tuple[float, ...] = (0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Summarize realization-gap metrics by project-level COMPOT quartile."""
    required = {
        "work_id", "company_id", "selected", "compot",
        "cognitive_fit_cosine", "cognitive_fit_publication_count",
        "author_prior_partner",
    }
    missing = sorted(required - set(candidate_long.columns))
    if missing:
        raise ValueError(f"COMPOT conflict panel is missing required columns: {missing}")
    if not thresholds or any(float(value) < 0 for value in thresholds):
        raise ValueError("Conflict thresholds must be nonnegative")

    frame = candidate_long.copy()
    frame["compot"] = pd.to_numeric(frame["compot"], errors="raise").astype(float)
    if frame["compot"].isna().any():
        raise ValueError("COMPOT contains missing candidate rows")
    compot_nunique = frame.groupby("work_id", observed=True)["compot"].nunique(dropna=False)
    if compot_nunique.gt(1).any():
        raise ValueError("Within-project COMPOT values disagree")
    if "university_prior_partner" not in frame.columns:
        frame["university_prior_partner"] = 0
    if "strong_university_candidate" not in frame.columns:
        frame["strong_university_candidate"] = pd.to_numeric(
            frame["university_prior_partner"], errors="coerce"
        ).fillna(0).astype(int)
    if "natural_candidate" not in frame.columns:
        if "forced_selected_candidate" in frame.columns:
            frame["natural_candidate"] = 1 - pd.to_numeric(
                frame["forced_selected_candidate"], errors="raise"
            ).astype(int)
        else:
            frame["natural_candidate"] = 1
    if "forced_selected_candidate" not in frame.columns:
        frame["forced_selected_candidate"] = 0

    project_compot = (
        frame[["work_id", "compot"]]
        .drop_duplicates("work_id")
        .sort_values("work_id")
        .reset_index(drop=True)
    )
    project_compot["compot_quartile"] = _quartiles(project_compot["compot"])

    base = build_project_realization_metrics(frame, conflict_threshold=0.05)
    keep = [
        "work_id", "natural_project_recall", "selected_fit_covered_count",
        "actual_selected_count", "fit_shortfall", "selected_fit_percentile",
        "top_decile_fit_not_selected",
    ]
    project_metrics = base[keep].copy()
    for threshold in sorted(set(float(value) for value in thresholds)):
        threshold_metrics = build_project_realization_metrics(
            frame, conflict_threshold=threshold
        )[["work_id", "relation_fit_conflict"]]
        project_metrics = project_metrics.merge(
            threshold_metrics.rename(columns={
                "relation_fit_conflict": f"relation_fit_conflict_{_threshold_suffix(threshold)}"
            }),
            on="work_id",
            validate="one_to_one",
        )
    project_metrics = project_metrics.merge(
        project_compot, on="work_id", validate="one_to_one"
    )

    rows: list[dict[str, float | int]] = []
    for quartile, group in project_metrics.groupby(
        "compot_quartile", observed=True, sort=True
    ):
        selected_total = float(group["actual_selected_count"].sum())
        selected_covered = float(group["selected_fit_covered_count"].sum())
        row: dict[str, float | int] = {
            "compot_quartile": int(quartile),
            "projects": int(len(group)),
            "mean_compot": float(group["compot"].mean()),
            "natural_project_recall": float(group["natural_project_recall"].mean()),
            "selected_firm_cognitive_fit_coverage": (
                selected_covered / selected_total if selected_total else np.nan
            ),
            "mean_fit_shortfall": float(group["fit_shortfall"].mean()),
            "mean_selected_fit_percentile": float(group["selected_fit_percentile"].mean()),
            "no_selected_top_fit_decile_share": float(
                group["top_decile_fit_not_selected"].mean()
            ),
        }
        for threshold in sorted(set(float(value) for value in thresholds)):
            column = f"relation_fit_conflict_{_threshold_suffix(threshold)}"
            row[f"relation_fit_conflict_share_{_threshold_suffix(threshold)}"] = float(
                pd.to_numeric(group[column], errors="coerce").mean()
            )
        rows.append(row)
    return pd.DataFrame(rows)
