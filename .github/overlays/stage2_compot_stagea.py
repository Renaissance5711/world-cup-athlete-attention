"""Project-level COMPOT analysis for firm-participation emergence."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm


def _field_column(frame: pd.DataFrame) -> str:
    for column in ("primary_field_id", "primary_subfield_id"):
        if column in frame.columns:
            return column
    raise ValueError("Stage A requires primary_field_id or primary_subfield_id")


def _publication_year(frame: pd.DataFrame) -> pd.Series:
    if "publication_year" in frame.columns:
        return pd.to_numeric(frame["publication_year"], errors="raise").astype(int)
    if "publication_date" in frame.columns:
        dates = pd.to_datetime(frame["publication_date"], errors="raise")
        return dates.dt.year.astype(int)
    raise ValueError("Stage A requires publication_year or publication_date")


def _compot_quartile(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="raise").astype(float)
    if numeric.isna().any():
        raise ValueError("COMPOT contains missing values")
    if len(numeric) < 4:
        raise ValueError("At least four projects are required for COMPOT quartiles")
    ranked = numeric.rank(method="first")
    return pd.qcut(ranked, 4, labels=[1, 2, 3, 4]).astype(int)


def build_stagea_project_panel(
    strict_panel: pd.DataFrame,
    *,
    expected_projects: int | None = 6536,
    expected_positive: int | None = 1881,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse a strict author-project panel to one audited row per project."""
    required = {"work_id", "compot", "firm_participation"}
    missing = sorted(required - set(strict_panel.columns))
    if missing:
        raise ValueError(f"Stage A strict panel is missing required columns: {missing}")
    field = _field_column(strict_panel)
    frame = strict_panel.copy()
    frame["publication_year"] = _publication_year(frame)
    frame["compot"] = pd.to_numeric(frame["compot"], errors="raise").astype(float)
    if frame["compot"].isna().any():
        raise ValueError("COMPOT contains missing values in the strict sample")
    frame["firm_participation"] = pd.to_numeric(
        frame["firm_participation"], errors="raise"
    ).astype(int)
    if not set(frame["firm_participation"].unique()).issubset({0, 1}):
        raise ValueError("firm_participation must be binary")

    consistency_columns = ["compot", "firm_participation", "publication_year", field]
    disagreement = (
        frame.groupby("work_id", observed=True)[consistency_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if disagreement.any():
        examples = disagreement[disagreement].index.astype(str).tolist()[:5]
        raise ValueError(
            f"Within-project Stage A fields disagree for work_ids={examples}"
        )

    project = (
        frame[["work_id", "compot", "firm_participation", "publication_year", field]]
        .drop_duplicates("work_id")
        .sort_values("work_id")
        .reset_index(drop=True)
        .rename(columns={field: "stagea_field_id"})
    )
    if project["work_id"].duplicated().any():
        raise ValueError("Stage A project collapse produced duplicate work_id values")
    projects = int(len(project))
    positives = int(project["firm_participation"].sum())
    if expected_projects is not None and projects != expected_projects:
        raise ValueError(
            f"Stage A expected {expected_projects} projects but found {projects}"
        )
    if expected_positive is not None and positives != expected_positive:
        raise ValueError(
            f"Stage A expected {expected_positive} firm-participation projects but found {positives}"
        )
    project["compot_z"] = (
        project["compot"] - float(project["compot"].mean())
    ) / (float(project["compot"].std(ddof=0)) or 1.0)
    project["compot_quartile"] = _compot_quartile(project["compot"])
    audit = {
        "projects": projects,
        "positive_projects": positives,
        "negative_projects": projects - positives,
        "compot_missing": int(project["compot"].isna().sum()),
        "work_id_duplicates": int(project["work_id"].duplicated().sum()),
        "year_min": int(project["publication_year"].min()),
        "year_max": int(project["publication_year"].max()),
        "conditional_population": "V3_STRICT_ELIGIBLE_PROJECTS",
    }
    return project, audit


def build_stagea_descriptives(project_panel: pd.DataFrame) -> pd.DataFrame:
    required = {"work_id", "compot", "compot_quartile", "firm_participation"}
    missing = sorted(required - set(project_panel.columns))
    if missing:
        raise ValueError(f"Stage A project panel is missing columns: {missing}")
    rows = []
    for quartile, group in project_panel.groupby("compot_quartile", observed=True, sort=True):
        rows.append({
            "compot_quartile": int(quartile),
            "projects": int(len(group)),
            "firm_participation_projects": int(group["firm_participation"].sum()),
            "firm_participation_rate": float(group["firm_participation"].mean()),
            "mean_compot": float(group["compot"].mean()),
            "median_compot": float(group["compot"].median()),
            "min_compot": float(group["compot"].min()),
            "max_compot": float(group["compot"].max()),
        })
    return pd.DataFrame(rows)


def _base_design(project_panel: pd.DataFrame) -> pd.DataFrame:
    controls = pd.DataFrame({
        "year": project_panel["publication_year"].astype(str),
        "field": project_panel["stagea_field_id"].fillna("__MISSING__").astype(str),
    })
    dummies = pd.get_dummies(controls, prefix=["year", "field"], drop_first=True, dtype=float)
    return dummies


def _coefficient_rows(
    model_name: str,
    terms: list[str],
    params: pd.Series,
    bse: pd.Series,
    pvalues: pd.Series,
    n_projects: int,
    fit_method: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for term in terms:
        rows.append({
            "model": model_name,
            "term": term,
            "coefficient": float(params.get(term, np.nan)),
            "standard_error": float(bse.get(term, np.nan)),
            "p_value": float(pvalues.get(term, np.nan)),
            "n_projects": int(n_projects),
            "fit_method": fit_method,
        })
    return rows


def fit_stagea_compot_models(project_panel: pd.DataFrame) -> pd.DataFrame:
    """Estimate prespecified Stage A LPM/logit/quartile models with year/field controls."""
    required = {
        "firm_participation", "compot_z", "compot_quartile",
        "publication_year", "stagea_field_id",
    }
    missing = sorted(required - set(project_panel.columns))
    if missing:
        raise ValueError(f"Stage A model panel is missing columns: {missing}")
    data = project_panel.copy()
    y = pd.to_numeric(data["firm_participation"], errors="raise").astype(float)
    controls = _base_design(data)
    rows: list[dict[str, Any]] = []

    x1 = pd.concat(
        [pd.Series(1.0, index=data.index, name="const"), data[["compot_z"]].astype(float), controls],
        axis=1,
    )
    lpm = sm.OLS(y, x1).fit(cov_type="HC1")
    rows.extend(_coefficient_rows(
        "A1_lpm", list(x1.columns), lpm.params, lpm.bse, lpm.pvalues, len(data), "ols_hc1"
    ))

    try:
        logit = sm.GLM(y, x1, family=sm.families.Binomial()).fit(cov_type="HC1")
        rows.extend(_coefficient_rows(
            "A2_logit", list(x1.columns), logit.params, logit.bse, logit.pvalues,
            len(data), "glm_binomial_hc1"
        ))
    except Exception:
        regularized = sm.GLM(y, x1, family=sm.families.Binomial()).fit_regularized(
            alpha=1e-8, L1_wt=0.0
        )
        params = pd.Series(np.asarray(regularized.params), index=x1.columns, dtype=float)
        nan = pd.Series(np.nan, index=x1.columns, dtype=float)
        rows.extend(_coefficient_rows(
            "A2_logit", list(x1.columns), params, nan, nan,
            len(data), "glm_binomial_l2_fallback"
        ))

    quartiles = pd.get_dummies(
        data["compot_quartile"].astype(int).astype(str), prefix="compot_q", drop_first=True, dtype=float
    )
    x3 = pd.concat(
        [pd.Series(1.0, index=data.index, name="const"), quartiles, controls], axis=1
    )
    q_lpm = sm.OLS(y, x3).fit(cov_type="HC1")
    rows.extend(_coefficient_rows(
        "A3_quartile_lpm", list(x3.columns), q_lpm.params, q_lpm.bse, q_lpm.pvalues,
        len(data), "ols_hc1"
    ))
    return pd.DataFrame(rows)
