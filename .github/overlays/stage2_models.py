"""Shared preparation helpers for legacy Stage 2 fixed-effects diagnostics.

The V5 relationship-transition estimators live in
``relationship_transition_models.py``. This module preserves the preparation
contract used by ``stage2_diagnostics.run_tie_stratified_models``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = float(values.mean())
    sd = float(values.std(ddof=0))
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(0.0, index=values.index, dtype=float)
    return (values - mean) / sd


def prepare_estimation_sample(frame: pd.DataFrame) -> pd.DataFrame:
    """Prepare an estimable project-stratified candidate-choice sample."""
    required = {
        "work_id",
        "selected",
        "author_prior_partner",
        "university_prior_partner",
        "prior_subfield_publication_count",
        "compot",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing estimation columns: {missing}")

    data = frame.copy()
    data["selected"] = pd.to_numeric(data["selected"], errors="raise").astype(int)
    if not data["selected"].isin([0, 1]).all():
        raise ValueError("selected must be binary")
    for column in ["author_prior_partner", "university_prior_partner"]:
        data[column] = pd.to_numeric(data[column], errors="raise").astype(int)
        if not data[column].isin([0, 1]).all():
            raise ValueError(f"{column} must be binary")

    counts = pd.to_numeric(
        data["prior_subfield_publication_count"], errors="raise"
    ).astype(float)
    if counts.lt(0).any():
        raise ValueError("prior_subfield_publication_count must be nonnegative")
    data["log_subfield_count_z"] = _zscore(np.log1p(counts))
    data["compot_z"] = _zscore(data["compot"])

    capability_source = (
        data["cognitive_fit_cosine"]
        if "cognitive_fit_cosine" in data.columns
        else data["log_subfield_count_z"]
    )
    data["capability_z"] = _zscore(capability_source)
    data["author_x_compot"] = data["author_prior_partner"] * data["compot_z"]
    data["university_x_compot"] = (
        data["university_prior_partner"] * data["compot_z"]
    )
    data["capability_x_compot"] = data["capability_z"] * data["compot_z"]

    model_columns = [
        "selected",
        "author_prior_partner",
        "university_prior_partner",
        "log_subfield_count_z",
        "author_x_compot",
        "university_x_compot",
        "capability_x_compot",
    ]
    finite = np.isfinite(data[model_columns].astype(float)).all(axis=1)
    data = data.loc[finite].copy()

    grouped = data.groupby("work_id", observed=True)["selected"].agg(["sum", "count"])
    eligible_ids = grouped.index[
        (grouped["sum"] > 0) & (grouped["sum"] < grouped["count"])
    ]
    data = data[data["work_id"].isin(eligible_ids)].copy()
    if data.empty:
        raise ValueError("No projects contain both selected and unselected alternatives")
    return data.reset_index(drop=True)
