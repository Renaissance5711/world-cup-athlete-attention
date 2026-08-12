"""Deterministic stratified sampling for the TEM realization pilot."""
from __future__ import annotations

import numpy as np
import pandas as pd

PERIOD_BINS = [1999, 2006, 2012, 2018, 2024]
PERIOD_LABELS = ["2000-2006", "2007-2012", "2013-2018", "2019-2024"]
QUARTILE_LABELS = ["Q1", "Q2", "Q3", "Q4"]


def _largest_remainder_allocation(counts: pd.Series, sample_size: int) -> pd.Series:
    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if sample_size > int(counts.sum()):
        raise ValueError("sample_size exceeds the available population")
    nonempty = counts[counts.gt(0)]
    if sample_size < len(nonempty):
        raise ValueError("sample_size is smaller than the number of nonempty strata")

    raw = nonempty / nonempty.sum() * sample_size
    allocated = np.floor(raw).astype(int)
    allocated[allocated.eq(0)] = 1

    while int(allocated.sum()) > sample_size:
        removable = allocated[allocated.gt(1)]
        if removable.empty:
            raise ValueError("Could not reduce stratum allocation to requested size")
        key = (allocated - raw).loc[removable.index].sort_values(ascending=False).index[0]
        allocated.loc[key] -= 1

    while int(allocated.sum()) < sample_size:
        room = nonempty - allocated
        eligible = room[room.gt(0)]
        if eligible.empty:
            raise ValueError("Could not increase stratum allocation to requested size")
        remainder = (raw - allocated).loc[eligible.index]
        key = remainder.sort_values(ascending=False).index[0]
        allocated.loc[key] += 1

    return allocated.reindex(counts.index, fill_value=0).astype(int)


def _prepare_strata(projects: pd.DataFrame) -> pd.DataFrame:
    data = projects.copy()
    data["pilot_period"] = pd.cut(
        pd.to_numeric(data["publication_year"], errors="raise"),
        bins=PERIOD_BINS,
        labels=PERIOD_LABELS,
    ).astype("string")
    if data["pilot_period"].isna().any():
        bad_years = sorted(data.loc[data["pilot_period"].isna(), "publication_year"].unique())
        raise ValueError(f"Publication years fall outside pilot period bins: {bad_years}")

    def assign_quartile(values: pd.Series) -> pd.Series:
        ranked = pd.to_numeric(values, errors="raise").rank(method="first")
        return pd.qcut(ranked, 4, labels=QUARTILE_LABELS).astype("string")

    data["compot_quartile"] = (
        data.groupby("pilot_period", observed=True)["compot"]
        .transform(assign_quartile)
        .astype("string")
    )
    data["pilot_stratum"] = (
        data["pilot_period"].astype(str)
        + "|"
        + data["primary_field_id"].astype(str)
        + "|"
        + data["compot_quartile"].astype(str)
    )
    return data


def select_stratified_pilot_projects(
    projects: pd.DataFrame,
    sample_size: int = 400,
    seed: int = 20260804,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select a reproducible period × field × COMPOT-quartile pilot sample."""
    required = {"work_id", "publication_year", "primary_field_id", "compot"}
    missing = sorted(required - set(projects.columns))
    if missing:
        raise ValueError(f"Missing sampling columns: {missing}")
    if projects.empty:
        raise ValueError("Pilot project input is empty")
    if not projects["work_id"].is_unique:
        raise ValueError("Pilot project input must contain unique work_id values")

    data = _prepare_strata(projects)
    effective_size = min(int(sample_size), len(data))
    population = data.groupby("pilot_stratum", observed=True).size().sort_index()

    if effective_size == len(data):
        sample = data.sort_values("work_id").reset_index(drop=True)
        allocation = population.copy()
    else:
        allocation = _largest_remainder_allocation(population, effective_size)
        rng = np.random.default_rng(seed)
        pieces: list[pd.DataFrame] = []
        for stratum, target_n in allocation.items():
            if target_n <= 0:
                continue
            group = data[data["pilot_stratum"].eq(stratum)]
            random_state = int(rng.integers(0, 2**31 - 1))
            pieces.append(group.sample(n=int(target_n), random_state=random_state))
        sample = (
            pd.concat(pieces, ignore_index=True)
            .sort_values(["pilot_period", "primary_field_id", "compot", "work_id"])
            .reset_index(drop=True)
        )

    audit = population.rename("population_n").reset_index()
    audit["target_n"] = audit["pilot_stratum"].map(allocation).fillna(0).astype(int)
    sampled_counts = sample.groupby("pilot_stratum", observed=True).size()
    audit["sampled_n"] = audit["pilot_stratum"].map(sampled_counts).fillna(0).astype(int)

    if len(sample) != effective_size or not sample["work_id"].is_unique:
        raise AssertionError("Pilot sampling contract failed")
    if int(audit["sampled_n"].sum()) != effective_size:
        raise AssertionError("Pilot sampling audit does not reconcile")
    return sample, audit
