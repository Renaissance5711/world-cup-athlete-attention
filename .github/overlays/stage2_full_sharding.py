"""Deterministic balanced sharding for the full 1,881-project TEM expansion."""
from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd

from stage2_pilot_sampling import _prepare_strata


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def assign_full_shards(
    projects: pd.DataFrame,
    *,
    shard_count: int = 5,
    seed: int = 20260811,
    expected_projects: int | None = 1881,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign every project to one balanced deterministic stratified shard."""
    if shard_count <= 1:
        raise ValueError("shard_count must be greater than one")
    required = {"work_id", "publication_year", "primary_field_id", "compot"}
    missing = sorted(required - set(projects.columns))
    if missing:
        raise ValueError(f"Full sharding input is missing columns: {missing}")
    if projects.empty:
        raise ValueError("Full sharding input is empty")
    if not projects["work_id"].is_unique:
        raise ValueError("Full sharding requires unique work_id values")
    if expected_projects is not None and len(projects) != expected_projects:
        raise ValueError(
            f"Expected {expected_projects} full Stage 2 projects but found {len(projects)}"
        )

    data = _prepare_strata(projects)
    data["_hash"] = data["work_id"].astype(str).map(
        lambda work_id: _stable_int(f"{seed}|{work_id}")
    )
    loads = [0 for _ in range(shard_count)]
    assignments: dict[str, int] = {}

    for stratum in sorted(data["pilot_stratum"].astype(str).unique()):
        group = data[data["pilot_stratum"].astype(str).eq(stratum)].sort_values(
            ["_hash", "work_id"]
        )
        work_ids = group["work_id"].astype(str).tolist()
        quotient, remainder = divmod(len(work_ids), shard_count)
        counts = [quotient for _ in range(shard_count)]
        remainder_order = sorted(
            range(shard_count),
            key=lambda shard: (
                loads[shard],
                _stable_int(f"{seed}|{stratum}|remainder|{shard}"),
            ),
        )
        for shard in remainder_order[:remainder]:
            counts[shard] += 1

        sequence: list[int] = []
        remaining = counts.copy()
        while len(sequence) < len(work_ids):
            available = [shard for shard in range(shard_count) if remaining[shard] > 0]
            order = sorted(
                available,
                key=lambda shard: (
                    -remaining[shard],
                    loads[shard] + (counts[shard] - remaining[shard]),
                    _stable_int(f"{seed}|{stratum}|cycle|{shard}"),
                ),
            )
            for shard in order:
                if remaining[shard] <= 0:
                    continue
                sequence.append(shard)
                remaining[shard] -= 1
                if len(sequence) == len(work_ids):
                    break
        for work_id, shard in zip(work_ids, sequence, strict=True):
            assignments[work_id] = shard
        for shard, count in enumerate(counts):
            loads[shard] += count

    assignment = data[
        ["work_id", "publication_year", "primary_field_id", "compot", "pilot_period", "compot_quartile", "pilot_stratum"]
    ].copy()
    assignment["shard_index"] = assignment["work_id"].astype(str).map(assignments).astype(int)
    assignment = assignment.rename(columns={"pilot_stratum": "shard_stratum"})
    assignment = assignment.sort_values("work_id").reset_index(drop=True)

    if assignment["shard_index"].isna().any():
        raise AssertionError("A full project was not assigned to a shard")
    if assignment["work_id"].duplicated().any():
        raise AssertionError("Shard assignment contains duplicate work_id values")
    sizes = assignment.groupby("shard_index", observed=True).size().reindex(
        range(shard_count), fill_value=0
    )
    if int(sizes.max() - sizes.min()) > 1:
        raise AssertionError(f"Shard sizes are imbalanced: {sizes.to_dict()}")
    stratum_spread = (
        assignment.groupby(["shard_stratum", "shard_index"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=range(shard_count), fill_value=0)
    )
    max_stratum_spread = int((stratum_spread.max(axis=1) - stratum_spread.min(axis=1)).max())
    if max_stratum_spread > 1:
        raise AssertionError("Within-stratum shard counts differ by more than one")

    years = pd.to_numeric(assignment["publication_year"], errors="raise").astype(int)
    has_train = bool(years.le(2018).any())
    has_test = bool(years.gt(2018).any())
    temporal_counts: dict[str, dict[str, int]] = {}
    if has_train and has_test:
        for shard in range(shard_count):
            shard_years = years[assignment["shard_index"].eq(shard)]
            pre = int(shard_years.le(2018).sum())
            post = int(shard_years.gt(2018).sum())
            if pre == 0 or post == 0:
                raise AssertionError(
                    f"Shard {shard} does not contain both temporal train/test periods"
                )
            temporal_counts[str(shard)] = {"through_2018": pre, "after_2018": post}

    audit: dict[str, Any] = {
        "projects": int(len(assignment)),
        "shards": int(shard_count),
        "seed": int(seed),
        "shard_sizes": {str(int(index)): int(value) for index, value in sizes.items()},
        "max_shard_size_difference": int(sizes.max() - sizes.min()),
        "max_within_stratum_difference": max_stratum_spread,
        "strata": int(assignment["shard_stratum"].nunique()),
        "temporal_counts": temporal_counts,
        "deterministic": True,
    }
    return assignment, audit
