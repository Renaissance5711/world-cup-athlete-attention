#!/usr/bin/env python3
"""Pilot entrypoint that uses free OpenAlex singleton project-detail requests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import run_stage2_realization_pilot as base
from src.openalex_stage2 import OpenAlexClient
from src.stage2_risk_set import normalize_openalex_id, parse_project_detail


def fetch_project_details_singletons(
    client: OpenAlexClient,
    projects: pd.DataFrame,
    cache_dir: Path,
) -> pd.DataFrame:
    """Fetch sampled projects as free singleton requests with per-work caching."""
    required = {"work_id", "focal_author_ids"}
    missing = sorted(required - set(projects.columns))
    if missing:
        raise ValueError(f"Project singleton input is missing columns: {missing}")
    cache_root = cache_dir / "project_details_singletons"
    cache_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for project in projects.sort_values("work_id").itertuples(index=False):
        work_id = str(project.work_id)
        cache_path = cache_root / f"{work_id}.json"
        if cache_path.exists():
            work = json.loads(cache_path.read_text(encoding="utf-8"))
        else:
            work = client.get_json(
                f"/works/{work_id}",
                {
                    "select": (
                        "id,title,abstract_inverted_index,publication_date,authorships"
                    )
                },
            )
            temporary = cache_path.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(work, ensure_ascii=False), encoding="utf-8"
            )
            temporary.replace(cache_path)
        fetched_id = normalize_openalex_id(work.get("id"))
        if fetched_id != work_id:
            raise ValueError(
                f"OpenAlex singleton returned {fetched_id!r} for {work_id!r}"
            )
        focal_authors = {
            value
            for value in str(project.focal_author_ids).split("|")
            if value
        }
        rows.append(parse_project_detail(work, focal_authors))
    return pd.DataFrame(rows)


def _load_or_fetch_details(
    client: OpenAlexClient,
    projects: pd.DataFrame,
    cache_dir: Path,
    output_dir: Path,
) -> pd.DataFrame:
    pickle_path = output_dir / "project_details.pkl"
    csv_path = output_dir / "project_details.csv"
    expected = set(projects["work_id"].astype(str))
    if pickle_path.exists():
        details = pd.read_pickle(pickle_path)
        base._validate_work_ids(details, expected, "project details checkpoint")
        if not csv_path.exists():
            details.drop(columns=["abstract_inverted_index"], errors="ignore").to_csv(
                csv_path, index=False
            )
        return details
    details = fetch_project_details_singletons(client, projects, cache_dir)
    base._validate_work_ids(details, expected, "fetched project details")
    details.to_pickle(pickle_path)
    details.drop(columns=["abstract_inverted_index"], errors="ignore").to_csv(
        csv_path, index=False
    )
    return details


base._load_or_fetch_details = _load_or_fetch_details


def main(argv: list[str] | None = None) -> int:
    return base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
