"""Candidate construction for the TEM potential-match realization pilot."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import pandas as pd


def _split_ids(value: object) -> list[str]:
    if value is None or pd.isna(value) or not str(value).strip():
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def _normalize_dates(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_datetime(frame[column], errors="coerce", utc=True).dt.tz_localize(None)


def _empty_attributes() -> dict[str, object]:
    return {
        "author_prior_partner": 0,
        "author_relationship_strength": 0,
        "author_last_evidence_date": pd.NaT,
        "author_recency_years": np.nan,
        "university_prior_partner": 0,
        "university_prior_work_count": 0,
        "university_last_evidence_date": pd.NaT,
        "university_recency_years": np.nan,
        "strong_university_candidate": 0,
        "subfield_active_company": 0,
        "field_rank": np.nan,
        "prior_subfield_publication_count": 0,
        "field_query_from_date": pd.NaT,
        "field_query_to_date": pd.NaT,
    }


def _project_metadata(project: pd.Series) -> dict[str, object]:
    return {
        "publication_year": int(project["publication_year"]),
        "publication_date": project["publication_date"],
        "primary_subfield_id": project["primary_subfield_id"],
        "primary_field_id": project["primary_field_id"],
        "compot": project["compot"],
        "focal_author_ids": project.get("focal_author_ids", ""),
    }


def assemble_pilot_candidates(
    projects: pd.DataFrame,
    project_details: pd.DataFrame,
    author_history: pd.DataFrame,
    university_history: pd.DataFrame,
    field_counts: pd.DataFrame,
    *,
    field_top_n: int,
    university_min_works: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a natural project-specific candidate set without forced outcomes."""
    if field_top_n <= 0:
        raise ValueError("field_top_n must be positive")
    if university_min_works <= 0:
        raise ValueError("university_min_works must be positive")
    required_projects = {
        "work_id", "publication_year", "publication_date", "primary_subfield_id",
        "primary_field_id", "compot", "focal_author_ids",
    }
    missing = sorted(required_projects - set(projects.columns))
    if missing:
        raise ValueError(f"Missing project columns: {missing}")
    required_details = {"work_id", "actual_company_ids", "focal_education_ids"}
    missing = sorted(required_details - set(project_details.columns))
    if missing:
        raise ValueError(f"Missing project-detail columns: {missing}")
    if not projects["work_id"].is_unique or not project_details["work_id"].is_unique:
        raise ValueError("Projects and project details must be unique by work_id")

    project_frame = projects.copy()
    project_frame["publication_date"] = _normalize_dates(project_frame, "publication_date")
    details = project_details[sorted(required_details)].copy()
    merged_projects = project_frame.merge(details, on="work_id", how="left", validate="one_to_one")
    if merged_projects["actual_company_ids"].isna().any():
        raise ValueError("Missing project details for sampled projects")

    ah = author_history.copy()
    uh = university_history.copy()
    if not ah.empty:
        ah["evidence_date"] = _normalize_dates(ah, "evidence_date")
    if not uh.empty:
        uh["evidence_date"] = _normalize_dates(uh, "evidence_date")
    fc = field_counts.copy()
    if not fc.empty:
        fc["as_of_date"] = _normalize_dates(fc, "as_of_date")
        if "field_rank" not in fc.columns:
            fc = fc.sort_values(
                ["subfield_id", "as_of_date", "prior_subfield_publication_count", "company_id"],
                ascending=[True, True, False, True],
            )
            fc["field_rank"] = fc.groupby(["subfield_id", "as_of_date"], observed=True).cumcount() + 1

    rows: list[dict[str, object]] = []
    audit_rows: list[dict[str, object]] = []

    for project in merged_projects.to_dict("records"):
        work_id = str(project["work_id"])
        publication_date = pd.Timestamp(project["publication_date"])
        window_start = publication_date - pd.DateOffset(years=5)
        author_ids = _split_ids(project.get("focal_author_ids"))
        university_ids = _split_ids(project.get("focal_education_ids"))
        actual_ids = set(_split_ids(project.get("actual_company_ids")))
        candidate_attributes: dict[str, dict[str, object]] = defaultdict(_empty_attributes)

        if author_ids and not ah.empty:
            sub = ah[
                ah["author_id"].astype(str).isin(author_ids)
                & ah["evidence_date"].ge(window_start)
                & ah["evidence_date"].lt(publication_date)
            ]
            for company_id, group in sub.groupby("company_id", observed=True):
                company_id = str(company_id)
                attrs = candidate_attributes[company_id]
                attrs["author_prior_partner"] = 1
                attrs["author_relationship_strength"] = int(group["prior_work_id"].astype(str).nunique())
                last_date = group["evidence_date"].max()
                attrs["author_last_evidence_date"] = last_date
                attrs["author_recency_years"] = (publication_date - last_date).days / 365.25

        university_stats: dict[str, dict[str, object]] = {}
        if university_ids and not uh.empty:
            sub = uh[
                uh["university_id"].astype(str).isin(university_ids)
                & uh["evidence_date"].ge(window_start)
                & uh["evidence_date"].lt(publication_date)
            ]
            for company_id, group in sub.groupby("company_id", observed=True):
                company_id = str(company_id)
                count = int(group["prior_work_id"].astype(str).nunique())
                last_date = group["evidence_date"].max()
                university_stats[company_id] = {
                    "university_prior_partner": 1,
                    "university_prior_work_count": count,
                    "university_last_evidence_date": last_date,
                    "university_recency_years": (publication_date - last_date).days / 365.25,
                    "strong_university_candidate": int(count >= university_min_works),
                }
                if count >= university_min_works:
                    candidate_attributes[company_id].update(university_stats[company_id])

        if not fc.empty:
            project_subfield = pd.to_numeric(
                pd.Series([project["primary_subfield_id"]]), errors="coerce"
            ).iloc[0]
            field_subfield = pd.to_numeric(fc["subfield_id"], errors="coerce")
            if pd.notna(project_subfield):
                subfield_match = field_subfield.eq(float(project_subfield))
            else:
                subfield_match = fc["subfield_id"].astype(str).eq(
                    str(project["primary_subfield_id"])
                )
            field_sub = fc[
                subfield_match
                & fc["as_of_date"].eq(publication_date)
                & pd.to_numeric(fc["field_rank"], errors="coerce").le(field_top_n)
            ]
            for field_row in field_sub.itertuples(index=False):
                company_id = str(field_row.company_id)
                attrs = candidate_attributes[company_id]
                attrs["subfield_active_company"] = 1
                attrs["field_rank"] = int(field_row.field_rank)
                attrs["prior_subfield_publication_count"] = int(field_row.prior_subfield_publication_count)
                attrs["field_query_from_date"] = getattr(field_row, "query_from_date", pd.NaT)
                attrs["field_query_to_date"] = getattr(field_row, "query_to_date", pd.NaT)

        for company_id, stats in university_stats.items():
            if company_id in candidate_attributes:
                candidate_attributes[company_id].update(stats)

        for company_id in sorted(candidate_attributes):
            attrs = candidate_attributes[company_id]
            row = {
                "work_id": work_id,
                "company_id": company_id,
                **_project_metadata(pd.Series(project)),
                "window_start_inclusive": window_start.strftime("%Y-%m-%d"),
                "window_end_exclusive": publication_date.strftime("%Y-%m-%d"),
                **attrs,
                "selected": int(company_id in actual_ids),
                "natural_candidate": 1,
                "forced_selected_candidate": 0,
            }
            rows.append(row)

        natural_selected_count = len(actual_ids & set(candidate_attributes))
        audit_rows.append(
            {
                "work_id": work_id,
                "actual_selected_count": len(actual_ids),
                "natural_candidate_count": len(candidate_attributes),
                "natural_selected_count": natural_selected_count,
                "natural_project_recall": int(natural_selected_count > 0),
                "natural_firm_instance_recall": (
                    natural_selected_count / len(actual_ids) if actual_ids else np.nan
                ),
            }
        )

    natural = pd.DataFrame(rows)
    if not natural.empty and natural.groupby(["work_id", "company_id"]).size().gt(1).any():
        raise AssertionError("Natural candidate set contains duplicate project-company rows")
    return natural, pd.DataFrame(audit_rows)


def inject_realized_firms(
    natural_candidates: pd.DataFrame,
    projects: pd.DataFrame,
    project_details: pd.DataFrame,
) -> pd.DataFrame:
    """Append realized firms missing from the natural discovery set."""
    details = project_details[["work_id", "actual_company_ids"]].copy()
    metadata = projects.merge(details, on="work_id", validate="one_to_one")
    existing = natural_candidates.copy()
    defaults = _empty_attributes()

    if existing.empty:
        existing = pd.DataFrame(columns=["work_id", "company_id"])
    if "selected" not in existing.columns:
        existing["selected"] = 0
    if "natural_candidate" not in existing.columns:
        existing["natural_candidate"] = 1
    if "forced_selected_candidate" not in existing.columns:
        existing["forced_selected_candidate"] = 0

    existing_keys = set(zip(existing["work_id"].astype(str), existing["company_id"].astype(str)))
    actual_by_project = {
        str(row.work_id): set(_split_ids(row.actual_company_ids))
        for row in metadata.itertuples(index=False)
    }
    existing["selected"] = [
        int(str(company_id) in actual_by_project.get(str(work_id), set()))
        for work_id, company_id in zip(existing["work_id"], existing["company_id"])
    ]

    injected_rows: list[dict[str, object]] = []
    for project in metadata.to_dict("records"):
        work_id = str(project["work_id"])
        for company_id in sorted(_split_ids(project["actual_company_ids"])):
            if (work_id, company_id) in existing_keys:
                continue
            publication_date = pd.Timestamp(pd.to_datetime(project["publication_date"], utc=True)).tz_localize(None)
            window_start = publication_date - pd.DateOffset(years=5)
            injected_rows.append(
                {
                    "work_id": work_id,
                    "company_id": company_id,
                    **_project_metadata(pd.Series(project)),
                    "window_start_inclusive": window_start.strftime("%Y-%m-%d"),
                    "window_end_exclusive": publication_date.strftime("%Y-%m-%d"),
                    **defaults,
                    "selected": 1,
                    "natural_candidate": 0,
                    "forced_selected_candidate": 1,
                }
            )

    out = pd.concat([existing, pd.DataFrame(injected_rows)], ignore_index=True, sort=False)
    for column, value in defaults.items():
        if column not in out.columns:
            out[column] = value
    if out.groupby(["work_id", "company_id"]).size().gt(1).any():
        raise AssertionError("Estimation candidate set contains duplicate project-company rows")
    return out.sort_values(["work_id", "company_id"]).reset_index(drop=True)


def enrich_candidate_relationships(
    candidate_long: pd.DataFrame,
    projects: pd.DataFrame,
    project_details: pd.DataFrame,
    author_history: pd.DataFrame,
    university_history: pd.DataFrame,
    *,
    university_min_works: int = 5,
) -> pd.DataFrame:
    """Recompute exact-window relationship attributes for every candidate row."""
    if university_min_works <= 0:
        raise ValueError("university_min_works must be positive")
    details = project_details[["work_id", "focal_education_ids"]].copy()
    metadata = projects.merge(details, on="work_id", validate="one_to_one").copy()
    metadata["publication_date"] = _normalize_dates(metadata, "publication_date")
    project_map = metadata.set_index("work_id").to_dict("index")

    ah = author_history.copy()
    uh = university_history.copy()
    if not ah.empty:
        ah["evidence_date"] = _normalize_dates(ah, "evidence_date")
    if not uh.empty:
        uh["evidence_date"] = _normalize_dates(uh, "evidence_date")

    out = candidate_long.copy()
    for column, default in _empty_attributes().items():
        if column not in out.columns:
            out[column] = default

    for index, row in out.iterrows():
        work_id = str(row["work_id"])
        company_id = str(row["company_id"])
        if work_id not in project_map:
            raise ValueError(f"Candidate work_id is not in sampled projects: {work_id}")
        project = project_map[work_id]
        publication_date = pd.Timestamp(project["publication_date"])
        window_start = publication_date - pd.DateOffset(years=5)
        author_ids = _split_ids(project.get("focal_author_ids"))
        university_ids = _split_ids(project.get("focal_education_ids"))

        if author_ids and not ah.empty:
            history = ah[
                ah["author_id"].astype(str).isin(author_ids)
                & ah["company_id"].astype(str).eq(company_id)
                & ah["evidence_date"].ge(window_start)
                & ah["evidence_date"].lt(publication_date)
            ]
        else:
            history = ah.iloc[0:0]
        if not history.empty:
            last_date = history["evidence_date"].max()
            out.at[index, "author_prior_partner"] = 1
            out.at[index, "author_relationship_strength"] = int(
                history["prior_work_id"].astype(str).nunique()
            )
            out.at[index, "author_last_evidence_date"] = last_date
            out.at[index, "author_recency_years"] = (
                publication_date - last_date
            ).days / 365.25
        else:
            out.at[index, "author_prior_partner"] = 0
            out.at[index, "author_relationship_strength"] = 0
            out.at[index, "author_last_evidence_date"] = pd.NaT
            out.at[index, "author_recency_years"] = np.nan

        if university_ids and not uh.empty:
            history = uh[
                uh["university_id"].astype(str).isin(university_ids)
                & uh["company_id"].astype(str).eq(company_id)
                & uh["evidence_date"].ge(window_start)
                & uh["evidence_date"].lt(publication_date)
            ]
        else:
            history = uh.iloc[0:0]
        if not history.empty:
            count = int(history["prior_work_id"].astype(str).nunique())
            last_date = history["evidence_date"].max()
            out.at[index, "university_prior_partner"] = 1
            out.at[index, "university_prior_work_count"] = count
            out.at[index, "university_last_evidence_date"] = last_date
            out.at[index, "university_recency_years"] = (
                publication_date - last_date
            ).days / 365.25
            out.at[index, "strong_university_candidate"] = int(
                count >= university_min_works
            )
        else:
            out.at[index, "university_prior_partner"] = 0
            out.at[index, "university_prior_work_count"] = 0
            out.at[index, "university_last_evidence_date"] = pd.NaT
            out.at[index, "university_recency_years"] = np.nan
            out.at[index, "strong_university_candidate"] = 0

    return out
