#!/usr/bin/env python3
"""Run a restartable 400-project potential-match realization pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from stage2_pilot_metrics import build_project_realization_metrics, evaluate_pilot_gates
from stage2_pilot_models import fit_temporal_ranking_models
from stage2_pilot_risk_set import (
    assemble_pilot_candidates,
    enrich_candidate_relationships,
    inject_realized_firms,
)
from stage2_pilot_sampling import select_stratified_pilot_projects
from src.cognitive_fit import compute_project_firm_cognitive_fit, work_document_text
from src.openalex_stage2 import OpenAlexClient
from src.stage2_manifest import build_extraction_manifests, build_university_manifests
from src.stage2_openalex_extract import (
    fetch_candidate_firm_text_history,
    fetch_entity_year_history,
    fetch_exact_field_counts,
    fetch_project_details,
)
from src.stage2_validation import audit_full_project_input


def required_output_filenames() -> list[str]:
    return [
        "pilot_projects.csv",
        "pilot_sampling_audit.csv",
        "pilot_strict_panel.csv",
        "project_details.csv",
        "author_company_history.csv",
        "university_company_history.csv",
        "subfield_exact_date_company_counts.csv",
        "natural_candidate_long_top50.csv",
        "natural_candidate_audit_top50.csv",
        "estimation_candidate_long_top50.csv",
        "natural_candidate_long_top100.csv",
        "natural_candidate_audit_top100.csv",
        "estimation_candidate_long_top100.csv",
        "candidate_firm_prepublication_text_history.csv",
        "cognitive_candidate_long_top50.csv",
        "cognitive_candidate_long_top100.csv",
        "pilot_project_realization_metrics_top50.csv",
        "pilot_project_realization_metrics_top100.csv",
        "pilot_ranking_coefficients_top50.csv",
        "pilot_ranking_metrics_top50.csv",
        "pilot_ranking_coefficients_top100.csv",
        "pilot_ranking_metrics_top100.csv",
        "pilot_gate_decision_top50.json",
        "pilot_gate_decision_top100.json",
        "pilot_time_provenance_audit.json",
        "pilot_summary.json",
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    temporary.replace(path)


def _sample_signature(work_ids: pd.Series) -> str:
    values = "|".join(sorted(work_ids.astype(str).tolist()))
    return hashlib.sha256(values.encode("utf-8")).hexdigest()


def add_deterministic_field_rank(field_counts: pd.DataFrame) -> pd.DataFrame:
    """Rank firms by count descending and company ID ascending per project field-date."""
    required = {
        "subfield_id",
        "as_of_date",
        "company_id",
        "prior_subfield_publication_count",
    }
    missing = sorted(required - set(field_counts.columns))
    if missing:
        raise ValueError(f"Field counts are missing columns: {missing}")
    if field_counts.empty:
        out = field_counts.copy()
        out["field_rank"] = pd.Series(dtype="int64")
        return out
    out = field_counts.copy()
    out["as_of_date"] = pd.to_datetime(out["as_of_date"], utc=True).dt.strftime("%Y-%m-%d")
    out["prior_subfield_publication_count"] = pd.to_numeric(
        out["prior_subfield_publication_count"], errors="raise"
    ).astype(int)
    out = (
        out.sort_values(
            [
                "subfield_id",
                "as_of_date",
                "prior_subfield_publication_count",
                "company_id",
            ],
            ascending=[True, True, False, True],
        )
        .drop_duplicates(["subfield_id", "as_of_date", "company_id"])
        .reset_index(drop=True)
    )
    out["field_rank"] = (
        out.groupby(["subfield_id", "as_of_date"], observed=True).cumcount() + 1
    ).astype(int)
    return out


def _write_manifests(frames: dict[str, pd.DataFrame], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_csv(directory / f"{name}.csv", index=False)


def _validate_work_ids(frame: pd.DataFrame, expected: set[str], label: str) -> None:
    if "work_id" not in frame.columns:
        raise ValueError(f"{label} has no work_id column")
    observed = set(frame["work_id"].astype(str).unique())
    if observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise ValueError(f"{label} work IDs mismatch; missing={missing}, extra={extra}")


def _load_or_build_csv(
    path: Path,
    builder: Callable[[], pd.DataFrame],
    *,
    expected_work_ids: set[str] | None = None,
    label: str,
) -> pd.DataFrame:
    if path.exists():
        frame = pd.read_csv(path)
    else:
        frame = builder()
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    if expected_work_ids is not None:
        _validate_work_ids(frame, expected_work_ids, label)
    return frame


def _prepare_project_detail_cache(shared_cache: Path, sample_seed: int) -> Path:
    """Return a unique cache root for batch-indexed project-detail requests."""
    shared_cache.mkdir(parents=True, exist_ok=True)
    detail_cache = shared_cache / f"realization_pilot_project_details_{sample_seed}"
    (detail_cache / "project_details").mkdir(parents=True, exist_ok=True)
    return detail_cache


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
        _validate_work_ids(details, expected, "project details checkpoint")
        if not csv_path.exists():
            details.drop(columns=["abstract_inverted_index"], errors="ignore").to_csv(
                csv_path, index=False
            )
        return details
    details = fetch_project_details(client, projects, cache_dir)
    _validate_work_ids(details, expected, "fetched project details")
    details.to_pickle(pickle_path)
    details.drop(columns=["abstract_inverted_index"], errors="ignore").to_csv(
        csv_path, index=False
    )
    return details


def _project_texts(projects: pd.DataFrame, details: pd.DataFrame) -> pd.DataFrame:
    frame = projects[["work_id", "publication_date"]].merge(
        details[["work_id", "title", "abstract_inverted_index"]],
        on="work_id",
        validate="one_to_one",
    )
    frame["project_text"] = frame.apply(
        lambda row: work_document_text(
            {
                "title": row["title"],
                "abstract_inverted_index": row["abstract_inverted_index"],
            }
        ),
        axis=1,
    )
    frame["project_text_missing"] = frame["project_text"].fillna("").eq("").astype(int)
    return frame


def _merge_cognitive_fit(
    candidates: pd.DataFrame,
    cognitive_union: pd.DataFrame,
    project_texts: pd.DataFrame,
) -> pd.DataFrame:
    out = candidates.merge(
        cognitive_union,
        on=["work_id", "company_id"],
        how="left",
        validate="one_to_one",
    )
    out = out.merge(
        project_texts[["work_id", "project_text_missing"]],
        on="work_id",
        how="left",
        validate="many_to_one",
    )
    out["cognitive_fit_publication_count"] = pd.to_numeric(
        out["cognitive_fit_publication_count"], errors="coerce"
    ).fillna(0).astype(int)
    measurable = (
        out["cognitive_fit_publication_count"].gt(0)
        & out["project_text_missing"].eq(0)
    )
    out.loc[~measurable, "cognitive_fit_cosine"] = np.nan
    return out


def _validate_time_provenance(
    candidates: list[pd.DataFrame],
    text_history: pd.DataFrame,
    projects: pd.DataFrame,
) -> dict[str, object]:
    violations: list[str] = []
    rows_checked = 0
    for frame in candidates:
        publication = pd.to_datetime(frame["publication_date"], utc=True, errors="coerce")
        rows_checked += len(frame)
        for column in [
            "author_last_evidence_date",
            "university_last_evidence_date",
            "field_query_to_date",
            "cognitive_evidence_last_date",
        ]:
            if column not in frame.columns:
                continue
            values = pd.to_datetime(frame[column], utc=True, errors="coerce")
            observed = frame[column].notna()
            bad = observed & values.ge(publication)
            if bad.any():
                violations.append(f"{column}:{int(bad.sum())}")
        start = pd.to_datetime(frame["window_start_inclusive"], utc=True, errors="coerce")
        end = pd.to_datetime(frame["window_end_exclusive"], utc=True, errors="coerce")
        if start.isna().any() or end.isna().any() or (start >= end).any() or (end > publication).any():
            violations.append("invalid_project_window")

    if not text_history.empty:
        focal_dates = projects.set_index("work_id")["publication_date"]
        focal = pd.to_datetime(
            text_history["focal_work_id"].map(focal_dates), utc=True, errors="coerce"
        )
        evidence = pd.to_datetime(text_history["publication_date"], utc=True, errors="coerce")
        bad = evidence.ge(focal) | evidence.isna() | focal.isna()
        if bad.any():
            violations.append(f"text_history:{int(bad.sum())}")

    if violations:
        raise ValueError(f"Time provenance violations detected: {violations}")
    return {
        "auditable": True,
        "violations": 0,
        "candidate_rows_checked": rows_checked,
        "text_history_rows_checked": int(len(text_history)),
        "strictly_prepublication": True,
    }


def _run_specification(
    label: str,
    natural: pd.DataFrame,
    estimation: pd.DataFrame,
    cognitive_union: pd.DataFrame,
    project_texts: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    cognitive = _merge_cognitive_fit(estimation, cognitive_union, project_texts)
    cognitive.to_csv(output_dir / f"cognitive_candidate_long_{label}.csv", index=False)
    metrics = build_project_realization_metrics(cognitive, conflict_threshold=0.05)
    metrics.to_csv(output_dir / f"pilot_project_realization_metrics_{label}.csv", index=False)
    coefficients, ranking = fit_temporal_ranking_models(cognitive, train_end_year=2018)
    coefficients.to_csv(output_dir / f"pilot_ranking_coefficients_{label}.csv", index=False)
    ranking.to_csv(output_dir / f"pilot_ranking_metrics_{label}.csv", index=False)
    decision = evaluate_pilot_gates(cognitive, ranking, metrics)
    _write_json(output_dir / f"pilot_gate_decision_{label}.json", decision)
    return cognitive, metrics, decision


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--strict-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=400)
    parser.add_argument("--sample-seed", type=int, default=20260804)
    parser.add_argument("--field-fetch-top-n", type=int, default=100)
    parser.add_argument("--main-field-top-n", type=int, default=50)
    parser.add_argument("--sensitivity-field-top-n", type=int, default=100)
    parser.add_argument("--university-min-works", type=int, default=5)
    parser.add_argument("--cognitive-batch-size", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    projects_full = pd.read_csv(args.projects)
    strict_full = pd.read_csv(args.strict_panel)
    full_audit = audit_full_project_input(projects_full)

    pilot_projects, sampling_audit = select_stratified_pilot_projects(
        projects_full, sample_size=args.sample_size, seed=args.sample_seed
    )
    expected_ids = set(pilot_projects["work_id"].astype(str))
    sample_signature = _sample_signature(pilot_projects["work_id"])
    config = {
        "sample_size_requested": args.sample_size,
        "sample_size_realized": len(pilot_projects),
        "sample_seed": args.sample_seed,
        "sample_signature": sample_signature,
        "field_fetch_top_n": args.field_fetch_top_n,
        "main_field_top_n": args.main_field_top_n,
        "sensitivity_field_top_n": args.sensitivity_field_top_n,
        "university_min_works": args.university_min_works,
    }
    config_path = args.output_dir / "pilot_run_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError("Pilot output directory belongs to a different sample/configuration")
    else:
        _write_json(config_path, config)

    pilot_projects.to_csv(args.output_dir / "pilot_projects.csv", index=False)
    sampling_audit.to_csv(args.output_dir / "pilot_sampling_audit.csv", index=False)
    strict = strict_full[
        strict_full["work_id"].astype(str).isin(expected_ids)
    ].copy()
    strict_ids = set(strict["work_id"].astype(str).unique())
    if strict_ids != expected_ids:
        raise ValueError(
            "Strict author-project panel does not cover every sampled project"
        )
    strict.to_csv(args.output_dir / "pilot_strict_panel.csv", index=False)

    manifests = build_extraction_manifests(pilot_projects)
    _write_manifests(manifests, args.output_dir / "manifests")
    project_detail_cache = _prepare_project_detail_cache(
        args.cache_dir, args.sample_seed
    )
    client = OpenAlexClient()

    details = _load_or_fetch_details(
        client, pilot_projects, project_detail_cache, args.output_dir
    )
    if details["actual_company_count"].lt(1).any():
        raise ValueError("Pilot contains a project without a realized company")
    university_manifests = build_university_manifests(pilot_projects, details)
    _write_manifests(university_manifests, args.output_dir / "manifests")

    author_history = _load_or_build_csv(
        args.output_dir / "author_company_history.csv",
        lambda: fetch_entity_year_history(
            client,
            manifests["author_year_units"],
            args.cache_dir,
            entity_column="author_id",
            openalex_filter_field="authorships.author.id",
        ),
        label="author history",
    )
    university_history = _load_or_build_csv(
        args.output_dir / "university_company_history.csv",
        lambda: fetch_entity_year_history(
            client,
            university_manifests["university_year_units"],
            args.cache_dir,
            entity_column="university_id",
            openalex_filter_field="authorships.institutions.id",
        ),
        label="university history",
    )
    field_counts = _load_or_build_csv(
        args.output_dir / "subfield_exact_date_company_counts.csv",
        lambda: add_deterministic_field_rank(
            fetch_exact_field_counts(
                client,
                manifests["subfield_date_units"],
                args.cache_dir,
                top_n=args.field_fetch_top_n,
            )
        ),
        label="field counts",
    )
    field_counts = add_deterministic_field_rank(field_counts)
    field_counts.to_csv(args.output_dir / "subfield_exact_date_company_counts.csv", index=False)

    candidate_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for label, top_n in [
        ("top50", args.main_field_top_n),
        ("top100", args.sensitivity_field_top_n),
    ]:
        natural, natural_audit = assemble_pilot_candidates(
            pilot_projects,
            details,
            author_history,
            university_history,
            field_counts,
            field_top_n=top_n,
            university_min_works=args.university_min_works,
        )
        natural.to_csv(args.output_dir / f"natural_candidate_long_{label}.csv", index=False)
        natural_audit.to_csv(args.output_dir / f"natural_candidate_audit_{label}.csv", index=False)
        estimation = inject_realized_firms(natural, pilot_projects, details)
        estimation = enrich_candidate_relationships(
            estimation,
            pilot_projects,
            details,
            author_history,
            university_history,
            university_min_works=args.university_min_works,
        )
        estimation.to_csv(args.output_dir / f"estimation_candidate_long_{label}.csv", index=False)
        candidate_tables[label] = (natural, estimation)

    union_candidates = pd.concat(
        [candidate_tables["top50"][1], candidate_tables["top100"][1]],
        ignore_index=True,
    ).drop_duplicates(["work_id", "company_id"])
    text_history = _load_or_build_csv(
        args.output_dir / "candidate_firm_prepublication_text_history.csv",
        lambda: fetch_candidate_firm_text_history(
            client,
            pilot_projects,
            union_candidates,
            args.cache_dir,
            batch_size=args.cognitive_batch_size,
        ),
        label="candidate text history",
    )
    if not text_history.empty:
        observed_focal = set(text_history["focal_work_id"].astype(str))
        if not observed_focal.issubset(expected_ids):
            raise ValueError("Candidate text checkpoint contains nonpilot projects")

    project_texts = _project_texts(pilot_projects, details)
    cognitive_union = compute_project_firm_cognitive_fit(
        union_candidates[["work_id", "company_id"]],
        project_texts[["work_id", "publication_date", "project_text"]],
        text_history,
    )

    cognitive_frames: dict[str, pd.DataFrame] = {}
    decisions: dict[str, dict[str, object]] = {}
    for label in ["top50", "top100"]:
        natural, estimation = candidate_tables[label]
        cognitive, _, decision = _run_specification(
            label,
            natural,
            estimation,
            cognitive_union,
            project_texts,
            args.output_dir,
        )
        cognitive_frames[label] = cognitive
        decisions[label] = decision

    time_audit = _validate_time_provenance(
        [cognitive_frames["top50"], cognitive_frames["top100"]],
        text_history,
        pilot_projects,
    )
    _write_json(args.output_dir / "pilot_time_provenance_audit.json", time_audit)

    summary = {
        "status": "PILOT_SOFTWARE_COMPLETE",
        "input_audit": full_audit,
        "pilot_config": config,
        "top50": decisions["top50"],
        "top100": decisions["top100"],
        "directionally_stable": (
            decisions["top50"]["recommendation"] == decisions["top100"]["recommendation"]
        ),
        "api_key_present": bool(os.getenv("OPENALEX_API_KEY", "").strip()),
        "time_provenance": time_audit,
    }
    _write_json(args.output_dir / "pilot_summary.json", summary)

    missing_outputs = [
        name for name in required_output_filenames() if not (args.output_dir / name).exists()
    ]
    if missing_outputs:
        raise ValueError(f"Pilot did not produce required outputs: {missing_outputs}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
