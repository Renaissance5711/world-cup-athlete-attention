#!/usr/bin/env python3
"""Run one deterministic extraction shard for the full TEM realization analysis."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd

import run_stage2_realization_pilot as base
import run_stage2_realization_pilot_v2  # noqa: F401  # applies UTC/singleton fixes to base
from stage2_pilot_metrics import build_project_realization_metrics
from stage2_pilot_risk_set import (
    assemble_pilot_candidates,
    enrich_candidate_relationships,
    inject_realized_firms,
)
from src.openalex_stage2 import OpenAlexClient
from src.stage2_manifest import build_extraction_manifests, build_university_manifests
from src.stage2_openalex_extract import (
    fetch_candidate_firm_text_history,
    fetch_entity_year_history,
    fetch_exact_field_counts,
)
from src.stage2_validation import audit_full_project_input


def required_shard_output_filenames() -> list[str]:
    return [
        "shard_projects.csv",
        "shard_strict_panel.csv",
        "shard_config.json",
        "shard_summary.json",
        "project_details.csv",
        "natural_candidate_audit_top50.csv",
        "natural_candidate_audit_top100.csv",
        "cognitive_candidate_long_top50.csv",
        "cognitive_candidate_long_top100.csv",
        "project_realization_metrics_top50.csv",
        "project_realization_metrics_top100.csv",
        "shard_time_provenance_audit.json",
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--strict-panel", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--field-fetch-top-n", type=int, default=100)
    parser.add_argument("--main-field-top-n", type=int, default=50)
    parser.add_argument("--sensitivity-field-top-n", type=int, default=100)
    parser.add_argument("--university-min-works", type=int, default=5)
    parser.add_argument("--cognitive-batch-size", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError("shard-index is outside the configured shard range")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    projects_full = pd.read_csv(args.projects)
    strict_full = pd.read_csv(args.strict_panel)
    assignment = pd.read_csv(args.assignment)
    required_assignment = {"work_id", "shard_index"}
    missing = sorted(required_assignment - set(assignment.columns))
    if missing:
        raise ValueError(f"Shard assignment is missing columns: {missing}")
    if assignment["work_id"].duplicated().any():
        raise ValueError("Shard assignment contains duplicate work_id values")
    full_ids = set(projects_full["work_id"].astype(str))
    assignment_ids = set(assignment["work_id"].astype(str))
    if full_ids != assignment_ids:
        raise ValueError("Shard assignment does not exactly cover the project input")

    shard_ids = set(
        assignment.loc[
            pd.to_numeric(assignment["shard_index"], errors="raise").eq(args.shard_index),
            "work_id",
        ].astype(str)
    )
    if not shard_ids:
        raise ValueError(f"Shard {args.shard_index} is empty")
    projects = (
        projects_full[projects_full["work_id"].astype(str).isin(shard_ids)]
        .sort_values("work_id")
        .reset_index(drop=True)
    )
    if set(projects["work_id"].astype(str)) != shard_ids:
        raise ValueError("Project file does not cover the assigned shard")
    strict = strict_full[strict_full["work_id"].astype(str).isin(shard_ids)].copy()
    if set(strict["work_id"].astype(str).unique()) != shard_ids:
        raise ValueError("Strict panel does not cover the assigned shard")

    audit = audit_full_project_input(projects)
    config = {
        "shard_index": int(args.shard_index),
        "shard_count": int(args.shard_count),
        "projects": int(len(projects)),
        "work_id_min": str(projects["work_id"].min()),
        "work_id_max": str(projects["work_id"].max()),
        "field_fetch_top_n": int(args.field_fetch_top_n),
        "main_field_top_n": int(args.main_field_top_n),
        "sensitivity_field_top_n": int(args.sensitivity_field_top_n),
        "university_min_works": int(args.university_min_works),
        "api_key_present": bool(os.getenv("OPENALEX_API_KEY", "").strip()),
        "input_audit": audit,
    }
    _write_json(args.output_dir / "shard_config.json", config)
    projects.to_csv(args.output_dir / "shard_projects.csv", index=False)
    strict.to_csv(args.output_dir / "shard_strict_panel.csv", index=False)

    manifests = build_extraction_manifests(projects)
    base._write_manifests(manifests, args.output_dir / "manifests")
    detail_cache = base._prepare_project_detail_cache(
        args.cache_dir, 20260811 + args.shard_index
    )
    client = OpenAlexClient()
    details = base._load_or_fetch_details(client, projects, detail_cache, args.output_dir)
    if details["actual_company_count"].lt(1).any():
        bad = details.loc[details["actual_company_count"].lt(1), "work_id"].head().tolist()
        raise ValueError(f"Shard contains projects without realized companies: {bad}")

    university_manifests = build_university_manifests(projects, details)
    base._write_manifests(university_manifests, args.output_dir / "manifests")
    author_history = base._load_or_build_csv(
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
    university_history = base._load_or_build_csv(
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
    field_counts = base._load_or_build_csv(
        args.output_dir / "subfield_exact_date_company_counts.csv",
        lambda: base.add_deterministic_field_rank(
            fetch_exact_field_counts(
                client,
                manifests["subfield_date_units"],
                args.cache_dir,
                top_n=args.field_fetch_top_n,
            )
        ),
        label="field counts",
    )
    field_counts = base.add_deterministic_field_rank(field_counts)
    field_counts.to_csv(args.output_dir / "subfield_exact_date_company_counts.csv", index=False)

    candidate_tables: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for label, top_n in [
        ("top50", args.main_field_top_n),
        ("top100", args.sensitivity_field_top_n),
    ]:
        natural, natural_audit = assemble_pilot_candidates(
            projects,
            details,
            author_history,
            university_history,
            field_counts,
            field_top_n=top_n,
            university_min_works=args.university_min_works,
        )
        natural_audit.to_csv(
            args.output_dir / f"natural_candidate_audit_{label}.csv", index=False
        )
        estimation = inject_realized_firms(natural, projects, details)
        estimation = enrich_candidate_relationships(
            estimation,
            projects,
            details,
            author_history,
            university_history,
            university_min_works=args.university_min_works,
        )
        candidate_tables[label] = (natural, estimation)

    union_candidates = pd.concat(
        [candidate_tables["top50"][1], candidate_tables["top100"][1]],
        ignore_index=True,
    ).drop_duplicates(["work_id", "company_id"])
    text_path = args.output_dir / "candidate_firm_prepublication_text_history.csv"
    text_history = base._load_or_build_csv(
        text_path,
        lambda: fetch_candidate_firm_text_history(
            client,
            projects,
            union_candidates,
            args.cache_dir,
            batch_size=args.cognitive_batch_size,
        ),
        label="candidate text history",
    )
    if not text_history.empty:
        observed = set(text_history["focal_work_id"].astype(str))
        if not observed.issubset(shard_ids):
            raise ValueError("Candidate text checkpoint contains projects outside the shard")

    project_texts = base._project_texts(projects, details)
    cognitive_union = base.compute_project_firm_cognitive_fit(
        union_candidates[["work_id", "company_id"]],
        project_texts[["work_id", "publication_date", "project_text"]],
        text_history,
    )
    cognitive_frames: dict[str, pd.DataFrame] = {}
    metric_frames: dict[str, pd.DataFrame] = {}
    for label in ["top50", "top100"]:
        _, estimation = candidate_tables[label]
        cognitive = base._merge_cognitive_fit(estimation, cognitive_union, project_texts)
        cognitive.to_csv(
            args.output_dir / f"cognitive_candidate_long_{label}.csv", index=False
        )
        metrics = build_project_realization_metrics(cognitive, conflict_threshold=0.05)
        metrics.to_csv(
            args.output_dir / f"project_realization_metrics_{label}.csv", index=False
        )
        cognitive_frames[label] = cognitive
        metric_frames[label] = metrics

    time_audit = base._validate_time_provenance(
        [cognitive_frames["top50"], cognitive_frames["top100"]],
        text_history,
        projects,
    )
    _write_json(args.output_dir / "shard_time_provenance_audit.json", time_audit)

    summary = {
        "status": "SHARD_SOFTWARE_COMPLETE",
        "shard_index": int(args.shard_index),
        "projects": int(len(projects)),
        "top50_candidate_rows": int(len(cognitive_frames["top50"])),
        "top100_candidate_rows": int(len(cognitive_frames["top100"])),
        "text_history_rows": int(len(text_history)),
        "time_provenance": time_audit,
        "output_contract_complete": True,
    }
    _write_json(args.output_dir / "shard_summary.json", summary)

    missing_outputs = [
        name for name in required_shard_output_filenames()
        if not (args.output_dir / name).exists()
    ]
    if missing_outputs:
        raise ValueError(f"Shard did not produce required outputs: {missing_outputs}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
