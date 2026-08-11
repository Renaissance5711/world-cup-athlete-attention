#!/usr/bin/env python3
"""Merge five extraction shards and estimate the global 1,881-project TEM models."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stage2_compot_conflict import summarize_compot_conflict
from stage2_compot_models import fit_compot_temporal_ranking_models
from stage2_compot_stagea import (
    build_stagea_descriptives,
    build_stagea_project_panel,
    fit_stagea_compot_models,
)
from stage2_pilot_metrics import build_project_realization_metrics, evaluate_pilot_gates
from stage2_pilot_models import fit_temporal_ranking_models


def required_full_output_filenames() -> list[str]:
    return [
        "full_cognitive_candidate_long_top50.csv",
        "full_cognitive_candidate_long_top100.csv",
        "full_project_realization_metrics_top50.csv",
        "full_project_realization_metrics_top100.csv",
        "full_ranking_coefficients_top50.csv",
        "full_ranking_metrics_top50.csv",
        "full_ranking_coefficients_top100.csv",
        "full_ranking_metrics_top100.csv",
        "full_gate_decision_top50.json",
        "full_gate_decision_top100.json",
        "full_compot_ranking_coefficients_top50.csv",
        "full_compot_ranking_metrics_top50.csv",
        "full_compot_ranking_coefficients_top100.csv",
        "full_compot_ranking_metrics_top100.csv",
        "full_compot_conflict_by_quartile_top50.csv",
        "full_compot_conflict_by_quartile_top100.csv",
        "full_stageA_project_panel.csv",
        "full_stageA_compot_descriptives.csv",
        "full_stageA_compot_models.csv",
        "full_stageA_audit.json",
        "full_time_provenance_audit.json",
        "full_stage2_summary.json",
    ]


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def merge_shard_candidate_frames(
    frames: list[pd.DataFrame],
    *,
    expected_projects: int | None = 1881,
) -> pd.DataFrame:
    """Merge candidate frames with strict project-company uniqueness checks."""
    if not frames:
        raise ValueError("No shard candidate frames were provided")
    merged = pd.concat(frames, ignore_index=True)
    required = {"work_id", "company_id"}
    missing = sorted(required - set(merged.columns))
    if missing:
        raise ValueError(f"Shard candidate frames are missing columns: {missing}")
    if merged.duplicated(["work_id", "company_id"]).any():
        duplicate = merged.loc[
            merged.duplicated(["work_id", "company_id"], keep=False),
            ["work_id", "company_id"],
        ].head().to_dict("records")
        raise ValueError(f"Duplicate project-company rows after shard merge: {duplicate}")
    projects = int(merged["work_id"].astype(str).nunique())
    if expected_projects is not None and projects != expected_projects:
        raise ValueError(
            f"Merged full candidate panel expected {expected_projects} projects but found {projects}"
        )
    return merged.sort_values(["work_id", "company_id"]).reset_index(drop=True)


def _aggregate_row(metrics: pd.DataFrame, model: str) -> dict[str, float]:
    subset = metrics[(metrics["model"] == model) & metrics["is_aggregate"].astype(bool)]
    if len(subset) != 1:
        raise ValueError(f"Expected one aggregate row for {model}; found {len(subset)}")
    row = subset.iloc[0]
    return {
        "mrr": float(row["mean_reciprocal_rank"]),
        "recall_at_5": float(row["mean_recall_at_5"]),
        "recall_at_10": float(row["mean_recall_at_10"]),
        "average_precision": float(row["mean_average_precision"]),
        "selected_best_rank": float(row["mean_selected_best_rank"]),
    }


def _stagea_term(models: pd.DataFrame, model: str, term: str) -> dict[str, Any]:
    subset = models[(models["model"] == model) & (models["term"] == term)]
    if len(subset) != 1:
        return {"coefficient": None, "standard_error": None, "p_value": None}
    row = subset.iloc[0]
    return {
        "coefficient": float(row["coefficient"]),
        "standard_error": float(row["standard_error"]),
        "p_value": float(row["p_value"]),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-dir", type=Path, action="append", required=True)
    parser.add_argument("--strict-panel", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=1000)
    parser.add_argument("--expected-projects", type=int, default=1881)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if len(args.shard_dir) != 5:
        raise ValueError(f"Expected five shard directories but received {len(args.shard_dir)}")

    summaries: list[dict[str, Any]] = []
    shard_indices: set[int] = set()
    provenance: list[dict[str, Any]] = []
    candidate_frames: dict[str, list[pd.DataFrame]] = {"top50": [], "top100": []}
    project_sets: list[set[str]] = []

    for directory in args.shard_dir:
        summary_path = directory / "shard_summary.json"
        if not summary_path.exists():
            raise ValueError(f"Missing shard summary: {summary_path}")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "SHARD_SOFTWARE_COMPLETE":
            raise ValueError(f"Shard did not complete successfully: {directory}")
        shard_index = int(summary["shard_index"])
        if shard_index in shard_indices:
            raise ValueError(f"Duplicate shard index: {shard_index}")
        shard_indices.add(shard_index)
        summaries.append(summary)

        shard_projects = pd.read_csv(directory / "shard_projects.csv")
        ids = set(shard_projects["work_id"].astype(str))
        for previous in project_sets:
            overlap = ids & previous
            if overlap:
                raise ValueError(f"Shard project sets overlap: {sorted(overlap)[:5]}")
        project_sets.append(ids)
        for label in ["top50", "top100"]:
            frame = pd.read_csv(directory / f"cognitive_candidate_long_{label}.csv")
            observed = set(frame["work_id"].astype(str).unique())
            if observed != ids:
                raise ValueError(f"{label} candidate rows do not match shard project IDs")
            candidate_frames[label].append(frame)
        audit = json.loads(
            (directory / "shard_time_provenance_audit.json").read_text(encoding="utf-8")
        )
        if int(audit.get("violations", -1)) != 0 or not audit.get("strictly_prepublication"):
            raise ValueError(f"Shard provenance audit failed: shard {shard_index}")
        provenance.append(audit)

    if shard_indices != set(range(5)):
        raise ValueError(f"Expected shard indices 0..4; found {sorted(shard_indices)}")
    union_projects = set().union(*project_sets)
    if len(union_projects) != args.expected_projects:
        raise ValueError(
            f"Merged shard project union expected {args.expected_projects} but found {len(union_projects)}"
        )

    full_frames: dict[str, pd.DataFrame] = {}
    primary_metrics: dict[str, pd.DataFrame] = {}
    primary_ranking: dict[str, pd.DataFrame] = {}
    primary_coefficients: dict[str, pd.DataFrame] = {}
    gates: dict[str, dict[str, Any]] = {}
    compot_metrics: dict[str, pd.DataFrame] = {}

    for label in ["top50", "top100"]:
        full = merge_shard_candidate_frames(
            candidate_frames[label], expected_projects=args.expected_projects
        )
        full.to_csv(
            args.output_dir / f"full_cognitive_candidate_long_{label}.csv", index=False
        )
        metrics = build_project_realization_metrics(full, conflict_threshold=0.05)
        metrics.to_csv(
            args.output_dir / f"full_project_realization_metrics_{label}.csv", index=False
        )
        coefficients, ranking = fit_temporal_ranking_models(
            full, train_end_year=2018, bootstrap_reps=args.bootstrap_reps
        )
        coefficients.to_csv(
            args.output_dir / f"full_ranking_coefficients_{label}.csv", index=False
        )
        ranking.to_csv(
            args.output_dir / f"full_ranking_metrics_{label}.csv", index=False
        )
        gate = evaluate_pilot_gates(full, ranking, metrics)
        _write_json(args.output_dir / f"full_gate_decision_{label}.json", gate)

        compot_coeff, compot_rank = fit_compot_temporal_ranking_models(
            full, train_end_year=2018, bootstrap_reps=args.bootstrap_reps
        )
        compot_coeff.to_csv(
            args.output_dir / f"full_compot_ranking_coefficients_{label}.csv", index=False
        )
        compot_rank.to_csv(
            args.output_dir / f"full_compot_ranking_metrics_{label}.csv", index=False
        )
        conflict = summarize_compot_conflict(full)
        conflict.to_csv(
            args.output_dir / f"full_compot_conflict_by_quartile_{label}.csv", index=False
        )

        full_frames[label] = full
        primary_metrics[label] = metrics
        primary_ranking[label] = ranking
        primary_coefficients[label] = coefficients
        gates[label] = gate
        compot_metrics[label] = compot_rank

    if set(full_frames["top50"]["work_id"].astype(str).unique()) != set(
        full_frames["top100"]["work_id"].astype(str).unique()
    ):
        raise ValueError("Top50 and Top100 full panels cover different project sets")

    strict = pd.read_csv(args.strict_panel)
    stagea_panel, stagea_audit = build_stagea_project_panel(strict)
    stagea_desc = build_stagea_descriptives(stagea_panel)
    stagea_models = fit_stagea_compot_models(stagea_panel)
    stagea_panel.to_csv(args.output_dir / "full_stageA_project_panel.csv", index=False)
    stagea_desc.to_csv(args.output_dir / "full_stageA_compot_descriptives.csv", index=False)
    stagea_models.to_csv(args.output_dir / "full_stageA_compot_models.csv", index=False)
    _write_json(args.output_dir / "full_stageA_audit.json", stagea_audit)

    provenance_summary = {
        "shards": 5,
        "violations": int(sum(int(item.get("violations", 0)) for item in provenance)),
        "candidate_rows_checked": int(
            sum(int(item.get("candidate_rows_checked", 0)) for item in provenance)
        ),
        "text_history_rows_checked": int(
            sum(int(item.get("text_history_rows_checked", 0)) for item in provenance)
        ),
        "strictly_prepublication": all(
            bool(item.get("strictly_prepublication")) for item in provenance
        ),
        "all_shards_auditable": all(bool(item.get("auditable")) for item in provenance),
    }
    if provenance_summary["violations"] != 0 or not provenance_summary["strictly_prepublication"]:
        raise ValueError(f"Global provenance audit failed: {provenance_summary}")
    _write_json(args.output_dir / "full_time_provenance_audit.json", provenance_summary)

    ranking_summary: dict[str, Any] = {}
    for label in ["top50", "top100"]:
        baseline = {
            model: _aggregate_row(primary_ranking[label], model)
            for model in ["technical", "relational", "combined"]
        }
        b0 = _aggregate_row(compot_metrics[label], "B0_combined")
        b1 = _aggregate_row(compot_metrics[label], "B1_relationship_compot")
        b2 = _aggregate_row(compot_metrics[label], "B2_fit_compot")
        ranking_summary[label] = {
            "primary": baseline,
            "combined_minus_technical_mrr": (
                baseline["combined"]["mrr"] - baseline["technical"]["mrr"]
            ),
            "combined_minus_technical_recall_at_10": (
                baseline["combined"]["recall_at_10"] - baseline["technical"]["recall_at_10"]
            ),
            "compot_secondary": {"B0": b0, "B1": b1, "B2": b2},
        }

    summary = {
        "status": "FULL_STAGE2_SOFTWARE_COMPLETE",
        "projects": int(args.expected_projects),
        "shard_projects": {
            str(int(item["shard_index"])): int(item["projects"]) for item in summaries
        },
        "stageA": {
            "projects": int(stagea_audit["projects"]),
            "firm_participation_projects": int(stagea_audit["positive_projects"]),
            "A1_compot_z": _stagea_term(stagea_models, "A1_lpm", "compot_z"),
            "A2_compot_z": _stagea_term(stagea_models, "A2_logit", "compot_z"),
            "frozen_role": "STAGE_A_PRIMARY_STAGE_B_COMPOT_SECONDARY_ONLY",
        },
        "ranking": ranking_summary,
        "top50_gate": gates["top50"],
        "top100_gate": gates["top100"],
        "directionally_stable": (
            gates["top50"]["recommendation"] == gates["top100"]["recommendation"]
        ),
        "time_provenance": provenance_summary,
        "primary_stageB_model": "combined",
        "compot_stageB_role": "prespecified_secondary_robustness",
    }
    _write_json(args.output_dir / "full_stage2_summary.json", summary)

    missing_outputs = [
        name for name in required_full_output_filenames()
        if not (args.output_dir / name).exists()
    ]
    if missing_outputs:
        raise ValueError(f"Full merge did not produce required outputs: {missing_outputs}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
