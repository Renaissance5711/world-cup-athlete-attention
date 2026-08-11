#!/usr/bin/env python3
"""Run Stage A and 400-project Stage B COMPOT validation without OpenAlex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage2_compot_conflict import summarize_compot_conflict
from stage2_compot_models import fit_compot_temporal_ranking_models
from stage2_compot_stagea import (
    build_stagea_descriptives,
    build_stagea_project_panel,
    fit_stagea_compot_models,
)


def required_compot_output_filenames() -> list[str]:
    return [
        "stageA_project_panel.csv",
        "stageA_sample_audit.json",
        "stageA_compot_descriptives.csv",
        "stageA_compot_models.csv",
        "stageA_summary.json",
        "pilot_compot_ranking_coefficients_top50.csv",
        "pilot_compot_ranking_metrics_top50.csv",
        "pilot_compot_conflict_by_quartile_top50.csv",
        "pilot_compot_ranking_coefficients_top100.csv",
        "pilot_compot_ranking_metrics_top100.csv",
        "pilot_compot_conflict_by_quartile_top100.csv",
        "pilot_compot_validation_summary.json",
    ]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    temporary.replace(path)


def _model_term(models: pd.DataFrame, model: str, term: str) -> dict[str, Any]:
    subset = models[(models["model"] == model) & (models["term"] == term)]
    if subset.empty:
        return {"included": False, "coefficient": None, "standard_error": None, "p_value": None}
    row = subset.iloc[0]
    return {
        "included": True,
        "coefficient": float(row["coefficient"]),
        "standard_error": float(row["standard_error"]) if pd.notna(row["standard_error"]) else None,
        "p_value": float(row["p_value"]) if "p_value" in row.index and pd.notna(row["p_value"]) else None,
    }


def _aggregate_metric(metrics: pd.DataFrame, model: str) -> dict[str, float]:
    row = metrics[(metrics["model"] == model) & metrics["is_aggregate"].astype(bool)]
    if len(row) != 1:
        raise ValueError(f"Expected one aggregate metric row for {model}, found {len(row)}")
    item = row.iloc[0]
    return {
        "mrr": float(item["mean_reciprocal_rank"]),
        "recall_at_5": float(item["mean_recall_at_5"]),
        "recall_at_10": float(item["mean_recall_at_10"]),
        "average_precision": float(item["mean_average_precision"]),
        "selected_best_rank": float(item["mean_selected_best_rank"]),
    }


def _interaction_signal(
    coefficient_frames: dict[str, pd.DataFrame],
    model: str,
    terms: list[str],
) -> dict[str, Any]:
    details: dict[str, Any] = {}
    stable_significant_terms: list[str] = []
    for term in terms:
        values = []
        for label in ("top50", "top100"):
            frame = coefficient_frames[label]
            subset = frame[(frame["model"] == model) & (frame["term"] == term)]
            if subset.empty or not bool(subset.iloc[0]["included"]):
                values.append({"label": label, "included": False})
                continue
            row = subset.iloc[0]
            coefficient = float(row["coefficient"])
            se = float(row["standard_error"]) if pd.notna(row["standard_error"]) else np.nan
            z = abs(coefficient / se) if np.isfinite(se) and se > 0 else np.nan
            values.append({
                "label": label,
                "included": True,
                "coefficient": coefficient,
                "standard_error": se if np.isfinite(se) else None,
                "abs_z": float(z) if np.isfinite(z) else None,
            })
        details[term] = values
        included = [value for value in values if value.get("included")]
        if len(included) == 2:
            coefficients = [float(value["coefficient"]) for value in included]
            signs_match = np.sign(coefficients[0]) == np.sign(coefficients[1]) and coefficients[0] != 0
            significant = all(
                value.get("abs_z") is not None and float(value["abs_z"]) >= 1.96
                for value in included
            )
            if signs_match and significant:
                stable_significant_terms.append(term)
    return {
        "supported": bool(stable_significant_terms),
        "stable_significant_terms": stable_significant_terms,
        "details": details,
    }


def _stagea_summary(panel: pd.DataFrame, descriptives: pd.DataFrame, models: pd.DataFrame) -> dict[str, Any]:
    a1 = _model_term(models, "A1_lpm", "compot_z")
    a2 = _model_term(models, "A2_logit", "compot_z")
    signs = [
        np.sign(item["coefficient"])
        for item in (a1, a2)
        if item["coefficient"] is not None and item["coefficient"] != 0
    ]
    same_direction = len(signs) == 2 and signs[0] == signs[1]
    significant = any(
        item["p_value"] is not None and item["p_value"] < 0.05
        for item in (a1, a2)
    )
    supported = bool(same_direction and significant)
    return {
        "conditional_population": "V3_STRICT_ELIGIBLE_PROJECTS",
        "projects": int(len(panel)),
        "firm_participation_projects": int(panel["firm_participation"].sum()),
        "mean_compot_firm_participation_1": float(
            panel.loc[panel["firm_participation"].eq(1), "compot"].mean()
        ),
        "mean_compot_firm_participation_0": float(
            panel.loc[panel["firm_participation"].eq(0), "compot"].mean()
        ),
        "quartile_firm_participation_rates": {
            str(int(row["compot_quartile"])): float(row["firm_participation_rate"])
            for _, row in descriptives.iterrows()
        },
        "A1_compot_z": a1,
        "A2_compot_z": a2,
        "same_direction_A1_A2": same_direction,
        "stageA_substantive_signal": supported,
        "interpretation_boundary": "Associational within the V3 strict-eligible project sample; not all scientific projects.",
    }


def _validate_candidate_panel(frame: pd.DataFrame, label: str) -> set[str]:
    required = {"work_id", "company_id", "selected", "compot", "publication_year"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{label} candidate file is missing columns: {missing}")
    if frame["compot"].isna().any():
        raise ValueError(f"{label} has missing COMPOT values")
    if frame.groupby(["work_id", "company_id"], observed=True).size().gt(1).any():
        raise ValueError(f"{label} contains duplicate project-company rows")
    projects = set(frame["work_id"].astype(str).unique())
    if len(projects) != 400:
        raise ValueError(f"{label} expected 400 pilot projects but found {len(projects)}")
    compot_nunique = frame.groupby("work_id", observed=True)["compot"].nunique(dropna=False)
    if compot_nunique.gt(1).any():
        raise ValueError(f"{label} has within-project COMPOT disagreement")
    return projects


def _classification(stagea: bool, relationship: bool, fit: bool) -> str:
    if stagea and relationship and fit:
        return "STAGE_A_PLUS_BOTH_MODERATORS"
    if stagea and relationship:
        return "STAGE_A_PLUS_RELATIONSHIP_MODERATOR"
    if stagea and fit:
        return "STAGE_A_PLUS_FIT_MODERATOR"
    if stagea:
        return "STAGE_A_ONLY"
    if not relationship and not fit:
        return "NEITHER_STAGE_A_NOR_STAGE_B_SUBSTANTIVE_SIGNAL"
    return "STAGE_B_ONLY_SIGNAL_REQUIRES_THEORY_REVIEW"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict-panel", type=Path, required=True)
    parser.add_argument("--projects", type=Path, required=True)
    parser.add_argument("--top50", type=Path, required=True)
    parser.add_argument("--top100", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    strict = pd.read_csv(args.strict_panel)
    projects = pd.read_csv(args.projects)
    stagea_panel, audit = build_stagea_project_panel(strict)
    project_universe = set(projects["work_id"].astype(str).unique())
    if len(project_universe) != 1881:
        raise ValueError(f"Stage 2 project universe expected 1881 projects but found {len(project_universe)}")
    positive_ids = set(
        stagea_panel.loc[stagea_panel["firm_participation"].eq(1), "work_id"].astype(str)
    )
    if positive_ids != project_universe:
        raise ValueError("Stage A firm_participation positives do not exactly match the 1,881-project Stage 2 universe")

    stagea_descriptives = build_stagea_descriptives(stagea_panel)
    stagea_models = fit_stagea_compot_models(stagea_panel)
    stagea_summary = _stagea_summary(stagea_panel, stagea_descriptives, stagea_models)
    stagea_panel.to_csv(args.output_dir / "stageA_project_panel.csv", index=False)
    _write_json(args.output_dir / "stageA_sample_audit.json", audit)
    stagea_descriptives.to_csv(args.output_dir / "stageA_compot_descriptives.csv", index=False)
    stagea_models.to_csv(args.output_dir / "stageA_compot_models.csv", index=False)
    _write_json(args.output_dir / "stageA_summary.json", stagea_summary)

    candidates = {
        "top50": pd.read_csv(args.top50),
        "top100": pd.read_csv(args.top100),
    }
    project_sets = {
        label: _validate_candidate_panel(frame, label)
        for label, frame in candidates.items()
    }
    if project_sets["top50"] != project_sets["top100"]:
        raise ValueError("Top50 and Top100 COMPOT validation files cover different pilot projects")

    coefficient_frames: dict[str, pd.DataFrame] = {}
    metric_frames: dict[str, pd.DataFrame] = {}
    conflict_frames: dict[str, pd.DataFrame] = {}
    metric_summary: dict[str, Any] = {}
    for label, frame in candidates.items():
        coefficients, metrics = fit_compot_temporal_ranking_models(
            frame, train_end_year=2018, bootstrap_reps=args.bootstrap_reps
        )
        conflict = summarize_compot_conflict(frame)
        coefficients.to_csv(
            args.output_dir / f"pilot_compot_ranking_coefficients_{label}.csv", index=False
        )
        metrics.to_csv(
            args.output_dir / f"pilot_compot_ranking_metrics_{label}.csv", index=False
        )
        conflict.to_csv(
            args.output_dir / f"pilot_compot_conflict_by_quartile_{label}.csv", index=False
        )
        coefficient_frames[label] = coefficients
        metric_frames[label] = metrics
        conflict_frames[label] = conflict
        b0 = _aggregate_metric(metrics, "B0_combined")
        b1 = _aggregate_metric(metrics, "B1_relationship_compot")
        b2 = _aggregate_metric(metrics, "B2_fit_compot")
        metric_summary[label] = {
            "B0": b0,
            "B1": b1,
            "B2": b2,
            "B1_minus_B0_mrr": b1["mrr"] - b0["mrr"],
            "B1_minus_B0_recall_at_10": b1["recall_at_10"] - b0["recall_at_10"],
            "B2_minus_B1_mrr": b2["mrr"] - b1["mrr"],
            "B2_minus_B1_recall_at_10": b2["recall_at_10"] - b1["recall_at_10"],
        }

    relationship_signal = _interaction_signal(
        coefficient_frames,
        "B1_relationship_compot",
        ["author_x_compot", "university_x_compot", "strong_university_x_compot"],
    )
    fit_signal = _interaction_signal(
        coefficient_frames, "B2_fit_compot", ["fit_x_compot"]
    )
    stagea_signal = bool(stagea_summary["stageA_substantive_signal"])
    classification = _classification(
        stagea_signal, bool(relationship_signal["supported"]), bool(fit_signal["supported"])
    )
    summary = {
        "status": "COMPOT_VALIDATION_SOFTWARE_COMPLETE",
        "pilot_projects": 400,
        "stageA": stagea_summary,
        "ranking": metric_summary,
        "relationship_compot_signal": relationship_signal,
        "fit_compot_signal": fit_signal,
        "classification": classification,
        "top50_top100_project_sets_match": True,
        "null_results_allowed": True,
        "ready_for_specification_review": True,
        "openalex_refetch_required": False,
    }
    _write_json(args.output_dir / "pilot_compot_validation_summary.json", summary)

    missing_outputs = [
        name for name in required_compot_output_filenames()
        if not (args.output_dir / name).exists()
    ]
    if missing_outputs:
        raise ValueError(f"COMPOT validation did not produce required outputs: {missing_outputs}")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
