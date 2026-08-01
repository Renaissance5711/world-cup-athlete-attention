from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .near_miss import fit_near_miss_models, fit_overlap_weights, prepare_near_miss_sample
from .shot_workflow import build_player_match_shot_panel


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value)!r}")


def run_near_miss_workflow(
    *,
    workspace: Path,
    output_dir: Path,
) -> dict[str, Path]:
    """Build the shot-opportunity panel and run primary/sensitivity models."""

    workspace = Path(workspace)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = workspace / "base_archive" / "data"

    panel, reconciliation = build_player_match_shot_panel(
        statsbomb_root=workspace / "data" / "raw" / "statsbomb" / "combined",
        match_outcomes=pd.read_csv(
            base / "processed" / "all_player_match_outcomes_2022.csv"
        ),
        fjelstul_matches=pd.read_csv(base / "raw" / "fjelstul" / "matches.csv"),
        appearances=pd.read_csv(
            base / "raw" / "fjelstul" / "player_appearances.csv"
        ),
        goals=pd.read_csv(base / "raw" / "fjelstul" / "goals.csv"),
    )

    paths: dict[str, Path] = {}
    paths["player_match_shot_panel"] = output_dir / "player_match_shot_panel.csv"
    panel.to_csv(paths["player_match_shot_panel"], index=False)

    paths["retrieval_audit"] = output_dir / "statsbomb_fjelstul_reconciliation.json"
    paths["retrieval_audit"].write_text(
        json.dumps(reconciliation, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )

    primary = prepare_near_miss_sample(
        panel,
        treatment_column="fjelstul_non_penalty_scorer",
        exclude_penalty_attempts=True,
    )
    primary_weighted, primary_balance, primary_diagnostics = fit_overlap_weights(
        primary
    )
    treated_n = int(primary_diagnostics["treated_observations"])
    if primary_diagnostics["max_absolute_smd_weighted"] >= 0.10:
        raise RuntimeError("Near-miss design failed the 0.10 balance gate")
    minimum_ess = 0.5 * treated_n
    if min(
        primary_diagnostics["treated_effective_sample_size"],
        primary_diagnostics["control_effective_sample_size"],
    ) < minimum_ess:
        raise RuntimeError("Near-miss design failed the effective-sample-size gate")
    primary_results = fit_near_miss_models(primary_weighted)

    paths["primary_sample"] = output_dir / "near_miss_primary_sample.csv"
    paths["primary_balance"] = output_dir / "near_miss_primary_balance.csv"
    paths["primary_diagnostics"] = output_dir / "near_miss_primary_diagnostics.json"
    paths["primary_results"] = output_dir / "near_miss_primary_results.csv"
    primary_weighted.to_csv(paths["primary_sample"], index=False)
    primary_balance.to_csv(paths["primary_balance"], index=False)
    paths["primary_diagnostics"].write_text(
        json.dumps(primary_diagnostics, indent=2, default=_json_default),
        encoding="utf-8",
    )
    primary_results.to_csv(paths["primary_results"], index=False)

    sensitivity_specs = [
        (
            "primary_fjelstul_exclude_penalties",
            "fjelstul_non_penalty_scorer",
            True,
        ),
        (
            "statsbomb_exclude_penalties",
            "statsbomb_non_penalty_scorer",
            True,
        ),
        (
            "fjelstul_include_penalties",
            "fjelstul_non_penalty_scorer",
            False,
        ),
    ]
    sensitivity_frames: list[pd.DataFrame] = []
    sensitivity_diagnostics: dict[str, Any] = {}
    for specification, treatment_column, exclude_penalties in sensitivity_specs:
        sample = prepare_near_miss_sample(
            panel,
            treatment_column=treatment_column,
            exclude_penalty_attempts=exclude_penalties,
        )
        weighted, balance, diagnostics = fit_overlap_weights(sample)
        results = fit_near_miss_models(weighted)
        results.insert(0, "specification", specification)
        sensitivity_frames.append(results)
        sensitivity_diagnostics[specification] = {
            **diagnostics,
            "max_absolute_smd_weighted": float(balance["smd_weighted"].abs().max()),
            "treatment_column": treatment_column,
            "exclude_penalty_attempts": exclude_penalties,
        }

    paths["sensitivity_results"] = output_dir / "near_miss_sensitivity_results.csv"
    pd.concat(sensitivity_frames, ignore_index=True).to_csv(
        paths["sensitivity_results"], index=False
    )
    sensitivity_path = output_dir / "near_miss_sensitivity_diagnostics.json"
    sensitivity_path.write_text(
        json.dumps(sensitivity_diagnostics, indent=2, default=_json_default),
        encoding="utf-8",
    )
    paths["sensitivity_diagnostics"] = sensitivity_path
    return paths
