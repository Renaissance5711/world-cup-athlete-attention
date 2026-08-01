from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .shot_panel import aggregate_player_match_shots, summarize_match_events
from .statsbomb_mapping import (
    build_match_mapping,
    map_lineup_players,
    summarize_lineup_participation,
)


SHOT_NUMERIC_COLUMNS = [
    "shots",
    "non_penalty_shots",
    "total_xg",
    "npxg",
    "max_shot_xg",
    "max_npxg",
    "regulation_goals",
    "non_penalty_goals",
    "penalty_attempts",
    "shots_on_target",
    "open_play_shots",
    "under_pressure_shots",
    "scored_any_goal",
    "scored_non_penalty_goal",
]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _match_end_seconds(events: list[dict]) -> int:
    eligible = [
        int(event.get("minute", 0) or 0) * 60 + int(event.get("second", 0) or 0)
        for event in events
        if int(event.get("period", 0) or 0) <= 4
    ]
    return max(eligible, default=90 * 60)


def build_player_match_shot_panel(
    *,
    statsbomb_root: Path,
    match_outcomes: pd.DataFrame,
    fjelstul_matches: pd.DataFrame,
    appearances: pd.DataFrame,
    goals: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a player-match panel with StatsBomb opportunity covariates.

    Fjelstul remains authoritative for scorer treatment so the robustness
    analysis uses the same goal coding as the manuscript. StatsBomb supplies
    shot quality, shot context, and a sensitivity treatment indicator.
    """

    statsbomb_matches = _load_json(statsbomb_root / "matches_43_106.json")
    match_mapping = build_match_mapping(statsbomb_matches, fjelstul_matches)
    if not match_mapping["mapping_status"].eq("mapped").all():
        missing = match_mapping.loc[
            match_mapping["mapping_status"].ne("mapped"), "sb_match_id"
        ].tolist()
        raise ValueError(f"Unmapped StatsBomb matches: {missing}")

    lineups: dict[int, list[dict]] = {}
    shot_rows: list[dict] = []
    participation_frames: list[pd.DataFrame] = []
    for match in statsbomb_matches:
        sb_match_id = int(match["match_id"])
        teams = _load_json(statsbomb_root / "lineups" / f"{sb_match_id}.json")
        events = _load_json(statsbomb_root / "events" / f"{sb_match_id}.json")
        lineups[sb_match_id] = teams
        shot_rows.extend(
            summarize_match_events(
                events,
                sb_match_id,
                match["home_team"]["home_team_name"],
                match["away_team"]["away_team_name"],
            )
        )
        participation_frames.append(
            summarize_lineup_participation(
                teams,
                sb_match_id,
                _match_end_seconds(events),
            )
        )

    lineup_mapping, lineup_audit = map_lineup_players(
        lineups, match_mapping, appearances
    )
    active_audit = lineup_audit.loc[lineup_audit["has_position_record"].eq(1)]
    if not active_audit["mapping_status"].eq("mapped").all():
        failed = active_audit.loc[
            active_audit["mapping_status"].ne("mapped")
        ].to_dict("records")
        raise ValueError(f"Unmapped active lineup players: {failed[:5]}")

    player_lookup = lineup_mapping[
        ["sb_match_id", "team_name", "sb_player_id", "player_id"]
    ].copy()

    participation = pd.concat(participation_frames, ignore_index=True)
    participation = participation.merge(
        player_lookup,
        on=["sb_match_id", "team_name", "sb_player_id"],
        how="left",
        validate="one_to_one",
    )
    if participation["player_id"].isna().any():
        raise ValueError("Participation records contain unmapped players")

    shots = pd.DataFrame(shot_rows)
    shot_aggregate = aggregate_player_match_shots(shots)
    shot_aggregate = shot_aggregate.merge(
        player_lookup,
        on=["sb_match_id", "team_name", "sb_player_id"],
        how="left",
        validate="one_to_one",
    )
    if shot_aggregate["player_id"].isna().any():
        raise ValueError("Shot records contain unmapped players")

    shot_aggregate = shot_aggregate.merge(
        match_mapping[["sb_match_id", "match_id"]],
        on="sb_match_id",
        how="left",
        validate="many_to_one",
    )

    panel = match_outcomes.merge(
        match_mapping[["sb_match_id", "match_id"]],
        on="match_id",
        how="left",
        validate="many_to_one",
    )
    panel = panel.merge(
        participation[
            [
                "sb_match_id",
                "player_id",
                "minutes_played",
                "sb_starter",
                "first_position",
            ]
        ],
        on=["sb_match_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    if panel["minutes_played"].isna().any():
        raise ValueError("Player-match outcomes lack participation minutes")

    aggregate_columns = [
        column
        for column in shot_aggregate.columns
        if column
        not in {
            "team_name",
            "sb_player_name",
            "match_id",
        }
    ]
    panel = panel.merge(
        shot_aggregate[aggregate_columns],
        on=["sb_match_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    for column in SHOT_NUMERIC_COLUMNS:
        panel[column] = panel[column].fillna(0)

    fjelstul_npg = (
        goals.loc[goals["penalty"].eq(0) & goals["own_goal"].eq(0)]
        .groupby(["match_id", "player_id"], as_index=False)
        .size()
        .rename(columns={"size": "fjelstul_non_penalty_goals"})
    )
    panel = panel.merge(
        fjelstul_npg,
        on=["match_id", "player_id"],
        how="left",
        validate="one_to_one",
    )
    panel["fjelstul_non_penalty_goals"] = panel[
        "fjelstul_non_penalty_goals"
    ].fillna(0)
    panel["fjelstul_non_penalty_scorer"] = panel[
        "fjelstul_non_penalty_goals"
    ].gt(0).astype(int)
    panel["statsbomb_non_penalty_scorer"] = panel[
        "non_penalty_goals"
    ].gt(0).astype(int)

    disagreement_columns = [
        "match_id",
        "match_name",
        "player_id",
        "given_name",
        "family_name",
        "fjelstul_non_penalty_scorer",
        "statsbomb_non_penalty_scorer",
        "non_penalty_shots",
        "npxg",
    ]
    disagreements = panel.loc[
        panel["fjelstul_non_penalty_scorer"].ne(
            panel["statsbomb_non_penalty_scorer"]
        ),
        disagreement_columns,
    ].copy()

    audit: dict[str, Any] = {
        "statsbomb_match_count": int(len(statsbomb_matches)),
        "active_lineup_mapped": int(active_audit["mapping_status"].eq("mapped").sum()),
        "shot_takers_mapped": int(shot_aggregate["player_id"].notna().sum()),
        "goal_source_disagreement_count": int(len(disagreements)),
        "goal_source_disagreements": disagreements.to_dict("records"),
    }
    return panel, audit
