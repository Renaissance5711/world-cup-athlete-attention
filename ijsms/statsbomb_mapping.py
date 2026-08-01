from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def build_match_mapping(statsbomb_matches: list[dict], fjelstul_matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for match in statsbomb_matches:
        rows.append(
            {
                "sb_match_id": int(match["match_id"]),
                "match_date": str(match["match_date"]),
                "home_team_name": match["home_team"]["home_team_name"],
                "away_team_name": match["away_team"]["away_team_name"],
            }
        )
    sb = pd.DataFrame(rows)
    columns = ["match_id", "match_date", "home_team_name", "away_team_name"]
    mapping = sb.merge(fjelstul_matches[columns], on=columns[1:], how="left", validate="one_to_one")
    mapping["mapping_status"] = mapping["match_id"].notna().map({True: "mapped", False: "unmapped"})
    return mapping


def map_lineup_players(
    lineups: Mapping[int, list[dict]],
    match_mapping: pd.DataFrame,
    appearances: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    match_lookup = dict(zip(match_mapping["sb_match_id"].astype(int), match_mapping["match_id"]))
    rows: list[dict] = []
    for sb_match_id, teams in lineups.items():
        match_id = match_lookup.get(int(sb_match_id))
        for team in teams:
            team_name = team["team_name"]
            for player in team["lineup"]:
                rows.append(
                    {
                        "sb_match_id": int(sb_match_id),
                        "match_id": match_id,
                        "team_name": team_name,
                        "sb_player_id": int(player["player_id"]),
                        "sb_player_name": player["player_name"],
                        "shirt_number": int(player["jersey_number"]),
                        "has_position_record": int(bool(player.get("positions"))),
                    }
                )
    lineup_df = pd.DataFrame(rows)
    app_columns = ["match_id", "team_name", "shirt_number", "player_id"]
    mapped = lineup_df.merge(
        appearances[app_columns],
        on=["match_id", "team_name", "shirt_number"],
        how="left",
        validate="many_to_one",
    )
    mapped["mapping_status"] = mapped["player_id"].notna().map({True: "mapped", False: "unmapped"})
    audit = mapped[
        [
            "sb_match_id",
            "match_id",
            "team_name",
            "sb_player_id",
            "sb_player_name",
            "shirt_number",
            "has_position_record",
            "player_id",
            "mapping_status",
        ]
    ].copy()
    return mapped, audit


def _clock_to_seconds(value: str | None, fallback: int) -> int:
    if not value:
        return fallback
    minutes, seconds = value.split(':', maxsplit=1)
    return int(minutes) * 60 + int(float(seconds))


def summarize_lineup_participation(
    teams: list[dict],
    sb_match_id: int,
    match_end_seconds: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for team in teams:
        for player in team['lineup']:
            positions = player.get('positions', [])
            if not positions:
                continue
            total_seconds = 0
            for segment in positions:
                start = _clock_to_seconds(segment.get('from'), 0)
                end = _clock_to_seconds(segment.get('to'), match_end_seconds)
                total_seconds += max(0, end - start)
            first = positions[0]
            rows.append(
                {
                    'sb_match_id': int(sb_match_id),
                    'team_name': team['team_name'],
                    'sb_player_id': int(player['player_id']),
                    'sb_player_name': player['player_name'],
                    'shirt_number': int(player['jersey_number']),
                    'minutes_played': total_seconds / 60.0,
                    'sb_starter': int(first.get('start_reason') == 'Starting XI'),
                    'first_position': first.get('position'),
                }
            )
    return pd.DataFrame(rows)
