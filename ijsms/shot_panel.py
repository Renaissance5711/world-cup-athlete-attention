from __future__ import annotations

import pandas as pd

ON_TARGET_OUTCOMES = {"Goal", "Saved", "Saved to Post", "Saved Off Target", "Post"}


def summarize_match_events(
    events: list[dict],
    sb_match_id: int,
    home_team: str,
    away_team: str,
) -> list[dict]:
    score = {home_team: 0, away_team: 0}
    shot_rows: list[dict] = []
    for event in sorted(events, key=lambda item: int(item.get("index", 0))):
        event_type = event.get("type", {}).get("name")
        team_name = event.get("team", {}).get("name")
        period = int(event.get("period", 0) or 0)
        if event_type == "Shot" and event.get("player"):
            opponent = away_team if team_name == home_team else home_team
            shot = event.get("shot", {})
            outcome = shot.get("outcome", {}).get("name")
            shot_type = shot.get("type", {}).get("name")
            is_goal = int(outcome == "Goal" and period <= 4)
            is_non_penalty_goal = int(is_goal == 1 and shot_type != "Penalty")
            shot_rows.append(
                {
                    "sb_match_id": int(sb_match_id),
                    "event_index": int(event.get("index", 0)),
                    "period": period,
                    "minute": int(event.get("minute", 0) or 0),
                    "second": int(event.get("second", 0) or 0),
                    "team_name": team_name,
                    "opponent_name": opponent,
                    "sb_player_id": int(event["player"]["id"]),
                    "sb_player_name": event["player"]["name"],
                    "shot_xg": float(shot.get("statsbomb_xg", 0.0) or 0.0),
                    "shot_type": shot_type,
                    "shot_outcome": outcome,
                    "shot_body_part": shot.get("body_part", {}).get("name"),
                    "play_pattern": event.get("play_pattern", {}).get("name"),
                    "under_pressure": int(bool(event.get("under_pressure", False))),
                    "score_for_before_shot": score.get(team_name, 0),
                    "score_against_before_shot": score.get(opponent, 0),
                    "score_diff_before_shot": score.get(team_name, 0) - score.get(opponent, 0),
                    "is_regulation_goal": is_goal,
                    "is_non_penalty_goal": is_non_penalty_goal,
                }
            )
            if is_goal:
                score[team_name] = score.get(team_name, 0) + 1
        elif event_type == "Own Goal For" and period <= 4 and team_name in score:
            score[team_name] += 1
    return shot_rows


def aggregate_player_match_shots(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return pd.DataFrame()
    frame = shots.copy()
    if "event_index" not in frame.columns:
        frame["event_index"] = range(len(frame))
    if "sb_player_name" not in frame.columns:
        frame["sb_player_name"] = frame["sb_player_id"].astype(str)
    frame = frame.sort_values(["sb_match_id", "sb_player_id", "event_index"], kind="stable").copy()
    frame["is_penalty"] = frame["shot_type"].eq("Penalty").astype(int)
    frame["is_non_penalty"] = 1 - frame["is_penalty"]
    frame["npxg_component"] = frame["shot_xg"] * frame["is_non_penalty"]
    frame["is_on_target"] = frame["shot_outcome"].isin(ON_TARGET_OUTCOMES).astype(int)
    frame["is_open_play"] = frame["shot_type"].eq("Open Play").astype(int)

    grouped = frame.groupby(["sb_match_id", "team_name", "sb_player_id"], sort=False)
    result = grouped.agg(
        sb_player_name=("sb_player_name", "first"),
        shots=("shot_xg", "size"),
        non_penalty_shots=("is_non_penalty", "sum"),
        total_xg=("shot_xg", "sum"),
        npxg=("npxg_component", "sum"),
        max_shot_xg=("shot_xg", "max"),
        max_npxg=("npxg_component", "max"),
        regulation_goals=("is_regulation_goal", "sum"),
        non_penalty_goals=("is_non_penalty_goal", "sum"),
        penalty_attempts=("is_penalty", "sum"),
        shots_on_target=("is_on_target", "sum"),
        open_play_shots=("is_open_play", "sum"),
        under_pressure_shots=("under_pressure", "sum"),
        first_shot_minute=("minute", "first"),
        first_shot_score_diff=("score_diff_before_shot", "first"),
    ).reset_index()
    result["scored_any_goal"] = result["regulation_goals"].gt(0).astype(int)
    result["scored_non_penalty_goal"] = result["non_penalty_goals"].gt(0).astype(int)
    return result
