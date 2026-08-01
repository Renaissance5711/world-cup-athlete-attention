from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd

ELIGIBLE_OUTCOMES = {"Goal", "Saved", "Saved to Post", "Saved Off Target"}
SAVE_OUTCOMES = {"Saved", "Saved to Post", "Saved Off Target"}


def _distance_and_angle(location: list[float] | None) -> tuple[float, float]:
    if not location or len(location) < 2:
        return float("nan"), float("nan")
    x, y = float(location[0]), float(location[1])
    dx = max(0.0, 120.0 - x)
    dy = y - 40.0
    distance = math.hypot(dx, dy)
    goal_width = 7.32
    left = math.hypot(dx, dy - goal_width / 2)
    right = math.hypot(dx, dy + goal_width / 2)
    denominator = max(1e-12, 2 * left * right)
    cosine = np.clip((left * left + right * right - goal_width * goal_width) / denominator, -1, 1)
    angle = float(math.acos(cosine))
    return distance, angle


def _keeper_freeze_frame(shot: dict) -> dict | None:
    for player in shot.get("freeze_frame", []) or []:
        if (
            player.get("position", {}).get("name") == "Goalkeeper"
            and not bool(player.get("teammate", False))
        ):
            return player
    return None


def extract_on_target_shots(
    events: list[dict],
    *,
    sb_match_id: int,
    match_date: str,
    home_team: str,
    away_team: str,
) -> list[dict]:
    """Extract regulation/extra-time shots resolved as a goal or goalkeeper save."""

    event_lookup = {event.get("id"): event for event in events if event.get("id")}
    score = {home_team: 0, away_team: 0}
    rows: list[dict] = []
    for event in sorted(events, key=lambda item: int(item.get("index", 0) or 0)):
        event_type = event.get("type", {}).get("name")
        period = int(event.get("period", 0) or 0)
        team_name = event.get("team", {}).get("name")
        if event_type == "Shot" and period <= 4 and event.get("player"):
            shot = event.get("shot", {})
            outcome = shot.get("outcome", {}).get("name")
            is_goal = int(outcome == "Goal")
            if outcome in ELIGIBLE_OUTCOMES:
                related = [
                    event_lookup[event_id]
                    for event_id in event.get("related_events", []) or []
                    if event_id in event_lookup
                    and event_lookup[event_id].get("type", {}).get("name") == "Goal Keeper"
                ]
                if len(related) != 1:
                    raise ValueError(
                        f"Shot {event.get('id')} has {len(related)} linked goalkeeper events"
                    )
                goalkeeper_event = related[0]
                goalkeeper = goalkeeper_event.get("player") or {}
                goalkeeper_team = goalkeeper_event.get("team", {}).get("name")
                opponent = away_team if team_name == home_team else home_team
                if goalkeeper_team != opponent:
                    raise ValueError("Linked goalkeeper is not on the opposing team")
                location = event.get("location") or []
                end_location = shot.get("end_location") or []
                distance, angle = _distance_and_angle(location)
                keeper_frame = _keeper_freeze_frame(shot)
                keeper_location = (keeper_frame or {}).get("location") or []
                keeper_distance = (
                    math.hypot(120.0 - float(keeper_location[0]), 40.0 - float(keeper_location[1]))
                    if len(keeper_location) >= 2
                    else float("nan")
                )
                rows.append(
                    {
                        "sb_match_id": int(sb_match_id),
                        "match_date": str(match_date),
                        "event_id": event.get("id"),
                        "event_index": int(event.get("index", 0) or 0),
                        "period": period,
                        "minute": int(event.get("minute", 0) or 0),
                        "second": int(event.get("second", 0) or 0),
                        "shooter_team_name": team_name,
                        "goalkeeper_team_name": goalkeeper_team,
                        "sb_shooter_id": int(event["player"]["id"]),
                        "sb_shooter_name": event["player"]["name"],
                        "sb_goalkeeper_id": int(goalkeeper["id"]),
                        "sb_goalkeeper_name": goalkeeper["name"],
                        "shot_xg": float(shot.get("statsbomb_xg", 0.0) or 0.0),
                        "shot_outcome": outcome,
                        "goalkeeper_outcome": goalkeeper_event.get("goalkeeper", {})
                        .get("outcome", {})
                        .get("name"),
                        "actual_save": int(outcome in SAVE_OUTCOMES),
                        "actual_goal": is_goal,
                        "shot_type": shot.get("type", {}).get("name"),
                        "body_part": shot.get("body_part", {}).get("name"),
                        "technique": shot.get("technique", {}).get("name"),
                        "play_pattern": event.get("play_pattern", {}).get("name"),
                        "under_pressure": int(bool(event.get("under_pressure", False))),
                        "one_on_one": int(bool(shot.get("one_on_one", False))),
                        "open_goal": int(bool(shot.get("open_goal", False))),
                        "first_time": int(bool(shot.get("first_time", False))),
                        "deflected": int(bool(shot.get("deflected", False))),
                        "shot_x": float(location[0]) if len(location) >= 1 else float("nan"),
                        "shot_y": float(location[1]) if len(location) >= 2 else float("nan"),
                        "shot_distance": distance,
                        "shot_angle": angle,
                        "end_x": float(end_location[0]) if len(end_location) >= 1 else float("nan"),
                        "end_y": float(end_location[1]) if len(end_location) >= 2 else float("nan"),
                        "end_z": float(end_location[2]) if len(end_location) >= 3 else 0.0,
                        "goalkeeper_x": float(keeper_location[0]) if len(keeper_location) >= 1 else float("nan"),
                        "goalkeeper_y": float(keeper_location[1]) if len(keeper_location) >= 2 else float("nan"),
                        "goalkeeper_distance_to_goal_center": keeper_distance,
                        "score_for_before_shot": int(score.get(team_name, 0)),
                        "score_against_before_shot": int(score.get(opponent, 0)),
                        "score_diff_before_shot": int(score.get(team_name, 0) - score.get(opponent, 0)),
                    }
                )
            if is_goal:
                score[team_name] = score.get(team_name, 0) + 1
        elif event_type == "Own Goal For" and period <= 4 and team_name in score:
            score[team_name] += 1
    return rows


def _first_constant(group: pd.DataFrame, column: str):
    values = group[column].drop_duplicates()
    if len(values) != 1:
        raise ValueError(f"{column} is not constant within shooter-goalkeeper-match pair")
    return values.iloc[0]


def aggregate_shooter_goalkeeper_pairs(shots: pd.DataFrame) -> pd.DataFrame:
    if shots.empty:
        return pd.DataFrame()
    required = {
        "match_id",
        "shooter_player_id",
        "goalkeeper_player_id",
        "expected_save_probability",
        "actual_save",
        "actual_goal",
        "performance_category",
    }
    missing = required.difference(shots.columns)
    if missing:
        raise ValueError(f"Missing pair aggregation columns: {sorted(missing)}")

    rows: list[dict] = []
    keys = ["match_id", "shooter_player_id", "goalkeeper_player_id"]
    for key, group in shots.groupby(keys, sort=False):
        group = group.sort_values("event_index" if "event_index" in group else "expected_save_probability")
        p = group["expected_save_probability"].astype(float)
        save = group["actual_save"].astype(int)
        goal = group["actual_goal"].astype(int)
        surprise = np.where(save.eq(1), 1 - p, p)
        focal_idx = int(np.argmax(surprise))
        focal_category = group.iloc[focal_idx]["performance_category"]
        row = {
            "match_id": key[0],
            "shooter_player_id": key[1],
            "goalkeeper_player_id": key[2],
            "focal_category": focal_category,
            "on_target_shots": int(len(group)),
            "goals": int(goal.sum()),
            "saves": int(save.sum()),
            "mean_expected_save_probability": float(p.mean()),
            "heroic_save_count": int(group["performance_category"].eq("heroic_save").sum()),
            "routine_save_count": int(group["performance_category"].eq("routine_save").sum()),
            "understandable_goal_count": int(group["performance_category"].eq("understandable_goal").sum()),
            "unexpected_goal_count": int(group["performance_category"].eq("unexpected_goal").sum()),
            "heroic_save_intensity": float(((1 - p) * save).sum()),
            "routine_save_intensity": float((p * save).sum()),
            "understandable_goal_intensity": float(((1 - p) * goal).sum()),
            "unexpected_goal_intensity": float((p * goal).sum()),
            "net_shooter_surprise": float((p * goal).sum() - ((1 - p) * save).sum()),
        }
        optional_constants = [
            "sb_match_id",
            "match_date",
            "shooter_immediate_attention_log_lift",
            "goalkeeper_immediate_attention_log_lift",
            "shooter_additional_pageviews",
            "goalkeeper_additional_pageviews",
            "shooter_baseline_log_views",
            "goalkeeper_baseline_log_views",
            "shooter_team_win",
            "goalkeeper_team_win",
            "knockout_stage",
            "shooter_name",
            "goalkeeper_name",
            "shooter_team_name",
            "goalkeeper_team_name",
        ]
        for column in optional_constants:
            if column in group:
                row[column] = _first_constant(group, column)
        if "shooter_immediate_attention_log_lift" in row:
            row["attention_log_lift_difference"] = float(
                row["shooter_immediate_attention_log_lift"]
                - row["goalkeeper_immediate_attention_log_lift"]
            )
        if "shooter_additional_pageviews" in row:
            row["additional_pageview_difference"] = float(
                row["shooter_additional_pageviews"]
                - row["goalkeeper_additional_pageviews"]
            )
        if "shooter_baseline_log_views" in row:
            row["baseline_log_visibility_difference"] = float(
                row["shooter_baseline_log_views"] - row["goalkeeper_baseline_log_views"]
            )
        if "shooter_team_win" in row:
            row["team_win_difference"] = int(
                row["shooter_team_win"] - row["goalkeeper_team_win"]
            )
        rows.append(row)
    return pd.DataFrame(rows)
