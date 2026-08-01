from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .expected_save import cross_fit_expected_save
from .goalkeeper_models import (
    fit_bilateral_attention_models,
    fit_bilateral_category_contrasts,
    fit_all_pair_outcome_balance_sensitivity,
    fit_single_shot_outcome_gradient_models,
    fit_news_framing_validation,
    summarize_category_means,
)
from .goalkeeper_panel import aggregate_shooter_goalkeeper_pairs, extract_on_target_shots
from .news_framing import (
    aggregate_goalkeeper_match_headlines,
    parse_wolfram_google_news_csv,
    prepare_goalkeeper_headlines,
)
from .statsbomb_mapping import build_match_mapping, map_lineup_players


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _display_name(given: pd.Series, family: pd.Series) -> pd.Series:
    given_clean = given.fillna("").astype(str).replace("not applicable", "")
    return (given_clean.str.strip() + " " + family.fillna("").astype(str).str.strip()).str.strip()


def _match_end_datetime(match: dict, events: list[dict]) -> pd.Timestamp:
    elapsed = max(
        (
            int(event.get("minute", 0) or 0) * 60
            + int(event.get("second", 0) or 0)
            for event in events
            if int(event.get("period", 0) or 0) <= 4
        ),
        default=90 * 60,
    )
    kick_off = pd.Timestamp(
        f"{match['match_date']} {match['kick_off']}", tz="Asia/Qatar"
    ).tz_convert("UTC")
    return kick_off + pd.Timedelta(seconds=elapsed + 300)


def _shootout_goalkeeper_counts(
    events: list[dict], sb_match_id: int, player_lookup: pd.DataFrame
) -> pd.DataFrame:
    event_lookup = {event.get("id"): event for event in events if event.get("id")}
    rows: list[dict] = []
    for event in events:
        if event.get("type", {}).get("name") != "Shot" or int(event.get("period", 0) or 0) != 5:
            continue
        related = [
            event_lookup[event_id]
            for event_id in event.get("related_events", []) or []
            if event_id in event_lookup
            and event_lookup[event_id].get("type", {}).get("name") == "Goal Keeper"
        ]
        if len(related) != 1:
            continue
        goalkeeper_event = related[0]
        outcome = event.get("shot", {}).get("outcome", {}).get("name")
        rows.append(
            {
                "sb_match_id": int(sb_match_id),
                "team_name": goalkeeper_event.get("team", {}).get("name"),
                "sb_player_id": int(goalkeeper_event.get("player", {}).get("id")),
                "shootout_save": int(outcome == "Saved"),
                "shootout_goal_allowed": int(outcome == "Goal"),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["sb_match_id", "goalkeeper_player_id", "shootout_saves", "shootout_goals_allowed"]
        )
    frame = pd.DataFrame(rows).merge(
        player_lookup.rename(columns={"player_id": "goalkeeper_player_id"}),
        on=["sb_match_id", "team_name", "sb_player_id"],
        how="left",
        validate="many_to_one",
    )
    if frame["goalkeeper_player_id"].isna().any():
        raise ValueError("Unmapped shootout goalkeeper")
    return (
        frame.groupby(["sb_match_id", "goalkeeper_player_id"], as_index=False)
        .agg(
            shootout_saves=("shootout_save", "sum"),
            shootout_goals_allowed=("shootout_goal_allowed", "sum"),
        )
    )


def _goalkeeper_performance(shots: pd.DataFrame, shootouts: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for (match_id, sb_match_id, goalkeeper_player_id), group in shots.groupby(
        ["match_id", "sb_match_id", "goalkeeper_player_id"], sort=False
    ):
        probability = group["expected_save_probability"].astype(float)
        saves = group["actual_save"].astype(int)
        goals = group["actual_goal"].astype(int)
        rows.append(
            {
                "match_id": match_id,
                "sb_match_id": int(sb_match_id),
                "goalkeeper_player_id": goalkeeper_player_id,
                "on_target_shots": int(len(group)),
                "heroic_save_count": int(group["performance_category"].eq("heroic_save").sum()),
                "routine_save_count": int(group["performance_category"].eq("routine_save").sum()),
                "understandable_goal_count": int(group["performance_category"].eq("understandable_goal").sum()),
                "unexpected_goal_count": int(group["performance_category"].eq("unexpected_goal").sum()),
                "heroic_save_intensity": float(((1 - probability) * saves).sum()),
                "routine_save_intensity": float((probability * saves).sum()),
                "understandable_goal_intensity": float(((1 - probability) * goals).sum()),
                "unexpected_goal_intensity": float((probability * goals).sum()),
            }
        )
    performance = pd.DataFrame(rows)
    performance = performance.merge(
        shootouts,
        on=["sb_match_id", "goalkeeper_player_id"],
        how="left",
        validate="one_to_one",
    )
    for column in ["shootout_saves", "shootout_goals_allowed"]:
        performance[column] = pd.to_numeric(performance[column], errors="coerce").fillna(0).astype(int)
    performance["heroic_save_any"] = performance["heroic_save_count"].gt(0).astype(int)
    performance["unexpected_goal_any"] = performance["unexpected_goal_count"].gt(0).astype(int)
    performance["notable_save_any"] = (
        performance["heroic_save_count"].gt(0) | performance["shootout_saves"].gt(0)
    ).astype(int)
    return performance


def run_goalkeeper_workflow(
    *,
    workspace: Path,
    news_csv_paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, Path]:
    workspace = Path(workspace)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    statsbomb_root = workspace / "data" / "raw" / "statsbomb" / "combined"
    base = workspace / "base_archive" / "data"

    statsbomb_matches = _read_json(statsbomb_root / "matches_43_106.json")
    fjelstul_matches = pd.read_csv(base / "raw" / "fjelstul" / "matches.csv")
    appearances = pd.read_csv(base / "raw" / "fjelstul" / "player_appearances.csv")
    outcomes = pd.read_csv(base / "processed" / "all_player_match_outcomes_2022.csv")
    match_mapping = build_match_mapping(statsbomb_matches, fjelstul_matches)
    if not match_mapping["mapping_status"].eq("mapped").all():
        raise RuntimeError("StatsBomb match mapping failed")

    lineups: dict[int, list[dict]] = {}
    events_by_match: dict[int, list[dict]] = {}
    shot_rows: list[dict] = []
    match_end_rows: list[dict] = []
    for match in statsbomb_matches:
        sb_match_id = int(match["match_id"])
        events = _read_json(statsbomb_root / "events" / f"{sb_match_id}.json")
        lineups[sb_match_id] = _read_json(statsbomb_root / "lineups" / f"{sb_match_id}.json")
        events_by_match[sb_match_id] = events
        shot_rows.extend(
            extract_on_target_shots(
                events,
                sb_match_id=sb_match_id,
                match_date=match["match_date"],
                home_team=match["home_team"]["home_team_name"],
                away_team=match["away_team"]["away_team_name"],
            )
        )
        match_end_rows.append(
            {
                "sb_match_id": sb_match_id,
                "match_end_datetime": _match_end_datetime(match, events),
            }
        )

    lineup_mapping, lineup_audit = map_lineup_players(lineups, match_mapping, appearances)
    active = lineup_audit.loc[lineup_audit["has_position_record"].eq(1)]
    if not active["mapping_status"].eq("mapped").all():
        raise RuntimeError("Active lineup mapping failed")
    player_lookup = lineup_mapping[
        ["sb_match_id", "team_name", "sb_player_id", "player_id"]
    ].drop_duplicates()

    shots = pd.DataFrame(shot_rows)
    shots = shots.merge(
        player_lookup.rename(
            columns={
                "team_name": "shooter_team_name",
                "sb_player_id": "sb_shooter_id",
                "player_id": "shooter_player_id",
            }
        ),
        on=["sb_match_id", "shooter_team_name", "sb_shooter_id"],
        how="left",
        validate="many_to_one",
    )
    shots = shots.merge(
        player_lookup.rename(
            columns={
                "team_name": "goalkeeper_team_name",
                "sb_player_id": "sb_goalkeeper_id",
                "player_id": "goalkeeper_player_id",
            }
        ),
        on=["sb_match_id", "goalkeeper_team_name", "sb_goalkeeper_id"],
        how="left",
        validate="many_to_one",
    )
    shots = shots.merge(
        match_mapping[["sb_match_id", "match_id"]],
        on="sb_match_id",
        how="left",
        validate="many_to_one",
    )
    if shots[["shooter_player_id", "goalkeeper_player_id", "match_id"]].isna().any().any():
        raise RuntimeError("Shot participant mapping failed")

    outcome_columns = [
        "match_id", "player_id", "family_name", "given_name", "baseline_log_views",
        "immediate_attention_log_lift", "winsorized_additional_pageviews", "team_win",
        "knockout_stage",
    ]
    shooter = outcomes[outcome_columns].copy()
    shooter["shooter_name"] = _display_name(shooter["given_name"], shooter["family_name"])
    shooter = shooter.drop(columns=["family_name", "given_name"]).rename(
        columns={
            "player_id": "shooter_player_id",
            "baseline_log_views": "shooter_baseline_log_views",
            "immediate_attention_log_lift": "shooter_immediate_attention_log_lift",
            "winsorized_additional_pageviews": "shooter_additional_pageviews",
            "team_win": "shooter_team_win",
            "knockout_stage": "shooter_knockout_stage",
        }
    )
    goalkeeper = outcomes[outcome_columns].copy()
    goalkeeper["goalkeeper_name"] = _display_name(goalkeeper["given_name"], goalkeeper["family_name"])
    goalkeeper = goalkeeper.drop(columns=["family_name", "given_name"]).rename(
        columns={
            "player_id": "goalkeeper_player_id",
            "baseline_log_views": "goalkeeper_baseline_log_views",
            "immediate_attention_log_lift": "goalkeeper_immediate_attention_log_lift",
            "winsorized_additional_pageviews": "goalkeeper_additional_pageviews",
            "team_win": "goalkeeper_team_win",
            "knockout_stage": "goalkeeper_knockout_stage",
        }
    )
    shots = shots.merge(shooter, on=["match_id", "shooter_player_id"], how="left", validate="many_to_one")
    shots = shots.merge(goalkeeper, on=["match_id", "goalkeeper_player_id"], how="left", validate="many_to_one")
    shots["knockout_stage"] = shots["shooter_knockout_stage"].astype(int)
    if shots.filter(regex="attention_log_lift|additional_pageviews|baseline_log_views").isna().any().any():
        raise RuntimeError("Pageview outcomes missing for a shot participant")

    predicted, diagnostics = cross_fit_expected_save(shots)
    if diagnostics["group_leakage_count"] != 0:
        raise RuntimeError("Expected-save cross-fitting leaked a match")
    if diagnostics["brier_score"] >= diagnostics["intercept_brier_score"]:
        raise RuntimeError("Expected-save model failed the Brier benchmark")
    if diagnostics.get("roc_auc", 0.0) < 0.65:
        raise RuntimeError("Expected-save model failed the AUC gate")

    pairs = aggregate_shooter_goalkeeper_pairs(predicted)
    bilateral = fit_bilateral_attention_models(pairs)
    contrasts = fit_bilateral_category_contrasts(pairs)
    single_shot_pairs = pairs.loc[pairs["on_target_shots"].eq(1)].copy()
    single_shot_category_contrasts = fit_bilateral_category_contrasts(single_shot_pairs)
    single_shot_category_means = summarize_category_means(single_shot_pairs)
    outcome_gradients = fit_single_shot_outcome_gradient_models(pairs)
    outcome_balance = fit_all_pair_outcome_balance_sensitivity(pairs)
    sample_structure = pd.DataFrame([
        {"metric": "on_target_shots", "value": int(len(predicted))},
        {"metric": "bilateral_pairs", "value": int(len(pairs))},
        {"metric": "single_shot_pairs", "value": int(pairs["on_target_shots"].eq(1).sum())},
        {"metric": "multi_shot_pairs", "value": int(pairs["on_target_shots"].gt(1).sum())},
        {"metric": "mixed_outcome_pairs", "value": int((pairs["goals"].gt(0) & pairs["saves"].gt(0)).sum())},
        {"metric": "single_shot_goals", "value": int((pairs["on_target_shots"].eq(1) & pairs["goals"].eq(1)).sum())},
        {"metric": "single_shot_saves", "value": int((pairs["on_target_shots"].eq(1) & pairs["saves"].eq(1)).sum())},
    ])
    category_means = summarize_category_means(pairs)

    shootout_frames = [
        _shootout_goalkeeper_counts(events_by_match[sb_match_id], sb_match_id, player_lookup)
        for sb_match_id in events_by_match
    ]
    shootouts = pd.concat(shootout_frames, ignore_index=True)
    performance = _goalkeeper_performance(predicted, shootouts)

    match_end = pd.DataFrame(match_end_rows)
    goalkeeper_matches = predicted[
        ["match_id", "sb_match_id", "match_date", "goalkeeper_player_id", "goalkeeper_team_name", "goalkeeper_name"]
    ].drop_duplicates()
    goalkeeper_matches = goalkeeper_matches.merge(match_end, on="sb_match_id", how="left", validate="many_to_one")
    match_teams = fjelstul_matches[
        ["match_id", "home_team_name", "away_team_name", "knockout_stage"]
    ]
    goalkeeper_matches = goalkeeper_matches.merge(match_teams, on="match_id", how="left", validate="many_to_one")
    goalkeeper_matches["opponent_team_name"] = np.where(
        goalkeeper_matches["goalkeeper_team_name"].eq(goalkeeper_matches["home_team_name"]),
        goalkeeper_matches["away_team_name"],
        goalkeeper_matches["home_team_name"],
    )

    news_frames = [parse_wolfram_google_news_csv(path) for path in news_csv_paths]
    raw_news = pd.concat(news_frames, ignore_index=True)
    retained_news = prepare_goalkeeper_headlines(raw_news, goalkeeper_matches, window_hours=48)
    news_summary = aggregate_goalkeeper_match_headlines(retained_news)

    validation_panel = performance.merge(
        news_summary, on=["match_id", "goalkeeper_player_id"], how="left", validate="one_to_one"
    )
    count_columns = [
        "headline_count", "praise_count", "blame_count", "neutral_count",
        "directional_count", "praise_any", "blame_any",
    ]
    share_columns = ["praise_share", "blame_share", "directional_balance"]
    for column in count_columns:
        validation_panel[column] = validation_panel[column].fillna(0).astype(int)
    for column in share_columns:
        validation_panel[column] = validation_panel[column].fillna(0.0)
    news_validation = fit_news_framing_validation(validation_panel)

    paths = {
        "on_target_shots": output_dir / "goalkeeper_on_target_shots.csv",
        "expected_save_diagnostics": output_dir / "expected_save_diagnostics.json",
        "bilateral_pairs": output_dir / "shooter_goalkeeper_bilateral_pairs.csv",
        "bilateral_results": output_dir / "goalkeeper_bilateral_model_results.csv",
        "bilateral_contrasts": output_dir / "goalkeeper_bilateral_contrasts.csv",
        "outcome_gradient_results": output_dir / "goalkeeper_outcome_gradient_results.csv",
        "outcome_balance_sensitivity": output_dir / "goalkeeper_outcome_balance_sensitivity.csv",
        "sample_structure": output_dir / "goalkeeper_bilateral_sample_structure.csv",
        "single_shot_category_contrasts": output_dir / "goalkeeper_single_shot_category_contrasts.csv",
        "single_shot_category_means": output_dir / "goalkeeper_single_shot_category_means.csv",
        "category_means": output_dir / "goalkeeper_category_means.csv",
        "goalkeeper_performance": output_dir / "goalkeeper_match_performance.csv",
        "news_headlines": output_dir / "goalkeeper_news_headlines_retained.csv",
        "news_match_summary": output_dir / "goalkeeper_news_match_summary.csv",
        "news_validation_panel": output_dir / "goalkeeper_news_validation_panel.csv",
        "news_validation_results": output_dir / "goalkeeper_news_validation_results.csv",
        "news_query_manifest": output_dir / "goalkeeper_news_query_manifest.csv",
        "lineup_mapping_audit": output_dir / "goalkeeper_lineup_mapping_audit.csv",
        "workflow_audit": output_dir / "goalkeeper_workflow_audit.json",
    }
    predicted.to_csv(paths["on_target_shots"], index=False)
    paths["expected_save_diagnostics"].write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    pairs.to_csv(paths["bilateral_pairs"], index=False)
    bilateral.to_csv(paths["bilateral_results"], index=False)
    contrasts.to_csv(paths["bilateral_contrasts"], index=False)
    outcome_gradients.to_csv(paths["outcome_gradient_results"], index=False)
    outcome_balance.to_csv(paths["outcome_balance_sensitivity"], index=False)
    sample_structure.to_csv(paths["sample_structure"], index=False)
    single_shot_category_contrasts.to_csv(paths["single_shot_category_contrasts"], index=False)
    single_shot_category_means.to_csv(paths["single_shot_category_means"], index=False)
    category_means.to_csv(paths["category_means"], index=False)
    performance.to_csv(paths["goalkeeper_performance"], index=False)
    retained_news.to_csv(paths["news_headlines"], index=False)
    news_summary.to_csv(paths["news_match_summary"], index=False)
    validation_panel.to_csv(paths["news_validation_panel"], index=False)
    news_validation.to_csv(paths["news_validation_results"], index=False)
    goalkeeper_matches.to_csv(paths["news_query_manifest"], index=False)
    lineup_audit.to_csv(paths["lineup_mapping_audit"], index=False)
    audit = {
        "on_target_shots": int(len(predicted)),
        "matches": int(predicted["sb_match_id"].nunique()),
        "goalkeepers": int(predicted["goalkeeper_player_id"].nunique()),
        "shooters": int(predicted["shooter_player_id"].nunique()),
        "bilateral_pairs": int(len(pairs)),
        "single_shot_pairs": int(pairs["on_target_shots"].eq(1).sum()),
        "multi_shot_pairs": int(pairs["on_target_shots"].gt(1).sum()),
        "mixed_outcome_pairs": int((pairs["goals"].gt(0) & pairs["saves"].gt(0)).sum()),
        "performance_category_counts": predicted["performance_category"].value_counts().to_dict(),
        "raw_indexed_headlines": int(len(raw_news)),
        "retained_post_match_headlines": int(len(retained_news)),
        "covered_goalkeeper_matches": int(news_summary.shape[0]),
        "covered_matches": int(retained_news["match_id"].nunique()) if len(retained_news) else 0,
        "headline_framing_counts": retained_news["framing"].value_counts().to_dict() if len(retained_news) else {},
        "expected_save": diagnostics,
    }
    paths["workflow_audit"].write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return paths
