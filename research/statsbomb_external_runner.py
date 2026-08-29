#!/usr/bin/env python3
"""Process a pinned sparse checkout of StatsBomb Open Data into the frozen 12D panel.

This module is deliberately network-free. The GitHub Actions workflow performs the
pinned sparse checkout; this script validates coverage and transforms only local JSON.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PIN = "3bfbffe1de5750ebd47d770be0bb924a10cde54f"
COMPETITIONS = {
    "England": (2, 27, 380),
    "France": (7, 27, 377),
    "Germany": (9, 27, 306),
    "Spain": (11, 27, 380),
    "Italy": (12, 27, 380),
}
FEATURES_12 = [
    "mean_pass_length", "mean_dx", "mean_abs_dy", "mean_x1", "mean_x2",
    "forward_share", "long_share", "final3_start_share", "final3_end_share",
    "high_share", "head_share", "cross_share",
]


def period_seconds(timestamp: str) -> float:
    hh, mm, ss = timestamp.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def flatten_pass_event(e: dict, league: str, match_id: int, match_date: str) -> dict | None:
    if e.get("type", {}).get("name") != "Pass":
        return None
    period = int(e.get("period", 0))
    if period not in (1, 2):
        return None
    loc = e.get("location")
    p = e.get("pass") if isinstance(e.get("pass"), dict) else {}
    end = p.get("end_location")
    if not loc or not end or len(loc) < 2 or len(end) < 2:
        return None

    x1, y1 = float(loc[0]) / 1.2, float(loc[1]) / 0.8
    x2, y2 = float(end[0]) / 1.2, float(end[1]) / 0.8
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    outcome = p.get("outcome") if isinstance(p.get("outcome"), dict) else None
    outcome_name = outcome.get("name") if outcome else None
    team = e.get("team") if isinstance(e.get("team"), dict) else {}
    player = e.get("player") if isinstance(e.get("player"), dict) else {}

    return {
        "game_id": int(match_id),
        "match_date": match_date,
        "league": league,
        "half": period,
        "t": period_seconds(e["timestamp"]),
        "event_index": int(e.get("index", 0)),
        "event_uuid": e.get("id"),
        "team_id": int(team["id"]),
        "team_name": team.get("name"),
        "player_id": player.get("id"),
        "possession": e.get("possession"),
        "complete": int(outcome_name is None),
        "pass_outcome": outcome_name,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "mean_pass_length": length,
        "mean_dx": dx,
        "mean_abs_dy": abs(dy),
        "mean_x1": x1,
        "mean_x2": x2,
        "forward_share": float(dx > 0),
        "long_share": float(length >= 30),
        "final3_start_share": float(x1 >= 70),
        "final3_end_share": float(x2 >= 70),
        "high_share": float(p.get("height", {}).get("name") == "High Pass"),
        "head_share": float(p.get("body_part", {}).get("name") == "Head"),
        "cross_share": float(bool(p.get("cross", False))),
    }


def _other_team(scoring_team_id: int, match: dict) -> tuple[int | None, str | None]:
    h = match.get("home_team", {})
    a = match.get("away_team", {})
    hid, aid = h.get("home_team_id"), a.get("away_team_id")
    if scoring_team_id == hid:
        return aid, a.get("away_team_name")
    if scoring_team_id == aid:
        return hid, h.get("home_team_name")
    return None, None


def flatten_goal_event(e: dict, league: str, match: dict) -> dict | None:
    et = e.get("type", {}).get("name")
    shot = e.get("shot") if isinstance(e.get("shot"), dict) else {}
    is_shot_goal = et == "Shot" and shot.get("outcome", {}).get("name") == "Goal"
    is_own_goal_for = et == "Own Goal For"
    if not (is_shot_goal or is_own_goal_for):
        return None
    period = int(e.get("period", 0))
    if period not in (1, 2):
        return None
    team = e.get("team") if isinstance(e.get("team"), dict) else {}
    if team.get("id") is None:
        return None
    scoring_id = int(team["id"])
    conceding_id, conceding_name = _other_team(scoring_id, match)
    return {
        "game_id": int(match["match_id"]),
        "match_date": match.get("match_date"),
        "league": league,
        "half": period,
        "t": period_seconds(e["timestamp"]),
        "event_index": int(e.get("index", 0)),
        "event_uuid": e.get("id"),
        "goal_event_type": et,
        "shot_type": shot.get("type", {}).get("name") if shot else None,
        "scoring_team_id": scoring_id,
        "scoring_team_name": team.get("name"),
        "conceding_team_id": conceding_id,
        "conceding_team_name": conceding_name,
    }


def build_pass_spells(pass_df: pd.DataFrame) -> pd.DataFrame:
    d = pass_df.sort_values(["game_id", "half", "t", "event_index"]).copy()
    prev = d.groupby(["game_id", "half"])["team_id"].shift(1)
    d["spell_start"] = (d["team_id"] != prev).astype(int)
    d["spell_id"] = d.groupby(["game_id", "half"])["spell_start"].cumsum()
    dep = (
        d.groupby(["game_id", "half", "spell_id"], as_index=False)
        .agg(spell_team=("team_id", "first"), spell_start_t=("t", "first"), spell_depth=("team_id", "size"))
    )
    for k in (2, 3, 4, 5):
        dep[f"reach{k}"] = (dep["spell_depth"] >= k).astype(int)
    return d.merge(dep, on=["game_id", "half", "spell_id"], how="left")


def build_two_minute_windows(pass_df: pd.DataFrame) -> pd.DataFrame:
    d = pass_df.copy()
    d["bin2"] = (d["t"] // 120).astype(int)
    keys = ["game_id", "match_date", "league", "half", "team_id", "team_name", "bin2"]
    g = d.groupby(keys, sort=False)
    w = g[FEATURES_12].mean().reset_index()
    w["pass_n"] = g.size().to_numpy()
    return w[w["pass_n"] >= 3].copy()


def add_spell_labels(pass_with_spells: pd.DataFrame, windows: pd.DataFrame) -> pd.DataFrame:
    starts = pass_with_spells[pass_with_spells["spell_start"].eq(1)].copy()
    starts["bin2"] = (starts["spell_start_t"] // 120).astype(int)
    lab = (
        starts.groupby(["game_id", "half", "team_id", "bin2"], as_index=False)
        .agg(
            spell_starts=("spell_id", "size"),
            mean_spell_depth=("spell_depth", "mean"),
            median_spell_depth=("spell_depth", "median"),
            reach2_share=("reach2", "mean"),
            reach3_share=("reach3", "mean"),
            reach4_share=("reach4", "mean"),
            reach5_share=("reach5", "mean"),
        )
    )
    return windows.merge(lab, on=["game_id", "half", "team_id", "bin2"], how="left")


def validate_coverage(actual: dict[str, int]) -> int:
    total = 0
    for league, (_, _, expected) in COMPETITIONS.items():
        got = int(actual.get(league, -1))
        if got != expected:
            raise RuntimeError(f"{league}: expected {expected} event-complete matches, got {got}")
        total += got
    if total != 1823:
        raise RuntimeError(f"Expected total 1823 matches, got {total}")
    return total


def load_matches(data_root: Path) -> tuple[pd.DataFrame, list[tuple[str, dict]]]:
    records: list[tuple[str, dict]] = []
    counts: dict[str, int] = {}
    rows = []
    for league, (comp, season, _) in COMPETITIONS.items():
        path = data_root / "matches" / str(comp) / f"{season}.json"
        matches = json.loads(path.read_text(encoding="utf-8"))
        counts[league] = len(matches)
        for m in matches:
            records.append((league, m))
            rows.append({
                "game_id": int(m["match_id"]),
                "match_date": m.get("match_date"),
                "league": league,
                "competition_id": comp,
                "season_id": season,
                "home_team_id": m.get("home_team", {}).get("home_team_id"),
                "home_team_name": m.get("home_team", {}).get("home_team_name"),
                "away_team_id": m.get("away_team", {}).get("away_team_id"),
                "away_team_name": m.get("away_team", {}).get("away_team_name"),
                "home_score": m.get("home_score"),
                "away_score": m.get("away_score"),
            })
    validate_coverage(counts)
    md = pd.DataFrame(rows).sort_values(["match_date", "game_id"]).reset_index(drop=True)
    md["game_chron_key"] = np.arange(1, len(md) + 1, dtype=int)
    return md, records


def process_checkout(data_root: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    matches, records = load_matches(data_root)
    matches.to_csv(output_dir / "statsbomb_2015_16_big5_matches.csv", index=False)
    chron = matches.set_index("game_id")["game_chron_key"].to_dict()
    expected_ids = set(matches.game_id.astype(int))
    event_ids = {int(p.stem) for p in (data_root / "events").glob("*.json")}
    missing = sorted(expected_ids - event_ids)
    extra = sorted(event_ids - expected_ids)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} event files; first IDs: {missing[:10]}")

    goals_all = []
    coverage_rows = []
    for league, (_, _, expected) in COMPETITIONS.items():
        league_records = [(lg, m) for lg, m in records if lg == league]
        pass_rows = []
        goal_rows = []
        for i, (_, m) in enumerate(league_records, 1):
            mid = int(m["match_id"])
            events = json.loads((data_root / "events" / f"{mid}.json").read_text(encoding="utf-8"))
            for e in events:
                p = flatten_pass_event(e, league, mid, m.get("match_date"))
                if p is not None:
                    p["game_chron_key"] = chron[mid]
                    pass_rows.append(p)
                g = flatten_goal_event(e, league, m)
                if g is not None:
                    g["game_chron_key"] = chron[mid]
                    goal_rows.append(g)
            if i % 50 == 0 or i == len(league_records):
                print(f"{league}: processed {i}/{len(league_records)} matches", flush=True)

        passes = pd.DataFrame(pass_rows)
        if passes.empty:
            raise RuntimeError(f"{league}: no passes extracted")
        spell_passes = build_pass_spells(passes)
        windows = add_spell_labels(spell_passes, build_two_minute_windows(passes))
        windows = windows.merge(matches[["game_id", "game_chron_key"]], on="game_id", how="left")

        slug = league.lower()
        passes.to_parquet(output_dir / f"passes_{slug}.parquet", index=False, compression="zstd")
        windows.to_parquet(output_dir / f"windows12_{slug}.parquet", index=False, compression="zstd")
        pd.DataFrame(goal_rows).to_csv(output_dir / f"goals_{slug}.csv", index=False)
        goals_all.extend(goal_rows)
        coverage_rows.append({
            "league": league,
            "expected_matches": expected,
            "processed_matches": len(league_records),
            "passes": len(passes),
            "eligible_windows": len(windows),
            "windows_with_spell_starts": int(windows["spell_starts"].notna().sum()),
            "goals_extracted": len(goal_rows),
        })
        del passes, spell_passes, windows, pass_rows

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(output_dir / "coverage_verified.csv", index=False)
    pd.DataFrame(goals_all).to_csv(output_dir / "goals_all.csv", index=False)

    manifest = {
        "source": "hudl/open-data",
        "pinned_commit": PIN,
        "target_matches": 1823,
        "processed_matches": int(coverage.processed_matches.sum()),
        "passes": int(coverage.passes.sum()),
        "eligible_windows": int(coverage.eligible_windows.sum()),
        "goals_extracted": int(coverage.goals_extracted.sum()),
        "features_12": FEATURES_12,
        "coordinate_mapping": "x/1.2, y/0.8 to Wyscout 0-100 scale",
        "spell_rule": "pass-only consecutive same-team runs; label by spell start bin; full run depth",
        "extra_event_files": len(extra),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to pinned checkout's data directory")
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    process_checkout(Path(args.data_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
