#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/fjelstul"
PROCESSED = ROOT / "data/processed"
CODE = ROOT / "code"
SOURCES = {
    "players.csv": "https://raw.githubusercontent.com/jfjelstul/worldcup/refs/heads/master/data-csv/players.csv",
    "player_appearances.csv": "https://raw.githubusercontent.com/jfjelstul/worldcup/refs/heads/master/data-csv/player_appearances.csv",
    "matches.csv": "https://raw.githubusercontent.com/jfjelstul/worldcup/refs/heads/master/data-csv/matches.csv",
    "goals.csv": "https://raw.githubusercontent.com/jfjelstul/worldcup/refs/heads/master/data-csv/goals.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete replication pipeline.")
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use the archived raw CSV files instead of making network requests.",
    )
    parser.add_argument(
        "--user-agent",
        default="AthleteAttentionAcademicResearch/2.0 (academic replication)",
    )
    return parser.parse_args()


def download(url: str, path: Path, user_agent: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with requests.get(
        url,
        stream=True,
        timeout=120,
        headers={"User-Agent": user_agent},
    ) as response:
        response.raise_for_status()
        with path.open("wb") as handle:
            for chunk in response.iter_content(1024 * 1024):
                handle.write(chunk)


def run(*args: object) -> None:
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(
        [sys.executable, *map(str, args)], cwd=CODE, check=True
    )


def require_archived(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"--skip-download requires the archived input file: {path}"
        )


def main() -> None:
    args = parse_args()
    requested = (
        ROOT
        / "data/raw/pageviews/all_player_daily_pageviews_requested_titles_2022.csv"
    )
    resolved = (
        ROOT
        / "data/raw/pageviews/all_player_daily_pageviews_resolved_titles_2022.csv"
    )
    mapping = ROOT / "metadata/title_redirect_mapping.csv"

    if args.skip_download:
        for path in [*(RAW / name for name in SOURCES), requested, resolved, mapping]:
            require_archived(path)
    else:
        for name, url in SOURCES.items():
            download(url, RAW / name, args.user_agent)

    run(
        "scripts/build_all_player_roster.py",
        "--raw-dir",
        RAW,
        "--output",
        PROCESSED / "all_player_roster_2022.csv",
    )
    if not args.skip_download:
        run(
            "scripts/download_daily_pageviews.py",
            "--roster",
            PROCESSED / "all_player_roster_2022.csv",
            "--cache-dir",
            ROOT / "data/raw/pageviews/WC-2022",
            "--manifest",
            ROOT / "metadata/all_player_pageview_manifest.csv",
            "--output",
            requested,
            "--user-agent",
            args.user_agent,
        )
        run(
            "scripts/download_resolved_title_pageviews.py",
            "--mapping",
            mapping,
            "--cache-dir",
            ROOT / "data/raw/pageviews/WC-2022-resolved",
            "--output",
            resolved,
            "--user-agent",
            args.user_agent,
        )

    run(
        "scripts/combine_pageviews.py",
        "--requested",
        requested,
        "--resolved",
        resolved,
        "--mapping",
        mapping,
        "--output",
        PROCESSED / "all_player_daily_pageviews_2022.csv",
    )
    run(
        "scripts/run_all_player_daily.py",
        "--raw-dir",
        RAW,
        "--pageviews",
        PROCESSED / "all_player_daily_pageviews_2022.csv",
        "--processed-dir",
        PROCESSED,
        "--output-dir",
        ROOT / "outputs",
    )
    run(
        "scripts/run_all_player_conversion.py",
        "--raw-dir",
        RAW,
        "--panel",
        PROCESSED / "all_player_day_panel_2022.csv",
        "--processed-dir",
        PROCESSED,
        "--output-dir",
        ROOT / "outputs",
    )
    run(
        "scripts/validate_daily_panel.py",
        "--panel",
        PROCESSED / "all_player_day_panel_2022.csv",
        "--roster",
        PROCESSED / "all_player_roster_2022.csv",
        "--output",
        ROOT / "metadata/full_panel_validation.json",
        "--hashes",
        ROOT / "metadata/SHA256SUMS_DAILY.txt",
    )
    subprocess.run(["bash", str(CODE / "make_checksums.sh")], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
