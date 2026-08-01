from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import requests

BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master"
COMPETITION_ID = 43
SEASON_ID = 106


def build_open_data_urls(match_ids: Iterable[int]) -> dict[str, object]:
    ids = [int(match_id) for match_id in match_ids]
    return {
        "matches": f"{BASE_URL}/data/matches/{COMPETITION_ID}/{SEASON_ID}.json",
        "events": {match_id: f"{BASE_URL}/data/events/{match_id}.json" for match_id in ids},
        "lineups": {match_id: f"{BASE_URL}/data/lineups/{match_id}.json" for match_id in ids},
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_local_open_data(root: Path) -> dict[str, object]:
    root = Path(root)
    match_path = root / "matches_43_106.json"
    if not match_path.exists():
        raise FileNotFoundError(match_path)
    matches = json.loads(match_path.read_text(encoding="utf-8"))
    match_ids = {int(match["match_id"]) for match in matches}
    event_files = {int(path.stem) for path in (root / "events").glob("*.json")}
    lineup_files = {int(path.stem) for path in (root / "lineups").glob("*.json")}
    missing_events = sorted(match_ids - event_files)
    missing_lineups = sorted(match_ids - lineup_files)
    return {
        "competition_id": COMPETITION_ID,
        "season_id": SEASON_ID,
        "match_count": len(match_ids),
        "event_file_count": len(event_files),
        "lineup_file_count": len(lineup_files),
        "missing_events": missing_events,
        "missing_lineups": missing_lineups,
        "complete": len(match_ids) == 64 and not missing_events and not missing_lineups,
    }


def download_statsbomb_world_cup(output_dir: Path, timeout: float = 60.0) -> dict[str, object]:
    output_dir = Path(output_dir)
    events_dir = output_dir / "events"
    lineups_dir = output_dir / "lineups"
    events_dir.mkdir(parents=True, exist_ok=True)
    lineups_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "IJSMS-replication/1.0 (academic research)"})
    matches_url = build_open_data_urls([])["matches"]
    response = session.get(str(matches_url), timeout=timeout)
    response.raise_for_status()
    match_path = output_dir / "matches_43_106.json"
    match_path.write_bytes(response.content)
    matches = response.json()
    match_ids = [int(match["match_id"]) for match in matches]
    urls = build_open_data_urls(match_ids)

    retrievals: list[dict[str, object]] = []
    for kind, directory in (("events", events_dir), ("lineups", lineups_dir)):
        for match_id, url in urls[kind].items():
            destination = directory / f"{match_id}.json"
            if not destination.exists():
                item = session.get(url, timeout=timeout)
                item.raise_for_status()
                destination.write_bytes(item.content)
            retrievals.append(
                {
                    "match_id": int(match_id),
                    "kind": kind,
                    "url": url,
                    "bytes": destination.stat().st_size,
                    "sha256": _sha256(destination),
                }
            )

    report = validate_local_open_data(output_dir)
    report["retrievals"] = retrievals
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
