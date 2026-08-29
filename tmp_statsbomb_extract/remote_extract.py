from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from statsbomb_raw_batch import build_match_anchor_candidates

PINNED_COMMIT = '3bfbffe1de5750ebd47d770be0bb924a10cde54f'
RAW_BASE = f'https://raw.githubusercontent.com/hudl/open-data/{PINNED_COMMIT}/data/events'
EXPECTED_MATCHES = 1823
EXPECTED_GOALS = 4863
EXPECTED_GOALS_BY_LEAGUE = {
    'England': 1026,
    'France': 949,
    'Germany': 866,
    'Spain': 1043,
    'Italy': 979,
}


def _shot_outcome_name(e):
    if not isinstance(e, dict) or (e.get('type') or {}).get('name') != 'Shot':
        return None
    sh = e.get('shot') if isinstance(e.get('shot'), dict) else {}
    out = sh.get('outcome') if isinstance(sh.get('outcome'), dict) else {}
    return out.get('name')


def _is_score_goal_event(e):
    if not isinstance(e, dict):
        return False
    event_type = (e.get('type') or {}).get('name')
    return event_type == 'Own Goal For' or (event_type == 'Shot' and _shot_outcome_name(e) == 'Goal')


def process_match_payload(game_id: int, league: str, events: list[dict]) -> dict:
    goal_count = sum(_is_score_goal_event(e) for e in events)
    anchors_df = build_match_anchor_candidates(events, match_id=int(game_id), league=str(league))
    anchors = anchors_df.to_dict(orient='records') if not anchors_df.empty else []
    return {
        'game_id': int(game_id),
        'league': str(league),
        'goal_count': int(goal_count),
        'event_count': int(len(events)),
        'anchors': anchors,
    }


def validate_summary(summary: dict) -> tuple[bool, list[str]]:
    problems = []
    if int(summary.get('matches_processed', -1)) != EXPECTED_MATCHES:
        problems.append(f"matches_processed={summary.get('matches_processed')} expected={EXPECTED_MATCHES}")
    if int(summary.get('goal_total', -1)) != EXPECTED_GOALS:
        problems.append(f"goal_total={summary.get('goal_total')} expected={EXPECTED_GOALS}")
    got = summary.get('goals_by_league') or {}
    for league, expected in EXPECTED_GOALS_BY_LEAGUE.items():
        if int(got.get(league, -1)) != expected:
            problems.append(f"{league} goals={got.get(league)} expected={expected}")
    if int(summary.get('failures', 0)) != 0:
        problems.append(f"failures={summary.get('failures')} expected=0")
    return not problems, problems


def fetch_events(game_id: int, attempts: int = 6, timeout: float = 90.0) -> list[dict]:
    url = f'{RAW_BASE}/{int(game_id)}.json'
    last = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'statsbomb-replication-extractor/1.0'})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as exc:
            last = exc
            if attempt == attempts:
                break
            time.sleep(min(2 ** (attempt - 1), 20))
    raise RuntimeError(f'failed game_id={game_id} after {attempts} attempts: {last}')


def _load_match_rows(path: Path) -> list[tuple[int, str]]:
    rows = []
    seen = set()
    with path.open(newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            gid = int(r['game_id'])
            league = str(r['league'])
            if gid in seen:
                raise ValueError(f'duplicate game_id in match list: {gid}')
            seen.add(gid)
            rows.append((gid, league))
    if len(rows) != EXPECTED_MATCHES:
        raise ValueError(f'match list has {len(rows)} rows; expected {EXPECTED_MATCHES}')
    return rows


def _one(row):
    gid, league = row
    events = fetch_events(gid)
    return process_match_payload(gid, league, events)


def run(match_list: Path, out_dir: Path, workers: int = 12) -> tuple[dict, bool]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _load_match_rows(match_list)
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = {ex.submit(_one, row): row for row in rows}
        done = 0
        for fut in as_completed(futs):
            gid, league = futs[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                failures.append({'game_id': gid, 'league': league, 'error': repr(exc)})
            done += 1
            if done % 100 == 0 or done == len(rows):
                print(f'processed {done}/{len(rows)} failures={len(failures)}', flush=True)

    anchors = []
    goals_by_league = Counter()
    event_total = 0
    for r in results:
        goals_by_league[r['league']] += r['goal_count']
        event_total += r['event_count']
        anchors.extend(r['anchors'])

    anchors_df = pd.DataFrame(anchors)
    if not anchors_df.empty:
        anchors_df = anchors_df.sort_values(['game_id', 'half', 't', 'event_index'], kind='mergesort')
    anchors_path = out_dir / 'StatsBomb_2015_16_Raw_Anchor_Candidates.csv.gz'
    anchors_df.to_csv(anchors_path, index=False, compression='gzip')
    pd.DataFrame(failures).to_csv(out_dir / 'StatsBomb_2015_16_Raw_Failures.csv', index=False)

    summary = {
        'source': 'hudl/open-data',
        'pinned_commit': PINNED_COMMIT,
        'match_list_rows': len(rows),
        'matches_processed': len(results),
        'failures': len(failures),
        'event_total': int(event_total),
        'goal_total': int(sum(goals_by_league.values())),
        'goals_by_league': dict(sorted(goals_by_league.items())),
        'eligible_anchor_candidates': int(len(anchors_df)),
        'goal_anchor_candidates': int(anchors_df['goal_treat'].sum()) if not anchors_df.empty else 0,
        'miss_anchor_candidates': int((1 - anchors_df['goal_treat']).sum()) if not anchors_df.empty else 0,
    }
    ok, problems = validate_summary(summary)
    summary['hard_validation_ok'] = bool(ok)
    summary['hard_validation_problems'] = problems
    (out_dir / 'StatsBomb_2015_16_Raw_Extraction_Summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary, ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--match-list', required=True, type=Path)
    p.add_argument('--out-dir', required=True, type=Path)
    p.add_argument('--workers', type=int, default=int(os.environ.get('WORKERS', '12')))
    args = p.parse_args()
    _, ok = run(args.match_list, args.out_dir, args.workers)
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()
