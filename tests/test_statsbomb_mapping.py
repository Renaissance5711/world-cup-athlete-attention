import pandas as pd

from ijsms.statsbomb_mapping import build_match_mapping, map_lineup_players, summarize_lineup_participation


def test_match_mapping_uses_date_and_ordered_teams():
    statsbomb = [{
        'match_id': 123,
        'match_date': '2022-11-20',
        'home_team': {'home_team_name': 'Qatar'},
        'away_team': {'away_team_name': 'Ecuador'},
    }]
    fjelstul = pd.DataFrame([
        {'match_id': 'M-2022-01', 'match_date': '2022-11-20', 'home_team_name': 'Qatar', 'away_team_name': 'Ecuador'}
    ])
    mapping = build_match_mapping(statsbomb, fjelstul)
    assert mapping.loc[0, 'sb_match_id'] == 123
    assert mapping.loc[0, 'match_id'] == 'M-2022-01'


def test_lineup_mapping_uses_match_team_and_jersey_number():
    lineups = {123: [{'team_name': 'Qatar', 'lineup': [
        {'player_id': 77, 'player_name': 'Pedro Miguel Correia', 'jersey_number': 2, 'positions': [{'start_reason': 'Starting XI'}]}
    ]}]}
    match_mapping = pd.DataFrame([{'sb_match_id': 123, 'match_id': 'M-2022-01'}])
    appearances = pd.DataFrame([
        {'match_id': 'M-2022-01', 'team_name': 'Qatar', 'shirt_number': 2, 'player_id': 'P-00052'}
    ])
    mapped, audit = map_lineup_players(lineups, match_mapping, appearances)
    assert mapped.loc[0, 'player_id'] == 'P-00052'
    assert mapped.loc[0, 'sb_player_id'] == 77
    assert audit.loc[0, 'mapping_status'] == 'mapped'


def test_summarize_lineup_participation_sums_tactical_segments_and_final_whistle():
    teams = [{'team_name': 'A', 'lineup': [{
        'player_id': 1,
        'player_name': 'Player One',
        'jersey_number': 9,
        'positions': [
            {'position': 'Center Forward', 'from': '00:00', 'to': '60:00', 'start_reason': 'Starting XI'},
            {'position': 'Right Wing', 'from': '60:00', 'to': None, 'start_reason': 'Tactical Shift'},
        ],
    }]}]
    result = summarize_lineup_participation(teams, sb_match_id=123, match_end_seconds=95 * 60)
    row = result.iloc[0]
    assert row['minutes_played'] == 95
    assert row['sb_starter'] == 1
    assert row['first_position'] == 'Center Forward'
