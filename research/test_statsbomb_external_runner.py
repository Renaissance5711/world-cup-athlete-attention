import math
import pandas as pd
import pytest

from statsbomb_external_runner import (
    period_seconds,
    flatten_pass_event,
    flatten_goal_event,
    build_pass_spells,
    build_two_minute_windows,
    add_spell_labels,
    validate_coverage,
)


def test_period_seconds_is_period_relative():
    assert period_seconds('00:02:03.500') == pytest.approx(123.5)


def test_flatten_pass_event_harmonizes_coordinates_and_flags():
    e = {
        'id':'p1','index':7,'period':1,'timestamp':'00:02:03.500',
        'type':{'name':'Pass'},'possession':4,
        'team':{'id':10,'name':'A'},'player':{'id':99,'name':'P'},
        'location':[24.0, 16.0],
        'pass':{
            'end_location':[60.0, 40.0],
            'height':{'name':'High Pass'},
            'body_part':{'name':'Head'},
            'cross':True,
        },
    }
    r = flatten_pass_event(e, 'England', 123, '2015-08-08')
    assert r['x1'] == pytest.approx(20.0)
    assert r['y1'] == pytest.approx(20.0)
    assert r['x2'] == pytest.approx(50.0)
    assert r['y2'] == pytest.approx(50.0)
    assert r['mean_dx'] == pytest.approx(30.0)
    assert r['mean_abs_dy'] == pytest.approx(30.0)
    assert r['mean_pass_length'] == pytest.approx(math.sqrt(1800))
    assert r['forward_share'] == 1.0
    assert r['long_share'] == 1.0
    assert r['high_share'] == 1.0
    assert r['head_share'] == 1.0
    assert r['cross_share'] == 1.0
    assert r['complete'] == 1


def test_flatten_pass_event_drops_extra_time_and_missing_locations():
    base = {'type':{'name':'Pass'}, 'team':{'id':1,'name':'A'}, 'pass':{'end_location':[2,2]}, 'location':[1,1], 'timestamp':'00:00:01.0'}
    assert flatten_pass_event({**base, 'period':3}, 'X', 1, '2015-01-01') is None
    bad = {**base, 'period':1, 'location':None}
    assert flatten_pass_event(bad, 'X', 1, '2015-01-01') is None


def test_flatten_goal_event_identifies_conceding_team():
    e = {
        'id':'g1','index':100,'period':2,'timestamp':'00:10:00.000',
        'type':{'name':'Shot'},'team':{'id':1,'name':'A'},
        'shot':{'outcome':{'name':'Goal'}, 'type':{'name':'Open Play'}},
    }
    match = {
        'match_id':55,
        'home_team':{'home_team_id':1,'home_team_name':'A'},
        'away_team':{'away_team_id':2,'away_team_name':'B'},
        'match_date':'2015-01-01',
    }
    r = flatten_goal_event(e, 'Spain', match)
    assert r['scoring_team_id'] == 1
    assert r['conceding_team_id'] == 2
    assert r['t'] == pytest.approx(600.0)
    assert r['shot_type'] == 'Open Play'


def test_pass_spells_reset_at_half_and_team_change():
    d = pd.DataFrame([
        {'game_id':1,'half':1,'t':1,'event_index':1,'team_id':10},
        {'game_id':1,'half':1,'t':2,'event_index':2,'team_id':10},
        {'game_id':1,'half':1,'t':3,'event_index':3,'team_id':20},
        {'game_id':1,'half':1,'t':4,'event_index':4,'team_id':20},
        {'game_id':1,'half':1,'t':5,'event_index':5,'team_id':20},
        {'game_id':1,'half':1,'t':6,'event_index':6,'team_id':20},
        {'game_id':1,'half':2,'t':1,'event_index':7,'team_id':20},
    ])
    x = build_pass_spells(d)
    starts = x[x.spell_start.eq(1)]
    assert starts.spell_depth.tolist() == [2,4,1]
    assert starts.reach4.tolist() == [0,1,0]


def test_two_minute_windows_and_spell_labels_use_spells_starting_in_window():
    base = []
    for i,t in enumerate([118,119,121,122], start=1):
        base.append({'game_id':1,'match_date':'2015-01-01','league':'England','half':1,
                     'team_id':10,'team_name':'A','t':t,'event_index':i,
                     'mean_pass_length':10.0,'mean_dx':1.0,'mean_abs_dy':2.0,'mean_x1':30.0,'mean_x2':31.0,
                     'forward_share':1.0,'long_share':0.0,'final3_start_share':0.0,'final3_end_share':0.0,
                     'high_share':0.0,'head_share':0.0,'cross_share':0.0})
    for j,t in enumerate([123,124,125], start=5):
        base.append({'game_id':1,'match_date':'2015-01-01','league':'England','half':1,
                     'team_id':20,'team_name':'B','t':t,'event_index':j,
                     'mean_pass_length':20.0,'mean_dx':-1.0,'mean_abs_dy':1.0,'mean_x1':60.0,'mean_x2':59.0,
                     'forward_share':0.0,'long_share':0.0,'final3_start_share':0.0,'final3_end_share':0.0,
                     'high_share':0.0,'head_share':0.0,'cross_share':0.0})
    d = pd.DataFrame(base)
    sp = build_pass_spells(d)
    w = build_two_minute_windows(d)
    out = add_spell_labels(sp,w)
    assert len(out) == 1
    r = out.iloc[0]
    assert r.team_id == 20
    assert r.bin2 == 1
    assert r.spell_starts == 1
    assert r.mean_spell_depth == pytest.approx(3.0)
    assert r.reach4_share == pytest.approx(0.0)


def test_validate_coverage_fails_loudly():
    good = {'England':380,'France':377,'Germany':306,'Spain':380,'Italy':380}
    assert validate_coverage(good) == 1823
    bad = dict(good); bad['Germany'] = 34
    with pytest.raises(RuntimeError, match='Germany'):
        validate_coverage(bad)
