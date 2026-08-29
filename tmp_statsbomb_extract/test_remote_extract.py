from remote_extract import process_match_payload, validate_summary


def ev(idx, period, ts, typ, team=1, loc=None, end=None, outcome=None, shot_type='Open Play', xg=None):
    d={'id':f'e{idx}','index':idx,'period':period,'timestamp':ts,
       'type':{'id':0,'name':typ},'team':{'id':team,'name':f'T{team}'}}
    if loc is not None: d['location']=loc
    if typ=='Pass': d['pass']={'end_location':end}
    if typ=='Carry': d['carry']={'end_location':end}
    if typ=='Shot':
        d['shot']={'outcome':{'id':0,'name':outcome},'type':{'id':0,'name':shot_type},'end_location':end or [120,40,1]}
        if xg is not None: d['shot']['statsbomb_xg']=xg
    return d


def test_process_match_payload_counts_all_goals_and_emits_frozen_eligible_anchor():
    events=[
        ev(1,1,'00:05:00.000','Pass',team=2,loc=[60,40],end=[100,40]),
        ev(2,1,'00:06:00.000','Carry',team=2,loc=[50,40],end=[55,40]),
        ev(3,1,'00:10:00.000','Shot',team=1,loc=[108,40],outcome='Goal',xg=.4),
        ev(4,1,'00:13:30.000','Pass',team=2,loc=[70,40],end=[110,40]),
        ev(5,1,'00:14:00.000','Carry',team=2,loc=[50,40],end=[70,40]),
        ev(6,1,'00:20:00.000','Shot',team=2,loc=[100,30],outcome='Saved',xg=.2),
        ev(7,1,'00:25:00.000','Shot',team=1,loc=[100,30],outcome='Goal',shot_type='Free Kick',xg=.1),
    ]
    r=process_match_payload(77,'England',events)
    assert r['goal_count']==2
    assert len(r['anchors'])==2
    assert [a['anchor_class'] for a in r['anchors']]==['Goal','Miss']
    assert r['anchors'][0]['defending_team_id']==2
    assert r['anchors'][0]['pre_n_carries']==1


def test_validate_summary_rejects_any_frozen_count_mismatch():
    summary={'matches_processed':1823,'goal_total':4862,'goals_by_league':{
        'England':1026,'France':949,'Germany':866,'Spain':1043,'Italy':978}}
    ok, problems=validate_summary(summary)
    assert not ok
    assert any('goal_total' in p for p in problems)
    assert any('Italy' in p for p in problems)
