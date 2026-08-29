import numpy as np
import pandas as pd
import pytest

from research.statsbomb_replication_analysis import (
    make_temporal_folds,
    sufficient_rows,
    config_from_interval,
    historical_marginality_targets,
    temporal_oos_predictions,
    lolo_temporal_predictions,
)

FEATURES=[
    'mean_pass_length','mean_dx','mean_abs_dy','mean_x1','mean_x2',
    'forward_share','long_share','final3_start_share','final3_end_share',
    'high_share','head_share','cross_share'
]


def synthetic_windows():
    rows=[]
    for ck in range(1,17):
        for li,league in enumerate(['England','Spain']):
            for rep in range(4):
                high=(ck+rep+li)%2
                r={'game_chron_key':ck,'league':league,'half':1,'bin2':rep,'team_id':1+li,
                   'spell_starts':10,'reach4_share':0.8 if high else 0.2}
                for j,f in enumerate(FEATURES): r[f]=float(high)+j*0.001
                rows.append(r)
    return pd.DataFrame(rows)


def test_folds_leave_first_chunk_training_only():
    matches=pd.DataFrame({'game_chron_key':np.arange(1,17)})
    f=make_temporal_folds(matches,8)
    assert len(f)==7
    assert f.iloc[0].train_before_chron==3
    assert f.iloc[-1].test_before_chron==17


def test_sufficient_rows_reconstructs_integer_counts():
    base={f:1.0 for f in FEATURES}
    d=pd.DataFrame([{**base,'league':'England','half':1,'bin2':2,'game_chron_key':10,'spell_starts':4,'reach4_share':.75}])
    z=sufficient_rows(d)
    assert z[z.y.eq(1)].w.sum()==3
    assert z[z.y.eq(0)].w.sum()==1


def test_config_uses_frozen_12d_rules():
    p=pd.DataFrame({'x1':[10,20,30],'y1':[10,10,10],'x2':[40,10,60],'y2':[10,30,10],
                    'high_share':[0,1,0],'head_share':[0,0,1],'cross_share':[0,1,0]})
    c=config_from_interval(p)
    assert c['mean_dx']==pytest.approx(50/3)
    assert c['forward_share']==pytest.approx(2/3)
    assert c['high_share']==pytest.approx(1/3)


def test_marginality_scores_extreme_target_higher():
    rows=[]
    for ck in range(1,31):
        for team in (1,2,3):
            r={'game_chron_key':ck,'league':'England','team_id':team}
            for j,f in enumerate(FEATURES):r[f]=50+(team-2)*.2+ck*.01+j*.001
            rows.append(r)
    w=pd.DataFrame(rows); targets=[]
    for key,val in [('near',50.2),('far',90.0)]:
        r={'target_key':key,'game_chron_key':31,'league':'England','team_id':1}
        for f in FEATURES:r[f]=val
        targets.append(r)
    q=historical_marginality_targets(w,pd.DataFrame(targets),min_history=20,max_calibration=1000).set_index('target_key')
    assert q.loc['far','component_atyp_pct']>q.loc['near','component_atyp_pct']


def test_temporal_and_lolo_predictions_cover_all_test_blocks():
    w=synthetic_windows(); matches=pd.DataFrame({'game_chron_key':np.arange(1,17)})
    folds=make_temporal_folds(matches,8)
    a=temporal_oos_predictions(w,folds); b=lolo_temporal_predictions(w,folds)
    assert set(a.block.unique())==set(range(1,8))
    assert set(b.heldout_league.unique())=={'England','Spain'}
    assert set(b.block.unique())==set(range(1,8))
    assert a.pred.notna().all() and b.pred.notna().all()
