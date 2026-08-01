#!/usr/bin/env python3
from pathlib import Path
import json
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'base_archive'
OUT=ROOT/'outputs/ranking'; OUT.mkdir(parents=True,exist_ok=True)
def codes(s): return pd.Categorical(s.astype(str)).codes
def fit(events,outcome):
    d=events.dropna(subset=[outcome,'baseline_visibility_z']).copy()
    r=smf.ols(f"{outcome} ~ scored_int + scored_int:baseline_visibility_z + starter + C(player_id) + C(match_id)",data=d).fit()
    cov_arr,_,_=cov_cluster_2groups(r,codes(d.player_id),codes(d.match_id))
    cov=pd.DataFrame(cov_arr,index=r.model.exog_names,columns=r.model.exog_names)
    return r,cov
events=pd.read_csv(BASE/'data/processed/all_player_match_outcomes_2022.csv')
roster=pd.read_csv(BASE/'data/processed/all_player_roster_2022.csv')
prop,_=fit(events,'immediate_attention_log_lift'); add,_=fit(events,'winsorized_additional_pageviews')
sc=(events.loc[events.scored_int.eq(1),['player_id','baseline_visibility_z','baseline_views']].drop_duplicates('player_id')
    .merge(roster[['player_id','player_name','team_name','position_code']],on='player_id',how='left'))
sc['predicted_log_effect']=prop.params['scored_int']+prop.params['scored_int:baseline_visibility_z']*sc.baseline_visibility_z
sc['predicted_percent_change']=100*np.expm1(sc.predicted_log_effect)
sc['predicted_additional_pageviews']=add.params['scored_int']+add.params['scored_int:baseline_visibility_z']*sc.baseline_visibility_z
sc['proportional_rank']=sc.predicted_percent_change.rank(method='min',ascending=False).astype(int)
sc['additive_rank']=sc.predicted_additional_pageviews.rank(method='min',ascending=False).astype(int)
sc['rank_shift']=sc.additive_rank-sc.proportional_rank
n=len(sc); k=int(np.ceil(.10*n)); A=set(sc.nsmallest(k,'proportional_rank').player_id); B=set(sc.nsmallest(k,'additive_rank').player_id)
summary={'unique_scorers':n,'top_10_percent_n':k,'spearman_rho':float(spearmanr(sc.proportional_rank,sc.additive_rank).statistic),'top_10_percent_overlap_n':len(A&B),'interpretation':'Conditional on the fitted monotonic linear interactions; not an independent behavioural finding.'}
sc.sort_values('proportional_rank').to_csv(OUT/'fitted_goal_scorer_metric_rankings.csv',index=False)
(OUT/'fitted_ranking_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
