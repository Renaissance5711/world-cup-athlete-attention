from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups
from scipy.stats import spearmanr, norm

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'base_archive/code'
sys.path.insert(0,str(SRC))
from src.workflow import add_event_window_counts, add_post_event_windows

OUT=ROOT/'outputs/r24'; OUT.mkdir(exist_ok=True)

def codes(s): return pd.Categorical(s.astype(str)).codes

def fit_event(df,outcome):
    data=df.dropna(subset=[outcome,'baseline_visibility_z']).copy()
    formula=f"{outcome} ~ scored_int + scored_int:baseline_visibility_z + starter + C(player_id) + C(match_id)"
    res=smf.ols(formula,data=data).fit()
    cov_arr,_,_=cov_cluster_2groups(res,codes(data.player_id),codes(data.match_id))
    names=res.model.exog_names
    cov=pd.DataFrame(cov_arr,index=names,columns=names)
    rows=[]
    for term in ['scored_int','scored_int:baseline_visibility_z']:
        b=float(res.params[term]); se=float(np.sqrt(max(cov.loc[term,term],0)))
        rows.append({'outcome':outcome,'term':term,'coefficient':b,'std_error':se,'ci_low':b-1.96*se,'ci_high':b+1.96*se,'p_value':float(2*norm.sf(abs(b/se))) if se else np.nan,'n':len(data),'players':data.player_id.nunique(),'matches':data.match_id.nunique()})
    return pd.DataFrame(rows)

def fit_daily(data, latest=False):
    x=data.copy(); x['date']=pd.to_datetime(x.date)
    if latest:
        x=add_post_event_windows(x,'goal_today','goal')
        x=add_post_event_windows(x,'played_today','appearance')
    else:
        x=add_event_window_counts(x,'goal_today','goal')
        x=add_event_window_counts(x,'played_today','appearance')
    baseline=x.groupby('player_id').baseline_log_views.first()
    z=(baseline-baseline.mean())/baseline.std(ddof=0)
    x['baseline_visibility_z']=x.player_id.map(z)
    g=['goal_w0_1','goal_w2_7','goal_w8_30']
    ints=[f'{v}:baseline_visibility_z' for v in g]
    a=['appearance_w0_1','appearance_w2_7','appearance_w8_30']
    for term,base in zip(ints,g): x[term]=x[base]*x['baseline_visibility_z']
    cols=g+ints+a
    y=x['log_views'].astype(float)
    X=x[cols].astype(float)
    y_t=y-y.groupby(x.player_id).transform('mean')-y.groupby(x.date).transform('mean')+y.mean()
    X_t=X-X.groupby(x.player_id).transform('mean')-X.groupby(x.date).transform('mean')+X.mean()
    res=__import__('statsmodels.api',fromlist=['OLS']).OLS(y_t.to_numpy(),X_t.to_numpy()).fit()
    cov_arr,_,_=cov_cluster_2groups(res,codes(x.player_id),codes(x.date))
    cov=pd.DataFrame(cov_arr,index=cols,columns=cols)
    params=pd.Series(res.params,index=cols)
    rows=[]
    for term in ints:
        b=float(params[term]); se=float(np.sqrt(max(cov.loc[term,term],0)))
        rows.append({'specification':'latest_event' if latest else 'event_counts','term':term,'coefficient':b,'std_error':se,'ci_low':b-1.96*se,'ci_high':b+1.96*se,'p_value':float(2*norm.sf(abs(b/se))) if se else np.nan,'n':len(x),'players':x.player_id.nunique(),'dates':x.date.nunique()})
    aterm,bterm=ints[0],ints[2]
    est=float(params[aterm]-params[bterm])
    var=float(cov.loc[aterm,aterm]+cov.loc[bterm,bterm]-2*cov.loc[aterm,bterm]); se=np.sqrt(max(var,0))
    contrast={'specification':'latest_event' if latest else 'event_counts','contrast':'0-1 minus 8-30','estimate':est,'std_error':se,'p_value':float(2*norm.sf(abs(est/se))) if se else np.nan}
    idx={n:i for i,n in enumerate(cols)}
    R=np.zeros((2,len(cols))); R[0,idx[ints[0]]]=1; R[0,idx[ints[1]]]=-1; R[1,idx[ints[1]]]=1; R[1,idx[ints[2]]]=-1
    diff=R@params.to_numpy(); V=R@cov.to_numpy()@R.T
    stat=float(diff.T@np.linalg.pinv(V)@diff)
    from scipy.stats import chi2
    contrast_joint={'specification':'latest_event' if latest else 'event_counts','contrast':'joint equality','estimate':stat,'std_error':np.nan,'p_value':float(chi2.sf(stat,2))}
    return pd.DataFrame(rows), pd.DataFrame([contrast,contrast_joint]), x

events=pd.read_csv(ROOT/'base_archive/data/processed/all_player_match_outcomes_2022.csv')
daily_raw=pd.read_csv(ROOT/'base_archive/data/processed/all_player_daily_pageviews_2022.csv',parse_dates=['date'])
day=pd.read_csv(ROOT/'base_archive/data/processed/all_player_day_analysis_panel_2022.csv',parse_dates=['date'])
roster=pd.read_csv(ROOT/'base_archive/data/processed/all_player_roster_2022.csv')
zero_players=sorted(daily_raw.loc[daily_raw.structural_zero.eq(1),'player_id'].unique())

scoring=events.loc[events.scored_int.eq(1)].copy()
obs=(scoring.groupby('player_id',as_index=False)
     .agg(observed_proportional_log_lift=('immediate_attention_log_lift','mean'),
          observed_additional_pageviews=('winsorized_additional_pageviews','mean'),
          scoring_appearances=('match_id','nunique'),baseline_views=('baseline_views','first'),baseline_visibility_z=('baseline_visibility_z','first'))
     .merge(roster[['player_id','player_name','team_name','position_code']],on='player_id',how='left'))
obs['observed_percent_change']=100*np.expm1(obs.observed_proportional_log_lift)
obs['proportional_rank']=obs.observed_proportional_log_lift.rank(method='min',ascending=False).astype(int)
obs['additive_rank']=obs.observed_additional_pageviews.rank(method='min',ascending=False).astype(int)
n=len(obs); k=int(np.ceil(.10*n))
rho,p=spearmanr(obs.proportional_rank,obs.additive_rank)
tp=set(obs.nsmallest(k,'proportional_rank').player_id); ta=set(obs.nsmallest(k,'additive_rank').player_id)
obs_summary={'aggregation':'mean across scoring appearances','unique_scorers':n,'top_decile_n':k,'spearman_rho':float(rho),'spearman_p':float(p),'top_decile_overlap_n':len(tp&ta),'top_decile_overlap_percent':100*len(tp&ta)/k,'zero_coded_players_in_scorers':int(obs.player_id.isin(zero_players).sum())}
obs.sort_values('proportional_rank').to_csv(OUT/'observed_goal_scorer_metric_rankings.csv',index=False)

first=(scoring.sort_values(['player_id','match_date','match_id']).drop_duplicates('player_id')
       [['player_id','immediate_attention_log_lift','winsorized_additional_pageviews']].copy())
first['prop_rank']=first.immediate_attention_log_lift.rank(method='min',ascending=False)
first['add_rank']=first.winsorized_additional_pageviews.rank(method='min',ascending=False)
rho1,p1=spearmanr(first.prop_rank,first.add_rank); tp1=set(first.nsmallest(k,'prop_rank').player_id); ta1=set(first.nsmallest(k,'add_rank').player_id)
obs_summary['first_scoring_appearance_spearman_rho']=float(rho1); obs_summary['first_scoring_appearance_top_decile_overlap_n']=len(tp1&ta1)

main=[]
for label,df in [('primary',events),('exclude_any_structural_zero_player',events.loc[~events.player_id.isin(zero_players)])]:
    for outcome in ['immediate_attention_log_lift','winsorized_additional_pageviews']:
        r=fit_event(df,outcome); r.insert(0,'specification',label); main.append(r)
main=pd.concat(main,ignore_index=True); main.to_csv(OUT/'structural_zero_event_model_sensitivity.csv',index=False)

r_count,c_count,x_count=fit_daily(day,latest=False)
r_latest,c_latest,x_latest=fit_daily(day,latest=True)
r_zero,c_zero,_=fit_daily(day.loc[~day.player_id.isin(zero_players)],latest=False)
r_zero['specification']='event_counts_exclude_structural_zero_players'; c_zero['specification']='event_counts_exclude_structural_zero_players'
pd.concat([r_count,r_latest,r_zero],ignore_index=True).to_csv(OUT/'dynamic_window_sensitivity.csv',index=False)
pd.concat([c_count,c_latest,c_zero],ignore_index=True).to_csv(OUT/'dynamic_window_contrast_sensitivity.csv',index=False)

counts=add_event_window_counts(day.copy(),'goal_today','goal')
counts['total_goal_window_exposures']=counts[['goal_w0_1','goal_w2_7','goal_w8_30']].sum(axis=1)
overlap_summary={
    'player_days':len(counts),
    'player_days_with_any_goal_window':int((counts.total_goal_window_exposures>0).sum()),
    'player_days_with_more_than_one_goal_event_in_any_window':int((counts[['goal_w0_1','goal_w2_7','goal_w8_30']].max(axis=1)>1).sum()),
    'player_days_with_total_goal_exposures_more_than_one':int((counts.total_goal_window_exposures>1).sum()),
    'maximum_total_goal_exposures':int(counts.total_goal_window_exposures.max()),
    'unique_players_with_overlapping_goal_exposure':int(counts.loc[counts.total_goal_window_exposures>1,'player_id'].nunique()),
}

summary={'structural_zero_players':zero_players,'n_structural_zero_players':len(zero_players),'n_structural_zero_dates':int(daily_raw.structural_zero.sum()),'n_structural_zero_dates_analysis_period':int(daily_raw.loc[daily_raw.date.between('2022-11-20','2023-01-20'),'structural_zero'].sum()),'observed_ranking':obs_summary,'overlap':overlap_summary}
(OUT/'sensitivity_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\nEVENT SENSITIVITY\n',main.to_string(index=False))
print('\nDYNAMIC\n',pd.concat([r_count,r_latest,r_zero],ignore_index=True).to_string(index=False))
print('\nCONTRASTS\n',pd.concat([c_count,c_latest,c_zero],ignore_index=True).to_string(index=False))
