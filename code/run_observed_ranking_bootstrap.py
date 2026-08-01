from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EVENTS = ROOT/'base_archive/data/processed/all_player_match_outcomes_2022.csv'
ROSTER = ROOT/'base_archive/data/processed/all_player_roster_2022.csv'
OUT = ROOT/'outputs/r24'
OUT.mkdir(exist_ok=True)
SEED = 20260730
B = 10000
TOP_SHARES = [0.05, 0.10, 0.15, 0.20]

rng = np.random.default_rng(SEED)
events = pd.read_csv(EVENTS)
roster = pd.read_csv(ROSTER)
scoring = events.loc[events['scored_int'].eq(1), [
    'player_id','match_id','match_date','immediate_attention_log_lift',
    'winsorized_additional_pageviews','baseline_views','baseline_visibility_z'
]].copy()

obs = (scoring.groupby('player_id', as_index=False)
       .agg(observed_proportional_log_lift=('immediate_attention_log_lift','mean'),
            observed_additional_pageviews=('winsorized_additional_pageviews','mean'),
            scoring_appearances=('match_id','nunique'),
            baseline_views=('baseline_views','first'),
            baseline_visibility_z=('baseline_visibility_z','first'))
       .merge(roster[['player_id','player_name','team_name','position_code']], on='player_id', how='left'))
obs['observed_percent_change'] = 100*np.expm1(obs['observed_proportional_log_lift'])
obs['proportional_rank'] = obs['observed_proportional_log_lift'].rank(method='min', ascending=False).astype(int)
obs['additive_rank'] = obs['observed_additional_pageviews'].rank(method='min', ascending=False).astype(int)
obs['rank_difference'] = obs['additive_rank'] - obs['proportional_rank']
obs.sort_values('proportional_rank').to_csv(OUT/'observed_goal_scorer_metric_rankings.csv', index=False)

n = len(obs)
rho, rho_p = spearmanr(obs['observed_proportional_log_lift'], obs['observed_additional_pageviews'])

# Paired-athlete nonparametric bootstrap of Spearman correlation.
# Because the inputs are ranks, row-wise Pearson correlation of resampled rank pairs is the bootstrap statistic.
rp = obs['proportional_rank'].to_numpy(dtype=float)
ra = obs['additive_rank'].to_numpy(dtype=float)
idx = rng.integers(0, n, size=(B, n))
x = rp[idx]; y = ra[idx]
xm = x.mean(axis=1, keepdims=True); ym = y.mean(axis=1, keepdims=True)
num = ((x-xm)*(y-ym)).sum(axis=1)
den = np.sqrt(((x-xm)**2).sum(axis=1)*((y-ym)**2).sum(axis=1))
rho_boot = num/den

# Match-weighted Bayesian bootstrap for list stability.
# Positive Dirichlet weights retain every scorer; the same weight is shared by appearances in one match.
players = obs['player_id'].tolist()
matches = scoring['match_id'].drop_duplicates().tolist()
pidx = {p:i for i,p in enumerate(players)}
midx = {m:i for i,m in enumerate(matches)}
P, M = len(players), len(matches)
mask = np.zeros((P,M), dtype=float)
yprop = np.zeros((P,M), dtype=float)
yadd = np.zeros((P,M), dtype=float)
for r in scoring.itertuples(index=False):
    i=pidx[r.player_id]; j=midx[r.match_id]
    mask[i,j] = 1.0
    yprop[i,j] = float(r.immediate_attention_log_lift)
    yadd[i,j] = float(r.winsorized_additional_pageviews)
weights = rng.dirichlet(np.ones(M), size=B)
denom = weights @ mask.T
pvals = (weights @ yprop.T)/denom
avals = (weights @ yadd.T)/denom

threshold_rows=[]
boot_frames=[]
for share in TOP_SHARES:
    k=int(np.ceil(share*n))
    top_prop=set(obs.nlargest(k,'observed_proportional_log_lift')['player_id'])
    top_add=set(obs.nlargest(k,'observed_additional_pageviews')['player_id'])
    overlap=len(top_prop & top_add); union=len(top_prop | top_add)
    # argpartition returns top-k unordered indices for each draw.
    ip=np.argpartition(-pvals, kth=k-1, axis=1)[:,:k]
    ia=np.argpartition(-avals, kth=k-1, axis=1)[:,:k]
    overlap_draws=np.empty(B,dtype=int)
    for b in range(B):
        overlap_draws[b]=len(set(ip[b].tolist()) & set(ia[b].tolist()))
    jaccard_draws=overlap_draws/(2*k-overlap_draws)
    threshold_rows.append({
        'top_share':share,'top_k':k,'observed_overlap_n':overlap,
        'observed_overlap_percent_of_k':100*overlap/k,'observed_jaccard':overlap/union,
        'bootstrap_overlap_median':float(np.median(overlap_draws)),
        'bootstrap_overlap_ci_low':float(np.quantile(overlap_draws,.025)),
        'bootstrap_overlap_ci_high':float(np.quantile(overlap_draws,.975)),
        'bootstrap_jaccard_median':float(np.median(jaccard_draws)),
        'bootstrap_jaccard_ci_low':float(np.quantile(jaccard_draws,.025)),
        'bootstrap_jaccard_ci_high':float(np.quantile(jaccard_draws,.975)),
        'bootstrap_draws':B,
        'bootstrap_method':'match-weighted Bayesian bootstrap over scoring appearances'})
    boot_frames.append(pd.DataFrame({'draw':np.arange(B),'top_share':share,'top_k':k,'overlap_n':overlap_draws,'jaccard':jaccard_draws}))

thresholds=pd.DataFrame(threshold_rows)
thresholds.to_csv(OUT/'observed_ranking_topk_sensitivity.csv',index=False)
pd.concat(boot_frames,ignore_index=True).to_csv(OUT/'observed_ranking_topk_bootstrap_draws.csv',index=False)
pd.DataFrame({'draw':np.arange(B),'spearman_rho':rho_boot}).to_csv(OUT/'observed_ranking_spearman_bootstrap_draws.csv',index=False)
summary={
 'seed':SEED,'bootstrap_draws':B,'unique_scorers':n,
 'observed_spearman_rho':float(rho),'observed_spearman_p':float(rho_p),
 'athlete_bootstrap_spearman_median':float(np.median(rho_boot)),
 'athlete_bootstrap_spearman_ci_95':[float(np.quantile(rho_boot,.025)),float(np.quantile(rho_boot,.975))],
 'spearman_bootstrap_method':'paired nonparametric bootstrap of athlete rank pairs',
 'topk':threshold_rows,
 'topk_bootstrap_method':'match-weighted Bayesian bootstrap over scoring appearances with shared Dirichlet match weights',
 'interpretation':'Observed scorer rankings are weakly positively related, while overlap remains limited across multiple decision thresholds. Bootstrap results are conditional on the recorded scoring appearances.'}
(OUT/'observed_ranking_bootstrap_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
print('\nTOP-K TABLE\n'+thresholds.to_string(index=False))
