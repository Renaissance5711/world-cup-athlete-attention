#!/usr/bin/env python3
"""Run the frozen Stage-2 StatsBomb 2015/16 replication analyses.

Outputs:
- strict temporal-OOS J validation
- leave-one-league-out + temporal transport validation
- goal/miss pooled radicality and Goal+Extreme episode construction
- strict-prior historical marginality M on goal response/recovery states
- static M/J nonredundancy
- Figure-5-style coordinate-specific recovery audit
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import statsmodels.api as sm

from research.statsbomb_external_runner import FEATURES_12, build_pass_spells
from research.statsbomb_replication_analysis import (
    make_temporal_folds,
    config_from_interval,
    historical_marginality_targets,
    temporal_oos_predictions,
    lolo_temporal_predictions,
    score_targets_temporal,
    safe_metrics,
)

LEAGUE_SLUG = {
    "England": "england",
    "France": "france",
    "Germany": "germany",
    "Spain": "spain",
    "Italy": "italy",
}


def period_seconds(timestamp: str) -> float:
    hh, mm, ss = timestamp.split(":")
    return int(hh) * 3600 + int(mm) * 60 + float(ss)


def other_team(team_id: int, row: pd.Series) -> tuple[int | None, str | None]:
    if int(team_id) == int(row.home_team_id):
        return int(row.away_team_id), row.away_team_name
    if int(team_id) == int(row.away_team_id):
        return int(row.home_team_id), row.home_team_name
    return None, None


def extract_shot_anchors(raw_data: Path, matches: pd.DataFrame) -> pd.DataFrame:
    """Extract all Shot events plus Own Goal For as pooled Goal/Miss anchors."""
    mlookup = matches.set_index("game_id")
    rows = []
    for i, m in enumerate(matches.itertuples(index=False), 1):
        gid = int(m.game_id)
        events = json.loads((raw_data / "events" / f"{gid}.json").read_text(encoding="utf-8"))
        mr = mlookup.loc[gid]
        for e in events:
            et = e.get("type", {}).get("name")
            if et not in ("Shot", "Own Goal For"):
                continue
            period = int(e.get("period", 0))
            if period not in (1, 2):
                continue
            team = e.get("team") if isinstance(e.get("team"), dict) else {}
            if team.get("id") is None:
                continue
            actor = int(team["id"])
            focal, focal_name = other_team(actor, mr)
            if focal is None:
                continue
            shot = e.get("shot") if isinstance(e.get("shot"), dict) else {}
            outcome = shot.get("outcome", {}).get("name") if shot else None
            goal = int(et == "Own Goal For" or (et == "Shot" and outcome == "Goal"))
            rows.append({
                "anchor_id": f"{gid}:{period}:{int(e.get('index', 0))}",
                "game_id": gid,
                "match_date": m.match_date,
                "league": m.league,
                "game_chron_key": int(m.game_chron_key),
                "half": period,
                "anchor_t": period_seconds(e["timestamp"]),
                "event_index": int(e.get("index", 0)),
                "event_type": et,
                "shot_outcome": outcome,
                "actor_team_id": actor,
                "actor_team_name": team.get("name"),
                "focal_team_id": int(focal),
                "focal_team_name": focal_name,
                "goal_treat": goal,
            })
        if i % 250 == 0:
            print(f"shot anchors: scanned {i}/{len(matches)} matches", flush=True)
    return pd.DataFrame(rows)


def continuity_from_starts(starts: pd.DataFrame, lo: float, hi: float) -> tuple[float, float, int]:
    if starts is None or starts.empty:
        return float("nan"), float("nan"), 0
    t = starts["spell_start_t"].to_numpy(float)
    a, b = np.searchsorted(t, [lo, hi], side="left")
    q = starts.iloc[a:b]
    if q.empty:
        return float("nan"), float("nan"), 0
    return float(q["spell_depth"].mean()), float(q["reach4"].mean()), int(len(q))


def build_anchor_episode_panel(processed: Path, anchors: pd.DataFrame) -> pd.DataFrame:
    """Construct pre (-3,-1m), response (+3,+5m), and optional recovery (+5,+7m) states."""
    intervals = {"pre": (-180.0, -60.0), "response": (180.0, 300.0), "recovery": (300.0, 420.0)}
    all_rows = []
    for league, slug in LEAGUE_SLUG.items():
        print(f"building anchor episodes: {league}", flush=True)
        passes = pd.read_parquet(processed / f"passes_{slug}.parquet").sort_values(["game_id","half","team_id","t","event_index"])
        spells = build_pass_spells(passes)
        starts = spells[spells["spell_start"].eq(1)].sort_values(["game_id","half","team_id","spell_start_t"])
        pass_groups = {k:g.sort_values("t") for k,g in passes.groupby(["game_id","half","team_id"], sort=False)}
        start_groups = {k:g.sort_values("spell_start_t") for k,g in starts.groupby(["game_id","half","team_id"], sort=False)}
        aa = anchors[anchors["league"].eq(league)]
        for j, arow in enumerate(aa.itertuples(index=False), 1):
            key = (int(arow.game_id), int(arow.half), int(arow.focal_team_id))
            gp = pass_groups.get(key)
            if gp is None or gp.empty:
                continue
            tt = gp["t"].to_numpy(float)
            rec = {c:getattr(arow,c) for c in anchors.columns}
            valid_pre_response = True
            for state, (da, db) in intervals.items():
                lo, hi = float(arow.anchor_t + da), float(arow.anchor_t + db)
                ia, ib = np.searchsorted(tt, [lo, hi], side="left")
                q = gp.iloc[ia:ib]
                cfg = config_from_interval(q)
                if cfg is None:
                    if state in ("pre","response"):
                        valid_pre_response = False
                    rec[f"has_config__{state}"] = 0
                    for f in FEATURES_12:
                        rec[f"{f}__{state}"] = np.nan
                    rec[f"pass_n__{state}"] = int(len(q))
                else:
                    rec[f"has_config__{state}"] = 1
                    for f in FEATURES_12:
                        rec[f"{f}__{state}"] = cfg[f]
                    rec[f"pass_n__{state}"] = cfg["pass_n"]
                md, r4, ns = continuity_from_starts(start_groups.get(key), lo, hi)
                rec[f"mean_spell_depth__{state}"] = md
                rec[f"reach4_share__{state}"] = r4
                rec[f"spell_starts__{state}"] = ns
            if not valid_pre_response:
                continue
            rec["bin2_response"] = int((arow.anchor_t + 180.0) // 120.0)
            all_rows.append(rec)
            if j % 2500 == 0:
                print(f"{league}: {j}/{len(aa)} anchors", flush=True)
        del passes, spells, starts, pass_groups, start_groups
    return pd.DataFrame(all_rows)


def add_radicality(episodes: pd.DataFrame, windows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Provider-harmonized substantial reconfiguration: top quartile of pre→response 12D distance."""
    sd = windows[FEATURES_12].std(ddof=0).replace(0, 1).to_numpy(float)
    pre = episodes[[f"{f}__pre" for f in FEATURES_12]].to_numpy(float)
    resp = episodes[[f"{f}__response" for f in FEATURES_12]].to_numpy(float)
    raw = np.linalg.norm((resp - pre) / sd, axis=1)
    d = episodes.copy()
    d["radicality_raw"] = raw
    mu, sig = np.nanmean(raw), np.nanstd(raw, ddof=0)
    d["radicality_z"] = (raw - mu) / (sig if sig > 0 else 1.0)
    cutoff = float(np.nanquantile(raw, 0.75))
    d["extreme_q4"] = d["radicality_raw"].ge(cutoff)
    return d, cutoff


def metrics_table(pred: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    grouper = group_cols[0] if len(group_cols) == 1 else group_cols
    for key, g in pred.groupby(grouper, dropna=False):
        if not isinstance(key, tuple): key = (key,)
        auc,ll,br = safe_metrics(g.y, g.pred, g.w)
        row = {c:v for c,v in zip(group_cols,key)}
        row.update({"n_rows":len(g),"spells":int(g.w.sum()),"auc":auc,"logloss":ll,"brier":br})
        rows.append(row)
    return pd.DataFrame(rows)


def overall_metrics(pred: pd.DataFrame, label: str) -> dict:
    auc,ll,br = safe_metrics(pred.y, pred.pred, pred.w)
    return {"spec":label,"n_rows":len(pred),"spells":int(pred.w.sum()),"auc":auc,"logloss":ll,"brier":br}


def make_goal_targets(goals: pd.DataFrame, state: str, context_state: str = "response") -> pd.DataFrame:
    rows=[]
    for r in goals.itertuples(index=False):
        if state == "recovery" and not bool(getattr(r,"has_config__recovery")):
            continue
        q={
            "target_key":f"{r.anchor_id}:{state}",
            "anchor_id":r.anchor_id,
            "state":state,
            "league":r.league,
            "game_chron_key":r.game_chron_key,
            "team_id":r.focal_team_id,
            "half":r.half,
            "bin2":r.bin2_response,
        }
        for f in FEATURES_12:
            q[f]=getattr(r,f"{f}__{state}")
        rows.append(q)
    return pd.DataFrame(rows)


def clustered_ols(df: pd.DataFrame, y: str, xcols: list[str]) -> pd.DataFrame:
    d=df.dropna(subset=[y,"game_id"]+xcols).copy()
    X=sm.add_constant(d[xcols].astype(float))
    fit=sm.OLS(d[y].astype(float),X).fit(cov_type="cluster",cov_kwds={"groups":d.game_id})
    return pd.DataFrame([{"term":t,"beta":float(fit.params[t]),"se":float(fit.bse[t]),"p":float(fit.pvalues[t]),"n":len(d),"r2":float(fit.rsquared)} for t in xcols])


def zscore(s: pd.Series) -> pd.Series:
    sd=float(s.std(ddof=0)); return (s-float(s.mean()))/(sd if sd>0 else 1.0)


def figure5_audit(goal_extreme: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    d=goal_extreme.dropna(subset=["M_response","M_recovery","J_response","J_recovery","mean_spell_depth__response","mean_spell_depth__recovery"]).copy()
    d["historical_inward"] = d.M_response - d.M_recovery
    d["delta_J"] = d.J_recovery - d.J_response
    d["continuity_gain"] = d["mean_spell_depth__recovery"] - d["mean_spell_depth__response"]
    d["reach4_gain"] = d["reach4_share__recovery"] - d["reach4_share__response"]
    d["recovered"] = d.continuity_gain > 0
    d["J_up"] = d.delta_J > 0
    d["no_historical_reversion"] = d.historical_inward <= 0
    d["directional_signature"] = np.select(
        [d.J_up & ~d.no_historical_reversion,
         d.J_up & d.no_historical_reversion,
         ~d.J_up & ~d.no_historical_reversion,
         ~d.J_up & d.no_historical_reversion],
        ["Historical inward + J up","No historical reversion + J up","Historical inward + J non-up","No historical reversion + J non-up"],
        default="Other")
    order=["Historical inward + J up","No historical reversion + J up","No historical reversion + J non-up","Historical inward + J non-up"]
    quad=(d.groupby("directional_signature",observed=False)
          .agg(n=("anchor_id","size"),recovered_n=("recovered","sum"),mean_continuity_gain=("continuity_gain","mean"))
          .reindex(order).fillna(0).reset_index())
    quad["recovery_rate"] = np.where(quad.n>0,quad.recovered_n/quad.n,np.nan)

    nr=d[d.no_historical_reversion].copy()
    a=int(((nr.J_up)&(nr.recovered)).sum()); b=int(((nr.J_up)&(~nr.recovered)).sum())
    c=int(((~nr.J_up)&(nr.recovered)).sum()); e=int(((~nr.J_up)&(~nr.recovered)).sum())
    if (a+b)>0 and (c+e)>0:
        OR,p=fisher_exact([[a,b],[c,e]])
    else:
        OR,p=np.nan,np.nan
    tests=[{"test":"Fisher recovery among no-reversion: J-up vs J-non-up","estimate":float(OR),"se":np.nan,"p":float(p),"n":len(nr),"details":f"Jup {a}/{a+b}; Jnonup {c}/{c+e}"}]
    if len(nr)>=10 and nr.J_up.nunique()>1:
        rr=clustered_ols(nr.assign(J_up_num=nr.J_up.astype(float)),"continuity_gain",["J_up_num"]).iloc[0]
        tests.append({"test":"Clustered continuous gain among no-reversion: J-up","estimate":rr.beta,"se":rr.se,"p":rr.p,"n":int(rr.n),"details":"OLS, game-clustered SE"})
    if len(d)>=20:
        hr=clustered_ols(d.assign(delta_J_z=zscore(d.delta_J),historical_inward_z=zscore(d.historical_inward)),"continuity_gain",["delta_J_z","historical_inward_z"])
        for r in hr.itertuples(index=False):
            tests.append({"test":f"Clustered horse race: {r.term}","estimate":r.beta,"se":r.se,"p":r.p,"n":int(r.n),"details":"OLS, game-clustered SE"})
    return d,quad,pd.DataFrame(tests)


def plot_figure5(d: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,6))
    rec=d[d.recovered]; non=d[~d.recovered]
    ax.scatter(rec.historical_inward,rec.delta_J,marker="o",alpha=.65,label="Recovered")
    ax.scatter(non.historical_inward,non.delta_J,marker="x",alpha=.65,label="Non-recovered")
    ax.axhline(0,linewidth=1); ax.axvline(0,linewidth=1)
    ax.set_xlabel("Historical inward movement  M(B) - M(C)  (>0 = historical reversion)")
    ax.set_ylabel("ΔJ = J(C) - J(B)")
    ax.set_title("StatsBomb 2015/16: Coordinate-Specific Recovery Replication")
    ax.legend()
    fig.tight_layout(); fig.savefig(path,dpi=180); plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--processed-dir",required=True)
    ap.add_argument("--raw-data",required=True)
    ap.add_argument("--output-dir",required=True)
    args=ap.parse_args()
    processed=Path(args.processed_dir); raw=Path(args.raw_data); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)

    matches=pd.read_csv(processed/"statsbomb_2015_16_big5_matches.csv")
    windows=pd.concat([pd.read_parquet(processed/f"windows12_{s}.parquet") for s in LEAGUE_SLUG.values()],ignore_index=True)
    assert len(matches)==1823
    assert len(windows)==158121
    folds=make_temporal_folds(matches,8); folds.to_csv(out/"temporal_folds.csv",index=False)

    print("Stage A: strict temporal-OOS J",flush=True)
    tp=temporal_oos_predictions(windows,folds); tp.to_parquet(out/"temporal_oos_predictions.parquet",index=False,compression="zstd")
    tb=metrics_table(tp,["block"]); tb.to_csv(out/"temporal_oos_block_metrics.csv",index=False)
    overall=[overall_metrics(tp,"Pooled strict temporal-OOS")]

    print("Stage B: LOLO + temporal transport",flush=True)
    lp=lolo_temporal_predictions(windows,folds); lp.to_parquet(out/"lolo_temporal_predictions.parquet",index=False,compression="zstd")
    lb=metrics_table(lp,["heldout_league","block"]); lb.to_csv(out/"lolo_block_metrics.csv",index=False)
    ll=metrics_table(lp,["heldout_league"]); ll.to_csv(out/"lolo_heldout_league_metrics.csv",index=False)
    overall.append(overall_metrics(lp,"LOLO + strict temporal-OOS"))
    pd.DataFrame(overall).to_csv(out/"J_validation_overall.csv",index=False)

    print("Stage C: pooled Goal/Miss anchors and radicality",flush=True)
    anchors=extract_shot_anchors(raw,matches); anchors.to_csv(out/"shot_anchors_all.csv.gz",index=False,compression="gzip")
    episodes=build_anchor_episode_panel(processed,anchors)
    episodes,rad_cut=add_radicality(episodes,windows)
    episodes.to_csv(out/"anchor_episode_panel.csv.gz",index=False,compression="gzip")
    goal=episodes[episodes.goal_treat.eq(1)].copy()
    print("eligible anchors",len(episodes),"goals",len(goal),"misses",int((episodes.goal_treat==0).sum()),"rad Q4 cutoff",rad_cut,flush=True)

    print("Stage D: strict temporal-OOS J on goal response/recovery states",flush=True)
    tr=make_goal_targets(goal,"response"); tc=make_goal_targets(goal,"recovery")
    jt=pd.concat([tr,tc],ignore_index=True)
    jp=score_targets_temporal(windows,jt,folds,lolo=False).rename(columns={"pred_J":"J"})
    jp.to_csv(out/"goal_state_J_scores.csv",index=False)

    print("Stage E: strict-prior historical M on goal response/recovery states",flush=True)
    mt=jt[["target_key","league","game_chron_key","team_id"]+FEATURES_12].copy()
    mp=historical_marginality_targets(windows,mt,min_history=200,max_calibration=1200)
    mp.to_csv(out/"goal_state_M_scores.csv",index=False)

    state=jt[["target_key","anchor_id","state"]].merge(jp[["target_key","J"]],on="target_key",how="left").merge(mp[["target_key","component_atyp_pct"]],on="target_key",how="left")
    state=state.rename(columns={"component_atyp_pct":"M"})
    wide=state.pivot(index="anchor_id",columns="state",values=["J","M"])
    wide.columns=[f"{a}_{b}" for a,b in wide.columns]; wide=wide.reset_index()
    goal=goal.merge(wide,on="anchor_id",how="left")
    goal.to_csv(out/"goal_episode_scored.csv.gz",index=False,compression="gzip")

    static=goal.dropna(subset=["M_response","J_response"]).copy()
    corr=float(static[["M_response","J_response"]].corr().iloc[0,1]) if len(static)>1 else np.nan
    static["historical_edge"]=static.M_response>=.75; static["higher_J"]=static.J_response>=.50
    static["cell"]=np.select([
        ~static.historical_edge & static.higher_J,
        ~static.historical_edge & ~static.higher_J,
        static.historical_edge & static.higher_J,
        static.historical_edge & ~static.higher_J],
        ["Interior + Higher J","Interior + Lower J","Peripheral + Higher J","Peripheral + Lower J"])
    cells=static.groupby("cell").agg(n=("anchor_id","size"),mean_M=("M_response","mean"),mean_J=("J_response","mean")).reset_index(); cells["share"]=cells.n/len(static)
    cells.to_csv(out/"static_MJ_cells.csv",index=False)
    pd.DataFrame([{"n":len(static),"pearson_r_M_J":corr,"peripheral_higherJ_n":int((static.historical_edge&static.higher_J).sum()),"interior_lowerJ_n":int((~static.historical_edge&~static.higher_J).sum())}]).to_csv(out/"static_MJ_nonredundancy_summary.csv",index=False)

    print("Stage F: Figure 5 replication",flush=True)
    gx=goal[goal.extreme_q4 & goal.has_config__recovery.eq(1)].copy()
    f5,quad,tests=figure5_audit(gx)
    f5.to_csv(out/"Figure5_StatsBomb_Directional_Audit.csv",index=False)
    quad.to_csv(out/"Figure5_StatsBomb_Quadrants.csv",index=False)
    tests.to_csv(out/"Figure5_StatsBomb_Tests.csv",index=False)
    plot_figure5(f5,out/"Figure5_StatsBomb_Coordinate_Specific_Recovery.png")

    manifest={
        "source":"StatsBomb Open Data 2015/16 Big Five",
        "processed_matches":len(matches),
        "eligible_2min_windows":len(windows),
        "shot_anchors_raw":len(anchors),
        "eligible_pre_response_anchors":len(episodes),
        "eligible_goal_anchors":len(goal),
        "eligible_miss_anchors":int((episodes.goal_treat==0).sum()),
        "pooled_radicality_q4_cutoff":rad_cut,
        "goal_extreme_with_recovery_config":len(gx),
        "figure5_complete_n":len(f5),
        "figure5_recovered_n":int(f5.recovered.sum()) if len(f5) else 0,
        "figure5_killer_region_n":int((f5.J_up & f5.no_historical_reversion).sum()) if len(f5) else 0,
        "figure5_killer_recovered_n":int((f5.J_up & f5.no_historical_reversion & f5.recovered).sum()) if len(f5) else 0,
        "features_12":FEATURES_12,
        "J_definition":"strict temporal-OOS P(spell depth >=4 | 12D config + league + half + time-bin); recovery J scored under response context G0",
        "LOLO_definition":"strict prior other-four-league training; no league indicator",
        "M_definition":"strict-prior same-league peer componentwise two-sided tail marginality percentile; focal team excluded",
        "radicality_definition":"Euclidean pre-to-response movement in 12D after scaling by pooled 2-min-window SD; Q4 pooled across eligible Goal and non-goal Shot anchors",
        "recovery_definition_primary":"mean complete passing-spell depth in +5~+7 minus +3~+5 > 0",
    }
    (out/"stage2_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")

    report=[]
    report.append("# StatsBomb 2015/16 Big Five — Stage 2 Replication Results\n")
    report.append(f"- Matches: **{len(matches):,}**")
    report.append(f"- Eligible 2-min windows: **{len(windows):,}**")
    report.append(f"- Eligible Goal/Miss pre→response anchors: **{len(episodes):,}**")
    report.append(f"- Eligible goal anchors: **{len(goal):,}**")
    report.append(f"- Pooled radicality Q4 cutoff: **{rad_cut:.4f}**")
    for r in overall:
        report.append(f"- {r['spec']}: AUC **{r['auc']:.4f}**, Brier **{r['brier']:.4f}**, spells **{r['spells']:,}**")
    report.append(f"- Static M–J correlation on scored goal responses: **r={corr:.4f}**, N={len(static):,}")
    report.append(f"- Figure 5 complete Goal+Extreme trajectories: **N={len(f5):,}**")
    if len(f5):
        killer=f5.J_up & f5.no_historical_reversion
        report.append(f"- J-up without historical reversion: **{int(killer.sum())}**; recovered **{int((killer&f5.recovered).sum())}**")
        report.append(f"- Recovered overall: **{int(f5.recovered.sum())}/{len(f5)}**")
    report.append("\n## Measurement note\n")
    report.append("The replication uses the frozen 12D provider-harmonized configuration. `simple_share` and `launch_share` are excluded because StatsBomb has no exact ontology-equivalent fields. Substantial reconfiguration is operationalized as the pooled top quartile of standardized pre→response 12D movement across eligible Goal and non-goal Shot anchors. This is a harmonized cross-provider operationalization, not a claim that provider event ontologies are identical.")
    (out/"STAGE2_RESULTS_REPORT.md").write_text("\n".join(report),encoding="utf-8")

    print(json.dumps(manifest,indent=2),flush=True)
    print(pd.DataFrame(overall).to_string(index=False),flush=True)
    print("\nLOLO heldout leagues\n",ll.to_string(index=False),flush=True)
    print("\nStatic cells\n",cells.to_string(index=False),flush=True)
    print("\nFigure5 quadrants\n",quad.to_string(index=False),flush=True)
    print("\nFigure5 tests\n",tests.to_string(index=False),flush=True)


if __name__=="__main__":
    main()
