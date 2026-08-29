#!/usr/bin/env python3
"""Frozen analysis helpers for the StatsBomb 2015/16 cross-provider replication."""
from __future__ import annotations

import hashlib
import math
import numpy as np
import pandas as pd

from research.statsbomb_external_runner import FEATURES_12, build_pass_spells

CTX_POOLED = ["league", "half", "bin2"]
CTX_LOLO = ["half", "bin2"]


def make_temporal_folds(matches: pd.DataFrame, n_chunks: int = 8) -> pd.DataFrame:
    """Expanding-window temporal folds; first chunk is training-only."""
    keys = np.array(sorted(pd.unique(matches["game_chron_key"].dropna()).astype(int)))
    if len(keys) < n_chunks:
        raise ValueError("Not enough games for requested temporal chunks")
    chunks = [np.asarray(x, dtype=int) for x in np.array_split(keys, n_chunks)]
    rows = []
    for b in range(1, n_chunks):
        te = chunks[b]
        rows.append({
            "block": b,
            "train_before_chron": int(te.min()),
            "test_before_chron": int(te.max()) + 1,
            "test_games": int(len(te)),
        })
    return pd.DataFrame(rows)


def sufficient_rows(windows: pd.DataFrame, features: list[str] | None = None) -> pd.DataFrame:
    """Convert window-level binomial aggregates to weighted success/failure rows."""
    features = features or FEATURES_12
    d = windows[windows["spell_starts"].fillna(0) > 0].copy()
    n = d["spell_starts"].round().astype(int)
    s = (d["reach4_share"] * n).round().astype(int)
    f = n - s
    cols = features + CTX_POOLED + ["game_chron_key"]
    a = d.loc[s > 0, cols].copy(); a["y"] = 1; a["w"] = s[s > 0].to_numpy()
    b = d.loc[f > 0, cols].copy(); b["y"] = 0; b["w"] = f[f > 0].to_numpy()
    return pd.concat([a, b], ignore_index=True)


def config_from_interval(passes: pd.DataFrame) -> dict | None:
    """Aggregate pass rows into frozen 12D configuration; require >=3 passes."""
    if len(passes) < 3:
        return None
    dx = passes["x2"].to_numpy(float) - passes["x1"].to_numpy(float)
    dy = passes["y2"].to_numpy(float) - passes["y1"].to_numpy(float)
    length = np.hypot(dx, dy)
    return {
        "mean_pass_length": float(length.mean()),
        "mean_dx": float(dx.mean()),
        "mean_abs_dy": float(np.abs(dy).mean()),
        "mean_x1": float(passes["x1"].mean()),
        "mean_x2": float(passes["x2"].mean()),
        "forward_share": float(np.mean(dx > 0)),
        "long_share": float(np.mean(length >= 30)),
        "final3_start_share": float(np.mean(passes["x1"].to_numpy(float) >= 70)),
        "final3_end_share": float(np.mean(passes["x2"].to_numpy(float) >= 70)),
        "high_share": float(passes["high_share"].mean()),
        "head_share": float(passes["head_share"].mean()),
        "cross_share": float(passes["cross_share"].mean()),
        "pass_n": int(len(passes)),
    }


def historical_marginality_targets(
    windows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    features: list[str] | None = None,
    min_history: int = 200,
    max_calibration: int = 1200,
) -> pd.DataFrame:
    """Frozen component-extremeness historical percentile.

    Strict-prior windows from the same league excluding the focal team are used for
    componentwise empirical two-sided tails. The mean negative log tail probability is
    percentile-calibrated against prior peer configurations at the same historical cutoff.
    """
    features = features or FEATURES_12
    reqw = ["league", "game_chron_key", "team_id"] + features
    reqt = ["target_key", "league", "game_chron_key", "team_id"] + features
    w0 = windows.dropna(subset=reqw).copy()
    t0 = targets.dropna(subset=reqt).copy()
    out: list[dict] = []

    for league in sorted(t0["league"].unique()):
        w = w0[w0["league"].eq(league)].sort_values("game_chron_key").reset_index(drop=True)
        t = t0[t0["league"].eq(league)]
        Xw = w[features].to_numpy(float)
        ckw = w["game_chron_key"].to_numpy(float)
        tw = w["team_id"].to_numpy(int)

        for ck, gt in t.groupby("game_chron_key", sort=True):
            allidx = np.flatnonzero(ckw < float(ck))
            if len(allidx) < min_history:
                continue
            if len(allidx) > max_calibration:
                seed = int(hashlib.md5(f"{league}|{ck}".encode()).hexdigest()[:8], 16)
                sel = np.random.default_rng(seed).choice(allidx, size=max_calibration, replace=False)
            else:
                sel = allidx

            for focal, gf in gt.groupby("team_id"):
                peeridx = allidx[tw[allidx] != int(focal)]
                npeer = len(peeridx)
                if npeer < min_history:
                    continue
                hp = Xw[peeridx]
                sorted_peer = [np.sort(hp[:, j]) for j in range(len(features))]
                hs = sel[tw[sel] != int(focal)]
                if len(hs) < max(10, min_history // 2):
                    hs = peeridx[: min(len(peeridx), max_calibration)]
                hc = Xw[hs]
                uh = np.empty_like(hc, float)
                for j, s in enumerate(sorted_peer):
                    uh[:, j] = (np.searchsorted(s, hc[:, j], side="right") + 0.5) / (npeer + 1.0)
                comp_hist = (-np.log(np.clip(2 * np.minimum(uh, 1 - uh), 1e-8, 1))).mean(axis=1)

                for rr in gf.itertuples(index=False):
                    x = np.array([getattr(rr, f) for f in features], float)
                    u = np.array([
                        (np.searchsorted(sorted_peer[j], x[j], side="right") + 0.5) / (npeer + 1.0)
                        for j in range(len(features))
                    ])
                    u = np.clip(u, 1e-5, 1 - 1e-5)
                    raw = float((-np.log(np.clip(2 * np.minimum(u, 1 - u), 1e-8, 1))).mean())
                    pct = float((np.sum(comp_hist <= raw) + 1) / (len(comp_hist) + 1))
                    out.append({
                        "target_key": rr.target_key,
                        "league": league,
                        "game_chron_key": float(ck),
                        "team_id": int(focal),
                        "peer_hist_n": npeer,
                        "component_atyp_raw": raw,
                        "component_atyp_pct": pct,
                    })
    return pd.DataFrame(out)


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(
        loss="log_loss", learning_rate=0.1, max_iter=100,
        max_leaf_nodes=31, min_samples_leaf=20, l2_regularization=0.0,
        random_state=123,
    )


def _categories(windows: pd.DataFrame, cols: list[str]) -> dict[str, pd.Index]:
    return {c: pd.Index(sorted(windows[c].dropna().unique())) for c in cols}


def _cast_categories(df: pd.DataFrame, cats: dict[str, pd.Index]) -> pd.DataFrame:
    q = df.copy()
    for c, categories in cats.items():
        q[c] = pd.Categorical(q[c], categories=categories)
    return q


def safe_metrics(y, p, w) -> tuple[float, float, float]:
    from sklearn.metrics import roc_auc_score, log_loss
    y = np.asarray(y, int); p = np.asarray(p, float); w = np.asarray(w, float)
    auc = float(roc_auc_score(y, p, sample_weight=w)) if len(np.unique(y)) > 1 else float("nan")
    ll = float(log_loss(y, p, sample_weight=w, labels=[0, 1]))
    br = float(np.average((y - p) ** 2, weights=w))
    return auc, ll, br


def temporal_oos_predictions(windows: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    z = sufficient_rows(windows)
    cats = _categories(windows, CTX_POOLED)
    z = _cast_categories(z, cats)
    xcols = FEATURES_12 + CTX_POOLED
    out = []
    for b in folds.itertuples(index=False):
        lo, hi = float(b.train_before_chron), float(b.test_before_chron)
        tr = z[z["game_chron_key"] < lo]
        te = z[(z["game_chron_key"] >= lo) & (z["game_chron_key"] < hi)].copy()
        if tr.empty or te.empty:
            continue
        m = _model(); m.fit(tr[xcols], tr["y"], sample_weight=tr["w"])
        te["pred"] = m.predict_proba(te[xcols])[:, 1]
        te["block"] = int(b.block)
        out.append(te[["league","game_chron_key","y","w","pred","block"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def lolo_temporal_predictions(windows: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """LOLO+temporal OOS. No league indicator is used, matching the frozen Wyscout transport test."""
    z = sufficient_rows(windows)
    cats = _categories(windows, CTX_LOLO)
    z = _cast_categories(z, cats)
    xcols = FEATURES_12 + CTX_LOLO
    out = []
    for heldout in sorted(windows["league"].dropna().unique()):
        for b in folds.itertuples(index=False):
            lo, hi = float(b.train_before_chron), float(b.test_before_chron)
            tr = z[(z["game_chron_key"] < lo) & (~z["league"].astype(str).eq(str(heldout)))]
            te = z[(z["game_chron_key"] >= lo) & (z["game_chron_key"] < hi) & z["league"].astype(str).eq(str(heldout))].copy()
            if tr.empty or te.empty:
                continue
            m = _model(); m.fit(tr[xcols], tr["y"], sample_weight=tr["w"])
            te["pred"] = m.predict_proba(te[xcols])[:, 1]
            te["block"] = int(b.block)
            te["heldout_league"] = heldout
            out.append(te[["heldout_league","game_chron_key","y","w","pred","block"]])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def score_targets_temporal(
    windows: pd.DataFrame,
    targets: pd.DataFrame,
    folds: pd.DataFrame,
    *,
    lolo: bool = False,
) -> pd.DataFrame:
    """Strict temporal-OOS score for arbitrary configuration targets."""
    z = sufficient_rows(windows)
    ctx = CTX_LOLO if lolo else CTX_POOLED
    cats = _categories(windows, ctx)
    z = _cast_categories(z, cats)
    t = _cast_categories(targets, cats)
    xcols = FEATURES_12 + ctx
    pred = np.full(len(t), np.nan, float)

    if not lolo:
        for b in folds.itertuples(index=False):
            lo, hi = float(b.train_before_chron), float(b.test_before_chron)
            tr = z[z["game_chron_key"] < lo]
            ix = (t["game_chron_key"] >= lo) & (t["game_chron_key"] < hi)
            if tr.empty or not ix.any():
                continue
            m = _model(); m.fit(tr[xcols], tr["y"], sample_weight=tr["w"])
            pred[ix.to_numpy()] = m.predict_proba(t.loc[ix, xcols])[:, 1]
    else:
        for heldout in sorted(targets["league"].dropna().unique()):
            heldmask = t["league"].astype(str).eq(str(heldout))
            for b in folds.itertuples(index=False):
                lo, hi = float(b.train_before_chron), float(b.test_before_chron)
                tr = z[(z["game_chron_key"] < lo) & (~z["league"].astype(str).eq(str(heldout)))]
                ix = heldmask & (t["game_chron_key"] >= lo) & (t["game_chron_key"] < hi)
                if tr.empty or not ix.any():
                    continue
                m = _model(); m.fit(tr[xcols], tr["y"], sample_weight=tr["w"])
                pred[ix.to_numpy()] = m.predict_proba(t.loc[ix, xcols])[:, 1]

    cols = [c for c in ["target_key", "league", "game_chron_key"] if c in targets.columns]
    out = targets[cols].copy(); out["pred_J"] = pred
    return out
