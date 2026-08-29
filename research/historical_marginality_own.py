"""Organization-specific strict-prior historical configuration marginality.

This module implements the frozen paper construct M_{i,H0}: each target configuration
is positioned relative to the focal team's own strictly prior configuration history.
It deliberately does not use other teams as the reference distribution.
"""
from __future__ import annotations

import hashlib
import numpy as np
import pandas as pd

from research.statsbomb_external_runner import FEATURES_12


def historical_marginality_targets(
    windows: pd.DataFrame,
    targets: pd.DataFrame,
    *,
    features: list[str] | None = None,
    min_history: int = 200,
    max_calibration: int = 1200,
) -> pd.DataFrame:
    """Score targets against the focal team's own strict-prior history.

    For each target at game chronology cutoff ``ck`` and focal team ``i``:
    1. use only team-i windows with ``game_chron_key < ck``;
    2. calculate componentwise empirical two-sided tail probabilities;
    3. aggregate component extremeness as mean negative log two-sided tail probability;
    4. percentile-calibrate that aggregate against a deterministic sample of the
       focal team's own prior configurations evaluated under the same frozen H0.

    The same strict-prior H0 is therefore the reference for B and C when they occur
    in the same match.
    """
    features = features or FEATURES_12
    reqw = ["league", "game_chron_key", "team_id"] + features
    reqt = ["target_key", "league", "game_chron_key", "team_id"] + features
    w0 = windows.dropna(subset=reqw).copy()
    t0 = targets.dropna(subset=reqt).copy()
    out: list[dict] = []

    for league in sorted(t0["league"].dropna().unique()):
        w = w0[w0["league"].eq(league)].sort_values("game_chron_key").reset_index(drop=True)
        t = t0[t0["league"].eq(league)]
        Xw = w[features].to_numpy(float)
        ckw = w["game_chron_key"].to_numpy(float)
        tw = w["team_id"].to_numpy(int)

        for ck, gt in t.groupby("game_chron_key", sort=True):
            prior_mask = ckw < float(ck)
            for focal, gf in gt.groupby("team_id"):
                ownidx = np.flatnonzero(prior_mask & (tw == int(focal)))
                nown = len(ownidx)
                if nown < min_history:
                    continue

                h0 = Xw[ownidx]
                sorted_own = [np.sort(h0[:, j]) for j in range(len(features))]

                if nown > max_calibration:
                    seed = int(
                        hashlib.md5(f"{league}|{int(focal)}|{ck}".encode()).hexdigest()[:8],
                        16,
                    )
                    calidx = np.random.default_rng(seed).choice(
                        ownidx, size=max_calibration, replace=False
                    )
                else:
                    calidx = ownidx

                hc = Xw[calidx]
                uh = np.empty_like(hc, float)
                for j, s in enumerate(sorted_own):
                    uh[:, j] = (
                        np.searchsorted(s, hc[:, j], side="right") + 0.5
                    ) / (nown + 1.0)
                uh = np.clip(uh, 1e-5, 1 - 1e-5)
                comp_hist = (
                    -np.log(np.clip(2 * np.minimum(uh, 1 - uh), 1e-8, 1))
                ).mean(axis=1)

                for rr in gf.itertuples(index=False):
                    x = np.array([getattr(rr, f) for f in features], float)
                    u = np.array([
                        (
                            np.searchsorted(sorted_own[j], x[j], side="right") + 0.5
                        ) / (nown + 1.0)
                        for j in range(len(features))
                    ])
                    u = np.clip(u, 1e-5, 1 - 1e-5)
                    raw = float(
                        (-np.log(np.clip(2 * np.minimum(u, 1 - u), 1e-8, 1))).mean()
                    )
                    pct = float((np.sum(comp_hist <= raw) + 1) / (len(comp_hist) + 1))
                    out.append({
                        "target_key": rr.target_key,
                        "league": league,
                        "game_chron_key": float(ck),
                        "team_id": int(focal),
                        "own_hist_n": nown,
                        "component_atyp_raw": raw,
                        "component_atyp_pct": pct,
                    })

    return pd.DataFrame(out)
