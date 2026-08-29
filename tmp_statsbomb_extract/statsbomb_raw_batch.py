from __future__ import annotations

from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd

from statsbomb_radicality_homolog import (
    X_TO_M,
    Y_TO_M,
    timestamp_to_half_seconds,
    extract_shot_anchors,
    signed_six_component_change,
)


def _team_id(e: Mapping[str, Any]):
    t=e.get('team')
    return t.get('id') if isinstance(t, Mapping) else None


def _team_name(e: Mapping[str, Any]):
    t=e.get('team')
    return t.get('name') if isinstance(t, Mapping) else None


def _two_match_teams(events):
    teams={}
    for e in events:
        tid=_team_id(e)
        if tid is not None:
            teams[tid]=_team_name(e)
    return teams


CONTINUITY_INTERCEPT = -1.6550033604199785
CONTINUITY_MAX_STREAK_W = 0.17385372336959271
CONTINUITY_LONG_STREAK_SHARE_W = 0.83146291307747178
CONTINUITY_RATE_W = 1.3962837347982049


def _is_completed_pass(e: Mapping[str, Any]) -> bool:
    if (e.get('type') or {}).get('name') != 'Pass':
        return False
    p = e.get('pass') if isinstance(e.get('pass'), Mapping) else {}
    return p.get('outcome') is None


def continuity_primitives(
    events: Iterable[Mapping[str, Any]],
    team_id,
    period: int,
    start_t: float,
    end_t: float,
) -> dict[str, float]:
    rows=[]
    for e in events:
        if int(e.get('period', -1)) != int(period):
            continue
        if (e.get('type') or {}).get('name') != 'Pass':
            continue
        t=timestamp_to_half_seconds(e.get('timestamp'))
        if t is None or not (float(start_t) <= float(t) < float(end_t)):
            continue
        rows.append((int(e.get('index', 0)), float(t), e))
    rows.sort(key=lambda z:(z[0], z[1]))

    pass_n=0
    success_n=0
    streaks=[]
    cur=0
    for _,_,e in rows:
        focal = _team_id(e) == team_id
        good = focal and _is_completed_pass(e)
        if focal:
            pass_n += 1
        if good:
            success_n += 1
            cur += 1
        else:
            if cur:
                streaks.append(cur)
                cur=0
    if cur:
        streaks.append(cur)

    max_streak=max(streaks) if streaks else 0
    if success_n:
        long_share=sum(v for v in streaks if v >= 3) / success_n
        continuation=sum(max(0, v-1) for v in streaks) / success_n
    else:
        long_share=0.0
        continuation=0.0
    score=(CONTINUITY_INTERCEPT
           + CONTINUITY_MAX_STREAK_W*max_streak
           + CONTINUITY_LONG_STREAK_SHARE_W*long_share
           + CONTINUITY_RATE_W*continuation)
    return {
        'pass_n_continuity': int(pass_n),
        'successful_pass_n': int(success_n),
        'max_streak': int(max_streak),
        'long_streak_share': float(long_share),
        'continuation_rate': float(continuation),
        'observed_continuity_homolog': float(score),
    }


def build_match_anchor_candidates(
    events: Iterable[Mapping[str, Any]],
    match_id,
    league: str | None = None,
) -> pd.DataFrame:
    events=list(events)
    teams=_two_match_teams(events)
    shots=extract_shot_anchors(events, match_id=match_id)
    if shots.empty:
        return shots

    rows=[]
    by_uuid={e.get('id'):e for e in events}
    for s in shots.itertuples(index=False):
        if not (600.0 <= float(s.t) < 1800.0):
            continue
        if str(s.shot_type) != 'Open Play':
            continue

        others=[tid for tid in teams if tid != s.shooting_team_id]
        if len(others) != 1:
            continue
        defending_team_id=others[0]
        e=by_uuid.get(s.event_uuid,{})
        sh=e.get('shot') if isinstance(e.get('shot'),Mapping) else {}
        loc=e.get('location')
        x_m=y_m=np.nan
        if isinstance(loc,(list,tuple)) and len(loc)>=2:
            x_m=float(loc[0])*X_TO_M
            y_m=float(loc[1])*Y_TO_M

        comp=signed_six_component_change(
            events,
            team_id=defending_team_id,
            period=int(s.half),
            anchor_t=float(s.t),
        )
        row={
            'game_id':match_id,
            'league':league,
            'half':int(s.half),
            't':float(s.t),
            'event_index':s.event_index,
            'event_uuid':s.event_uuid,
            'goal_treat':int(s.anchor_class=='Goal'),
            'anchor_class':s.anchor_class,
            'shot_outcome':s.shot_outcome,
            'shot_type':s.shot_type,
            'shooting_team_id':s.shooting_team_id,
            'shooting_team_name':s.shooting_team_name,
            'defending_team_id':defending_team_id,
            'defending_team_name':teams.get(defending_team_id),
            'statsbomb_xg':sh.get('statsbomb_xg'),
            'shot_x_m':x_m,
            'shot_y_m':y_m,
        }
        row.update(comp)
        response_cont=continuity_primitives(
            events, defending_team_id, int(s.half), float(s.t)+180.0, float(s.t)+300.0
        )
        recovery_cont=continuity_primitives(
            events, defending_team_id, int(s.half), float(s.t)+300.0, float(s.t)+420.0
        )
        row.update({f'response_{k}':v for k,v in response_cont.items()})
        row.update({f'recovery_{k}':v for k,v in recovery_cont.items()})
        rows.append(row)
    return pd.DataFrame(rows)
