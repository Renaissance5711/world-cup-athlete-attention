from __future__ import annotations

import math
from typing import Iterable, Mapping, Any

import numpy as np
import pandas as pd

COMPONENTS = [
    'mean_pass_dx',
    'forward_pass_share',
    'long_pass_share',
    'pass_length_mean',
    'acceleration_share',
    'final_third_entry_pass_share',
]

X_TO_M = 105.0 / 120.0
Y_TO_M = 68.0 / 80.0
FINAL_THIRD_X_SB = 80.0
LONG_PASS_M = 30.0


def timestamp_to_half_seconds(timestamp: str) -> float:
    hh, mm, ss = timestamp.split(':')
    return int(hh) * 3600.0 + int(mm) * 60.0 + float(ss)


def _type_name(e: Mapping[str, Any]) -> str | None:
    t = e.get('type')
    return t.get('name') if isinstance(t, Mapping) else None


def _team_id(e: Mapping[str, Any]):
    t = e.get('team')
    return t.get('id') if isinstance(t, Mapping) else None


def _team_name(e: Mapping[str, Any]):
    t = e.get('team')
    return t.get('name') if isinstance(t, Mapping) else None


def _window_metrics(events: Iterable[Mapping[str, Any]], team_id, period: int, lo: float, hi: float):
    selected=[]
    for e in events:
        if int(e.get('period', -1)) != int(period) or _team_id(e) != team_id:
            continue
        ts=e.get('timestamp')
        if ts is None:
            continue
        t=timestamp_to_half_seconds(str(ts))
        if lo <= t < hi:
            selected.append(e)

    n_events=len(selected)
    carries=sum(_type_name(e)=='Carry' for e in selected)
    acceleration_share = carries / n_events if n_events else 0.0

    dxs=[]; lens=[]; forwards=[]; longs=[]; entries=[]
    for e in selected:
        if _type_name(e) != 'Pass':
            continue
        loc=e.get('location')
        p=e.get('pass')
        end=p.get('end_location') if isinstance(p, Mapping) else None
        if not (isinstance(loc,(list,tuple)) and len(loc)>=2 and isinstance(end,(list,tuple)) and len(end)>=2):
            continue
        x1,y1=float(loc[0]),float(loc[1]); x2,y2=float(end[0]),float(end[1])
        dx=(x2-x1)*X_TO_M; dy=(y2-y1)*Y_TO_M; L=math.hypot(dx,dy)
        dxs.append(dx); lens.append(L); forwards.append(float(dx>0)); longs.append(float(L>=LONG_PASS_M))
        entries.append(float(x1 < FINAL_THIRD_X_SB and x2 >= FINAL_THIRD_X_SB))

    if not dxs:
        pass_metrics=[np.nan]*5
    else:
        pass_metrics=[float(np.mean(dxs)),float(np.mean(forwards)),float(np.mean(longs)),float(np.mean(lens)),float(np.mean(entries))]

    return {
        'mean_pass_dx':pass_metrics[0],
        'forward_pass_share':pass_metrics[1],
        'long_pass_share':pass_metrics[2],
        'pass_length_mean':pass_metrics[3],
        'acceleration_share':float(acceleration_share),
        'final_third_entry_pass_share':pass_metrics[4],
        'n_team_events':n_events,
        'n_passes':len(dxs),
        'n_carries':carries,
    }


def signed_six_component_change(events: Iterable[Mapping[str, Any]], team_id, period: int, anchor_t: float):
    events=list(events)
    pre=_window_metrics(events,team_id,period,anchor_t-300.0,anchor_t)
    post=_window_metrics(events,team_id,period,anchor_t+180.0,anchor_t+300.0)
    out={}
    for c in COMPONENTS:
        a,b=pre[c],post[c]
        out[c]=float(b-a) if np.isfinite(a) and np.isfinite(b) else np.nan
    out.update({
        'pre_n_team_events':pre['n_team_events'],'response_n_team_events':post['n_team_events'],
        'pre_n_passes':pre['n_passes'],'response_n_passes':post['n_passes'],
        'pre_n_carries':pre['n_carries'],'response_n_carries':post['n_carries'],
    })
    return out


def standardize_components_and_composite(raw: pd.DataFrame) -> pd.DataFrame:
    out=raw.copy()
    zcols=[]
    for c in COMPONENTS:
        mu=out[c].mean(skipna=True); sd=out[c].std(ddof=0,skipna=True)
        z=f'z_{c}'
        out[z]=(out[c]-mu)/sd if np.isfinite(sd) and sd>0 else np.nan
        zcols.append(z)
    out['radicality_homolog_z']=out[zcols].mean(axis=1,skipna=True)
    return out


def extract_shot_anchors(events: Iterable[Mapping[str, Any]], match_id) -> pd.DataFrame:
    rows=[]
    for e in events:
        if _type_name(e) != 'Shot':
            continue
        sh=e.get('shot') if isinstance(e.get('shot'),Mapping) else {}
        outcome=sh.get('outcome') if isinstance(sh.get('outcome'),Mapping) else {}
        st=sh.get('type') if isinstance(sh.get('type'),Mapping) else {}
        name=outcome.get('name')
        rows.append({
            'game_id':match_id,
            'half':int(e.get('period')),
            't':timestamp_to_half_seconds(str(e.get('timestamp'))),
            'event_index':e.get('index'),
            'event_uuid':e.get('id'),
            'shooting_team_id':_team_id(e),
            'shooting_team_name':_team_name(e),
            'shot_outcome':name,
            'shot_type':st.get('name'),
            'anchor_class':'Goal' if name=='Goal' else 'Miss',
        })
    return pd.DataFrame(rows)
