from __future__ import annotations

import csv
import html
import re
import unicodedata
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd

PRAISE_TERMS = {
    "hero", "heroic", "brilliant", "superb", "outstanding", "magnificent",
    "sensational", "incredible", "excellent", "masterclass", "star",
    "rescues", "rescued", "saves", "saved", "denies", "denied", "stuns",
    "triumph", "glory", "impressive", "crucial", "decisive", "poise",
}
BLAME_TERMS = {
    "blunder", "error", "mistake", "fault", "costly", "nightmare", "howler",
    "fails", "failed", "failure", "poor", "terrible", "disastrous", "culpable",
    "gift", "gifts", "embarrassed",
    "flop", "collapse", "disaster", "criticised", "criticized", "blamed",
}
GOALKEEPER_CONTEXT_TERMS = {
    "goalkeeper", "keeper", "goalie", "shotstopper", "save", "saves", "saved",
    "penalty", "penalties", "goal", "goals", "blunder", "error", "mistake",
    "howler", "hero", "clean", "sheet", "concedes", "conceded", "beaten",
}

_WOLFRAM_ROW = re.compile(
    r"^\{goalkeeper_player_id : (?P<goalkeeper_player_id>.*?), "
    r"query_name : (?P<query_name>.*?), "
    r"title : (?P<title>.*), "
    r"link : (?P<link>https://news\.google\.com/.*?\?oc=5), "
    r"pubDate : (?P<pubDate>.*?), source : (?P<source>.*)\}$"
)


def _ascii(value: str) -> str:
    return unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")


def _tokens(value: str) -> set[str]:
    return set(re.findall(r"[a-z]+", _ascii(value).lower()))


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", _ascii(html.unescape(str(value))).lower()))


def _normalize_domain(value: str) -> str:
    domain = str(value or "").lower().strip()
    if "://" in domain:
        domain = urlparse(domain).netloc
    return domain.removeprefix("www.")


def parse_wolfram_google_news_csv(path: str | Path) -> pd.DataFrame:
    """Parse Wolfram's Association-form CSV export into a clean table."""

    records: list[dict[str, str]] = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        for csv_row in csv.reader(handle):
            if not csv_row:
                continue
            raw = ",".join(csv_row).strip()
            match = _WOLFRAM_ROW.match(raw)
            if not match:
                raise ValueError(f"Unparseable Google News row: {raw[:180]}")
            record = {key: html.unescape(value.strip()) for key, value in match.groupdict().items()}
            records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(
            columns=["goalkeeper_player_id", "query_name", "title", "link", "pubDate", "source", "seen_datetime"]
        )
    frame["seen_datetime"] = pd.to_datetime(frame["pubDate"], errors="coerce", utc=True)
    if frame["seen_datetime"].isna().any():
        bad = frame.loc[frame["seen_datetime"].isna(), "pubDate"].head().tolist()
        raise ValueError(f"Unparseable Google News dates: {bad}")
    return frame


def is_relevant_goalkeeper_headline(title: str, query_name: str) -> bool:
    raw_norm = _ascii(title).lower()
    query_norm = _ascii(query_name).lower().strip()
    parts = re.findall(r"[a-z]+", query_norm)
    title_tokens = _tokens(title)
    if query_norm and query_norm in raw_norm and len(parts) > 1:
        return True
    if query_norm and query_norm in raw_norm and len(parts) == 1:
        return bool(title_tokens.intersection(GOALKEEPER_CONTEXT_TERMS))
    if not parts:
        return False
    surname = parts[-1]
    if surname not in title_tokens:
        return False
    return bool(title_tokens.intersection(GOALKEEPER_CONTEXT_TERMS))


def classify_headline_framing(title: str) -> str:
    tokens = _tokens(title)
    praise = len(tokens.intersection(PRAISE_TERMS))
    blame = len(tokens.intersection(BLAME_TERMS))
    if praise > blame and praise > 0:
        return "praise"
    if blame > praise and blame > 0:
        return "blame"
    return "neutral"


def deduplicate_articles(articles: pd.DataFrame) -> pd.DataFrame:
    frame = articles.copy()
    if frame.empty:
        return frame
    frame["normalized_title"] = frame["title"].map(_normalize_title)
    source_column = "source" if "source" in frame else "domain"
    frame["normalized_domain"] = frame[source_column].map(_normalize_domain)
    if "seen_datetime" not in frame:
        if "pubDate" in frame:
            frame["seen_datetime"] = pd.to_datetime(frame["pubDate"], errors="coerce", utc=True)
        else:
            frame["seen_datetime"] = pd.to_datetime(
                frame["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
            )
    subset = ["normalized_title", "normalized_domain"]
    if "goalkeeper_player_id" in frame:
        subset.insert(0, "goalkeeper_player_id")
    frame = frame.sort_values("seen_datetime", kind="stable")
    return frame.drop_duplicates(subset, keep="first").reset_index(drop=True)


def assign_articles_to_goalkeeper_matches(
    articles: pd.DataFrame,
    goalkeeper_matches: pd.DataFrame,
    *,
    window_days: float = 2,
    require_opponent_after_hours: float = 12,
) -> pd.DataFrame:
    """Assign post-match headlines to the nearest goalkeeper appearance.

    When ``match_end_datetime`` is available, the window begins at the end of
    the match. Otherwise, the legacy date-midnight rule is retained for
    backwards-compatible tests and older inputs.
    """

    if articles.empty:
        result = articles.copy()
        result["match_id"] = pd.Series(dtype=object)
        return result
    frame = articles.copy()
    if "seen_datetime" not in frame:
        if "pubDate" in frame:
            frame["seen_datetime"] = pd.to_datetime(frame["pubDate"], errors="coerce", utc=True)
        else:
            frame["seen_datetime"] = pd.to_datetime(
                frame["seendate"], format="%Y%m%dT%H%M%SZ", errors="coerce", utc=True
            )
    matches = goalkeeper_matches.copy()
    use_end = "match_end_datetime" in matches
    if use_end:
        matches["reference_datetime"] = pd.to_datetime(matches["match_end_datetime"], utc=True)
    else:
        matches["reference_datetime"] = pd.to_datetime(matches["match_date"], utc=True)
    assigned: list[dict] = []
    max_hours = float(window_days) * 24
    for article in frame.dropna(subset=["seen_datetime"]).to_dict("records"):
        candidates = matches.loc[
            matches["goalkeeper_player_id"].eq(article["goalkeeper_player_id"])
        ].copy()
        candidates["hours_after_match"] = (
            article["seen_datetime"] - candidates["reference_datetime"]
        ).dt.total_seconds() / 3600
        candidates = candidates.loc[candidates["hours_after_match"].between(0, max_hours, inclusive="both")]
        if candidates.empty:
            continue
        chosen = candidates.sort_values(["hours_after_match", "reference_datetime"]).iloc[0]
        if (
            float(chosen["hours_after_match"]) > require_opponent_after_hours
            and "opponent_team_name" in chosen.index
            and pd.notna(chosen["opponent_team_name"])
        ):
            opponent_tokens = {
                token for token in _tokens(chosen["opponent_team_name"])
                if len(token) >= 3 and token not in {"united", "republic"}
            }
            title_tokens = _tokens(article.get("title", ""))
            aliases = set()
            opponent_norm = _ascii(chosen["opponent_team_name"]).lower()
            if opponent_norm == "united states":
                aliases.update({"usa", "us"})
            if opponent_norm == "south korea":
                aliases.add("korea")
            if not title_tokens.intersection(opponent_tokens.union(aliases)):
                continue
        article["match_id"] = chosen["match_id"]
        article["match_date"] = chosen["match_date"]
        article["hours_after_match"] = float(chosen["hours_after_match"])
        article["days_after_match"] = float(chosen["hours_after_match"] / 24)
        assigned.append(article)
    return pd.DataFrame(assigned)


def prepare_goalkeeper_headlines(
    articles: pd.DataFrame,
    goalkeeper_matches: pd.DataFrame,
    *,
    window_hours: float = 48,
) -> pd.DataFrame:
    frame = deduplicate_articles(articles)
    frame = frame.loc[
        [is_relevant_goalkeeper_headline(title, name) for title, name in zip(frame["title"], frame["query_name"])]
    ].copy()
    frame["framing"] = frame["title"].map(classify_headline_framing)
    return assign_articles_to_goalkeeper_matches(frame, goalkeeper_matches, window_days=window_hours / 24)


def aggregate_goalkeeper_match_headlines(assigned: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "goalkeeper_player_id", "match_id", "headline_count", "praise_count",
        "blame_count", "neutral_count", "praise_share", "blame_share",
        "directional_count", "directional_balance", "praise_any", "blame_any",
    ]
    if assigned.empty:
        return pd.DataFrame(columns=columns)
    counts = (
        assigned.groupby(["goalkeeper_player_id", "match_id", "framing"])
        .size().unstack(fill_value=0)
    )
    for category in ["praise", "blame", "neutral"]:
        if category not in counts:
            counts[category] = 0
    counts = counts.reset_index()
    counts["headline_count"] = counts[["praise", "blame", "neutral"]].sum(axis=1)
    counts["praise_count"] = counts["praise"].astype(int)
    counts["blame_count"] = counts["blame"].astype(int)
    counts["neutral_count"] = counts["neutral"].astype(int)
    counts["praise_share"] = counts["praise_count"] / counts["headline_count"]
    counts["blame_share"] = counts["blame_count"] / counts["headline_count"]
    counts["directional_count"] = counts["praise_count"] + counts["blame_count"]
    counts["directional_balance"] = np.where(
        counts["directional_count"].gt(0),
        (counts["praise_count"] - counts["blame_count"]) / counts["directional_count"],
        0.0,
    )
    counts["praise_any"] = counts["praise_count"].gt(0).astype(int)
    counts["blame_any"] = counts["blame_count"].gt(0).astype(int)
    return counts[columns].reset_index(drop=True)
