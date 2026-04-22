"""
Core Implementation of Google Trends Extraction

Contains: extract_google_trends(), fetch_related_queries(), batch_normalise()
Calls: Google Trends API via SerpAPI
Seeds: hardcoded as SEEDS = ['recipe', 'recipes']
"""

import logging
import os
import re
from datetime import date

import pandas as pd
import serpapi

logger = logging.getLogger(__name__)

SEEDS: list[str] = ["recipe", "recipes"]

_SEED_PATTERN = re.compile(r"\brecipes?\b", re.IGNORECASE)


def _get_client() -> serpapi.Client:
    return serpapi.Client(api_key=os.environ["SERPAPI_KEY"])


def _validate_related_query(query: str) -> bool:
    """Return True only if the query contains 'recipe' or 'recipes'."""
    return bool(_SEED_PATTERN.search(query))


def extract_google_trends(timeframe: str = "today 5-y", geo: str = "") -> pd.DataFrame:
    """
    Extract Google Trends interest-over-time and related queries for SEEDS via SerpAPI.

    Returns:
        DataFrame with columns: keyword, date, interest_score, geo, related_queries
    """
    client = _get_client()
    q = ", ".join(SEEDS)

    results = client.search({
        "engine": "google_trends",
        "q": q,
        "date": timeframe,
        "tz": "360",
        "geo": geo,
        "data_type": "TIMESERIES",
    })

    timeline = _parse_timeline(results, SEEDS)

    if timeline.empty:
        raise RuntimeError("Empty TIMESERIES response from Google Trends.")

    # Monthly resample
    timeline = (
        timeline.set_index("date")
        .groupby("keyword")
        .resample("ME")["interest_score"]
        .mean()
        .round()
        .reset_index()
    )

    rq_results = client.search({
        "engine": "google_trends",
        "q": q,
        "date": timeframe,
        "tz": "360",
        "geo": geo,
        "data_type": "RELATED_QUERIES",
    })
    related_queries = rq_results.get("related_queries", {})

    rows = []
    for kw in SEEDS:
        kw_data = timeline[timeline["keyword"] == kw].copy()
        kw_data["geo"] = geo if geo else "global"

        kw_rq = related_queries.get(kw) or {}
        rising = [
            r["query"]
            for r in (kw_rq.get("rising") or [])[:5]
            if _validate_related_query(r.get("query", ""))
        ]
        kw_data["related_queries"] = [rising] * len(kw_data)

        rows.append(kw_data)

    result_df = pd.concat(rows, ignore_index=True)
    result_df["date"] = pd.to_datetime(result_df["date"])

    return result_df.sort_values(["keyword", "date"]).reset_index(drop=True)


def fetch_related_queries(seeds: list[str], timeframe: str = "today 5-y", geo: str = "") -> pd.DataFrame:
    """
    Fetch RELATED_QUERIES for each seed via SerpAPI and return a flat DataFrame.
    Only rows where related_query contains 'recipe' or 'recipes' are kept.

    Columns: seed_query, related_query, query_type (rising|top), raw_value, fetched_date.
    """
    client = _get_client()
    fetched_date = date.today()
    rows: list[dict] = []

    results = client.search({
        "engine": "google_trends",
        "q": ", ".join(seeds),
        "date": timeframe,
        "tz": "360",
        "geo": geo,
        "data_type": "RELATED_QUERIES",
    })

    related = results.get("related_queries", {})

    for seed in seeds:
        seed_rq = related.get(seed) or {}
        for query_type in ("rising", "top"):
            for item in seed_rq.get(query_type) or []:
                query = item.get("query", "")
                if not _validate_related_query(query):
                    logger.debug("Dropping unrelated query %r (no recipe/recipes)", query)
                    continue
                raw_value = item.get("extracted_value") or item.get("value") or 0
                if isinstance(raw_value, str):
                    raw_value = 5000 if raw_value.lower() == "breakout" else 0
                rows.append({
                    "seed_query": seed,
                    "related_query": query,
                    "query_type": query_type,
                    "raw_value": int(raw_value),
                    "fetched_date": fetched_date,
                })

    if not rows:
        return pd.DataFrame(
            columns=["seed_query", "related_query", "query_type", "raw_value", "fetched_date"]
        )

    df = pd.DataFrame(rows)
    logger.info(
        "fetch_related_queries: kept %d rows (all contain 'recipe'/'recipes')", len(df)
    )
    return df


def batch_normalise(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Z-score normalise raw_value within each (seed_query, query_type) group
    and return the full DataFrame with an added normalised_score column.
    """
    if raw_df.empty:
        return raw_df.assign(normalised_score=pd.Series(dtype=float))

    def _zscore(series: pd.Series) -> pd.Series:
        std = series.std()
        if pd.isna(std) or std == 0:
            return pd.Series(0.0, index=series.index)
        return ((series - series.mean()) / std).round(6)

    df = raw_df.copy()
    df["normalised_score"] = (
        df.groupby(["seed_query", "query_type"])["raw_value"]
        .transform(_zscore)
    )
    return df


def _parse_timeline(results: dict, keywords: list[str]) -> pd.DataFrame:
    """Converts a SerpAPI TIMESERIES response into a long-form DataFrame."""
    iot = results.get("interest_over_time", {})
    timeline = iot.get("timeline_data", []) if isinstance(iot, dict) else iot
    rows = []
    for point in timeline:
        ts = pd.to_datetime(int(point["timestamp"]), unit="s")
        for val in point.get("values", []):
            query = val.get("query", "")
            if query in keywords:
                rows.append({
                    "date": ts,
                    "keyword": query,
                    "interest_score": val.get("extracted_value", 0),
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["date", "keyword", "interest_score"])
