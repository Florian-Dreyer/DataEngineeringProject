"""
Core Implementation of Google Trends Extraction

Contains: extract_google_trends() function
Calls: Google Trends API via SerpAPI
Reads: trends_keywords.txt for keyword list
Purpose: Actual data extraction logic with rate limiting and error handling
"""

import logging
import time
import os
from datetime import date

import pandas as pd
import serpapi

logger = logging.getLogger(__name__)

_BREAKOUT_VALUE = 5000  # SerpAPI encodes >5000% rising as the string "Breakout"


def _get_client() -> serpapi.Client:
    return serpapi.Client(api_key=os.environ["SERPAPI_KEY"])


def extract_google_trends(
    keywords_file=None,
    batch_size=4,
    sleep_time=5,
    timeframe="today 5-y",
    geo="",
    use_proxies=False,
    proxies=None,
):
    """
    Extract Google Trends interest-over-time data for a list of keywords with
    cross-batch normalization via SerpAPI.

    Uses 'pasta' as anchor keyword in every batch for cross-batch comparability.
    Interest scores are scaled so 'pasta' maintains consistent reference value.

    Args:
        keywords_file: Path to file containing keywords (one per line).
                      If None, uses config/trends_keywords.txt relative to project root.
        batch_size: Number of keywords per batch (excluding anchor; Google caps at 4)
        sleep_time: Seconds to sleep between batches
        timeframe: Trends timeframe (e.g., 'today 5-y')
        geo: Geographic region (empty string for global)
        use_proxies: Ignored — kept for API compatibility
        proxies: Ignored — kept for API compatibility

    Returns:
        DataFrame with columns: keyword, date, interest_score, geo, related_queries
    """
    if keywords_file is None:
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "trends_keywords.txt"
        )
        keywords_file = config_path

    if not os.path.exists(keywords_file):
        raise FileNotFoundError("Keywords file not found: {}".format(keywords_file))

    with open(keywords_file, "r", encoding="utf-8") as f:
        keywords = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    if not keywords:
        raise ValueError("No keywords found in file")

    if "pasta" not in keywords:
        keywords.insert(0, "pasta")

    anchor_keyword = "pasta"
    other_keywords = [kw for kw in keywords if kw != anchor_keyword]

    if len(other_keywords) < batch_size:
        raise ValueError("Need at least {} non-anchor keywords for batching".format(batch_size))

    batches = []
    for i in range(0, len(other_keywords), batch_size):
        batch = [anchor_keyword] + other_keywords[i : i + batch_size]
        batches.append(batch)

    print("Created {} batches with anchor keyword '{}'".format(len(batches), anchor_keyword))

    client = _get_client()
    all_data = []
    anchor_scores = {}
    successful_batches = []

    for batch_idx, batch in enumerate(batches):
        print("Processing batch {}: {}".format(batch_idx + 1, batch))

        max_retries = 3
        retry_count = 0
        success = False
        last_exc = None

        while retry_count < max_retries and not success:
            try:
                results = client.search({
                    "engine": "google_trends",
                    "q": ", ".join(batch),
                    "date": timeframe,
                    "tz": "360",
                    "geo": geo,
                    "data_type": "TIMESERIES",
                })

                timeline = _parse_timeline(results, batch)

                if timeline.empty:
                    print("  Warning: Empty response for batch {}".format(batch))
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = sleep_time * (2 ** retry_count)
                        print("  Retrying after {} seconds...".format(wait_time))
                        time.sleep(wait_time)
                    continue

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
                    "q": ", ".join(batch),
                    "date": timeframe,
                    "tz": "360",
                    "geo": geo,
                    "data_type": "RELATED_QUERIES",
                })
                related_queries = rq_results.get("related_queries", {})

                anchor_rows = timeline[timeline["keyword"] == anchor_keyword]
                if not anchor_rows.empty:
                    anchor_score = anchor_rows["interest_score"].max()
                    if anchor_score and anchor_score > 0:
                        anchor_scores[batch_idx] = anchor_score

                for kw in batch:
                    kw_data = timeline[timeline["keyword"] == kw].copy()
                    kw_data["geo"] = geo if geo else "global"

                    rising = []
                    kw_rq = related_queries.get(kw) or {}
                    rising_list = kw_rq.get("rising") or []
                    rising = [r["query"] for r in rising_list[:5]]
                    kw_data["related_queries"] = [rising] * len(kw_data)

                    all_data.append(kw_data)

                successful_batches.append(batch_idx)
                success = True

            except Exception as e:
                last_exc = e
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = sleep_time * (2 ** retry_count)
                    print("  Error: {}. Retrying after {} seconds...".format(e, wait_time))
                    time.sleep(wait_time)
                else:
                    print("  Error processing batch {}: {} (max retries exceeded)".format(batch, e))

        if batch_idx < len(batches) - 1 and success:
            print("  Sleeping {} seconds...".format(sleep_time))
            time.sleep(sleep_time)

    if not all_data:
        raise RuntimeError("No data retrieved from Google Trends.")

    result_df = pd.concat(all_data, ignore_index=True)
    result_df["date"] = pd.to_datetime(result_df["date"])

    if anchor_scores:
        target_anchor_score = sum(anchor_scores.values()) / len(anchor_scores)
        normalized_data = []
        for batch_idx in successful_batches:
            if batch_idx in anchor_scores:
                batch_anchor_score = anchor_scores[batch_idx]
                scale_factor = target_anchor_score / batch_anchor_score if batch_anchor_score > 0 else 1.0
                batch_keywords = batches[batch_idx]
                batch_data = result_df[result_df["keyword"].isin(batch_keywords)].copy()
                batch_data["interest_score"] = (
                    batch_data["interest_score"] * scale_factor
                ).round().clip(0, 100).astype(int)
                normalized_data.append(batch_data)
        if normalized_data:
            result_df = pd.concat(normalized_data, ignore_index=True)

    return result_df.sort_values(["keyword", "date"]).reset_index(drop=True)


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


def load_keywords_from_file(file_path):
    """Load keywords from a text file."""
    with open(file_path, "r") as f:
        return [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]


# ---------------------------------------------------------------------------
# Related-queries extraction (used by the batch pipeline DAG task)
# ---------------------------------------------------------------------------


def fetch_related_queries(
    seeds: list[str],
    *,
    rate_limit_sleep_secs: int = 60,
) -> pd.DataFrame:
    """
    Calls SerpAPI Google Trends related_queries for all seeds in a single request
    and returns a DataFrame with columns: seed_query, related_query, query_type,
    fetched_date, raw_value.

    On failure retries with exponential backoff (up to 4 attempts).
    """
    client = _get_client()
    fetched_date = date.today()
    rows: list[dict] = []

    logger.info("Fetching related queries for: %s", seeds)
    _fetch_related_batch(client, seeds, fetched_date, rows, rate_limit_sleep_secs)

    if not rows:
        return pd.DataFrame(
            columns=["seed_query", "related_query", "query_type", "fetched_date", "raw_value"]
        )

    df = pd.DataFrame(rows)
    df["raw_value"] = df["raw_value"].astype(int)
    return df[["seed_query", "related_query", "query_type", "fetched_date", "raw_value"]]


def batch_normalise(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a normalised_score column via z-score within each seed_query group.
    Groups with std == 0 receive normalised_score = 0.0.
    """
    df = raw_df.copy()
    df["normalised_score"] = 0.0
    for _seed, group in df.groupby("seed_query"):
        mean = group["raw_value"].mean()
        std = group["raw_value"].std()
        if pd.isna(std) or std == 0.0:
            df.loc[group.index, "normalised_score"] = 0.0
        else:
            df.loc[group.index, "normalised_score"] = (
                (group["raw_value"] - mean) / std
            ).round(6)
    return df


def _fetch_related_batch(
    client: serpapi.Client,
    seeds: list[str],
    fetched_date: date,
    rows: list[dict],
    rate_limit_sleep_secs: int,
) -> None:
    """Fetches related_queries via SerpAPI; appends result dicts to rows."""
    max_attempts = 4
    for attempt in range(max_attempts):
        try:
            results = client.search({
                "engine": "google_trends",
                "q": ", ".join(seeds),
                "date": "today 3-m",
                "tz": "360",
                "data_type": "RELATED_QUERIES",
            })
            related: dict = results.get("related_queries", {})
            for seed in seeds:
                seed_data = related.get(seed) or {}
                for query_type in ("top", "rising"):
                    entries = seed_data.get(query_type) or []
                    for r in entries:
                        raw_value = r.get("value", 0)
                        if isinstance(raw_value, str):
                            raw_value = _BREAKOUT_VALUE
                        try:
                            raw_value = int(raw_value)
                        except (ValueError, TypeError):
                            raw_value = 0
                        rows.append({
                            "seed_query": seed,
                            "related_query": str(r["query"]),
                            "query_type": query_type,
                            "fetched_date": fetched_date,
                            "raw_value": raw_value,
                        })
            return
        except Exception as exc:
            if attempt < max_attempts - 1:
                sleep_secs = rate_limit_sleep_secs * (2 ** attempt)
                logger.warning(
                    "Error on keywords %s: %s — sleeping %ds then retrying (attempt %d/%d).",
                    seeds, exc, sleep_secs, attempt + 1, max_attempts,
                )
                time.sleep(sleep_secs)
            else:
                logger.error("Failed after %d attempts for %s: %s — skipping.", max_attempts, seeds, exc)
