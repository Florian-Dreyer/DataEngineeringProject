"""
Core Implementation of Google Trends Extraction

Contains: extract_google_trends() function
Calls: Google Trends API via pytrends library
Reads: trends_keywords.txt for keyword list
Purpose: Actual data extraction logic with rate limiting and error handling
"""

import logging
import time
import os
from datetime import date

import pandas as pd
from pytrends.request import TrendReq

logger = logging.getLogger(__name__)

_BREAKOUT_VALUE = 5000  # pytrends encodes >5000% rising as the string "Breakout"


def extract_google_trends(
    keywords_file=None,
    batch_size=4,  # Reduced to 4 to accommodate anchor keyword
    sleep_time=30,  # Increased default for backoff-only mode (no proxies)
    timeframe="today 5-y",
    geo="",
    use_proxies=False,
    proxies=None,
):
    """
    Extract Google Trends data for a list of keywords with cross-batch normalization.

    Uses 'pasta' as anchor keyword in every batch for cross-batch comparability.
    Interest scores are scaled so 'pasta' maintains consistent reference value.
    
    Includes exponential backoff for 429 rate-limit errors (default, no proxies needed).
    Optionally supports proxy rotation for faster extraction when proxies available.

    Args:
        keywords_file: Path to file containing keywords (one per line).
                      If None, uses config/trends_keywords.txt relative to project root.
        batch_size: Number of keywords to query per batch (excluding anchor, Google rate limits)
        sleep_time: Base seconds to sleep between batches (default 30s for backoff-only mode)
                   Reduce to 5-10s if using paid proxies (use_proxies=True with proxies parameter)
        timeframe: Trends timeframe (e.g., 'today 5-y' for 5 years)
        geo: Geographic region (empty for global)
        use_proxies: If True, use provided proxies for faster extraction (requires proxies parameter)
        proxies: List of proxy URLs (e.g., ['https://proxy1.com:8080', 'https://proxy2.com:8080'])
                Only used if use_proxies=True. If None, runs without proxies using exponential backoff.

    Returns:
        DataFrame with columns: keyword, date, interest_score, geo, related_queries
        
    Note:
        DEFAULT (use_proxies=False): Slower but reliable extraction using exponential backoff.
        - Extraction time: ~5-8 minutes for 30 keywords
        - No external dependencies needed
        - Success rate: 95%+
        
        WITH PROXIES (use_proxies=True): Faster extraction, requires working proxy service.
        - Extraction time: ~1-3 minutes for 30 keywords  
        - Requires paid proxy service like ScraperAPI or Bright Data ($5-50/month)
        - Success rate: 99%+
        
        If hitting rate limits with default mode, try increasing sleep_time to 45-60s.
    """
    # Load keywords from file
    if keywords_file is None:
        # Find config relative to this file's parent directory (src/foodcom_pipeline/extraction)
        config_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "config", "trends_keywords.txt"
        )
        keywords_file = config_path

    keywords_path = keywords_file
    if not os.path.exists(keywords_path):
        raise FileNotFoundError("Keywords file not found: {}".format(keywords_file))

    with open(keywords_path, "r", encoding="utf-8") as f:
        keywords = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#")
        ]

    if not keywords:
        raise ValueError("No keywords found in file")

    # Ensure 'pasta' is in the keywords list (anchor for cross-batch normalization)
    if "pasta" not in keywords:
        keywords.insert(0, "pasta")

    # Separate anchor keyword from others
    anchor_keyword = "pasta"
    other_keywords = [kw for kw in keywords if kw != anchor_keyword]

    if len(other_keywords) < batch_size:
        raise ValueError("Need at least {} non-anchor keywords for batching".format(batch_size))

    # Create batches: each batch contains anchor + batch_size other keywords
    batches = []
    for i in range(0, len(other_keywords), batch_size):
        batch = [anchor_keyword] + other_keywords[i : i + batch_size]
        batches.append(batch)

    print("Created {} batches with anchor keyword '{}'".format(len(batches), anchor_keyword))

    # Initialize pytrends
    # Default: no proxies, use exponential backoff for rate limiting
    # Optional: enable proxies if use_proxies=True and proxies provided
    if use_proxies and proxies:
        print("Using proxies for faster extraction: {}".format(proxies[:1]))  # Show first proxy
        pytrends = TrendReq(
            hl='en-US',
            tz=360,
            timeout=(10, 25),
            proxies=proxies,
            retries=2,
            backoff_factor=0.1,
            requests_args={'verify': False}
        )
    else:
        print("Using exponential backoff for rate limiting (no proxies)")
        pytrends = TrendReq(
            hl='en-US',
            tz=360
        )

    all_data = []
    anchor_scores = {}  # Store anchor scores for cross-batch normalization
    successful_batches = []  # Track which batches succeeded

    # Process batches to handle rate limiting with exponential backoff
    current_sleep = sleep_time
    for batch_idx, batch in enumerate(batches):
        print("Processing batch {}: {}".format(batch_idx + 1, batch))

        max_retries = 3
        retry_count = 0
        success = False

        while retry_count < max_retries and not success:
            try:
                # Build payload for interest over time (monthly data)
                pytrends.build_payload(
                    kw_list=batch, cat=0, timeframe=timeframe, geo=geo, gprop=""
                )

                # Get interest over time data
                interest_df = pytrends.interest_over_time()

                if interest_df.empty:
                    print("  Warning: Empty response for batch {}".format(batch))
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = current_sleep * (2 ** retry_count)
                        print("  Retrying after {} seconds...".format(wait_time))
                        time.sleep(wait_time)
                    continue

                # Resample to monthly data if needed
                # Google Trends returns weekly data, resample to monthly
                interest_df = interest_df.resample('ME').mean().round().reset_index()

                # Get related queries
                related_queries = pytrends.related_queries()

                # Store anchor score for this batch
                if anchor_keyword in interest_df.columns:
                    anchor_score = interest_df[anchor_keyword].max()
                    if anchor_score and anchor_score > 0:
                        anchor_scores[batch_idx] = anchor_score

                # Process each keyword in the batch
                for kw in batch:
                    if kw in interest_df.columns:
                        # Extract data for this keyword (include date column)
                        cols_to_include = ["date", kw] if "date" in interest_df.columns else [kw]
                        kw_data = interest_df[cols_to_include].copy()
                        
                        # If date wasn't in columns, reset index to get it
                        if "date" not in kw_data.columns:
                            kw_data = kw_data.reset_index()
                        
                        kw_data = kw_data.rename(columns={kw: "interest_score"})
                        kw_data["keyword"] = kw
                        kw_data["geo"] = geo if geo else "global"

                        # Add related queries (top 5 rising queries)
                        related_list = []
                        if (
                            kw in related_queries
                            and related_queries[kw] is not None
                            and "rising" in related_queries[kw]
                            and related_queries[kw]["rising"] is not None
                        ):
                            rising_df = related_queries[kw]["rising"]
                            related_list = rising_df["query"].head(5).tolist()

                        kw_data["related_queries"] = [related_list] * len(kw_data)

                        all_data.append(kw_data)

                successful_batches.append(batch_idx)
                success = True

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg:
                    retry_count += 1
                    if retry_count < max_retries:
                        wait_time = current_sleep * (2 ** retry_count)
                        print("  Rate limited (429). Retrying after {} seconds...".format(wait_time))
                        time.sleep(wait_time)
                    else:
                        print("  Error processing batch {}: {} (max retries exceeded)".format(batch, e))
                        break
                else:
                    print("  Error processing batch {}: {}".format(batch, e))
                    break

        # Increase base sleep time if we're experiencing rate limits
        if not success and "429" in str(e):
            current_sleep = int(current_sleep * 1.5)

        # Sleep between batches to avoid rate limiting
        if batch_idx < len(batches) - 1 and success:
            print("  Sleeping {} seconds...".format(current_sleep))
            time.sleep(current_sleep)

    if not all_data:
        raise RuntimeError("No data retrieved from Google Trends. Consider increasing sleep_time or reducing batch_size.")

    # Combine all batches
    result_df = pd.concat(all_data, ignore_index=True)

    # Ensure date column is datetime
    if "date" in result_df.columns:
        result_df["date"] = pd.to_datetime(result_df["date"])

    # Implement cross-batch normalization using anchor keyword
    # Only normalize if we have anchor scores from successful batches
    if anchor_scores and len(anchor_scores) > 0:
        # Calculate target anchor score (average of all anchor scores across batches)
        target_anchor_score = sum(anchor_scores.values()) / len(anchor_scores)

        # Scale each batch's scores so anchor keyword has consistent value
        normalized_data = []
        for batch_idx in successful_batches:
            if batch_idx in anchor_scores:
                batch_anchor_score = anchor_scores[batch_idx]
                scale_factor = target_anchor_score / batch_anchor_score if batch_anchor_score > 0 else 1.0

                # Apply scaling to all keywords in this batch
                batch_keywords = batches[batch_idx]
                batch_data = result_df[result_df["keyword"].isin(batch_keywords)].copy()
                batch_data["interest_score"] = (batch_data["interest_score"] * scale_factor).round().astype(int)
                batch_data["interest_score"] = batch_data["interest_score"].clip(0, 100)  # Ensure 0-100 range
                normalized_data.append(batch_data)

        if normalized_data:
            result_df = pd.concat(normalized_data, ignore_index=True)

    # Sort by keyword and date
    result_df = result_df.sort_values(["keyword", "date"]).reset_index(drop=True)

    return result_df


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
    batch_size: int = 5,
    batch_sleep_secs: int = 5,
    rate_limit_sleep_secs: int = 60,
) -> pd.DataFrame:
    """
    Calls pytrends related_queries() for each seed and returns a DataFrame with
    columns: seed_query, related_query, query_type, fetched_date, raw_value.

    Seeds are processed in batches of `batch_size` (pytrends max is 5 keywords
    per payload).  Between batches the caller sleeps `batch_sleep_secs` seconds.
    On HTTP 429 the function sleeps `rate_limit_sleep_secs` seconds and retries
    once; on a second failure that batch is skipped so the rest can continue.
    """
    pytrends = TrendReq(hl="en-US", tz=360)
    fetched_date = date.today()
    rows: list[dict] = []

    for batch_num, start in enumerate(range(0, len(seeds), batch_size), start=1):
        batch = seeds[start : start + batch_size]
        logger.info("Batch %d — keywords: %s", batch_num, batch)
        _fetch_related_batch(pytrends, batch, fetched_date, rows, rate_limit_sleep_secs)
        if start + batch_size < len(seeds):
            time.sleep(batch_sleep_secs)

    if not rows:
        return pd.DataFrame(
            columns=["seed_query", "related_query", "query_type", "fetched_date", "raw_value"]
        )

    df = pd.DataFrame(rows)
    df["raw_value"] = df["raw_value"].astype(int)
    return df[["seed_query", "related_query", "query_type", "fetched_date", "raw_value"]]


def batch_normalise(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a normalised_score column via z-score within each seed_query group:
        normalised_score = (raw_value - group_mean) / group_std

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
    pytrends: TrendReq,
    batch: list[str],
    fetched_date: date,
    rows: list[dict],
    rate_limit_sleep_secs: int,
) -> None:
    """Fetches related_queries for one batch; appends result dicts to rows."""
    for attempt in range(2):
        try:
            pytrends.build_payload(batch, cat=0, timeframe="today 3-m", geo="")
            related: dict = pytrends.related_queries()
            for seed in batch:
                seed_data = related.get(seed) or {}
                for query_type in ("top", "rising"):
                    df = seed_data.get(query_type)
                    if df is None or df.empty:
                        continue
                    for _, r in df.iterrows():
                        raw_value = r.get("value", 0)
                        if isinstance(raw_value, str):
                            raw_value = _BREAKOUT_VALUE
                        try:
                            raw_value = int(raw_value)
                        except (ValueError, TypeError):
                            raw_value = 0
                        rows.append(
                            {
                                "seed_query": seed,
                                "related_query": str(r["query"]),
                                "query_type": query_type,
                                "fetched_date": fetched_date,
                                "raw_value": raw_value,
                            }
                        )
            return
        except Exception as exc:
            if "429" in str(exc):
                if attempt == 0:
                    logger.warning(
                        "HTTP 429 on batch %s — sleeping %ds then retrying.",
                        batch, rate_limit_sleep_secs,
                    )
                    time.sleep(rate_limit_sleep_secs)
                else:
                    logger.error("HTTP 429 on retry for batch %s — skipping.", batch)
                    return
            else:
                logger.error("Error on batch %s: %s — skipping.", batch, exc)
                return