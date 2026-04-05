"""
Core Implementation of Google Trends Extraction

Contains: extract_google_trends() function
Calls: Google Trends API via pytrends library
Reads: trends_keywords.txt for keyword list
Purpose: Actual data extraction logic with rate limiting and error handling
"""

import time
import os

import pandas as pd
from pytrends.request import TrendReq


def extract_google_trends(
    keywords_file=None,
    batch_size=5,
    sleep_time=5,
    timeframe="today 5-y",
    geo="",
):
    """
    Extract Google Trends data for a list of keywords.

    Args:
        keywords_file: Path to file containing keywords (one per line).
                      If None, uses config/trends_keywords.txt relative to project root.
        batch_size: Number of keywords to query per batch (Google rate limits)
        sleep_time: Seconds to sleep between batches
        timeframe: Trends timeframe (e.g., 'today 5-y' for 5 years)
        geo: Geographic region (empty for global)

    Returns:
        DataFrame with columns: keyword, date, interest_score, geo, related_queries
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

    # Initialize pytrends
    pytrends = TrendReq(hl="en-US", tz=360)

    all_data = []

    # Process keywords in batches to handle rate limiting
    for i in range(0, len(keywords), batch_size):
        batch = keywords[i : i + batch_size]
        print("Processing batch {}: {}".format(i//batch_size + 1, batch))

        try:
            # Build payload for interest over time
            pytrends.build_payload(
                kw_list=batch, cat=0, timeframe=timeframe, geo=geo, gprop=""
            )

            # Get interest over time data
            interest_df = pytrends.interest_over_time()

            # Get related queries
            related_queries = pytrends.related_queries()

            # Process each keyword in the batch
            for kw in batch:
                if kw in interest_df.columns:
                    # Extract data for this keyword
                    kw_data = interest_df[[kw]].reset_index()
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

        except Exception as e:
            print("Error processing batch {}: {}".format(batch, e))
            continue

        # Sleep between batches to avoid rate limiting
        if i + batch_size < len(keywords):
            print("Sleeping {} seconds...".format(sleep_time))
            time.sleep(sleep_time)

    if not all_data:
        raise RuntimeError("No data retrieved from Google Trends")

    # Combine all batches
    result_df = pd.concat(all_data, ignore_index=True)

    # Ensure date column is datetime
    result_df["date"] = pd.to_datetime(result_df["date"])

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