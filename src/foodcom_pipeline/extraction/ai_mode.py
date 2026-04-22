"""
ai_mode.py
----------
Extracts AI Mode text blocks from Google Search via SerpAPI and derives
food/cuisine term-frequency scores from the returned content.

Three public functions used by the batch pipeline:
  fetch_ai_mode_blocks  — calls SerpAPI, returns raw text-block rows
  score_terms           — counts token frequency, applies z-score normalisation
  merge_with_trends     — left-joins term scores against google_trends_normalised
"""

import logging
import re
from collections import Counter
from datetime import date
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Stopwords to exclude from term frequency scoring.
# Kept inline to avoid an NLTK dependency.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "it", "its", "you", "your", "we",
        "our", "they", "their", "this", "that", "these", "those", "i", "my",
        "as", "if", "when", "what", "how", "so", "up", "out", "no", "not",
        "more", "some", "any", "all", "like", "just", "also", "such", "than",
        "into", "over", "after", "before", "about", "get", "make", "use",
        "great", "good", "best", "easy", "quick", "simple", "try", "here",
        "can", "made", "include", "including", "need", "needs", "want",
        "time", "day", "week", "night", "add", "top", "new", "one", "two",
        "three", "many", "much", "well", "re", "ll", "ve", "don", "t",
    }
)


# ---------------------------------------------------------------------------
# Public: SerpAPI fetch
# ---------------------------------------------------------------------------


def fetch_ai_mode_blocks(seeds: list[str], api_key: str) -> pd.DataFrame:
    """
    Calls the SerpAPI ``google_ai_mode`` engine for each seed query and
    returns a flattened DataFrame of text blocks.

    Columns: seed_query, block_index, title (nullable), body, fetched_date.

    Each SerpAPI call is wrapped individually so a single failure does not
    abort the rest.  Failed queries are logged and skipped.
    """
    from serpapi import GoogleSearch  # deferred: not available in all envs

    fetched_date = date.today()
    rows: list[dict] = []

    for seed in seeds:
        try:
            results = GoogleSearch(
                {"engine": "google_ai_mode", "q": seed, "api_key": api_key}
            ).get_dict()

            # SerpAPI may nest text_blocks under an "ai_mode" key or at the top level
            ai_section = results.get("ai_mode", results)
            text_blocks = ai_section.get("text_blocks", [])

            if not text_blocks:
                logger.warning("No text_blocks returned for query %r", seed)

            for idx, block in enumerate(text_blocks):
                title = block.get("title") or None
                body = (
                    block.get("snippet")
                    or block.get("body")
                    or block.get("text")
                    or ""
                )
                rows.append(
                    {
                        "seed_query": seed,
                        "block_index": idx,
                        "title": title,
                        "body": str(body),
                        "fetched_date": fetched_date,
                    }
                )

        except Exception as exc:
            logger.error("SerpAPI call failed for query %r: %s", seed, exc)

    if not rows:
        return pd.DataFrame(
            columns=["seed_query", "block_index", "title", "body", "fetched_date"]
        )

    return pd.DataFrame(rows)[
        ["seed_query", "block_index", "title", "body", "fetched_date"]
    ]


# ---------------------------------------------------------------------------
# Public: term scoring
# ---------------------------------------------------------------------------


def score_terms(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Counts food/cuisine token frequency across all text blocks, then applies
    z-score normalisation within the full corpus.

    Tokenisation: lowercase, split on non-alphanumeric, drop stopwords and
    tokens shorter than 3 characters.

    Columns: term, raw_frequency, normalised_score, fetched_date.
    """
    fetched_date = date.today()

    combined_text = " ".join(
        (raw_df["title"].fillna("") + " " + raw_df["body"].fillna("")).tolist()
    ).lower()

    tokens = re.split(r"[^a-z0-9]+", combined_text)
    filtered = [t for t in tokens if len(t) >= 3 and t not in _STOPWORDS]

    counts = Counter(filtered)
    if not counts:
        return pd.DataFrame(
            columns=["term", "raw_frequency", "normalised_score", "fetched_date"]
        )

    df = pd.DataFrame(
        [{"term": t, "raw_frequency": c} for t, c in counts.items()]
    )

    mean = df["raw_frequency"].mean()
    std = df["raw_frequency"].std()
    if pd.isna(std) or std == 0.0:
        df["normalised_score"] = 0.0
    else:
        df["normalised_score"] = ((df["raw_frequency"] - mean) / std).round(6)

    df["fetched_date"] = fetched_date
    return (
        df[["term", "raw_frequency", "normalised_score", "fetched_date"]]
        .sort_values("raw_frequency", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Public: merge with Google Trends
# ---------------------------------------------------------------------------


def merge_with_trends(
    term_scores_df: pd.DataFrame,
    trends_normalised_path: Path | str,
) -> pd.DataFrame:
    """
    Left-joins ai_mode_term_scores with google_trends_normalised on
    term ↔ related_query (aggregating trends scores to one row per term).

    combined_score = mean of whichever of (ai_mode_score, trends_score) are
    non-null; falls back to the single available score when only one exists.

    Columns: term, ai_mode_score, trends_score, combined_score, fetched_date.
    """
    trends_path = Path(trends_normalised_path)

    if trends_path.is_file():
        trends_df = pd.read_parquet(trends_path)
        # Aggregate to one normalised_score per related_query (mean across seeds/types)
        trends_agg = (
            trends_df.groupby("related_query")["normalised_score"]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "related_query": "term",
                    "normalised_score": "trends_score",
                }
            )
        )
    else:
        logger.warning(
            "google_trends_normalised.parquet not found at %s; "
            "market_signals will have no trends_score.",
            trends_normalised_path,
        )
        trends_agg = pd.DataFrame(columns=["term", "trends_score"])

    merged = (
        term_scores_df[["term", "normalised_score", "fetched_date"]]
        .rename(columns={"normalised_score": "ai_mode_score"})
        .merge(trends_agg, on="term", how="left")
    )

    # combined_score: pandas mean(skipna=True) gives the single value when only one exists
    merged["combined_score"] = (
        merged[["ai_mode_score", "trends_score"]].mean(axis=1, skipna=True)
    )

    return merged[
        ["term", "ai_mode_score", "trends_score", "combined_score", "fetched_date"]
    ]
