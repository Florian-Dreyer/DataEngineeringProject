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

def build_ai_query(seed: str) -> str:
    return (
        f"{seed}. "
        "Generate a list of 20 high-volume search terms and trending dishes. "
        "Each item must be distilled to its core dish name, meaning the primary keyword used to identify broad search intent. "
        "Remove descriptive or branded adjectives such as 'viral', 'marry me', 'one-pot', 'sheet-pan', 'crispy', 'creamy', or similar stylistic modifiers unless they are essential to the dish identity. "
        "Do not include explanations, categories, numbering, or extra commentary. "
        "Only output dish names, separated by commas. "
        "Examples: Tortellini, Garlic Chicken, Korean Beef Bowl, Feta Pasta, Birria Tacos."
    )


def fetch_ai_mode_blocks(seeds, api_key) -> pd.DataFrame:
    import serpapi

    fetched_date = date.today()
    rows: list[dict] = []

    try:
        client = serpapi.Client(api_key=api_key)
    except Exception as e:
        raise RuntimeError("Failed to initialize SerpAPI client") from e

    for seed in seeds:
        prompted_query = build_ai_query(seed)

        try:
            results = client.search(
                {
                    "engine": "google_ai_mode",
                    "q": prompted_query,
                }
            )

            ai_section = results.get("ai_mode", results)

            text_blocks = ai_section.get("text_blocks", [])
            if not text_blocks and isinstance(ai_section.get("text_blocks"), dict):
                text_blocks = [ai_section["text_blocks"]]

            # fallback shapes in case AI Mode returns text elsewhere
            if not text_blocks:
                fallback_text = (
                    ai_section.get("text")
                    or ai_section.get("answer")
                    or ai_section.get("snippet")
                    or results.get("text")
                    or results.get("answer")
                    or results.get("snippet")
                    or ""
                )

                if fallback_text:
                    text_blocks = [{"title": None, "text": fallback_text}]

            if not text_blocks:
                logger.warning(
                    "No AI Mode text content returned for seed %r (prompt=%r)",
                    seed,
                    prompted_query,
                )

            for idx, block in enumerate(text_blocks):
                if not isinstance(block, dict):
                    block = {"title": None, "text": str(block)}

                title = block.get("title") or None
                body = (
                    block.get("snippet")
                    or block.get("body")
                    or block.get("text")
                    or block.get("answer")
                    or ""
                )

                rows.append(
                    {
                        "seed_query": seed,
                        "prompted_query": prompted_query,
                        "block_index": idx,
                        "title": title,
                        "body": str(body).strip(),
                        "fetched_date": fetched_date,
                    }
                )

        except Exception as exc:
            logger.error(
                "SerpAPI call failed for seed %r (prompt=%r): %s",
                seed,
                prompted_query,
                exc,
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "seed_query",
                "prompted_query",
                "block_index",
                "title",
                "body",
                "fetched_date",
            ]
        )

    return pd.DataFrame(rows)[
        [
            "seed_query",
            "prompted_query",
            "block_index",
            "title",
            "body",
            "fetched_date",
        ]
    ]


# ---------------------------------------------------------------------------
# Public: term scoring
# ---------------------------------------------------------------------------


def score_terms(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scores dish phrases rather than single-word tokens.

    Assumes AI Mode output has been prompt-engineered to return dish names
    separated by commas, line breaks, semicolons, or bullets.

    Columns: term, raw_frequency, normalised_score, fetched_date.
    """
    fetched_date = date.today()

    if raw_df.empty:
        return pd.DataFrame(
            columns=["term", "raw_frequency", "normalised_score", "fetched_date"]
        )

    def _clean_term(term: str) -> str:
        term = str(term).strip().lower()

        # remove numbering / bullets at start
        term = re.sub(r"^\s*(?:\d+[\.\)]\s*|[-•*]\s*)", "", term)

        # normalize punctuation / whitespace
        term = re.sub(r"[“”\"']", "", term)
        term = re.sub(r"\s+", " ", term).strip(" ,;:-")

        return term

    def _is_valid_term(term: str) -> bool:
        if not term:
            return False

        # reject very short fragments
        if len(term) < 3:
            return False

        # reject generic query/category phrases
        generic_phrases = {
            "dinner ideas",
            "easy weeknight meals",
            "easy meals",
            "quick meals",
            "quick dinner ideas",
            "dinner recipes",
            "meal ideas",
            "recipe ideas",
            "high volume search terms",
            "trending dishes",
            "dish names",
        }
        if term in generic_phrases:
            return False

        # reject fragments with too few alphabetic chars
        if len(re.sub(r"[^a-z]", "", term)) < 3:
            return False

        # reject phrases that are almost entirely stopwords
        words = term.split()
        content_words = [w for w in words if w not in _STOPWORDS]
        if len(content_words) == 0:
            return False

        return True

    phrases: list[str] = []

    for _, row in raw_df.iterrows():

        # split on common list separators from AI output
        candidates = re.split(r",|\n|;|\||•", str(row.get("body", "")))

        for candidate in candidates:
            cleaned = _clean_term(candidate)
            if _is_valid_term(cleaned):
                phrases.append(cleaned)

    counts = Counter(phrases)

    if not counts:
        return pd.DataFrame(
            columns=["term", "raw_frequency", "normalised_score", "fetched_date"]
        )

    df = pd.DataFrame(
        [{"term": term, "raw_frequency": freq} for term, freq in counts.items()]
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
        .sort_values(["raw_frequency", "term"], ascending=[False, True])
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
