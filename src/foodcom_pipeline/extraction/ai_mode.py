"""
ai_mode.py
----------
Extracts AI Mode text blocks from Google Search via SerpAPI and derives
food/cuisine term-frequency scores from the returned content.

Two public functions used by the batch pipeline:
  fetch_ai_mode_blocks  — calls SerpAPI, returns raw text-block rows
  score_terms           — counts token frequency, applies z-score normalisation
"""

import logging
import re
from collections import Counter
from datetime import date
import pandas as pd

# Patterns that indicate an AI assistant UI string, not food content
_JUNK_PATTERNS = re.compile(
    r'would you like'
    r'|something went wrong'
    r'|ai response wasn\'t generated'
    r'|let me know if'
    r'|i hope (that |this )?helps'
    r'|here are some'
    r'|feel free to ask'
    r'|shopping list',
    flags=re.IGNORECASE,
)
_MIN_BODY_LENGTH = 5

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
    return f"{seed}: 20 dish names, comma-separated"


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
            ).as_dict()

            logger.debug("SerpAPI top-level keys for seed %r: %s", seed, list(results.keys()))

            ai_section = results.get("ai_mode") or results.get("ai_overview") or results

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

            seed_rows_before = len(rows)

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

                body_clean = str(body).strip()
                if len(body_clean) < _MIN_BODY_LENGTH or _JUNK_PATTERNS.search(body_clean):
                    logger.debug("Dropping junk block for %r: %r", seed, body_clean[:80])
                    continue

                rows.append(
                    {
                        "seed_query": seed,
                        "prompted_query": prompted_query,
                        "block_index": idx,
                        "title": title,
                        "body": body_clean,
                        "fetched_date": fetched_date,
                    }
                )

            # text_blocks list-type entries carry no content in SerpAPI's response;
            # fall back to reconstructed_markdown which has the actual dish list.
            if len(rows) == seed_rows_before:
                md = results.get("reconstructed_markdown") or ""
                md_clean = re.sub(r"\\(.)", r"\1", md).strip()
                if len(md_clean) >= _MIN_BODY_LENGTH and not _JUNK_PATTERNS.search(md_clean):
                    rows.append(
                        {
                            "seed_query": seed,
                            "prompted_query": prompted_query,
                            "block_index": 0,
                            "title": None,
                            "body": md_clean,
                            "fetched_date": fetched_date,
                        }
                    )
                    logger.info(
                        "Used reconstructed_markdown fallback for seed %r (%d chars).",
                        seed, len(md_clean),
                    )
                else:
                    logger.warning(
                        "No AI Mode text content returned for seed %r (prompt=%r)",
                        seed,
                        prompted_query,
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
        meaningful_words = [w for w in words if len(w) > 2 and w not in _STOPWORDS]

        # require at least 2 meaningful words (to filter out generic phrases like "chicken recipes")
        if len(meaningful_words) < 2:
            return False

        # reject common generic descriptors that aren't dish names
        generic_descriptors = {
            "recipes", "recipe", "dish", "dishes", "food", "foods",
            "meal", "meals", "ideas", "idea", "easy", "quick", "simple",
            "best", "top", "popular", "favorite", "favourite", "classic",
        }
        # if all words are generic descriptors, reject
        if all(w.lower() in generic_descriptors for w in words):
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

