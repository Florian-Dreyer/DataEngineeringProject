"""
match.py
--------
Two-tier matcher: links each external_recipe_terms row to the closest
Food.com recipe in recipe_term_index.

Tier 1 — Embeddings (optional):
  sentence-transformers all-MiniLM-L6-v2; cosine similarity over
  recipe_name + ingredients_summary representations.

Tier 2 — Lexical baseline (always available):
  1. Exact canonical_term match           (O(1) hash lookup)
  2. Token-overlap candidate selection    (inverted index)
  3. Combined score on candidates         max(token Jaccard, fuzzy ratio)
     — RapidFuzz token_sort_ratio when available, difflib.SequenceMatcher otherwise

If sentence-transformers is not installed the pipeline continues with the
lexical baseline; no Airflow task failure is raised.

Output: recipe_gap_analysis.parquet
"""

import logging
import re
from difflib import SequenceMatcher

import pandas as pd

from foodcom_pipeline.batch.extract import STAGING_DIR, atomic_parquet
from foodcom_pipeline.utils.recipe_terms import canonicalize_recipe_term

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
    logger.debug("RapidFuzz available for fuzzy matching")
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False
    logger.info("RapidFuzz not available — using difflib.SequenceMatcher")

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
    logger.debug("sentence-transformers available for embedding matching")
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.info("sentence-transformers not installed — lexical matching only")

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# ─────────────────────────────────────────────────────────────────────────
# Staging paths
# ─────────────────────────────────────────────────────────────────────────

EXTERNAL_TERMS_STAGING = STAGING_DIR / "external_recipe_terms.parquet"
RECIPE_TERM_INDEX_STAGING = STAGING_DIR / "recipe_term_index.parquet"
RECIPE_GAP_ANALYSIS_STAGING = STAGING_DIR / "recipe_gap_analysis.parquet"

# ─────────────────────────────────────────────────────────────────────────
# Schema and thresholds
# ─────────────────────────────────────────────────────────────────────────

_GAP_COLS = [
    "source",
    "raw_term",
    "canonical_term",
    "source_score",
    "best_foodcom_recipe_id",
    "best_foodcom_recipe_name",
    "best_foodcom_canonical_term",
    "best_foodcom_similarity",
    "match_status",
    "gap_score",
    "top_gap_rank",
    "opportunity_label",
    "insight_summary",
    "gap_reason",
    "matching_method",
    "fetched_date",
]

_STRONG_MATCH = 0.80
_WEAK_MATCH = 0.60

# Caps fuzzy scoring to the top-N candidates ranked by shared-token count.
# Prevents quadratic blow-up when a common token (e.g. "chicken") maps to
# tens of thousands of recipes.
_MAX_FUZZY_CANDIDATES = 500

_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "with", "in", "on", "for", "of",
    "to", "my", "your", "our", "its", "is", "are", "was", "be",
    "i", "you", "we", "they",
})

# ─────────────────────────────────────────────────────────────────────────
# Tokenization and scoring helpers
# ─────────────────────────────────────────────────────────────────────────


def _tokenize(text: str) -> list[str]:
    """Lowercase alphabetic tokens, excluding stopwords and single chars."""
    return [
        t for t in re.findall(r"[a-z]+", str(text).lower())
        if t not in _STOPWORDS and len(t) > 1
    ]


def _token_jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _fuzzy_ratio(a: str, b: str) -> float:
    """Similarity in [0, 1]. token_sort_ratio (RapidFuzz) or SequenceMatcher."""
    if _RAPIDFUZZ_AVAILABLE:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


def _combined_score(q: str, c: str, q_tok: frozenset, c_tok: frozenset) -> float:
    """Max of token Jaccard and fuzzy ratio — best of two orthogonal signals."""
    return max(_token_jaccard(q_tok, c_tok), _fuzzy_ratio(q, c))


# ─────────────────────────────────────────────────────────────────────────
# Term index
# ─────────────────────────────────────────────────────────────────────────


class _TermIndex:
    """
    In-memory index of unique recipe canonical terms.

    Builds an inverted token → [recipe indices] map at construction time
    so candidate selection is fast (no linear scan).
    """

    def __init__(self, index_df: pd.DataFrame) -> None:
        valid = (
            index_df
            .dropna(subset=["canonical_term"])
            .loc[lambda df: df["canonical_term"].str.strip() != ""]
            .drop_duplicates(subset=["canonical_term"])
            .reset_index(drop=True)
        )

        self._terms: list[str] = valid["canonical_term"].tolist()
        self._ids: list = valid["recipe_id"].tolist()
        self._names: list[str] = valid["recipe_name"].tolist()
        self._token_sets: list[frozenset] = [
            frozenset(_tokenize(t)) for t in self._terms
        ]

        # O(1) exact lookup
        self._exact: dict[str, int] = {t: i for i, t in enumerate(self._terms)}

        # Inverted index: token → list of term indices
        self._inv: dict[str, list[int]] = {}
        for i, toks in enumerate(self._token_sets):
            for tok in toks:
                self._inv.setdefault(tok, []).append(i)

        logger.info(
            "TermIndex: %d unique canonical terms, %d token types",
            len(self._terms), len(self._inv),
        )

    def match(
        self, query: str
    ) -> tuple[int | None, str | None, str | None, float, str]:
        """
        Find best-matching recipe for *query* (a canonical term).

        Returns:
            (recipe_id, recipe_name, matched_canonical, similarity, method)

        method is one of: 'exact' | 'fuzzy' | 'no_token_overlap' | 'empty_query'
        """
        if not query:
            return None, None, None, 0.0, "empty_query"

        # ── 1. Exact ──────────────────────────────────────────────────────
        if query in self._exact:
            i = self._exact[query]
            return self._ids[i], self._names[i], self._terms[i], 1.0, "exact"

        # ── 2. Candidate selection via shared tokens ───────────────────────
        q_tok = frozenset(_tokenize(query))
        overlap_count: dict[int, int] = {}
        for tok in q_tok:
            for i in self._inv.get(tok, []):
                overlap_count[i] = overlap_count.get(i, 0) + 1

        if not overlap_count:
            return None, None, None, 0.0, "no_token_overlap"

        # Rank candidates by token overlap, cap to avoid quadratic cost
        top_candidates = sorted(
            overlap_count, key=overlap_count.__getitem__, reverse=True
        )[:_MAX_FUZZY_CANDIDATES]

        # ── 3. Combined score on top candidates ────────────────────────────
        best_i, best_score = None, 0.0
        for i in top_candidates:
            s = _combined_score(query, self._terms[i], q_tok, self._token_sets[i])
            if s > best_score:
                best_score, best_i = s, i

        if best_i is None:
            return None, None, None, 0.0, "no_match"

        return (
            self._ids[best_i],
            self._names[best_i],
            self._terms[best_i],
            round(best_score, 4),
            "fuzzy",
        )


# ─────────────────────────────────────────────────────────────────────────
# Embedding index (optional — requires sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────


def _recipe_text(row: pd.Series) -> str:
    """Build the text string used for embedding a recipe row."""
    name = str(row.get("recipe_name") or row.get("canonical_term") or "")
    ingr = row.get("ingredients_summary")
    if pd.notna(ingr) and ingr:
        # First 5 ingredients give context without bloating the embedding
        top5 = " ".join(str(ingr).split("|")[:5])
        return f"{name} {top5}".strip()
    return name.strip()


class _EmbeddingIndex:
    """
    Embedding-based matcher using sentence-transformers.

    Encodes recipe texts (name + top 5 ingredients) at construction time.
    Query texts are batch-encoded in match_all() for efficiency.
    Cosine similarity is dot-product of L2-normalised embeddings.
    """

    def __init__(self, index_df: pd.DataFrame) -> None:
        import numpy as np

        valid = (
            index_df
            .dropna(subset=["canonical_term"])
            .loc[lambda df: df["canonical_term"].str.strip() != ""]
            .drop_duplicates(subset=["canonical_term"])
            .reset_index(drop=True)
        )

        self._terms: list[str] = valid["canonical_term"].tolist()
        self._ids: list = valid["recipe_id"].tolist()
        self._names: list[str] = valid["recipe_name"].tolist()

        texts = valid.apply(_recipe_text, axis=1).tolist()

        logger.info(
            "Loading sentence-transformers model: %s", _EMBEDDING_MODEL_NAME
        )
        self._model = _SentenceTransformer(_EMBEDDING_MODEL_NAME)

        logger.info("Encoding %d recipe texts for embedding index...", len(texts))
        self._embeddings = self._model.encode(
            texts,
            batch_size=256,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        logger.info(
            "EmbeddingIndex: %d recipes encoded (dim=%d)",
            len(self._terms), self._embeddings.shape[1],
        )

    def match_all(
        self, queries: list[str]
    ) -> list[tuple[int | None, str | None, str | None, float, str]]:
        """
        Batch-match a list of query canonical_terms against the recipe index.
        Returns one result tuple per query; tuple format mirrors _TermIndex.match().
        """
        import numpy as np

        if not queries:
            return []

        logger.info("Encoding %d external query terms...", len(queries))
        q_embs = self._model.encode(
            queries,
            batch_size=64,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

        # Cosine similarity matrix: (n_queries × n_recipes)
        sims = q_embs @ self._embeddings.T

        results = []
        for i, query in enumerate(queries):
            if not query:
                results.append((None, None, None, 0.0, "empty_query"))
                continue
            best_j = int(np.argmax(sims[i]))
            best_score = round(float(sims[i, best_j]), 4)
            results.append((
                self._ids[best_j],
                self._names[best_j],
                self._terms[best_j],
                best_score,
                "embedding",
            ))
        return results


def _try_build_embedding_index(index_df: pd.DataFrame) -> "_EmbeddingIndex | None":
    """
    Attempt to build an embedding index; return None on any failure
    so the caller can fall back to lexical without crashing.
    """
    if not _SENTENCE_TRANSFORMERS_AVAILABLE:
        return None
    try:
        return _EmbeddingIndex(index_df)
    except Exception as exc:
        logger.warning(
            "Could not build embedding index (%s) — falling back to lexical", exc
        )
        return None


# ─────────────────────────────────────────────────────────────────────────
# Gap classification helpers
# ─────────────────────────────────────────────────────────────────────────


def _match_status(similarity: float) -> str:
    if similarity >= _STRONG_MATCH:
        return "strong_match"
    if similarity >= _WEAK_MATCH:
        return "weak_match"
    return "gap"


def _gap_score(source_score, similarity: float) -> float:
    """source_score * (1 - similarity). Defaults source_score to 1.0 if missing."""
    ss = float(source_score) if pd.notna(source_score) else 1.0
    return round(ss * (1.0 - similarity), 4)


def _gap_reason(
    raw_term: str,
    canonical_term: str,
    best_term: str | None,
    similarity: float,
    status: str,
    method: str,
) -> str:
    if status == "strong_match":
        thr = f">={_STRONG_MATCH:.2f}"
    elif status == "weak_match":
        thr = f">={_WEAK_MATCH:.2f}"
    else:
        thr = f"<{_WEAK_MATCH:.2f}"
    return (
        f"raw={raw_term!r} | canonical={canonical_term!r} | "
        f"best_match={best_term!r} | similarity={similarity:.2f} | "
        f"threshold={thr} | status={status} | method={method}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Dashboard-ready field helpers (Section 11)
# ─────────────────────────────────────────────────────────────────────────

_OPP_HIGH = 0.70
_OPP_MED = 0.40


def _opportunity_label(gap_score: float, match_status: str) -> str | None:
    """Null for strong matches; High / Medium / Low for gaps and weak matches."""
    if match_status == "strong_match":
        return None
    if gap_score >= _OPP_HIGH:
        return "High"
    if gap_score >= _OPP_MED:
        return "Medium"
    return "Low"


def _insight_summary(
    raw_term: str,
    match_status: str,
    similarity: float,
    source_score,
    opportunity: str | None,
    best_name: str | None,
    source: str | None,
) -> str:
    """One-sentence card-ready insight for the BI dashboard."""
    source_label = str(source).replace("_", " ").title() if source else "External"
    sim_str = f"{similarity:.2f}"
    ss = float(source_score) if pd.notna(source_score) else 1.0

    if match_status == "strong_match":
        return (
            f"'{raw_term}' is well-covered on Food.com "
            f"(best match: '{best_name}', similarity={sim_str})."
        )
    if match_status == "weak_match":
        return (
            f"'{raw_term}' has a partial Food.com match "
            f"('{best_name}', similarity={sim_str}); "
            f"{opportunity} opportunity per {source_label}."
        )
    return (
        f"'{raw_term}' has no strong Food.com match (similarity={sim_str}); "
        f"{opportunity} opportunity based on {source_label} signal "
        f"(source score={ss:.2f})."
    )


def _add_dashboard_fields(gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute top_gap_rank, opportunity_label, insight_summary in-place and return.
    top_gap_rank is 1-indexed within gap + weak_match rows, ordered by gap_score desc.
    """
    if gap_df.empty:
        gap_df["top_gap_rank"] = None
        gap_df["opportunity_label"] = None
        gap_df["insight_summary"] = None
        return gap_df

    # top_gap_rank — only for non-strong-match rows
    rankable = gap_df["match_status"].isin({"gap", "weak_match"})
    ranks = (
        gap_df.loc[rankable, "gap_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    gap_df["top_gap_rank"] = None
    gap_df.loc[rankable, "top_gap_rank"] = ranks

    # opportunity_label and insight_summary — vectorised via apply
    def _row_fields(row) -> pd.Series:
        opp = _opportunity_label(
            float(row["gap_score"]) if pd.notna(row["gap_score"]) else 0.0,
            str(row["match_status"]),
        )
        summary = _insight_summary(
            raw_term=str(row.get("raw_term", "")),
            match_status=str(row["match_status"]),
            similarity=float(row["best_foodcom_similarity"]) if pd.notna(row["best_foodcom_similarity"]) else 0.0,
            source_score=row.get("source_score"),
            opportunity=opp,
            best_name=row.get("best_foodcom_recipe_name"),
            source=row.get("source"),
        )
        return pd.Series({"opportunity_label": opp, "insight_summary": summary})

    dashboard_cols = gap_df.apply(_row_fields, axis=1)
    gap_df["opportunity_label"] = dashboard_cols["opportunity_label"]
    gap_df["insight_summary"] = dashboard_cols["insight_summary"]

    return gap_df


# ─────────────────────────────────────────────────────────────────────────
# Main task
# ─────────────────────────────────────────────────────────────────────────


def build_recipe_gap_analysis(**context) -> None:
    """
    Match each external_recipe_terms row against recipe_term_index and
    compute gap scores.  Uses embeddings when sentence-transformers is
    installed; falls back to the lexical baseline otherwise.

    Outputs recipe_gap_analysis.parquet.
    """
    # ── 1. Load inputs ────────────────────────────────────────────────────
    if not EXTERNAL_TERMS_STAGING.is_file():
        logger.warning("external_recipe_terms.parquet not found: %s", EXTERNAL_TERMS_STAGING)
        return
    if not RECIPE_TERM_INDEX_STAGING.is_file():
        logger.warning("recipe_term_index.parquet not found: %s", RECIPE_TERM_INDEX_STAGING)
        return

    ext_df = pd.read_parquet(EXTERNAL_TERMS_STAGING)
    index_df = pd.read_parquet(RECIPE_TERM_INDEX_STAGING)
    logger.info(
        "Loaded %d external terms, %d recipe index rows",
        len(ext_df), len(index_df),
    )

    # ── 2. Build matcher — embeddings preferred, lexical fallback ─────────
    emb_index = _try_build_embedding_index(index_df)
    lex_index = _TermIndex(index_df)

    if emb_index is not None:
        logger.info("Using embedding matcher (sentence-transformers)")
        matching_method_global = "embedding"
        queries = [str(r.get("canonical_term", "")).strip() for _, r in ext_df.iterrows()]
        match_results = emb_index.match_all(queries)
    else:
        logger.info("Using lexical matcher (embedding not available)")
        matching_method_global = "lexical"
        match_results = [
            lex_index.match(str(r.get("canonical_term", "")).strip())
            for _, r in ext_df.iterrows()
        ]

    # ── 3. Build output rows ──────────────────────────────────────────────
    rows: list[dict] = []
    for (_, row), (recipe_id, recipe_name, matched_canon, similarity, method) in zip(
        ext_df.iterrows(), match_results
    ):
        raw_term = str(row.get("raw_term", "")).strip()
        canonical_term = str(row.get("canonical_term", "")).strip()

        status = _match_status(similarity)
        gs = _gap_score(row.get("source_score"), similarity)
        reason = _gap_reason(
            raw_term, canonical_term, matched_canon, similarity, status, method
        )

        rows.append({
            "source": row.get("source"),
            "raw_term": raw_term,
            "canonical_term": canonical_term,
            "source_score": row.get("source_score"),
            "best_foodcom_recipe_id": recipe_id,
            "best_foodcom_recipe_name": recipe_name,
            "best_foodcom_canonical_term": matched_canon,
            "best_foodcom_similarity": similarity,
            "match_status": status,
            "gap_score": gs,
            "gap_reason": reason,
            "matching_method": matching_method_global,
            "fetched_date": row.get("fetched_date"),
        })

    gap_df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=_GAP_COLS)

    # ── Section 11: dashboard-ready fields ───────────────────────────────
    gap_df = _add_dashboard_fields(gap_df)

    # Enforce exact schema
    for col in _GAP_COLS:
        if col not in gap_df.columns:
            gap_df[col] = None
    gap_df = gap_df[_GAP_COLS]

    # ── 4. Write parquet ──────────────────────────────────────────────────
    RECIPE_GAP_ANALYSIS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(gap_df, RECIPE_GAP_ANALYSIS_STAGING)
    logger.info(
        "Recipe gap analysis written: %s (%d rows, matching_method=%s)",
        RECIPE_GAP_ANALYSIS_STAGING, len(gap_df), matching_method_global,
    )

    # ── 5. Validation ─────────────────────────────────────────────────────
    _validate_gap_analysis(gap_df, lex_index)

    # ── 6. End-to-end pipeline summary ────────────────────────────────────
    _log_pipeline_summary(gap_df)

    if context.get("ti"):
        strong = int((gap_df["match_status"] == "strong_match").sum())
        weak = int((gap_df["match_status"] == "weak_match").sum())
        gaps = int((gap_df["match_status"] == "gap").sum())
        context["ti"].xcom_push(key="gap_analysis_rows", value=len(gap_df))
        context["ti"].xcom_push(key="strong_matches", value=strong)
        context["ti"].xcom_push(key="weak_matches", value=weak)
        context["ti"].xcom_push(key="gaps", value=gaps)
        context["ti"].xcom_push(key="matching_method", value=matching_method_global)


# ─────────────────────────────────────────────────────────────────────────
# End-to-end pipeline summary (Section 12)
# ─────────────────────────────────────────────────────────────────────────

_TAGS_STAGING = STAGING_DIR / "recipe_tags.parquet"
_CLUSTERS_STAGING = STAGING_DIR / "recipe_term_clusters.parquet"


def _log_pipeline_summary(gap_df: pd.DataFrame) -> None:
    """
    Log a single consolidated end-to-end summary after gap analysis completes.
    Reads sibling staging parquets so all key metrics appear together.
    """
    SEP = "=" * 64
    logger.info(SEP)
    logger.info("PIPELINE SUMMARY  —  Food.com BI gap analysis")
    logger.info(SEP)

    # ── Food.com recipe term index ────────────────────────────────────────
    if RECIPE_TERM_INDEX_STAGING.is_file():
        idx = pd.read_parquet(RECIPE_TERM_INDEX_STAGING)
        tagged_mask = idx["tags"].notna() & (idx["tags"].astype(str).str.strip() != "")
        tagged_n = int(tagged_mask.sum())
        logger.info(
            "Food.com recipes indexed   : %d  (tagged=%d, %.1f%%)",
            len(idx), tagged_n, 100.0 * tagged_n / max(len(idx), 1),
        )
    else:
        logger.warning("recipe_term_index.parquet  : NOT FOUND")

    # ── Recipe tags ───────────────────────────────────────────────────────
    if _TAGS_STAGING.is_file():
        tags = pd.read_parquet(_TAGS_STAGING)
        regex_n = int((tags["tagging_method"] == "regex").sum()) if "tagging_method" in tags.columns else len(tags)
        llm_n = int((tags["tagging_method"] == "llm").sum()) if "tagging_method" in tags.columns else 0
        logger.info(
            "Recipe tags generated      : %d  (regex=%d, llm=%d)",
            len(tags), regex_n, llm_n,
        )
    else:
        logger.warning("recipe_tags.parquet        : NOT FOUND")

    # ── External terms by source ──────────────────────────────────────────
    if EXTERNAL_TERMS_STAGING.is_file():
        ext = pd.read_parquet(EXTERNAL_TERMS_STAGING)
        by_src = ext.groupby("source").size().to_dict()
        ai_n = by_src.get("google_ai", 0)
        tr_n = by_src.get("google_trends", 0)
        logger.info("External terms — Google AI : %d", ai_n)
        logger.info("External terms — Trends    : %d", tr_n)
    else:
        logger.warning("external_recipe_terms.parquet : NOT FOUND")

    # ── Gap analysis breakdown ─────────────────────────────────────────────
    if not gap_df.empty:
        total = len(gap_df)
        strong = int((gap_df["match_status"] == "strong_match").sum())
        weak = int((gap_df["match_status"] == "weak_match").sum())
        gaps = int((gap_df["match_status"] == "gap").sum())
        logger.info(
            "Gap analysis rows          : %d  "
            "(strong=%d %.0f%%  weak=%d %.0f%%  gap=%d %.0f%%)",
            total,
            strong, 100.0 * strong / total,
            weak, 100.0 * weak / total,
            gaps, 100.0 * gaps / total,
        )

        # Top 10 gaps with dashboard insight
        logger.info("--- top 10 recipe opportunities (gap_score desc) ---")
        top_gaps = (
            gap_df[gap_df["match_status"] == "gap"]
            .nlargest(10, "gap_score")
            .reset_index(drop=True)
        )
        for i, row in top_gaps.iterrows():
            label = str(row.get("opportunity_label") or "?")
            insight = str(row.get("insight_summary") or "")
            logger.info(
                "  %2d. [%-6s]  %-35s  gap=%.3f",
                i + 1,
                label,
                str(row["raw_term"])[:35],
                float(row["gap_score"]) if pd.notna(row["gap_score"]) else 0.0,
            )
            if insight:
                logger.info("       %s", insight[:120])
    else:
        logger.info("Gap analysis rows          : 0 (empty)")

    # ── Recipe term clusters ──────────────────────────────────────────────
    if _CLUSTERS_STAGING.is_file():
        clusters = pd.read_parquet(_CLUSTERS_STAGING)
        high_opp = 0
        if "avg_gap_score" in clusters.columns:
            high_opp = int((clusters["avg_gap_score"].fillna(0) >= 0.40).sum())
        logger.info(
            "Recipe term clusters       : %d  (%d with avg_gap_score≥0.40)",
            len(clusters), high_opp,
        )
    else:
        logger.info("recipe_term_clusters.parquet : not yet written (runs after this task)")

    logger.info(SEP)


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────

_MANUAL_CASES = [
    "banana bread recipe",
    "garlic chicken",
    "birria tacos",
    "feta pasta",
]


def _validate_gap_analysis(gap_df: pd.DataFrame, lex_index: "_TermIndex") -> None:
    logger.info("=== recipe_gap_analysis validation ===")
    logger.info("Total rows: %d", len(gap_df))

    for status in ("strong_match", "weak_match", "gap"):
        n = int((gap_df["match_status"] == status).sum())
        logger.info("  %-15s  %d rows", status, n)

    # Confirm matching_method column present and consistent
    if "matching_method" in gap_df.columns:
        methods = gap_df["matching_method"].value_counts().to_dict()
        logger.info("  matching_method breakdown: %s", methods)

    # ── Manual test cases (always run via lexical for reproducibility) ────
    logger.info("--- manual test cases (lexical) ---")
    for raw_term in _MANUAL_CASES:
        canonical_term = canonicalize_recipe_term(raw_term)
        recipe_id, recipe_name, matched_canon, similarity, method = (
            lex_index.match(canonical_term)
        )
        status = _match_status(similarity)
        reason = _gap_reason(
            raw_term, canonical_term, matched_canon, similarity, status, method
        )
        logger.info("  %s", reason)

    # ── Top 10 gaps ───────────────────────────────────────────────────────
    logger.info("--- top 10 gaps by gap_score ---")
    top_gaps = (
        gap_df[gap_df["match_status"] == "gap"]
        .nlargest(10, "gap_score")
    )
    for _, row in top_gaps.iterrows():
        logger.info(
            "  [%-13s]  %-35s  source_score=%.3f  gap_score=%.3f",
            row["source"],
            str(row["raw_term"])[:35],
            float(row["source_score"]) if pd.notna(row["source_score"]) else 0.0,
            float(row["gap_score"]) if pd.notna(row["gap_score"]) else 0.0,
        )

    logger.info("=== end validation ===")
