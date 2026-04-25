"""
cluster_terms.py
----------------
Cluster external recipe terms (Google AI + Google Trends) into semantic groups.

Three-stage strategy:
  1. Group identical canonical_terms (exact dedup)
  2. Fuzzy-merge canonical_terms above a similarity threshold
     (catches "banana bread" / "banana bread recipe" after canonicalization)
  3. If sentence-transformers is installed, merge embedding-similar clusters
     whose cosine similarity exceeds _EMBED_MERGE_THRESHOLD

Output: recipe_term_clusters.parquet
"""

import logging
import re
from difflib import SequenceMatcher

import pandas as pd

from foodcom_pipeline.batch.extract import STAGING_DIR, atomic_parquet

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz as _rf_fuzz
    _RAPIDFUZZ_AVAILABLE = True
except ImportError:
    _RAPIDFUZZ_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
    _SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    _SENTENCE_TRANSFORMERS_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────
# Paths and schema
# ─────────────────────────────────────────────────────────────────────────

EXTERNAL_TERMS_STAGING = STAGING_DIR / "external_recipe_terms.parquet"
RECIPE_GAP_ANALYSIS_STAGING = STAGING_DIR / "recipe_gap_analysis.parquet"
RECIPE_TERM_CLUSTERS_STAGING = STAGING_DIR / "recipe_term_clusters.parquet"

_CLUSTER_COLS = [
    "cluster_id",
    "cluster_label",
    "representative_term",
    "source_terms",
    "sources_present",
    "dominant_tags",
    "avg_gap_score",
    "max_gap_score",
    "best_foodcom_match",
    "foodcom_coverage_count",
    "explanation",
]

# ─────────────────────────────────────────────────────────────────────────
# Similarity helpers
# ─────────────────────────────────────────────────────────────────────────

_FUZZY_MERGE_THRESHOLD = 0.85   # canonical_terms above this are merged
_EMBED_MERGE_THRESHOLD = 0.92   # embedding cosine similarity merge threshold

_EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def _term_similarity(a: str, b: str) -> float:
    """String similarity in [0, 1]: token_sort_ratio (RapidFuzz) or SequenceMatcher."""
    if _RAPIDFUZZ_AVAILABLE:
        return _rf_fuzz.token_sort_ratio(a, b) / 100.0
    return SequenceMatcher(None, a, b).ratio()


# ─────────────────────────────────────────────────────────────────────────
# Union-Find for merging clusters
# ─────────────────────────────────────────────────────────────────────────


class _UnionFind:
    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def groups(self) -> dict[int, list[int]]:
        from collections import defaultdict
        groups: dict[int, list[int]] = defaultdict(list)
        for i in range(len(self._parent)):
            groups[self.find(i)].append(i)
        return dict(groups)


# ─────────────────────────────────────────────────────────────────────────
# Clustering logic
# ─────────────────────────────────────────────────────────────────────────


def _fuzzy_cluster(terms: list[str]) -> dict[int, list[int]]:
    """
    Merge terms whose pairwise similarity exceeds _FUZZY_MERGE_THRESHOLD.
    Returns a dict of {root_index: [member_indices]}.
    O(N²) — acceptable for the small number of unique canonical terms
    in external_recipe_terms (typically < 500).
    """
    uf = _UnionFind(len(terms))
    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if _term_similarity(terms[i], terms[j]) >= _FUZZY_MERGE_THRESHOLD:
                uf.union(i, j)
    return uf.groups()


def _embed_cluster(
    terms: list[str], initial_groups: dict[int, list[int]]
) -> dict[int, list[int]]:
    """
    Further merge clusters whose representative embeddings exceed
    _EMBED_MERGE_THRESHOLD.  Only called when sentence-transformers is available.
    """
    import numpy as np

    # One representative per group (index of group root in terms list)
    roots = sorted(initial_groups.keys())
    rep_texts = [terms[r] for r in roots]

    model = _SentenceTransformer(_EMBEDDING_MODEL_NAME)
    embs = model.encode(rep_texts, normalize_embeddings=True, show_progress_bar=False)
    sims = embs @ embs.T

    root_uf = _UnionFind(len(roots))
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            if sims[i, j] >= _EMBED_MERGE_THRESHOLD:
                root_uf.union(i, j)

    # Rebuild final groups: merge original groups whose roots were merged
    merged_roots = root_uf.groups()
    final_groups: dict[int, list[int]] = {}
    for new_root_pos, old_root_positions in merged_roots.items():
        members: list[int] = []
        for pos in old_root_positions:
            members.extend(initial_groups[roots[pos]])
        final_groups[roots[old_root_positions[0]]] = members

    return final_groups


# ─────────────────────────────────────────────────────────────────────────
# Row assembly helpers
# ─────────────────────────────────────────────────────────────────────────


def _representative(terms: list[str]) -> str:
    """Choose the shortest term as representative (fewest tokens = most general)."""
    return min(terms, key=lambda t: (len(t.split()), len(t)))


def _cluster_label(rep: str) -> str:
    """Capitalize the representative term as the cluster label."""
    return rep.title()


def _build_cluster_row(
    cluster_id: int,
    member_rows: pd.DataFrame,
    gap_rows: pd.DataFrame,
) -> dict:
    """Build a single cluster output row from member external terms + gap data."""
    canonical_terms = member_rows["canonical_term"].dropna().unique().tolist()
    raw_terms = member_rows["raw_term"].dropna().unique().tolist()
    sources = sorted(member_rows["source"].dropna().unique().tolist())

    rep = _representative(canonical_terms)
    label = _cluster_label(rep)

    # Dominant tags — from tags column if available
    all_tags: list[str] = []
    if "tags" in member_rows.columns:
        for tags_str in member_rows["tags"].dropna():
            all_tags.extend(str(tags_str).split("|"))
    from collections import Counter
    tag_counts = Counter(t for t in all_tags if t)
    dominant_tags = "|".join(t for t, _ in tag_counts.most_common(5))

    # Gap stats from gap analysis rows
    cluster_gaps = gap_rows[
        gap_rows["canonical_term"].isin(canonical_terms)
    ]
    avg_gap = float(cluster_gaps["gap_score"].mean()) if not cluster_gaps.empty else None
    max_gap = float(cluster_gaps["gap_score"].max()) if not cluster_gaps.empty else None

    # Best Food.com match — pick the one with highest similarity
    best_match = None
    coverage = 0
    if not cluster_gaps.empty:
        strong = cluster_gaps[cluster_gaps["match_status"] == "strong_match"]
        coverage = len(strong)
        if not cluster_gaps.empty:
            best_row = cluster_gaps.loc[cluster_gaps["best_foodcom_similarity"].idxmax()]
            best_match = best_row.get("best_foodcom_recipe_name")

    # Explanation
    sim_val = float(cluster_gaps["best_foodcom_similarity"].max()) if not cluster_gaps.empty else 0.0
    status_str = cluster_gaps["match_status"].mode().iloc[0] if not cluster_gaps.empty else "unknown"
    explanation = (
        f"Cluster '{label}' groups {len(raw_terms)} term(s) from {', '.join(sources)}. "
        f"Best Food.com match: {best_match!r} (similarity={sim_val:.2f}). "
        f"Status: {status_str}. "
        f"Tags: {dominant_tags or 'none'}."
    )

    return {
        "cluster_id": cluster_id,
        "cluster_label": label,
        "representative_term": rep,
        "source_terms": "|".join(raw_terms),
        "sources_present": "|".join(sources),
        "dominant_tags": dominant_tags or None,
        "avg_gap_score": round(avg_gap, 4) if avg_gap is not None else None,
        "max_gap_score": round(max_gap, 4) if max_gap is not None else None,
        "best_foodcom_match": best_match,
        "foodcom_coverage_count": coverage,
        "explanation": explanation,
    }


# ─────────────────────────────────────────────────────────────────────────
# Main task
# ─────────────────────────────────────────────────────────────────────────


def build_recipe_term_clusters(**context) -> None:
    """
    Cluster external_recipe_terms.parquet into semantic groups and write
    recipe_term_clusters.parquet.
    """
    logger.info("Building recipe term clusters...")

    # ── 1. Load inputs ────────────────────────────────────────────────────
    if not EXTERNAL_TERMS_STAGING.is_file():
        logger.warning("external_recipe_terms.parquet not found: %s", EXTERNAL_TERMS_STAGING)
        return

    ext_df = pd.read_parquet(EXTERNAL_TERMS_STAGING)
    logger.info("Loaded %d external terms", len(ext_df))

    gap_df = pd.DataFrame()
    if RECIPE_GAP_ANALYSIS_STAGING.is_file():
        gap_df = pd.read_parquet(RECIPE_GAP_ANALYSIS_STAGING)
        logger.info("Loaded %d gap analysis rows for cluster enrichment", len(gap_df))
    else:
        logger.info("Gap analysis not found — cluster gap stats will be empty")

    # ── 2. Deduplicate by canonical_term ──────────────────────────────────
    # Keep all rows but group by canonical_term; we cluster the unique terms
    unique_terms = (
        ext_df["canonical_term"]
        .dropna()
        .loc[lambda s: s.str.strip() != ""]
        .unique()
        .tolist()
    )
    logger.info("%d unique canonical terms to cluster", len(unique_terms))

    # ── 3. Fuzzy merge ────────────────────────────────────────────────────
    groups = _fuzzy_cluster(unique_terms)
    logger.info("Fuzzy clustering: %d groups from %d terms", len(groups), len(unique_terms))

    # ── 4. Embedding merge (optional) ────────────────────────────────────
    if _SENTENCE_TRANSFORMERS_AVAILABLE and len(groups) > 1:
        try:
            groups = _embed_cluster(unique_terms, groups)
            logger.info("Embedding merge: %d final groups", len(groups))
        except Exception as exc:
            logger.warning("Embedding cluster merge failed (%s) — keeping fuzzy groups", exc)

    # ── 5. Build output rows ──────────────────────────────────────────────
    cluster_rows: list[dict] = []
    for cluster_idx, (root, member_indices) in enumerate(
        sorted(groups.items(), key=lambda kv: -len(kv[1]))
    ):
        member_canonical = [unique_terms[i] for i in member_indices]
        member_ext = ext_df[ext_df["canonical_term"].isin(member_canonical)]
        row = _build_cluster_row(cluster_idx, member_ext, gap_df)
        cluster_rows.append(row)

    if not cluster_rows:
        cluster_df = pd.DataFrame(columns=_CLUSTER_COLS)
    else:
        cluster_df = pd.DataFrame(cluster_rows)

    # Enforce schema
    for col in _CLUSTER_COLS:
        if col not in cluster_df.columns:
            cluster_df[col] = None
    cluster_df = cluster_df[_CLUSTER_COLS]

    # ── 6. Write ──────────────────────────────────────────────────────────
    RECIPE_TERM_CLUSTERS_STAGING.parent.mkdir(parents=True, exist_ok=True)
    atomic_parquet(cluster_df, RECIPE_TERM_CLUSTERS_STAGING)
    logger.info(
        "Recipe term clusters written: %s (%d clusters)", RECIPE_TERM_CLUSTERS_STAGING, len(cluster_df)
    )

    # ── 7. Validation ─────────────────────────────────────────────────────
    _validate_clusters(cluster_df)

    if context.get("ti"):
        context["ti"].xcom_push(key="cluster_count", value=len(cluster_df))


# ─────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────

_GROUP_CASES = [
    ("banana bread recipe", "best banana bread recipe", "healthy banana bread"),
    ("garlic chicken", "creamy garlic chicken", "one pot garlic chicken"),
]


def _validate_clusters(cluster_df: pd.DataFrame) -> None:
    logger.info("=== recipe_term_clusters validation ===")
    logger.info("Total clusters: %d", len(cluster_df))

    # Check grouping test cases
    logger.info("--- grouping test cases ---")
    for case_terms in _GROUP_CASES:
        # Find which cluster each term landed in (by checking source_terms)
        cluster_ids_seen: set[int] = set()
        for term in case_terms:
            hits = cluster_df[
                cluster_df["source_terms"].str.contains(
                    re.escape(term), case=False, na=False
                )
            ]
            if not hits.empty:
                cluster_ids_seen.update(hits["cluster_id"].tolist())

        if len(cluster_ids_seen) == 1:
            status = "PASS"
        elif len(cluster_ids_seen) == 0:
            status = "N/A (terms not in data)"
        else:
            status = f"SPLIT across {len(cluster_ids_seen)} clusters"

        logger.info(
            "  [%s]  %s",
            status,
            " / ".join(f"'{t}'" for t in case_terms),
        )

    # Top 10 clusters by max_gap_score
    logger.info("--- top 10 clusters by max_gap_score ---")
    top = cluster_df.dropna(subset=["max_gap_score"]).nlargest(10, "max_gap_score")
    for _, row in top.iterrows():
        logger.info(
            "  cluster_id=%-4s  %-30s  max_gap=%.3f  terms=%s",
            row["cluster_id"],
            str(row["cluster_label"])[:30],
            float(row["max_gap_score"]),
            str(row["source_terms"])[:50],
        )

    logger.info("=== end validation ===")
