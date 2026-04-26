"""
compute_gap.py
--------------
Compares consumer search demand (ai_mode_term_scores) against Food.com recipe
supply (recipe_tags) to identify categories where demand outpaces content.

Demand proxy: ai_mode_term_scores.parquet → normalised_score (surfaced as combined_score).
Note: market_signals was removed; ai_mode_term_scores is the canonical demand signal.

Output columns: term, combined_score, recipe_count, gap_score, opportunity_tier, computed_date
"""

import logging
import os
import re
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, text

from foodcom_pipeline.batch.extract import (
    STAGING_DIR,
    atomic_parquet,
)

logger = logging.getLogger(__name__)

GAP_ANALYSIS_STAGING = STAGING_DIR / "gap_analysis.parquet"
RECIPE_TAGS_STAGING  = STAGING_DIR / "recipe_tags.parquet"
AI_MODE_TERM_SCORES_STAGING = STAGING_DIR / "ai_mode_term_scores.parquet"
POSTGRES_CONN = os.getenv(
    "FOODCOM_POSTGRES_CONN", "postgresql://user:password@postgres:5432/foodcom"
)

_RECIPE_SUFFIX = re.compile(r"\brecipes?\s*$", re.IGNORECASE)


def _clean_term(term: str) -> str:
    term = str(term).lower().strip()
    term = _RECIPE_SUFFIX.sub("", term).strip()
    return term


def compute_gap_analysis(**context) -> None:
    """
    Airflow task: merge demand signal with recipe supply and write gap_analysis.
    """
    # ── 1. Recipe supply ──────────────────────────────────────────────────
    if not RECIPE_TAGS_STAGING.is_file():
        logger.warning("recipe_tags.parquet not found — skipping gap analysis.")
        return

    recipe_tags = pd.read_parquet(RECIPE_TAGS_STAGING)
    recipe_supply = (
        recipe_tags.groupby("tag")
        .size()
        .reset_index(name="recipe_count")
    )

    # ── 2. Demand signal ──────────────────────────────────────────────────
    if not AI_MODE_TERM_SCORES_STAGING.is_file():
        logger.warning("ai_mode_term_scores.parquet not found — skipping gap analysis.")
        return

    raw_demand = pd.read_parquet(AI_MODE_TERM_SCORES_STAGING)

    # Clean terms: lowercase, strip whitespace, strip trailing "recipe"/"recipes"
    raw_demand = raw_demand.copy()
    raw_demand["term"] = raw_demand["term"].apply(_clean_term)
    raw_demand = raw_demand[raw_demand["term"] != ""]

    # Deduplicate: keep highest normalised_score per cleaned term
    demand = (
        raw_demand.sort_values("normalised_score", ascending=False)
        .drop_duplicates(subset=["term"])
        .rename(columns={"normalised_score": "combined_score"})
        [["term", "combined_score"]]
        .reset_index(drop=True)
    )

    # ── 3. Outer merge on term ↔ tag ───────────────────────────────────────
    gap_df = demand.merge(
        recipe_supply,
        left_on="term",
        right_on="tag",
        how="outer",
    )

    # Unify the term column (outer join leaves NaNs for supply-only rows)
    gap_df["term"] = gap_df["term"].fillna(gap_df["tag"])
    gap_df = gap_df.drop(columns=["tag"])

    # ── 4. Gap score ──────────────────────────────────────────────────────
    rc = gap_df["recipe_count"].astype(float)
    rc_mean = rc.mean(skipna=True)
    rc_std  = rc.std(skipna=True)

    if pd.isna(rc_std) or rc_std == 0:
        norm_rc = rc.fillna(0) * 0.0
    else:
        norm_rc = ((rc - rc_mean) / rc_std).round(6)

    has_demand = gap_df["combined_score"].notna()
    has_supply = gap_df["recipe_count"].notna()

    gap_df["gap_score"] = 0.0
    gap_df.loc[has_demand & has_supply,  "gap_score"] = (
        gap_df.loc[has_demand & has_supply, "combined_score"]
        - norm_rc[has_demand & has_supply]
    )
    gap_df.loc[~has_demand & has_supply, "gap_score"] = -norm_rc[~has_demand & has_supply]
    gap_df.loc[has_demand & ~has_supply, "gap_score"] = gap_df.loc[has_demand & ~has_supply, "combined_score"]

    gap_df["gap_score"] = gap_df["gap_score"].round(6)

    # ── 5. Opportunity tier ───────────────────────────────────────────────
    def _tier(score: float) -> str:
        if score > 1.0:
            return "High"
        if score > 0.3:
            return "Medium"
        return "Low"

    gap_df["opportunity_tier"] = gap_df["gap_score"].apply(_tier)
    gap_df["computed_date"]    = date.today()

    gap_df = gap_df[
        ["term", "combined_score", "recipe_count", "gap_score", "opportunity_tier", "computed_date"]
    ].sort_values("gap_score", ascending=False).reset_index(drop=True)

    logger.info("Gap analysis produced %d rows", len(gap_df))

    # ── 6. Stage + load ───────────────────────────────────────────────────
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(gap_df, GAP_ANALYSIS_STAGING)
    logger.info("Gap analysis staged to %s", GAP_ANALYSIS_STAGING)

    _write_to_db(gap_df)

    context["ti"].xcom_push(key="gap_rows", value=len(gap_df))
    context["ti"].xcom_push(
        key="high_opportunity_count",
        value=int((gap_df["opportunity_tier"] == "High").sum()),
    )


def _write_to_db(gap_df: pd.DataFrame) -> None:
    try:
        engine = create_engine(POSTGRES_CONN)

        with engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS gap_analysis (
                    term             TEXT             NOT NULL,
                    combined_score   DOUBLE PRECISION,
                    recipe_count     INTEGER,
                    gap_score        DOUBLE PRECISION NOT NULL,
                    opportunity_tier TEXT             NOT NULL,
                    computed_date    DATE             NOT NULL,
                    PRIMARY KEY (term, computed_date)
                )
            """))
            # Backward-compatible type hardening for pre-existing tables.
            # Older schemas may use narrower integer types (e.g., SMALLINT)
            # that overflow on large recipe counts.
            conn.execute(text("""
                ALTER TABLE gap_analysis
                ALTER COLUMN combined_score TYPE DOUBLE PRECISION
                USING combined_score::DOUBLE PRECISION
            """))
            conn.execute(text("""
                ALTER TABLE gap_analysis
                ALTER COLUMN recipe_count TYPE BIGINT
                USING recipe_count::BIGINT
            """))
            conn.execute(text("""
                ALTER TABLE gap_analysis
                ALTER COLUMN gap_score TYPE DOUBLE PRECISION
                USING gap_score::DOUBLE PRECISION
            """))

        safe_df = gap_df.copy()
        safe_df["combined_score"] = pd.to_numeric(
            safe_df["combined_score"], errors="coerce"
        )
        safe_df["recipe_count"] = pd.to_numeric(
            safe_df["recipe_count"], errors="coerce"
        ).round()
        safe_df["gap_score"] = pd.to_numeric(safe_df["gap_score"], errors="coerce")
        safe_df["recipe_count"] = safe_df["recipe_count"].astype("Int64")

        records = safe_df.where(pd.notna(safe_df), None).to_dict(orient="records")
        upsert_sql = """
            INSERT INTO gap_analysis
                (term, combined_score, recipe_count, gap_score, opportunity_tier, computed_date)
            VALUES
                (:term, :combined_score, :recipe_count, :gap_score, :opportunity_tier, :computed_date)
            ON CONFLICT (term, computed_date) DO UPDATE SET
                combined_score   = EXCLUDED.combined_score,
                recipe_count     = EXCLUDED.recipe_count,
                gap_score        = EXCLUDED.gap_score,
                opportunity_tier = EXCLUDED.opportunity_tier
        """

        batch_size = 5000
        for i in range(0, len(records), batch_size):
            with engine.begin() as conn:
                conn.execute(text(upsert_sql), records[i : i + batch_size])

        logger.info("Wrote %d gap analysis rows to PostgreSQL", len(records))

    except Exception as exc:
        logger.error("Failed to write gap_analysis to PostgreSQL: %s", exc, exc_info=True)
