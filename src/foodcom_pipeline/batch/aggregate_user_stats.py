"""
aggregate_user_stats.py
-----------------------
Feature engineering step for the Food.com batch pipeline.

Computes per-user aggregate statistics from the sentiment-scored interactions.
This runs as a dedicated DAG step between sentiment.py and cluster.py so that:
  - The aggregation can be retried independently of clustering
  - The user stats parquet is available to both cluster.py and load.py
    (dim_user needs these features regardless of clustering)
  - The step is cheap to rerun compared to DistilBERT inference

Output columns (one row per user_id):
  avg_rating_given      : mean star rating across all reviews
  review_count          : total number of reviews submitted
  recipe_diversity      : number of unique recipes reviewed
  avg_sentiment_score   : mean DistilBERT sentiment score (nulls excluded)
  avg_rating_gap        : mean rating_sentiment_gap
  std_rating_given      : std of star ratings (consistency signal)
  pct_positive_sentiment: fraction of reviews with sentiment_score > 0
  first_review_date     : earliest review date
  last_review_date      : most recent review date
  active_days           : days between first and last review
"""

import logging

import pandas as pd
from foodcom_pipeline.batch.extract import STAGING_DIR, atomic_parquet
from foodcom_pipeline.batch.sentiment import load_sentiment_interactions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_STATS_STAGING = STAGING_DIR / 'user_stats.parquet'


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_aggregate_user_stats(**context) -> None:
    """
    Aggregates interaction-level data to one row per user and stages
    the result as a parquet file for cluster.py and load.py.
    """
    logger.info('Starting user stats aggregation step.')

    interactions = load_sentiment_interactions()

    user_stats = _compute_user_stats(interactions)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(user_stats, USER_STATS_STAGING)

    logger.info(
        f'User stats aggregation complete. '
        f'{len(user_stats):,} users staged to {USER_STATS_STAGING}'
    )

    context['ti'].xcom_push(key='n_users', value=len(user_stats))


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _compute_user_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all per-user features in a single groupby pass where possible,
    then enriches with secondary derived columns.
    """
    n_interactions = len(df)
    n_users = df['user_id'].nunique()
    logger.info(
        f'Aggregating {n_interactions:,} interactions across {n_users:,} users...'
    )

    # Primary aggregation
    agg = (
        df.groupby('user_id')
        .agg(
            avg_rating_given=('rating', 'mean'),
            std_rating_given=('rating', 'std'),
            review_count=('rating', 'count'),
            recipe_diversity=('recipe_id', 'nunique'),
            avg_sentiment_score=('sentiment_score', 'mean'),
            avg_rating_gap=('rating_sentiment_gap', 'mean'),
            first_review_date=('date', 'min'),
            last_review_date=('date', 'max'),
        )
        .reset_index()
    )

    # Derived columns
    agg = _compute_pct_positive_sentiment(agg, df)
    agg = _compute_active_days(agg)
    agg = _fill_nulls(agg)
    agg = _round_floats(agg)

    _log_aggregation_summary(agg)
    return agg


def _compute_pct_positive_sentiment(
    agg: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    """
    Fraction of a user's reviews with sentiment_score > 0.
    Computed separately since it requires a conditional count.
    """
    positive = (
        interactions[interactions['sentiment_score'] > 0]
        .groupby('user_id')
        .size()
        .reset_index(name='positive_sentiment_count')
    )
    agg = agg.merge(positive, on='user_id', how='left')
    agg['positive_sentiment_count'] = agg['positive_sentiment_count'].fillna(0)
    agg['pct_positive_sentiment'] = (
        agg['positive_sentiment_count'] / agg['review_count']
    ).round(4)
    agg = agg.drop(columns=['positive_sentiment_count'])
    return agg


def _compute_active_days(agg: pd.DataFrame) -> pd.DataFrame:
    """Days between a user's first and last review — proxy for long-term engagement."""
    agg['active_days'] = (
        (agg['last_review_date'] - agg['first_review_date'])
        .dt.days.fillna(0)
        .astype(int)
    )
    return agg


def _fill_nulls(agg: pd.DataFrame) -> pd.DataFrame:
    """
    Handle nulls introduced by aggregation:
      - std_rating_given is NaN for users with only 1 review → fill with 0
      - avg_sentiment_score is NaN if all reviews had no text → fill with 0 (neutral)
      - avg_rating_gap follows the same logic as sentiment
    """
    agg['std_rating_given'] = agg['std_rating_given'].fillna(0.0)
    agg['avg_sentiment_score'] = agg['avg_sentiment_score'].fillna(0.0)
    agg['avg_rating_gap'] = agg['avg_rating_gap'].fillna(0.0)
    return agg


def _round_floats(agg: pd.DataFrame) -> pd.DataFrame:
    """Round float columns to 4 decimal places for cleaner storage."""
    float_cols = agg.select_dtypes(include='float').columns
    agg[float_cols] = agg[float_cols].round(4)
    return agg


def _log_aggregation_summary(agg: pd.DataFrame) -> None:
    logger.info('User stats summary:')
    logger.info(f'  Total users          : {len(agg):,}')
    logger.info(
        f"  avg_rating_given     : mean={agg['avg_rating_given'].mean():.3f}, "
        f"std={agg['avg_rating_given'].std():.3f}"
    )
    logger.info(
        f"  review_count         : mean={agg['review_count'].mean():.1f}, "
        f"median={agg['review_count'].median():.0f}, "
        f"max={agg['review_count'].max()}"
    )
    logger.info(f"  recipe_diversity     : mean={agg['recipe_diversity'].mean():.1f}")
    logger.info(
        f"  avg_sentiment_score  : mean={agg['avg_sentiment_score'].mean():.3f}"
    )
    logger.info(f"  avg_rating_gap       : mean={agg['avg_rating_gap'].mean():.3f}")
    logger.info(
        f"  pct_positive_sent    : " f"mean={agg['pct_positive_sentiment'].mean():.3f}"
    )
    logger.info(
        f"  active_days          : mean={agg['active_days'].mean():.0f}, "
        f"max={agg['active_days'].max()}"
    )


# ---------------------------------------------------------------------------
# Utility: load user stats (used by cluster.py and load.py)
# ---------------------------------------------------------------------------


def load_user_stats() -> pd.DataFrame:
    return pd.read_parquet(USER_STATS_STAGING)
