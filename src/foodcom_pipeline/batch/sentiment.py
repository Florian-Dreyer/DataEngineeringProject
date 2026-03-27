"""
sentiment.py
------------
Sentiment analysis step for the Food.com batch pipeline.

Responsibilities:
  1. Score interactions with VADER (fast CPU-friendly sentiment)
  2. Compute rating_sentiment_gap

Sentiment score convention throughout: [-1, +1]
  -1 = maximally negative
   0 = neutral
  +1 = maximally positive

VADER outputs a compound score already in [-1, +1].
"""

import logging

import numpy as np
import pandas as pd
from foodcom_pipeline.batch.clean import load_cleaned_interactions
from foodcom_pipeline.batch.extract import STAGING_DIR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SENTIMENT_STAGING = STAGING_DIR / 'interactions_sentiment.parquet'


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_sentiment(**context) -> None:
    """
    Main sentiment entry point:
      1. Score interactions with VADER
      2. Compute rating_sentiment_gap
    """
    logger.info('Starting sentiment step.')

    df = load_cleaned_interactions()

    scoreable = df['review'].notna() & (df['review'].str.strip() != '')
    unscoreable_count = (~scoreable).sum()
    if unscoreable_count > 0:
        logger.warning(
            f'{unscoreable_count} interactions have no review text — '
            f'sentiment_score will be null for these rows.'
        )

    df = _score_vader(df, scoreable)
    df = _compute_rating_sentiment_gap(df)

    df.to_parquet(SENTIMENT_STAGING, index=False)
    logger.info(f'Sentiment step complete. Staged to {SENTIMENT_STAGING}')

    scored = df['sentiment_score'].notna().sum()
    context['ti'].xcom_push(key='sentiment_scored_count', value=int(scored))
    context['ti'].xcom_push(
        key='sentiment_unscored_count', value=int(unscoreable_count)
    )


# ---------------------------------------------------------------------------
# VADER scoring
# ---------------------------------------------------------------------------


def _score_vader(df: pd.DataFrame, scoreable_mask: pd.Series) -> pd.DataFrame:
    """
    Score all scoreable reviews using VADER compound polarity.
    Returns df with a new `sentiment_score` column in [-1, +1].
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
    reviews = df.loc[scoreable_mask, 'review'].astype(str).tolist()
    logger.info(f'Scoring {len(reviews):,} reviews with VADER...')

    scores = [analyzer.polarity_scores(text)['compound'] for text in reviews]

    df['sentiment_score'] = np.nan
    df.loc[scoreable_mask, 'sentiment_score'] = scores

    mean = df['sentiment_score'].mean()
    std = df['sentiment_score'].std()
    logger.info(f'VADER scoring complete. Mean score: {mean:.3f}, Std: {std:.3f}')
    return df


# ---------------------------------------------------------------------------
# Rating-sentiment gap
# ---------------------------------------------------------------------------


def _compute_rating_sentiment_gap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute rating_sentiment_gap = normalized_rating - sentiment_score

    Ratings are normalized from [1, 5] to [-1, +1] to match the sentiment
    score scale:  normalized = (rating - 3) / 2

    Interpretation:
      Large positive gap → high star rating but negative review text
                           (courtesy inflation or sarcasm)
      Large negative gap → low stars but positive text
                           (unfair or culturally different rating behavior)
    """
    df['rating_normalized'] = (df['rating'] - 3) / 2
    df['rating_sentiment_gap'] = df['rating_normalized'] - df['sentiment_score']

    gap = df['rating_sentiment_gap'].dropna()
    logger.info(
        f'Rating-sentiment gap: mean={gap.mean():.3f}, std={gap.std():.3f}, '
        f'max={gap.max():.3f}, min={gap.min():.3f}'
    )
    return df


# ---------------------------------------------------------------------------
# Utility: load sentiment data (used by downstream steps)
# ---------------------------------------------------------------------------


def load_sentiment_interactions() -> pd.DataFrame:
    return pd.read_parquet(SENTIMENT_STAGING)
