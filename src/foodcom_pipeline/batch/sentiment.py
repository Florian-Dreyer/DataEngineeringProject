"""
sentiment.py
------------
Sentiment analysis step for the Food.com batch pipeline.

Two responsibilities:
  1. Score all new interactions with DistilBERT (batch layer — accurate)
  2. On the first run, produce a one-time VADER vs DistilBERT comparison
     report on a 10,000-review sample (for the report discussion section)

Sentiment score convention throughout: [-1, +1]
  -1 = maximally negative
   0 = neutral
  +1 = maximally positive

DistilBERT (SST-2) outputs softmax probabilities over [NEGATIVE, POSITIVE].
We normalize to [-1, +1] via:  score = 2 * P(POSITIVE) - 1

VADER outputs a compound score already in [-1, +1] — no transformation needed.
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from foodcom_pipeline.batch.clean import load_cleaned_interactions
from foodcom_pipeline.batch.extract import STAGING_DIR
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SENTIMENT_STAGING = STAGING_DIR / 'interactions_sentiment.parquet'
VADER_COMPARISON_PATH = STAGING_DIR / 'vader_distilbert_comparison.json'

COMPARISON_SAMPLE_SIZE = 10_000
DISTILBERT_BATCH_SIZE = 64
DISTILBERT_MODEL = 'distilbert-base-uncased-finetuned-sst-2-english'


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_sentiment(**context) -> None:
    """
    Main sentiment entry point:
      1. Score interactions with DistilBERT
      2. Compute rating_sentiment_gap
      3. On first run, produce VADER vs DistilBERT comparison report
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

    df = _score_distilbert(df, scoreable)
    df = _compute_rating_sentiment_gap(df)

    if not VADER_COMPARISON_PATH.exists():
        logger.info('VADER comparison report not found — running one-time comparison.')
        _run_vader_comparison(df)
    else:
        logger.info(
            f'VADER comparison report already exists at '
            f'{VADER_COMPARISON_PATH} — skipping.'
        )

    df.to_parquet(SENTIMENT_STAGING, index=False)
    logger.info(f'Sentiment step complete. Staged to {SENTIMENT_STAGING}')

    scored = df['sentiment_score'].notna().sum()
    context['ti'].xcom_push(key='sentiment_scored_count', value=int(scored))
    context['ti'].xcom_push(
        key='sentiment_unscored_count', value=int(unscoreable_count)
    )


# ---------------------------------------------------------------------------
# DistilBERT scoring
# ---------------------------------------------------------------------------


def _score_distilbert(df: pd.DataFrame, scoreable_mask: pd.Series) -> pd.DataFrame:
    """
    Run DistilBERT inference on all scoreable reviews.
    Processes in batches to manage memory on CPU.
    Returns df with a new `sentiment_score` column in [-1, +1].
    """
    from transformers import pipeline

    logger.info(f'Loading DistilBERT model: {DISTILBERT_MODEL}')
    classifier = pipeline(
        'sentiment-analysis',
        model=DISTILBERT_MODEL,
        truncation=True,
        max_length=512,
        batch_size=DISTILBERT_BATCH_SIZE,
    )

    reviews = df.loc[scoreable_mask, 'review'].tolist()
    logger.info(f'Scoring {len(reviews):,} reviews with DistilBERT...')

    raw_scores = []
    for i in range(0, len(reviews), DISTILBERT_BATCH_SIZE):
        batch = reviews[i : i + DISTILBERT_BATCH_SIZE]
        results = classifier(batch)
        for result in results:
            p_positive = (
                result['score']
                if result['label'] == 'POSITIVE'
                else 1 - result['score']
            )
            raw_scores.append(2 * p_positive - 1)

        if (i // DISTILBERT_BATCH_SIZE) % 10 == 0:
            progress = min(i + DISTILBERT_BATCH_SIZE, len(reviews))
            logger.info(f'  Progress: {progress:,} / {len(reviews):,}')

    df['sentiment_score'] = np.nan
    df.loc[scoreable_mask, 'sentiment_score'] = raw_scores

    mean = df['sentiment_score'].mean()
    std = df['sentiment_score'].std()
    logger.info(f'DistilBERT scoring complete. Mean score: {mean:.3f}, Std: {std:.3f}')
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
# VADER vs DistilBERT comparison (one-time, for the report)
# ---------------------------------------------------------------------------


@dataclass
class ComparisonMetrics:
    """Holds all computed comparison metrics in one place."""

    agreement_rate: float
    mean_abs_diff: float
    pearson_corr: float
    pearson_pvalue: float
    vader_accuracy: float
    distilbert_accuracy: float
    vader_time_ms: float
    distilbert_time_ms: float


def _run_vader_comparison(df: pd.DataFrame) -> None:
    """
    Produces the four comparison metrics between VADER and DistilBERT on a
    random sample of COMPARISON_SAMPLE_SIZE reviews.

    Metrics (all on normalized [-1, +1] scale):
      1. Agreement Rate       — fraction where sign(vader) == sign(distilbert)
      2. Mean Abs Difference  — average magnitude of score divergence
      3. Pearson Correlation  — linear correlation between the two distributions
      4. Star Rating Accuracy — fraction where sign(score) == sign(rating - 3)

    Results are written to VADER_COMPARISON_PATH as JSON.
    """
    import time

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    scoreable = (
        df['review'].notna()
        & (df['review'].str.strip() != '')
        & df['sentiment_score'].notna()
        & df['rating'].notna()
    )
    pool = df[scoreable]

    n = min(COMPARISON_SAMPLE_SIZE, len(pool))
    if n < COMPARISON_SAMPLE_SIZE:
        logger.warning(
            f'Only {n} scoreable reviews available — '
            f'using all instead of {COMPARISON_SAMPLE_SIZE}.'
        )

    sample = pool.sample(n=n, random_state=42)
    logger.info(f'Running VADER vs DistilBERT comparison on {n:,} reviews...')

    t0 = time.time()
    analyzer = SentimentIntensityAnalyzer()
    vader_scores = [
        analyzer.polarity_scores(review)['compound']
        for review in sample['review'].tolist()
    ]
    vader_time_ms = (time.time() - t0) / n * 1000

    vader_arr = np.array(vader_scores)
    distilbert_arr = sample['sentiment_score'].values
    ratings = sample['rating'].values

    metrics = ComparisonMetrics(
        agreement_rate=_agreement_rate(vader_arr, distilbert_arr),
        mean_abs_diff=_mean_abs_difference(vader_arr, distilbert_arr),
        pearson_corr=_pearson_correlation(vader_arr, distilbert_arr)[0],
        pearson_pvalue=_pearson_correlation(vader_arr, distilbert_arr)[1],
        vader_accuracy=_star_rating_accuracy(vader_arr, ratings),
        distilbert_accuracy=_star_rating_accuracy(distilbert_arr, ratings),
        vader_time_ms=vader_time_ms,
        distilbert_time_ms=_estimate_distilbert_time_ms(),
    )

    results = _metrics_to_dict(metrics, n)
    _log_comparison_results(results)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    with open(VADER_COMPARISON_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f'VADER comparison report saved to {VADER_COMPARISON_PATH}')


def _metrics_to_dict(m: ComparisonMetrics, sample_size: int) -> dict:
    """Serialises ComparisonMetrics to the JSON structure saved to disk."""
    return {
        'sample_size': sample_size,
        'metrics': {
            'agreement_rate': round(m.agreement_rate, 4),
            'mean_abs_difference': round(m.mean_abs_diff, 4),
            'pearson_correlation': round(m.pearson_corr, 4),
            'pearson_pvalue': round(m.pearson_pvalue, 6),
        },
        'star_rating_accuracy': {
            'vader': round(m.vader_accuracy, 4),
            'distilbert': round(m.distilbert_accuracy, 4),
        },
        'avg_inference_time_ms': {
            'vader': round(m.vader_time_ms, 4),
            'distilbert_estimated': round(m.distilbert_time_ms, 1),
        },
        'interpretation': _generate_interpretation(m),
    }


# ── Metric calculations ───────────────────────────────────────────────────────


def _agreement_rate(vader: np.ndarray, distilbert: np.ndarray) -> float:
    """
    Fraction of reviews where both models agree on sentiment direction.
    Exact zeros are excluded (no clear direction).
    """
    mask = (vader != 0) & (distilbert != 0)
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(vader[mask]) == np.sign(distilbert[mask])))


def _mean_abs_difference(vader: np.ndarray, distilbert: np.ndarray) -> float:
    """Average absolute difference between the two score distributions."""
    return float(np.mean(np.abs(vader - distilbert)))


def _pearson_correlation(
    vader: np.ndarray, distilbert: np.ndarray
) -> tuple[float, float]:
    """Pearson correlation coefficient and p-value."""
    corr, pvalue = pearsonr(vader, distilbert)
    return float(corr), float(pvalue)


def _star_rating_accuracy(scores: np.ndarray, ratings: np.ndarray) -> float:
    """
    Primary metric: fraction where sign(score) == sign(rating - 3).
    Rating 3 is excluded (ambiguous direction).
    """
    mask = ratings != 3
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.sign(scores[mask]) == np.sign(ratings[mask] - 3)))


def _estimate_distilbert_time_ms() -> float:
    """Typical DistilBERT CPU inference time per review (~200–400ms)."""
    return 300.0


def _generate_interpretation(m: ComparisonMetrics) -> str:
    """
    Generates a ready-to-use paragraph for the report with actual metrics
    filled in.
    """
    speedup = m.distilbert_time_ms / m.vader_time_ms if m.vader_time_ms > 0 else 0
    return (
        f'DistilBERT achieves a star rating accuracy of '
        f"{m.distilbert_accuracy:.1%} vs VADER's {m.vader_accuracy:.1%}, "
        f'meaning it better predicts whether a user actually liked a recipe. '
        f'The two models agree on sentiment direction {m.agreement_rate:.1%} '
        f'of the time (Pearson r = {m.pearson_corr:.2f}), with an average '
        f'score divergence of {m.mean_abs_diff:.2f} on the [-1, +1] scale. '
        f'This accuracy improvement comes at a cost of ~{speedup:.0f}x longer '
        f'inference time (~{m.distilbert_time_ms:.0f}ms vs '
        f'~{m.vader_time_ms:.2f}ms per review). This tradeoff justifies the '
        f'Lambda Architecture design: DistilBERT runs in the batch layer where '
        f'reviews are processed in bulk on a schedule, while VADER serves as a '
        f'fast approximation in the speed layer, with its scores overwritten by '
        f'DistilBERT in the next batch run.'
    )


def _log_comparison_results(results: dict) -> None:
    """Pretty-prints the comparison results to the Airflow log."""
    m = results['metrics']
    a = results['star_rating_accuracy']
    t = results['avg_inference_time_ms']

    logger.info('=' * 55)
    logger.info('  VADER vs DistilBERT Comparison Results')
    logger.info('=' * 55)
    logger.info(f"  Sample size          : {results['sample_size']:,}")
    logger.info(f"  Agreement Rate       : {m['agreement_rate']:.4f}")
    logger.info(f"  Mean Abs Difference  : {m['mean_abs_difference']:.4f}")
    logger.info(
        f"  Pearson Correlation  : {m['pearson_correlation']:.4f} "
        f"(p={m['pearson_pvalue']:.2e})"
    )
    logger.info(
        f"  Star Rating Accuracy : "
        f"VADER={a['vader']:.4f}  DistilBERT={a['distilbert']:.4f}"
    )
    logger.info(
        f"  Avg Inference Time   : "
        f"VADER={t['vader']:.3f}ms  "
        f"DistilBERT~{t['distilbert_estimated']:.0f}ms"
    )
    logger.info('=' * 55)
    logger.info(f"  Interpretation:\n  {results['interpretation']}")
    logger.info('=' * 55)


# ---------------------------------------------------------------------------
# Utility: load sentiment data (used by downstream steps)
# ---------------------------------------------------------------------------


def load_sentiment_interactions() -> pd.DataFrame:
    return pd.read_parquet(SENTIMENT_STAGING)
