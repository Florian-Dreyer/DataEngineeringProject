"""
extract.py
----------
Handles extraction of RAW_recipes.csv and RAW_interactions.csv.
Both extractions are designed to run as independent Airflow tasks in parallel.

Data is staged as parquet files rather than passed via XCom, since DataFrames
can be too large for Airflow's XCom storage (backed by the metadata DB).
"""

import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config — in production these come from Airflow Variables or environment vars
# ---------------------------------------------------------------------------

DATA_DIR = Path(os.getenv('FOODCOM_DATA_DIR', '/opt/airflow/data'))
STAGING_DIR = Path(os.getenv('FOODCOM_STAGING_DIR', '/opt/airflow/staging'))
POSTGRES_CONN = os.getenv(
    'FOODCOM_POSTGRES_CONN', 'postgresql://user:password@postgres:5432/foodcom'
)

RECIPES_CSV = DATA_DIR / 'RAW_recipes.csv'
INTERACTIONS_CSV = DATA_DIR / 'RAW_interactions.csv'

RECIPES_STAGING = STAGING_DIR / 'recipes_extracted.parquet'
INTERACTIONS_STAGING = STAGING_DIR / 'interactions_extracted.parquet'


# ---------------------------------------------------------------------------
# Watermark helper
# ---------------------------------------------------------------------------


def get_last_processed_date(engine) -> datetime | None:
    """
    Returns the latest interaction date already loaded into fact_interactions,
    or None if the table is empty (first run).
    """
    with engine.connect() as conn:
        result = conn.execute(
            text("""
            SELECT MAX(date) FROM fact_interactions
        """)
        )
        value = result.scalar()

    if value is None:
        logger.info('fact_interactions is empty — this is a full initial load.')
    else:
        logger.info(f'Watermark: last processed date = {value}')

    return value


# ---------------------------------------------------------------------------
# Recipes extraction
# ---------------------------------------------------------------------------


def extract_recipes(**context) -> None:
    """
    Extracts RAW_recipes.csv and stages it as parquet.

    Recipes don't have a date field, so we always extract the full file.
    Downstream clean/load steps handle upserts, so reprocessing is safe.
    """
    logger.info(f'Reading recipes from {RECIPES_CSV}')

    df = pd.read_csv(
        RECIPES_CSV,
        usecols=['id', 'name', 'minutes', 'tags', 'nutrition', 'steps', 'ingredients'],
        dtype={'id': 'int64'},
    )

    _log_extraction_stats('recipes', df)
    _validate_recipes(df)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(RECIPES_STAGING, index=False)

    logger.info(f'Recipes staged to {RECIPES_STAGING}')

    # Push record count to XCom for monitoring/alerting
    context['ti'].xcom_push(key='recipes_record_count', value=len(df))


def _validate_recipes(df: pd.DataFrame) -> None:
    """Basic schema and null checks — fail fast before touching the warehouse."""
    required_cols = {
        'id',
        'name',
        'minutes',
        'tags',
        'nutrition',
        'steps',
        'ingredients',
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Recipes CSV missing expected columns: {missing}')

    null_ids = df['id'].isnull().sum()
    if null_ids > 0:
        raise ValueError(f'Found {null_ids} null recipe IDs — data quality issue.')

    logger.info('Recipes validation passed.')


# ---------------------------------------------------------------------------
# Interactions extraction
# ---------------------------------------------------------------------------


def extract_interactions(**context) -> None:
    """
    Extracts RAW_interactions.csv, filtering to only rows not yet processed
    (based on the watermark in fact_interactions).

    On first run (empty warehouse), extracts everything except the held-out
    streaming simulation set, which is identified by a separate flag column
    or a pre-split file.
    """
    engine = create_engine(POSTGRES_CONN)
    last_processed_date = get_last_processed_date(engine)

    logger.info(f'Reading interactions from {INTERACTIONS_CSV}')

    df = pd.read_csv(
        INTERACTIONS_CSV,
        usecols=['user_id', 'recipe_id', 'date', 'rating', 'review'],
        dtype={'user_id': 'int64', 'recipe_id': 'int64', 'rating': 'int8'},
        parse_dates=['date'],
    )

    total_rows = len(df)

    # Apply watermark filter — only process new records
    if last_processed_date is not None:
        df = df[df['date'] > pd.Timestamp(last_processed_date)]
        logger.info(
            f'Watermark filter applied: {total_rows} total → {len(df)} new interactions'
        )
    else:
        logger.info(f'Initial load: processing all {total_rows} interactions')

    if df.empty:
        logger.info('No new interactions found. Nothing to stage.')
        context['ti'].xcom_push(key='interactions_record_count', value=0)
        context['ti'].xcom_push(key='has_new_data', value=False)
        return

    _log_extraction_stats('interactions', df)
    _validate_interactions(df)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(INTERACTIONS_STAGING, index=False)

    logger.info(f'Interactions staged to {INTERACTIONS_STAGING}')

    context['ti'].xcom_push(key='interactions_record_count', value=len(df))
    context['ti'].xcom_push(key='has_new_data', value=True)


def _validate_interactions(df: pd.DataFrame) -> None:
    """Checks ratings are in valid range and required fields are present."""
    required_cols = {'user_id', 'recipe_id', 'date', 'rating', 'review'}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f'Interactions CSV missing expected columns: {missing}')

    invalid_ratings = df[~df['rating'].between(1, 5)]
    if not invalid_ratings.empty:
        logger.warning(
            f'Found {len(invalid_ratings)} interactions with out-of-range ratings '
            f'(expected 1-5). These will be dropped in the clean step.'
        )

    null_keys = df[['user_id', 'recipe_id', 'date']].isnull().any(axis=1).sum()
    if null_keys > 0:
        raise ValueError(f'Found {null_keys} interactions with null key fields.')

    logger.info('Interactions validation passed.')


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _log_extraction_stats(name: str, df: pd.DataFrame) -> None:
    """Logs shape, dtypes and null counts — useful for the report's runtime table."""
    logger.info(f'--- {name.upper()} EXTRACTION STATS ---')
    logger.info(f'  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns')
    logger.info(f'  Columns    : {list(df.columns)}')
    logger.info(f'  Null counts:\n{df.isnull().sum().to_string()}')
    logger.info(f'  Memory     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
