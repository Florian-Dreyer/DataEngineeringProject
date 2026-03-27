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
import subprocess
import zipfile
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

KAGGLE_DATASET = os.getenv(
    'FOODCOM_KAGGLE_DATASET',
    'shuyangli94/food-com-recipes-and-user-interactions',
)
ENABLE_KAGGLE_DOWNLOAD = (
    os.getenv('FOODCOM_ENABLE_KAGGLE_DOWNLOAD', 'true').lower() == 'true'
)


# ---------------------------------------------------------------------------
# Watermark helper
# ---------------------------------------------------------------------------


def get_last_processed_date(engine) -> datetime | None:
    """
    Returns the latest interaction date already loaded into fact_interactions,
    or None if the table is empty (first run).
    """
    # Star schema stores dates in dim_date and references them via date_id.
    # On first run, tables may not exist yet; treat that as "no watermark".
    try:
        with engine.connect() as conn:
            value = conn.execute(
                text(
                    """
                    SELECT MAX(d.full_date) AS last_processed_date
                    FROM fact_interactions f
                    JOIN dim_date d
                      ON d.date_id = f.date_id
                    """
                )
            ).scalar()
    except Exception as e:
        logger.info(
            'Could not read watermark from warehouse (first run or schema missing). '
            f'Proceeding with full extract. Error: {e}'
        )
        return None

    if value is None:
        logger.info('fact_interactions is empty — this is a full initial load.')
    else:
        logger.info(f'Watermark: last processed date = {value}')

    return value


# ---------------------------------------------------------------------------
# Source data bootstrap (Kaggle API)
# ---------------------------------------------------------------------------


def ensure_source_data(**context) -> None:
    """
    Ensures RAW_recipes.csv and RAW_interactions.csv exist in DATA_DIR.

    If either file is missing and FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true, downloads
    both files from the Kaggle dataset using the Kaggle CLI.
    """
    logger.info(
        f'Checking source files in {DATA_DIR}: '
        f'recipes_csv_exists={RECIPES_CSV.exists()}, '
        f'interactions_csv_exists={INTERACTIONS_CSV.exists()}'
    )

    # If CSVs are missing but ZIPs exist from a previous Kaggle run, extract first.
    logger.info('Attempting pre-download ZIP extraction (if needed).')
    _extract_csv_from_zip_if_needed(RECIPES_CSV)
    _extract_csv_from_zip_if_needed(INTERACTIONS_CSV)

    recipes_exists = RECIPES_CSV.exists()
    interactions_exists = INTERACTIONS_CSV.exists()
    logger.info(
        'Post pre-download extraction check: '
        f'recipes_csv_exists={recipes_exists}, '
        f'interactions_csv_exists={interactions_exists}'
    )

    if recipes_exists and interactions_exists:
        logger.info('Raw CSV files already present. Skipping Kaggle download.')
        return

    if not ENABLE_KAGGLE_DOWNLOAD:
        raise FileNotFoundError(
            'Raw CSV files are missing and Kaggle auto-download is disabled. '
            'Set FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true or place files at '
            f'{RECIPES_CSV} and {INTERACTIONS_CSV}.'
        )

    logger.info(
        f'Raw CSV missing. Downloading from Kaggle dataset "{KAGGLE_DATASET}" '
        f'to {DATA_DIR}...'
    )
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    _download_file_from_kaggle('RAW_recipes.csv')
    _download_file_from_kaggle('RAW_interactions.csv')

    # Kaggle may leave .zip files even with --unzip in some environments.
    logger.info('Attempting post-download ZIP extraction (if needed).')
    _extract_csv_from_zip_if_needed(RECIPES_CSV)
    _extract_csv_from_zip_if_needed(INTERACTIONS_CSV)

    logger.info(
        'Final source file check: '
        f'recipes_csv_exists={RECIPES_CSV.exists()}, '
        f'interactions_csv_exists={INTERACTIONS_CSV.exists()}'
    )

    if not RECIPES_CSV.exists() or not INTERACTIONS_CSV.exists():
        raise FileNotFoundError(
            'Kaggle download completed but expected files were not found: '
            f'{RECIPES_CSV}, {INTERACTIONS_CSV}'
        )

    logger.info('Kaggle source data download complete.')
    context['ti'].xcom_push(key='source_dataset', value=KAGGLE_DATASET)


def _download_file_from_kaggle(filename: str) -> None:
    cmd = [
        'kaggle',
        'datasets',
        'download',
        '-d',
        KAGGLE_DATASET,
        '-f',
        filename,
        '-p',
        str(DATA_DIR),
        '--unzip',
    ]

    logger.info(f'Downloading {filename} from Kaggle...')
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        raise RuntimeError(
            f'Kaggle download failed for {filename}.\n'
            f'STDOUT: {result.stdout}\nSTDERR: {result.stderr}\n'
            'Ensure Kaggle CLI is installed and credentials are available '
            '(~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY).'
        )


def _extract_csv_from_zip_if_needed(csv_path: Path) -> None:
    """
    If csv_path is missing, try to materialize it from ZIP archives in DATA_DIR.
    """
    if csv_path.exists():
        logger.info(f'{csv_path.name} already exists. Skipping ZIP extraction.')
        return

    # Prefer the conventional "<name>.csv.zip", then fall back to any ZIP
    # containing the target CSV.
    candidate_zips = [csv_path.with_suffix(csv_path.suffix + '.zip')]
    candidate_zips.extend(sorted(DATA_DIR.glob('*.zip')))

    # Deduplicate while preserving order
    seen = set()
    ordered_zips = []
    for zp in candidate_zips:
        if zp in seen:
            continue
        seen.add(zp)
        ordered_zips.append(zp)

    zip_paths = [zp for zp in ordered_zips if zp.exists()]
    if not zip_paths:
        logger.info(f'No ZIP archives found for {csv_path.name} in {DATA_DIR}.')
        return

    logger.info(
        f'ZIP candidates for {csv_path.name}: {[str(p.name) for p in zip_paths]}'
    )
    for zip_path in zip_paths:
        logger.info(f'Attempting extraction for {csv_path.name} from {zip_path}.')
        with zipfile.ZipFile(zip_path, 'r') as zf:
            members = zf.namelist()
            logger.info(f'ZIP member count for {zip_path.name}: {len(members)}')

            # Case 1: exact filename present in archive
            if csv_path.name in members:
                zf.extract(csv_path.name, path=DATA_DIR)
                logger.info(f'Extracted {csv_path.name} from {zip_path}.')
                return

            # Case 2: archive has nested paths; match by basename
            basename_matches = [
                m for m in members if Path(m).name.lower() == csv_path.name.lower()
            ]
            if basename_matches:
                member = basename_matches[0]
                logger.info(
                    f'Found basename match for {csv_path.name} in {zip_path.name}: {member}'
                )
                zf.extract(member, path=DATA_DIR)
                extracted_path = DATA_DIR / member
                extracted_path.parent.mkdir(parents=True, exist_ok=True)
                extracted_path.replace(csv_path)
                logger.info(
                    f'Extracted {member} from {zip_path} and renamed to {csv_path.name}.'
                )
                return

    raise FileNotFoundError(
        f'Could not find {csv_path.name} inside ZIP archives in {DATA_DIR}.'
    )


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
