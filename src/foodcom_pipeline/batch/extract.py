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
from datetime import date, datetime
from pathlib import Path
import pickle
import sys
import threading
import types
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic, sleep
from typing import Any

import pandas as pd
import requests
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
INGR_MAP_PKL = DATA_DIR / 'ingr_map.pkl'

RECIPES_STAGING = STAGING_DIR / 'recipes_extracted.parquet'
INTERACTIONS_STAGING = STAGING_DIR / 'interactions_extracted.parquet'
USDA_NUTRIENTS_STAGING = STAGING_DIR / 'usda_nutrients.parquet'

KAGGLE_DATASET = os.getenv(
    'FOODCOM_KAGGLE_DATASET',
    'shuyangli94/food-com-recipes-and-user-interactions',
)
ENABLE_KAGGLE_DOWNLOAD = (
    os.getenv('FOODCOM_ENABLE_KAGGLE_DOWNLOAD', 'true').lower() == 'true'
)


@dataclass
class _GlobalRateLimiter:
    """
    Thread-safe global rate limiter.

    Supports both:
      - optional max_rps smoothing
      - optional strict rolling-hour request cap
    """

    max_rps: float
    max_requests_per_hour: int = 0

    def __post_init__(self):
        self._lock = threading.Lock()
        self._next_allowed = 0.0
        self._window_s = 3600.0
        self._request_times: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            wait_s = 0.0
            with self._lock:
                now = monotonic()

                # Enforce strict rolling 1-hour cap if configured.
                if self.max_requests_per_hour > 0:
                    cutoff = now - self._window_s
                    while self._request_times and self._request_times[0] <= cutoff:
                        self._request_times.popleft()

                    if len(self._request_times) >= self.max_requests_per_hour:
                        wait_s = max(0.0, self._request_times[0] + self._window_s - now)
                    else:
                        # Optional smoothing on top of hourly cap.
                        if self.max_rps > 0:
                            min_interval = 1.0 / self.max_rps
                            wait_s = max(0.0, self._next_allowed - now)
                            if wait_s == 0.0:
                                self._next_allowed = now + min_interval
                        if wait_s == 0.0:
                            self._request_times.append(now)
                            return
                else:
                    # Legacy behavior when no hourly cap is configured.
                    if self.max_rps <= 0:
                        return
                    min_interval = 1.0 / self.max_rps
                    wait_s = max(0.0, self._next_allowed - now)
                    if wait_s == 0.0:
                        self._next_allowed = now + min_interval
                        return

            if wait_s > 0.0:
                sleep(wait_s)


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
    Ensures RAW_recipes.csv, RAW_interactions.csv, and ingr_map.pkl exist in DATA_DIR.

    If any file is missing and FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true, downloads
    missing files from the Kaggle dataset using the Kaggle CLI.
    """
    logger.info(
        f'Checking source files in {DATA_DIR}: '
        f'recipes_csv_exists={RECIPES_CSV.exists()}, '
        f'interactions_csv_exists={INTERACTIONS_CSV.exists()}, '
        f'ingr_map_exists={INGR_MAP_PKL.exists()}'
    )

    # If CSVs are missing but ZIPs exist from a previous Kaggle run, extract first.
    logger.info('Attempting pre-download ZIP extraction (if needed).')
    _extract_csv_from_zip_if_needed(RECIPES_CSV)
    _extract_csv_from_zip_if_needed(INTERACTIONS_CSV)

    recipes_exists = RECIPES_CSV.exists()
    interactions_exists = INTERACTIONS_CSV.exists()
    ingr_map_exists = INGR_MAP_PKL.exists()
    logger.info(
        'Post pre-download extraction check: '
        f'recipes_csv_exists={recipes_exists}, '
        f'interactions_csv_exists={interactions_exists}, '
        f'ingr_map_exists={ingr_map_exists}'
    )

    if recipes_exists and interactions_exists and ingr_map_exists:
        logger.info('All required source files present. Skipping Kaggle download.')
        return

    if not ENABLE_KAGGLE_DOWNLOAD:
        missing = []
        if not recipes_exists:
            missing.append(str(RECIPES_CSV))
        if not interactions_exists:
            missing.append(str(INTERACTIONS_CSV))
        if not ingr_map_exists:
            missing.append(str(INGR_MAP_PKL))
        raise FileNotFoundError(
            f'Missing required source files ({missing}) and Kaggle auto-download is disabled. '
            'Set FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true or place files in DATA_DIR.'
        )

    logger.info(f'Source files missing. Downloading from Kaggle "{KAGGLE_DATASET}"...')
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not recipes_exists:
        _download_file_from_kaggle('RAW_recipes.csv')
    if not interactions_exists:
        _download_file_from_kaggle('RAW_interactions.csv')
    if not ingr_map_exists:
        _download_file_from_kaggle('ingr_map.pkl')

    # Kaggle may leave .zip files even with --unzip in some environments.
    logger.info('Attempting post-download ZIP extraction (if needed).')
    _extract_csv_from_zip_if_needed(RECIPES_CSV)
    _extract_csv_from_zip_if_needed(INTERACTIONS_CSV)

    logger.info(
        'Final source file check: '
        f'recipes_csv_exists={RECIPES_CSV.exists()}, '
        f'interactions_csv_exists={INTERACTIONS_CSV.exists()}, '
        f'ingr_map_exists={INGR_MAP_PKL.exists()}'
    )

    if not RECIPES_CSV.exists() or not INTERACTIONS_CSV.exists() or not INGR_MAP_PKL.exists():
        raise FileNotFoundError(
            'Kaggle download completed but expected files were not found: '
            f'{RECIPES_CSV}, {INTERACTIONS_CSV}, {INGR_MAP_PKL}'
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
# USDA extraction (canonical ingredient nutrient enrichment)
# ---------------------------------------------------------------------------


def _load_ingr_map(path: Path) -> dict[str, str]:
    """
    Loads ingr_map.pkl and normalizes it to {raw_string: canonical_string}.

    The exact pickle structure can vary across preprocessing versions; we handle
    the common cases (dict-like or DataFrame-like) and fall back with clear errors.
    """
    if not path.exists():
        raise FileNotFoundError(f'ingr_map.pkl not found at {path}')

    def _pickle_load_with_pandas_shim(fh):
        """
        `ingr_map.pkl` in the Kaggle dataset was created with older pandas.
        Pandas 2.x removed some internal module paths (e.g. pandas.core.indexes.numeric),
        causing ModuleNotFoundError during unpickling.

        We shim the missing module path and alias legacy Index classes to `pandas.Index`
        so we can load the pickle and then extract the raw→canonical mapping.
        """
        try:
            return pickle.load(fh)
        except ModuleNotFoundError as e:
            if "pandas.core.indexes.numeric" not in str(e):
                raise

            import pandas as _pd

            shim = types.ModuleType("pandas.core.indexes.numeric")
            # Legacy index class names that appear in older pickles.
            for cls_name in [
                "Int64Index",
                "UInt64Index",
                "Float64Index",
                "NumericIndex",
            ]:
                setattr(shim, cls_name, _pd.Index)

            sys.modules["pandas.core.indexes.numeric"] = shim
            fh.seek(0)
            return pickle.load(fh)

    with path.open('rb') as f:
        obj: Any = _pickle_load_with_pandas_shim(f)

    if isinstance(obj, dict):
        out: dict[str, str] = {}
        for k, v in obj.items():
            out[str(k).strip().lower()] = str(v).strip().lower()
        return out

    # Some variants store two columns in a DataFrame-like object.
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower(): c for c in obj.columns}
        raw_col = cols.get('raw_ingr') or cols.get('raw_ingredient') or cols.get('raw')
        canon_col = cols.get('canonical') or cols.get('canonical_form') or cols.get(
            'replaced'
        )
        if raw_col and canon_col:
            out = {}
            for _, row in obj.iterrows():
                raw = row.get(raw_col)
                canon = row.get(canon_col)
                if pd.notnull(raw) and pd.notnull(canon):
                    out[str(raw).strip().lower()] = str(canon).strip().lower()
            if out:
                return out

    raise TypeError(
        f'Unsupported ingr_map.pkl structure in {path} (type={type(obj)!r}).'
    )


def _find_nutrient_amount(
    food_nutrients: list[dict[str, Any]],
    *,
    name_keywords: list[str],
    unit_keywords: list[str] | None = None,
) -> float | None:
    """
    Best-effort extraction from USDA `foodNutrients`.

    - Matches nutrientName with any `name_keywords` (case-insensitive substring)
    - Optionally filters by unitName containing any `unit_keywords`
    - Returns the first matching `amount` (or `value`) as float
    """
    for n in food_nutrients:
        nutrient_name = str(n.get('nutrientName') or n.get('name') or '').lower()
        if not nutrient_name:
            continue

        if not any(kw.lower() in nutrient_name for kw in name_keywords):
            continue

        if unit_keywords is not None:
            unit_name = str(n.get('unitName') or '').lower()
            if not unit_name:
                continue
            if not any(uk.lower() in unit_name for uk in unit_keywords):
                continue

        amount = n.get('amount')
        if amount is None:
            amount = n.get('value')

        try:
            return float(amount)
        except (TypeError, ValueError):
            return None

    return None


def _empty_usda_nutrient_row() -> dict[str, float | None]:
    return {
        'calories_per_100g': None,
        'protein_g_per_100g': None,
        'fat_g_per_100g': None,
        'saturated_fat_g_per_100g': None,
        'sugar_g_per_100g': None,
        'sodium_g_per_100g': None,
        'carbs_g_per_100g': None,
    }


def _query_usda_nutrients_for_ingredient(
    ingredient: str,
    *,
    api_key: str,
    session: requests.Session,
    timeout_s: float,
    max_retries: int,
    rate_limiter: _GlobalRateLimiter,
) -> dict[str, float | None]:
    """
    Queries USDA FoodData Central (POST /foods/search) for one ingredient.

    Returns a dict of per-100g nutrient values. Missing values are None.
    """
    base_url = os.getenv(
        'FOODCOM_USDA_FDC_SEARCH_URL',
        'https://api.nal.usda.gov/fdc/v1/foods/search',
    )
    data_types = os.getenv(
        'FOODCOM_USDA_DATA_TYPES',
        'Survey (FNDDS),SR Legacy,Foundation',
    )

    params = {'api_key': api_key}
    payload = {
        'query': ingredient,
        'dataType': data_types,
        'pageSize': 1,
    }

    response = None
    for attempt in range(max_retries + 1):
        rate_limiter.acquire()
        response = session.post(
            base_url,
            params=params,
            json=payload,
            timeout=timeout_s,
        )

        # Retry on USDA rate-limit and transient service failures.
        if response.status_code in {429, 500, 502, 503, 504} and attempt < max_retries:
            backoff = min(8.0, 0.5 * (2**attempt))
            sleep(backoff)
            continue
        break

    assert response is not None
    response.raise_for_status()
    payload = response.json()

    foods = payload.get('foods') or []
    if not foods:
        return _empty_usda_nutrient_row()

    food0 = foods[0]
    nutrients = food0.get('foodNutrients') or []
    if not isinstance(nutrients, list):
        nutrients = []

    # Calories / Energy
    calories = _find_nutrient_amount(
        nutrients,
        name_keywords=['energy'],
        unit_keywords=['kcal'],
    )

    protein = _find_nutrient_amount(
        nutrients,
        name_keywords=['protein'],
    )

    fat = _find_nutrient_amount(
        nutrients,
        name_keywords=['total lipid', 'fat'],
    )

    saturated_fat = _find_nutrient_amount(
        nutrients,
        name_keywords=['total saturated'],
    )

    sugar = _find_nutrient_amount(
        nutrients,
        name_keywords=['sugars', 'total sugars'],
    )

    sodium_mg = _find_nutrient_amount(
        nutrients,
        name_keywords=['sodium'],
        unit_keywords=['mg'],
    )
    sodium_g = sodium_mg / 1000.0 if sodium_mg is not None else None

    carbs = _find_nutrient_amount(
        nutrients,
        name_keywords=['carbohydrate', 'carbohydrates'],
    )

    return {
        'calories_per_100g': calories,
        'protein_g_per_100g': protein,
        'fat_g_per_100g': fat,
        'saturated_fat_g_per_100g': saturated_fat,
        'sugar_g_per_100g': sugar,
        'sodium_g_per_100g': sodium_g,
        'carbs_g_per_100g': carbs,
    }


def extract_usda_nutrients(**context) -> None:
    """
    Extracts USDA FoodData Central nutrient data for canonical ingredients from ingr_map.pkl.

    Writes `usda_nutrients.parquet` into the Airflow staging directory so downstream
    transforms can join recipe ingredients against USDA nutrient ground truth.
    """
    api_key = os.getenv('USDA_API_KEY') or os.getenv('FOODCOM_USDA_API_KEY') or ''
    if not api_key:
        raise EnvironmentError(
            'USDA API key not found. Set `USDA_API_KEY` (or FOODCOM_USDA_API_KEY).'
        )

    # Tunables for safe parallelism:
    # - workers: concurrency (5-10 recommended)
    # - global max_rps: aggregate request throughput cap across all workers
    # - retries/timeout: resilience on transient USDA/API failures
    workers = int(os.getenv('FOODCOM_USDA_WORKERS', '6'))
    timeout_s = float(os.getenv('FOODCOM_USDA_TIMEOUT_SECONDS', '30'))
    max_retries = int(os.getenv('FOODCOM_USDA_MAX_RETRIES', '3'))
    global_max_rps = float(os.getenv('FOODCOM_USDA_MAX_RPS', '4'))
    max_requests_per_hour = int(
        os.getenv('FOODCOM_USDA_MAX_REQUESTS_PER_HOUR', '999')
    )
    max_ing = os.getenv('FOODCOM_USDA_MAX_INGREDIENTS')
    max_ing_int: int | None = int(max_ing) if max_ing else None

    ingr_map = _load_ingr_map(INGR_MAP_PKL)
    canonical_ingredients = sorted(set(ingr_map.values()))
    if max_ing_int is not None:
        canonical_ingredients = canonical_ingredients[:max_ing_int]

    if not canonical_ingredients:
        raise ValueError('No canonical ingredients found in ingr_map.pkl')

    logger.info(
        f'Querying USDA for {len(canonical_ingredients):,} canonical ingredients '
        f'using workers={workers}, max_rps={global_max_rps}, '
        f'max_requests_per_hour={max_requests_per_hour}, timeout={timeout_s}s.'
    )

    rows: list[dict[str, Any]] = []
    computed_date = date.today()
    rate_limiter = _GlobalRateLimiter(
        max_rps=global_max_rps,
        max_requests_per_hour=max_requests_per_hour,
    )
    thread_local = threading.local()

    def _get_session() -> requests.Session:
        sess = getattr(thread_local, 'session', None)
        if sess is None:
            sess = requests.Session()
            sess.headers.update({'User-Agent': 'foodcom-pipeline/1.0'})
            thread_local.session = sess
        return sess

    def _fetch_one(ingredient: str) -> dict[str, Any]:
        try:
            nutrient_row = _query_usda_nutrients_for_ingredient(
                ingredient,
                api_key=api_key,
                session=_get_session(),
                timeout_s=timeout_s,
                max_retries=max_retries,
                rate_limiter=rate_limiter,
            )
        except Exception as e:
            logger.warning(f'USDA query failed for ingredient="{ingredient}": {e}')
            nutrient_row = _empty_usda_nutrient_row()

        nutrient_row['canonical_ingredient'] = ingredient
        nutrient_row['computed_date'] = computed_date
        return nutrient_row

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(_fetch_one, ingredient): ingredient
            for ingredient in canonical_ingredients
        }
        for i, fut in enumerate(as_completed(futures), start=1):
            rows.append(fut.result())
            if i % 50 == 0 or i == len(canonical_ingredients):
                logger.info(f'USDA progress: {i}/{len(canonical_ingredients)}')

    usda_df = pd.DataFrame(rows)
    usda_df.to_parquet(USDA_NUTRIENTS_STAGING, index=False)

    coverage = int(usda_df['calories_per_100g'].notna().sum())
    coverage_rate = coverage / max(1, len(usda_df))
    logger.info(
        f'USDA extraction complete. Coverage(calories)={coverage:,}/{len(usda_df):,} '
        f'({coverage_rate:.1%}). Staged to {USDA_NUTRIENTS_STAGING}'
    )
    context['ti'].xcom_push(key='usda_coverage_rate', value=float(coverage_rate))
    context['ti'].xcom_push(key='usda_rows', value=int(len(usda_df)))


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
