"""
extract.py
----------
Handles extraction of RAW_recipes.csv and RAW_interactions.csv.
Both extractions are designed to run as independent Airflow tasks in parallel.

Data is staged as parquet files rather than passed via XCom, since DataFrames
can be too large for Airflow's XCom storage (backed by the metadata DB).
"""

import json
import logging
import os
import re
import subprocess
import zipfile
from datetime import date, datetime
from pathlib import Path
import pickle
import sys
import types
from typing import Any, Union

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


def atomic_parquet(df: pd.DataFrame, dest: Union[Path, str]) -> None:
    """Write df to a .tmp file then atomically rename to dest.

    Avoids EDEADLK (errno 35) on macOS Docker bind mounts when concurrent
    tasks read from the same staging directory during a write.
    """
    tmp = str(dest) + '.tmp'
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)


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
TRENDS_RAW_STAGING = STAGING_DIR / 'google_trends_raw.parquet'
TRENDS_NORMALISED_STAGING = STAGING_DIR / 'google_trends_normalised.parquet'
AI_MODE_RAW_STAGING = STAGING_DIR / 'ai_mode_raw.parquet'
AI_MODE_TERM_SCORES_STAGING = STAGING_DIR / 'ai_mode_term_scores.parquet'

from foodcom_pipeline.extraction.trends import SEEDS as _TRENDS_SEEDS

_AI_MODE_SEEDS: list[str] = [
    'dinner ideas',
    'lunch ideas',
    'breakfast ideas',
    'easy weeknight meals',
    'healthy meal ideas',
    'quick recipes',
    'what to cook tonight',
    'meal prep ideas',
    'party food ideas',
]

# USDA FoodData Central bulk JSON (Foundation, SR Legacy, FNDDS survey)
_USDA_JSON_SPECS: tuple[tuple[str, str], ...] = (
    ('FoodData_Central_foundation_food_json_2025-12-18.json', 'FoundationFoods'),
    ('FoodData_Central_sr_legacy_food_json_2018-04.json', 'SRLegacyFoods'),
    ('surveyDownload.json', 'SurveyFoods'),
)
_USDA_DATATYPE_RANK: dict[str, int] = {
    'Foundation': 3,
    'SR Legacy': 2,
    'Survey (FNDDS)': 1,
}


def _default_usda_json_dir() -> Path:
    return Path(__file__).resolve().parents[3] / 'usda_data'


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

    # If files are missing but ZIPs exist from a previous Kaggle run, extract first.
    logger.info('Attempting pre-download ZIP extraction (if needed).')
    _extract_csv_from_zip_if_needed(RECIPES_CSV)
    _extract_csv_from_zip_if_needed(INTERACTIONS_CSV)
    _extract_csv_from_zip_if_needed(INGR_MAP_PKL)

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
    _extract_csv_from_zip_if_needed(INGR_MAP_PKL)

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
    atomic_parquet(df, RECIPES_STAGING)

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
        # Watermark filtered everything — warehouse is already up to date.
        # Keep the existing staging file so downstream tasks (clean, sentiment)
        # continue to work with the data from the last successful run.
        logger.info(
            'No new interactions after watermark filter — warehouse is current. '
            'Preserving existing interactions staging file for downstream tasks.'
        )
        context['ti'].xcom_push(key='interactions_record_count', value=0)
        context['ti'].xcom_push(key='has_new_data', value=False)
        return

    _log_extraction_stats('interactions', df)
    _validate_interactions(df)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(df, INTERACTIONS_STAGING)

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

    Supports:
    - API/search flat items (`nutrientName`, `unitName`)
    - FDC bulk download items with nested `nutrient.name` / `nutrient.unitName`

    - Matches nutrient name with any `name_keywords` (case-insensitive substring)
    - Optionally filters by unit containing any `unit_keywords`
    - Returns the first matching `amount` (or `value`) as float
    """
    for n in food_nutrients:
        nested = n.get('nutrient')
        if isinstance(nested, dict):
            nutrient_name = str(nested.get('name') or '').lower()
            unit_name = str(nested.get('unitName') or '').lower()
        else:
            nutrient_name = str(
                n.get('nutrientName') or n.get('name') or ''
            ).lower()
            unit_name = str(n.get('unitName') or '').lower()

        if not nutrient_name:
            continue

        if not any(kw.lower() in nutrient_name for kw in name_keywords):
            continue

        if unit_keywords is not None:
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


def _load_local_usda_foods(usda_dir: Path) -> list[dict[str, Any]]:
    """Loads Foundation, SR Legacy, and Survey (FNDDS) foods from bulk JSON downloads."""
    foods: list[dict[str, Any]] = []
    for filename, top_key in _USDA_JSON_SPECS:
        path = usda_dir / filename
        if not path.is_file():
            logger.warning('USDA JSON file not found, skipping: %s', path)
            continue
        try:
            # Read fully into memory before parsing to avoid EDEADLK (errno 35)
            # on Docker volume mounts where streaming large files can deadlock.
            raw = path.read_bytes()
            payload = json.loads(raw)
        except OSError as exc:
            logger.warning('Could not read %s (%s); skipping.', path, exc)
            continue
        except json.JSONDecodeError as exc:
            logger.warning('Could not parse %s (%s); skipping.', path, exc)
            continue
        batch = payload.get(top_key) or []
        if not isinstance(batch, list):
            logger.warning(
                'USDA JSON key %r in %s is not a list; skipping.', top_key, path
            )
            continue
        foods.extend(batch)
        logger.info('Loaded %d foods from %s', len(batch), filename)
    return foods


def _usda_match_sort_key(
    food: dict[str, Any],
    desc_lower: str,
    q: str,
    tokens: list[str],
) -> tuple[int, float, int, int, int]:
    """
    Higher tuple sorts greater. Picks the best matching USDA food row for an ingredient.

    Preference order: full phrase in description, token hit rate, data type,
    nutrient row count (richer), then shorter descriptions (generic items).
    """
    hits = sum(1 for t in tokens if t in desc_lower)
    rate = hits / max(1, len(tokens))
    phrase = 1 if q in desc_lower else 0
    dt = str(food.get('dataType') or '')
    rank = _USDA_DATATYPE_RANK.get(dt, 0)
    n_nut = len(food.get('foodNutrients') or [])
    return (phrase, rate, rank, n_nut, -len(desc_lower))


def _select_best_usda_food(
    ingredient: str,
    foods: list[dict[str, Any]],
    desc_lowers: list[str],
) -> dict[str, Any] | None:
    q = ingredient.strip().lower()
    tokens = [t for t in re.split(r'[^a-z0-9]+', q) if len(t) >= 2]
    if not q or not tokens:
        return None

    best_food: dict[str, Any] | None = None
    best_key: tuple[int, float, int, int, int] | None = None
    for food, desc_lower in zip(foods, desc_lowers, strict=True):
        key = _usda_match_sort_key(food, desc_lower, q, tokens)
        if best_key is None or key > best_key:
            best_key = key
            best_food = food

    if best_food is None or best_key is None:
        return None
    phrase, rate, *_rest = best_key
    if phrase == 0 and rate == 0:
        return None
    return best_food


def _nutrient_row_from_usda_food(food: dict[str, Any]) -> dict[str, float | None]:
    """Per-100g nutrient dict from one USDA food record (API or bulk JSON shape)."""
    nutrients = food.get('foodNutrients') or []
    if not isinstance(nutrients, list):
        nutrients = []

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


def _usda_force_refresh() -> bool:
    return os.getenv('FOODCOM_USDA_FORCE_REFRESH', '').lower() in (
        '1',
        'true',
        'yes',
    )


def extract_usda_nutrients(**context) -> None:
    """
    Extracts USDA FoodData Central nutrient data for canonical ingredients from ingr_map.pkl.

    Uses local bulk JSON under ``FOODCOM_USDA_JSON_DIR`` (Foundation, SR Legacy, FNDDS survey),
    not the public API. Writes `usda_nutrients.parquet` into the Airflow staging directory so
    downstream transforms can join recipe ingredients against USDA nutrient ground truth.

    If ``usda_nutrients.parquet`` already exists, the task skips recomputation unless
    ``FOODCOM_USDA_FORCE_REFRESH`` is set to a truthy value (``1``/``true``/``yes``),
    in which case the file is rebuilt from scratch.
    """
    force_refresh = _usda_force_refresh()
    if USDA_NUTRIENTS_STAGING.is_file() and not force_refresh:
        try:
            existing = pd.read_parquet(USDA_NUTRIENTS_STAGING)
        except Exception as e:
            logger.warning(
                'Could not read existing %s (%s); rebuilding USDA nutrients.',
                USDA_NUTRIENTS_STAGING,
                e,
            )
        else:
            cov = int(existing['calories_per_100g'].notna().sum())
            rate = cov / max(1, len(existing))
            logger.info(
                'Skipping USDA extract: %s already exists and '
                'FOODCOM_USDA_FORCE_REFRESH is not set (%s rows, calories coverage %.1f%%).',
                USDA_NUTRIENTS_STAGING,
                len(existing),
                100.0 * rate,
            )
            context['ti'].xcom_push(key='usda_coverage_rate', value=float(rate))
            context['ti'].xcom_push(key='usda_rows', value=int(len(existing)))
            context['ti'].xcom_push(key='usda_extract_skipped', value=True)
            return

    if force_refresh and USDA_NUTRIENTS_STAGING.is_file():
        logger.info(
            'FOODCOM_USDA_FORCE_REFRESH is set; rebuilding %s',
            USDA_NUTRIENTS_STAGING,
        )

    usda_dir = Path(
        os.getenv('FOODCOM_USDA_JSON_DIR', str(_default_usda_json_dir()))
    )
    max_ing = os.getenv('FOODCOM_USDA_MAX_INGREDIENTS')
    max_ing_int: int | None = int(max_ing) if max_ing else None

    ingr_map = _load_ingr_map(INGR_MAP_PKL)
    canonical_ingredients = sorted(set(ingr_map.values()))
    if max_ing_int is not None:
        canonical_ingredients = canonical_ingredients[:max_ing_int]

    if not canonical_ingredients:
        raise ValueError('No canonical ingredients found in ingr_map.pkl')

    logger.info('Loading USDA bulk JSON from %s', usda_dir)
    foods = _load_local_usda_foods(usda_dir)
    if not foods:
        logger.warning(
            'No USDA JSON files found in %s — writing empty usda_nutrients.parquet. '
            'Downstream balance_score will fall back to Food.com PDV values. '
            'To enable nutrient enrichment, download FDC bulk JSON files: %s',
            usda_dir,
            ', '.join(n for n, _ in _USDA_JSON_SPECS),
        )
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        empty_rows = [
            dict({'canonical_ingredient': ing, 'computed_date': date.today()},
                 **_empty_usda_nutrient_row())
            for ing in canonical_ingredients
        ]
        atomic_parquet(pd.DataFrame(empty_rows), USDA_NUTRIENTS_STAGING)
        context['ti'].xcom_push(key='usda_coverage_rate', value=0.0)
        context['ti'].xcom_push(key='usda_rows', value=len(canonical_ingredients))
        context['ti'].xcom_push(key='usda_extract_skipped', value=True)
        return

    desc_lowers = [str(f.get('description') or '').lower() for f in foods]
    logger.info(
        f'Looking up {len(canonical_ingredients):,} canonical ingredients against '
        f'{len(foods):,} local USDA foods.'
    )

    rows: list[dict[str, Any]] = []
    computed_date = date.today()

    for i, ingredient in enumerate(canonical_ingredients, start=1):
        food_match = _select_best_usda_food(ingredient, foods, desc_lowers)
        nutrient_row = (
            _nutrient_row_from_usda_food(food_match)
            if food_match is not None
            else _empty_usda_nutrient_row()
        )
        nutrient_row['canonical_ingredient'] = ingredient
        nutrient_row['computed_date'] = computed_date
        rows.append(nutrient_row)
        if i % 50 == 0 or i == len(canonical_ingredients):
            logger.info(f'USDA local lookup progress: {i}/{len(canonical_ingredients)}')

    usda_df = pd.DataFrame(rows)
    atomic_parquet(usda_df, USDA_NUTRIENTS_STAGING)

    coverage = int(usda_df['calories_per_100g'].notna().sum())
    coverage_rate = coverage / max(1, len(usda_df))
    logger.info(
        f'USDA extraction complete. Coverage(calories)={coverage:,}/{len(usda_df):,} '
        f'({coverage_rate:.1%}). Staged to {USDA_NUTRIENTS_STAGING}'
    )
    context['ti'].xcom_push(key='usda_coverage_rate', value=float(coverage_rate))
    context['ti'].xcom_push(key='usda_rows', value=int(len(usda_df)))
    context['ti'].xcom_push(key='usda_extract_skipped', value=False)


# ---------------------------------------------------------------------------
# Google Trends extraction
# ---------------------------------------------------------------------------


def extract_ai_mode(**context) -> None:
    """
    Calls the SerpAPI Google AI Mode engine for each seed in _AI_MODE_SEEDS,
    scores food/cuisine term frequency from the returned text blocks.

    Writes two staging files:
      ai_mode_raw.parquet          — one row per text block
      ai_mode_term_scores.parquet  — term frequencies + normalised scores

    SERPAPI_KEY must be set as an environment variable.
    """
    from foodcom_pipeline.batch.load import is_ai_mode_stale, save_ai_mode_metadata
    from foodcom_pipeline.extraction.ai_mode import (
        fetch_ai_mode_blocks,
        score_terms,
    )

    if not is_ai_mode_stale():
        logger.info('AI Mode data is fresh, skipping fetch.')
        return

    api_key = os.environ.get('SERPAPI_KEY')
    if not api_key:
        raise RuntimeError('SERPAPI_KEY environment variable not set')

    raw_df = fetch_ai_mode_blocks(_AI_MODE_SEEDS, api_key)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(raw_df, AI_MODE_RAW_STAGING)
    logger.info('AI Mode raw blocks staged: %d rows → %s', len(raw_df), AI_MODE_RAW_STAGING)

    if raw_df.empty:
        logger.warning('No AI Mode blocks returned; term scoring and merge skipped.')
        return

    term_scores = score_terms(raw_df)
    atomic_parquet(term_scores, AI_MODE_TERM_SCORES_STAGING)
    logger.info(
        'AI Mode term scores staged: %d terms → %s',
        len(term_scores), AI_MODE_TERM_SCORES_STAGING,
    )

    save_ai_mode_metadata(date.today())


def extract_google_trends(**context) -> None:
    """
    Extracts Google Trends related queries for _TRENDS_SEEDS and stages two
    Parquet files: google_trends_raw.parquet and google_trends_normalised.parquet.

    Skips seeds whose last fetch is within 7 days (checked via trends_metadata.parquet).
    Imports from load.py are deferred to avoid the circular extract ↔ load import.
    """
    # Deferred to avoid circular import: load.py imports STAGING_DIR from extract.py
    from foodcom_pipeline.batch.load import is_trends_stale, save_trends_metadata
    from foodcom_pipeline.extraction.trends import fetch_related_queries, batch_normalise

    stale_seeds = [s for s in _TRENDS_SEEDS if is_trends_stale(s)]
    if not stale_seeds:
        logger.info('Trends data is fresh, skipping fetch.')
        return

    raw_df = fetch_related_queries(stale_seeds)

    if raw_df.empty:
        logger.warning('No trend rows returned; skipping parquet write.')
        return

    raw_df = _clean_trend_queries(raw_df)

    if raw_df.empty:
        logger.warning('No food-relevant trend rows after filtering; skipping parquet write.')
        return

    normalised_df = batch_normalise(raw_df)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    atomic_parquet(raw_df, TRENDS_RAW_STAGING)
    atomic_parquet(normalised_df, TRENDS_NORMALISED_STAGING)
    logger.info('Trends staged: %d rows → %s', len(raw_df), STAGING_DIR)

    for seed in raw_df['seed_query'].unique():
        save_trends_metadata(seed, date.today())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clean_trend_queries(df: pd.DataFrame) -> pd.DataFrame:
    """Strip seed-word noise from related_query and drop non-food rows.

    Seeds are 'recipe'/'recipes', so every related query is polluted with
    those words (e.g. 'chicken recipe', 'easy recipe').  After stripping them,
    non-food residue like 'easy', 'box', or 'allrecipes' is removed by
    requiring a match against at least one keyword in TAG_PATTERNS.
    """
    import re as _re
    # Deferred to avoid circular import (tag_recipes → extract)
    from foodcom_pipeline.batch.tag_recipes import TAG_PATTERNS

    _noise = _re.compile(r'\brecipes?\b', flags=_re.IGNORECASE)

    df = df.copy()
    df['related_query'] = (
        df['related_query']
        .str.replace(_noise, '', regex=True)
        .str.strip()
    )
    df = df[df['related_query'] != '']

    def _is_food(query: str) -> bool:
        q = query.lower()
        return any(
            any(p.search(q) for p in patterns)
            for patterns in TAG_PATTERNS.values()
        )

    mask = df['related_query'].apply(_is_food)
    dropped = (~mask).sum()
    if dropped:
        logger.info('Trend query filter: dropped %d non-food rows.', dropped)
    return df[mask].reset_index(drop=True)


def _log_extraction_stats(name: str, df: pd.DataFrame) -> None:
    """Logs shape, dtypes and null counts — useful for the report's runtime table."""
    logger.info(f'--- {name.upper()} EXTRACTION STATS ---')
    logger.info(f'  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns')
    logger.info(f'  Columns    : {list(df.columns)}')
    logger.info(f'  Null counts:\n{df.isnull().sum().to_string()}')
    logger.info(f'  Memory     : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB')
