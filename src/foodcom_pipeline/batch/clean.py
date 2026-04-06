"""
clean.py
--------
Cleaning step for the Food.com batch pipeline.

Reads staged parquet files from extract.py, applies all cleaning and
normalization logic, and writes cleaned parquet files for the next stage
(sentiment.py / cluster.py).

Cleaning steps are divided into two categories:
  [DATASET]    — Issues observed in RAW_recipes.csv / RAW_interactions.csv
  [DEFENSIVE]  — Issues not present in the static dataset but expected in
                 a live system receiving new entries via the stream layer.
                 These are no-ops on the Kaggle data but protect the pipeline
                 in production.
"""

import ast
import logging
import re
from typing import Any

import pandas as pd
from foodcom_pipeline.batch.extract import (
    INGR_MAP_PKL,
    INTERACTIONS_STAGING,
    RECIPES_STAGING,
    STAGING_DIR,
    USDA_NUTRIENTS_STAGING,
    _load_ingr_map,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Staging paths
# ---------------------------------------------------------------------------

RECIPES_CLEAN = STAGING_DIR / 'recipes_clean.parquet'
INTERACTIONS_CLEAN = STAGING_DIR / 'interactions_clean.parquet'

# Nutrition fields as defined in the dataset documentation
NUTRITION_COLS = [
    'calories_pdv',
    'total_fat_pdv',
    'sugar_pdv',
    'sodium_pdv',
    'protein_pdv',
    'saturated_fat_pdv',
    'carbs_pdv',
]

# Thresholds for defensive cleaning — tune per business context
MIN_REVIEW_LENGTH = 2  # catches ".", "x", single emoji
MAX_REVIEWS_PER_USER_PER_DAY = 50  # spam/bot detection
MAX_MINUTES = 4320  # 3 days
MAX_CALORIES = 10000
MAX_RECIPE_NAME_LENGTH = 200


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_clean(**context) -> None:
    """
    Main cleaning entry point. Cleans both DataFrames independently,
    then writes cleaned parquet files for downstream tasks.
    """
    logger.info('Starting clean step.')

    recipes_df = pd.read_parquet(RECIPES_STAGING)
    interactions_df = pd.read_parquet(INTERACTIONS_STAGING)

    ingr_map: dict[str, str] | None = None
    if INGR_MAP_PKL.exists():
        try:
            ingr_map = _load_ingr_map(INGR_MAP_PKL)
            logger.info(f'Loaded ingr_map.pkl ({len(ingr_map):,} mappings).')
        except Exception as e:
            logger.warning(f'Could not load ingr_map.pkl: {e}')

    usda_nutrients = None
    if USDA_NUTRIENTS_STAGING.exists():
        try:
            usda_nutrients = pd.read_parquet(USDA_NUTRIENTS_STAGING)
            logger.info(
                f'Loaded USDA nutrients ({len(usda_nutrients):,} rows) from '
                f'{USDA_NUTRIENTS_STAGING}'
            )
        except Exception as e:
            logger.warning(f'Could not load {USDA_NUTRIENTS_STAGING}: {e}')

    recipes_clean = clean_recipes(recipes_df, ingr_map=ingr_map, usda_nutrients=usda_nutrients)
    interactions_clean = clean_interactions(interactions_df)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    recipes_clean.to_parquet(RECIPES_CLEAN, index=False)
    interactions_clean.to_parquet(INTERACTIONS_CLEAN, index=False)

    logger.info('Clean step complete.')
    logger.info(f'  Recipes    : {len(recipes_df):,} → {len(recipes_clean):,} rows')
    logger.info(
        f'  Interactions: {len(interactions_df):,} → {len(interactions_clean):,} rows'
    )

    context['ti'].xcom_push(key='recipes_clean_count', value=len(recipes_clean))
    context['ti'].xcom_push(
        key='interactions_clean_count', value=len(interactions_clean)
    )


# ---------------------------------------------------------------------------
# Interactions cleaning
# ---------------------------------------------------------------------------


def clean_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies all interaction cleaning steps in sequence.
    Logs a row count report after each step.
    """
    logger.info(f'Cleaning interactions: starting with {len(df):,} rows')
    original_count = len(df)

    # --- [DATASET] known issues in RAW_interactions.csv ---
    df = _parse_interaction_dates(df)
    df = _drop_invalid_ratings(df)
    df = _drop_duplicate_interactions(df)
    df = _enrich_date_fields(df)
    df = _drop_null_keys(df)

    # --- [DEFENSIVE] guards for live/new entries ---
    df = _drop_future_dates(df)
    df = _coerce_rating_type(df)
    df = _clean_review_text(df)
    df = _drop_implausibly_short_reviews(df)
    df = _flag_suspected_bots(df)
    df = _drop_bot_interactions(df)

    removed = original_count - len(df)
    logger.info(
        f'Interactions cleaning complete: '
        f'{len(df):,} rows retained, {removed:,} removed '
        f'({removed / original_count * 100:.1f}%)'
    )
    return df.reset_index(drop=True)


# ── [DATASET] Interactions ─────────────────────────────────────────────


def _parse_interaction_dates(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Coerce date column to datetime, dropping unparseable rows."""
    before = len(df)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    dropped = df['date'].isnull().sum()
    if dropped > 0:
        logger.warning(
            f'[DATASET] Dropped {dropped} interactions with unparseable dates.'
        )
    df = df.dropna(subset=['date'])
    logger.info(f'Date parsing: {before:,} → {len(df):,} rows')
    return df


def _drop_invalid_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Remove interactions with ratings outside the valid 1–5 range."""
    before = len(df)
    df = df[df['rating'].between(1, 5)]
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f'[DATASET] Dropped {dropped} interactions with invalid ratings.'
        )
    logger.info(f'Rating range filter: {before:,} → {len(df):,} rows')
    return df


def _drop_duplicate_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Remove duplicate (user_id, recipe_id, date) combinations."""
    before = len(df)
    df = df.drop_duplicates(subset=['user_id', 'recipe_id', 'date'], keep='first')
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f'[DATASET] Dropped {dropped} duplicate interactions.')
    logger.info(f'Deduplication: {before:,} → {len(df):,} rows')
    return df


def _enrich_date_fields(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Extract year, month, day_of_week, quarter for dim_date."""
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.day_name()
    df['quarter'] = df['date'].dt.quarter
    logger.info('[DATASET] Date enrichment complete.')
    return df


def _drop_null_keys(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Drop rows with null values in key foreign key columns."""
    before = len(df)
    df = df.dropna(subset=['user_id', 'recipe_id', 'date'])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(
            f'[DATASET] Dropped {dropped} interactions with null key fields.'
        )
    return df


# ── [DEFENSIVE] Interactions ──────────────────────────────────────────────


def _drop_future_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Drop interactions with dates in the future.

    Not present in the static Kaggle dataset (all historical), but a live
    system could receive malformed entries with future timestamps — e.g. from
    a client with a misconfigured clock or a bad Kafka producer.
    """
    before = len(df)
    now = pd.Timestamp.now()
    future_mask = df['date'] > now
    dropped = future_mask.sum()

    if dropped > 0:
        logger.warning(
            f"[DEFENSIVE] Dropped {dropped} interactions with future dates "
            f"(max seen: {df.loc[future_mask, 'date'].max()})"
        )
        df = df[~future_mask]

    logger.info(f'Future date filter: {before:,} → {len(df):,} rows')
    return df


def _coerce_rating_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Coerce rating to integer.

    A live API endpoint might accept float ratings (e.g. 4.0 or "3") from
    clients that don't enforce int types. We round to nearest int and
    re-validate. Entries that can't be coerced are dropped.
    """
    before_nulls = df['rating'].isnull().sum()
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating'] = df['rating'].round().astype('Int64')  # nullable int

    new_nulls = df['rating'].isnull().sum() - before_nulls
    if new_nulls > 0:
        logger.warning(
            f'[DEFENSIVE] {new_nulls} ratings could not be coerced to int — dropping.'
        )
        df = df.dropna(subset=['rating'])

    # Re-apply range validation after coercion
    df = df[df['rating'].between(1, 5)]
    return df


def _clean_review_text(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Normalize review text for a live system:
      - Strip leading/trailing whitespace
      - Collapse multiple consecutive whitespace characters
      - Remove null bytes (breaks Postgres text columns)
      - Strip HTML tags (user might paste formatted text from a browser)
      - Normalize unicode to NFC form (handles composed vs decomposed chars)

    None of these are needed for the Kaggle dataset, which has clean
    plain text throughout.
    """
    import unicodedata

    html_tag_pattern = re.compile(r'<[^>]+')
    null_byte_pattern = re.compile(r'\x00')

    def normalize(text):
        if not isinstance(text, str):
            return ''
        text = null_byte_pattern.sub('', text)
        text = html_tag_pattern.sub(' ', text)
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['review'] = df['review'].apply(normalize)
    logger.info('[DEFENSIVE] Review text normalization complete.')
    return df


def _drop_implausibly_short_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Null out reviews that are too short to carry any sentiment signal.

    In the Kaggle dataset all reviews have meaningful text. In a live system
    users might submit single-character reviews, accidental button presses,
    or lone emojis. These are useless for DistilBERT.
    We null the review text rather than dropping the row — the rating is
    still valid and should be retained.
    """
    short_mask = df['review'].str.len() < MIN_REVIEW_LENGTH
    short_count = short_mask.sum()

    if short_count > 0:
        logger.warning(
            f'[DEFENSIVE] {short_count} reviews below minimum length '
            f'({MIN_REVIEW_LENGTH} chars) — text set to null, rating retained.'
        )
        df.loc[short_mask, 'review'] = None

    return df


def _flag_suspected_bots(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Flag users submitting an implausibly high number of reviews
    per day — a strong signal for bot activity or automated scraping.

    Adds a boolean column `suspected_bot` rather than immediately dropping,
    so the decision can be changed without modifying this function.
    """
    reviews_per_user_per_day = (
        df.groupby(['user_id', df['date'].dt.date])
        .size()
        .reset_index(name='daily_review_count')
    )

    bot_users = reviews_per_user_per_day.loc[
        reviews_per_user_per_day['daily_review_count'] > MAX_REVIEWS_PER_USER_PER_DAY,
        'user_id',
    ].unique()

    df['suspected_bot'] = df['user_id'].isin(bot_users)

    if len(bot_users) > 0:
        logger.warning(
            f'[DEFENSIVE] {len(bot_users)} users flagged as suspected bots '
            f'(>{MAX_REVIEWS_PER_USER_PER_DAY} reviews/day).'
        )

    return df


def _drop_bot_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Drop interactions from suspected bots.

    Separated from _flag_suspected_bots so the policy can be changed
    independently — e.g. quarantine to a separate table instead of dropping.
    """
    before = len(df)
    df = df[~df['suspected_bot']].drop(columns=['suspected_bot'])
    dropped = before - len(df)

    if dropped > 0:
        logger.warning(
            f'[DEFENSIVE] Dropped {dropped} interactions from suspected bots.'
        )

    logger.info(f'Bot filter: {before:,} → {len(df):,} rows')
    return df


# ---------------------------------------------------------------------------
# Recipes cleaning
# ---------------------------------------------------------------------------


def clean_recipes(
    df: pd.DataFrame,
    *,
    ingr_map: dict[str, str] | None,
    usda_nutrients: pd.DataFrame | None,
) -> pd.DataFrame:
    logger.info(f'Cleaning recipes: starting with {len(df):,} rows')

    # --- [DATASET] ---
    df = _drop_null_recipe_ids(df)
    df = _parse_nutrition(df)
    df = _normalize_ingredients(df, ingr_map=ingr_map)
    df = _clean_minutes(df)

    # [USDA] Enrich canonical ingredients -> absolute per-100g nutrients
    df = _enrich_recipes_with_usda(df, usda_nutrients=usda_nutrients)

    # --- [DEFENSIVE] ---
    df = _drop_duplicate_recipe_ids(df)
    df = _clean_recipe_names(df)
    df = _cap_nutrition_values(df)
    df = _drop_negative_nutrition(df)
    df = _flag_nutrition_outliers(df)

    df = _compute_balance_score(df)

    logger.info(f'Recipes cleaning complete: {len(df):,} rows retained')
    return df.reset_index(drop=True)


# ── [DATASET] Recipes ────────────────────────────────────────────────────────────────────


def _drop_null_recipe_ids(df: pd.DataFrame) -> pd.DataFrame:
    """[DATASET] Drop recipes with null IDs."""
    before = len(df)
    df = df.dropna(subset=['id'])
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f'[DATASET] Dropped {dropped} recipes with null IDs.')
    return df


def _parse_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DATASET] Parse the nutrition column from string list representation
    into individual float columns.
    e.g. '[490.4, 34.0, 43.0, 21.0, 38.0, 51.0, 11.0]'
    """

    def parse_nutrition_str(value):
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list) and len(parsed) == 7:
                return [float(x) if x is not None else None for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return [None] * 7

    nutrition_parsed = df['nutrition'].apply(parse_nutrition_str).tolist()
    nutrition_df = pd.DataFrame(nutrition_parsed, columns=NUTRITION_COLS)
    df = pd.concat([df.drop(columns=['nutrition']), nutrition_df], axis=1)

    null_nutrition = nutrition_df.isnull().any(axis=1).sum()
    if null_nutrition > 0:
        logger.warning(
            f'[DATASET] {null_nutrition} recipes had unparseable nutrition data.'
        )

    logger.info('[DATASET] Nutrition parsing complete.')
    return df


def _load_ingr_map(path) -> dict[str, str]:
    def _pickle_load_with_pandas_shim(fh):
        try:
            return pickle.load(fh)
        except ModuleNotFoundError as e:
            if "pandas.core.indexes.numeric" not in str(e):
                raise
            import pandas as _pd

            shim = types.ModuleType("pandas.core.indexes.numeric")
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
            if v is None:
                continue
            out[str(k).strip().lower()] = str(v).strip().lower()
        return out

    # Kaggle's ingr_map.pkl is often a DataFrame with raw ingredient variants
    # mapped to a canonical replacement string (usually in "replaced").
    if isinstance(obj, pd.DataFrame):
        cols = {c.lower(): c for c in obj.columns}
        raw_col = cols.get('raw_ingr') or cols.get('raw_ingredient') or cols.get('raw')
        canon_col = (
            cols.get('replaced') or cols.get('canonical') or cols.get('canonical_form')
        )

        if raw_col and canon_col:
            out: dict[str, str] = {}
            for _, row in obj.iterrows():
                raw = row.get(raw_col)
                canon = row.get(canon_col)
                if pd.notnull(raw) and pd.notnull(canon):
                    out[str(raw).strip().lower()] = str(canon).strip().lower()
            if out:
                return out

        raise TypeError(
            'Unsupported ingr_map.pkl DataFrame shape. '
            f'Columns found: {list(obj.columns)}'
        )

    raise TypeError(f'Unsupported ingr_map.pkl structure: {type(obj)!r}')


def _normalize_ingredients(
    df: pd.DataFrame, *, ingr_map: dict[str, str] | None
) -> pd.DataFrame:
    """
    [DATASET] Parse and normalize ingredient strings.
    Lowercases, strips whitespace, stores as pipe-separated string.
    """

    def parse_and_normalize(value):
        try:
            ingredients = ast.literal_eval(value)
            if isinstance(ingredients, list):
                return [re.sub(r'\s+', ' ', ing.lower().strip()) for ing in ingredients]
        except (ValueError, SyntaxError):
            pass
        return []

    df['ingredients_parsed'] = df['ingredients'].apply(parse_and_normalize)
    df['ingredients_normalized'] = df['ingredients_parsed'].apply('|'.join)
    df['ingredient_count'] = df['ingredients_parsed'].apply(len)

    if ingr_map:
        # Map normalized ingredient strings onto canonical forms required
        # for consistent USDA lookup, and log fallback behavior.
        mapped_count = 0
        fallback_count = 0
        fallback_samples: list[str] = []

        def _map_ingredient_list(lst):
            nonlocal mapped_count, fallback_count, fallback_samples
            if not isinstance(lst, list):
                return []

            out = []
            for ing in lst:
                canon = ingr_map.get(ing)
                if canon is None:
                    fallback_count += 1
                    if len(fallback_samples) < 20:
                        fallback_samples.append(ing)
                    out.append(ing)
                else:
                    mapped_count += 1
                    out.append(canon)
            return out

        df['ingredients_canonical_list'] = df['ingredients_parsed'].apply(
            _map_ingredient_list
        )

        total = mapped_count + fallback_count
        if total > 0:
            logger.info(
                'Ingredient canonical mapping: mapped=%s fallback=%s fallback_rate=%.1f%%',
                f'{mapped_count:,}',
                f'{fallback_count:,}',
                100.0 * fallback_count / total,
            )
        if fallback_count > 0:
            logger.info("some ingredients not found in ingr_map")
            logger.info(
                'Fallback ingredients not found in ingr_map (sample): %s',
                sorted(set(fallback_samples)),
            )
    else:
        df['ingredients_canonical_list'] = df['ingredients_parsed']

    df['ingredients_canonical_normalized'] = df['ingredients_canonical_list'].apply('|'.join)

    empty = (df['ingredient_count'] == 0).sum()
    if empty > 0:
        logger.warning(f'[DATASET] {empty} recipes have no parseable ingredients.')

    logger.info(
        f"[DATASET] Ingredient normalization complete. "
        f"Avg per recipe: {df['ingredient_count'].mean():.1f}"
    )
    return df


def _enrich_recipes_with_usda(
    df: pd.DataFrame, *, usda_nutrients: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Joins each recipe's canonical ingredient list with USDA nutrient values.

    Because recipes don't include ingredient quantities, we treat each ingredient
    as a 100g-equivalent proxy and use the *mean* nutrient value across the
    canonical ingredients as a pragmatic estimate for recipe-level nutrition.
    """
    # If USDA data isn't available yet, compute balance_score from PDV later.
    if usda_nutrients is None or df.empty:
        for col in [
            'calories',
            'protein',
            'fat',
            'saturated_fat',
            'sugar',
            'sodium',
            'carbs',
        ]:
            if col not in df.columns:
                df[col] = None
        return df

    required_usda_cols = {
        'canonical_ingredient',
        'calories_per_100g',
        'protein_g_per_100g',
        'fat_g_per_100g',
        'saturated_fat_g_per_100g',
        'sugar_g_per_100g',
        'sodium_g_per_100g',
        'carbs_g_per_100g',
    }
    missing_cols = required_usda_cols - set(usda_nutrients.columns)
    if missing_cols:
        raise ValueError(f'USDA nutrients missing expected columns: {missing_cols}')

    exploded = df[['id', 'ingredients_canonical_list']].explode(
        'ingredients_canonical_list'
    )
    exploded = exploded.rename(
        columns={'ingredients_canonical_list': 'canonical_ingredient'}
    )
    exploded = exploded.dropna(subset=['canonical_ingredient'])

    merged = exploded.merge(
        usda_nutrients,
        on='canonical_ingredient',
        how='left',
    )

    agg = (
        merged.groupby('id')
        .agg(
            calories=('calories_per_100g', 'mean'),
            protein=('protein_g_per_100g', 'mean'),
            fat=('fat_g_per_100g', 'mean'),
            saturated_fat=('saturated_fat_g_per_100g', 'mean'),
            sugar=('sugar_g_per_100g', 'mean'),
            sodium=('sodium_g_per_100g', 'mean'),
            carbs=('carbs_g_per_100g', 'mean'),
            usda_matched_ingredients=('calories_per_100g', lambda s: s.notna().sum()),
        )
        .reset_index()
    )

    df = df.merge(agg, on='id', how='left')

    # Recipe-level USDA ingredient coverage proxy.
    df['usda_coverage_rate'] = (
        df['usda_matched_ingredients'] / df['ingredient_count']
    ).fillna(0.0)

    return df


def _compute_balance_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a bounded recipe balance_score using nutrient risk weighting:
      balance_score = 0.4*protein_norm
                       -0.25*saturated_fat_norm
                       -0.2*sugar_norm
                       -0.15*sodium_norm
    clipped to [-1, 1].
    """
    # Prefer USDA-derived absolute nutrients when present; otherwise fall back
    # to Food.com PDV-derived percentages.
    def _norm_from_usda(col_abs: str, col_pdv: str) -> pd.Series:
        if col_abs in df.columns and df[col_abs].notna().any():
            denom = float(df[col_abs].quantile(0.95))
            if denom and denom > 0:
                return (df[col_abs] / denom).clip(0, 1)
        if col_pdv in df.columns and df[col_pdv].notna().any():
            return (df[col_pdv] / 100.0).clip(0, 1)
        return pd.Series([0.0] * len(df), index=df.index)

    protein_norm = _norm_from_usda('protein', 'protein_pdv')
    sat_fat_norm = _norm_from_usda('saturated_fat', 'saturated_fat_pdv')
    sugar_norm = _norm_from_usda('sugar', 'sugar_pdv')
    sodium_norm = _norm_from_usda('sodium', 'sodium_pdv')

    df['balance_score'] = (
        protein_norm * 0.4
        - sat_fat_norm * 0.25
        - sugar_norm * 0.2
        - sodium_norm * 0.15
    ).clip(-1, 1)

    return df

def _clean_minutes(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DATASET] Cap extreme cook times and null out zero/negative values.
    The Kaggle dataset contains known outliers (e.g. minutes=1,000,000).
    """
    extreme = (df['minutes'] > MAX_MINUTES).sum()
    if extreme > 0:
        logger.warning(
            f'[DATASET] Capping {extreme} recipes with cook time > {MAX_MINUTES} min.'
        )
        df['minutes'] = df['minutes'].clip(upper=MAX_MINUTES)

    invalid = (df['minutes'] <= 0).sum()
    if invalid > 0:
        logger.warning(f'[DATASET] Nulling {invalid} recipes with minutes <= 0.')
        df.loc[df['minutes'] <= 0, 'minutes'] = None

    return df


# ── [DEFENSIVE] Recipes ───────────────────────────────────────────────────────────────────


def _drop_duplicate_recipe_ids(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Drop recipes with duplicate IDs.

    The Kaggle dataset has unique IDs. A live system could receive the same
    recipe twice — e.g. from a retry on a failed API call or a double-submit
    from the frontend.
    """
    before = len(df)
    df = df.drop_duplicates(subset=['id'], keep='first')
    dropped = before - len(df)
    if dropped > 0:
        logger.warning(f'[DEFENSIVE] Dropped {dropped} recipes with duplicate IDs.')
    logger.info(f'Recipe ID deduplication: {before:,} → {len(df):,} rows')
    return df


def _clean_recipe_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Normalize recipe names and drop entries that are empty,
    purely special characters, or excessively long.

    In the Kaggle dataset all names are clean strings. A live system could
    receive names that are null, only whitespace, only symbols ("!!!"),
    contain null bytes, or are SEO-spam length strings.
    """
    import unicodedata

    null_byte_pattern = re.compile(r'\x00')

    def normalize_name(name):
        if not isinstance(name, str):
            return None
        name = null_byte_pattern.sub('', name)
        name = unicodedata.normalize('NFC', name)
        name = re.sub(r'\s+', ' ', name).strip()
        return name if len(name) > 0 else None

    before = len(df)
    df['name'] = df['name'].apply(normalize_name)

    long_mask = df['name'].str.len() > MAX_RECIPE_NAME_LENGTH
    if long_mask.sum() > 0:
        logger.warning(
            f'[DEFENSIVE] Truncating {long_mask.sum()} recipe names '
            f'exceeding {MAX_RECIPE_NAME_LENGTH} characters.'
        )
        df.loc[long_mask, 'name'] = df.loc[long_mask, 'name'].str[
            :MAX_RECIPE_NAME_LENGTH
        ]

    df = df.dropna(subset=['name'])

    only_special = ~df['name'].str.contains(r'[a-zA-Z0-9]', regex=True, na=False)
    if only_special.sum() > 0:
        logger.warning(
            f'[DEFENSIVE] Dropped {only_special.sum()} recipes with '
            f'names containing no alphanumeric characters.'
        )
        df = df[~only_special]

    logger.info(f'Recipe name cleaning: {before:,} → {len(df):,} rows')
    return df


def _cap_nutrition_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Cap calorie values at a sensible maximum.

    The Kaggle dataset has high but plausible values. A live system where
    users enter nutrition manually could have data entry errors
    (e.g. calories=999999). Cap rather than drop to preserve the recipe.
    """
    if 'calories_pdv' not in df.columns:
        return df

    extreme = (df['calories_pdv'] > MAX_CALORIES).sum()
    if extreme > 0:
        logger.warning(
            f'[DEFENSIVE] Capping {extreme} recipes with calories_pdv > {MAX_CALORIES}.'
        )
        df['calories_pdv'] = df['calories_pdv'].clip(upper=MAX_CALORIES)

    return df


def _flag_nutrition_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DATASET] Flags recipes where any PDV nutritional value exceeds the 95th percentile.

    Adds a boolean `nutrition_outlier` column — the recipe is retained but should be
    excluded from nutritional benchmarking and balance_score comparisons.
    """
    outlier_mask = pd.Series(False, index=df.index)
    for col in NUTRITION_COLS:
        if col not in df.columns:
            continue
        threshold = df[col].quantile(0.95)
        outlier_mask |= df[col] > threshold

    df['nutrition_outlier'] = outlier_mask
    n = int(outlier_mask.sum())
    logger.info(
        f'[DATASET] Nutrition outlier flagging: {n:,} recipes ({n / len(df) * 100:.1f}%) '
        f'exceed the 95th percentile on at least one PDV column.'
    )
    return df


def _drop_negative_nutrition(df: pd.DataFrame) -> pd.DataFrame:
    """
    [DEFENSIVE] Set negative nutrition values to null.

    Nutritional values are always non-negative by definition. Negative values
    indicate a data entry or parsing error and must not reach the warehouse.
    """
    for col in NUTRITION_COLS:
        if col not in df.columns:
            continue
        negative_mask = df[col] < 0
        count = negative_mask.sum()
        if count > 0:
            logger.warning(
                f"[DEFENSIVE] Setting {count} negative values in '{col}' to null."
            )
            df.loc[negative_mask, col] = None

    return df


# ---------------------------------------------------------------------------
# Utility: load cleaned data (used by downstream steps)
# ---------------------------------------------------------------------------


def load_cleaned_interactions() -> pd.DataFrame:
    return pd.read_parquet(INTERACTIONS_CLEAN)


def load_cleaned_recipes() -> pd.DataFrame:
    return pd.read_parquet(RECIPES_CLEAN)
