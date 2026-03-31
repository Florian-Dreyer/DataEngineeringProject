"""
load.py
-------
Load step for the Food.com batch pipeline.

Upserts cleaned, sentiment-scored, and clustered data into the PostgreSQL
star schema. All writes use INSERT ... ON CONFLICT DO UPDATE (upsert) so
the step is fully idempotent — rerunning the same DAG twice produces the
same result.

Load order matters due to foreign key constraints:
  1. dim_date                              (no FK dependencies)
  2. dim_recipe                            (no FK dependencies)
  3. dim_canonical_ingredient_nutrients   (no FK dependencies; from staging parquet)
  4. dim_user                              (no FK dependencies)
  5. fact_interactions                     (FKs → dim_date, dim_recipe, dim_user)

Star schema (from project plan):
  fact_interactions : interaction_id, user_id, recipe_id, date_id, rating,
                      sentiment_score, rating_sentiment_gap, source
  dim_user          : user_id, review_count, avg_rating_given, recipe_diversity,
                      cluster_id, cluster_label, avg_sentiment_score,
                      avg_rating_gap, std_rating_given, pct_positive_sentiment,
                      active_days, first_review_date, last_review_date
  dim_recipe        : recipe_id, name, avg_rating, review_count,
                      avg_cook_minutes, top_ingredients, tags
  dim_date          : date_id, full_date, year, month, day_of_week, quarter
"""

import hashlib
import logging
import os
from datetime import date

import pandas as pd
from foodcom_pipeline.batch.aggregate_user_stats import load_user_stats
from foodcom_pipeline.batch.clean import (
    load_cleaned_recipes,
)
from foodcom_pipeline.batch.cluster import load_user_clusters
from foodcom_pipeline.batch.extract import USDA_NUTRIENTS_STAGING
from foodcom_pipeline.batch.sentiment import load_sentiment_interactions
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POSTGRES_CONN = os.getenv(
    'FOODCOM_POSTGRES_CONN', 'postgresql://user:password@postgres:5432/foodcom'
)

# Batch size for bulk upserts — avoids building a single enormous SQL statement
UPSERT_BATCH_SIZE = 5_000

# Bayesian shrinkage strength for recipe-level sentiment_rating.
# Higher m => more shrinkage toward global mean for low-sample recipes.
SENTIMENT_SHRINKAGE_M = float(os.getenv('FOODCOM_SENTIMENT_SHRINKAGE_M', '100'))

# Non-PK columns on dim_recipe that loaders expect (legacy DBs may lack some).
_DIM_RECIPE_NON_PK_COLUMNS: tuple[tuple[str, str], ...] = (
    ('name', 'TEXT'),
    ('avg_rating', 'DOUBLE PRECISION'),
    ('review_count', 'INTEGER'),
    ('sentiment_rating', 'DOUBLE PRECISION'),
    ('avg_cook_minutes', 'DOUBLE PRECISION'),
    ('top_ingredients', 'TEXT'),
    ('tags', 'TEXT'),
    ('ingredient_count', 'INTEGER'),
    ('calories', 'DOUBLE PRECISION'),
    ('protein', 'DOUBLE PRECISION'),
    ('fat', 'DOUBLE PRECISION'),
    ('sugar', 'DOUBLE PRECISION'),
    ('sodium', 'DOUBLE PRECISION'),
    ('carbs', 'DOUBLE PRECISION'),
    ('saturated_fat', 'DOUBLE PRECISION'),
    ('balance_score', 'DOUBLE PRECISION'),
)


def _ensure_dim_recipe_columns(conn) -> None:
    """Adds any missing dim_recipe columns so upserts match the current code."""
    for col, pg_type in _DIM_RECIPE_NON_PK_COLUMNS:
        conn.execute(
            text(
                f'ALTER TABLE public.dim_recipe ADD COLUMN IF NOT EXISTS {col} {pg_type}'
            )
        )


_DIM_CANONICAL_INGREDIENT_NON_PK_COLUMNS: tuple[tuple[str, str], ...] = (
    ('computed_date', 'DATE'),
    ('calories_per_100g', 'DOUBLE PRECISION'),
    ('protein_g_per_100g', 'DOUBLE PRECISION'),
    ('fat_g_per_100g', 'DOUBLE PRECISION'),
    ('saturated_fat_g_per_100g', 'DOUBLE PRECISION'),
    ('sugar_g_per_100g', 'DOUBLE PRECISION'),
    ('sodium_g_per_100g', 'DOUBLE PRECISION'),
    ('carbs_g_per_100g', 'DOUBLE PRECISION'),
)


def _ensure_dim_canonical_ingredient_nutrients_columns(conn) -> None:
    """Adds any missing columns on dim_canonical_ingredient_nutrients."""
    for col, pg_type in _DIM_CANONICAL_INGREDIENT_NON_PK_COLUMNS:
        conn.execute(
            text(
                f'ALTER TABLE public.dim_canonical_ingredient_nutrients '
                f'ADD COLUMN IF NOT EXISTS {col} {pg_type}'
            )
        )


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_load(**context) -> None:
    """
    Main load entry point. Loads all staging data into the star schema
    in dependency order.
    """
    logger.info('Starting load step.')

    engine = create_engine(POSTGRES_CONN)

    _ensure_schema(engine)

    interactions = load_sentiment_interactions()
    recipes = load_cleaned_recipes()
    user_stats = load_user_stats()
    user_clusters = load_user_clusters()

    # Merge cluster labels into user stats
    users = user_stats.merge(
        user_clusters[['user_id', 'cluster_id', 'cluster_label']],
        on='user_id',
        how='left',
    )

    # Build dimension tables first, then facts
    date_map = _load_dim_date(engine, interactions)
    _load_dim_recipe(engine, recipes, interactions)
    _load_dim_canonical_ingredient_nutrients(engine)
    _load_dim_user(engine, users)
    _load_fact_interactions(engine, interactions, date_map)

    _log_row_counts(engine)

    logger.info('Load step complete.')

    # Push summary to XCom
    with engine.connect() as conn:
        n_facts = conn.execute(text('SELECT COUNT(*) FROM fact_interactions')).scalar()
    context['ti'].xcom_push(key='fact_interactions_total', value=int(n_facts))


# ---------------------------------------------------------------------------
# Schema creation (idempotent)
# ---------------------------------------------------------------------------


def _ensure_schema(engine) -> None:
    """
    Creates all star schema tables if they don't already exist.
    Safe to run on every DAG execution.
    """
    ddl = """
    CREATE TABLE IF NOT EXISTS dim_date (
        date_id     SERIAL PRIMARY KEY,
        full_date   DATE        NOT NULL UNIQUE,
        year        SMALLINT    NOT NULL,
        month       SMALLINT    NOT NULL,
        day_of_week VARCHAR(10) NOT NULL,
        quarter     SMALLINT    NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dim_recipe (
        recipe_id        BIGINT PRIMARY KEY,
        name             TEXT,
        avg_rating       FLOAT,
        review_count     INTEGER,
        sentiment_rating FLOAT,
        avg_cook_minutes FLOAT,
        top_ingredients  TEXT,    -- pipe-separated top 10 ingredients
        tags             TEXT,
        ingredient_count INTEGER,
        calories         FLOAT,
        protein          FLOAT,
        fat              FLOAT,
        sugar            FLOAT,
        sodium           FLOAT,
        carbs            FLOAT,
        saturated_fat    FLOAT,
        balance_score    FLOAT
    );

    CREATE TABLE IF NOT EXISTS dim_canonical_ingredient_nutrients (
        canonical_ingredient     TEXT PRIMARY KEY,
        computed_date             DATE,
        calories_per_100g         DOUBLE PRECISION,
        protein_g_per_100g        DOUBLE PRECISION,
        fat_g_per_100g            DOUBLE PRECISION,
        saturated_fat_g_per_100g  DOUBLE PRECISION,
        sugar_g_per_100g          DOUBLE PRECISION,
        sodium_g_per_100g         DOUBLE PRECISION,
        carbs_g_per_100g          DOUBLE PRECISION
    );

    CREATE TABLE IF NOT EXISTS dim_user (
        user_id                 BIGINT PRIMARY KEY,
        review_count            INTEGER,
        avg_rating_given        FLOAT,
        std_rating_given        FLOAT,
        recipe_diversity        INTEGER,
        avg_sentiment_score     FLOAT,
        avg_rating_gap          FLOAT,
        pct_positive_sentiment  FLOAT,
        active_days             INTEGER,
        first_review_date       DATE,
        last_review_date        DATE,
        cluster_id              SMALLINT,
        cluster_label           VARCHAR(50)
    );

    CREATE TABLE IF NOT EXISTS fact_interactions (
        interaction_id        BIGINT PRIMARY KEY,
        user_id               BIGINT   REFERENCES dim_user(user_id),
        recipe_id             BIGINT   REFERENCES dim_recipe(recipe_id),
        date_id               INTEGER  REFERENCES dim_date(date_id),
        rating                SMALLINT,
        sentiment_score       FLOAT,
        rating_normalized     FLOAT,
        rating_sentiment_gap  FLOAT,
        source                VARCHAR(10) DEFAULT 'batch'
    );

    CREATE TABLE IF NOT EXISTS recent_interactions (
        interaction_id        BIGINT PRIMARY KEY,
        user_id               BIGINT,
        recipe_id             BIGINT,
        date_id               INTEGER,
        rating                SMALLINT,
        sentiment_score       FLOAT,
        rating_normalized     FLOAT,
        rating_sentiment_gap  FLOAT,
        source                VARCHAR(10) DEFAULT 'stream'
    );

    -- Serving layer view: union of batch fact table and unprocessed stream records
    CREATE OR REPLACE VIEW serving_interactions AS
        SELECT * FROM fact_interactions
        UNION ALL
        SELECT * FROM recent_interactions
        WHERE interaction_id NOT IN (SELECT interaction_id FROM fact_interactions);
    """

    with engine.begin() as conn:
        conn.execute(text(ddl))

    # one transaction each so a failure does not poison
    # _ensure_dim_recipe_columns / _ensure_dim_canonical_ingredient_nutrients_columns.


    with engine.begin() as conn:
        _ensure_dim_recipe_columns(conn)
        _ensure_dim_canonical_ingredient_nutrients_columns(conn)

    logger.info('Schema ensured (tables and serving view created if not existing).')


# ---------------------------------------------------------------------------
# dim_date
# ---------------------------------------------------------------------------


def _load_dim_date(engine, interactions: pd.DataFrame) -> dict[date, int]:
    """
    Upserts all unique dates from the interactions into dim_date.
    Returns a mapping of {full_date: date_id} for use when loading fact_interactions.
    """
    unique_dates = interactions[
        ['date', 'year', 'month', 'day_of_week', 'quarter']
    ].drop_duplicates('date')
    unique_dates = unique_dates.rename(columns={'date': 'full_date'})
    unique_dates['full_date'] = unique_dates['full_date'].dt.date

    upsert_sql = """
        INSERT INTO dim_date (full_date, year, month, day_of_week, quarter)
        VALUES (:full_date, :year, :month, :day_of_week, :quarter)
        ON CONFLICT (full_date) DO NOTHING
    """

    _bulk_upsert(engine, unique_dates, upsert_sql)
    logger.info(f'dim_date: upserted {len(unique_dates):,} dates.')

    # Load back the full date → date_id mapping
    date_map_df = pd.read_sql('SELECT date_id, full_date FROM dim_date', engine)
    date_map_df['full_date'] = pd.to_datetime(date_map_df['full_date']).dt.date
    date_map = dict(zip(date_map_df['full_date'], date_map_df['date_id']))

    return date_map


# ---------------------------------------------------------------------------
# dim_recipe
# ---------------------------------------------------------------------------


def _load_dim_recipe(engine, recipes: pd.DataFrame, interactions: pd.DataFrame) -> None:
    """
    Upserts recipe dimension records.
    Computes avg_rating and review_count from interactions so these
    reflect the current batch's data.
    """
    with engine.begin() as conn:
        _ensure_dim_recipe_columns(conn)

    # Compute recipe-level aggregates from interactions
    recipe_agg = (
        interactions.groupby('recipe_id')
        .agg(
            avg_rating=('rating', 'mean'),
            review_count=('rating', 'count'),
        )
        .reset_index()
        .round({'avg_rating': 4})
    )

    # Compute sentiment_rating per recipe using inverse-frequency weighting + Bayesian shrinkage:
    # sentiment_rating = (Σ(w_i * r_i) + m*C) / (Σw_i + m)
    # where w_i = 1 / global_count[rating_i], r_i = sentiment_score, C = global mean sentiment_score.
    sentiment_df = interactions[['recipe_id', 'rating', 'sentiment_score']].copy()
    sentiment_df = sentiment_df.dropna(subset=['sentiment_score', 'rating'])
    if not sentiment_df.empty:
        rating_counts = sentiment_df['rating'].value_counts().to_dict()
        sentiment_df['w'] = sentiment_df['rating'].map(
            lambda r: 1.0 / float(rating_counts.get(r, 1))
        )
        C = float(sentiment_df['sentiment_score'].mean())
        m = SENTIMENT_SHRINKAGE_M

        recipe_sent = (
            sentiment_df.assign(wr=sentiment_df['w'] * sentiment_df['sentiment_score'])
            .groupby('recipe_id')
            .agg(weighted_sum=('wr', 'sum'), weight_sum=('w', 'sum'))
            .reset_index()
        )
        recipe_sent['sentiment_rating'] = (
            (recipe_sent['weighted_sum'] + m * C) / (recipe_sent['weight_sum'] + m)
        ).round(4)
        recipe_sent = recipe_sent[['recipe_id', 'sentiment_rating']]
    else:
        recipe_sent = pd.DataFrame(columns=['recipe_id', 'sentiment_rating'])

    # Join with recipe metadata without creating duplicate recipe_id columns.
    recipe_agg = recipe_agg.rename(columns={'recipe_id': 'id'})
    dim = recipes.merge(recipe_agg, on='id', how='left')
    dim = dim.merge(recipe_sent, left_on='id', right_on='recipe_id', how='left')
    dim = dim.drop(columns=['recipe_id'], errors='ignore')

    # avg_cook_minutes from the recipes table
    dim = dim.rename(
        columns={
            'id': 'recipe_id',
            'minutes': 'avg_cook_minutes',
        }
    )

    # Defensive type coercion for DB compatibility
    # (avoid floats like 137739.0 being bound into integer columns)
    dim['recipe_id'] = pd.to_numeric(dim['recipe_id'], errors='coerce').astype('Int64')
    dim['review_count'] = pd.to_numeric(dim['review_count'], errors='coerce').astype(
        'Int64'
    )
    dim['ingredient_count'] = pd.to_numeric(
        dim['ingredient_count'], errors='coerce'
    ).astype('Int64')

    # Defensive coercion for USDA-derived float columns.
    for col in [
        'calories',
        'protein',
        'fat',
        'sugar',
        'sodium',
        'carbs',
        'saturated_fat',
        'balance_score',
    ]:
        if col in dim.columns:
            dim[col] = pd.to_numeric(dim[col], errors='coerce')

    # Top ingredients: first 10 from canonical pipe-separated string when available.
    top_src = (
        'ingredients_canonical_normalized'
        if 'ingredients_canonical_normalized' in dim.columns
        else 'ingredients_normalized'
    )
    dim['top_ingredients'] = dim[top_src].apply(
        lambda x: '|'.join(x.split('|')[:10]) if isinstance(x, str) else None
    )

    # Ensure column names are unique before row mapping/upsert.
    dim = dim.loc[:, ~dim.columns.duplicated()]

    # Keep null tags as NULL (not literal "nan"), stringify real values.
    dim['tags'] = dim['tags'].where(pd.notnull(dim['tags']), None)
    dim['tags'] = dim['tags'].apply(lambda v: str(v) if v is not None else None)

    # Select and rename to match schema
    # Ensure nutrient columns exist even if USDA enrichment was skipped.
    for col in [
        'calories',
        'protein',
        'fat',
        'sugar',
        'sodium',
        'carbs',
        'saturated_fat',
        'balance_score',
    ]:
        if col not in dim.columns:
            dim[col] = None

    dim = dim[
        [
            'recipe_id',
            'name',
            'avg_rating',
            'review_count',
            'sentiment_rating',
            'avg_cook_minutes',
            'top_ingredients',
            'tags',
            'ingredient_count',
            'calories',
            'protein',
            'fat',
            'sugar',
            'sodium',
            'carbs',
            'saturated_fat',
            'balance_score',
        ]
    ].drop_duplicates('recipe_id')

    upsert_sql = """
        INSERT INTO dim_recipe (
            recipe_id, name, avg_rating, review_count, sentiment_rating,
            avg_cook_minutes, top_ingredients, tags, ingredient_count,
            calories, protein, fat, sugar, sodium, carbs, saturated_fat, balance_score
        )
        VALUES (
            :recipe_id, :name, :avg_rating, :review_count, :sentiment_rating,
            :avg_cook_minutes, :top_ingredients, :tags, :ingredient_count,
            :calories, :protein, :fat, :sugar, :sodium, :carbs, :saturated_fat, :balance_score
        )
        ON CONFLICT (recipe_id) DO UPDATE SET
            avg_rating       = EXCLUDED.avg_rating,
            review_count     = EXCLUDED.review_count,
            sentiment_rating = EXCLUDED.sentiment_rating,
            avg_cook_minutes = EXCLUDED.avg_cook_minutes,
            top_ingredients  = EXCLUDED.top_ingredients,
            tags             = EXCLUDED.tags,
            ingredient_count = EXCLUDED.ingredient_count,
            calories         = EXCLUDED.calories,
            protein          = EXCLUDED.protein,
            fat              = EXCLUDED.fat,
            sugar            = EXCLUDED.sugar,
            sodium           = EXCLUDED.sodium,
            carbs            = EXCLUDED.carbs,
            saturated_fat    = EXCLUDED.saturated_fat,
            balance_score    = EXCLUDED.balance_score
    """

    _bulk_upsert(engine, dim, upsert_sql)
    logger.info(f'dim_recipe: upserted {len(dim):,} recipes.')


# ---------------------------------------------------------------------------
# dim_canonical_ingredient_nutrients
# ---------------------------------------------------------------------------


def _load_dim_canonical_ingredient_nutrients(engine) -> None:
    """
    Upserts per-canonical-ingredient USDA nutrient rows from staging parquet
    (one row per distinct canonical string from ingr_map / extract_usda_nutrients).
    """
    if not USDA_NUTRIENTS_STAGING.is_file():
        logger.warning(
            'No %s; skipping dim_canonical_ingredient_nutrients load.',
            USDA_NUTRIENTS_STAGING,
        )
        return

    df = pd.read_parquet(USDA_NUTRIENTS_STAGING)
    expected = [
        'canonical_ingredient',
        'computed_date',
        'calories_per_100g',
        'protein_g_per_100g',
        'fat_g_per_100g',
        'saturated_fat_g_per_100g',
        'sugar_g_per_100g',
        'sodium_g_per_100g',
        'carbs_g_per_100g',
    ]
    missing = set(expected) - set(df.columns)
    if missing:
        raise ValueError(
            f'USDA nutrients parquet missing columns {missing}: {USDA_NUTRIENTS_STAGING}'
        )

    df = df[expected].drop_duplicates(subset=['canonical_ingredient'], keep='first')
    if df.empty:
        logger.info('dim_canonical_ingredient_nutrients: parquet has no rows to upsert.')
        return

    df['computed_date'] = pd.to_datetime(df['computed_date']).dt.date

    with engine.begin() as conn:
        _ensure_dim_canonical_ingredient_nutrients_columns(conn)

    upsert_sql = """
        INSERT INTO dim_canonical_ingredient_nutrients (
            canonical_ingredient, computed_date,
            calories_per_100g, protein_g_per_100g, fat_g_per_100g,
            saturated_fat_g_per_100g, sugar_g_per_100g, sodium_g_per_100g, carbs_g_per_100g
        )
        VALUES (
            :canonical_ingredient, :computed_date,
            :calories_per_100g, :protein_g_per_100g, :fat_g_per_100g,
            :saturated_fat_g_per_100g, :sugar_g_per_100g, :sodium_g_per_100g, :carbs_g_per_100g
        )
        ON CONFLICT (canonical_ingredient) DO UPDATE SET
            computed_date = EXCLUDED.computed_date,
            calories_per_100g = EXCLUDED.calories_per_100g,
            protein_g_per_100g = EXCLUDED.protein_g_per_100g,
            fat_g_per_100g = EXCLUDED.fat_g_per_100g,
            saturated_fat_g_per_100g = EXCLUDED.saturated_fat_g_per_100g,
            sugar_g_per_100g = EXCLUDED.sugar_g_per_100g,
            sodium_g_per_100g = EXCLUDED.sodium_g_per_100g,
            carbs_g_per_100g = EXCLUDED.carbs_g_per_100g
    """

    _bulk_upsert(engine, df, upsert_sql)
    logger.info(
        'dim_canonical_ingredient_nutrients: upserted %s canonical ingredients.',
        f'{len(df):,}',
    )


# ---------------------------------------------------------------------------
# dim_user
# ---------------------------------------------------------------------------


def _load_dim_user(engine, users: pd.DataFrame) -> None:
    """
    Upserts user dimension records including cluster assignments.
    On update, all aggregate fields are refreshed to reflect latest data.
    """
    users = users.copy()

    # Coerce date columns to Python date objects for psycopg2 compatibility
    for col in ['first_review_date', 'last_review_date']:
        if col in users.columns:
            users[col] = pd.to_datetime(users[col]).dt.date

    upsert_sql = """
        INSERT INTO dim_user (
            user_id, review_count, avg_rating_given, std_rating_given,
            recipe_diversity, avg_sentiment_score, avg_rating_gap,
            pct_positive_sentiment, active_days,
            first_review_date, last_review_date,
            cluster_id, cluster_label
        )
        VALUES (
            :user_id, :review_count, :avg_rating_given, :std_rating_given,
            :recipe_diversity, :avg_sentiment_score, :avg_rating_gap,
            :pct_positive_sentiment, :active_days,
            :first_review_date, :last_review_date,
            :cluster_id, :cluster_label
        )
        ON CONFLICT (user_id) DO UPDATE SET
            review_count           = EXCLUDED.review_count,
            avg_rating_given       = EXCLUDED.avg_rating_given,
            std_rating_given       = EXCLUDED.std_rating_given,
            recipe_diversity       = EXCLUDED.recipe_diversity,
            avg_sentiment_score    = EXCLUDED.avg_sentiment_score,
            avg_rating_gap         = EXCLUDED.avg_rating_gap,
            pct_positive_sentiment = EXCLUDED.pct_positive_sentiment,
            active_days            = EXCLUDED.active_days,
            last_review_date       = EXCLUDED.last_review_date,
            cluster_id             = EXCLUDED.cluster_id,
            cluster_label          = EXCLUDED.cluster_label
    """

    _bulk_upsert(engine, users, upsert_sql)
    logger.info(f'dim_user: upserted {len(users):,} users.')


# ---------------------------------------------------------------------------
# fact_interactions
# ---------------------------------------------------------------------------


def _load_fact_interactions(engine, interactions: pd.DataFrame, date_map: dict) -> None:
    """
    Upserts interaction fact records.

    interaction_id is a deterministic hash of (user_id, recipe_id, date)
    so the same interaction always gets the same ID across runs, making
    upserts truly idempotent without needing a sequence or UUID.
    """
    facts = interactions.copy()

    # Map date → date_id FK
    facts['full_date'] = facts['date'].dt.date
    facts['date_id'] = facts['full_date'].map(date_map)

    missing_date_ids = facts['date_id'].isnull().sum()
    if missing_date_ids > 0:
        logger.warning(
            f'{missing_date_ids} interactions could not be mapped to a date_id — '
            f'these will be skipped.'
        )
        facts = facts.dropna(subset=['date_id'])

    facts['date_id'] = facts['date_id'].astype(int)

    # Deterministic interaction_id
    facts['interaction_id'] = facts.apply(
        lambda row: _hash_interaction(
            row['user_id'], row['recipe_id'], row['full_date']
        ),
        axis=1,
    )

    facts['source'] = 'batch'

    facts = facts[
        [
            'interaction_id',
            'user_id',
            'recipe_id',
            'date_id',
            'rating',
            'sentiment_score',
            'rating_normalized',
            'rating_sentiment_gap',
            'source',
        ]
    ]

    upsert_sql = """
        INSERT INTO fact_interactions (
            interaction_id, user_id, recipe_id, date_id,
            rating, sentiment_score, rating_normalized,
            rating_sentiment_gap, source
        )
        VALUES (
            :interaction_id, :user_id, :recipe_id, :date_id,
            :rating, :sentiment_score, :rating_normalized,
            :rating_sentiment_gap, :source
        )
        ON CONFLICT (interaction_id) DO UPDATE SET
            sentiment_score      = EXCLUDED.sentiment_score,
            rating_normalized    = EXCLUDED.rating_normalized,
            rating_sentiment_gap = EXCLUDED.rating_sentiment_gap,
            source               = EXCLUDED.source
    """

    _bulk_upsert(engine, facts, upsert_sql)
    logger.info(f'fact_interactions: upserted {len(facts):,} records.')


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hash_interaction(user_id: int, recipe_id: int, full_date) -> int:
    """
    Generates a deterministic 64-bit integer ID from (user_id, recipe_id, date).
    Using a hash rather than a sequence means the same interaction always gets
    the same ID regardless of which DAG run inserts it first — essential for
    idempotent upserts and for the serving view deduplication logic.
    """
    key = f'{user_id}_{recipe_id}_{full_date}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:15], 16)


def _bulk_upsert(engine, df: pd.DataFrame, sql: str) -> None:
    """
    Executes the given upsert SQL in batches of UPSERT_BATCH_SIZE rows.
    Each batch is committed independently so a failure mid-load doesn't
    roll back already-written data.
    """
    records = df.where(pd.notnull(df), None).to_dict(orient='records')
    total = len(records)

    for i in range(0, total, UPSERT_BATCH_SIZE):
        batch = records[i : i + UPSERT_BATCH_SIZE]

        # Execute each batch in its own transaction so partial progress is preserved.
        with engine.begin() as conn:
            conn.execute(text(sql), batch)

        end = min(i + UPSERT_BATCH_SIZE, total)
        if i % (UPSERT_BATCH_SIZE * 5) == 0 or end == total:
            logger.info(f'  Upserted {end:,} / {total:,} rows...')


def _log_row_counts(engine) -> None:
    """Logs current row counts for all star schema tables after the load."""
    tables = [
        'dim_date',
        'dim_recipe',
        'dim_canonical_ingredient_nutrients',
        'dim_user',
        'fact_interactions',
        'recent_interactions',
    ]
    logger.info('Star schema row counts after load:')
    with engine.connect() as conn:
        for table in tables:
            try:
                n = conn.execute(text(f'SELECT COUNT(*) FROM {table}')).scalar()
                logger.info(f'  {table:<25}: {n:>10,}')
            except Exception as e:
                logger.warning(f'  {table:<25}: could not query ({e})')
