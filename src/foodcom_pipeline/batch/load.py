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

from foodcom_pipeline.batch.features import (
    load_recipe_sentiment_ratings,
    load_substitution_engine,
)
from foodcom_pipeline.batch.extract import (
    STAGING_DIR,
    TRENDS_RAW_STAGING,
    TRENDS_NORMALISED_STAGING,
    AI_MODE_RAW_STAGING,
    AI_MODE_TERM_SCORES_STAGING,
    USDA_NUTRIENTS_STAGING,
)
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
    ('weighted_review_count', 'DOUBLE PRECISION'),
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


def _ensure_fact_substitution_recommendations_columns(conn) -> None:
    """Aligns fact_substitution_recommendations with ingredient-pair schema."""
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS substitute_ingredient TEXT'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS substitute_similarity DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS rating_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS sentiment_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS protein_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS saturated_fat_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS sugar_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS sodium_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS calories_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'ALTER TABLE public.fact_substitution_recommendations '
            'ADD COLUMN IF NOT EXISTS health_delta DOUBLE PRECISION'
        )
    )
    conn.execute(
        text(
            'CREATE UNIQUE INDEX IF NOT EXISTS '
            'uq_fact_sub_candidate_substitute '
            'ON public.fact_substitution_recommendations (candidate_ingredient, substitute_ingredient)'
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
    recipe_sentiment_ratings = load_recipe_sentiment_ratings()
    user_stats = load_user_stats()
    user_clusters = load_user_clusters()
    substitution_recommendations = load_substitution_engine()

    # Merge cluster labels into user stats
    users = user_stats.merge(
        user_clusters[['user_id', 'cluster_id', 'cluster_label']],
        on='user_id',
        how='left',
    )

    # Build dimension tables first, then facts
    date_map = _load_dim_date(engine, interactions)
    _load_dim_recipe(engine, recipes, interactions, recipe_sentiment_ratings)
    _load_dim_canonical_ingredient_nutrients(engine)
    _load_dim_user(engine, users)
    _load_fact_interactions(engine, interactions, date_map)
    _load_fact_substitution_recommendations(engine, substitution_recommendations)

    _log_row_counts(engine)

    logger.info('Load step complete.')

    # Push summary to XCom
    with engine.connect() as conn:
        n_facts = conn.execute(text('SELECT COUNT(*) FROM fact_interactions')).scalar()
        n_sub_facts = conn.execute(
            text('SELECT COUNT(*) FROM fact_substitution_recommendations')
        ).scalar()
    context['ti'].xcom_push(key='fact_interactions_total', value=int(n_facts))
    context['ti'].xcom_push(
        key='fact_substitution_recommendations_total', value=int(n_sub_facts)
    )


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
        weighted_review_count FLOAT,
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

    CREATE TABLE IF NOT EXISTS fact_substitution_recommendations (
        candidate_ingredient     TEXT NOT NULL,
        substitute_ingredient    TEXT NOT NULL,
        substitute_similarity    DOUBLE PRECISION,
        recommendation_score     DOUBLE PRECISION,
        rating_delta             DOUBLE PRECISION,
        sentiment_delta          DOUBLE PRECISION,
        protein_delta            DOUBLE PRECISION,
        saturated_fat_delta      DOUBLE PRECISION,
        sugar_delta              DOUBLE PRECISION,
        sodium_delta             DOUBLE PRECISION,
        calories_delta           DOUBLE PRECISION,
        health_delta             DOUBLE PRECISION,
        updated_at               TIMESTAMP WITHOUT TIME ZONE DEFAULT NOW(),
        PRIMARY KEY (candidate_ingredient, substitute_ingredient)
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

    CREATE OR REPLACE VIEW serving_substitution_recommendations AS
        SELECT
            candidate_ingredient,
            substitute_ingredient,
            substitute_similarity,
            recommendation_score,
            rating_delta,
            sentiment_delta,
            protein_delta,
            saturated_fat_delta,
            sugar_delta,
            sodium_delta,
            calories_delta,
            health_delta,
            updated_at
        FROM fact_substitution_recommendations
        ORDER BY recommendation_score DESC NULLS LAST;
    """

    with engine.begin() as conn:
        conn.execute(text(ddl))

    # one transaction each so a failure does not poison
    # _ensure_dim_recipe_columns / _ensure_dim_canonical_ingredient_nutrients_columns.


    with engine.begin() as conn:
        _ensure_dim_recipe_columns(conn)
        _ensure_dim_canonical_ingredient_nutrients_columns(conn)
        _ensure_fact_substitution_recommendations_columns(conn)

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


def _load_dim_recipe(
    engine,
    recipes: pd.DataFrame,
    interactions: pd.DataFrame,
    recipe_sentiment_ratings: pd.DataFrame,
) -> None:
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

    # Keep warehouse sentiment fields aligned with feature engineering outputs.
    if recipe_sentiment_ratings is None or recipe_sentiment_ratings.empty:
        recipe_sent = pd.DataFrame(
            columns=['recipe_id', 'sentiment_rating', 'weighted_review_count']
        )
    else:
        recipe_sent = recipe_sentiment_ratings[
            ['recipe_id', 'sentiment_rating', 'weighted_review_count']
        ].copy()

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
    dim['weighted_review_count'] = pd.to_numeric(
        dim['weighted_review_count'], errors='coerce'
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
            'weighted_review_count',
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
            weighted_review_count, avg_cook_minutes, top_ingredients, tags, ingredient_count,
            calories, protein, fat, sugar, sodium, carbs, saturated_fat, balance_score
        )
        VALUES (
            :recipe_id, :name, :avg_rating, :review_count, :sentiment_rating,
            :weighted_review_count, :avg_cook_minutes, :top_ingredients, :tags, :ingredient_count,
            :calories, :protein, :fat, :sugar, :sodium, :carbs, :saturated_fat, :balance_score
        )
        ON CONFLICT (recipe_id) DO UPDATE SET
            avg_rating       = EXCLUDED.avg_rating,
            review_count     = EXCLUDED.review_count,
            sentiment_rating = EXCLUDED.sentiment_rating,
            weighted_review_count = EXCLUDED.weighted_review_count,
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


def _load_fact_substitution_recommendations(
    engine, substitution_recommendations: pd.DataFrame
) -> None:
    """
    Upserts substitution recommendation pairs generated by the feature step.
    This makes the substitution engine queryable directly from PostgreSQL.
    """
    if substitution_recommendations is None or substitution_recommendations.empty:
        logger.warning(
            'No substitution recommendations found in staging; '
            'skipping fact_substitution_recommendations load.'
        )
        return

    expected_cols = [
        'candidate_ingredient',
        'substitute_ingredient',
        'substitute_similarity',
        'recommendation_score',
        'rating_delta',
        'sentiment_delta',
        'protein_delta',
        'saturated_fat_delta',
        'sugar_delta',
        'sodium_delta',
        'calories_delta',
        'health_delta',
    ]
    missing = set(expected_cols) - set(substitution_recommendations.columns)
    if missing:
        raise ValueError(
            'Substitution recommendations are missing expected columns: '
            f'{sorted(missing)}'
        )

    df = substitution_recommendations[expected_cols].copy()
    df = df.dropna(subset=['candidate_ingredient', 'substitute_ingredient'])
    df['candidate_ingredient'] = df['candidate_ingredient'].astype(str)
    df['substitute_ingredient'] = df['substitute_ingredient'].astype(str)

    upsert_sql = """
        INSERT INTO fact_substitution_recommendations (
            candidate_ingredient, substitute_ingredient, substitute_similarity,
            recommendation_score, rating_delta, sentiment_delta, protein_delta,
            saturated_fat_delta, sugar_delta, sodium_delta, calories_delta,
            health_delta, updated_at
        )
        VALUES (
            :candidate_ingredient, :substitute_ingredient, :substitute_similarity,
            :recommendation_score, :rating_delta, :sentiment_delta, :protein_delta,
            :saturated_fat_delta, :sugar_delta, :sodium_delta, :calories_delta,
            :health_delta, NOW()
        )
        ON CONFLICT (candidate_ingredient, substitute_ingredient) DO UPDATE SET
            substitute_similarity = EXCLUDED.substitute_similarity,
            recommendation_score = EXCLUDED.recommendation_score,
            rating_delta = EXCLUDED.rating_delta,
            sentiment_delta = EXCLUDED.sentiment_delta,
            protein_delta = EXCLUDED.protein_delta,
            saturated_fat_delta = EXCLUDED.saturated_fat_delta,
            sugar_delta = EXCLUDED.sugar_delta,
            sodium_delta = EXCLUDED.sodium_delta,
            calories_delta = EXCLUDED.calories_delta,
            health_delta = EXCLUDED.health_delta,
            updated_at = NOW()
    """

    _bulk_upsert(engine, df, upsert_sql)
    logger.info(
        'fact_substitution_recommendations: upserted %s recommendation pairs.',
        f'{len(df):,}',
    )


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
        'fact_substitution_recommendations',
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


# ---------------------------------------------------------------------------
# Trends metadata — tracks when each seed query was last fetched so the
# extract_google_trends task can skip fetches that are still fresh.
# Metadata is stored in PostgreSQL to avoid file-locking issues (EDEADLK / errno 35)
# on Docker bind-mount volumes.  is_trends_stale() already queries google_trends_raw
# directly, so save_trends_metadata only needs to touch the DB.
# ---------------------------------------------------------------------------

_STALENESS_DAYS = 7

_ENSURE_METADATA_DDL = """
CREATE TABLE IF NOT EXISTS pipeline_metadata (
    key          TEXT PRIMARY KEY,
    fetched_date DATE NOT NULL
)
"""


def _metadata_engine():
    return create_engine(POSTGRES_CONN)


def _ensure_metadata_table(conn) -> None:
    conn.execute(text(_ENSURE_METADATA_DDL))


def _days_since(last_fetched) -> int:
    if hasattr(last_fetched, 'date'):
        last_fetched = last_fetched.date()
    elif isinstance(last_fetched, str):
        last_fetched = date.fromisoformat(last_fetched)
    return (date.today() - last_fetched).days


def is_trends_stale(seed_query: str) -> bool:
    """Returns True if google_trends_raw has no rows for *seed_query* or data is older than 7 days."""
    try:
        engine = _metadata_engine()
        with engine.connect() as conn:
            result = conn.execute(
                text(
                    "SELECT MAX(fetched_date) FROM google_trends_raw "
                    "WHERE seed_query = :seed"
                ),
                {"seed": seed_query},
            ).scalar()
        if result is None:
            return True
        return _days_since(result) > _STALENESS_DAYS
    except Exception:
        return True


def is_ai_mode_stale() -> bool:
    """Returns True if AI Mode data is missing, older than 7 days, or absent from the DB."""
    try:
        engine = _metadata_engine()
        with engine.connect() as conn:
            _ensure_metadata_table(conn)
            row = conn.execute(
                text("SELECT fetched_date FROM pipeline_metadata WHERE key = 'ai_mode'")
            ).fetchone()
        if row is None or _days_since(row[0]) > _STALENESS_DAYS:
            return True
        # Confirm data actually landed
        with engine.connect() as conn:
            count = conn.execute(text("SELECT COUNT(*) FROM ai_mode_raw")).scalar()
        return not count
    except Exception:
        return True


def save_trends_metadata(seed_query: str, fetched_date: date) -> None:
    """Records the date of the last successful Trends fetch for *seed_query*."""
    try:
        engine = _metadata_engine()
        with engine.begin() as conn:
            _ensure_metadata_table(conn)
            conn.execute(
                text("""
                    INSERT INTO pipeline_metadata (key, fetched_date)
                    VALUES (:key, :fetched_date)
                    ON CONFLICT (key) DO UPDATE SET fetched_date = EXCLUDED.fetched_date
                """),
                {"key": f"trends:{seed_query}", "fetched_date": fetched_date},
            )
        logger.info('trends_metadata updated: %s → %s', seed_query, fetched_date)
    except Exception as exc:
        logger.warning('Could not save trends metadata: %s', exc)


def save_ai_mode_metadata(fetched_date: date) -> None:
    """Records the date of the last successful AI Mode fetch."""
    try:
        engine = _metadata_engine()
        with engine.begin() as conn:
            _ensure_metadata_table(conn)
            conn.execute(
                text("""
                    INSERT INTO pipeline_metadata (key, fetched_date)
                    VALUES ('ai_mode', :fetched_date)
                    ON CONFLICT (key) DO UPDATE SET fetched_date = EXCLUDED.fetched_date
                """),
                {"fetched_date": fetched_date},
            )
        logger.info('ai_mode_metadata updated: %s', fetched_date)
    except Exception as exc:
        logger.warning('Could not save ai_mode metadata: %s', exc)


# ---------------------------------------------------------------------------
# Trends + AI Mode load — pushes all signal staging files into Postgres
# ---------------------------------------------------------------------------


def load_app_data(**context) -> None:
    """
    Loads app-layer staging parquets into PostgreSQL.

    Tables populated:
      recipe_gap_analysis    — one row per external term matched against recipe index
      recipe_term_clusters   — one row per semantic cluster of external terms
    """
    from foodcom_pipeline.batch.match import RECIPE_GAP_ANALYSIS_STAGING
    from foodcom_pipeline.batch.cluster_terms import RECIPE_TERM_CLUSTERS_STAGING

    engine = create_engine(POSTGRES_CONN)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recipe_gap_analysis (
                source                    TEXT,
                raw_term                  TEXT             NOT NULL,
                canonical_term            TEXT             NOT NULL,
                source_score              DOUBLE PRECISION,
                best_foodcom_recipe_id    BIGINT,
                best_foodcom_recipe_name  TEXT,
                best_foodcom_canonical_term TEXT,
                best_foodcom_similarity   DOUBLE PRECISION,
                match_status              TEXT,
                gap_score                 DOUBLE PRECISION,
                top_gap_rank              INTEGER,
                opportunity_label         TEXT,
                insight_summary           TEXT,
                gap_reason                TEXT,
                matching_method           TEXT,
                fetched_date              DATE,
                PRIMARY KEY (source, canonical_term)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS recipe_term_clusters (
                cluster_id              INTEGER          NOT NULL,
                cluster_label           TEXT,
                representative_term     TEXT             NOT NULL PRIMARY KEY,
                source_terms            TEXT,
                sources_present         TEXT,
                dominant_tags           TEXT,
                avg_gap_score           DOUBLE PRECISION,
                max_gap_score           DOUBLE PRECISION,
                best_foodcom_match      TEXT,
                foodcom_coverage_count  INTEGER,
                explanation             TEXT
            )
        """))

    _load_staging_table(
        engine, RECIPE_GAP_ANALYSIS_STAGING, 'recipe_gap_analysis',
        cols=(
            'source, raw_term, canonical_term, source_score, '
            'best_foodcom_recipe_id, best_foodcom_recipe_name, best_foodcom_canonical_term, '
            'best_foodcom_similarity, match_status, gap_score, top_gap_rank, '
            'opportunity_label, insight_summary, gap_reason, matching_method, fetched_date'
        ),
        conflict='(source, canonical_term)',
        update=(
            'raw_term = EXCLUDED.raw_term, source_score = EXCLUDED.source_score, '
            'best_foodcom_recipe_id = EXCLUDED.best_foodcom_recipe_id, '
            'best_foodcom_recipe_name = EXCLUDED.best_foodcom_recipe_name, '
            'best_foodcom_canonical_term = EXCLUDED.best_foodcom_canonical_term, '
            'best_foodcom_similarity = EXCLUDED.best_foodcom_similarity, '
            'match_status = EXCLUDED.match_status, gap_score = EXCLUDED.gap_score, '
            'top_gap_rank = EXCLUDED.top_gap_rank, opportunity_label = EXCLUDED.opportunity_label, '
            'insight_summary = EXCLUDED.insight_summary, gap_reason = EXCLUDED.gap_reason, '
            'matching_method = EXCLUDED.matching_method, fetched_date = EXCLUDED.fetched_date'
        ),
    )
    _load_staging_table(
        engine, RECIPE_TERM_CLUSTERS_STAGING, 'recipe_term_clusters',
        cols=(
            'cluster_id, cluster_label, representative_term, source_terms, sources_present, '
            'dominant_tags, avg_gap_score, max_gap_score, best_foodcom_match, '
            'foodcom_coverage_count, explanation'
        ),
        conflict='(representative_term)',
        update=(
            'cluster_id = EXCLUDED.cluster_id, cluster_label = EXCLUDED.cluster_label, '
            'source_terms = EXCLUDED.source_terms, sources_present = EXCLUDED.sources_present, '
            'dominant_tags = EXCLUDED.dominant_tags, avg_gap_score = EXCLUDED.avg_gap_score, '
            'max_gap_score = EXCLUDED.max_gap_score, best_foodcom_match = EXCLUDED.best_foodcom_match, '
            'foodcom_coverage_count = EXCLUDED.foodcom_coverage_count, '
            'explanation = EXCLUDED.explanation'
        ),
    )

    if context.get('ti'):
        with engine.connect() as conn:
            n_gap = conn.execute(text('SELECT COUNT(*) FROM recipe_gap_analysis')).scalar()
            n_clusters = conn.execute(text('SELECT COUNT(*) FROM recipe_term_clusters')).scalar()
        context['ti'].xcom_push(key='recipe_gap_analysis_total', value=int(n_gap))
        context['ti'].xcom_push(key='recipe_term_clusters_total', value=int(n_clusters))
        logger.info('load_app_data complete: %d gap rows, %d clusters.', n_gap, n_clusters)


def load_trends(**context) -> None:
    """
    Loads all Google Trends and AI Mode staging parquet files into Postgres.
    Creates tables on first run; subsequent runs upsert.

    Tables:
      google_trends_raw         — raw related-query scores per seed
      google_trends_normalised  — z-score normalised related-query scores
      ai_mode_raw               — raw text blocks from Google AI Mode
      ai_mode_term_scores       — term frequency + normalised scores
    """
    engine = create_engine(POSTGRES_CONN)

    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS google_trends_raw (
                seed_query    TEXT    NOT NULL,
                related_query TEXT    NOT NULL,
                query_type    TEXT    NOT NULL,
                fetched_date  DATE    NOT NULL,
                raw_value     INTEGER NOT NULL,
                PRIMARY KEY (seed_query, related_query, query_type, fetched_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS google_trends_normalised (
                seed_query       TEXT             NOT NULL,
                related_query    TEXT             NOT NULL,
                query_type       TEXT             NOT NULL,
                fetched_date     DATE             NOT NULL,
                raw_value        INTEGER          NOT NULL,
                normalised_score DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (seed_query, related_query, query_type, fetched_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ai_mode_raw (
                seed_query   TEXT    NOT NULL,
                block_index  INTEGER NOT NULL,
                body         TEXT    NOT NULL,
                fetched_date DATE    NOT NULL,
                PRIMARY KEY (seed_query, block_index, fetched_date)
            )
        """))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ai_mode_term_scores (
                term             TEXT             NOT NULL,
                raw_frequency    INTEGER          NOT NULL,
                normalised_score DOUBLE PRECISION NOT NULL,
                fetched_date     DATE             NOT NULL,
                PRIMARY KEY (term, fetched_date)
            )
        """))

    _load_staging_table(
        engine, TRENDS_RAW_STAGING, 'google_trends_raw',
        cols='seed_query, related_query, query_type, fetched_date, raw_value',
        conflict='(seed_query, related_query, query_type, fetched_date)',
        update='raw_value = EXCLUDED.raw_value',
    )
    _load_staging_table(
        engine, TRENDS_NORMALISED_STAGING, 'google_trends_normalised',
        cols='seed_query, related_query, query_type, fetched_date, raw_value, normalised_score',
        conflict='(seed_query, related_query, query_type, fetched_date)',
        update='raw_value = EXCLUDED.raw_value, normalised_score = EXCLUDED.normalised_score',
    )
    _load_staging_table(
        engine, AI_MODE_RAW_STAGING, 'ai_mode_raw',
        cols='seed_query, block_index, body, fetched_date',
        conflict='(seed_query, block_index, fetched_date)',
        update='body = EXCLUDED.body',
    )
    _load_staging_table(
        engine, AI_MODE_TERM_SCORES_STAGING, 'ai_mode_term_scores',
        cols='term, raw_frequency, normalised_score, fetched_date',
        conflict='(term, fetched_date)',
        update='raw_frequency = EXCLUDED.raw_frequency, normalised_score = EXCLUDED.normalised_score',
    )


def _load_staging_table(engine, path, table: str, cols: str, conflict: str, update: str) -> None:
    """Reads a staging parquet file and bulk-upserts it into *table*."""
    if not path.is_file():
        logger.warning('Staging file missing for %s — skipping.', table)
        return
    df = pd.read_parquet(path)
    named = ', '.join(f':{c.strip()}' for c in cols.split(','))
    sql = f"""
        INSERT INTO {table} ({cols})
        VALUES ({named})
        ON CONFLICT {conflict}
        DO UPDATE SET {update}
    """
    _bulk_upsert(engine, df, sql)
    logger.info('Loaded %d rows into %s', len(df), table)
