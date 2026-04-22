"""
features.py
-----------
Feature engineering step for the Food.com batch pipeline.

Computes all derived features needed before clustering:
  1. Per-user aggregate stats (avg_rating, review_count, sentiment stats, etc.)
  2. Per-recipe Bayesian sentiment_rating (inverse-frequency + Bayesian shrinkage)
  3. Ingredient-level features (avg rating, avg sentiment, recipe count)
  4. Substitution coverage across all canonical ingredients

Outputs (staged as parquet for downstream steps):
  user_stats.parquet               — one row per user → cluster.py, load.py
  recipe_sentiment_ratings.parquet — one row per recipe → load.py, dashboard
  ingredient_features.parquet      — one row per canonical ingredient → load.py, dashboard

Per-recipe sentiment_rating formula (Bayesian shrinkage with inverse-frequency weights):

  sentiment_rating = (Σ w_i · v_i  +  m · C) / (Σ w_i  +  m)

  where:
    w_i = 1 / user_review_count   — downweights prolific raters
    v_i = VADER sentiment_score for interaction i
    m   = BAYESIAN_PSEUDO_COUNT   — shrinkage strength (pulls sparse recipes to global mean)
    C   = global mean sentiment score across all scoreable interactions

Output columns — user_stats.parquet (one row per user_id):
  avg_rating_given      : mean star rating across all reviews
  review_count          : total number of reviews submitted
  recipe_diversity      : number of unique recipes reviewed
  avg_sentiment_score   : mean VADER sentiment score (nulls excluded)
  avg_rating_gap        : mean rating_sentiment_gap
  std_rating_given      : std of star ratings (consistency signal)
  pct_positive_sentiment: fraction of reviews with sentiment_score > 0
  first_review_date     : earliest review date
  last_review_date      : most recent review date
  active_days           : days between first and last review

Output columns — recipe_sentiment_ratings.parquet (one row per recipe_id):
  recipe_id             : recipe identifier
  sentiment_rating      : Bayesian-shrunk inverse-frequency-weighted sentiment score
  weighted_review_count : sum of inverse-frequency weights (proxy for review depth)

Output columns — ingredient_features.parquet (one row per canonical_ingredient):
  canonical_ingredient      : canonical ingredient string from ingr_map.pkl
  recipe_count              : number of distinct recipes containing this ingredient
  avg_rating                : mean recipe avg_rating across those recipes
  avg_sentiment             : mean recipe avg_sentiment across those recipes
  trend_index               : reserved nullable column for compatibility
"""

import logging
import ast
import json
import os
import re

import numpy as np
import pandas as pd
import requests
from foodcom_pipeline.batch.clean import load_cleaned_recipes
from foodcom_pipeline.batch.extract import STAGING_DIR, USDA_NUTRIENTS_STAGING
from foodcom_pipeline.batch.sentiment import load_sentiment_interactions
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from sentence_transformers import SentenceTransformer
except ImportError as exc:  # pragma: no cover - validated at runtime
    SentenceTransformer = None
    _IMPORT_ERROR = exc

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_STATS_STAGING = STAGING_DIR / 'user_stats.parquet'
RECIPE_SENTIMENT_STAGING = STAGING_DIR / 'recipe_sentiment_ratings.parquet'
INGREDIENT_FEATURES_STAGING = STAGING_DIR / 'ingredient_features.parquet'
SUBSTITUTION_ENGINE_STAGING = STAGING_DIR / 'substitution_engine.parquet'
CANONICAL_INGREDIENT_EMBEDDINGS_STAGING = (
    STAGING_DIR / 'canonical_ingredient_embeddings.parquet'
)

# Ingredient categories for user taste segmentation feature vector.
# Keyword matching on already-normalised canonical ingredient strings.
INGREDIENT_CATEGORIES: dict[str, list[str]] = {
    'dairy': [
        'milk', 'butter', 'cheese', 'cream', 'yogurt', 'parmesan', 'mozzarella',
        'cheddar', 'ricotta', 'cottage', 'brie', 'feta', 'gouda', 'whipping',
    ],
    'protein': [
        'chicken', 'beef', 'pork', 'salmon', 'fish', 'turkey', 'lamb', 'shrimp',
        'egg', 'tuna', 'cod', 'tilapia', 'bacon', 'sausage', 'ham', 'steak',
        'tofu', 'tempeh',
    ],
    'vegetable': [
        'onion', 'garlic', 'tomato', 'carrot', 'pepper', 'spinach', 'broccoli',
        'celery', 'mushroom', 'zucchini', 'cucumber', 'lettuce', 'potato',
        'cabbage', 'corn', 'pea', 'kale', 'asparagus', 'eggplant',
    ],
    'baking': [
        'flour', 'sugar', 'baking powder', 'baking soda', 'yeast', 'vanilla',
        'cocoa', 'chocolate', 'brown sugar', 'powdered sugar', 'molasses',
        'shortening',
    ],
    'international': [
        'soy sauce', 'fish sauce', 'miso', 'gochujang', 'mirin', 'tahini',
        'cumin', 'turmeric', 'curry', 'garam masala', 'coconut milk',
        'sesame oil', 'rice vinegar', 'sriracha', 'harissa',
    ],
}

# Bayesian shrinkage pseudo-count (m in the formula).
# Higher values pull recipes with few reviews more strongly toward the global mean.
BAYESIAN_PSEUDO_COUNT = 10

# Substitution coverage now includes all canonical ingredients observed in recipes.

# Ollama compatibility gate config.
OLLAMA_BASE_URL = os.getenv('FOODCOM_OLLAMA_BASE_URL', 'http://host.docker.internal:11434')
OLLAMA_MODEL = os.getenv('FOODCOM_OLLAMA_MODEL', 'llama3.1:8b')
OLLAMA_TIMEOUT_SECONDS = float(os.getenv('FOODCOM_OLLAMA_TIMEOUT_SECONDS', '8'))
SUBSTITUTION_MODEL_NAME = os.getenv(
    'FOODCOM_SUBSTITUTION_MODEL_NAME', 'all-MiniLM-L6-v2'
)
SUBSTITUTION_TEXT_WEIGHT = float(os.getenv('FOODCOM_SUBSTITUTION_TEXT_WEIGHT', '0.7'))
SUBSTITUTION_NUTRITION_WEIGHT = float(
    os.getenv('FOODCOM_SUBSTITUTION_NUTRITION_WEIGHT', '0.3')
)
SUBSTITUTION_TOP_K = int(os.getenv('FOODCOM_SUBSTITUTION_TOP_K', '5'))
SUBSTITUTION_NUTRIENT_COLS = [
    'calories_per_100g',
    'protein_g_per_100g',
    'fat_g_per_100g',
    'saturated_fat_g_per_100g',
    'sugar_g_per_100g',
    'sodium_g_per_100g',
    'carbs_g_per_100g',
]


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_features(**context) -> None:
    """
    Main feature engineering entry point. Computes and stages:
      - Per-user aggregate stats
      - Per-recipe Bayesian sentiment ratings
      - Ingredient-level features with substitution flags
    """
    logger.info('Starting feature engineering step.')

    interactions = load_sentiment_interactions()
    recipes = load_cleaned_recipes()

    user_stats = _compute_user_stats(interactions)
    category_ratings = _compute_user_category_ratings(interactions, recipes)
    user_stats = user_stats.merge(category_ratings, on='user_id', how='left')
    for cat in INGREDIENT_CATEGORIES:
        col = f'avg_rating_{cat}'
        global_mean = float(category_ratings[col].mean()) if col in category_ratings.columns else float(user_stats['avg_rating_given'].mean())
        user_stats[col] = user_stats[col].fillna(global_mean).round(4)

    recipe_ratings = _compute_recipe_sentiment_ratings(interactions)
    ingredient_features = _compute_ingredient_features(interactions, recipes)
    substitution_engine = _compute_substitution_engine(ingredient_features, recipes)

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    user_stats.to_parquet(USER_STATS_STAGING, index=False)
    recipe_ratings.to_parquet(RECIPE_SENTIMENT_STAGING, index=False)
    ingredient_features.to_parquet(INGREDIENT_FEATURES_STAGING, index=False)
    substitution_engine.to_parquet(SUBSTITUTION_ENGINE_STAGING, index=False)

    n_candidates = int(len(ingredient_features))
    logger.info(
        f'Feature engineering complete. '
        f'{len(user_stats):,} users, '
        f'{len(recipe_ratings):,} recipe sentiment ratings, '
        f'{len(ingredient_features):,} ingredient features '
        f'({n_candidates} substitution candidates).'
    )

    context['ti'].xcom_push(key='n_users', value=len(user_stats))
    context['ti'].xcom_push(key='n_recipe_ratings', value=len(recipe_ratings))
    context['ti'].xcom_push(key='n_substitution_candidates', value=n_candidates)
    context['ti'].xcom_push(key='n_substitution_pairs', value=len(substitution_engine))


def run_embed_canonical_ingredients(**context) -> None:
    """
    Builds and stages canonical ingredient hybrid embeddings from USDA nutrient rows.
    This is separated from run_features so embedding generation can run independently.
    """
    logger.info('Building canonical ingredient embedding index.')
    embedding_df = _build_canonical_embedding_index()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    embedding_df.to_parquet(CANONICAL_INGREDIENT_EMBEDDINGS_STAGING, index=False)
    logger.info(
        'Canonical ingredient embeddings staged: %s rows.',
        f'{len(embedding_df):,}',
    )
    context['ti'].xcom_push(key='n_canonical_embeddings', value=int(len(embedding_df)))


# ---------------------------------------------------------------------------
# Per-user aggregate stats (feeds cluster.py and load.py → dim_user)
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

    agg = _compute_pct_positive_sentiment(agg, df)
    agg = _compute_active_days(agg)
    agg = _fill_nulls(agg)
    agg = _round_floats(agg)

    _log_user_stats_summary(agg)
    return agg


def _compute_pct_positive_sentiment(
    agg: pd.DataFrame, interactions: pd.DataFrame
) -> pd.DataFrame:
    """Fraction of a user's reviews with sentiment_score > 0."""
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
    return agg.drop(columns=['positive_sentiment_count'])


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
    float_cols = agg.select_dtypes(include='float').columns
    agg[float_cols] = agg[float_cols].round(4)
    return agg


def _log_user_stats_summary(agg: pd.DataFrame) -> None:
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
        f"  pct_positive_sent    : mean={agg['pct_positive_sentiment'].mean():.3f}"
    )
    logger.info(
        f"  active_days          : mean={agg['active_days'].mean():.0f}, "
        f"max={agg['active_days'].max()}"
    )


# ---------------------------------------------------------------------------
# Per-user ingredient category rating features (clustering feature vector)
# ---------------------------------------------------------------------------


def _compute_user_category_ratings(
    interactions: pd.DataFrame, recipes: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes per-user average rating given to recipes in each of 5 ingredient
    categories: dairy, protein, vegetable, baking, international.

    Category membership is determined by keyword matching on the already-normalised
    canonical ingredient strings (ingredients_canonical_list from clean.py).
    A recipe can belong to multiple categories. Nulls (user never reviewed a recipe
    in that category) are filled with the global mean for that category.
    """
    cat_cols = [f'avg_rating_{cat}' for cat in INGREDIENT_CATEGORIES]

    if 'ingredients_canonical_list' not in recipes.columns:
        logger.warning(
            'ingredients_canonical_list missing — category rating features will be '
            'filled with global mean.'
        )
        empty = interactions[['user_id']].drop_duplicates()
        global_mean = float(interactions['rating'].mean())
        for col in cat_cols:
            empty[col] = round(global_mean, 4)
        return empty

    def _recipe_categories(ingredient_list: list) -> list[str]:
        cats: set[str] = set()
        for ing in ingredient_list:
            ing_lower = str(ing).lower()
            for cat, keywords in INGREDIENT_CATEGORIES.items():
                if any(kw in ing_lower for kw in keywords):
                    cats.add(cat)
        return list(cats)

    recipe_cats = (
        recipes[['id', 'ingredients_canonical_list']]
        .rename(columns={'id': 'recipe_id'})
        .copy()
    )
    recipe_cats['category'] = recipe_cats['ingredients_canonical_list'].apply(
        _recipe_categories
    )
    recipe_cats = (
        recipe_cats[['recipe_id', 'category']]
        .explode('category')
        .dropna(subset=['category'])
    )

    merged = interactions[['user_id', 'recipe_id', 'rating']].merge(
        recipe_cats, on='recipe_id', how='inner'
    )

    user_cat_pivot = (
        merged.groupby(['user_id', 'category'])['rating']
        .mean()
        .unstack(level='category')
        .reset_index()
    )
    user_cat_pivot.columns.name = None

    for cat in INGREDIENT_CATEGORIES:
        if cat not in user_cat_pivot.columns:
            user_cat_pivot[cat] = None

    user_cat_pivot = user_cat_pivot.rename(
        columns={cat: f'avg_rating_{cat}' for cat in INGREDIENT_CATEGORIES}
    )

    for col in cat_cols:
        global_mean = float(user_cat_pivot[col].mean())
        user_cat_pivot[col] = user_cat_pivot[col].fillna(global_mean).round(4)

    coverage = {
        cat: int(user_cat_pivot[f'avg_rating_{cat}'].notna().sum())
        for cat in INGREDIENT_CATEGORIES
    }
    logger.info(
        f'User category ratings computed for {len(user_cat_pivot):,} users. '
        f'Coverage (non-null before fill): {coverage}'
    )
    return user_cat_pivot[['user_id'] + cat_cols]


# ---------------------------------------------------------------------------
# Per-recipe Bayesian sentiment rating
# ---------------------------------------------------------------------------


def _compute_recipe_sentiment_ratings(interactions: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a per-recipe sentiment_rating using inverse-frequency weighting
    and Bayesian shrinkage:

      sentiment_rating = (Σ w_i · v_i  +  m · C) / (Σ w_i  +  m)

    Recipes with few reviews are pulled toward the global mean C.
    Prolific reviewers are downweighted via w_i = 1 / user_review_count.
    """
    scoreable = interactions[interactions['sentiment_score'].notna()].copy()

    if scoreable.empty:
        logger.warning(
            'No sentiment-scored interactions — recipe sentiment ratings will be empty.'
        )
        return pd.DataFrame(
            columns=['recipe_id', 'sentiment_rating', 'weighted_review_count']
        )

    C = float(scoreable['sentiment_score'].mean())
    m = BAYESIAN_PSEUDO_COUNT
    logger.info(
        f'Computing recipe sentiment ratings. '
        f'Global mean C={C:.4f}, pseudo-count m={m}, '
        f'{len(scoreable):,} scoreable interactions.'
    )

    user_counts = scoreable.groupby('user_id').size().rename('user_review_count')
    scoreable = scoreable.join(user_counts, on='user_id')
    scoreable['weight'] = 1.0 / scoreable['user_review_count']
    scoreable['weighted_score'] = scoreable['weight'] * scoreable['sentiment_score']

    agg = (
        scoreable.groupby('recipe_id')
        .agg(
            weighted_sum=('weighted_score', 'sum'),
            weight_total=('weight', 'sum'),
        )
        .reset_index()
    )

    agg['sentiment_rating'] = (
        (agg['weighted_sum'] + m * C) / (agg['weight_total'] + m)
    ).round(4)
    agg = agg.rename(columns={'weight_total': 'weighted_review_count'})

    logger.info(
        f'Recipe sentiment ratings computed for {len(agg):,} recipes. '
        f"Mean={agg['sentiment_rating'].mean():.4f}, "
        f"Std={agg['sentiment_rating'].std():.4f}"
    )
    return agg[['recipe_id', 'sentiment_rating', 'weighted_review_count']]


# ---------------------------------------------------------------------------
# Ingredient-level features and substitution candidate flagging
# ---------------------------------------------------------------------------


def _compute_ingredient_features(
    interactions: pd.DataFrame, recipes: pd.DataFrame
) -> pd.DataFrame:
    """
    Computes per-canonical-ingredient aggregate features and flags substitution
    candidates (ingredients appearing in low-rated recipes).

    trend_index is retained as a nullable reserved field for compatibility.
    """
    if 'ingredients_canonical_list' not in recipes.columns:
        logger.warning(
            'ingredients_canonical_list not found in recipes — '
            'skipping ingredient feature computation.'
        )
        return pd.DataFrame(
            columns=[
                'canonical_ingredient',
                'recipe_count',
                'avg_rating',
                'avg_sentiment',
                'trend_index',
            ]
        )

    # Recipe-level avg rating and sentiment from interactions
    recipe_agg = (
        interactions.groupby('recipe_id')
        .agg(
            recipe_avg_rating=('rating', 'mean'),
            recipe_avg_sentiment=('sentiment_score', 'mean'),
        )
        .reset_index()
    )

    # Explode canonical ingredient lists to one row per (recipe, ingredient)
    exploded = (
        recipes[['id', 'ingredients_canonical_list']]
        .rename(columns={'id': 'recipe_id'})
        .explode('ingredients_canonical_list')
        .rename(columns={'ingredients_canonical_list': 'canonical_ingredient'})
        .dropna(subset=['canonical_ingredient'])
    )

    merged = exploded.merge(recipe_agg, on='recipe_id', how='left')

    ingr_features = (
        merged.groupby('canonical_ingredient')
        .agg(
            recipe_count=('recipe_id', 'nunique'),
            avg_rating=('recipe_avg_rating', 'mean'),
            avg_sentiment=('recipe_avg_sentiment', 'mean'),
        )
        .reset_index()
    )

    ingr_features['avg_rating'] = ingr_features['avg_rating'].round(4)
    ingr_features['avg_sentiment'] = ingr_features['avg_sentiment'].round(4)

    # Reserved nullable field (Google Trends no longer used in substitution logic).
    ingr_features['trend_index'] = None

    n_candidates = int(len(ingr_features))
    logger.info(
        f'Ingredient features computed for {len(ingr_features):,} canonical ingredients. '
        f'{n_candidates} substitution candidates (all canonical ingredients).'
    )
    return ingr_features


def _compute_ingredient_nutrition_profiles(recipes: pd.DataFrame) -> pd.DataFrame:
    """
    Computes average nutrient profile per canonical ingredient by exploding
    recipe ingredient lists and averaging recipe-level nutrient columns.
    """
    required_cols = [
        'id',
        'ingredients_canonical_list',
        'protein',
        'saturated_fat',
        'sugar',
        'sodium',
        'calories',
    ]
    if not set(required_cols).issubset(recipes.columns):
        return pd.DataFrame(
            columns=[
                'canonical_ingredient',
                'protein',
                'saturated_fat',
                'sugar',
                'sodium',
                'calories',
            ]
        )

    exploded = (
        recipes[required_cols]
        .rename(columns={'id': 'recipe_id'})
        .explode('ingredients_canonical_list')
        .rename(columns={'ingredients_canonical_list': 'canonical_ingredient'})
        .dropna(subset=['canonical_ingredient'])
    )

    nutrient_profiles = (
        exploded.groupby('canonical_ingredient')
        .agg(
            protein=('protein', 'mean'),
            saturated_fat=('saturated_fat', 'mean'),
            sugar=('sugar', 'mean'),
            sodium=('sodium', 'mean'),
            calories=('calories', 'mean'),
        )
        .reset_index()
    )
    return nutrient_profiles.round(4)


def _normalize_ingredient_text(value: str) -> str:
    text = str(value).lower().strip()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def _empty_substitution_engine_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
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
    )


def _safe_float(value) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_delta(new_value, base_value) -> float | None:
    new_float = _safe_float(new_value)
    base_float = _safe_float(base_value)
    if new_float is None or base_float is None:
        return None
    return new_float - base_float


def _round_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _build_canonical_embedding_index() -> pd.DataFrame:
    if not USDA_NUTRIENTS_STAGING.is_file():
        raise FileNotFoundError(
            f'USDA nutrients staging not found: {USDA_NUTRIENTS_STAGING}'
        )

    nutrients = pd.read_parquet(USDA_NUTRIENTS_STAGING)
    required_nutrient_cols = {'canonical_ingredient', *SUBSTITUTION_NUTRIENT_COLS}
    if not required_nutrient_cols.issubset(nutrients.columns):
        missing = sorted(required_nutrient_cols - set(nutrients.columns))
        raise ValueError(
            f'USDA nutrients staging missing substitution columns: {missing}'
        )

    canonical_df = (
        nutrients[['canonical_ingredient', *SUBSTITUTION_NUTRIENT_COLS]]
        .dropna(subset=['canonical_ingredient'])
        .drop_duplicates(subset=['canonical_ingredient'], keep='first')
        .copy()
    )
    canonical_df['canonical_ingredient'] = (
        canonical_df['canonical_ingredient'].astype(str).str.strip().str.lower()
    )
    canonical_df = canonical_df[canonical_df['canonical_ingredient'] != '']
    if canonical_df.empty:
        return pd.DataFrame(columns=['canonical_ingredient', 'combined_embedding'])

    if SentenceTransformer is None:
        raise ImportError(
            'sentence-transformers is required but could not be imported.'
        ) from _IMPORT_ERROR

    ingredient_names = canonical_df['canonical_ingredient'].reset_index(drop=True)
    cleaned_names = ingredient_names.map(_normalize_ingredient_text)
    model = SentenceTransformer(SUBSTITUTION_MODEL_NAME)
    text_embeddings = model.encode(
        cleaned_names.tolist(),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    text_embeddings = np.asarray(text_embeddings, dtype=np.float32)

    nutrition_matrix = canonical_df[SUBSTITUTION_NUTRIENT_COLS].copy()
    nutrition_array = SimpleImputer(strategy='median').fit_transform(nutrition_matrix)
    nutrition_embeddings = StandardScaler().fit_transform(nutrition_array).astype(np.float32)

    combined_embeddings = np.hstack(
        [
            SUBSTITUTION_TEXT_WEIGHT * text_embeddings,
            SUBSTITUTION_NUTRITION_WEIGHT * nutrition_embeddings,
        ]
    )
    norms = np.linalg.norm(combined_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    combined_embeddings = combined_embeddings / norms

    out = canonical_df.reset_index(drop=True).copy()
    out['combined_embedding'] = [row.tolist() for row in combined_embeddings]
    return out


def _compute_substitution_engine(
    ingredient_features: pd.DataFrame, recipes: pd.DataFrame
) -> pd.DataFrame:
    """
    Generates ingredient substitution recommendations.

    Builds recipe-level substitution recommendations using a hybrid embedding space:
      - Text embeddings over canonical ingredient names
      - Standardized USDA nutrient vectors
      - Weighted concatenation + cosine nearest neighbors
    Produces one row per (candidate_ingredient, substitute_ingredient) pair.
    """
    if (
        recipes is None
        or recipes.empty
        or 'id' not in recipes.columns
        or 'ingredients_canonical_list' not in recipes.columns
    ):
        return _empty_substitution_engine_frame()

    if not CANONICAL_INGREDIENT_EMBEDDINGS_STAGING.is_file():
        logger.warning(
            'No canonical embedding index found at %s; substitution recommendations will be empty.',
            CANONICAL_INGREDIENT_EMBEDDINGS_STAGING,
        )
        return _empty_substitution_engine_frame()

    canonical_df = pd.read_parquet(CANONICAL_INGREDIENT_EMBEDDINGS_STAGING)
    if canonical_df.empty:
        logger.warning('USDA nutrient staging has no canonical ingredients.')
        return _empty_substitution_engine_frame()

    required_cols = {'canonical_ingredient', 'combined_embedding', *SUBSTITUTION_NUTRIENT_COLS}
    if not required_cols.issubset(canonical_df.columns):
        missing = sorted(required_cols - set(canonical_df.columns))
        raise ValueError(
            'canonical_ingredient_embeddings.parquet missing required columns: '
            f'{missing}'
        )

    canonical_df = canonical_df.copy()
    canonical_df['canonical_ingredient'] = (
        canonical_df['canonical_ingredient'].astype(str).str.strip().str.lower()
    )
    ingredient_names = canonical_df['canonical_ingredient'].reset_index(drop=True)
    combined_embeddings = np.asarray(
        canonical_df['combined_embedding'].tolist(), dtype=np.float32
    )

    nn = NearestNeighbors(metric='cosine', algorithm='brute')
    nn.fit(combined_embeddings)

    name_to_index = pd.Series(
        np.arange(len(ingredient_names)), index=ingredient_names.values
    ).to_dict()

    unique_recipe_ingredients = (
        recipes[['id', 'ingredients_canonical_list']]
        .rename(columns={'id': 'recipe_id'})
        .explode('ingredients_canonical_list')
        .dropna(subset=['ingredients_canonical_list'])
        .copy()
    )
    unique_recipe_ingredients['candidate_ingredient'] = (
        unique_recipe_ingredients['ingredients_canonical_list']
        .astype(str)
        .str.strip()
        .str.lower()
    )
    unique_recipe_ingredients = unique_recipe_ingredients[
        unique_recipe_ingredients['candidate_ingredient'].isin(name_to_index)
    ][['candidate_ingredient']].drop_duplicates()

    if unique_recipe_ingredients.empty:
        logger.warning(
            'No recipe ingredients overlap with canonical USDA ingredients.'
        )
        return _empty_substitution_engine_frame()

    feature_base = ingredient_features.copy()
    if not feature_base.empty and 'canonical_ingredient' in feature_base.columns:
        feature_base['canonical_ingredient'] = (
            feature_base['canonical_ingredient'].astype(str).str.strip().str.lower()
        )
    feature_lookup = (
        feature_base.set_index('canonical_ingredient')
        if not ingredient_features.empty
        else pd.DataFrame()
    )
    nutrition_lookup = canonical_df.set_index('canonical_ingredient')

    rows: list[dict] = []
    neighbor_count = min(SUBSTITUTION_TOP_K + 1, len(ingredient_names))
    for _, row in unique_recipe_ingredients.iterrows():
        candidate = str(row['candidate_ingredient'])
        idx = int(name_to_index[candidate])
        distances, indices = nn.kneighbors(
            [combined_embeddings[idx]], n_neighbors=neighbor_count
        )
        selected_for_candidate = 0
        candidate_feat = (
            feature_lookup.loc[candidate]
            if not feature_lookup.empty and candidate in feature_lookup.index
            else None
        )
        candidate_nut = nutrition_lookup.loc[candidate]
        for dist, match_idx in zip(distances[0], indices[0]):
            if int(match_idx) == idx:
                continue
            substitute = str(ingredient_names.iloc[int(match_idx)])
            similarity = float(1.0 - dist)
            substitute_feat = (
                feature_lookup.loc[substitute]
                if not feature_lookup.empty and substitute in feature_lookup.index
                else None
            )
            substitute_nut = nutrition_lookup.loc[substitute]

            rating_delta = (
                _safe_delta(substitute_feat['avg_rating'], candidate_feat['avg_rating'])
                if candidate_feat is not None and substitute_feat is not None
                else None
            )
            sentiment_delta = (
                _safe_delta(
                    substitute_feat['avg_sentiment'], candidate_feat['avg_sentiment']
                )
                if candidate_feat is not None and substitute_feat is not None
                else None
            )
            protein_delta = _safe_delta(
                substitute_nut['protein_g_per_100g'],
                candidate_nut['protein_g_per_100g'],
            )
            saturated_fat_delta = _safe_delta(
                substitute_nut['saturated_fat_g_per_100g'],
                candidate_nut['saturated_fat_g_per_100g'],
            )
            sugar_delta = _safe_delta(
                substitute_nut['sugar_g_per_100g'],
                candidate_nut['sugar_g_per_100g'],
            )
            sodium_delta = _safe_delta(
                substitute_nut['sodium_g_per_100g'],
                candidate_nut['sodium_g_per_100g'],
            )
            calories_delta = _safe_delta(
                substitute_nut['calories_per_100g'],
                candidate_nut['calories_per_100g'],
            )
            health_delta = (
                0.4 * (protein_delta if protein_delta is not None else 0.0)
                - 0.25
                * (saturated_fat_delta if saturated_fat_delta is not None else 0.0)
                - 0.2 * (sugar_delta if sugar_delta is not None else 0.0)
                - 0.15 * (sodium_delta if sodium_delta is not None else 0.0)
            )
            recommendation_score = (
                0.6 * similarity
                + 0.2 * (rating_delta if rating_delta is not None else 0.0)
                + 0.1 * (sentiment_delta if sentiment_delta is not None else 0.0)
                + 0.1 * health_delta
            )
            rows.append(
                {
                    'candidate_ingredient': candidate,
                    'substitute_ingredient': substitute,
                    'substitute_similarity': round(similarity, 4),
                    'recommendation_score': round(float(recommendation_score), 4),
                    'rating_delta': _round_or_none(rating_delta),
                    'sentiment_delta': _round_or_none(sentiment_delta),
                    'protein_delta': _round_or_none(protein_delta),
                    'saturated_fat_delta': _round_or_none(saturated_fat_delta),
                    'sugar_delta': _round_or_none(sugar_delta),
                    'sodium_delta': _round_or_none(sodium_delta),
                    'calories_delta': _round_or_none(calories_delta),
                    'health_delta': round(float(health_delta), 4),
                }
            )
            selected_for_candidate += 1
            if selected_for_candidate >= SUBSTITUTION_TOP_K:
                break

    recommendations = pd.DataFrame(rows)
    if recommendations.empty:
        return _empty_substitution_engine_frame()

    recommendations = recommendations.drop_duplicates(
        subset=['candidate_ingredient', 'substitute_ingredient']
    ).sort_values(
        ['candidate_ingredient', 'recommendation_score'],
        ascending=[True, False],
    ).reset_index(drop=True)

    logger.info(
        'Substitution engine recommendations generated: %s rows across %s candidates.',
        f'{len(recommendations):,}',
        f'{recommendations["candidate_ingredient"].nunique():,}',
    )
    return recommendations


def _ollama_substitution_compatible(
    candidate_ingredient: str,
    alt_ingredient: str,
    cache: dict[tuple[str, str], bool],
) -> bool:
    """
    Uses a local Ollama model to decide if alt_ingredient is a plausible substitute
    for candidate_ingredient in general cooking contexts.
    """
    key = (candidate_ingredient.strip().lower(), alt_ingredient.strip().lower())
    if key in cache:
        return cache[key]

    prompt = (
        'Decide substitution compatibility for cooking ingredients. '
        'Respond as strict JSON only: {"compatible": true|false}. '
        f'Candidate ingredient: "{candidate_ingredient}". '
        f'Alternative ingredient: "{alt_ingredient}". '
        'Return true only if the alternative is generally a plausible substitute '
        'for the candidate (not merely commonly co-used).'
    )
    payload = {
        'model': OLLAMA_MODEL,
        'prompt': prompt,
        'stream': False,
        'format': 'json',
    }

    compatible = False
    try:
        resp = requests.post(
            f'{OLLAMA_BASE_URL.rstrip("/")}/api/generate',
            json=payload,
            timeout=OLLAMA_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        raw = resp.json().get('response', '{}')
        parsed = json.loads(raw) if isinstance(raw, str) else raw
        compatible = bool(parsed.get('compatible', False))
    except Exception as e:
        logger.warning(
            'Ollama compatibility gate failed for pair (%s -> %s): %s',
            candidate_ingredient,
            alt_ingredient,
            e,
        )
        compatible = False

    cache[key] = compatible
    return compatible


def _safe_to_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        txt = value.strip()
        if not txt:
            return []
        if txt.startswith('[') and txt.endswith(']'):
            try:
                parsed = ast.literal_eval(txt)
                if isinstance(parsed, list):
                    return [str(v).strip().lower() for v in parsed if str(v).strip()]
            except Exception:
                pass
        if '|' in txt:
            return [v.strip().lower() for v in txt.split('|') if v.strip()]
        return [txt.lower()]
    if hasattr(value, '__iter__'):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    return []


def _build_ingredient_recipe_map(recipes: pd.DataFrame) -> dict[str, set[int]]:
    if recipes.empty or 'id' not in recipes.columns or 'ingredients_canonical_list' not in recipes.columns:
        return {}
    exploded = (
        recipes[['id', 'ingredients_canonical_list']]
        .rename(columns={'id': 'recipe_id'})
        .explode('ingredients_canonical_list')
        .dropna(subset=['ingredients_canonical_list'])
    )
    exploded['ingredients_canonical_list'] = exploded['ingredients_canonical_list'].astype(str).str.lower()
    out: dict[str, set[int]] = {}
    for ing, grp in exploded.groupby('ingredients_canonical_list'):
        out[str(ing)] = set(grp['recipe_id'].astype(int).tolist())
    return out


def _build_ingredient_context_map(recipes: pd.DataFrame) -> dict[str, dict[str, bool]]:
    if recipes.empty or 'id' not in recipes.columns or 'ingredients_canonical_list' not in recipes.columns:
        return {}
    tags_col = 'tags' if 'tags' in recipes.columns else None
    if tags_col is None:
        return {}

    ctx = recipes[['id', 'ingredients_canonical_list', tags_col]].rename(columns={'id': 'recipe_id'}).copy()
    ctx['tags_list'] = ctx[tags_col].apply(_safe_to_list)
    ctx['is_baking'] = ctx['tags_list'].apply(
        lambda tags: any(
            any(k in tag for k in ['bake', 'baking', 'cookie', 'cake', 'dessert', 'bread', 'pastry'])
            for tag in tags
        )
    )
    exploded = ctx[['recipe_id', 'ingredients_canonical_list', 'is_baking']].explode('ingredients_canonical_list')
    exploded = exploded.dropna(subset=['ingredients_canonical_list'])
    exploded['ingredients_canonical_list'] = exploded['ingredients_canonical_list'].astype(str).str.lower()

    out: dict[str, dict[str, bool]] = {}
    for ing, grp in exploded.groupby('ingredients_canonical_list'):
        baking_rate = float(grp['is_baking'].mean()) if len(grp) else 0.0
        out[str(ing)] = {'is_baking': baking_rate >= 0.5}
    return out


def _ingredient_cooccurrence_score(candidate_recipe_ids: set[int], alt_recipe_ids: set[int]) -> float:
    if not candidate_recipe_ids or not alt_recipe_ids:
        return 0.0
    inter = len(candidate_recipe_ids.intersection(alt_recipe_ids))
    union = len(candidate_recipe_ids.union(alt_recipe_ids))
    if union == 0:
        return 0.0
    return inter / union


def _context_penalty(
    candidate_context: dict[str, bool],
    alt_context: dict[str, bool],
) -> float:
    """
    Penalize substitutes that violate dominant recipe context.
    Example: baking-heavy candidate should map to baking-compatible substitutes.
    """
    penalty = 0.0
    if candidate_context.get('is_baking', False):
        if not alt_context.get('is_baking', False):
            penalty += 0.15
    return penalty


# ---------------------------------------------------------------------------
# Utilities: load staged outputs (used by cluster.py, load.py)
# ---------------------------------------------------------------------------


def load_user_stats() -> pd.DataFrame:
    return pd.read_parquet(USER_STATS_STAGING)


def load_recipe_sentiment_ratings() -> pd.DataFrame:
    return pd.read_parquet(RECIPE_SENTIMENT_STAGING)


def load_ingredient_features() -> pd.DataFrame:
    return pd.read_parquet(INGREDIENT_FEATURES_STAGING)


def load_substitution_engine() -> pd.DataFrame:
    return pd.read_parquet(SUBSTITUTION_ENGINE_STAGING)
