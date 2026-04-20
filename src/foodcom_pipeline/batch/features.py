"""
features.py
-----------
Feature engineering step for the Food.com batch pipeline.

Computes all derived features needed before clustering:
  1. Per-user aggregate stats (avg_rating, review_count, sentiment stats, etc.)
  2. Per-recipe Bayesian sentiment_rating (inverse-frequency + Bayesian shrinkage)
  3. Ingredient-level features (avg rating, avg sentiment, recipe count)
  4. Substitution candidate flags (low-rated ingredients for the substitution engine)

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
  is_substitution_candidate : True when avg_rating < threshold and recipe_count >= min_count
  trend_index               : Google Trends index (null until extract_google_trends is added)
"""

import ast
import logging
import os
from pathlib import Path

import pandas as pd
from foodcom_pipeline.batch.clean import load_cleaned_recipes
from foodcom_pipeline.batch.extract import STAGING_DIR
from foodcom_pipeline.batch.sentiment import load_sentiment_interactions

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

USER_STATS_STAGING = STAGING_DIR / 'user_stats.parquet'
RECIPE_SENTIMENT_STAGING = STAGING_DIR / 'recipe_sentiment_ratings.parquet'
INGREDIENT_FEATURES_STAGING = STAGING_DIR / 'ingredient_features.parquet'

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

# Substitution candidate thresholds
SUBSTITUTION_RATING_THRESHOLD = 3.5  # avg_rating below this → candidate
SUBSTITUTION_MIN_RECIPE_COUNT = 5    # ingredient must appear in at least this many recipes


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

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    user_stats.to_parquet(USER_STATS_STAGING, index=False)
    recipe_ratings.to_parquet(RECIPE_SENTIMENT_STAGING, index=False)
    ingredient_features.to_parquet(INGREDIENT_FEATURES_STAGING, index=False)

    n_candidates = int(ingredient_features['is_substitution_candidate'].sum())
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

    trend_index is left null until extract_google_trends is wired in.
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
                'is_substitution_candidate',
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

    ingr_features['is_substitution_candidate'] = (
        (ingr_features['avg_rating'] < SUBSTITUTION_RATING_THRESHOLD)
        & (ingr_features['recipe_count'] >= SUBSTITUTION_MIN_RECIPE_COUNT)
    )

    # Placeholder for Google Trends integration
    ingr_features['trend_index'] = None

    n_candidates = int(ingr_features['is_substitution_candidate'].sum())
    logger.info(
        f'Ingredient features computed for {len(ingr_features):,} canonical ingredients. '
        f'{n_candidates} substitution candidates '
        f'(avg_rating < {SUBSTITUTION_RATING_THRESHOLD}, '
        f'recipe_count >= {SUBSTITUTION_MIN_RECIPE_COUNT}).'
    )
    return ingr_features


# ---------------------------------------------------------------------------
# Utilities: load staged outputs (used by cluster.py, load.py)
# ---------------------------------------------------------------------------


def load_user_stats() -> pd.DataFrame:
    return pd.read_parquet(USER_STATS_STAGING)


def load_recipe_sentiment_ratings() -> pd.DataFrame:
    return pd.read_parquet(RECIPE_SENTIMENT_STAGING)


def load_ingredient_features() -> pd.DataFrame:
    return pd.read_parquet(INGREDIENT_FEATURES_STAGING)


# ---------------------------------------------------------------------------
# Google Trends keyword generation from recipe tags
# ---------------------------------------------------------------------------


def generate_recipe_tag_keywords() -> pd.DataFrame:
    """
    Extracts unique recipe tags from cleaned recipes and saves them as keywords
    for Google Trends extraction.

    Tags are parsed from the string representation stored in the recipes table,
    deduplicated, sorted alphabetically, and written to config/trends_keywords.txt.

    Returns:
        DataFrame with columns: ['keyword'] containing unique recipe tags.
    """
    recipes = load_cleaned_recipes()

    if 'tags' not in recipes.columns:
        logger.warning('Tags column not found in recipes — skipping keyword generation.')
        return pd.DataFrame(columns=['keyword'])

    all_tags: set[str] = set()

    for tags_value in recipes['tags'].dropna():
        try:
            # Tags stored as string representation of list
            if isinstance(tags_value, str):
                tags_list = ast.literal_eval(tags_value)
                if isinstance(tags_list, list):
                    all_tags.update([str(tag).strip().lower() for tag in tags_list if tag])
        except (ValueError, SyntaxError):
            # Skip malformed tag entries
            pass

    if not all_tags:
        logger.warning('No valid tags found in recipes.')
        return pd.DataFrame(columns=['keyword'])

    # Sort alphabetically and remove empty strings
    sorted_tags = sorted([tag for tag in all_tags if tag])

    # Write to config/trends_keywords.txt
    config_dir = Path(__file__).resolve().parents[3] / 'config'
    config_dir.mkdir(parents=True, exist_ok=True)
    keywords_file = config_dir / 'trends_keywords.txt'

    with open(keywords_file, 'w', encoding='utf-8') as f:
        for tag in sorted_tags:
            f.write(f"{tag}\n")

    logger.info(
        f'Generated {len(sorted_tags):,} unique recipe tags from {len(recipes):,} recipes. '
        f'Keywords written to {keywords_file}'
    )

    return pd.DataFrame({'keyword': sorted_tags})
