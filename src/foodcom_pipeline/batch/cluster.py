"""
cluster.py
----------
User segmentation step for the Food.com batch pipeline.

Segments users into k clusters using K-Means based on their reviewing
behaviour. Cluster labels are stored in dim_user and used as a feature
in the XGBoost rating prediction model.

Responsibilities:
  1. Compute per-user aggregation features from interactions
  2. Determine optimal k via the elbow method (first run only)
  3. Fit K-Means and assign cluster labels
  4. Produce cluster profile statistics for the report / Streamlit dashboard
  5. Stage results for the load step

Features used for clustering:
  Taste profile (avg rating per ingredient category):
  - avg_rating_dairy        : avg rating given to dairy-heavy recipes
  - avg_rating_protein      : avg rating given to protein-heavy recipes
  - avg_rating_vegetable    : avg rating given to vegetable-heavy recipes
  - avg_rating_baking       : avg rating given to baking-heavy recipes
  - avg_rating_international: avg rating given to international-ingredient recipes
  Behavioural signals:
  - review_count            : total reviews submitted
  - recipe_diversity        : number of unique recipes reviewed
  - avg_sentiment_score     : mean VADER sentiment score

Cluster label heuristics (assigned after fitting based on centroid values):
  - Indulgent Baker      : highest avg_rating_baking
  - International Explorer: highest avg_rating_international
  - Protein-Forward Cook : highest avg_rating_protein
  - Health-Conscious Cook: highest avg_rating_vegetable
  - General Cook       : remainder
"""

import json
import logging

import numpy as np
import pandas as pd
from foodcom_pipeline.batch.features import load_user_stats
from foodcom_pipeline.batch.extract import STAGING_DIR
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CLUSTER_STAGING = STAGING_DIR / 'user_clusters.parquet'
ELBOW_STATS_PATH = STAGING_DIR / 'elbow_stats.json'
CLUSTER_PROFILE_PATH = STAGING_DIR / 'cluster_profiles.json'

K_RANGE = [4, 5, 6]  # k values to evaluate via silhouette score
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    # Taste profile — avg rating per ingredient category
    'avg_rating_dairy',
    'avg_rating_protein',
    'avg_rating_vegetable',
    'avg_rating_baking',
    'avg_rating_international',
    # Behavioural signals — activity and consistency
    'review_count',
    'recipe_diversity',
    'avg_sentiment_score',
]


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_clustering(**context) -> None:
    """
    Main clustering entry point.
      1. Load pre-computed user stats from features.py
      2. Determine optimal k via silhouette score
      3. Fit K-Means
      4. Assign interpretable labels
      5. Produce cluster profile report
      6. Stage results
    """
    logger.info('Starting clustering step.')

    # Load pre-aggregated user stats — computed in the previous DAG step
    user_features = load_user_stats()

    if len(user_features) == 0:
        logger.warning(
            'No users in user_stats — skipping K-Means and writing empty cluster output.'
        )
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            columns=['user_id', 'cluster_id', 'cluster_label'],
        ).to_parquet(CLUSTER_STAGING, index=False)
        context['ti'].xcom_push(key='n_clusters', value=0)
        context['ti'].xcom_push(key='n_users_clustered', value=0)
        return

    from foodcom_pipeline.batch.sentiment import load_sentiment_interactions
    from foodcom_pipeline.batch.clean import load_cleaned_recipes
    from foodcom_pipeline.batch.features import load_ingredient_features

    interactions = load_sentiment_interactions()
    recipes = load_cleaned_recipes()
    ingredient_features = load_ingredient_features()

    logger.info(
        f'Loaded user feature matrix: '
        f'{user_features.shape[0]:,} users × {len(CLUSTER_FEATURES)} features'
    )

    k = _determine_optimal_k(user_features)

    user_features, scaler, kmeans = _fit_kmeans(user_features, k)
    user_features = _assign_labels(user_features, kmeans, scaler)

    _produce_cluster_profiles(user_features, interactions, recipes, ingredient_features)

    user_features.to_parquet(CLUSTER_STAGING, index=False)
    logger.info(f'Clustering complete. Staged to {CLUSTER_STAGING}')

    context['ti'].xcom_push(key='n_clusters', value=k)
    context['ti'].xcom_push(key='n_users_clustered', value=len(user_features))


# ---------------------------------------------------------------------------
# Step 2: Determine optimal k
# ---------------------------------------------------------------------------


def _determine_optimal_k(user_features: pd.DataFrame) -> int:
    """
    Always recomputes silhouette scores for k in K_RANGE on current data and selects
    the k with the highest score. If the chosen k differs from the previous run, a
    warning is logged so the team knows cluster labels and interpretations may need review.

    Results are written to ELBOW_STATS_PATH after every run.
    """
    previous_k: int | None = None
    if ELBOW_STATS_PATH.exists():
        with open(ELBOW_STATS_PATH) as f:
            previous_k = json.load(f).get('chosen_k')

    logger.info(f'Running silhouette analysis for k in {K_RANGE}...')
    X = _scale_features(user_features)
    silhouettes = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sample_size = min(10_000, len(X))
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X), sample_size, replace=False)
        sil = silhouette_score(X[idx], labels[idx])
        silhouettes.append(float(sil))
        logger.info(f'  k={k}: silhouette={sil:.4f}')

    chosen_k = K_RANGE[int(np.argmax(silhouettes))]
    logger.info(f'Chosen k={chosen_k} (highest silhouette score)')

    if previous_k is not None and chosen_k != previous_k:
        logger.warning(
            f'Optimal k changed from {previous_k} to {chosen_k} — '
            'cluster labels and segment interpretations should be reviewed.'
        )

    stats = {'k_range': K_RANGE, 'silhouettes': silhouettes, 'chosen_k': chosen_k}
    with open(ELBOW_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    _log_silhouette_table(K_RANGE, silhouettes, chosen_k)
    return chosen_k


def _log_silhouette_table(
    k_range: list[int],
    silhouettes: list[float],
    chosen_k: int,
) -> None:
    logger.info('=' * 36)
    logger.info('  Silhouette Score Results')
    logger.info('=' * 36)
    logger.info(f"  {'k':>4}  {'Silhouette':>12}")
    for k, sil in zip(k_range, silhouettes):
        marker = ' ← chosen' if k == chosen_k else ''
        logger.info(f'  {k:>4}  {sil:>12.4f}{marker}')
    logger.info('=' * 36)


# ---------------------------------------------------------------------------
# Step 3: Fit K-Means
# ---------------------------------------------------------------------------


def _scale_features(user_features: pd.DataFrame) -> np.ndarray:
    """StandardScale the feature matrix for K-Means."""
    scaler = StandardScaler()
    return scaler.fit_transform(user_features[CLUSTER_FEATURES])


def _fit_kmeans(
    user_features: pd.DataFrame, k: int
) -> tuple[pd.DataFrame, StandardScaler, KMeans]:
    """
    Fit K-Means with k clusters.
    Uses n_init=10 to mitigate sensitivity to random initialization.
    Returns the updated DataFrame, fitted scaler, and fitted KMeans.
    """
    logger.info(f'Fitting K-Means with k={k}...')

    scaler = StandardScaler()
    X = scaler.fit_transform(user_features[CLUSTER_FEATURES])

    kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    user_features['cluster_id'] = kmeans.fit_predict(X)

    logger.info(
        f'K-Means fit complete. '
        f'Inertia: {kmeans.inertia_:.1f}. '
        f'Cluster sizes:\n'
        + '\n'.join(
            f'  cluster {c}: {n:,} users'
            for c, n in user_features['cluster_id'].value_counts().sort_index().items()
        )
    )
    return user_features, scaler, kmeans


# ---------------------------------------------------------------------------
# Step 4: Assign interpretable labels
# ---------------------------------------------------------------------------


def _assign_labels(
    user_features: pd.DataFrame,
    kmeans: KMeans,
    scaler: StandardScaler,
) -> pd.DataFrame:
    """
    Assign human-readable labels to clusters based on centroid values.

    Strategy:
      - Inverse-transform centroids back to original feature scale
      - Rank clusters on each feature
      - Apply heuristic rules to assign labels
      - Any cluster not matched by a rule gets "Segment {id}"
    """
    # Inverse transform centroids to original scale
    centroids = pd.DataFrame(
        scaler.inverse_transform(kmeans.cluster_centers_),
        columns=CLUSTER_FEATURES,
    )
    centroids['cluster_id'] = range(len(centroids))

    logger.info('Cluster centroids (original scale):')
    logger.info('\n' + centroids.to_string(index=False))

    label_map = _heuristic_label_assignment(centroids)
    user_features['cluster_label'] = user_features['cluster_id'].map(label_map)

    logger.info('Cluster label assignment:')
    for cid, label in sorted(label_map.items()):
        n = (user_features['cluster_id'] == cid).sum()
        logger.info(f"  Cluster {cid} → '{label}' ({n:,} users)")

    return user_features


def _heuristic_label_assignment(centroids: pd.DataFrame) -> dict[int, str]:
    """
    Assigns labels based on which ingredient category each cluster rates most highly.

    Rules (applied in order, first match wins per category feature):
      Indulgent Baker       : highest avg_rating_baking
      International Explorer: highest avg_rating_international
      Protein-Forward Cook  : highest avg_rating_protein
      Health-Conscious Cook : highest avg_rating_vegetable
      General Cook        : remainder
    """
    label_map = {}
    used_ids = set()

    def rank_cluster(feature: str) -> int:
        sorted_c = centroids.sort_values(feature, ascending=False)
        for cid in sorted_c['cluster_id']:
            if cid not in used_ids:
                return int(cid)
        return -1

    assignments = [
        ('avg_rating_baking',        'Indulgent Baker'),
        ('avg_rating_international', 'International Explorer'),
        ('avg_rating_protein',       'Protein-Forward Cook'),
        ('avg_rating_vegetable',     'Health-Conscious Cook'),
    ]

    for feature, label in assignments:
        if feature not in centroids.columns:
            continue
        cid = rank_cluster(feature)
        if cid >= 0:
            label_map[cid] = label
            used_ids.add(cid)

    for cid in centroids['cluster_id']:
        if int(cid) not in used_ids:
            label_map[int(cid)] = 'General Cook'

    return label_map


# ---------------------------------------------------------------------------
# Step 5: Cluster profile statistics
# ---------------------------------------------------------------------------


def _produce_cluster_profiles(
    user_features: pd.DataFrame,
    interactions: pd.DataFrame,
    recipes: pd.DataFrame,
    ingredient_features: pd.DataFrame,
) -> None:
    """
    Computes and saves a detailed profile for each cluster.

    Per-cluster statistics:
      - Size (n users, % of total)
      - Feature means and standard deviations
      - Rating distribution (% of each star rating)
      - Avg rating-sentiment gap
      - Top 5 canonical ingredients (by review frequency in cluster)
      - Activity peak month (modal review month)
      - Avg exclamation count per review (enthusiasm proxy)
      - Substitute exposure rate (% of reviewed recipes containing ≥1 substitution candidate)

    Saved to CLUSTER_PROFILE_PATH as JSON for use in the Streamlit dashboard
    and the report discussion section.
    """
    logger.info('Computing cluster profiles...')

    profiles = {}
    total_users = len(user_features)

    # Join cluster labels onto interactions for rating distribution and metrics
    interactions_with_clusters = interactions.merge(
        user_features[['user_id', 'cluster_id', 'cluster_label']],
        on='user_id',
        how='left',
    )

    # Pre-compute substitution candidate set for exposure rate
    sub_candidates: set[str] = set()
    if (
        len(ingredient_features) > 0
        and 'is_substitution_candidate' in ingredient_features.columns
        and 'canonical_ingredient' in ingredient_features.columns
    ):
        sub_candidates = set(
            ingredient_features.loc[
                ingredient_features['is_substitution_candidate'], 'canonical_ingredient'
            ]
        )

    for cluster_id in sorted(user_features['cluster_id'].unique()):
        cluster_users = user_features[user_features['cluster_id'] == cluster_id]
        cluster_label = cluster_users['cluster_label'].iloc[0]
        cluster_interactions = interactions_with_clusters[
            interactions_with_clusters['cluster_id'] == cluster_id
        ]

        n_users = len(cluster_users)
        pct_users = n_users / total_users * 100

        # Feature statistics
        feature_stats = {}
        for feature in CLUSTER_FEATURES:
            feature_stats[feature] = {
                'mean': round(float(cluster_users[feature].mean()), 4),
                'std': round(float(cluster_users[feature].std()), 4),
                'min': round(float(cluster_users[feature].min()), 4),
                'max': round(float(cluster_users[feature].max()), 4),
            }

        # Rating distribution
        rating_dist = (
            cluster_interactions['rating']
            .value_counts(normalize=True)
            .sort_index()
            .round(4)
            .to_dict()
        )
        rating_dist = {int(k): float(v) for k, v in rating_dist.items()}

        # Avg rating-sentiment gap for the cluster
        gap_mean = (
            float(cluster_interactions['rating_sentiment_gap'].mean())
            if 'rating_sentiment_gap' in cluster_interactions.columns
            else None
        )

        # Top 5 canonical ingredients by frequency across cluster's reviewed recipes
        top_ingredients = _compute_top_ingredients(cluster_interactions, recipes)

        # Activity peak month (modal month of reviews)
        peak_month = _compute_peak_month(cluster_interactions)

        # Avg exclamation count per review (enthusiasm signal)
        avg_exclamations = _compute_avg_exclamations(cluster_interactions)

        # Substitute exposure rate
        sub_exposure = _compute_substitute_exposure(
            cluster_interactions, recipes, sub_candidates
        )

        profile = {
            'cluster_id': int(cluster_id),
            'cluster_label': cluster_label,
            'n_users': n_users,
            'pct_users': round(pct_users, 2),
            'feature_stats': feature_stats,
            'rating_distribution': rating_dist,
            'avg_rating_sentiment_gap': round(gap_mean, 4) if gap_mean is not None else None,
            'top_ingredients': top_ingredients,
            'peak_month': peak_month,
            'avg_exclamation_count': round(avg_exclamations, 4) if avg_exclamations is not None else None,
            'substitute_exposure_rate': round(sub_exposure, 4) if sub_exposure is not None else None,
        }
        profiles[str(cluster_id)] = profile

        _log_cluster_profile(profile)

    # Overall summary
    summary = {
        'total_users': total_users,
        'n_clusters': int(user_features['cluster_id'].nunique()),
        'clusters': profiles,
    }

    with open(CLUSTER_PROFILE_PATH, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f'Cluster profiles saved to {CLUSTER_PROFILE_PATH}')


def _compute_top_ingredients(
    cluster_interactions: pd.DataFrame,
    recipes: pd.DataFrame,
    top_n: int = 5,
) -> list[str]:
    """Top-N canonical ingredients by frequency across recipes reviewed by this cluster."""
    if 'ingredients_canonical_list' not in recipes.columns or cluster_interactions.empty:
        return []
    cluster_recipe_ids = cluster_interactions['recipe_id'].unique()
    cluster_recipes = recipes[recipes['id'].isin(cluster_recipe_ids)]
    if cluster_recipes.empty:
        return []
    return (
        cluster_recipes['ingredients_canonical_list']
        .explode()
        .dropna()
        .value_counts()
        .head(top_n)
        .index.tolist()
    )


def _compute_peak_month(cluster_interactions: pd.DataFrame) -> str | None:
    """Modal calendar month of reviews (e.g. 'December')."""
    if 'date' not in cluster_interactions.columns or cluster_interactions.empty:
        return None
    months = pd.to_datetime(cluster_interactions['date'], errors='coerce').dt.month_name()
    months = months.dropna()
    if months.empty:
        return None
    return str(months.mode().iloc[0])


def _compute_avg_exclamations(cluster_interactions: pd.DataFrame) -> float | None:
    """Mean number of exclamation marks per review — proxy for enthusiasm."""
    if 'review' not in cluster_interactions.columns or cluster_interactions.empty:
        return None
    return float(cluster_interactions['review'].fillna('').str.count('!').mean())


def _compute_substitute_exposure(
    cluster_interactions: pd.DataFrame,
    recipes: pd.DataFrame,
    sub_candidates: set[str],
) -> float | None:
    """
    Fraction of reviewed recipes that contain at least one substitution-candidate
    ingredient. Returns None when substitution candidate data is unavailable.
    """
    if (
        not sub_candidates
        or 'ingredients_canonical_list' not in recipes.columns
        or cluster_interactions.empty
    ):
        return None
    cluster_recipe_ids = cluster_interactions['recipe_id'].unique()
    cluster_recipes = recipes[recipes['id'].isin(cluster_recipe_ids)]
    if cluster_recipes.empty:
        return None
    has_sub = cluster_recipes['ingredients_canonical_list'].apply(
        lambda ing_list: any(ing in sub_candidates for ing in ing_list)
        if ing_list is not None and len(ing_list) > 0
        else False
    )
    return float(has_sub.mean())


def _log_cluster_profile(profile: dict) -> None:
    """Pretty-prints a single cluster profile to the Airflow log."""
    cid = profile['cluster_id']
    label = profile['cluster_label']
    n = profile['n_users']
    pct = profile['pct_users']
    gap = profile['avg_rating_sentiment_gap']

    logger.info('─' * 52)
    logger.info(f'  Cluster {cid}: {label}  ({n:,} users, {pct:.1f}%)')
    logger.info('─' * 52)
    for feat, stats in profile['feature_stats'].items():
        logger.info(f"  {feat:<26}: mean={stats['mean']:.3f}  std={stats['std']:.3f}")
    logger.info(
        '  Rating distribution      : '
        + '  '.join(f'{k}★={v:.1%}' for k, v in profile['rating_distribution'].items())
    )
    if gap is not None:
        logger.info(f'  Avg rating-sentiment gap : {gap:.4f}')
    if profile.get('top_ingredients'):
        logger.info(f"  Top ingredients          : {', '.join(profile['top_ingredients'])}")
    if profile.get('peak_month'):
        logger.info(f"  Activity peak month      : {profile['peak_month']}")
    if profile.get('avg_exclamation_count') is not None:
        logger.info(f"  Avg exclamation count    : {profile['avg_exclamation_count']:.4f}")
    if profile.get('substitute_exposure_rate') is not None:
        logger.info(f"  Substitute exposure rate : {profile['substitute_exposure_rate']:.2%}")
    logger.info('─' * 52)


# ---------------------------------------------------------------------------
# Utility: load cluster data (used by load.py)
# ---------------------------------------------------------------------------


def load_user_clusters() -> pd.DataFrame:
    return pd.read_parquet(CLUSTER_STAGING)
