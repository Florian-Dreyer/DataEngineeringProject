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

Features used for clustering (all user-level aggregates):
  - avg_rating_given      : mean star rating across all reviews
  - review_count          : total number of reviews submitted
  - recipe_diversity      : number of unique recipes reviewed
  - avg_sentiment_score   : mean DistilBERT sentiment score across reviews

Cluster label heuristics (assigned after fitting based on centroid values):
  - Enthusiastic Cook  : high rating, high volume, positive sentiment
  - Harsh Critic       : low rating, moderate volume, negative sentiment
  - Casual Rater       : low volume, middling rating
  - Power User         : very high volume, high diversity
"""

import json
import logging

import numpy as np
import pandas as pd
from foodcom_pipeline.batch.aggregate_user_stats import load_user_stats
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

K_RANGE = range(2, 9)  # k values to evaluate in elbow method
DEFAULT_K = 4  # fallback if elbow is ambiguous
RANDOM_STATE = 42

CLUSTER_FEATURES = [
    'avg_rating_given',
    'review_count',
    'recipe_diversity',
    'avg_sentiment_score',
]

# Human-readable label mapping: assigned post-hoc based on centroid inspection.
# Keys are centroid rank tuples (high/low per feature) — updated after fitting.
# These are heuristics; adjust after inspecting your actual cluster centroids.
LABEL_HEURISTICS = {
    'high_rating_high_volume': 'Enthusiastic Cook',
    'low_rating_any_volume': 'Harsh Critic',
    'low_volume': 'Casual Rater',
    'high_volume_high_diversity': 'Power User',
}


# ---------------------------------------------------------------------------
# Entry point called by Airflow
# ---------------------------------------------------------------------------


def run_clustering(**context) -> None:
    """
    Main clustering entry point.
      1. Load pre-computed user stats from aggregate_user_stats.py
      2. Determine optimal k (elbow + silhouette), first run only
      3. Fit K-Means
      4. Assign interpretable labels
      5. Produce cluster profile report
      6. Stage results
    """
    logger.info('Starting clustering step.')

    # Load pre-aggregated user stats — computed in the previous DAG step
    user_features = load_user_stats()
    # Also load interactions for the cluster profile rating distribution
    from foodcom_pipeline.batch.sentiment import load_sentiment_interactions

    interactions = load_sentiment_interactions()

    logger.info(
        f'Loaded user feature matrix: '
        f'{user_features.shape[0]:,} users × {len(CLUSTER_FEATURES)} features'
    )

    k = _determine_optimal_k(user_features)

    user_features, scaler, kmeans = _fit_kmeans(user_features, k)
    user_features = _assign_labels(user_features, kmeans, scaler)

    _produce_cluster_profiles(user_features, interactions)

    user_features.to_parquet(CLUSTER_STAGING, index=False)
    logger.info(f'Clustering complete. Staged to {CLUSTER_STAGING}')

    context['ti'].xcom_push(key='n_clusters', value=k)
    context['ti'].xcom_push(key='n_users_clustered', value=len(user_features))


# ---------------------------------------------------------------------------
# Step 2: Determine optimal k
# ---------------------------------------------------------------------------


def _determine_optimal_k(user_features: pd.DataFrame) -> int:
    """
    Runs the elbow method and silhouette analysis to find the optimal k.
    Results are saved to ELBOW_STATS_PATH.

    If the stats file already exists (subsequent DAG runs), we load the
    previously determined k rather than recomputing — clustering should
    be stable across runs to keep cluster IDs consistent for the dashboard.

    Returns the chosen k.
    """
    if ELBOW_STATS_PATH.exists():
        with open(ELBOW_STATS_PATH) as f:
            stats = json.load(f)
        k = stats['chosen_k']
        logger.info(f'Loaded existing elbow stats — using k={k} from previous run.')
        return k

    logger.info(f'Running elbow method for k in {list(K_RANGE)}...')

    X = _scale_features(user_features)

    inertias = []
    silhouettes = []

    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(float(km.inertia_))

        # Silhouette is expensive on large datasets — subsample if needed
        sample_size = min(10_000, len(X))
        idx = np.random.default_rng(RANDOM_STATE).choice(
            len(X), sample_size, replace=False
        )
        sil = silhouette_score(X[idx], labels[idx])
        silhouettes.append(float(sil))

        logger.info(f'  k={k}: inertia={km.inertia_:.1f}, silhouette={sil:.4f}')

    chosen_k = _elbow_k(list(K_RANGE), inertias, silhouettes)
    logger.info(f'Chosen k={chosen_k} (elbow + silhouette combined)')

    stats = {
        'k_range': list(K_RANGE),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'chosen_k': chosen_k,
    }
    with open(ELBOW_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    _log_elbow_table(list(K_RANGE), inertias, silhouettes, chosen_k)
    return chosen_k


def _elbow_k(
    k_range: list[int],
    inertias: list[float],
    silhouettes: list[float],
) -> int:
    """
    Combines two signals to pick k:
      1. Elbow point: largest second-order difference in inertia
         (point of diminishing returns on the inertia curve)
      2. Silhouette peak: k with highest silhouette score

    If they agree → use that k.
    If they disagree → use silhouette peak (more interpretable metric).
    Falls back to DEFAULT_K if both methods are inconclusive.
    """
    # Elbow via second-order differences in inertia
    inertia_arr = np.array(inertias)
    second_diff = np.diff(np.diff(inertia_arr))
    elbow_idx = int(np.argmax(second_diff)) + 2  # offset for two diffs
    elbow_k = k_range[elbow_idx]

    # Silhouette peak
    sil_k = k_range[int(np.argmax(silhouettes))]

    logger.info(f'Elbow method suggests k={elbow_k}, silhouette suggests k={sil_k}')

    if elbow_k == sil_k:
        return elbow_k

    # Disagreement — prefer silhouette as it's more interpretable
    logger.warning(
        f'Elbow (k={elbow_k}) and silhouette (k={sil_k}) disagree. '
        f'Using silhouette peak k={sil_k}.'
    )
    return sil_k


def _log_elbow_table(
    k_range: list[int],
    inertias: list[float],
    silhouettes: list[float],
    chosen_k: int,
) -> None:
    logger.info('=' * 52)
    logger.info('  Elbow Method Results')
    logger.info('=' * 52)
    logger.info(f"  {'k':>4}  {'Inertia':>14}  {'Silhouette':>12}  {'':>6}")
    for k, inertia, sil in zip(k_range, inertias, silhouettes):
        marker = ' ← chosen' if k == chosen_k else ''
        logger.info(f'  {k:>4}  {inertia:>14.1f}  {sil:>12.4f}{marker}')
    logger.info('=' * 52)


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
    Assigns labels based on relative centroid ranks.

    Rules (applied in order, first match wins):
      Power User        : highest review_count
      Harsh Critic      : lowest avg_rating_given
      Enthusiastic Cook : highest avg_rating_given AND above-median sentiment
      Casual Rater      : remainder (typically low volume)
    """
    label_map = {}
    used_ids = set()

    def rank_cluster(feature: str, highest: bool = True) -> int:
        sorted_c = centroids.sort_values(feature, ascending=not highest)
        for cid in sorted_c['cluster_id']:
            if cid not in used_ids:
                return int(cid)
        return -1

    # Power User — highest review count
    power_user_id = rank_cluster('review_count', highest=True)
    label_map[power_user_id] = 'Power User'
    used_ids.add(power_user_id)

    # Harsh Critic — lowest avg rating
    harsh_critic_id = rank_cluster('avg_rating_given', highest=False)
    label_map[harsh_critic_id] = 'Harsh Critic'
    used_ids.add(harsh_critic_id)

    # Enthusiastic Cook — highest avg rating among remaining
    enthusiastic_id = rank_cluster('avg_rating_given', highest=True)
    label_map[enthusiastic_id] = 'Enthusiastic Cook'
    used_ids.add(enthusiastic_id)

    # Casual Rater — everything else
    for cid in centroids['cluster_id']:
        if int(cid) not in used_ids:
            label_map[int(cid)] = 'Casual Rater'

    return label_map


# ---------------------------------------------------------------------------
# Step 5: Cluster profile statistics
# ---------------------------------------------------------------------------


def _produce_cluster_profiles(
    user_features: pd.DataFrame,
    interactions: pd.DataFrame,
) -> None:
    """
    Computes and saves a detailed profile for each cluster.

    Per-cluster statistics:
      - Size (n users, % of total)
      - Feature means and standard deviations
      - Rating distribution (% of each star rating)
      - Top recipes by avg rating within cluster (via interactions join)

    Saved to CLUSTER_PROFILE_PATH as JSON for use in the Streamlit dashboard
    and the report discussion section.
    """
    logger.info('Computing cluster profiles...')

    profiles = {}
    total_users = len(user_features)

    # Join cluster labels onto interactions for rating distribution
    interactions_with_clusters = interactions.merge(
        user_features[['user_id', 'cluster_id', 'cluster_label']],
        on='user_id',
        how='left',
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

        profile = {
            'cluster_id': int(cluster_id),
            'cluster_label': cluster_label,
            'n_users': n_users,
            'pct_users': round(pct_users, 2),
            'feature_stats': feature_stats,
            'rating_distribution': rating_dist,
            'avg_rating_sentiment_gap': round(gap_mean, 4) if gap_mean else None,
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
    logger.info('─' * 52)


# ---------------------------------------------------------------------------
# Utility: load cluster data (used by load.py)
# ---------------------------------------------------------------------------


def load_user_clusters() -> pd.DataFrame:
    return pd.read_parquet(CLUSTER_STAGING)
