"""
ingredient_clusters.py
----------------------
Builds canonical ingredient clusters from staged USDA nutrient vectors.

This runs at the extract/feature boundary:
  usda_nutrients.parquet -> ingredient_clusters.parquet

Cluster IDs are later consumed by the substitution engine as a first-pass
compatibility gate before pair scoring.
"""

import logging

import numpy as np
import pandas as pd
from foodcom_pipeline.batch.extract import STAGING_DIR, USDA_NUTRIENTS_STAGING
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

INGREDIENT_CLUSTERS_STAGING = STAGING_DIR / 'ingredient_clusters.parquet'

NUTRIENT_COLS = [
    'calories_per_100g',
    'protein_g_per_100g',
    'fat_g_per_100g',
    'saturated_fat_g_per_100g',
    'sugar_g_per_100g',
    'sodium_g_per_100g',
    'carbs_g_per_100g',
]
K_RANGE = [60, 65, 70, 75, 80, 85, 90, 95, 100]
RANDOM_STATE = 42
TOP_SILHOUETTE_CANDIDATES = 3
SILHOUETTE_CLEAR_GAP = 0.002


def run_ingredient_clustering(**context) -> None:
    """
    Fits K-Means on standardized USDA nutrient vectors per canonical ingredient.

    Only ingredients present in usda_nutrients.parquet with at least one nutrient
    value are clustered. Missing nutrient values are median-imputed per column.
    """
    if not USDA_NUTRIENTS_STAGING.is_file():
        logger.warning(
            'No %s found; writing empty ingredient cluster map.',
            USDA_NUTRIENTS_STAGING,
        )
        _write_empty_clusters()
        context['ti'].xcom_push(key='ingredient_cluster_count', value=0)
        context['ti'].xcom_push(key='ingredient_cluster_rows', value=0)
        return

    usda = pd.read_parquet(USDA_NUTRIENTS_STAGING)
    required = {'canonical_ingredient', *NUTRIENT_COLS}
    missing = required - set(usda.columns)
    if missing:
        raise ValueError(f'USDA staging missing required columns: {sorted(missing)}')

    base = usda[['canonical_ingredient', *NUTRIENT_COLS]].copy()
    base = base.dropna(subset=['canonical_ingredient']).drop_duplicates(
        'canonical_ingredient'
    )
    nutrient_non_null = base[NUTRIENT_COLS].notna().sum(axis=1)
    base = base[nutrient_non_null > 0].copy()
    base['nutrient_feature_count'] = nutrient_non_null.loc[base.index].astype(int)

    if base.empty:
        logger.warning('USDA staging has no nutrient-populated ingredient rows.')
        _write_empty_clusters()
        context['ti'].xcom_push(key='ingredient_cluster_count', value=0)
        context['ti'].xcom_push(key='ingredient_cluster_rows', value=0)
        return

    X = base[NUTRIENT_COLS].copy()
    for col in NUTRIENT_COLS:
        median = float(X[col].median()) if X[col].notna().any() else 0.0
        X[col] = X[col].fillna(median)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    chosen_k = _choose_k(X_scaled)
    km = KMeans(n_clusters=chosen_k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X_scaled)

    out = base[['canonical_ingredient', 'nutrient_feature_count']].copy()
    out['ingredient_cluster_id'] = labels.astype(int)
    cluster_counts = (
        out['ingredient_cluster_id'].value_counts().sort_index().to_dict()
    )
    for cluster_id, ingredient_count in cluster_counts.items():
        logger.info(
            'Ingredient cluster debug: selected_k=%s cluster_id=%s ingredient_count=%s',
            chosen_k,
            int(cluster_id),
            int(ingredient_count),
        )

    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(INGREDIENT_CLUSTERS_STAGING, index=False)
    logger.info(
        'Ingredient clustering complete: %s rows across %s clusters.',
        f'{len(out):,}',
        out['ingredient_cluster_id'].nunique(),
    )
    context['ti'].xcom_push(
        key='ingredient_cluster_count',
        value=int(out['ingredient_cluster_id'].nunique()),
    )
    context['ti'].xcom_push(key='ingredient_cluster_rows', value=int(len(out)))
    context['ti'].xcom_push(key='ingredient_cluster_k', value=int(chosen_k))


def _choose_k(X_scaled: np.ndarray) -> int:
    n_rows = len(X_scaled)
    if n_rows < 3:
        return 1

    candidates = [k for k in K_RANGE if 2 <= k < n_rows]
    if not candidates:
        return 1

    scores: list[dict[str, float | int]] = []
    for k in candidates:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X_scaled)
        silhouette = float(silhouette_score(X_scaled, labels))
        calinski_harabasz = float(calinski_harabasz_score(X_scaled, labels))
        davies_bouldin = float(davies_bouldin_score(X_scaled, labels))
        scores.append(
            {
                'k': int(k),
                'silhouette': silhouette,
                'calinski_harabasz': calinski_harabasz,
                'davies_bouldin': davies_bouldin,
            }
        )
        logger.info(
            (
                'Ingredient cluster metrics: k=%s silhouette=%.4f '
                'calinski_harabasz=%.2f davies_bouldin=%.4f'
            ),
            k,
            silhouette,
            calinski_harabasz,
            davies_bouldin,
        )

    ranked_by_silhouette = sorted(
        scores, key=lambda row: row['silhouette'], reverse=True
    )

    if len(ranked_by_silhouette) == 1:
        return int(ranked_by_silhouette[0]['k'])

    if (
        ranked_by_silhouette[0]['silhouette'] - ranked_by_silhouette[1]['silhouette']
        >= SILHOUETTE_CLEAR_GAP
    ):
        chosen = ranked_by_silhouette[0]
        logger.info(
            'Ingredient cluster k selection: clear silhouette winner k=%s (gap=%.4f).',
            int(chosen['k']),
            float(
                ranked_by_silhouette[0]['silhouette']
                - ranked_by_silhouette[1]['silhouette']
            ),
        )
        return int(chosen['k'])

    top_candidates = ranked_by_silhouette[:TOP_SILHOUETTE_CANDIDATES]
    chosen = min(
        top_candidates,
        key=lambda row: (-row['calinski_harabasz'], row['davies_bouldin']),
    )
    logger.info(
        (
            'Ingredient cluster k selection: top-%s silhouette tie-break -> k=%s '
            '(silhouette=%.4f calinski_harabasz=%.2f davies_bouldin=%.4f).'
        ),
        TOP_SILHOUETTE_CANDIDATES,
        int(chosen['k']),
        float(chosen['silhouette']),
        float(chosen['calinski_harabasz']),
        float(chosen['davies_bouldin']),
    )
    return int(chosen['k'])


def _write_empty_clusters() -> None:
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        columns=[
            'canonical_ingredient',
            'nutrient_feature_count',
            'ingredient_cluster_id',
        ]
    ).to_parquet(INGREDIENT_CLUSTERS_STAGING, index=False)


def load_ingredient_clusters() -> pd.DataFrame:
    return pd.read_parquet(INGREDIENT_CLUSTERS_STAGING)
