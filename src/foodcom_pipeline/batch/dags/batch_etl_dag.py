"""
foodcom_dag.py
--------------
Airflow DAG for the Food.com batch pipeline (Lambda Architecture batch layer).

Pipeline stages:
  1. extract_recipes + extract_interactions  [PARALLEL]
  2. check_has_new_data                      [SHORT CIRCUIT]
  3. clean                                   [SEQUENTIAL]
  4. run_vader_sentiment                     [SEQUENTIAL]
  5. features                                [SEQUENTIAL]
  6. run_kmeans_clustering                   [SEQUENTIAL]
  7. load_to_star_schema                     [SEQUENTIAL]

Dependency graph:

  extract_recipes      ──┐
                         ├──► check_has_new_data ──► clean ──► sentiment
  extract_interactions ──┘

  ──► features ──► cluster ──► load

Rationale for separating features from cluster:
  - Feature engineering is cheap and can be retried independently of K-Means fitting
  - User stats (dim_user) are needed by the load step regardless of clustering
  - Each step has a single, clearly scoped responsibility
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule
from foodcom_pipeline.batch.extract import (
    ensure_source_data,
    extract_interactions,
    extract_recipes,
    extract_usda_nutrients,
)

# ---------------------------------------------------------------------------
# Default arguments
# ---------------------------------------------------------------------------

default_args = {
    'owner': 'foodcom-team',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
}

# ---------------------------------------------------------------------------
# Guard
# ---------------------------------------------------------------------------

 #REMEMBER TO UNCOMMENT THIS WHEN YOU ARE DONE TESTING!

def check_has_new_data(**context) -> bool:
    return True


    #REMEMBER TO UNCOMMENT THIS WHEN YOU ARE DONE TESTING!
    
    
    
    # """
    # Reads the XCom flag set by extract_interactions.
    # Returning False short-circuits all downstream tasks gracefully.
    # """
    # has_new = context['ti'].xcom_pull(
    #     task_ids='extract_interactions',
    #     key='has_new_data',
    # )
    # if not has_new:
    #     print('No new interactions found — skipping all downstream tasks.')
    # return bool(has_new)


# ---------------------------------------------------------------------------
# Stage callables (lazy imports keep the DAG file lightweight)
# ---------------------------------------------------------------------------


def clean(**context):
    from foodcom_pipeline.batch.clean import run_clean

    run_clean(**context)


def run_sentiment(**context):
    from foodcom_pipeline.batch.sentiment import run_sentiment

    run_sentiment(**context)


def run_features(**context):
    from foodcom_pipeline.batch.features import run_features as _run_features

    _run_features(**context)


def run_ingredient_clustering(**context):
    from foodcom_pipeline.batch.ingredient_clusters import run_ingredient_clustering

    run_ingredient_clustering(**context)


def run_clustering(**context):
    from foodcom_pipeline.batch.cluster import run_clustering

    run_clustering(**context)


def load(**context):
    from foodcom_pipeline.batch.load import run_load

    run_load(**context)


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

with DAG(
    dag_id='foodcom_batch_pipeline',
    description=(
        'Lambda batch layer: parallel extract → clean → VADER sentiment '
        '→ feature engineering → K-Means clustering → star schema load'
    ),
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['foodcom', 'batch', 'lambda'],
) as dag:
    task_ensure_source_data = PythonOperator(
        task_id='ensure_source_data',
        python_callable=ensure_source_data,
        doc_md=(
            'Ensures RAW_recipes.csv, RAW_interactions.csv, and ingr_map.pkl exist. '
            'If any file is missing, downloads it from Kaggle. '
            'Handles ZIP extraction fallback for all three files.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 1: Extract — PARALLEL
    # ------------------------------------------------------------------

    task_extract_recipes = PythonOperator(
        task_id='extract_recipes',
        python_callable=extract_recipes,
        doc_md='Full extract of RAW_recipes.csv. Stages to parquet.',
    )

    task_extract_interactions = PythonOperator(
        task_id='extract_interactions',
        python_callable=extract_interactions,
        doc_md=(
            'Incremental extract of RAW_interactions.csv. '
            'Filters to rows newer than MAX(full_date) via dim_date join. '
            'Pushes `has_new_data` flag to XCom.'
        ),
    )

    task_extract_usda_nutrients = PythonOperator(
        task_id='extract_usda_nutrients',
        python_callable=extract_usda_nutrients,
        doc_md=(
            'Extracts USDA nutrient ground truth for canonical ingredients from '
            '`ingr_map.pkl`, writes `usda_nutrients.parquet` to staging. '
            'Skips recomputation if the parquet already exists unless '
            '`FOODCOM_USDA_FORCE_REFRESH` is true/1/yes.'
        ),
    )

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    task_check_new_data = ShortCircuitOperator(
        task_id='check_has_new_data',
        python_callable=check_has_new_data, #remember to uncomment this when you are done testing!
        ignore_downstream_trigger_rules=False,
        doc_md='Short-circuits if no new interactions were found.',
    )

    # ------------------------------------------------------------------
    # Stage 3: Clean
    # ------------------------------------------------------------------

    task_clean = PythonOperator(
        task_id='clean',
        python_callable=clean,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
        doc_md=(
            'Cleans both recipes and interactions. '
            '[DATASET] dedup, invalid ratings, date parsing, cook time caps. '
            '[DEFENSIVE] future dates, rating coercion, review normalization, '
            'short reviews, bot detection.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 4: Sentiment
    # ------------------------------------------------------------------

    task_sentiment = PythonOperator(
        task_id='run_vader_sentiment',
        python_callable=run_sentiment,
        execution_timeout=timedelta(hours=2),
        doc_md=(
            'Scores interactions with VADER compound polarity. '
            'Computes rating_sentiment_gap.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 5: Aggregate user stats
    # ------------------------------------------------------------------

    task_features = PythonOperator(
        task_id='features',
        python_callable=run_features,
        doc_md=(
            'Computes all derived features before clustering. '
            'Per-user stats: avg_rating_given, review_count, recipe_diversity, '
            'avg_sentiment_score, avg_rating_gap, std_rating_given, '
            'pct_positive_sentiment, active_days. '
            'Per-recipe Bayesian sentiment_rating (inverse-frequency + shrinkage). '
            'Ingredient-level features + substitution candidate flags. '
            'Outputs feed cluster.py and load.py.'
        ),
    )

    task_ingredient_clusters = PythonOperator(
        task_id='cluster_canonical_ingredients',
        python_callable=run_ingredient_clustering,
        doc_md=(
            'Fits K-Means on staged USDA nutrient vectors per canonical ingredient '
            'and writes `ingredient_clusters.parquet` for substitution compatibility gating.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 6: Clustering
    # ------------------------------------------------------------------

    task_cluster = PythonOperator(
        task_id='run_kmeans_clustering',
        python_callable=run_clustering,
        doc_md=(
            'Fits K-Means on per-user ingredient category rating features. '
            'Selects optimal k in [4,5,6] by highest silhouette score on first run. '
            'Assigns labels (Indulgent Baker, International Explorer, '
            'Protein-Forward Cook, Health-Conscious Cook, Quick & Simple). '
            'Produces per-cluster profile statistics.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 7: Load
    # ------------------------------------------------------------------

    task_load = PythonOperator(
        task_id='load_to_star_schema',
        python_callable=load,
        doc_md='Upserts cleaned, scored, clustered data into the star schema.',
    )

    # ------------------------------------------------------------------
    # Dependency graph
    # ------------------------------------------------------------------

    task_ensure_source_data >> [
        task_extract_recipes,
        task_extract_interactions,
        task_extract_usda_nutrients,
    ]

    [task_extract_recipes, task_extract_interactions] >> task_check_new_data

    # clean waits for: recipes extract + USDA extract + guard True (ShortCircuit success)
    
    task_extract_usda_nutrients >> task_clean
    task_check_new_data >> task_clean

    task_clean >> task_sentiment
    task_extract_usda_nutrients >> task_ingredient_clusters
    [task_sentiment, task_ingredient_clusters] >> task_features
    task_features >> task_cluster >> task_load
