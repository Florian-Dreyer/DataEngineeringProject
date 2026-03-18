"""
foodcom_dag.py
--------------
Airflow DAG for the Food.com batch pipeline (Lambda Architecture batch layer).

Pipeline stages:
  1. extract_recipes + extract_interactions  [PARALLEL]
  2. check_has_new_data                      [SHORT CIRCUIT]
  3. clean                                   [SEQUENTIAL]
  4. run_distilbert_sentiment                [SEQUENTIAL]
  5. aggregate_user_stats                    [SEQUENTIAL]
  6. run_kmeans_clustering                   [SEQUENTIAL]
  7. load_to_star_schema                     [SEQUENTIAL]

Dependency graph:

  extract_recipes      ──┐
                         ├──► check_has_new_data ──► clean ──► sentiment
  extract_interactions ──┘

  ──► aggregate_user_stats ──► cluster ──► load

Rationale for separating aggregate_user_stats from cluster:
  - Aggregation is cheap and can be retried independently of K-Means fitting
  - User stats (dim_user) are needed by the load step regardless of clustering
  - Each step has a single, clearly scoped responsibility
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from foodcom_pipeline.batch.extract import extract_interactions, extract_recipes

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


def check_has_new_data(**context) -> bool:
    """
    Reads the XCom flag set by extract_interactions.
    Returning False short-circuits all downstream tasks gracefully.
    """
    has_new = context['ti'].xcom_pull(
        task_ids='extract_interactions',
        key='has_new_data',
    )
    if not has_new:
        print('No new interactions found — skipping all downstream tasks.')
    return bool(has_new)


# ---------------------------------------------------------------------------
# Stage callables (lazy imports keep the DAG file lightweight)
# ---------------------------------------------------------------------------


def clean(**context):
    from foodcom_pipeline.batch.clean import run_clean

    run_clean(**context)


def run_sentiment(**context):
    from foodcom_pipeline.batch.sentiment import run_sentiment

    run_sentiment(**context)


def aggregate_user_stats(**context):
    from foodcom_pipeline.batch.aggregate_user_stats import (
        run_aggregate_user_stats,
    )

    run_aggregate_user_stats(**context)


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
        'Lambda batch layer: parallel extract → clean → DistilBERT sentiment '
        '→ aggregate user stats → K-Means clustering → star schema load'
    ),
    default_args=default_args,
    schedule_interval='@hourly',
    catchup=False,
    max_active_runs=1,
    tags=['foodcom', 'batch', 'lambda'],
) as dag:
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
            'Filters to rows newer than MAX(date) in fact_interactions. '
            'Pushes `has_new_data` flag to XCom.'
        ),
    )

    # ------------------------------------------------------------------
    # Guard
    # ------------------------------------------------------------------

    task_check_new_data = ShortCircuitOperator(
        task_id='check_has_new_data',
        python_callable=check_has_new_data,
        doc_md='Short-circuits if no new interactions were found.',
    )

    # ------------------------------------------------------------------
    # Stage 3: Clean
    # ------------------------------------------------------------------

    task_clean = PythonOperator(
        task_id='clean',
        python_callable=clean,
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
        task_id='run_distilbert_sentiment',
        python_callable=run_sentiment,
        execution_timeout=timedelta(hours=2),
        doc_md=(
            'Scores interactions with DistilBERT (SST-2). '
            'Computes rating_sentiment_gap. '
            'Produces VADER vs DistilBERT comparison report on first run.'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 5: Aggregate user stats
    # ------------------------------------------------------------------

    task_aggregate_user_stats = PythonOperator(
        task_id='aggregate_user_stats',
        python_callable=aggregate_user_stats,
        doc_md=(
            'Computes per-user aggregate features from scored interactions: '
            'avg_rating_given, review_count, recipe_diversity, '
            'avg_sentiment_score, avg_rating_gap, std_rating_given, '
            'pct_positive_sentiment, active_days. '
            'Output feeds both cluster.py and load.py (dim_user).'
        ),
    )

    # ------------------------------------------------------------------
    # Stage 6: Clustering
    # ------------------------------------------------------------------

    task_cluster = PythonOperator(
        task_id='run_kmeans_clustering',
        python_callable=run_clustering,
        doc_md=(
            'Fits K-Means on user stats. '
            'Determines optimal k via elbow + silhouette on first run. '
            'Assigns interpretable labels (Enthusiastic Cook, Harsh Critic, '
            'Casual Rater, Power User). '
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

    [task_extract_recipes, task_extract_interactions] >> task_check_new_data
    (
        task_check_new_data
        >> task_clean
        >> task_sentiment
        >> task_aggregate_user_stats
        >> task_cluster
        >> task_load
    )
