"""
foodcom_batch_pipeline.py
=========================
Airflow DAG orchestrating the Food.com batch data pipeline.

Lambda architecture:
  - Batch layer: daily full/incremental extract → clean → sentiment analysis → clustering → load
  - Streaming layer: Kafka events → sentiment → append to recent_interactions (handled separately)
  - Serving layer: star schema views combining batch fact + recent stream data
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from foodcom_pipeline.batch.extract import (
    ensure_source_data,
    extract_recipes,
    extract_interactions,
    extract_usda_nutrients,
    extract_google_trends,
    extract_ai_mode,
)
from foodcom_pipeline.batch.clean import run_clean
from foodcom_pipeline.batch.sentiment import run_sentiment
from foodcom_pipeline.batch.aggregate_user_stats import run_aggregate_user_stats
from foodcom_pipeline.batch.cluster import run_clustering
from foodcom_pipeline.batch.load import run_load

# ─────────────────────────────────────────────────────────────────────────
# DAG Definition
# ─────────────────────────────────────────────────────────────────────────

default_args = {
    'owner': 'data-engineering',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'email_on_failure': False,
    'start_date': datetime(2025, 1, 1),
    'execution_timeout': timedelta(hours=3),
}

dag = DAG(
    'foodcom_batch_pipeline',
    default_args=default_args,
    description='Food.com batch pipeline: extract → clean → sentiment → cluster → load',
    schedule_interval='@daily',
    catchup=False,
    tags=['foodcom', 'batch', 'production'],
)

# ─────────────────────────────────────────────────────────────────────────
# Extract Phase
# ─────────────────────────────────────────────────────────────────────────

task_ensure_source_data = PythonOperator(
    task_id='ensure_source_data',
    python_callable=ensure_source_data,
    dag=dag,
)

task_extract_recipes = PythonOperator(
    task_id='extract_recipes',
    python_callable=extract_recipes,
    dag=dag,
)

task_extract_interactions = PythonOperator(
    task_id='extract_interactions',
    python_callable=extract_interactions,
    dag=dag,
)

task_extract_usda_nutrients = PythonOperator(
    task_id='extract_usda_nutrients',
    python_callable=extract_usda_nutrients,
    dag=dag,
)

task_extract_google_trends = PythonOperator(
    task_id='extract_google_trends',
    python_callable=extract_google_trends,
    dag=dag,
)

task_extract_ai_mode = PythonOperator(
    task_id='extract_ai_mode',
    python_callable=extract_ai_mode,
    dag=dag,
)

# ─────────────────────────────────────────────────────────────────────────
# Clean Phase
# ─────────────────────────────────────────────────────────────────────────

task_clean = PythonOperator(
    task_id='clean',
    python_callable=run_clean,
    dag=dag,
)

# ─────────────────────────────────────────────────────────────────────────
# Sentiment & Clustering Phase
# ─────────────────────────────────────────────────────────────────────────

task_sentiment = PythonOperator(
    task_id='sentiment',
    python_callable=run_sentiment,
    dag=dag,
)

task_aggregate_user_stats = PythonOperator(
    task_id='aggregate_user_stats',
    python_callable=run_aggregate_user_stats,
    dag=dag,
)

task_cluster = PythonOperator(
    task_id='cluster',
    python_callable=run_clustering,
    dag=dag,
)

# ─────────────────────────────────────────────────────────────────────────
# Load Phase
# ─────────────────────────────────────────────────────────────────────────

task_load = PythonOperator(
    task_id='load',
    python_callable=run_load,
    dag=dag,
)

# ─────────────────────────────────────────────────────────────────────────
# Task Dependencies
# ─────────────────────────────────────────────────────────────────────────

# Extract phase: ensure source data first, then independent extracts run in parallel
task_ensure_source_data >> [
    task_extract_recipes,
    task_extract_interactions,
    task_extract_usda_nutrients,
    task_extract_google_trends,
]

# AI Mode depends on recipes (food context) and trends (term scoring baseline)
[task_extract_recipes, task_extract_google_trends] >> task_extract_ai_mode

# Clean phase: waits for all extract work; recipes and trends feed clean
# transitively through ai_mode so only the remaining three are listed directly
[
    task_extract_interactions,
    task_extract_usda_nutrients,
    task_extract_ai_mode,
] >> task_clean

# Sentiment & aggregation depend on clean
task_clean >> [task_sentiment, task_aggregate_user_stats]

# Clustering depends on aggregation
task_aggregate_user_stats >> task_cluster

# Load phase: depends on sentiment and cluster
[task_sentiment, task_cluster] >> task_load
