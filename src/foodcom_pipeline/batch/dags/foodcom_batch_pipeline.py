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
from foodcom_pipeline.batch.load import run_load, load_trends
from foodcom_pipeline.batch.tag_recipes import run_tag_recipes, tag_signals
from foodcom_pipeline.batch.dags.tasks.compute_gap import compute_gap_analysis

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
    description='Food.com batch pipeline: extract → clean → sentiment → cluster → load (trends/AI mode run async)',
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

task_load_trends = PythonOperator(
    task_id='load_trends',
    python_callable=load_trends,
    dag=dag,
)

task_tag_recipes = PythonOperator(
    task_id='tag_recipes',
    python_callable=run_tag_recipes,
    dag=dag,
)

task_tag_signals = PythonOperator(
    task_id='tag_signals',
    python_callable=tag_signals,
    dag=dag,
)

task_compute_gap = PythonOperator(
    task_id='compute_gap_analysis',
    python_callable=compute_gap_analysis,
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

# Trends + AI Mode branch: runs fully async alongside the main pipeline
[task_extract_recipes, task_extract_google_trends] >> task_extract_ai_mode

# load_trends runs after both extracts so all 5 tables are populated together
[task_extract_google_trends, task_extract_ai_mode] >> task_load_trends

# tag_signals runs after load_trends so both staging files are guaranteed to exist
task_load_trends >> task_tag_signals

# Clean phase: depends only on the core extracts; unblocked by trends/AI mode
[
    task_extract_recipes,
    task_extract_interactions,
    task_extract_usda_nutrients,
] >> task_clean

# Tagging runs in parallel with sentiment after clean
task_clean >> [task_sentiment, task_tag_recipes]

# Gap analysis needs both recipe tags and AI Mode demand signal
task_tag_recipes >> task_compute_gap

# Aggregation reads sentiment output, so it must run after sentiment
task_sentiment >> task_aggregate_user_stats

# Clustering depends on aggregation
task_aggregate_user_stats >> task_cluster

# Load phase: depends on sentiment and cluster
[task_sentiment, task_cluster] >> task_load
