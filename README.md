# 🍽️ Food.com Recipe Analytics Pipeline

> IS3107 Data Engineering — AY2025/2026 Semester 2

An end-to-end **Lambda Architecture** data pipeline over the [Food.com Recipes and User Interactions](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset. The pipeline ingests recipe reviews in real time via Kafka, enriches them with NLP-based sentiment analysis, loads a PostgreSQL star schema via Airflow, and serves insights through a live Streamlit dashboard.

---

## Architecture

```
New Review (Kafka)
       │
  ┌────┴────┐
  │         │
Speed     Batch Layer (Airflow DAG)
Layer     ├─ Extract & Clean
│         ├─ DistilBERT Sentiment
VADER     ├─ K-Means User Clustering
Sentiment ├─ XGBoost Rating Prediction
│         └─ Star Schema Load (PostgreSQL)
  │         │
  └────┬────┘
       │
  Serving Layer (PostgreSQL VIEW)
       │
  Streamlit Dashboard
```

The **speed layer** processes incoming reviews immediately using VADER for low-latency sentiment scoring. The **batch layer** runs on a schedule, applying DistilBERT for higher-accuracy sentiment and overwriting the VADER approximations.

---

## Features

- **Real-time stream simulation** — Kafka producer replays held-out reviews at configurable speed
- **Dual sentiment models** — VADER (stream) and DistilBERT (batch), with quantified accuracy comparison
- **Star schema data warehouse** — `fact_interactions` + `dim_user`, `dim_recipe`, `dim_date`
- **`rating_sentiment_gap`** — derived feature exposing mismatches between star ratings and review text
- **User clustering** — K-Means segmentation into interpretable profiles (e.g. Harsh Critic, Enthusiastic Cook)
- **Rating prediction** — XGBoost regression evaluated with RMSE/MAE
- **Live Streamlit dashboard** — Recipe Explorer, Trend Analysis, and User Segments tabs

---

## Tech Stack

| Component | Technology |
|---|---|
| Orchestration | Apache Airflow |
| Message broker | Apache Kafka + Zookeeper |
| Database | PostgreSQL |
| Batch processing | Python, Pandas, scikit-learn |
| Sentiment (batch) | DistilBERT via HuggingFace Transformers |
| Sentiment (stream) | VADER via NLTK |
| ML model | XGBoost |
| Dashboard | Streamlit |
| Infrastructure | Docker Compose |

---

## Getting Started

### Prerequisites
- Docker & Docker Compose
- Python 3.10+

### 1) Prepare Local Files

```bash
# From the repo root (same folder as docker-compose.yml)
cp .env.example .env
mkdir -p data staging

# Kaggle credentials are required for auto-download:
#   ~/.kaggle/kaggle.json
# You can generate it from Kaggle Account -> API -> Create New Token.
# The docker compose file mounts this into the Airflow container.
mkdir -p ~/.kaggle
# Copy your downloaded Kaggle token file to:
#   ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 2) Start Services

#### 🆕 First-time setup (build required)
> ⚠️ Only run this once, or when dependencies change (e.g. Dockerfile or requirements.txt)

```bash
# Build and start all services (may take a few minutes)
docker compose up -d --build

# Subsequent runs (fast) - start services without rebuilding
docker compose up -d

# Restart services
docker compose restart

# Stop services
docker compose down


```

Expected:
- `postgres` and `airflow` are both `Up` in `docker compose ps`
- Airflow log shows webserver boot (e.g. `Listening at: http://0.0.0.0:8080`)
- First build can take several minutes (it builds a custom Airflow image with DAG dependencies preinstalled).

Health check:

```bash
curl -I http://localhost:8080/health
```

You should get `HTTP/1.1 200 OK`.

If you see `Empty reply from server` or connection reset, wait ~20-40 seconds and retry:

```bash
curl -v http://localhost:8080/health
```

### 3) Open Airflow UI

- URL: `http://localhost:8080`
- Default credentials (from `.env`): `airflow` / `airflow`

### 4) Trigger the DAG

In Airflow UI:
1. Open DAG `foodcom_batch_pipeline`
2. Click **Trigger DAG**
3. Watch tasks in this order:
   - `ensure_source_data` (downloads RAW CSVs from Kaggle if missing)
   - `extract_recipes` + `extract_interactions` (parallel)
   - `clean`
   - `run_distilbert_sentiment` (slowest step on CPU)
   - `aggregate_user_stats`
   - `run_kmeans_clustering`
   - `load_to_star_schema`


### 5) Verify Warehouse Load

Run from your Postgres client:

```sql
SELECT COUNT(*) FROM fact_interactions;
SELECT COUNT(*) FROM dim_user;
SELECT COUNT(*) FROM dim_recipe;
SELECT COUNT(*) FROM dim_date;
```

---

## Troubleshooting

- **`docker compose up` says `no configuration file provided`**
  - Run from the repo root (where `docker-compose.yml` is located).
- **`localhost:8080` not opening**
  - Check `docker compose ps` (airflow must be `Up`).
  - Check `curl -I http://localhost:8080/health` (should be 200).
  - Check `docker compose logs --tail=200 airflow`.
- **Browser says connection reset but health is 200**
  - Try `http://127.0.0.1:8080` and hard refresh / incognito.
- **`/bin/bash: --username: command not found` in airflow logs**
  - Your compose command block is malformed; use the one in this repo (single-line `airflow users create ...`).
- **`/bin/bash: airflow: command not found` in airflow logs**
  - Ensure airflow container runs with `user: "50000:0"` (as in this repo).
- **Large warning spam from `azure/... SyntaxWarning: invalid escape sequence`**
  - This is noisy but non-fatal; ignore unless there is a traceback/error after it.
- **`zsh: command not found: rg`**
  - `rg` (ripgrep) is optional locally; use:
    `docker compose logs airflow | grep -E "ERROR|Traceback|Exception|Listening at|Booting worker"`.
- **DAG fails at extract due to missing CSV**
  - Confirm exact paths: `data/RAW_recipes.csv`, `data/RAW_interactions.csv`.
- **Kaggle download fails**
  - Confirm `~/.kaggle/kaggle.json` exists on your host.
  - Ensure file permissions are strict: `chmod 600 ~/.kaggle/kaggle.json`.
  - Confirm `.env` has `FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true`.
- **Where data is mounted in container**
  - Input CSVs: `/opt/airflow/data`
  - Staging parquet files: `/opt/airflow/staging`

### (Optional) Launch the dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

---

## Project Structure

```
├── docker-compose.yml
├── docker/
│   └── airflow/
│       └── Dockerfile          # Custom Airflow image (pre-installs runtime deps)
├── .env.example
├── src/
│   └── foodcom_pipeline/        # Main package
│       ├── __init__.py
│       ├── batch/                   # Batch layer (runs via Airflow)
│       │   ├── __init__.py
│       │   ├── extract.py           # Load CSVs into DataFrames
│       │   ├── clean.py             # Cleaning & normalization
│       │   ├── sentiment.py         # DistilBERT scoring
│       │   ├── aggregate_user_stats.py  # User-level feature engineering
│       │   ├── cluster.py           # K-Means user segmentation
│       │   └── load.py              # PostgreSQL star schema loaders
│       ├── stream/                  # Stream layer (speed layer)
│       │   ├── __init__.py
│       │   ├── producer.py          # Kafka producer (simulation replay)
│       │   └── consumer.py          # Kafka consumer, VADER, hot table write
│       └── dashboard/               # Streamlit app
│           ├── __init__.py
│           └── app.py
├── data/
│   ├── RAW_recipes.csv           # Kaggle CSV (you add this)
│   └── RAW_interactions.csv      # Kaggle CSV (you add this)
├── staging/                      # Parquet staging outputs (created at runtime)
└── pyproject.toml
```
