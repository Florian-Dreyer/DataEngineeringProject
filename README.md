# Food.com Recipe Analytics Pipeline

> IS3107 Data Engineering — AY2025/2026 Semester 2

An **ETL pipeline** for the [Food.com](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) dataset, orchestrated with Apache Airflow and backed by a PostgreSQL data warehouse, powering a Streamlit analytics dashboard.

---

## Architecture

```text
┌─────────────────────────────────────────────────────────┐
│                    ETL PIPELINE (Airflow)                │
│                                                          │
│  ensure_source_data                                      │
│       │                                                  │
│       ├── extract_recipes ──────────────────────┐        │
│       ├── extract_interactions ─► check_new ─► clean    │
│       └── extract_usda_nutrients ───────────────┘        │
│                    │                                     │
│             run_vader_sentiment                          │
│                    │                                     │
│                features                                  │
│                    │                                     │
│          run_kmeans_clustering                           │
│                    │                                     │
│           load_to_star_schema                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              DATA WAREHOUSE (PostgreSQL)                 │
│  fact_interactions · dim_user · dim_recipe · dim_date   │
└─────────────────────────────────────────────────────────┘
```

---

## Pipeline Stages

| # | Task | Description |
| --- | --- | --- |
| 1 | `ensure_source_data` | Downloads `RAW_recipes.csv`, `RAW_interactions.csv`, `ingr_map.pkl` from Kaggle if missing |
| 2 | `extract_recipes` | Full extract of recipes → parquet |
| 2 | `extract_interactions` | Incremental extract of interactions (watermark-based) → parquet |
| 2 | `extract_usda_nutrients` | Matches canonical ingredients to USDA FoodData Central nutrient values |
| 3 | `check_has_new_data` | Short-circuits if no new interactions found |
| 4 | `clean` | Deduplication, invalid rating removal, date parsing, cook-time caps, nutrition outlier flagging (95th percentile), bot detection |
| 5 | `run_vader_sentiment` | VADER compound polarity scoring + `rating_sentiment_gap` |
| 6 | `features` | Per-user aggregate stats, ingredient category ratings, Bayesian recipe sentiment ratings, substitution candidate flagging |
| 7 | `run_kmeans_clustering` | K-Means user segmentation (k selected by silhouette score over [4,5,6]) |
| 8 | `load_to_star_schema` | Upsert to PostgreSQL star schema |

### User Segments

Clusters are auto-labelled by centroid values after fitting:

| Label | Dominant feature |
| --- | --- |
| Indulgent Baker | Highest avg baking recipe rating |
| International Explorer | Highest avg international recipe rating |
| Protein-Forward Cook | Highest avg protein recipe rating |
| Health-Conscious Cook | Highest avg vegetable recipe rating |
| General Cook | Remainder (typically the majority cluster) |

---

## Tech Stack

| Component | Technology |
| --- | --- |
| Orchestration | Apache Airflow 2.9 |
| Database | PostgreSQL 16 |
| Batch processing | Python, Pandas, scikit-learn |
| Sentiment analysis | VADER (`vaderSentiment`) |
| Clustering | K-Means (scikit-learn) |
| Nutrient data | USDA FoodData Central |
| Dashboard | Streamlit + Plotly |
| Infrastructure | Docker Compose |

---

## Project Structure

```text
├── docker-compose.yml
├── docker/
│   └── airflow/
│       └── Dockerfile              # Custom Airflow image
├── pyproject.toml
├── src/
│   └── foodcom_pipeline/
│       ├── batch/
│       │   ├── dags/
│       │   │   └── batch_etl_dag.py    # Airflow DAG definition
│       │   ├── extract.py              # Kaggle + USDA extraction
│       │   ├── clean.py                # Data cleaning
│       │   ├── sentiment.py            # VADER scoring
│       │   ├── features.py             # Feature engineering
│       │   ├── cluster.py              # K-Means segmentation
│       │   └── load.py                 # Star schema loader
│       ├── extraction/
│       │   └── trends.py               # Google Trends (deferred)
│       └── dashboard/
│           └── app.py                  # Streamlit dashboard
├── data/                               # Raw CSVs (auto-downloaded from Kaggle)
├── staging/                            # Parquet staging files (created at runtime)
├── usda_data/                          # USDA FoodData Central bulk JSON (optional)
└── tests/
```

---

## Getting Started

### Prerequisites

- Docker Desktop
- A Kaggle account with an API token (`~/.kaggle/kaggle.json`)

### 1. Clone and prepare

```bash
git clone <repo-url>
cd DataEngineeringProject

mkdir -p data staging

# Kaggle credentials — generate at kaggle.com → Account → API → Create New Token
mkdir -p ~/.kaggle
# Copy your kaggle.json here: 
#   ~/.kaggle/kaggle.json
mv kaggle.json ~/.kaggle/

# Secure your Kaggle API credentials by restricting file access
chmod 600 ~/.kaggle/kaggle.json
```

### 2. Start services

**First time (builds the custom Airflow image — takes a few minutes):**

```bash
docker compose up -d --build
```

**Subsequent starts:**

```bash
docker compose up -d
```

Verify services are up:

```bash
docker compose ps
# postgres   Up   5432/tcp
# airflow    Up   0.0.0.0:8080->8080/tcp
```

Health check:

```bash
curl -I http://localhost:8080/health   # expect HTTP/1.1 200 OK
```

### 3. Trigger the pipeline

```bash
# Unpause and trigger via CLI
docker compose exec airflow airflow dags unpause foodcom_batch_pipeline
docker compose exec airflow airflow dags trigger foodcom_batch_pipeline
```

Or open the Airflow UI at **[http://localhost:8080](http://localhost:8080)** (login: `airflow` / `airflow`), find `foodcom_batch_pipeline`, unpause it, and click **Trigger DAG**.

Expected run time on first execution: ~15–20 minutes (VADER scoring dominates).

### 4. Launch the dashboard

```bash
streamlit run src/foodcom_pipeline/dashboard/app.py
```

The dashboard auto-detects the `./staging/` directory — no environment variables needed.

### 5. Verify the warehouse

```sql
SELECT COUNT(*) FROM fact_interactions;
SELECT COUNT(*) FROM dim_user;
SELECT COUNT(*) FROM dim_recipe;
SELECT COUNT(*) FROM dim_date;
```

---

## Dashboard

| Tab | Status | Content |
| --- | --- | --- |
| Overview | stub | Pipeline run summary |
| Recipe Analytics | stub | Top recipes, sentiment ratings |
| Audience & Market Intelligence | implemented | Radar chart, segment profiles, CPG brand adjacency, PDF export |
| Pipeline Status | implemented | Row counts per stage, data loss rate, USDA coverage, Airflow task runtimes |

---

## Troubleshooting

| Problem | Fix |
| --- | --- |
| `localhost:8080` not responding | Wait 30–40 s after `docker compose up`; check `docker compose logs airflow` |
| DAG not visible in UI | Run `docker compose exec airflow airflow dags reserialize` |
| `ModuleNotFoundError` in DAG | Rebuild image: `docker compose up -d --build` |
| Kaggle download fails | Check `~/.kaggle/kaggle.json` exists and `chmod 600` is set |
| Data CSVs missing | Confirm `FOODCOM_ENABLE_KAGGLE_DOWNLOAD=true` in environment or add CSVs manually to `./data/` |
| Staging files not found in dashboard | Ensure pipeline has completed at least one successful run |

---

## Notes

- **Google Trends integration** is implemented (`src/foodcom_pipeline/extraction/trends.py`) but not yet wired into the DAG. The `trend_index` column in `ingredient_features.parquet` is currently `null`.
- The pipeline uses **VADER** for sentiment scoring. A higher-accuracy model can be swapped in by replacing `sentiment.py` without changing downstream steps.
- **Incremental extraction** uses a watermark on `MAX(full_date)` from `fact_interactions ⋈ dim_date`. The first run is always a full extract.
