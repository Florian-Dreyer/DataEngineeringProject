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

### Setup

```bash
# 1. Clone the repo
git clone https://github.com/<your-org>/foodcom-pipeline.git
cd foodcom-pipeline

# 2. Copy and fill in environment variables
cp .env.example .env

# 3. Download the dataset from Kaggle and place in data/raw/
#    RAW_recipes.csv and RAW_interactions.csv

# 4. Start all services
docker compose up -d

# 5. Access Airflow UI at http://localhost:8080
#    Default credentials: airflow / airflow
```

### Run the pipeline

```bash
# Trigger the batch DAG manually from the Airflow UI,
# or wait for the scheduled run.

# Start the Kafka stream simulation (in a separate terminal)
python src/stream/producer.py --rate 1  # 1 review/second
```

### Launch the dashboard

```bash
streamlit run src/dashboard/app.py
# Opens at http://localhost:8501
```

---

## Project Structure

```
├── dags/                        # Airflow DAG definitions
├── src/
│   ├── batch/                   # Batch layer (runs via Airflow)
│   │   ├── extract.py           # Load CSVs into DataFrames
│   │   ├── transform.py         # Cleaning & feature engineering
│   │   ├── sentiment.py         # DistilBERT scoring
│   │   ├── clustering.py        # K-Means user segmentation
│   │   ├── model.py             # XGBoost rating prediction
│   │   └── load.py              # PostgreSQL star schema loaders
│   ├── stream/                  # Stream layer (speed layer)
│   │   ├── producer.py          # Kafka producer (simulation replay)
│   │   └── consumer.py          # Kafka consumer, VADER, hot table write
│   └── dashboard/               # Streamlit app
│       └── app.py
├── data/
│   └── raw/                     # Place Kaggle CSVs here (gitignored)
├── docker-compose.yml
├── .env.example
└── pyproject.toml
```
