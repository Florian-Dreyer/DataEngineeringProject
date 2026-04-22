FROM apache/airflow:2.9.3

# Install Python dependencies needed by the batch DAG runtime.
# Keep Airflow pinned to base image version by avoiding project deps resolution.
RUN pip install --no-cache-dir \
    kaggle \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    serpapi \
    google-generativeai \
    requests \
    sqlalchemy \
    psycopg2-binary \
    kafka-python \
    nltk \
    vaderSentiment \
    xgboost \
    python-dotenv \
    tqdm \
    transformers \
    torch

