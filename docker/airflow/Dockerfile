FROM apache/airflow:2.9.3

# Install Python dependencies needed by the batch DAG runtime.
# Keep Airflow pinned to base image version by avoiding project deps resolution.
RUN pip install --no-cache-dir \
    kaggle \
    pandas \
    numpy \
    pytrends \
    scipy \
    scikit-learn \
    requests \
    sqlalchemy \
    psycopg2-binary \
    transformers \
    sentence-transformers \
    torch \
    vaderSentiment

