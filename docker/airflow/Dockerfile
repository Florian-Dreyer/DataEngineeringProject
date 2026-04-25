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
    google-genai \
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
    torch \
    fpdf2

# Copy source code with correct ownership (airflow user is 50000)
COPY --chown=50000:0 src /opt/airflow/app/src

# Create all necessary runtime directories
RUN mkdir -p /opt/airflow/data \
             /opt/airflow/staging \
             /opt/airflow/usda_data && \
    chown -R 50000:0 /opt/airflow/data \
                     /opt/airflow/staging \
                     /opt/airflow/usda_data

# Add /opt/airflow/app/src to Python path so Airflow can import foodcom_pipeline
ENV PYTHONPATH=/opt/airflow/app/src:${PYTHONPATH}
