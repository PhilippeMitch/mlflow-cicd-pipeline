# Use the official MLflow image as the base
FROM ghcr.io/mlflow/mlflow:v2.8.1

# Install psycopg2-binary
RUN pip install psycopg2-binary==2.9.6

# Create a non-root user
RUN useradd -m -u 1000 mlflow

# Switch to non-root user
USER mlflow

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1