# Use the official MLflow image as the base
FROM ghcr.io/mlflow/mlflow:v2.8.1

# Install system dependencies (curl) and psycopg2-binary
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install psycopg2-binary==2.9.6

# Create a non-root user
RUN useradd -m -u 1000 mlflow

# Ensure the artifact directory is writable by the non-root user
RUN mkdir -p /mlflow/artifacts && chown -R mlflow:mlflow /mlflow/artifacts

# Switch to non-root user
USER mlflow

# Add healthcheck (use / instead of /health, as MLflow serves the UI at /)
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1