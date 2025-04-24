# Use Python base image
FROM python:3.9-slim

# Install MLflow and dependencies
RUN pip install mlflow==2.8.1 psycopg2-binary==2.9.6

# Copy requirements.txt and install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Copy model artifacts (optional, if model is fetched from MLflow)
COPY feature_store/ ./feature_store/

# Environment variables for MLflow model serving
ARG MODEL_NAME
ARG MODEL_VERSION
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Expose port for model serving
EXPOSE 8000

# Command to serve the model
CMD ["sh", "-c", "mlflow models serve -m models:/${MODEL_NAME}/${MODEL_VERSION} -h 0.0.0.0 -p 8000 --env-manager local"]