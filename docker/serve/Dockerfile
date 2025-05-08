FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \

# Copy requirements.txt and install additional dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Set working directory
WORKDIR /app

# Copy model artifacts (optional, if model is fetched from MLflow)
# COPY feature_store/ ./feature_store/

# Environment variables for MLflow model serving
ARG MODEL_NAME
ARG MODEL_VERSION
ARG MLFLOW_TRACKING_URI
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Expose port for model serving
EXPOSE 8000

# Command to serve the model
CMD ["sh", "-c", "mlflow models serve -m models:/${MODEL_NAME}/${MODEL_VERSION} -h 0.0.0.0 -p 8000 --env-manager local"]