FROM python:3.10-bullseye

# Install system dependencies required by scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    libgomp1 \
    gfortran \
    libglib2.0-0 \
    libxext6 \
    libsm6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set workdir before copying files
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Environment variables for MLflow model serving
ARG MODEL_NAME
ARG MODEL_VERSION
ARG MLFLOW_TRACKING_URI
ENV MODEL_NAME=${MODEL_NAME}
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MLFLOW_TRACKING_URI=${MLFLOW_TRACKING_URI}

# Expose port for model serving
EXPOSE 8000

# Run MLflow model server
CMD ["sh", "-c", "mlflow models serve -m models:/${MODEL_NAME}/${MODEL_VERSION} -h 0.0.0.0 -p 8000 --env-manager local"]
