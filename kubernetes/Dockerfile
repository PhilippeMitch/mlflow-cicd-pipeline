FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY scripts/ ./scripts
COPY feature_store/ ./feature_store
CMD ["python", "scripts/train_model.py"]