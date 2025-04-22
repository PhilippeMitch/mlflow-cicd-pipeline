import mlflow
import pandas as pd
from sklearn.metrics import accuracy_score
import pytest

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "sample-model"
TEST_DATA_PATH = "tests/test_dataset.csv"
PERFORMANCE_THRESHOLD = 0.8

def test_model_performance():
    """Test MLflow model performance on test dataset."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Load the latest Production model
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version.version}")

    # Load test dataset
    test_data = pd.read_csv(TEST_DATA_PATH)
    X_test = test_data.drop(columns=["target"])
    y_test = test_data["target"]

    # Predict and evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Assert performance threshold
    assert accuracy >= PERFORMANCE_THRESHOLD, f"Model accuracy {accuracy} below threshold {PERFORMANCE_THRESHOLD}"
    print(f"Model accuracy: {accuracy}")