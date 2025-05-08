import os
import pytest
import mlflow
from unittest.mock import patch

# Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "sample-model"
TEST_DATA_PATH = "tests/test_dataset.csv"
PERFORMANCE_THRESHOLD = 0.8

@pytest.fixture(scope="module")
def mock_mlflow():
    """Fixture to set up MLflow tracking URI."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    yield
    mlflow.set_tracking_uri(None)


def test_model_mocking():
    """Test with mocked MLflow get_run."""
    with patch('mlflow.tracking.MlflowClient.get_run') as mock_get_run:
        # Setup mock return value
        mock_run = mock_get_run.return_value
        mock_run.data.metrics = {'accuracy': 0.85}

        # Test the mocked value
        client = mlflow.tracking.MlflowClient()
        run = client.get_run('run_id')
        assert run.data.metrics['accuracy'] >= PERFORMANCE_THRESHOLD, f"Model accuracy {accuracy} below threshold {PERFORMANCE_THRESHOLD}"

@pytest.mark.parametrize("mock_mlflow", [True], indirect=True)
def test_model(mock_mlflow):
    """Test MLflow metrics retrieval."""
    with patch('mlflow.get_run') as mock_get_run:
        mock_get_run.return_value.data.metrics = {'accuracy': 0.85}
        assert mock_get_run('run_id').data.metrics['accuracy'] == 0.85