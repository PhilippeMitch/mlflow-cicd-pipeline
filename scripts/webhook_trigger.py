import mlflow
import requests
import os
from dotenv import load_dotenv
load_dotenv(".env")

# Configuration
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
JENKINS_WEBHOOK_URL = os.environ.get("JENKINS_WEBHOOK_URL")
GITHUB_ACTION_DISPATCH_URL = os.environ.get("GITHUB_ACTION_DISPATCH_URL")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
MODEL_NAME = "adult-classifier"
NEW_STAGE = "Production"

def trigger_webhooks(model_name, version, stage):
    """Trigger Jenkins and GitHub Actions webhooks."""
    payload = {
        "model_name": model_name,
        "version": version,
        "stage": stage
    }

    # Trigger Jenkins webhook
    try:
        response = requests.post(JENKINS_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print(f"Jenkins webhook triggered successfully: {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to trigger Jenkins webhook: {e}")

    # Trigger GitHub Actions workflow
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github.v3+json"
    }
    try:
        response = requests.post(
            GITHUB_ACTION_DISPATCH_URL,
            json={"ref": "main", "inputs": payload},
            headers=headers
        )
        response.raise_for_status()
        print(f"GitHub Actions webhook triggered successfully: {response.status_code}")
    except requests.RequestException as e:
        print(f"Failed to trigger GitHub Actions webhook: {e}")

def main():
    """Simulate MLflow model stage transition and trigger webhooks."""
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    # Create a sample model version (for demo purposes)
    client = mlflow.tracking.MlflowClient()
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        print(f"Model {MODEL_NAME} already exists.")

    # Log a dummy model
    with mlflow.start_run():
        model = mlflow.sklearn.log_model(
            sk_model=sklearn.ensemble.RandomForestClassifier(),
            artifact_path="model"
        )
        model_version = client.create_model_version(
            MODEL_NAME, model.model_uri, run_id=mlflow.active_run().info.run_id
        )

    # Transition model to Production stage
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage=NEW_STAGE
    )
    print(f"Model {MODEL_NAME} version {model_version.version} transitioned to {NEW_STAGE}")

    # Trigger webhooks
    trigger_webhooks(MODEL_NAME, model_version.version, NEW_STAGE)

if __name__ == "__main__":
    main()