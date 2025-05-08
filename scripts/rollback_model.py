from mlflow.tracking import MlflowClient

# Configuration
MODEL_NAME = "adult-classifier"
MLFLOW_TRACKING_URI = "https://mitch-mlops.duckdns.org/mlflow"

def rollback_model():
    """Roll back to the last archived Production version and display metadata."""
    # Initialize MLflow client
    client = MlflowClient(MLFLOW_TRACKING_URI)

    # Get archived versions
    archived_versions = [
        v for v in client.search_model_versions(f"name='{MODEL_NAME}'")
        if v.current_stage == "Archived"
    ]

    if not archived_versions:
        raise Exception("No archived versions available for rollback.")

    # Select the last archived version
    last_version = archived_versions[-1].version

    # Transition to Production, archiving existing Production versions
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=last_version,
        stage="Production",
        archive_existing_versions=True
    )

    print(f"🔁 Rolled back to version {last_version} of model {MODEL_NAME}")
    print("✅ Archived existing Production versions")
    print(f"✅ Model {MODEL_NAME} version {last_version} is now in Production stage")

    # Retrieve and display metadata
    version_details = client.get_model_version(MODEL_NAME, last_version)
    print("\nModel Version Metadata:")
    print(f"  Created by: {version_details.tags.get('created_by', 'Unknown')}")
    print(f"  Stage: {version_details.current_stage}")
    print(f"  Description: {version_details.description or 'No description provided'}")
    print(f"  Schema: {version_details.tags.get('schema', 'No schema provided')}")

if __name__ == "__main__":
    try:
        rollback_model()
    except Exception as e:
        print(f"Error during rollback: {e}")