from mlflow.tracking import MlflowClient

model_name = "IrisModelWithPreprocess"
client = MlflowClient()

# Get last archived Production version
prod_versions = client.get_latest_versions(model_name, stages=["Production"])
archived_versions = [v for v in client.search_model_versions(f"name='{model_name}'") if v.current_stage == "Archived"]

if not archived_versions:
    raise Exception("No archived versions available for rollback.")

last_version = archived_versions[-1].version
client.transition_model_version_stage(
    name=model_name,
    version=last_version,
    stage="Production",
    archive_existing_versions=True
)

print(f"🔁 Rolled back to version {last_version} of model {model_name}")
print(f"✅ Archived model {model_name} version {last_version} in Production")
print(f"✅ Model {model_name} version {last_version} is now in Production stage")