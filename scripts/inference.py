import mlflow
import pandas as pd
import joblib
import redis

# Configuration
MLFLOW_TRACKING_URI = "https://mitch-mlops.duckdns.org/mlflow"
MODEL_NAME = "adult-classifier"
PREPROCESSOR_PATH = "feature_store/preprocessor.joblib"

r = redis.Redis(host='redis', port=6379, db=0)


def perform_inference(input_data):
    """Perform inference using the latest Production model."""
    cache_key = f"prediction:{input_data.to_json()}"
    if r.exists(cache_key):
        predictions = r.get(cache_key).decode()
        return predictions

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    model_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{model_version.version}")
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    input_transformed = preprocessor.transform(input_data)
    predictions = model.predict(input_transformed)
    r.set(cache_key, str(predictions), ex=3600)
    return predictions


if __name__ == "__main__":
    # Sample input data matching Adult dataset schema
    input_data = pd.DataFrame({
        "age": [39, 50],
        "workclass": ["State-gov", "Self-emp-not-inc"],
        "fnlwgt": [77516, 83311],
        "education": ["Bachelors", "Bachelors"],
        "education-num": [13, 13],
        "marital-status": ["Never-married", "Married-civ-spouse"],
        "occupation": ["Adm-clerical", "Exec-managerial"],
        "relationship": ["Not-in-family", "Husband"],
        "race": ["White", "White"],
        "sex": ["Male", "Male"],
        "capital-gain": [2174, 0],
        "capital-loss": [0, 0],
        "hours-per-week": [40, 13],
        "native-country": ["United-States", "United-States"]
    })
    predictions = perform_inference(input_data)
    print(f"Predictions: {predictions}")