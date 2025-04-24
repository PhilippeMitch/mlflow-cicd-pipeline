import mlflow
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.impute import SimpleImputer
import pandas as pd
import os
import joblib
from pathlib import Path
import redis
from mlflow.models import MetricThreshold, infer_signature
import json

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "adult-classifier"
FEATURE_STORE_PATH = "feature_store/features.csv"
PREPROCESSOR_PATH = "feature_store/preprocessor.joblib"
ARTIFACT_PATH = "model"
EXPERIMENT_NAME = "adult-classifier-experiment"
REDIS_TTL = 86400  # 24 hours in seconds

# Ensure feature store directory exists
Path(FEATURE_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)

# Initialize Redis client
def initialize_redis():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        r.ping()
    except redis.ConnectionError as e:
        print(f"Warning: Failed to connect to Redis: {e}")
        r = None
    return r

r = initialize_redis()

def load_data():
    """Load UCI Adult dataset and split into features and target."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
    columns = [
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country", "income"
    ]
    data = pd.read_csv(url, names=columns, skipinitialspace=True)
    X = data.drop("income", axis=1)
    y = (data["income"] == ">50K").astype(int)

    # Convert integer columns to float64
    numerical_features = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
    X[numerical_features] = X[numerical_features].astype("float64")

    X = X.reset_index(drop=True)
    print("Columns in X:", X.columns.tolist())
    return X, y

def create_preprocessor(X):
    """Create a preprocessor for numerical and categorical features."""
    expected_columns = {
        "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
        "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
        "hours-per-week", "native-country"
    }
    actual_columns = set(X.columns)
    missing_cols = expected_columns - actual_columns
    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    numerical_features = X.select_dtypes(include=["float64", "int64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    print("Numerical features:", numerical_features)
    print("Categorical features:", categorical_features)
    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_pipeline, numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ]
    )
    return preprocessor

def save_features_to_store(X_transformed, feature_names):
    """Save transformed features to the feature store."""
    transformed_df = pd.DataFrame(X_transformed, columns=feature_names)
    transformed_df.to_csv(FEATURE_STORE_PATH, index=False)
    print(f"Saved transformed features to {FEATURE_STORE_PATH}")

def load_features_from_store():
    """Load transformed features from the feature store for inference."""
    if os.path.exists(FEATURE_STORE_PATH):
        return pd.read_csv(FEATURE_STORE_PATH)
    else:
        raise FileNotFoundError(f"Feature store not found at {FEATURE_STORE_PATH}")

def train_candidate_model(X_train, y_train, preprocessor):
    """Train the XGBoost model with hyperparameter tuning."""
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.01, 0.1]
    }

    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    return grid_search

def train_baseline_model(X_train, y_train, preprocessor):
    """Train a baseline DummyClassifier."""
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DummyClassifier(strategy="uniform"))
    ])
    model.fit(X_train, y_train)
    return model

def cache_metrics_and_metadata(run_type, model_name, metrics, metadata):
    """Cache metrics and metadata in Redis with TTL."""
    if r is None:
        print("Redis unavailable, skipping caching.")
        return

    try:
        for metric_name, value in metrics.items():
            key = f"model:{model_name}:{run_type}:{metric_name}"
            r.set(key, str(value), ex=REDIS_TTL)
            print(f"Cached {key} = {value}")

        for meta_name, value in metadata.items():
            key = f"model:{model_name}:{run_type}:{meta_name}"
            if isinstance(value, dict):
                value = json.dumps(value)
            r.set(key, str(value), ex=REDIS_TTL)
            print(f"Cached {key} = {value}")
    except redis.RedisError as e:
        print(f"Warning: Failed to cache in Redis: {e}")

def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    preprocessor = create_preprocessor(X)

    with mlflow.start_run(run_name="candidate") as candidate_run:
        grid_search = train_candidate_model(X_train, y_train, preprocessor)
        candidate_model = grid_search.best_estimator_

        y_pred = candidate_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        candidate_metrics = {"accuracy": accuracy, "f1_score": f1}
        candidate_metadata = {"run_id": candidate_run.info.run_id, "best_params": grid_search.best_params_}
        cache_metrics_and_metadata("candidate", MODEL_NAME, candidate_metrics, candidate_metadata)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Create realistic dataset for schema inference
        X_train_sample = X_train.copy()
        for col in ["age", "capital-gain", "hours-per-week"]:
            X_train_sample.loc[X_train_sample.sample(frac=0.1, random_state=42).index, col] = None
        signature = infer_signature(X_train_sample, candidate_model.predict(X_train))

        # Define custom conda environment
        conda_env = {
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.9",
                "pip=23.3.1",
                {
                    "pip": [
                        "mlflow==2.21.3",
                        "psycopg2-binary==2.9.6",
                        "scikit-learn>=1.0.0",
                        "xgboost>=1.7.0",
                        "pandas>=1.5.0",
                        "numpy>=1.23.0"
                    ]
                }
            ],
            "name": "mlflow-env"
        }

        candidate_model_uri = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            artifact_path=ARTIFACT_PATH,
            signature=signature,
            registered_model_name=MODEL_NAME,
            conda_env=conda_env
        ).model_uri

        joblib.dump(candidate_model.named_steps["preprocessor"], PREPROCESSOR_PATH)
        mlflow.log_artifact(PREPROCESSOR_PATH)

        X_transformed = candidate_model.named_steps["preprocessor"].transform(X_train)
        feature_names = candidate_model.named_steps["preprocessor"].get_feature_names_out().tolist()
        save_features_to_store(X_transformed, feature_names)

        # Evaluate candidate model using untransformed data
        X_test_eval = X_test.reset_index(drop=True)
        eval_data = X_test_eval.copy()
        eval_data["label"] = y_test.astype(int).reset_index(drop=True)

        candidate_result = mlflow.evaluate(
            model=candidate_model_uri,
            data=eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
            evaluator_config={
                "default": {
                    "log_model_explainability": True,
                    "explainability_algorithm": "permutation",  # Changed from "shap" to "permutation"
                    "max_evals": 100
                }
            }
        )

    with mlflow.start_run(run_name="baseline") as baseline_run:
        baseline_model = train_baseline_model(X_train, y_train, preprocessor)
        y_pred_baseline = baseline_model.predict(X_test)
        baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
        baseline_f1 = f1_score(y_test, y_pred_baseline, average="weighted")

        baseline_metrics = {"accuracy": baseline_accuracy, "f1_score": baseline_f1}
        baseline_metadata = {"run_id": baseline_run.info.run_id}
        cache_metrics_and_metadata("baseline", MODEL_NAME, baseline_metrics, baseline_metadata)

        signature = infer_signature(X_train_sample, baseline_model.predict(X_train))

        baseline_model_uri = mlflow.sklearn.log_model(
            sk_model=baseline_model,
            artifact_path=ARTIFACT_PATH,
            signature=signature,
            conda_env=conda_env
        ).model_uri

        # Evaluate baseline model using untransformed data
        X_test_eval = X_test.reset_index(drop=True)
        eval_data = X_test_eval.copy()
        eval_data["label"] = y_test.astype(int).reset_index(drop=True)

        baseline_result = mlflow.evaluate(
            model=baseline_model_uri,
            data=eval_data,
            targets="label",
            model_type="classifier",
            evaluators=["default"],
            evaluator_config={
                "default": {
                    "log_model_explainability": True,
                    "explainability_algorithm": "permutation",  # Changed from "shap" to "permutation"
                    "max_evals": 100
                }
            }
        )

    thresholds = {
        "accuracy_score": MetricThreshold(
            threshold=0.8,
            min_absolute_change=0.05,
            min_relative_change=0.05,
            greater_is_better=True,
        ),
    }

    validation_status = "passed"
    try:
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=baseline_result,
            validation_thresholds=thresholds,
        )
        print("Candidate model passed validation against baseline.")
    except Exception as e:
        validation_status = "failed"
        print(f"Validation failed: {e}")

    if r is not None:
        try:
            r.set(f"model:{MODEL_NAME}:validation_status", validation_status, ex=REDIS_TTL)
            print(f"Cached model:{MODEL_NAME}:validation_status = {validation_status}")
        except redis.RedisError as e:
            print(f"Warning: Failed to cache validation status: {e}")

    print(f"Candidate model run ID: {candidate_run.info.run_id}")
    print(f"Baseline model run ID: {baseline_run.info.run_id}")
    print(f"Candidate Accuracy: {accuracy}, F1 Score: {f1}")
    print(f"Baseline Accuracy: {baseline_accuracy}, F1 Score: {baseline_f1}")

if __name__ == "__main__":
    main()