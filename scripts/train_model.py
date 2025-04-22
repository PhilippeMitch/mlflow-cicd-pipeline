import mlflow
from xgboost import XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import os
import joblib
from pathlib import Path
import shap
from mlflow.models import MetricThreshold, infer_signature

# Configuration
MLFLOW_TRACKING_URI = "http://localhost:5000"
MODEL_NAME = "adult-classifier"
FEATURE_STORE_PATH = "feature_store/features.csv"
PREPROCESSOR_PATH = "feature_store/preprocessor.joblib"
ARTIFACT_PATH = "model"
EXPERIMENT_NAME = "adult-classifier-experiment"

# Ensure feature store directory exists
Path(FEATURE_STORE_PATH).parent.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load UCI Adult dataset and split into features and target."""
    X, y = shap.datasets.adult()
    X = X.reset_index(drop=True)  # Ensure proper DataFrame indexing
    return X, y


def create_preprocessor():
    """Create a preprocessor for numerical and categorical features."""
    numerical_features = [
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ]
    categorical_features = [
        "workclass", "education", "marital-status", "occupation",
        "relationship", "race", "sex", "native-country"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_features)
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
    # Define the model pipeline
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(random_state=42, eval_metric="logloss"))
    ])

    # Hyperparameter grid
    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 6],
        "classifier__learning_rate": [0.01, 0.1]
    }

    # Perform grid search
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


def main():
    """Main function to train, evaluate, and log models to MLflow."""
    # Set MLflow tracking URI and experiment
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load data
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # Create evaluation dataset
    eval_data = X_test.copy()
    eval_data["label"] = y_test

    # Create preprocessor
    preprocessor = create_preprocessor()

    # Start MLflow run for candidate model
    with mlflow.start_run(run_name="candidate") as candidate_run:
        # Train candidate model
        grid_search = train_candidate_model(X_train, y_train, preprocessor)
        candidate_model = grid_search.best_estimator_

        # Evaluate on test set
        y_pred = candidate_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log parameters and metrics
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Infer signature
        signature = infer_signature(X_train, candidate_model.predict(X_train))

        # Log candidate model
        candidate_model_uri = mlflow.sklearn.log_model(
            sk_model=candidate_model,
            artifact_path=ARTIFACT_PATH,
            signature=signature,
            registered_model_name=MODEL_NAME
        ).model_uri

        # Save preprocessor
        joblib.dump(candidate_model.named_steps["preprocessor"], PREPROCESSOR_PATH)
        mlflow.log_artifact(PREPROCESSOR_PATH)

        # Transform and save features to feature store
        X_transformed = candidate_model.named_steps["preprocessor"].transform(X_train)
        feature_names = (
            candidate_model.named_steps["preprocessor"]
            .get_feature_names_out()
            .tolist()
        )
        save_features_to_store(X_transformed, feature_names)

        # Evaluate candidate model using mlflow.evaluate
        candidate_result = mlflow.evaluate(
            candidate_model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
        )

    # Start MLflow run for baseline model
    with mlflow.start_run(run_name="baseline") as baseline_run:
        # Train baseline model
        baseline_model = train_baseline_model(X_train, y_train, preprocessor)

        # Infer signature
        signature = infer_signature(X_train, baseline_model.predict(X_train))

        # Log baseline model
        baseline_model_uri = mlflow.sklearn.log_model(
            sk_model=baseline_model,
            artifact_path=ARTIFACT_PATH,
            signature=signature
        ).model_uri

        # Evaluate baseline model using mlflow.evaluate
        baseline_result = mlflow.evaluate(
            baseline_model_uri,
            eval_data,
            targets="label",
            model_type="classifier",
        )

    # Define validation thresholds
    thresholds = {
        "accuracy_score": MetricThreshold(
            threshold=0.8,  # Accuracy should be >= 0.8
            min_absolute_change=0.05,  # At least 0.05 better than baseline
            min_relative_change=0.05,  # At least 5% better than baseline
            greater_is_better=True,
        ),
    }

    # Validate candidate model against baseline
    try:
        mlflow.validate_evaluation_results(
            candidate_result=candidate_result,
            baseline_result=baseline_result,
            validation_thresholds=thresholds,
        )
        print("Candidate model passed validation against baseline.")
    except Exception as e:
        print(f"Validation failed: {e}")

    print(f"Candidate model run ID: {candidate_run.info.run_id}")
    print(f"Baseline model run ID: {baseline_run.info.run_id}")
    print(f"Accuracy: {accuracy}, F1 Score: {f1}")

if __name__ == "__main__":
    main()