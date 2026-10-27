import json
import os
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

# --------------------------------------------------
# 1. Paths and folders
# --------------------------------------------------
DATA_DIR = Path("prepared_data")
MODEL_DIR = Path("deployment")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# 2. Load train and test data from the artifact
# --------------------------------------------------
X_train = pd.read_csv(DATA_DIR / "X_train.csv")
X_test = pd.read_csv(DATA_DIR / "X_test.csv")
y_train = pd.read_csv(DATA_DIR / "y_train.csv").iloc[:, 0]
y_test = pd.read_csv(DATA_DIR / "y_test.csv").iloc[:, 0]

if not (
    len(X_train) == len(y_train)
    and len(X_test) == len(y_test)
):
    raise ValueError("Feature and target row counts do not match.")

if list(X_train.columns) != list(X_test.columns):
    raise ValueError("Train and test feature columns do not match.")

print("Training features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training target:", y_train.shape)
print("Testing target:", y_test.shape)

# --------------------------------------------------
# 3. Define model and parameter grid
# --------------------------------------------------
model = RandomForestClassifier(
    random_state=42,
    n_jobs=1,
    class_weight="balanced",
)

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
    "min_samples_split": [2, 5],
}

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

# --------------------------------------------------
# 4. Configure MLflow
# --------------------------------------------------
# mlflow.set_tracking_uri("file:./mlruns")
# mlflow.set_experiment("PredictiveMaintenance-RandomForest")

# Configure MLflow to use SQLite tracking
mlflow.set_tracking_uri("sqlite:///mlflow.db")

mlflow.set_experiment("PredictiveMaintenance-RandomForest")

# --------------------------------------------------
# 5. Tune the model and log every parameter combination
# --------------------------------------------------
with mlflow.start_run(run_name="RandomForest_GridSearch") as parent_run:

    mlflow.log_param("model_type", "RandomForestClassifier")
    mlflow.log_param("cv_strategy", "StratifiedKFold")
    mlflow.log_param("cv_folds", 5)
    mlflow.log_param("scoring", "f1_macro")
    mlflow.log_param("random_state", 42)
    mlflow.log_param("class_weight", "balanced")

    search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)

    # Save every candidate combination and its CV results.
    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(REPORT_DIR / "cv_results.csv", index=False)

    mlflow.log_artifact(str(REPORT_DIR / "cv_results.csv"))

    # Log each parameter combination as an individual MLflow run.
    for index, row in cv_results.iterrows():
        with mlflow.start_run(
            run_name=f"GridCandidate_{index + 1:02d}",
            nested=True,
        ):
            candidate_params = row["params"]

            for key, value in candidate_params.items():
                mlflow.log_param(key, str(value))

            mlflow.log_metric(
                "mean_cv_f1_macro",
                float(row["mean_test_score"]),
            )
            mlflow.log_metric(
                "std_cv_f1_macro",
                float(row["std_test_score"]),
            )
            mlflow.log_metric(
                "mean_train_f1_macro",
                float(row["mean_train_score"]),
            )
            mlflow.log_metric(
                "cv_rank",
                float(row["rank_test_score"]),
            )

    best_model = search.best_estimator_

    mlflow.log_metric(
        "best_cv_f1_macro",
        float(search.best_score_),
    )

    for key, value in search.best_params_.items():
        mlflow.log_param(f"best_{key}", str(value))

    # --------------------------------------------------
    # 6. Evaluate the best model on held-out test data
    # --------------------------------------------------
    y_pred = best_model.predict(X_test)
    y_probability = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(
            precision_score(
                y_test, y_pred, average="macro", zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_test, y_pred, average="macro", zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_test, y_pred, average="macro", zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(y_test, y_probability)
        ),
        "best_cv_f1_macro": float(search.best_score_),
    }

    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    with open(REPORT_DIR / "test_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    pd.DataFrame(
        matrix,
        index=["Actual_0", "Actual_1"],
        columns=["Predicted_0", "Predicted_1"],
    ).to_csv(REPORT_DIR / "confusion_matrix.csv")

    report = classification_report(
        y_test,
        y_pred,
        zero_division=0,
        output_dict=True,
    )
    with open(REPORT_DIR / "classification_report.json", "w") as file:
        json.dump(report, file, indent=4)

    mlflow.log_artifact(str(REPORT_DIR / "test_metrics.json"))
    mlflow.log_artifact(str(REPORT_DIR / "confusion_matrix.csv"))
    mlflow.log_artifact(str(REPORT_DIR / "classification_report.json"))

    # Save the complete fitted model.
    model_path = MODEL_DIR / "best_model.pkl"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(str(model_path))

    mlflow.sklearn.log_model(
    sk_model=best_model,
    artifact_path="random_forest_model",
    skops_trusted_types=["sklearn.tree._tree.Tree"]
)

    print("\nBest parameters:", search.best_params_)
    print("Best cross-validation macro F1:", search.best_score_)
    print("\nHeld-out test metrics:")
    print(json.dumps(metrics, indent=4))
    print("\nConfusion matrix:")
    print(matrix)
    print("\nSaved model:", model_path)
    print("Saved CV results:", REPORT_DIR / "cv_results.csv")
    print("MLflow parent run:", parent_run.info.run_id)
