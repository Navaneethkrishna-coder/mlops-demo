import mlflow
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from monitor_drift import check_data_drift


# MLflow configuration
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("iris-continuous-retraining")

MODEL_NAME = "IrisRandomForestModel"
ACCURACY_THRESHOLD = 0.90


def train_and_register_model():

    print("Initiating retraining process...")

    # 1. Load dataset
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    # 2. Train model
    with mlflow.start_run() as run:

        n_estimators = 100
        max_depth = 5

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        # 3. Evaluate model
        predictions = model.predict(X_test)

        acc = accuracy_score(y_test, predictions)

        print(f"Candidate Model Accuracy: {acc:.4f}")

        # 4. Log parameters and metrics
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)

        # 5. Register model if accuracy passes threshold
        if acc >= ACCURACY_THRESHOLD:

            model_info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.tree._tree.Tree"
                ]
            )

            print(
                "Model successfully logged and registered!"
            )

            print(f"Model URI: {model_info.model_uri}")

            print(f"Run ID: {run.info.run_id}")

        else:

            print(
                f"Model accuracy below {ACCURACY_THRESHOLD}. "
                "Registration skipped."
            )


def run_pipeline():

    print("Checking production data for drift...")

    drift_detected = check_data_drift()

    if drift_detected:

        print("Data drift detected!")

        print("Starting automated retraining...")

        train_and_register_model()

    else:

        print("No significant data drift detected.")

        print("Model operating within normal bounds.")


if __name__ == "__main__":

    run_pipeline()
