from sklearn.datasets import load_iris
from evidently import Report
from evidently.presets import DataDriftPreset


def check_data_drift():

    print("Loading Iris reference and production data...")

    iris = load_iris(as_frame=True)

    reference = iris.frame.sample(n=100, random_state=42)

    current = iris.frame.sample(n=100, random_state=99).copy()

    # Simulate drift in three features
    current["sepal length (cm)"] *= 1.5
    current["petal length (cm)"] *= 1.5
    current["petal width (cm)"] *= 1.5

    print("Generating Evidently drift report...")

    report = Report([
        DataDriftPreset()
    ])

    snapshot = report.run(
        reference_data=reference,
        current_data=current
    )

    snapshot.save_html("drift_report.html")

    report_dict = snapshot.dict()

    # Extract overall drift share
    dataset_drift = False

    for metric in report_dict["metrics"]:
        if metric["metric_name"].startswith("DriftedColumnsCount"):
            drift_share = metric["value"]["share"]

            print(f"Drifted columns share: {drift_share:.2f}")

            dataset_drift = drift_share >= 0.5
            break

    print("Drift report saved: drift_report.html")
    print(f"Dataset Drift Detected: {dataset_drift}")

    return dataset_drift


if __name__ == "__main__":
    check_data_drift()
