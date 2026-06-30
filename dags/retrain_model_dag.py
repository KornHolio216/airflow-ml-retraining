from airflow import DAG
from airflow.operators.python import PythonOperator

import datetime
import json
import os
import shutil

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


DATA_PATH = "/opt/airflow/data/new_data.csv"

ARCHIVE_DIR = "/opt/airflow/models/archive"
PRODUCTION_DIR = "/opt/airflow/models/production"
METRICS_DIR = "/opt/airflow/reports/metrics"
ALERTS_DIR = "/opt/airflow/alerts"

PRODUCTION_MODEL_PATH = "/opt/airflow/models/production/production_model.pkl"
PRODUCTION_SCORE_PATH = "/opt/airflow/models/production/production_score.txt"
PRODUCTION_METRICS_PATH = "/opt/airflow/models/production/production_metrics.json"
MODEL_ALERT_PATH = "/opt/airflow/alerts/model_performance_alert.txt"

MINIMUM_ACCURACY = 0.80


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42,
    )

    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{ARCHIVE_DIR}/rf_model_{timestamp}.pkl"

    joblib.dump(
        {
            "model": clf,
            "X_test": X_test,
            "y_test": y_test,
        },
        model_path,
    )

    print(f"Model zapisano w: {model_path}")

    return model_path


def validate_model(**context):
    task_instance = context["ti"]
    model_path = task_instance.xcom_pull(task_ids="train_model")

    saved_data = joblib.load(model_path)

    clf = saved_data["model"]
    X_test = saved_data["X_test"]
    y_test = saved_data["y_test"]

    y_pred = clf.predict(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))
    precision = float(precision_score(y_test, y_pred, zero_division=0))
    recall = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1]).astype(int).tolist()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(METRICS_DIR, exist_ok=True)
    metrics_path = f"{METRICS_DIR}/metrics_{timestamp}.json"

    metrics = {
        "evaluated_at": timestamp,
        "model_path": model_path,
        "minimum_accuracy": MINIMUM_ACCURACY,
        "metrics": {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": matrix,
        },
    }

    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print(f"Accuracy nowego modelu: {accuracy}")
    print(f"Precision nowego modelu: {precision}")
    print(f"Recall nowego modelu: {recall}")
    print(f"F1-score nowego modelu: {f1}")
    print(f"Confusion matrix nowego modelu: {matrix}")
    print(f"Raport metryk zapisano w: {metrics_path}")

    task_instance.xcom_push(key="new_model_path", value=model_path)
    task_instance.xcom_push(key="new_accuracy", value=accuracy)
    task_instance.xcom_push(key="new_metrics_path", value=metrics_path)
    task_instance.xcom_push(key="new_metrics", value=metrics)

    return accuracy


def compare_and_promote_model(**context):
    os.makedirs(PRODUCTION_DIR, exist_ok=True)
    os.makedirs(ALERTS_DIR, exist_ok=True)

    task_instance = context["ti"]

    new_model_path = task_instance.xcom_pull(
        task_ids="validate_model",
        key="new_model_path",
    )

    new_accuracy = task_instance.xcom_pull(
        task_ids="validate_model",
        key="new_accuracy",
    )

    new_metrics_path = task_instance.xcom_pull(
        task_ids="validate_model",
        key="new_metrics_path",
    )

    if os.path.exists(PRODUCTION_SCORE_PATH):
        with open(PRODUCTION_SCORE_PATH, "r", encoding="utf-8") as file:
            old_accuracy = float(file.read())
    else:
        old_accuracy = 0.0

    print(f"Accuracy starego modelu produkcyjnego: {old_accuracy}")
    print(f"Accuracy nowego modelu: {new_accuracy}")
    print(f"Minimalny wymagany accuracy: {MINIMUM_ACCURACY}")

    if new_accuracy < MINIMUM_ACCURACY:
        alert_message = (
            "Nowy model nie spelnia minimalnego progu jakosci. "
            f"Accuracy: {new_accuracy}, wymagane minimum: {MINIMUM_ACCURACY}."
        )
        with open(MODEL_ALERT_PATH, "w", encoding="utf-8") as file:
            file.write(alert_message)

        print(alert_message)
        print(f"Alert zapisano w: {MODEL_ALERT_PATH}")
        return

    if new_accuracy > old_accuracy:
        shutil.copy(new_model_path, PRODUCTION_MODEL_PATH)
        shutil.copy(new_metrics_path, PRODUCTION_METRICS_PATH)

        with open(PRODUCTION_SCORE_PATH, "w", encoding="utf-8") as file:
            file.write(str(new_accuracy))

        print("Nowy model jest lepszy i spelnia prog jakosci. Model zostal wdrozony do produkcji.")
    else:
        print("Nowy model nie jest lepszy. Pozostaje tylko w archiwum.")


default_args = {
    "owner": "Mateusz",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": datetime.timedelta(minutes=1),
}


with DAG(
    dag_id="retrain_model_dag",
    default_args=default_args,
    description="DAG do re-trenowania modelu ML z walidacja i wdrozeniem",
    schedule_interval="@daily",
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    tags=["ml", "airflow", "lab08"],
) as dag:
    train_task = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
    )

    validate_task = PythonOperator(
        task_id="validate_model",
        python_callable=validate_model,
    )

    promote_task = PythonOperator(
        task_id="compare_and_promote_model",
        python_callable=compare_and_promote_model,
    )

    train_task >> validate_task >> promote_task
