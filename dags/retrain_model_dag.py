from airflow import DAG
from airflow.operators.python import PythonOperator

import datetime
import os
import shutil

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib


DATA_PATH = "/opt/airflow/data/new_data.csv"

ARCHIVE_DIR = "/opt/airflow/models/archive"
PRODUCTION_DIR = "/opt/airflow/models/production"

PRODUCTION_MODEL_PATH = "/opt/airflow/models/production/production_model.pkl"
PRODUCTION_SCORE_PATH = "/opt/airflow/models/production/production_score.txt"


def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df.drop("target", axis=1)
    y = df["target"]

    #podział danych na treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=42
    )

    #trenowanie modelu
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    #zapis modelu do archiwum
    os.makedirs(ARCHIVE_DIR, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"{ARCHIVE_DIR}/rf_model_{timestamp}.pkl"

    joblib.dump(
        {
            "model": clf,
            "X_test": X_test,
            "y_test": y_test
        },
        model_path
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
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy nowego modelu: {accuracy}")

    task_instance.xcom_push(key="new_model_path", value=model_path)
    task_instance.xcom_push(key="new_accuracy", value=accuracy)

    return accuracy


def compare_and_promote_model(**context):
    os.makedirs(PRODUCTION_DIR, exist_ok=True)

    task_instance = context["ti"]

    new_model_path = task_instance.xcom_pull(
        task_ids="validate_model",
        key="new_model_path"
    )

    new_accuracy = task_instance.xcom_pull(
        task_ids="validate_model",
        key="new_accuracy"
    )

    if os.path.exists(PRODUCTION_SCORE_PATH):
        with open(PRODUCTION_SCORE_PATH, "r") as file:
            old_accuracy = float(file.read())
    else:
        old_accuracy = 0.0

    print(f"Accuracy starego modelu produkcyjnego: {old_accuracy}")
    print(f"Accuracy nowego modelu: {new_accuracy}")

    if new_accuracy > old_accuracy:
        shutil.copy(new_model_path, PRODUCTION_MODEL_PATH)

        with open(PRODUCTION_SCORE_PATH, "w") as file:
            file.write(str(new_accuracy))

        print("Nowy model jest lepszy. Model został wdrożony do produkcji.")
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
    description="DAG do re-trenowania modelu ML z walidacją i wdrożeniem",
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