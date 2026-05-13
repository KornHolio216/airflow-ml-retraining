# LAB08 Airflow ML Retraining

Projekt z laboratorium 08 z przedmiotu **Nowoczesne Technologie Przetwarzania Danych**.

Celem projektu jest uruchomienie Apache Airflow w Dockerze oraz przygotowanie prostego procesu automatycznego re-trenowania modelu ML. DAG trenuje model, wykonuje walidację i warunkowo podmienia model produkcyjny tylko wtedy, gdy nowa wersja osiąga lepszy wynik.

## Technologie

- Apache Airflow 2.9.3
- Docker / Docker Compose
- PostgreSQL 13
- Python 3.11
- pandas
- scikit-learn
- joblib

## Struktura projektu

```text
lab08_airflow/
├── dags/
│   └── retrain_model_dag.py
├── data/
│   └── new_data.csv
├── logs/
├── models/
│   ├── archive/
│   └── production/
├── Dockerfile
├── docker-compose.yaml
├── requirements.txt
└── README.md
```

## Opis działania

DAG `retrain_model_dag` składa się z trzech zadań:

1. `train_model` – wczytuje dane z pliku CSV, trenuje model `RandomForestClassifier` i zapisuje go w folderze `models/archive` z timestampem.
2. `validate_model` – oblicza `accuracy` nowego modelu na zbiorze testowym.
3. `compare_and_promote_model` – porównuje wynik nowego modelu z wynikiem modelu produkcyjnego. Jeśli nowy model jest lepszy, zostaje skopiowany do folderu `models/production` jako `production_model.pkl`.

## Uruchomienie projektu

Najpierw uruchom usługę inicjalizującą Airflow:

```bash
docker compose up --build airflow-init
```

Następnie uruchom całe środowisko:

```bash
docker compose up --build
```

Panel Airflow będzie dostępny pod adresem:

```text
http://localhost:8080
```

Dane logowania:

```text
login: admin
hasło: admin
```

## Sprawdzenie działania

Po uruchomieniu Airflow należy wejść w DAG:

```text
retrain_model_dag
```

Następnie można uruchomić go ręcznie z poziomu panelu. Po poprawnym wykonaniu w widoku Graph powinny być widoczne trzy zielone taski:

```text
train_model -> validate_model -> compare_and_promote_model
```

W logach zadania `validate_model` widoczna jest wartość `accuracy` nowego modelu, a w logach `compare_and_promote_model` informacja, czy model został wdrożony do produkcji.

## Wynik działania

Po wykonaniu DAG-a modele archiwalne są zapisywane w:

```text
models/archive/
```

Aktualny model produkcyjny oraz jego wynik są zapisywane w:

```text
models/production/
├── production_model.pkl
└── production_score.txt
```