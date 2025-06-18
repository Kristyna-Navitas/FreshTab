from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import subprocess


def run_dataset_creation():
    """Run the dataset creation script."""
    subprocess.run(["python3", "dataset_creation.py"], check=True)


default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG(
        dag_id="dataset_creation_monthly",
        default_args=default_args,
        schedule_interval="0 0 1 * *",  # Run at the start of each month
        catchup=False,
) as dag:
    task = PythonOperator(
        task_id="run_dataset_creation",
        python_callable=run_dataset_creation,
    )