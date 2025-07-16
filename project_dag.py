import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from pipeline_v2 import pipeline    # указать требуемую версию пайплайна

# variables:
''' Укажем путь к файлам проекта:
# -> $PROJECT_PATH при запуске в Airflow
# -> иначе - текущая директория при локальном запуске
'''
path = os.environ.get('PROJECT_PATH', '.')


args = {
    'owner': "Mawneil",                     # Информация о владельце DAG
    'start_date': dt.datetime(2025, 1, 1),  # Время начала выполнения пайплайна
    'retries': 1,                           # Количество повторений в случае неудач
    'retry_delay': dt.timedelta(minutes=1), # Пауза между повторениями
    'depends_on_past': True,               # Зависимость от успешного окончания предыдущего запуска
}

with DAG (
    dag_id='Analysis-of-the-SberAutosubscription',  # Имя DAG
    schedule="0 0 * * *",                           # Периодичность запуска
    default_args=args,
    tags=['sberautosubscription', 'ml'],
    doc_md="""## DAG для анализа SberAutoSubscription\n\nЗапускает ML-пайплайн и сервис предсказаний.""",
) as dag:
    learning_pipeline = PythonOperator(
        task_id='learning_ML',
        python_callable=pipeline,
    )

    start_prediction_service = BashOperator(
        task_id='prediction_service',
        bash_command=f'cd {path} && uvicorn main:app --host 0.0.0.0',
    )

    learning_pipeline >> start_prediction_service