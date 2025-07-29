from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from pipeline import data_loader, preprocessing, train

default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def retrain_pipeline():
    tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'TLT', 'LQD', 'GLD', 'XLE', 'XLV']
    price_path = "/mnt/data/quant_finance_pipeline/data/price.csv"
    model_path = "/mnt/data/quant_finance_pipeline/models/finance_model.h5"
    log_dir = "/mnt/data/quant_finance_pipeline/data/keras_tuner_logs"

    data_loader.load_or_update_price_csv(tickers, price_path)
    X, y = preprocessing.create_sequences_from_csv(price_path, window=30)
    split = int(len(X) * 0.8)
    train.train_model(
        X[:split], y[:split],
        input_shape=X.shape[1:],
        save_path=model_path,
        tuner_logdir=log_dir
    )

with DAG("weekly_retrain_pipeline", start_date=datetime(2023, 1, 1),
         schedule_interval="@weekly", catchup=True, default_args=default_args) as dag:

    retrain = PythonOperator(
        task_id="retrain_model",
        python_callable=retrain_pipeline
    )

    retrain