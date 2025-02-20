import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os

import torch 
import numpy as np
import mlflow

from dataset import load_mnist_dataset_test


@click.command()
@click.option('--experiment', default='Default', help='Target MLflow experiment')
def main(experiment):
    """ Runs a prediction 
    """
    logger = logging.getLogger(__name__)
    
    logger.info(f'Running a prediction for {experiment}')
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    client = mlflow.tracking.MlflowClient()
    current_experiment=dict(mlflow.get_experiment_by_name(experiment))
    logger.info(f'Found experiemnt {current_experiment}')
    experiment_id=current_experiment['experiment_id']
    best_run = client.search_runs(
        experiment_id, order_by=["metrics.eval_accuracy DESC"], max_results=1
    )[0]

    model_uri = "runs:/{}/model".format(best_run.info.run_id)
    
    logger.info(f'Best model run URI: {model_uri}')
    
    input_data = load_mnist_dataset_test()
    
    mlflow.models.predict(
        model_uri=model_uri,
        input_data=[input_data[0][0].reshape((28,28)).numpy()],
        env_manager="uv",
    )
    
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()

