import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import torch 
import numpy as np

from src.models.model import LinearRegressionModel
from src.models.dataset import train_test_split

@click.command()
@click.option('--model_path', help='Model name to load ')
@click.option('--device', default='cpu', help='Model name to load ')
def main(model_path, device):
    """ Runs a prediction 
    """
    logger = logging.getLogger(__name__)
    
    # Create the model
    model = torch.load(model_path, weights_only=False).to(device)
    
    # Create the dataset
    X_train, y_train, X_test, y_test = train_test_split()
    
    t = X_test.to(device) 
    y = y_test.to(device)
    
    logger.info(f"Loaded model:\n{model} on {device}")    
    logger.info(f'Running {len(X_test)} predictions')
    model.eval()
    with torch.inference_mode():
        inf_preds = model(t)
        acc =  y / inf_preds
        acc = torch.mean(acc) * 100
        logger.info(f'Accuracy: {acc:.3f}%')

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

