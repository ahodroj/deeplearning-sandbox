import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy

from model import NeuralNetwork
from dataset import load_mnist_dataset


# Train the model
def train(dataloader, model, loss_fn, metrics_fn, optimizer, device):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        pred = model(X)
        loss = loss_fn(pred, y)
        accuracy = metrics_fn(pred, y)

        # Backpropagation.
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch
            mlflow.log_metric("loss", f"{loss:3f}", step=(batch // 100))
            mlflow.log_metric("accuracy", f"{accuracy:3f}", step=(batch // 100))
            print(
                f"loss: {loss:3f} accuracy: {accuracy:3f} [{current} / {len(dataloader)}]"
            )


@click.command()
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--device', default='mps', help='Target device')
def main(epochs, lr, batch_size, device):
    """ Runs a train-test loop 
    """
    logger = logging.getLogger(__name__)
    
    training_data = load_mnist_dataset()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=64)
    
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    logger.info('starting training run and tracking in mlflow')
    with mlflow.start_run():
        params = {
            "epochs": epochs,
            "learning_rate": lr,
            "batch_size": batch_size,
            "loss_function": loss_fn.__class__.__name__,
            "metric_function": metric_fn.__class__.__name__,
            "optimizer": "SGD",
        }
        # Log training parameters.
        mlflow.log_params(params)

        # Log model summary.
        with open("model_summary.txt", "w") as f:
            f.write(str(summary(model)))
        mlflow.log_artifact("model_summary.txt")

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train(train_dataloader, model, loss_fn, metric_fn, optimizer, device)

        # Save the trained model to MLflow.
        mlflow.pytorch.log_model(model, "model")
    
    logger.info('End of training')
    mlflow.end_run()
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()

