import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv, set_key
import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics import Accuracy

from model import NeuralNetwork
from dataset import load_mnist_dataset, load_mnist_dataset_test


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

def evaluate(dataloader, model, loss_fn, metrics_fn, epoch, device):
    num_batches = len(dataloader)
    model.eval()
    eval_loss, eval_accuracy = 0, 0 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            eval_loss += loss_fn(pred, y).item()
            eval_accuracy += metrics_fn(pred, y)
        
        eval_loss /= num_batches
        eval_accuracy /= num_batches
        mlflow.log_metric("eval_loss", f"{eval_loss:.2f}", step=epoch)
        mlflow.log_metric("eval_accuracy", f"{eval_accuracy:.2f}", step=epoch)
        
        print(f"Eval metrics: Accuracy: {eval_accuracy:.2f}, Avg loss: {eval_loss:.2f}\n")
                

@click.command()
@click.option('--epochs', default=10, help='Number of training epochs')
@click.option('--lr', default=1e-3, help='Learning rate')
@click.option('--batch_size', default=64, help='Batch size')
@click.option('--device', default='mps', help='Target device')
@click.option('--experiment', default='Default', help='Target MLflow experiment')
def main(epochs, lr, batch_size, device, experiment):
    """ Runs a train-test loop 
    """
    logger = logging.getLogger(__name__)
    
    training_data = load_mnist_dataset()
    testing_data = load_mnist_dataset_test()

    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=64)
    test_dataloader = DataLoader(testing_data, batch_size=64)
    
    loss_fn = nn.CrossEntropyLoss()
    metric_fn = Accuracy(task="multiclass", num_classes=10).to(device)
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(experiment)

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
            evaluate(test_dataloader, model, loss_fn, metric_fn, t, device)

        # Save the trained model to MLflow.
        model_info = mlflow.pytorch.log_model(model, "model")
        set_key(find_dotenv(), "MODEL_INFO_URI", model_info.model_uri)
        logger.info(f'End of training, model saved at {model_info.model_uri}')

    mlflow.end_run()
    

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    load_dotenv(find_dotenv())

    main()
