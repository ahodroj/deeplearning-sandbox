import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor

# Load mnist dataset
def load_mnist_dataset():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
        
