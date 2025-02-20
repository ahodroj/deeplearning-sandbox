import torch 
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np 

from torch.utils.data import Dataset, DataLoader

# Load mnist dataset
def load_mnist_dataset():
    return datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )
        
def load_mnist_dataset_test():
    return datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )