import torch
from torch import nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

train_data = datasets.MNIST(
    root='./data', 
    train=True, 
    download=True, 
    transform=transforms.ToTensor())

test_data = datasets.MNIST(
    root='./data', 
    train=False, 
    download=True, 
    transform=transforms.ToTensor())

# shuffle + batch
train_loader = DataLoader(train_data, batch_size=32, shuffle=True) # returns a DataLoader object
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
