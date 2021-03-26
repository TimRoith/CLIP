from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


def get_data_set(dataset, file, batch_size, test_size=1):
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    if dataset == "MNIST":
        train, test = get_mnist(file)
    elif dataset == "FashionMNIST":
        train, test = get_fashion_mnist(file)
    else:
        raise ValueError("Dataset:" + dataset + " not defined")
    train_loader, valid_loader, test_loader = train_valid_test_split(train, test, batch_size, train_split=0.9,
                                                                     test_size=test_size)

    return train_loader, valid_loader, test_loader

def get_mnist(file):
    transform_train = get_transform("MNIST", 1, 28, True)
    transforms_test = get_transform("MNIST", 1, 28, False)

    train = datasets.MNIST(file, train=True, download=False,
                           transform=transform_train)
    test = datasets.MNIST(file, train=False, download=False, 
                          transform=transforms_test)
    return train, test

def get_fashion_mnist(file):
    transform_train = get_transform("FashionMNIST", 1, 28, True)
    transforms_test = get_transform("FashionMNIST", 1, 28, False)

    train = datasets.FashionMNIST(file, train=True, download=False,transform=transform_train)
    test = datasets.FashionMNIST(file, train=False, download=False, transform=transforms_test)
    return train, test


def get_transform(data_set, channels, size, train):
    t = []
    t.append(transforms.ToTensor())
    if data_set == "MNIST":
        t.append(transforms.Normalize((0.1307,), (0.3081,))),
    transform = transforms.Compose(t)
    return transform

def get_mean_std(data_set):
    mean = 0.
    std = 1.
    if data_set == "MNIST":
        channels = 1
    elif data_set == "FashionMNIST":
        channels = 3
        
    # calculate mean and std
    if channels == 3:
        return torch.tensor(mean).view(3, 1, 1).cuda(), torch.tensor(std).view(3, 1, 1).cuda()
    else:
        return torch.tensor(mean), torch.tensor(std)


def get_lower_and_upper_limits():
    lower_limit = 0.
    upper_limit = 1.
    return lower_limit, upper_limit


def train_valid_test_split(train, test, batch_size, train_split=0.9, test_size=1):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    train, val = torch.utils.data.random_split(train, [train_count, val_count])

    if test_size != 1:
        test_count = int(len(test) * test_size)
        _count = len(test) - test_count
        test, _ = torch.utils.data.random_split(test, [test_count, _count])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(val, batch_size=1000, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, valid_loader, test_loader