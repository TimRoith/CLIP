from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch

import os


def get_data_set(conf, test_size=1):
    train, valid, test, train_loader, valid_loader, test_loader = [None] * 6
    if conf.data_set == "MNIST":
        conf.im_shape = [1,28,28]
        
        # set mean and std for this dataset
        conf.data_set_mean = 0.1307
        conf.data_set_std = 0.3081
        train, test = get_mnist(conf)
    elif conf.data_set == "Fashion-MNIST":
        conf.data_set_mean = 0.5
        conf.data_set_std = 0.5
        
        conf.im_shape = [1,28,28]
        train, test = get_fashion_mnist(conf.data_file)
    else:
        raise ValueError("Dataset:" + conf.data_set + " not defined")
    train_loader, valid_loader, test_loader = train_valid_test_split(train, test, conf.batch_size, train_split=conf.train_split,
                                                                     test_size=test_size,num_workers=conf.num_workers)

    return train_loader, valid_loader, test_loader

def get_mnist(conf):
    transform = get_transform(conf, 1, 28, True)
    #
    train = datasets.MNIST(conf.data_file, train=True, download=conf.download, transform=transform)
    test = datasets.MNIST(conf.data_file, train=False, download=conf.download, transform=transform)
    return train, test

def get_fashion_mnist(file):
    transform_train = get_transform("FashionMNIST", 1, 28, True)
    transforms_test = get_transform("FashionMNIST", 1, 28, False)

    train = datasets.FashionMNIST(file, train=True, download=conf.download,transform=transform_train)
    test = datasets.FashionMNIST(file, train=False, download=conf.download, transform=transforms_test)
    return train, test


def get_transform(conf, channels, size, train):
    t = []
    t.append(transforms.ToTensor())
    # compose the transform
    transform = transforms.Compose(t)
    return transform


def train_valid_test_split(train, test, batch_size, train_split=0.9, test_size=1, num_workers=1):
    total_count = len(train)
    train_count = int(train_split * total_count)
    val_count = total_count - train_count
    train, val = torch.utils.data.random_split(train, [train_count, val_count],generator=torch.Generator().manual_seed(42))

    if test_size != 1:
        test_count = int(len(test) * test_size)
        _count = len(test) - test_count
        test, _ = torch.utils.data.random_split(test, [test_count, _count])

    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=num_workers)
    valid_loader = DataLoader(val, batch_size=1000, shuffle=True, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    return train_loader, valid_loader, test_loader
