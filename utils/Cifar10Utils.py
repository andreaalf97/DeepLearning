# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms

class Cifar10Util:
    num_workers = 2
    batch_size = 4

    classes = (
        'plane',
        'car',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck'
    )

    def __init__(self):
        print("Beginning Loading of Utils")

        # The output of torchvision datasets are PILImage images of range [0, 1].
        # We transform them to Tensors of normalized range [-1, 1].
        self.transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]  # 0.5 is the mean and std
        )

        self.trainSet = torchvision.datasets.CIFAR10(
            root='../data',
            train=True,
            download=True,
            transform=self.transform)

        self.testSet = torchvision.datasets.CIFAR10(
            root='../data',
            train=False,
            download=True,
            transform=self.transform
        )

        self.trainLoader = torch.utils.data.DataLoader(
            self.trainSet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers
        )
    
        self.testLoader = torch.utils.data.DataLoader(
            self.testSet,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers
        )

        print("Utils Loaded")