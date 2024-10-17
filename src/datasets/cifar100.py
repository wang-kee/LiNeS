import os
import torch
from torchvision.datasets import CIFAR100 as PyTorchCIFAR100
from src.utils.variables_and_paths import DATA_DIR


class CIFAR100:
    def __init__(self, preprocess, location=DATA_DIR, batch_size=128, num_workers=16):

        self.train_dataset = PyTorchCIFAR100(root=location, download=True, train=True, transform=preprocess)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, num_workers=num_workers
        )

        self.test_dataset = PyTorchCIFAR100(root=location, download=True, train=False, transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.test_dataset.classes
