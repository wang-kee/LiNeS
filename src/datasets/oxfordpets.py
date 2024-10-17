import os
import torch
import torchvision.datasets as datasets
from src.utils.variables_and_paths import DATA_DIR


class OxfordIIITPet:
    def __init__(self, preprocess, location=DATA_DIR, batch_size=128, num_workers=16):

        location = os.path.join(location, "OxfordIIITPet")
        self.train_dataset = datasets.OxfordIIITPet(
            root=location, download=True, split="trainval", transform=preprocess
        )

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.OxfordIIITPet(root=location, download=True, split="test", transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.train_dataset.classes
