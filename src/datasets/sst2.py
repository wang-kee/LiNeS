import os
import torch
import torchvision.datasets as datasets


class RenderedSST2:
    def __init__(
        self,
        preprocess,
        location=os.path.expanduser("~/data"),
        batch_size=128,
        num_workers=16,
    ):
        location = "home/kewang/data"

        location = os.path.join(location, "RenderedSST2")
        self.train_dataset = datasets.RenderedSST2(root=location, download=True, split="train", transform=preprocess)

        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test_dataset = datasets.RenderedSST2(root=location, download=True, split="test", transform=preprocess)

        self.test_loader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

        self.classnames = self.train_dataset.classes
