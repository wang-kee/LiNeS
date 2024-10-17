from pathlib import Path
from typing import Literal
import os

TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}{bar:-10b}"
MODELS = ["ViT-B-32", "ViT-B-16", "ViT-L-14"]
OPENCLIP_CACHEDIR = Path(Path.home(), "openclip-cachedir", "open_clip").as_posix()
CACHEDIR = None
DATA_DIR = os.environ.get("DATA_DIR", "/mnt/lts4/scratch/data")

ALL_DATASETS = [
    "EuroSAT",
    "DTD",
    "SUN397",
    "MNIST",
    "RESISC45",
    "GTSRB",
    "Cars",
    "SVHN",
    "STL10",
    "OxfordIIITPet",
    "Flowers102",
    "CIFAR100",
    "PCAM",
    "FER2013",
    "CIFAR10",
    "Food101",
    "FashionMNIST",
    "RenderedSST2",
    "EMNIST",
    "KMNIST",
]

DATASETS_8 = ALL_DATASETS[:8]
DATASETS_14 = ALL_DATASETS[:14]
DATASETS_20 = ALL_DATASETS[:20]


def cleanup_dataset_name(dataset_name: str):
    return dataset_name.replace("Val", "") + "Val"


def get_zeroshot_path(root, dataset, model):
    return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_zeroshot.pt").as_posix()


# def get_finetuned_path(root, dataset, model):
#     return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt").as_posix()


def get_finetuned_path(root, dataset, model):
    return Path(root, model, cleanup_dataset_name(dataset), f"nonlinear_finetuned.pt").as_posix()
    # return f"/mnt/lts4/scratch/home/ndimitri/dev/tall_masks/new_checkpoints/{model}/exponential/{cleanup_dataset_name(dataset)}/a=0.4/{cleanup_dataset_name(dataset)}.pt"


def get_single_task_accuracies_path(model):
    return Path("results/single_task", model, f"nonlinear_ft_accuracies.json").as_posix()


def get_zero_shot_accuracies_path(model):
    return Path(f"results/zero_shot/{model}_20tasks_zeroshot.json").as_posix()
