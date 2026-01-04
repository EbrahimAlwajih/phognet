from __future__ import annotations

import os
import medmnist
import numpy as np
import torch
from medmnist import INFO
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from phognet.utils.seed_worker import seed_worker


class CustomMedMNISTDataset(Dataset):
    def __init__(self, DataClass, split: str, transform=None):
        self.dataset = DataClass(split=split, transform=None, download=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        img, label = self.dataset[idx]
        if isinstance(img, torch.Tensor):
            img = img.numpy().astype(np.uint8)
            img = np.transpose(img, (1, 2, 0))
            img = Image.fromarray(img)
        if self.transform:
            img = self.transform(img)

        if isinstance(label, np.ndarray) and label.ndim > 1:
            label = torch.tensor(np.argmax(label), dtype=torch.long)
        else:
            label = torch.tensor(label, dtype=torch.long).squeeze()

        return img, label


def download_datasets(dataset: str, size: int = 32):
    num_class_list = {
        "mnist": 10,
        "cifar10": 10,
        "cifar100": 100,
        "imagenet": 1000,
        "pathmnist": 9,
        "chestmnist": 14,
        "dermamnist": 7,
        "octmnist": 4,
        "pneumoniamnist": 2,
        "retinamnist": 5,
        "breastmnist": 2,
        "bloodmnist": 8,
        "tissuemnist": 8,
        "organamnist": 11,
        "organcmnist": 11,
        "organsmnist": 11,
        "FashionMNIST": 10,
        "KMNIST": 10,
        "QMNIST": 10,
        "SVHN": 10,
        "Places365": 365,
        "STL10": 10,
        "USPS": 10,
        "Omniglot": 1632,
        "GTSRB": 43,
        "Food101": 101,
    }

    if dataset == "mnist":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        train_dataset = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "FashionMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.FashionMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.FashionMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "KMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.KMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.KMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "QMNIST":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.QMNIST(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.QMNIST(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "SVHN":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.SVHN(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.SVHN(
            root="./data", split="test", download=True, transform=transform
        )
    elif dataset == "Places365":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_dataset = datasets.Places365(
            root="./data", split="train-standard", download=True, transform=transform
        )
        test_dataset = datasets.Places365(
            root="./data", split="val", download=True, transform=transform
        )
    elif dataset == "STL10":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.STL10(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.STL10(
            root="./data", split="test", download=True, transform=transform
        )
    elif dataset == "USPS":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.USPS(root="./data", train=True, download=True, transform=transform)
        test_dataset = datasets.USPS(root="./data", train=False, download=True, transform=transform)
    elif dataset == "Omniglot":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.Omniglot(
            root="./data", background=True, download=True, transform=transform
        )
        test_dataset = datasets.Omniglot(
            root="./data", background=False, download=True, transform=transform
        )
    elif dataset == "GTSRB":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        train_dataset = datasets.GTSRB(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.GTSRB(
            root="./data", split="test", download=True, transform=transform
        )
    elif dataset == "Food101":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
        train_dataset = datasets.Food101(
            root="./data", split="train", download=True, transform=transform
        )
        test_dataset = datasets.Food101(
            root="./data", split="test", download=True, transform=transform
        )
    elif dataset == "cifar10":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_dataset = datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset == "cifar100":
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
            ]
        )
        train_dataset = datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform
        )
    elif dataset in [
        "pathmnist",
        "chestmnist",
        "dermamnist",
        "octmnist",
        "pneumoniamnist",
        "retinamnist",
        "breastmnist",
        "bloodmnist",
        "tissuemnist",
        "organamnist",
        "organcmnist",
        "organsmnist",
    ]:
        info = INFO[dataset]
        DataClass = getattr(medmnist, info["python_class"])
        transform = transforms.Compose(
            [
                transforms.Resize([size, size]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
        train_dataset = CustomMedMNISTDataset(DataClass, split="train", transform=transform)
        test_dataset = CustomMedMNISTDataset(DataClass, split="test", transform=transform)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    if dataset == "chestmnist":
        task = "multi-label, binary-class"
    elif dataset in {"breastmnist", "pneumoniamnist"}:
        task = "binary-class"
    else:
        task = "multi-label"

    return train_dataset, test_dataset, num_class_list[dataset], task


def convert_to_7_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 1:
        gray_image = image[0]
        rgb_image = np.stack((gray_image, gray_image, gray_image), axis=-1)
    elif image.ndim == 3 and image.shape[0] == 3:
        rgb_image = image.transpose(1, 2, 0)
    else:
        raise ValueError("Unsupported image format. The image must be either grayscale or RGB.")

    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]

    rg_channel = (r_channel + g_channel) / 2.0
    rb_channel = (r_channel + b_channel) / 2.0
    gb_channel = (g_channel + b_channel) / 2.0
    rgb_channel = (r_channel + g_channel + b_channel) / 3.0

    return np.stack(
        (r_channel, g_channel, b_channel, rg_channel, rb_channel, gb_channel, rgb_channel), axis=0
    )


def convert_to_3_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] == 1:
        gray_image = image[0]
        rgb_image = np.stack((gray_image, gray_image, gray_image), axis=-1)
    elif image.ndim == 3 and image.shape[0] == 3:
        rgb_image = image.transpose(1, 2, 0)
    else:
        raise ValueError("Unsupported image format. The image must be either grayscale or RGB.")

    r_channel = rgb_image[:, :, 0]
    g_channel = rgb_image[:, :, 1]
    b_channel = rgb_image[:, :, 2]
    return np.stack((r_channel, g_channel, b_channel), axis=0)


class CustomDataset(Dataset):
    def __init__(self, dataset, n_channel: int = 3):
        self.dataset = dataset
        self.n_channel = n_channel

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, label = self.dataset[idx]
        image = image.numpy() if hasattr(image, "numpy") else np.array(image)

        if self.n_channel == 7:
            image = convert_to_7_channels(image)
        elif self.n_channel == 3:
            image = convert_to_3_channels(image)
        else:
            raise ValueError("n_channel must be 3 or 7")

        # --- fix warning: don't wrap an existing tensor with torch.tensor() ---
        if torch.is_tensor(label):
            label_t = label.clone().detach()
        else:
            label_t = torch.tensor(label)

        return torch.tensor(image, dtype=torch.float32), label_t.long().squeeze()


def prepare_datasets(dataset_name: str, img_size: int = 32, n_channel: int = 3):
    train_dataset, test_dataset, num_classes, task = download_datasets(dataset_name, size=img_size)
    train_dataset = CustomDataset(train_dataset, n_channel=n_channel)
    test_dataset = CustomDataset(test_dataset, n_channel=n_channel)
    return train_dataset, test_dataset, num_classes, task


def get_dataloaders(
    dataset_name: str, batch_size: int = 64, img_size: int = 32, n_channel: int = 3
):
    train_dataset, test_dataset, num_classes, task = prepare_datasets(
        dataset_name, img_size=img_size, n_channel=n_channel
    )

    num_workers = min(4, max(0, (os.cpu_count() or 2) // 2))
    pin = True

    drop_last = dataset_name == "organamnist"
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
        worker_init_fn=seed_worker,
        pin_memory=pin,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        worker_init_fn=seed_worker,
        pin_memory=pin,
        num_workers=num_workers,
    )

    return train_loader, test_loader, n_channel, num_classes, task
