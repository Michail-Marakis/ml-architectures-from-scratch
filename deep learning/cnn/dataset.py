from __future__ import annotations
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

MNIST_MEAN_1CH = (0.1307,)
MNIST_STD_1CH = (0.3081,)


def _norm_stats(channels: int, scheme: str) -> Tuple[Tuple[float, ...], Tuple[float, ...]]:
    if scheme == "mnist":
        if channels == 1:
            return MNIST_MEAN_1CH, MNIST_STD_1CH
        # replicate MNIST stats to 3 channels
        return (MNIST_MEAN_1CH[0],) * channels, (MNIST_STD_1CH[0],) * channels

    if scheme == "imagenet":
        if channels != 3:
            raise ValueError("ImageNet normalization expects 3 channels")
        return IMAGENET_MEAN, IMAGENET_STD

    raise ValueError(f"Unknown normalization scheme: {scheme}")


def make_fashionmnist_loaders(
    root: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    img_size: int = 28,
    channels: int = 1,
    normalize: str = "mnist",  # "mnist" ή "imagenet"
    augment: bool = False,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create (train, val, test) loaders for FashionMNIST with a reproducible train/val split."""

    mean, std = _norm_stats(channels=channels, scheme=normalize)

    train_tfms: List[object] = []

    #FashionMNIST images are grayscale PIL images.
    if channels == 3:
        train_tfms.append(transforms.Grayscale(num_output_channels=3))

    if img_size != 28:
        train_tfms.append(transforms.Resize((img_size, img_size)))

    if augment:
        train_tfms.extend(
            [
                transforms.RandomRotation(degrees=10),
                transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            ]
        )

    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    test_tfms: List[object] = []

    if channels == 3:
        test_tfms.append(transforms.Grayscale(num_output_channels=3))

    if img_size != 28:
        test_tfms.append(transforms.Resize((img_size, img_size)))

    test_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_transform = transforms.Compose(train_tfms)
    test_transform = transforms.Compose(test_tfms)

    train_full = datasets.FashionMNIST(root=root, train=True, download=True, transform=train_transform)
    test_set = datasets.FashionMNIST(root=root, train=False, download=True, transform=test_transform)

    val_len = int(len(train_full) * val_ratio)
    train_len = len(train_full) - val_len

    #deterministic split
    g = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_full, [train_len, val_len], generator=g)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"FashionMNIST loaders | train={len(train_set)} val={len(val_set)} test={len(test_set)} | "
        f"img_size={img_size} channels={channels} augment={augment} normalize={normalize}"
    )

    return train_loader, val_loader, test_loader
