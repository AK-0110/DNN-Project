"""Dataset and data-pipeline code.

Two streams of data:

1. CIFAR-10 train (50,000 images, hard labels). Used only for backbone
   pretraining. Never used as a soft-label target for disagreement.

2. CIFAR-10H (10,000 images = the CIFAR-10 *test* split, with soft labels
   averaged from ~50 human annotators). This is the disagreement dataset.

The 10,000 are split with a fixed seed into 6,000 / 2,000 / 2,000.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]

# CIFAR-10 channel statistics (computed on the train set).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)


def get_train_transform(crop_padding: int = 4, hflip: bool = True) -> transforms.Compose:
    """Augmentations that do NOT change class semantics."""
    tfms = []
    if crop_padding > 0:
        tfms.append(transforms.RandomCrop(32, padding=crop_padding))
    if hflip:
        tfms.append(transforms.RandomHorizontalFlip())
    tfms += [
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ]
    return transforms.Compose(tfms)


def get_eval_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])


# ---------------------------------------------------------------------------
# Hard-label CIFAR-10 (50,000 images) — for pretraining only.
# ---------------------------------------------------------------------------

def cifar10_train_dataset(data_root: str, transform=None) -> datasets.CIFAR10:
    """Standard CIFAR-10 train set with hard labels."""
    return datasets.CIFAR10(
        root=data_root, train=True, download=True,
        transform=transform if transform is not None else get_train_transform(),
    )


# ---------------------------------------------------------------------------
# CIFAR-10H — soft labels over the CIFAR-10 *test* split.
# ---------------------------------------------------------------------------

class CIFAR10HSoft(Dataset):
    """CIFAR-10H: CIFAR-10 test images aligned with soft annotator distributions.

    The CIFAR-10H file `cifar10h-probs.npy` is shape (10000, 10) and is aligned
    in row-order with `torchvision.datasets.CIFAR10(train=False)`.
    """

    def __init__(
        self,
        data_root: str,
        cifar10h_probs_path: str,
        indices: Optional[np.ndarray] = None,
        transform=None,
    ):
        self.cifar = datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=None
        )
        probs_path = Path(cifar10h_probs_path)
        if not probs_path.exists():
            raise FileNotFoundError(
                f"CIFAR-10H probs file not found at {probs_path}.\n"
                "Download cifar10h-probs.npy from "
                "https://github.com/jcpeterson/cifar-10h and place it there."
            )
        self.soft_labels = np.load(probs_path).astype(np.float32)
        assert self.soft_labels.shape == (10000, 10), \
            f"Expected (10000, 10) probs, got {self.soft_labels.shape}"

        if indices is None:
            indices = np.arange(10000)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform if transform is not None else get_eval_transform()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        idx = int(self.indices[i])
        img, hard_label = self.cifar[idx]   # img is a PIL Image
        if self.transform is not None:
            img = self.transform(img)
        soft = torch.from_numpy(self.soft_labels[idx])
        return img, soft, int(hard_label), idx


def make_splits(n_total: int, n_train: int, n_val: int, n_test: int, seed: int):
    assert n_train + n_val + n_test <= n_total
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Sanity checks and dataset statistics.
# ---------------------------------------------------------------------------

def sanity_check_probs(probs: np.ndarray, atol: float = 1e-4) -> dict:
    """Verify that every row sums to 1 and is non-negative."""
    sums = probs.sum(axis=1)
    return {
        "shape": probs.shape,
        "min_value": float(probs.min()),
        "max_value": float(probs.max()),
        "min_row_sum": float(sums.min()),
        "max_row_sum": float(sums.max()),
        "n_rows_off_by_more_than_atol": int(np.sum(np.abs(sums - 1.0) > atol)),
        "atol": atol,
    }


def shannon_entropy(probs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Per-row entropy in bits. probs shape (N, C)."""
    p = np.clip(probs, eps, 1.0)
    return -np.sum(probs * np.log2(p), axis=1)


def per_class_average_entropy(probs: np.ndarray, hard_labels: np.ndarray) -> np.ndarray:
    """Average entropy of distributions whose argmax is each class."""
    H = shannon_entropy(probs)
    avg = np.zeros(10, dtype=np.float32)
    for c in range(10):
        mask = (hard_labels == c)
        avg[c] = H[mask].mean() if mask.any() else 0.0
    return avg


def soft_confusion_matrix(probs: np.ndarray, hard_labels: np.ndarray) -> np.ndarray:
    """Average annotator distribution conditioned on the majority/hard label.

    Output shape (10, 10): row c = average distribution for images whose hard
    label is c. The diagonal will be high (most annotators agree with the hard
    label); off-diagonals show which classes get confused with class c.
    """
    M = np.zeros((10, 10), dtype=np.float32)
    for c in range(10):
        mask = (hard_labels == c)
        if mask.any():
            M[c] = probs[mask].mean(axis=0)
    return M
