"""MNIST loading.

MNIST is small enough (70k x 28 x 28) to keep entirely in memory as tensors,
which is much faster than a per-sample torchvision transform pipeline,
especially on CPU. Augmentation is applied per batch with `affine_grid`.

Pixel values stay in [0, 1]. Normalisation lives *inside* the models, so
adversarial perturbation budgets (eps) are always expressed in pixel units.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

VAL_SIZE = 5_000
SPLIT_SEED = 0


@dataclass
class Splits:
    train: TensorDataset
    val: TensorDataset
    test: TensorDataset


def _to_tensors(ds: MNIST) -> tuple[torch.Tensor, torch.Tensor]:
    x = ds.data.float().div_(255.0).unsqueeze(1)  # (N, 1, 28, 28) in [0, 1]
    y = ds.targets.long()
    return x, y


def load_mnist(root: str = "data", val_size: int = VAL_SIZE, seed: int = SPLIT_SEED) -> Splits:
    """Return train / val / test splits.

    The official 60k training set is split into train and validation; the
    official 10k test set is held out and only used for final evaluation.
    (The original notebook trained *and* evaluated on the 10k test set,
    which is why it reported 100% accuracy.)
    """
    train_full = MNIST(root, train=True, download=True)
    test = MNIST(root, train=False, download=True)

    x, y = _to_tensors(train_full)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(x), generator=g)
    val_idx, train_idx = perm[:val_size], perm[val_size:]

    xt, yt = _to_tensors(test)
    return Splits(
        train=TensorDataset(x[train_idx], y[train_idx]),
        val=TensorDataset(x[val_idx], y[val_idx]),
        test=TensorDataset(xt, yt),
    )


def make_loader(ds: TensorDataset, batch_size: int = 128, shuffle: bool = False, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


def random_affine(x: torch.Tensor, max_rot: float = 10.0, max_shift: float = 0.1,
                  scale: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """Small random rotation / translation / scaling applied to a whole batch.

    x: (B, 1, H, W) in [0, 1]. max_shift is a fraction of the image size.
    """
    b = x.shape[0]
    dev = x.device
    angle = (torch.rand(b, device=dev) * 2 - 1) * math.radians(max_rot)
    s = torch.empty(b, device=dev).uniform_(*scale)
    tx = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2  # affine_grid coords span [-1, 1]
    ty = (torch.rand(b, device=dev) * 2 - 1) * max_shift * 2
    cos, sin = torch.cos(angle) / s, torch.sin(angle) / s
    theta = torch.stack([
        torch.stack([cos, -sin, tx], dim=1),
        torch.stack([sin, cos, ty], dim=1),
    ], dim=1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    return F.grid_sample(x, grid, align_corners=False, padding_mode="zeros")
