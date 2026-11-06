"""Model definitions.

All models take images of shape (B, 1, 28, 28) with pixels in [0, 1] and
return raw logits (use `F.cross_entropy`). Input normalisation is done inside
the model so that attacks operate directly in pixel space.
"""

from __future__ import annotations

import torch
import torch.nn as nn

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class Normalize(nn.Module):
    def __init__(self, mean: float = MNIST_MEAN, std: float = MNIST_STD):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class MLP(nn.Module):
    """The original project's architecture: 784 -> 100 -> 50 -> 10.

    Kept as a baseline so the CNN's gains (accuracy and robustness) can be
    measured against it.
    """

    def __init__(self, hidden: tuple[int, ...] = (100, 50), num_classes: int = 10):
        super().__init__()
        layers: list[nn.Module] = [Normalize(), nn.Flatten()]
        d = 28 * 28
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _conv_block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        nn.ReLU(inplace=True),
    )


class CNN(nn.Module):
    """Small VGG-style CNN: [conv-conv-pool] x 2 -> FC(128) -> FC(10). ~0.47M params."""

    def __init__(self, num_classes: int = 10, width: int = 32, dropout: float = 0.3):
        super().__init__()
        w = width
        self.features = nn.Sequential(
            Normalize(),
            _conv_block(1, w), _conv_block(w, w), nn.MaxPool2d(2),          # 28 -> 14
            _conv_block(w, 2 * w), _conv_block(2 * w, 2 * w), nn.MaxPool2d(2),  # 14 -> 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * w * 7 * 7, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(name: str) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLP()
    if name == "cnn":
        return CNN()
    raise ValueError(f"unknown model {name!r} (expected 'mlp' or 'cnn')")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(path: str, map_location: str | torch.device = "cpu") -> nn.Module:
    """Load a checkpoint written by `digits.train` (state_dict + architecture name)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = build_model(ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
