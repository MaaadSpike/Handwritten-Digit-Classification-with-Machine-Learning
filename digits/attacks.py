"""White-box L-infinity adversarial attacks.

Both attacks keep the result inside the valid pixel range [0, 1] and inside
the eps-ball around the clean input. Models must return logits.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _loss_grad(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x = x.detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
         targeted: bool = False) -> torch.Tensor:
    """Fast Gradient Sign Method (Goodfellow et al., 2015).

    Untargeted: `y` are the true labels and the loss on them is *increased*.
    Targeted:   `y` are the target labels and the loss on them is *decreased*.
    """
    if eps == 0:
        return x.clone()
    grad = _loss_grad(model, x, y)
    step = -eps * grad.sign() if targeted else eps * grad.sign()
    return (x + step).clamp(0.0, 1.0).detach()


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
        alpha: float | None = None, steps: int = 20, random_start: bool = True,
        targeted: bool = False) -> torch.Tensor:
    """Projected Gradient Descent (Madry et al., 2018): iterated FGSM with
    projection back onto the eps-ball. A much stronger attack than FGSM.
    """
    if eps == 0:
        return x.clone()
    if alpha is None:
        alpha = 2.5 * eps / steps
    x_adv = x.clone()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-eps, eps)).clamp(0.0, 1.0)
    for _ in range(steps):
        grad = _loss_grad(model, x_adv, y)
        x_adv = x_adv - alpha * grad.sign() if targeted else x_adv + alpha * grad.sign()
        x_adv = torch.min(torch.max(x_adv, x - eps), x + eps).clamp(0.0, 1.0).detach()
    return x_adv
