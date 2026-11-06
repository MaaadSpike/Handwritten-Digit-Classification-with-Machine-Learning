"""Train a model on MNIST.

Examples
--------
    python -m digits.train --model mlp --epochs 15 --no-augment --out checkpoints/mlp.pt
    python -m digits.train --model cnn --epochs 8 --out checkpoints/cnn.pt
    python -m digits.train --model cnn --epochs 10 --adv-eps 0.3 --out checkpoints/cnn_adv.pt

`--adv-eps > 0` enables adversarial training: each batch is replaced by
adversarial examples generated on the fly, either with PGD (Madry et al., 2018;
the default) or with single-step FGSM + random start ("fast" AT, Wong et al.,
2020). The checkpoint with the best *robust* validation accuracy (PGD-10) is
kept. At eps = 0.3 the fast variant suffered catastrophic overfitting (see
results/logs/), so the reported robust model uses PGD.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attacks import pgd
from .data import load_mnist, make_loader, random_affine
from .models import build_model, count_params


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device) -> tuple[float, float]:
    """Return (mean loss, accuracy) on clean inputs."""
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(1) == y).sum().item()
        n += len(y)
    return total_loss / n, correct / n


def robust_accuracy(model: nn.Module, loader, device, eps: float, steps: int = 10,
                    max_batches: int | None = None) -> float:
    model.eval()
    correct, n = 0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        x_adv = pgd(model, x, y, eps=eps, steps=steps)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        n += len(y)
    return correct / n


def fgsm_rs_example(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float,
                    alpha: float) -> torch.Tensor:
    """FGSM with random initialisation, used for fast adversarial training."""
    delta = torch.empty_like(x).uniform_(-eps, eps)
    delta = ((x + delta).clamp(0, 1) - x).requires_grad_(True)
    loss = F.cross_entropy(model(x + delta), y)
    (grad,) = torch.autograd.grad(loss, delta)
    delta = (delta + alpha * grad.sign()).clamp(-eps, eps)
    return (x + delta).clamp(0, 1).detach()


def train(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = get_device()
    splits = load_mnist(args.data)
    train_dl = make_loader(splits.train, args.batch_size, shuffle=True, seed=args.seed)
    val_dl = make_loader(splits.val, 500)

    model = build_model(args.model).to(device)
    print(f"{args.model}: {count_params(model):,} params | device={device} | "
          f"train={len(splits.train)} val={len(splits.val)} test={len(splits.test)} (test untouched)")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=len(train_dl))

    adv = args.adv_eps > 0
    history: list[dict] = []
    best_score, best_state = -1.0, None
    total_steps = args.epochs * len(train_dl)
    warmup_steps = int(args.adv_warmup * total_steps)
    step = 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        run_loss, run_correct, n = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            if args.augment:
                x = random_affine(x)
            if adv:
                # Linearly ramp eps up over the warm-up fraction of training.
                eps = args.adv_eps * min(1.0, (step + 1) / max(1, warmup_steps))
                if args.adv_method == "fgsm_rs":
                    x = fgsm_rs_example(model, x, y, eps, alpha=1.25 * eps)
                else:
                    # Attack generated in train mode (as in Madry et al.). Generating it in eval mode
                    # let BatchNorm's train/eval statistics drift apart and collapsed clean accuracy.
                    x = pgd(model, x, y, eps, alpha=2.5 * eps / args.adv_steps, steps=args.adv_steps)
            logits = model(x)
            loss = F.cross_entropy(logits, y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            step += 1
            run_loss += loss.item() * len(y)
            run_correct += (logits.argmax(1) == y).sum().item()
            n += len(y)

        val_loss, val_acc = evaluate(model, val_dl, device)
        rec = {"epoch": epoch, "train_loss": run_loss / n, "train_acc": run_correct / n,
               "val_loss": val_loss, "val_acc": val_acc, "time_s": round(time.time() - t0, 1)}
        score = val_acc
        if adv:
            rec["val_pgd_acc"] = robust_accuracy(model, val_dl, device, eps=args.adv_eps,
                                                 steps=10, max_batches=2)  # 1,000 val images
            score = rec["val_pgd_acc"]
        history.append(rec)
        msg = (f"epoch {epoch:2d} | train loss {rec['train_loss']:.4f} acc {rec['train_acc']:.4f} | "
               f"val loss {val_loss:.4f} acc {val_acc:.4f}")
        if adv:
            msg += f" | val PGD-10@{args.adv_eps} {rec['val_pgd_acc']:.4f}"
        print(msg + f" | {rec['time_s']}s", flush=True)

        if score >= best_score:
            best_score = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"arch": args.model, "state_dict": best_state, "config": vars(args)}, out)
    hist_path = out.with_suffix(".history.json")
    hist_path.write_text(json.dumps(history, indent=2))
    print(f"saved best checkpoint (score={best_score:.4f}) to {out}")
    return {"history": history, "best_score": best_score}


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=["mlp", "cnn"], default="cnn")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                   help="random rotation/shift/scale (default: on)")
    p.add_argument("--adv-eps", type=float, default=0.0,
                   help="L-inf radius for adversarial training; 0 disables it")
    p.add_argument("--adv-method", choices=["pgd", "fgsm_rs"], default="pgd",
                   help="pgd: Madry et al. (robust, ~(steps+1)x cost); fgsm_rs: Wong et al. fast AT "
                        "(2x cost, but hit catastrophic overfitting at eps=0.3 here)")
    p.add_argument("--adv-steps", type=int, default=5, help="PGD steps used during training")
    p.add_argument("--adv-warmup", type=float, default=0.2,
                   help="fraction of training over which eps is ramped up")
    p.add_argument("--data", default="data")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/model.pt")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
