"""Evaluate trained checkpoints on the held-out MNIST test set and make figures.

    python -m digits.evaluate                  # all checkpoints, PGD on the full test set
    python -m digits.evaluate --n-robust 2000  # faster: attacks on the first 2,000 test images

Writes results/metrics.json and results/figures/*.png.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from torch.utils.data import TensorDataset  # noqa: E402

from .attacks import fgsm, pgd  # noqa: E402
from .data import load_mnist, make_loader  # noqa: E402
from .models import count_params, load_checkpoint  # noqa: E402
from .train import get_device, set_seed  # noqa: E402

MODELS = {  # key -> (checkpoint, display name)
    "mlp": ("checkpoints/mlp.pt", "MLP (original arch.)"),
    "cnn": ("checkpoints/cnn.pt", "CNN"),
    "cnn_adv": ("checkpoints/cnn_adv.pt", "CNN + adversarial training"),
}
EPS_GRID = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

# Reference categorical palette (slots 1-3 validate all-pairs for CVD) + line styles
# as a secondary encoding so identity never depends on colour alone.
STYLE = {
    "mlp": dict(color="#2a78d6", ls="-", marker="o"),
    "cnn": dict(color="#eb6834", ls="--", marker="s"),
    "cnn_adv": dict(color="#1baf7a", ls=":", marker="^"),
}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.titlecolor": INK, "axes.titleweight": "bold", "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False,
})


@torch.no_grad()
def predict(model, x: torch.Tensor, device, bs: int = 1000) -> torch.Tensor:
    out = [model(x[i:i + bs].to(device)).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out)


def attacked_accuracy(model, ds: TensorDataset, device, attack: str, eps: float, bs: int = 500) -> float:
    correct = 0
    for x, y in make_loader(ds, bs):
        x, y = x.to(device), y.to(device)
        if attack == "fgsm":
            x_adv = fgsm(model, x, y, eps)
        else:
            x_adv = pgd(model, x, y, eps, steps=20)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
    return correct / len(ds)


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = 10) -> np.ndarray:
    m = np.zeros((k, k), dtype=int)
    np.add.at(m, (y_true, y_pred), 1)
    return m


# ---------------------------------------------------------------- figures

def fig_training_curves(out: Path, keys: list[str], names: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for k in keys:
        hp = Path(MODELS[k][0]).with_suffix(".history.json")
        if not hp.exists():
            continue
        h = json.loads(hp.read_text())
        ep = [r["epoch"] for r in h]
        s = STYLE[k]
        axes[0].plot(ep, [r["train_loss"] for r in h], color=s["color"], ls=s["ls"], lw=2, label=names[k])
        axes[1].plot(ep, [100 * r["val_acc"] for r in h], color=s["color"], ls=s["ls"], lw=2,
                     marker=s["marker"], ms=5, label=names[k])
    axes[0].set(title="Training loss (adv. model: on its PGD examples)", xlabel="epoch",
                ylabel="cross-entropy", yscale="log")
    axes[1].set(title="Validation accuracy (clean)", xlabel="epoch", ylabel="accuracy (%)", ylim=(93.5, 100))
    axes[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_robustness(out: Path, metrics: dict, names: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for ax, attack, title in [(axes[0], "fgsm", "FGSM"), (axes[1], "pgd", "PGD-20 (stronger)")]:
        ends = []
        for k, m in metrics["models"].items():
            s = STYLE[k]
            eps = [float(e) for e in m[attack]]
            acc = [100 * v for v in m[attack].values()]
            ax.plot(eps, acc, color=s["color"], ls=s["ls"], marker=s["marker"], lw=2, ms=6,
                    label=names[k])
            ends.append((acc[-1], eps[-1]))
        # End-of-line value labels; labels closer than 6 points of accuracy are merged into one.
        ends.sort()
        groups: list[list[tuple[float, float]]] = []
        for e in ends:
            if groups and e[0] - groups[-1][-1][0] < 6:
                groups[-1].append(e)
            else:
                groups.append([e])
        for g in groups:
            text = " / ".join(f"{a:.1f}%" for a, _ in sorted(g, reverse=True))
            ax.annotate(text, (g[0][1], sum(a for a, _ in g) / len(g)), xytext=(6, 0),
                        textcoords="offset points", va="center", fontsize=9, color=INK)
        ax.set(title=f"{title} attack", xlabel="ε  (L∞ budget, pixel units)", xlim=(-0.01, 0.34), ylim=(-3, 103))
    axes[0].set_ylabel("test accuracy under attack (%)")
    axes[0].legend(loc="center right")
    fig.tight_layout()
    fig.savefig(out / "robustness_curves.png", bbox_inches="tight")
    plt.close(fig)


def fig_confusion(out: Path, cm: np.ndarray, name: str, fname: str) -> None:
    off = cm.copy()
    np.fill_diagonal(off, 0)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(off, cmap=SEQ_BLUE)
    ax.grid(False)
    ax.set(xticks=range(10), yticks=range(10), xlabel="predicted", ylabel="true",
           title=f"{name}: test errors (diagonal hidden)")
    vmax = off.max() if off.max() > 0 else 1
    for i in range(10):
        for j in range(10):
            if i != j and cm[i, j] > 0:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > 0.55 * vmax else INK)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="# test images")
    fig.tight_layout()
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)


def fig_misclassified(out: Path, x: torch.Tensor, y: np.ndarray, logits: torch.Tensor, name: str,
                      fname: str, n: int = 24) -> None:
    probs = logits.softmax(1).numpy()
    pred = probs.argmax(1)
    wrong = np.where(pred != y)[0]
    wrong = wrong[np.argsort(-probs[wrong, pred[wrong]])][:n]  # most confident mistakes first
    cols = 8
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.5))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for ax, i in zip(np.atleast_1d(axes).ravel(), wrong):
        ax.imshow(x[i, 0], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(f"true {y[i]} → {pred[i]}\n({probs[i, pred[i]]:.0%})", fontsize=8.5,
                     color=INK, fontweight="normal")
    fig.suptitle(f"{name}: most confident test-set mistakes ({len(np.where(pred != y)[0])} errors / {len(y):,})",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)


def first_correct_per_class(models: dict, x: torch.Tensor, y: torch.Tensor, device) -> list[int]:
    """Index of the first test image of each class classified correctly by *every* model."""
    ok = torch.ones(len(y), dtype=torch.bool)
    for m in models.values():
        ok &= predict(m, x, device).argmax(1) == y
    return [int(torch.where(ok & (y == c))[0][0]) for c in range(10)]


def fig_adversarial_examples(out: Path, models: dict, names: dict, x: torch.Tensor, y: torch.Tensor,
                             idx: list[int], device, eps: float = 0.15) -> None:
    """Rows: clean, then PGD-attacked against each model. One column per digit."""
    xb, yb = x[idx].to(device), y[idx].to(device)
    rows = [("clean", None, xb)]
    for k, m in models.items():
        rows.append((names[k], m, pgd(m, xb, yb, eps, steps=40)))
    fig, axes = plt.subplots(len(rows), 10, figsize=(12, 1.55 * len(rows) + 0.4))
    for r, (label, m, xa) in enumerate(rows):
        with torch.no_grad():
            p = (m or models["cnn"])(xa).softmax(1).cpu()
        for c in range(10):
            ax = axes[r, c]
            ax.imshow(xa[c, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            if m is not None:
                pred = int(p[c].argmax())
                fooled = pred != int(yb[c])
                ax.set_title(f"→ {pred} ({p[c, pred]:.0%})", fontsize=8.5,
                             color="#d03b3b" if fooled else INK, fontweight="bold" if fooled else "normal")
            if c == 0:
                ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=9, color=INK)
    fig.suptitle(f"PGD-40 attack, ε = {eps}: each model attacked with its own gradients "
                 f"(red = fooled)", fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / "adversarial_examples.png", bbox_inches="tight")
    plt.close(fig)


def fig_targeted(out: Path, model, name: str, x: torch.Tensor, idx: list[int], device,
                 eps: float, fname: str) -> dict:
    """10x10 grid: row = source digit, column = target class (targeted PGD)."""
    fig, axes = plt.subplots(10, 10, figsize=(9.5, 10.2))
    success, total = 0, 0
    for src in range(10):
        xs = x[idx[src]].unsqueeze(0).repeat(10, 1, 1, 1).to(device)
        tgt = torch.arange(10, device=device)
        xa = pgd(model, xs, tgt, eps, steps=100, alpha=0.01, targeted=True)
        xa[src] = xs[src]
        with torch.no_grad():
            p = model(xa).softmax(1).cpu()
        for t in range(10):
            ax = axes[src, t]
            ax.imshow(xa[t, 0].cpu(), cmap="gray_r", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            pred = int(p[t].argmax())
            if t != src:
                total += 1
                success += pred == t
            ax.set_title(f"{pred} ({p[t, pred]:.0%})", fontsize=7.5, pad=2,
                         color="#d03b3b" if (pred == t and t != src) else INK,
                         fontweight="bold" if (pred == t and t != src) else "normal")
            if t == 0:
                ax.set_ylabel(f"{src}", rotation=0, ha="right", va="center", fontsize=11, color=INK)
            if src == 0:
                ax.set_xlabel(f"target {t}", fontsize=9, color=INK_2)
                ax.xaxis.set_label_position("top")
    fig.suptitle(f"Targeted PGD on {name}, ε = {eps}: row = true digit, column = target class\n"
                 f"(title = prediction and confidence; red = attack reached its target: {success}/{total} cases)",
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(out / fname, bbox_inches="tight")
    plt.close(fig)
    return {"eps": eps, "success": success, "total": total}


# ---------------------------------------------------------------- main

def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-robust", type=int, default=10_000, help="# test images used for attack curves")
    p.add_argument("--data", default="data")
    p.add_argument("--out", default="results")
    p.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True,
                   help="reuse attack curves from results/robust_cache.json (default: on)")
    args = p.parse_args(argv)

    set_seed(0)
    device = get_device()
    out = Path(args.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)

    test = load_mnist(args.data).test
    x, y = test.tensors
    y_np = y.numpy()
    robust_ds = TensorDataset(x[:args.n_robust], y[:args.n_robust])

    keys = [k for k, (path, _) in MODELS.items() if Path(path).exists()]
    names = {k: MODELS[k][1] for k in keys}
    models = {k: load_checkpoint(MODELS[k][0], device).to(device) for k in keys}

    cache_path = out / "robust_cache.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    metrics: dict = {"n_test": len(y), "n_robust": args.n_robust, "pgd_steps": 20, "models": {}}
    for k, m in models.items():
        logits = predict(m, x, device)
        pred = logits.argmax(1).numpy()
        acc = float((pred == y_np).mean())
        cm = confusion(y_np, pred)
        entry = {
            "name": names[k], "params": count_params(m), "test_acc": acc,
            "test_errors": int((pred != y_np).sum()),
            "per_class_acc": {str(c): float(cm[c, c] / cm[c].sum()) for c in range(10)},
            "fgsm": {}, "pgd": {},
        }
        # Attack curves are slow (PGD-20 on 10k images per eps), so they are cached per model and
        # per n_robust; delete results/robust_cache.json (or pass --no-cache) to recompute.
        cache_key = f"{k}|n={args.n_robust}"
        if args.cache and cache_key in cache:
            entry["fgsm"], entry["pgd"] = cache[cache_key]["fgsm"], cache[cache_key]["pgd"]
            print(f"{k:8s} attack curves loaded from {cache_path}", flush=True)
        else:
            for eps in EPS_GRID:
                entry["fgsm"][str(eps)] = attacked_accuracy(m, robust_ds, device, "fgsm", eps)
                entry["pgd"][str(eps)] = attacked_accuracy(m, robust_ds, device, "pgd", eps)
                print(f"{k:8s} eps={eps:<5} FGSM {entry['fgsm'][str(eps)]:.4f}  PGD-20 {entry['pgd'][str(eps)]:.4f}",
                      flush=True)
            cache[cache_key] = {"fgsm": entry["fgsm"], "pgd": entry["pgd"]}
            cache_path.write_text(json.dumps(cache, indent=2))
        metrics["models"][k] = entry
        print(f"{k:8s} clean test acc {acc:.4f} ({entry['test_errors']} errors)", flush=True)
        fig_confusion(figs, cm, names[k], f"confusion_{k}.png")
        if k == "cnn":
            fig_misclassified(figs, x, y_np, logits, names[k], "misclassified_cnn.png")

    fig_training_curves(figs, keys, names)
    fig_robustness(figs, metrics, names)
    idx = first_correct_per_class(models, x, y, device)
    fig_adversarial_examples(figs, models, names, x, y, idx, device, eps=0.15)
    targeted = {}
    if "cnn" in models:
        targeted["cnn"] = fig_targeted(figs, models["cnn"], names["cnn"], x, idx, device, 0.2,
                                       "targeted_attack_cnn.png")
    if "cnn_adv" in models:
        targeted["cnn_adv"] = fig_targeted(figs, models["cnn_adv"], names["cnn_adv"], x, idx, device, 0.2,
                                           "targeted_attack_cnn_adv.png")
    metrics["targeted_pgd100"] = targeted

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"wrote {out / 'metrics.json'} and figures to {figs}/")


if __name__ == "__main__":
    main()
