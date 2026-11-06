"""Export CNN weights (and a small adversarial gallery) for the in-browser demo.

    python -m digits.export_web          # writes demo/model_data.js

BatchNorm is folded into the preceding convolution, weights are stored as
float16 (base64), and inference runs in plain JavaScript (demo/index.html) with
no ML framework. The gallery holds PGD adversarial examples for 10 test digits
at several eps values, crafted against each CNN, stored as 8-bit images.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .attacks import pgd
from .data import load_mnist
from .models import CNN, load_checkpoint

GALLERY_EPS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


def _b64(a: np.ndarray, dtype) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=dtype).tobytes()).decode()


def fold_cnn(model: CNN) -> list[dict]:
    """Flatten a CNN into a list of simple JSON-able layers with BN folded in."""
    layers: list[dict] = []
    for mod in model.features:
        if mod.__class__.__name__ == "Normalize":
            layers.append({"type": "norm", "mean": float(mod.mean), "std": float(mod.std)})
        elif isinstance(mod, nn.Sequential):  # conv-bn-relu block
            conv, bn = mod[0], mod[1]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            w = conv.weight * scale[:, None, None, None]
            b = bn.bias - bn.running_mean * scale
            if conv.bias is not None:
                b = b + conv.bias * scale
            layers.append({"type": "conv3x3_relu", "cin": conv.in_channels, "cout": conv.out_channels,
                           "w": _b64(w.detach().numpy(), np.float16), "b": _b64(b.detach().numpy(), np.float32)})
        elif isinstance(mod, nn.MaxPool2d):
            layers.append({"type": "maxpool2"})
    lin = [m for m in model.classifier if isinstance(m, nn.Linear)]
    for i, m in enumerate(lin):
        layers.append({"type": "linear", "din": m.in_features, "dout": m.out_features, "relu": i < len(lin) - 1,
                       "w": _b64(m.weight.detach().numpy(), np.float16), "b": _b64(m.bias.detach().numpy(), np.float32)})
    return layers


def main(argv=None) -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="demo/model_data.js")
    p.add_argument("--metrics", default="results/metrics.json")
    args = p.parse_args(argv)
    torch.manual_seed(0)

    models = {"cnn": load_checkpoint("checkpoints/cnn.pt"), "cnn_adv": load_checkpoint("checkpoints/cnn_adv.pt")}
    test = load_mnist().test
    x, y = test.tensors

    with torch.no_grad():
        ok = torch.ones(len(y), dtype=torch.bool)
        for m in models.values():
            ok &= m(x).argmax(1) == y
    idx = [int(torch.where(ok & (y == c))[0][0]) for c in range(10)]
    xs, ys = x[idx], y[idx]

    gallery = {"labels": ys.tolist(), "eps": GALLERY_EPS, "against": {}}
    for k, m in models.items():
        imgs = []
        for eps in GALLERY_EPS:
            xa = pgd(m, xs, ys, eps, steps=40) if eps > 0 else xs
            imgs.append((xa.squeeze(1).numpy() * 255).round().astype(np.uint8))  # (10, 28, 28)
        gallery["against"][k] = _b64(np.stack(imgs), np.uint8)  # (n_eps, 10, 28, 28)

    # Reference logits so the JS implementation can be checked against PyTorch.
    with torch.no_grad():
        ref = {k: m(xs[:3]).tolist() for k, m in models.items()}

    payload = {
        "models": {k: fold_cnn(m) for k, m in models.items()},
        "gallery": gallery,
        "reference": {"images": _b64((xs[:3].squeeze(1).numpy() * 255).round().astype(np.uint8), np.uint8),
                      "logits": ref},
    }
    mp = Path(args.metrics)
    if mp.exists():
        met = json.loads(mp.read_text())["models"]
        payload["metrics"] = {k: {"test_acc": v["test_acc"], "pgd": v["pgd"]} for k, v in met.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.MODEL_DATA = " + json.dumps(payload, separators=(",", ":")) + ";\n")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
