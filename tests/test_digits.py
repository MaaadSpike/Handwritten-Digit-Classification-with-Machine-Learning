import torch

from digits.attacks import fgsm, pgd
from digits.data import random_affine
from digits.models import CNN, MLP, build_model


def _batch(n=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 1, 28, 28, generator=g), torch.randint(0, 10, (n,), generator=g)


def test_output_shapes():
    x, _ = _batch()
    for m in (MLP(), CNN()):
        assert m.eval()(x).shape == (8, 10)


def test_build_model_rejects_unknown():
    try:
        build_model("resnet")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_attacks_respect_eps_ball_and_pixel_range():
    torch.manual_seed(0)
    model = CNN().eval()
    x, y = _batch()
    for eps in (0.05, 0.3):
        for x_adv in (fgsm(model, x, y, eps), pgd(model, x, y, eps, steps=5)):
            assert (x_adv - x).abs().max() <= eps + 1e-6
            assert x_adv.min() >= 0 and x_adv.max() <= 1


def test_eps_zero_is_identity():
    model = MLP().eval()
    x, y = _batch()
    assert torch.equal(fgsm(model, x, y, 0.0), x)
    assert torch.equal(pgd(model, x, y, 0.0), x)


def test_untargeted_fgsm_increases_loss():
    torch.manual_seed(0)
    model = MLP().eval()
    x, y = _batch(64)
    loss = torch.nn.functional.cross_entropy
    with torch.no_grad():
        clean = loss(model(x), y)
    adv = loss(model(fgsm(model, x, y, 0.1)), y).detach()
    assert adv > clean


def test_targeted_pgd_decreases_target_loss():
    torch.manual_seed(0)
    model = MLP().eval()
    x, _ = _batch(64)
    target = torch.full((64,), 3)
    loss = torch.nn.functional.cross_entropy
    with torch.no_grad():
        before = loss(model(x), target)
    after = loss(model(pgd(model, x, target, 0.1, steps=10, targeted=True)), target).detach()
    assert after < before


def test_random_affine_shape_and_range():
    x, _ = _batch()
    out = random_affine(x)
    assert out.shape == x.shape
    assert out.min() >= 0 and out.max() <= 1 + 1e-6
