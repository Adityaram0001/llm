"""Tests for `llmlab.train.optimizers` (Lion, Muon) -- phase 5 Wave D."""

from __future__ import annotations

import pytest
import torch

from llmlab.train.optimizers import Lion, Muon, zeropower_via_newtonschulz5


def test_newtonschulz5_output_is_near_semi_orthogonal():
    """The whole point of the iteration: singular values of the output should cluster near 1
    (loosely, given bf16 + only 5 steps on a small matrix), not match the input's Marchenko-
    Pastur-distributed singular values (which for a random 32x48 gaussian span a much wider
    range, roughly [(sqrt(32)-sqrt(48))^2, (sqrt(32)+sqrt(48))^2] in *squared* terms)."""
    torch.manual_seed(0)
    G = torch.randn(32, 48)
    X = zeropower_via_newtonschulz5(G, steps=5)
    svals = torch.linalg.svdvals(X.float())
    assert svals.max() < 1.5
    assert svals.min() > 0.5
    assert (svals.max() - svals.min()) < (torch.linalg.svdvals(G).max() - torch.linalg.svdvals(G).min()) / 2


def test_newtonschulz5_handles_tall_and_wide_matrices():
    torch.manual_seed(0)
    for shape in [(48, 32), (32, 48), (16, 16)]:
        X = zeropower_via_newtonschulz5(torch.randn(*shape), steps=5)
        assert X.shape == shape
        assert torch.isfinite(X).all()


def test_muon_rejects_non_2d_params():
    p = torch.nn.Parameter(torch.randn(10))
    p.grad = torch.randn(10)
    opt = Muon([p], lr=0.02)
    with pytest.raises(ValueError, match="2D"):
        opt.step()


def test_muon_step_reduces_a_simple_quadratic_loss():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(16, 16))
    target = torch.randn(16, 16)
    opt = Muon([W], lr=0.05, momentum=0.9)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = (W - target).pow(2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]


def test_muon_state_dict_round_trips():
    torch.manual_seed(0)
    W = torch.nn.Parameter(torch.randn(8, 8))
    W.grad = torch.randn(8, 8)
    opt = Muon([W], lr=0.02)
    opt.step()
    state = opt.state_dict()

    W2 = torch.nn.Parameter(W.detach().clone())
    opt2 = Muon([W2], lr=0.02)
    opt2.load_state_dict(state)
    assert opt2.state[opt2.param_groups[0]["params"][0]]["momentum_buffer"] is not None


def test_lion_step_reduces_a_simple_quadratic_loss():
    torch.manual_seed(0)
    w = torch.nn.Parameter(torch.randn(10))
    target = torch.randn(10)
    opt = Lion([w], lr=1e-2, betas=(0.9, 0.99))

    losses = []
    for _ in range(50):
        opt.zero_grad()
        loss = (w - target).pow(2).mean()
        loss.backward()
        losses.append(loss.item())
        opt.step()
    assert losses[-1] < losses[0]


def test_lion_update_magnitude_is_lr_regardless_of_gradient_scale():
    """Lion's signature property: the step size is `lr` (times sign), independent of how big
    the gradient actually is -- unlike SGD/AdamW where a 100x larger gradient (before Adam's
    own normalization) moves the parameter further."""
    torch.manual_seed(0)
    w_small = torch.nn.Parameter(torch.zeros(5))
    w_big = torch.nn.Parameter(torch.zeros(5))
    opt_small = Lion([w_small], lr=0.1, betas=(0.9, 0.99))
    opt_big = Lion([w_big], lr=0.1, betas=(0.9, 0.99))

    w_small.grad = torch.ones(5) * 0.01
    w_big.grad = torch.ones(5) * 10.0
    opt_small.step()
    opt_big.step()
    assert torch.allclose(w_small.detach(), w_big.detach())


def test_lion_weight_decay_shrinks_params_towards_zero():
    """With a zero gradient, `sign(beta1*0 + (1-beta1)*0) == 0` so the sign-update contributes
    nothing -- isolating decoupled weight decay's effect: `p *= (1 - lr*weight_decay)`."""
    w = torch.nn.Parameter(torch.ones(5))
    opt = Lion([w], lr=0.1, weight_decay=0.1)
    w.grad = torch.zeros(5)
    opt.step()
    assert torch.allclose(w.detach(), torch.full((5,), 0.99))
