"""Tests of the MoE layer (nanochat/moe.py) and its use in GPT. The kernels are Triton: these need a CUDA GPU.

    python -m pytest tests/test_moe.py -v"""
import pytest
import torch
from nanochat.moe import K3MoE, margin_counts, quantile_balance, routed_experts, routed_experts_reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="the MoE kernels need a CUDA GPU")
dev = "cuda"


def routing(T, E, K, kind):
    s = torch.rand(T, E, device=dev)
    if kind == "skewed":  # expert 0 takes every token, the last expert none
        s[:, 0] += 2.0
        s[:, -1] -= 2.0
    elif kind == "one":  # everything on experts 0..K-1
        s[:, :K] += 5.0
    p = torch.rand(T, K, device=dev) + 0.1
    return s.topk(K, dim=1).indices, p / p.sum(1, keepdim=True)


@pytest.mark.parametrize("T,D,H,E,K,kind", [
    (1000, 96, 200, 5, 2, "uniform"), (1000, 96, 200, 5, 2, "skewed"), (777, 64, 130, 7, 3, "one"),
    (513, 100, 72, 4, 1, "uniform"), (300, 128, 256, 64, 4, "uniform"),  # ragged and mostly-empty tiles, K = 1
    (4096, 768, 1024, 8, 2, "skewed"), (2048, 512, 384, 32, 8, "uniform"),
])
def test_kernels_match_reference(T, D, H, E, K, kind):
    """Output and every gradient (x, wu, wd, p) against the float32 reference on the same bf16-valued inputs."""
    torch.manual_seed(0)
    x = torch.randn(T, D, device=dev).to(torch.bfloat16)
    wu = (torch.randn(E, H, D, device=dev) / D ** 0.5).to(torch.bfloat16).float()
    wd = (torch.randn(E, D, H, device=dev) / H ** 0.5).to(torch.bfloat16).float()
    sel, p = routing(T, E, K, kind)
    gy = torch.randn(T, D, device=dev).to(torch.bfloat16)
    results = []
    for fn, xin in ((routed_experts_reference, x.float()), (routed_experts, x)):
        leaves = [t.clone().requires_grad_() for t in (xin, wu, wd, p)]
        y = fn(*leaves[:3], sel, leaves[3])
        (y.float() * gy.float()).sum().backward()
        results.append([y] + [t.grad for t in leaves])
    for name, ref, got in zip(("y", "dx", "dwu", "dwd", "dp"), *results):
        err = ((got.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30)).item()
        assert err < 2e-2, f"{name}: relative error {err:.2e}"


@pytest.mark.parametrize("E,k", [(16, 2), (256, 16)])
def test_quantile_balancing_reaches_target(E, k):
    """From a router that overloads some experts, every expert gets close to k / E of the tokens."""
    torch.manual_seed(0)
    T, offset, bias = 32768, torch.randn(E, device=dev) * 1.5, torch.zeros(E, device=dev)
    for _ in range(16):  # fresh tokens every step
        s = torch.sigmoid(torch.randn(T, E, device=dev) + offset)
        top = (s + bias).topk(k + 1, dim=1)
        loads = torch.bincount(top.indices[:, :k].reshape(-1), minlength=E).float() / (T * k / E)
        bias = quantile_balance(margin_counts(s, top.values[:, k], bias), bias, k)
    assert 0.85 < loads.min().item() and loads.max().item() < 1.15, (loads.min().item(), loads.max().item())


def test_layer():
    """Gradients reach every parameter; a second pass can reuse the first one's experts; the bias moves only at balance()."""
    torch.manual_seed(0)
    layer = K3MoE(256, num_experts=16, top_k=2, expert_hidden=320, shared_hidden=320).to(dev)
    with torch.no_grad():
        layer.wd.normal_(0, 0.05)
        layer.shared_down.normal_(0, 0.05)
    x = torch.randn(4, 512, 256, device=dev).to(torch.bfloat16).requires_grad_()
    for _ in range(3):
        y = layer(x)
    y.float().square().sum().backward()
    assert y.shape == x.shape
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0 for p in layer.parameters())
    bias = layer.bias.clone()
    _, sel = layer.routed(x)
    assert torch.equal(layer.routed(x, sel)[1], sel) and torch.equal(layer.routed(x)[1], sel) and torch.equal(layer.bias, bias)
    assert layer.counts.sum().item() == 2 * 16 * x[..., 0].numel()  # two routing calls counted, the one given its experts not
    layer.balance()
    assert layer.counts.sum().item() == 0 and not torch.equal(layer.bias, bias)


def test_no_host_sync():
    """A layer's forward, backward and balance never wait for the GPU (no synchronizing call)."""
    layer = K3MoE(256, num_experts=8, top_k=2, expert_hidden=256, shared_hidden=256).to(dev)
    x = torch.randn(2048, 256, device=dev).to(torch.bfloat16).requires_grad_()
    layer(x).float().sum().backward()  # (warm up: compile the kernels)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        layer(x).float().sum().backward()
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_gpt_with_moe():
    """A small GPT with MoE layers: active parameters, a training step through MuonAdamW (3-D expert stacks), the load stats."""
    from nanochat.gpt import GPT, GPTConfig
    torch.manual_seed(0)
    cfg = GPTConfig(sequence_len=256, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2, n_embd=256, window_pattern="L",
                    num_experts=8, top_k=2, expert_hidden=256, shared_hidden=512)
    model = GPT(cfg).to(dev)
    model.init_weights()
    counts = model.num_scaling_params()
    dense = GPT(GPTConfig(sequence_len=256, vocab_size=1024, n_layer=2, n_head=2, n_kv_head=2, n_embd=256, window_pattern="L"))
    assert counts["active_matrices"] == dense.num_scaling_params()["transformer_matrices"]  # 512 + 2 * 256 = 4 * 256: the dense MLP's size
    assert counts["transformer_matrices"] > counts["active_matrices"]
    optimizer = model.setup_optimizer()
    x, y = (torch.randint(0, 1024, (2, 256), device=dev) for _ in range(2))
    for _ in range(2):
        model(x, y).backward()
        optimizer.step()
        model.zero_grad(set_to_none=True)
    assert all(torch.isfinite(p).all() for p in model.parameters())
    load = model.pop_moe_stats()
    assert load.shape == (2, 8) and torch.allclose(load.mean(1), torch.ones(2, device=dev))
    assert all(m.load.sum().item() == 0 for m in model.moes())
