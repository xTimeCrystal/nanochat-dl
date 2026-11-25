"""
Test FBT (GPTConfig.fbt_passes): passes of the whole model over the sequence, every later pass fed the previous pass's
final state of the position before.

Run: python -m pytest tests/test_fbt.py -v

A tiny model on CUDA if there is one (in the compute dtype), else on the CPU, with SDPA attention (as on every device).
AttnRes and the MoE run Triton kernels, so their tests need CUDA.
"""
import pytest
import torch
import nanochat.flash_attention as fa_module
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOL = 1e-2 if COMPUTE_DTYPE == torch.bfloat16 else 1e-5
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="AttnRes and the MoE run Triton kernels (CUDA)")
TRUNKS = [False, pytest.param(True, marks=cuda)] # attn_res
B, T, D = 2, 64, 128


@pytest.fixture(autouse=True)
def sdpa():
    """SDPA attention for these tests (restored after)."""
    saved = fa_module._override_impl
    fa_module._override_impl = 'sdpa'
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()
    yield
    fa_module._override_impl = saved
    fa_module.USE_FA3 = fa_module._resolve_use_fa3()


def config(passes, attn_res=False, moe=False):
    mlp = dict(shared_intermediate_size=2 * D, num_experts=8, num_experts_per_token=2, moe_intermediate_size=D // 4) if moe else dict(shared_intermediate_size=4 * D)
    return GPTConfig(sequence_len=T, vocab_size=512, n_layer=2, n_head=2, n_kv_head=2, n_embd=D, window_pattern="L",
                     attn_res=attn_res, fbt_passes=passes, **mlp)


def build(passes, attn_res=False, moe=False, like=None):
    """A model at init, or with the weights of `like` (a model with more passes, so W_s may be left out)."""
    with torch.device("meta"):
        model = GPT(config(passes, attn_res, moe))
    model.to_empty(device=DEVICE)
    torch.manual_seed(0)
    model.init_weights()
    if like is not None:
        model.load_state_dict(like.state_dict(), strict=passes > 1)
    return model


def trained(model):
    """Every weight moved off its init (the zero-init projections and W_s included), as after some training."""
    g = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for p in model.parameters():
            p.add_((0.05 * torch.randn(p.shape, generator=g)).to(p.device, p.dtype))
    return model


def tokens():
    g = torch.Generator().manual_seed(2)
    x = torch.randint(0, 512, (B, T), generator=g).to(DEVICE)
    return x, torch.roll(x, -1, 1)


def test_one_pass_is_the_plain_model():
    assert GPTConfig().fbt_passes == 1
    plain = build(1)
    assert not hasattr(plain, "fb_state")
    assert plain.num_scaling_params()["fbt"] == 0


@pytest.mark.parametrize("attn_res", TRUNKS)
def test_passes_start_as_the_plain_model(attn_res):
    # At init W_s is zero and the token input is unit-RMS (the smear starts at zero), so every pass computes the plain one
    model = build(3, attn_res)
    plain = build(1, attn_res, like=model)
    x, y = tokens()
    with torch.no_grad():
        assert torch.allclose(model(x), plain(x), atol=TOL)
        assert torch.allclose(model(x, y), 3 * plain(x, y), rtol=TOL) # training: the sum over the passes


@pytest.mark.parametrize("attn_res", TRUNKS)
def test_each_pass_fixes_one_more_position(attn_res):
    # A position reads its predecessor's previous-pass state, so position t stops changing after pass t + 1: after K passes,
    # positions 0..K-1 hold the serial recurrence (each position fed the final state of the one before it)
    K = 3
    model = trained(build(K + 1, attn_res))
    fewer = build(K, attn_res, like=model)
    x, _ = tokens()
    with torch.no_grad():
        a, b = fewer(x), model(x)
    assert torch.allclose(a[:, :K], b[:, :K], atol=TOL)
    assert (a[:, K] - b[:, K]).abs().max() > 10 * TOL


def test_training_loss_is_the_sum_of_the_passes():
    K = 3
    model = trained(build(K))
    x, y = tokens()
    with torch.no_grad():
        total = model(x, y)
        passes = [build(p, like=model)(x, y, loss_reduction="none").mean() for p in range(1, K + 1)] # (pass p: the last of p)
    assert torch.allclose(total, sum(passes), rtol=TOL)


def test_gradient_reaches_w_s():
    model = trained(build(2))
    x, y = tokens()
    model(x, y).backward()
    assert model.fb_state.weight.grad.abs().sum() > 0


def test_kv_cache_is_refused():
    model = build(2)
    cache = KVCache(batch_size=1, num_heads=2, seq_len=T, head_dim=D // 2, num_layers=2, device=DEVICE, dtype=COMPUTE_DTYPE)
    x, _ = tokens()
    with pytest.raises(AssertionError, match="FBT"):
        model(x[:1, :8], kv_cache=cache)


def test_flops_count_every_pass():
    with torch.device("meta"):
        plain, model = GPT(config(1)), GPT(config(3))
    assert model.estimate_flops() == 3 * plain.estimate_flops() + 6 * D * D * 2 # (W_s in passes 2 and 3)


@cuda
def test_moe_routes_every_pass():
    # Quantile Balancing reads the routing of every pass: a layer's selections since the last read cover all of them
    model = trained(build(2, moe=True))
    x, y = tokens()
    model(x, y)
    for _, mlp in model.layers():
        assert mlp.load.sum().item() == 2 * B * T * 2 # passes x tokens x num_experts_per_token
