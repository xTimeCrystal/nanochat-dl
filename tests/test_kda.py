"""
The Kimi K3 hybrid (GPTConfig.kda_hybrid): FLA's chunk_kda against the KDA recurrence written out step by step (CUDA), the
layer layout, NoPE, and the KDA layer's wiring with the recurrence standing in for the kernel (CPU).

Run: python -m pytest tests/test_kda.py -v
"""
import sys
import types

import pytest
import torch
import torch.nn.functional as F

import nanochat.gpt as gpt
from nanochat.gpt import GPT, GPTConfig, KimiDeltaAttention, is_kda


def kda_reference(q, k, v, g, beta, A_log, dt_bias, scale=None, use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
                  safe_gate=False, lower_bound=None, **kw):
    """Kimi K3 eqs. 1 and 5 one step at a time, with chunk_kda's interface (gate and L2 norm in the "kernel"):
    log alpha = lower_bound Sigmoid(e^A (g + b)); S_t = Diag(alpha_t) S_{t-1} + beta_t k_t (v_t - (Diag(alpha_t) S_{t-1})^T k_t)^T,
    which is (I - beta k k^T) Diag(alpha) S + beta k v^T; o_t = S_t^T (scale q_t)."""
    assert use_qk_l2norm_in_kernel and use_gate_in_kernel and safe_gate
    B, T, H, K = q.shape
    f = lambda x: x.to(torch.promote_types(x.dtype, torch.float32))  # (fp32 at least)
    q, k = F.normalize(f(q), dim=-1), F.normalize(f(k), dim=-1)
    a = lower_bound * torch.sigmoid(f(A_log).exp()[:, None] * (f(g) + f(dt_bias).view(H, K)))
    scale = K ** -0.5 if scale is None else scale
    S, out = torch.zeros(B, H, K, v.shape[-1], device=q.device, dtype=q.dtype), []
    for t in range(T):
        S = S * a[:, t].exp()[..., None]
        u = f(v[:, t]) - torch.einsum("bhkv,bhk->bhv", S, k[:, t])
        S = S + f(beta[:, t])[..., None, None] * torch.einsum("bhk,bhv->bhkv", k[:, t], u)
        out.append(torch.einsum("bhkv,bhk->bhv", S, q[:, t] * scale))
    return torch.stack(out, 1).to(v.dtype), None


@pytest.fixture
def reference_kernel(monkeypatch):
    """KDA layers on CPU: the step-by-step recurrence as fla.ops.kda.chunk_kda."""
    fla = types.ModuleType("fla")
    fla.ops = types.ModuleType("fla.ops")
    fla.ops.kda = types.ModuleType("fla.ops.kda")
    fla.ops.kda.chunk_kda = kda_reference
    for name, mod in (("fla", fla), ("fla.ops", fla.ops), ("fla.ops.kda", fla.ops.kda)):
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


def make(**kw):
    c = dict(sequence_len=64, vocab_size=96, n_layer=8, n_head=2, n_kv_head=2, n_embd=64, window_pattern="SSSL",
             kda_hybrid=True)
    c.update(kw)
    torch.manual_seed(0)
    m = GPT(GPTConfig(**c))
    m.init_weights()
    with torch.no_grad():
        for block in m.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.2)
            block.mlp.c_proj.weight.normal_(0, 0.2)
        m.lm_head.weight.normal_(0, 0.2)
    return m.eval()


def test_layout():
    """3 KDA layers then 1 global attention layer, and the last layer always global attention."""
    for n, want in ((8, "KKKAKKKA"), (6, "KKKAKA"), (12, "KKKAKKKAKKKA"), (9, "KKKAKKKAA")):
        c = GPTConfig(n_layer=n, kda_hybrid=True)
        assert "".join("K" if is_kda(i, c) else "A" for i in range(n)) == want, n
    assert not any(is_kda(i, GPTConfig(n_layer=8)) for i in range(8))


def test_nope_and_global(reference_kernel):
    """The attention layers use no rotary embedding (shifting every position leaves the output unchanged) and see the
    full context; value embeddings only on attention layers."""
    m = make()
    assert all(w == (64, 0) for w in m.window_sizes)
    assert all(not is_kda(int(i), m.config) for i in m.value_embeds)
    attn = m.transformer.h[3].attn
    assert attn.nope and attn.c_gate is not None
    x = torch.randn(2, 10, 64)
    cs0 = (m.cos[:, :10], m.sin[:, :10])
    cs7 = (m.cos[:, 7:17], m.sin[:, 7:17])
    with torch.no_grad():
        torch.testing.assert_close(attn(x, None, cs0, (64, 0), None), attn(x, None, cs7, (64, 0), None))


@torch.no_grad()
def test_attn_gate_alone(reference_kernel):
    """attn_gate without the hybrid: every layer is attention with RoPE and the baseline's windows, each with the sigmoid
    output gate before the output projection: y = W_o [Sigmoid(W_g x) * attention output]."""
    m = make(kda_hybrid=False, attn_gate=True)
    assert all(not isinstance(b.attn, KimiDeltaAttention) and b.attn.c_gate is not None and not b.attn.nope
               for b in m.transformer.h)
    assert m.window_sizes == make(kda_hybrid=False).window_sizes
    attn = m.transformer.h[0].attn
    attn.c_proj.weight.copy_(torch.eye(64))  # (the output is then the gated attention output itself)
    x, cs = torch.randn(2, 10, 64), (m.cos[:, :10], m.sin[:, :10])
    gated = attn(x, None, cs, m.window_sizes[0], None)
    gate, attn.c_gate = attn.c_gate, None
    torch.testing.assert_close(gated, torch.sigmoid(gate(x)) * attn(x, None, cs, m.window_sizes[0], None))


@torch.no_grad()
def test_kda_layer_against_recurrence(reference_kernel):
    """The layer: q, k, v from width-4 causal depthwise convolutions and Swish, beta = Sigmoid(W_beta x), the decay
    logit W_up W_down x, output W_o [Sigmoid(W_g x) * RMSNorm_head(o)]: rebuilt by hand around the recurrence."""
    m = make()
    layer = m.transformer.h[0].attn
    assert isinstance(layer, KimiDeltaAttention)
    x = torch.randn(2, 9, 64)
    B, T, C, H, D = 2, 9, 64, layer.n_head, layer.head_dim
    def conv(y, w):  # causal: output t reads inputs t-3..t
        yp = torch.cat([torch.zeros(B, 3, C), y], 1)
        return F.silu(sum(yp[:, j:j + T] * w[:, 0, j] for j in range(4)))
    q = conv(layer.c_q(x), layer.conv_q).view(B, T, H, D)
    k = conv(layer.c_k(x), layer.conv_k).view(B, T, H, D)
    v = conv(layer.c_v(x), layer.conv_v).view(B, T, H, D)
    o, _ = kda_reference(q, k, v, layer.c_decay_up(layer.c_decay_down(x)).view(B, T, H, D), torch.sigmoid(layer.c_beta(x)),
                         layer.A_log, layer.dt_bias, use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                         safe_gate=True, lower_bound=-5.0)
    want = layer.c_proj(torch.sigmoid(layer.c_gate(x)) * gpt.norm(o).view(B, T, C))
    torch.testing.assert_close(layer(x, None, None, None, None), want, rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_kda_without_gate_params(reference_kernel, monkeypatch):
    """kda_gate_params=False: no A or b; the kernel gets the precomputed decay log alpha = -5 Sigmoid(W_up W_down x) in
    its safe range (no gate computed in the kernel)."""
    m = make(kda_gate_params=False)
    layer = m.transformer.h[0].attn
    assert not hasattr(layer, "A_log") and not hasattr(layer, "dt_bias")
    seen = {}
    def kernel(q, k, v, g, beta, **kw):  # (records what the kernel is given)
        seen.update(g=g, kw=kw)
        return v, None
    monkeypatch.setattr(sys.modules["fla.ops.kda"], "chunk_kda", kernel)
    x = torch.randn(2, 9, 64)
    layer(x, None, None, None, None)
    z = layer.c_decay_up(layer.c_decay_down(x)).view(2, 9, layer.n_head, layer.head_dim)
    torch.testing.assert_close(seen["g"], -5.0 * torch.sigmoid(z))
    assert seen["kw"]["safe_gate"] and seen["kw"]["lower_bound"] == -5.0 and not seen["kw"].get("use_gate_in_kernel")
    m.train()
    m(torch.randint(0, 96, (2, 12)), torch.randint(0, 96, (2, 12)))  # (runs; every parameter in a group)
    m.setup_optimizer()


@torch.no_grad()
def test_causal(reference_kernel):
    """Changing token j leaves the logits at every earlier position unchanged."""
    m = make()
    idx = torch.randint(0, 96, (2, 12))
    idx2 = idx.clone()
    idx2[:, 6] = (idx2[:, 6] + 1) % 96
    a, b = m(idx), m(idx2)
    torch.testing.assert_close(a[:, :6], b[:, :6], rtol=0, atol=1e-5)
    assert not torch.allclose(a[:, 6:], b[:, 6:])


def test_training_step(reference_kernel):
    """A finite loss, gradients in every KDA parameter, every parameter in an optimizer group (Muon: the 2-D ones)."""
    m = make().train()
    loss = m(torch.randint(0, 96, (2, 12)), torch.randint(0, 96, (2, 12)))
    loss.backward()
    assert torch.isfinite(loss)
    for name, p in m.transformer.h[0].attn.named_parameters():
        assert p.grad is not None and p.grad.abs().sum() > 0, name
    m.setup_optimizer()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FLA's kernels need CUDA")
def test_chunk_kda_matches_recurrence():
    """FLA's chunk_kda (K3's lower-bounded gate, L2 norm in the kernel) == the step-by-step recurrence in fp64, output and
    every gradient, within 1% (relative norm; the kernel's internal matmul precision gives ~0.2%), in the model's own
    regime: A = 0, the bias as at init, a small decay logit."""
    import math
    from fla.ops.kda import chunk_kda
    torch.manual_seed(0)
    B, T, H, K, dev = 2, 200, 2, 64, "cuda"
    dt = torch.exp(torch.rand(H * K, device=dev) * (math.log(0.1) - math.log(0.001)) + math.log(0.001))
    base = [torch.randn(B, T, H, K, device=dev) for _ in range(3)] + [0.3 * torch.randn(B, T, H, K, device=dev),
            torch.rand(B, T, H, device=dev), torch.zeros(H, device=dev), dt + torch.log(-torch.expm1(-dt))]
    do = torch.randn(B, T, H, K, device=dev)

    def run(fn, dtype):
        t = [x.to(dtype).clone().requires_grad_() for x in base]
        o, _ = fn(*t[:5], A_log=t[5], dt_bias=t[6], use_qk_l2norm_in_kernel=True, use_gate_in_kernel=True,
                  safe_gate=True, lower_bound=-5.0)
        return [o.double()] + [x.double() for x in torch.autograd.grad((o * do.to(o.dtype)).sum(), t)]
    for name, a, b in zip("o q k v g beta A_log dt_bias".split(), run(chunk_kda, torch.float32),
                          run(kda_reference, torch.float64)):
        assert (a - b).norm() / b.norm() < 0.01, name
